"""Streaming raw-HTML JSONL runner; one complete comparison per input record."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .io import dumps, write_run
from .model import ChunkConfig, PipelineError, digest
from .normalize import normalize_html
from .pipeline import chunk_document, retrieval_views
from . import html_extract

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = Path(html_extract.__file__).resolve()


def load_configs(directory, absolute_chars):
    configs = []
    for path in sorted(Path(directory).glob('*.json')):
        if path.stem == 'url':
            raise PipelineError('Strategy name url is reserved')
        obj = json.loads(path.read_text(encoding='utf-8'))
        if absolute_chars is not None and obj.get('budget_policy') == 'protect':
            obj['absolute_chars'] = absolute_chars
        configs.append((path.stem, ChunkConfig.from_dict(obj)))
    if not configs:
        raise PipelineError('No strategy JSON files found')
    return configs


def load_extractor():
    return html_extract


def summary_chunks(records, views, field):
    if field == 'index_text':
        return [v['index_text'] for v in views]
    if field == 'text':
        return [r['text'] for r in records]
    raise PipelineError('Unknown summary field: ' + field)


def run_batch(input_path, out_dir, configs, html_field='pg', url_field='url',
              summary_field='index_text', profile=None, progress_every=100):
    """Keep memory bounded to one input page; retain failures outside summary."""
    extractor = load_extractor()
    if profile:
        extractor.pick_profile('', profile)  # Reject invalid global options before output.
    if summary_field not in ('index_text', 'text'):
        raise PipelineError('Invalid summary field')
    out_dir = Path(out_dir)
    # Open input first; exclusive mkdir prevents overwriting an earlier experiment.
    with Path(input_path).open(encoding='utf-8-sig') as source:
        out_dir.mkdir(parents=True, exist_ok=False)
        state = {
            'status': 'running', 'input': str(Path(input_path).resolve()),
            'html_field': html_field, 'url_field': url_field, 'summary_field': summary_field,
            'configs': {name: asdict(cfg) for name, cfg in configs},
            'extractor_sha256': digest(EXTRACTOR.read_text(encoding='utf-8')),
            'extractor_version': extractor.EXTRACTOR_VERSION, 'profile_override': profile,
            'lines': 0, 'blank_lines': 0, 'succeeded': 0, 'failed': 0,
        }
        (out_dir / 'run.json').write_text(dumps(state), encoding='utf-8')
        try:
            with (out_dir / 'summary.jsonl').open('w', encoding='utf-8') as summary, \
                    (out_dir / 'errors.jsonl').open('w', encoding='utf-8') as errors, \
                    (out_dir / 'pages.jsonl').open('w', encoding='utf-8') as pages:
                for line_no, line in enumerate(source, 1):
                    state['lines'] = line_no
                    if not line.strip():
                        state['blank_lines'] += 1
                        continue
                    url, page_dir, phase = None, None, 'input'
                    try:
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise PipelineError('Input record must be an object')
                        url = value.get(url_field)
                        html = value.get(html_field)
                        if not isinstance(url, str) or not url.strip():
                            raise PipelineError('Missing or invalid URL field: ' + url_field)
                        if not isinstance(html, str) or not html.strip():
                            raise PipelineError('Missing or invalid HTML field: ' + html_field)
                        # Line number preserves duplicate URLs; digest avoids unsafe URL paths.
                        doc_id = '%09d_%s' % (line_no, digest(url)[:12])
                        page_dir = out_dir / 'pages' / doc_id
                        page_dir.mkdir(parents=True)
                        phase = 'extract'
                        simplified, extraction = extractor.extract(html, url=url,
                                                                    profile_name=profile, anchors=True)
                        (page_dir / 'simplified.html').write_text(simplified, encoding='utf-8')
                        (page_dir / 'extraction.json').write_text(dumps(extraction), encoding='utf-8')
                        phase = 'normalize'
                        doc = normalize_html(simplified, doc_id, url)
                        result = {'url': url}
                        prepared = []
                        for name, cfg in configs:
                            phase = 'strategy:' + name
                            records, report = chunk_document(doc, cfg)
                            views = retrieval_views(doc, records)
                            result[name] = summary_chunks(records, views, summary_field)
                            prepared.append((name, records, report, views))
                    except (ValueError, TypeError, KeyError, AttributeError, AssertionError, RecursionError) as exc:
                        error = {'line': line_no, 'url': url, 'phase': phase,
                                 'page_dir': str(page_dir.relative_to(out_dir)) if page_dir else None,
                                 'error_type': type(exc).__name__, 'error': str(exc)}
                        if page_dir:
                            (page_dir / 'error.json').write_text(dumps(error), encoding='utf-8')
                        errors.write(json.dumps(error, ensure_ascii=False) + '\n')
                        errors.flush()
                        state['failed'] += 1
                    else:
                        # Filesystem failures abort the run rather than silently losing output.
                        for name, records, report, views in prepared:
                            write_run(page_dir / name, doc, records, report, views)
                        pages.write(json.dumps({'line': line_no, 'url': url, 'doc_id': doc_id,
                                                'page_dir': str(page_dir.relative_to(out_dir))},
                                               ensure_ascii=False) + '\n')
                        pages.flush()
                        summary.write(json.dumps(result, ensure_ascii=False) + '\n')
                        summary.flush()
                        state['succeeded'] += 1
                    if progress_every and (state['succeeded'] + state['failed']) % progress_every == 0:
                        print('Processed %d: succeeded=%d failed=%d' %
                              (line_no, state['succeeded'], state['failed']), file=sys.stderr)
                state['status'] = 'completed_with_errors' if state['failed'] else 'completed'
        except BaseException:
            state['status'] = 'interrupted'
            raise
        finally:
            (out_dir / 'run.json').write_text(dumps(state), encoding='utf-8')
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='UTF-8 JSONL, raw HTML in pg')
    ap.add_argument('--out-dir', type=Path, required=True, help='New experiment directory')
    ap.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    ap.add_argument('--absolute-chars', type=int, help='Explicit H required by protect configurations')
    ap.add_argument('--html-field', default='pg')
    ap.add_argument('--url-field', default='url')
    ap.add_argument('--summary-field', choices=['index_text', 'text'], default='index_text')
    ap.add_argument('--profile', help='Optional forced extractor site profile')
    ap.add_argument('--progress-every', type=int, default=100)
    args = ap.parse_args()
    if args.progress_every < 0:
        ap.error('--progress-every must be nonnegative')
    try:
        configs = load_configs(args.config_dir, args.absolute_chars)
        state = run_batch(args.input, args.out_dir, configs, args.html_field, args.url_field,
                          args.summary_field, args.profile, args.progress_every)
    except (ValueError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(state, ensure_ascii=False))
    return 1 if state['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
