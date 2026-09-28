import argparse
import json
from pathlib import Path
import sys

from .model import ChunkConfig, PipelineError
from .normalize import normalize_html
from .pipeline import chunk_document, retrieval_views
from .io import write_run


def main():
    ap = argparse.ArgumentParser(description='Simplified HTML → lossless source chunks + separate retrieval views')
    ap.add_argument('--html', type=Path, required=True)
    ap.add_argument('--doc-id', required=True)
    ap.add_argument('--url', default='')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--absolute-chars', type=int, help='Explicit experimental H, not a production default')
    ap.add_argument('--out-dir', type=Path, required=True, help='Must not already exist')
    ap.add_argument('--llm-contexts', type=Path, help='Cached context mapping keyed by llm_requests cache_key')
    ap.add_argument('--view-absolute-chars', type=int, help='Explicit limit for fixed-boundary enhanced views')
    args = ap.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        if args.absolute_chars is not None:
            config['absolute_chars'] = args.absolute_chars
        cfg = ChunkConfig.from_dict(config)
        html = args.html.read_text(encoding='utf-8')
        doc = normalize_html(html, args.doc_id, args.url)
        records, report = chunk_document(doc, cfg)
        cache = json.loads(args.llm_contexts.read_text(encoding='utf-8')) if args.llm_contexts else None
        views = retrieval_views(doc, records, cache, absolute_chars=args.view_absolute_chars)
        report['retrieval_views'] = {'count': len(views), 'llm_context_enabled': cache is not None,
                                     'boundaries_frozen': True, 'token_limit_verified': False,
                                     'max_chars': max((v['chars'] for v in views), default=0),
                                     'over_target': sum(v['chars'] > cfg.target_chars for v in views)}
        write_run(args.out_dir, doc, records, report, views, simplified_html=html)
        print(json.dumps({'out_dir': str(args.out_dir), **report['stats']}, ensure_ascii=False))
    except (PipelineError, OSError, json.JSONDecodeError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
