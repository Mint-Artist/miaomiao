import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chunk_pipeline.batch import load_configs, load_extractor, run_batch
from run_examples import CASES


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class BatchTests(unittest.TestCase):
    def test_cli_body_fallback_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'input').write_text(json.dumps({'url': 'https://example.com', 'pg':
                '<body><div id="gov-text"><p>正文内容。</p></div></body>'}) + '\n')
            cmd = [sys.executable, str(ROOT / 'run_jsonl.py'), '--input', str(tmp / 'input'),
                   '--out-dir', str(tmp / 'out'), '--absolute-chars', '512', '--content-fallback', 'body']
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((tmp / 'out/run.json').read_text())
            self.assertEqual((state['succeeded'], state['fallback_extracted']), (1, 1))
            page = read_rows(tmp / 'out/pages.jsonl')[0]
            report = json.loads((tmp / 'out' / page['page_dir'] / 'extraction.json').read_text())
            self.assertTrue(report['content_fallback_used'])
            cmd[cmd.index('--out-dir') + 1] = str(tmp / 'invalid')
            result = subprocess.run(cmd + ['--content-selector', '['], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((tmp / 'invalid').exists())

    def test_attribute_error_isolated_and_next_page_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            html = '<article><p>' + '保留的正文。' * 10 + '</p></article>'
            (tmp / 'input').write_text(''.join(json.dumps({'url': url, 'pg': html}) + '\n'
                                             for url in ['https://example.com/1', 'https://example.com/2']))
            extractor = load_extractor()
            original = extractor.extract
            def extract_or_fail(html, **kwargs):
                if kwargs['url'].endswith('/1'):
                    raise AttributeError('simulated malformed-page error')
                return original(html, **kwargs)
            with patch.object(extractor, 'extract', side_effect=extract_or_fail):
                state = run_batch(tmp / 'input', tmp / 'out', load_configs(ROOT / 'configs', 512),
                                  progress_every=0)
            self.assertEqual((state['succeeded'], state['failed']), (1, 1))
            error = read_rows(tmp / 'out/errors.jsonl')[0]
            self.assertEqual((error['line'], error['phase'], error['error_type']),
                             (1, 'extract', 'AttributeError'))
            self.assertEqual(read_rows(tmp / 'out/summary.jsonl')[0]['url'], 'https://example.com/2')

    def test_raw_samples_all_strategies_and_saved_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / 'input.jsonl'
            source.write_text(''.join(json.dumps({'url': url, 'pg':
                (ROOT / '简版HTML提取_2026-09-23/samples' / (name + '.raw.html')).read_text()}) + '\n'
                for name, url in CASES), encoding='utf-8')
            configs = load_configs(ROOT / 'configs', 512)
            out = tmp / 'out'
            result = run_batch(source, out, configs, progress_every=0)
            self.assertEqual(result['succeeded'], 3)
            self.assertEqual(result['failed'], 0)
            summaries = read_rows(out / 'summary.jsonl')
            for summary, page in zip(summaries, read_rows(out / 'pages.jsonl')):
                folder = out / page['page_dir']
                html = (folder / 'simplified.html').read_bytes()
                self.assertIn(b'<h1', html)
                self.assertEqual(set(summary), {'url'} | {name for name, _ in configs})
                for name, _ in configs:
                    views = read_rows(folder / name / 'retrieval.jsonl')
                    self.assertEqual(summary[name], [v['index_text'] for v in views])
                    records = read_rows(folder / name / 'chunks.jsonl')
                    self.assertEqual(''.join(r['text'] for r in records),
                                     (folder / name / 'normalized.txt').read_text())
                    doc = json.loads((folder / name / 'document.json').read_text())
                    self.assertEqual(doc['html_sha256'], hashlib.sha256(html).hexdigest())
            self.assertEqual((out / 'errors.jsonl').read_text(), '')
            with self.assertRaises(FileExistsError):
                run_batch(source, out, configs)

    def test_bad_records_duplicates_custom_fields_and_source_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            row = {'link': 'https://example.com/../../same', 'html':
                   '<html><h1>标题</h1><article><p>' + '测试正文。' * 80 + '</p></article></html>'}
            good = json.dumps(row, ensure_ascii=False) + '\n'
            (tmp / 'input').write_text('\ufeff' + good + '\nINVALID\n{}\n[]\n' + good, encoding='utf-8')
            out = tmp / 'out'
            configs = load_configs(ROOT / 'configs', 512)
            state = run_batch(tmp / 'input', out, configs, 'html', 'link', 'text', progress_every=0)
            self.assertEqual((state['succeeded'], state['failed'], state['blank_lines']), (2, 3, 1))
            self.assertEqual([r['line'] for r in read_rows(out / 'errors.jsonl')], [3, 4, 5])
            pages = read_rows(out / 'pages.jsonl')
            self.assertNotEqual(pages[0]['doc_id'], pages[1]['doc_id'])
            for summary, page in zip(read_rows(out / 'summary.jsonl'), pages):
                for name, _ in configs:
                    normalized = out / page['page_dir'] / name / 'normalized.txt'
                    self.assertEqual(''.join(summary[name]), normalized.read_text())

    def test_strategy_failure_keeps_html_but_no_partial_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Mandatory table header cannot fit the context budget.
            html = '<article><table><tr><th>' + '表头' * 160 + '</th></tr><tr><td>数据</td></tr></table></article>'
            (tmp / 'input').write_text(json.dumps({'url': 'https://example.com', 'pg': html}) + '\n')
            state = run_batch(tmp / 'input', tmp / 'out', load_configs(ROOT / 'configs', 512), progress_every=0)
            self.assertEqual(state['failed'], 1)
            self.assertEqual((tmp / 'out/summary.jsonl').read_text(), '')
            error = read_rows(tmp / 'out/errors.jsonl')[0]
            self.assertTrue(error['phase'].startswith('strategy:'))
            self.assertTrue((tmp / 'out' / error['page_dir'] / 'simplified.html').exists())

    def test_cli_portability_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'input').write_text('{"url":"x","pg":null}\n')
            cmd = [sys.executable, str(ROOT / 'run_jsonl.py'), '--input', str(tmp / 'input'),
                   '--out-dir', str(tmp / 'out')]
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((tmp / 'out').exists())
            result = subprocess.run(cmd + ['--absolute-chars', '512'], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads((tmp / 'out/run.json').read_text())['status'], 'completed_with_errors')


if __name__ == '__main__':
    unittest.main()
