from collections import Counter
import importlib.util
from pathlib import Path
import sys
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chunk_pipeline import html_extract
from run_examples import CASES


class ExtractorTests(unittest.TestCase):
    def test_removed_parent_descendants_are_skipped(self):
        for noise in ('<div class="related"><span><b>应删除</b></span></div>',
                      '<nav><ul><li>应删除</li></ul></nav>',
                      '<section class="custom-noise"><div><p>应删除</p></div></section>'):
            with self.subTest(noise=noise):
                soup = BeautifulSoup('<article><p>保留正文</p>' + noise + '</article>', 'html.parser')
                stats = Counter()
                html_extract.clean_tree(soup, ['.custom-noise'], stats)
                self.assertEqual(soup.get_text(), '保留正文')
                self.assertEqual(sum(n for key, n in stats.items() if key.startswith('dropped:')), 1)

    def test_empty_unknown_subtree_does_not_revisit_destroyed_tags(self):
        soup = BeautifulSoup('<p>保留正文</p><custom><nested></nested></custom>', 'html.parser')
        stats = Counter()
        html_extract.clean_tree(soup, [], stats)
        self.assertEqual(str(soup), '<p>保留正文</p>')
        self.assertEqual(stats['unknown:custom'], 1)
        self.assertNotIn('unknown:None', stats)

    def test_empty_nested_blocks_count_only_live_removals(self):
        soup = BeautifulSoup('<div><p><strong></strong></p></div><p>保留正文</p>', 'html.parser')
        self.assertEqual(html_extract.prune_empty(soup), 1)
        self.assertEqual(soup.get_text(), '保留正文')

    def test_raw_nested_noise_extraction_preserves_body(self):
        body = '正常正文内容。' * 10
        html = '<h1>标题</h1><article><p>' + body + '</p><div class="related"><span>应删除</span></div></article>'
        simplified, report = html_extract.extract(html, url='https://example.com', anchors=True)
        self.assertIn(body, simplified)
        self.assertNotIn('应删除', simplified)
        self.assertEqual(report['extractor_version'], '0.1.1')

    def test_existing_sample_outputs_match_frozen_extractor(self):
        path = ROOT / '简版HTML提取_2026-09-23/simplify_html.py'
        spec = importlib.util.spec_from_file_location('legacy_extractor_test', path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for name, url in CASES:
            with self.subTest(sample=name):
                raw = (path.parent / 'samples' / (name + '.raw.html')).read_text(encoding='utf-8')
                old_html, old_report = legacy.extract(raw, url=url, anchors=True)
                new_html, new_report = html_extract.extract(raw, url=url, anchors=True)
                self.assertEqual(new_html, old_html)
                new_report.pop('extractor_version')
                old_report.pop('extractor_version')
                self.assertEqual(new_report, old_report)


if __name__ == '__main__':
    unittest.main()
