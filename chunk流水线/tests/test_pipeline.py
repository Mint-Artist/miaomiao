import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunk_pipeline import ChunkConfig, PipelineError, normalize_html
from chunk_pipeline.pipeline import chunk_document, retrieval_views, llm_cache_key, validate_records
from chunk_pipeline.model import digest
from chunk_pipeline.io import write_run


class PipelineTests(unittest.TestCase):
    def run_html(self, html, **kwargs):
        doc = normalize_html(html, 'test')
        cfg = ChunkConfig(**kwargs)
        records, report = chunk_document(doc, cfg)
        self.assertEqual(''.join(r['text'] for r in records), doc.text)
        self.assertTrue(all(r['contextual_chars'] <= cfg.limit for r in records))
        return doc, records, report

    def test_three_canonical_fixtures_all_policies(self):
        for p in sorted((ROOT / '规格/vNext/样例').glob('*.normalized.txt')):
            name = p.name.replace('.normalized.txt', '')
            doc = normalize_html((ROOT / 'html结构切分_2026-09-23/output' / (name + '.simplified.html')).read_text(), name)
            self.assertEqual(doc.text, p.read_text())
            self.assertEqual(''.join(doc.text[a.start:a.end] for a in doc.atoms), doc.text)
            for config in (ROOT / 'configs').glob('*.json'):
                with self.subTest(sample=name, config=config.stem):
                    obj = json.loads(config.read_text())
                    if obj.get('budget_policy') == 'protect':
                        obj['absolute_chars'] = 512
                    cfg = ChunkConfig.from_dict(obj)
                    records, report = chunk_document(doc, cfg)
                    self.assertEqual(''.join(r['text'] for r in records), p.read_text())
                    self.assertEqual(report['text_sha256'], digest(p.read_text()))
                    self.assertEqual(len(retrieval_views(doc, records)), len(records))

    def test_inline_spaces_entities_and_unicode(self):
        d = normalize_html('<p><b>Hello </b>world <a href="/x">again</a> &amp; 中\u00a0文😀</p>', 'x')
        self.assertEqual(d.text, 'Hello world again & 中\u00a0文😀\n')
        anchor = next(n for n in d.nodes.values() if n['tag'] == 'a')
        self.assertEqual(d.text[anchor['start']:anchor['end']], 'again')

    def test_inline_only_space_and_nested_tags(self):
        d = normalize_html('<p>Hello<span> </span><em><b>world</b> </em>again</p>', 'x')
        self.assertEqual(d.text, 'Hello world again\n')

    def test_br_pre_and_boundary_newlines(self):
        d = normalize_html('<p>A<br><br>B<br></p><pre>  a\r\n\tb\n</pre><p>C</p>', 'x')
        self.assertEqual(d.text, 'A\n\nB\n  a\n\tb\n\nC\n')
        pre = next(n for n in d.nodes.values() if n['tag'] == 'pre')
        self.assertEqual(d.text[pre['start']:pre['end']], '  a\n\tb\n')

    def test_nested_list_note_ranges_and_attrs(self):
        html = '<div class="note"><ol start="3"><li value="4">A<p>B</p><ul><li>C</li></ul></li><li>D</li></ol></div>'
        d, rows, _ = self.run_html(html)
        self.assertEqual(d.text, 'A\nB\nC\nD\n')
        self.assertEqual([a.kind for a in d.atoms], ['note'])
        self.assertEqual(next(n for n in d.nodes.values() if n['tag'] == 'ol')['attrs']['start'], '3')
        self.assertEqual(len(rows), 1)

    def test_resources_do_not_inject_alt_or_urls(self):
        d, _, _ = self.run_html('<p>A<img src="image.jpg" alt="NOT BODY">B<a href="/target">link</a></p>')
        self.assertEqual(d.text, 'ABlink\n')
        image = next(r for r in d.resources if r['kind'] == 'img')
        self.assertEqual((image['start'], image['end']), (1, 1))
        self.assertEqual(image['alt'], 'NOT BODY')
        self.assertEqual(len(d.resources), 2)

    def test_image_only_and_empty_document(self):
        d, rows, _ = self.run_html('<figure><img src="x"></figure>')
        self.assertEqual(d.text, '')
        self.assertEqual(rows, [])
        self.assertEqual(d.resources[0]['start'], 0)
        self.assertEqual(normalize_html('', 'empty').text, '')

    def test_plain_text_gaps_and_wrappers_are_not_lost(self):
        d, _, _ = self.run_html('<html><body><section>前<p>中</p>后</section></body></html>')
        self.assertEqual(d.text, '前\n中\n后\n')

    def test_metadata_title_table_header_preserved(self):
        d, rows, _ = self.run_html('<h1>Title</h1><div class="metadata"><p>Author</p></div><table><thead><tr><th>Column</th></tr></thead><tbody><tr><td>Value</td></tr></tbody></table>')
        self.assertEqual(d.text, 'Title\nAuthor\nColumn\nValue\n')
        self.assertEqual(''.join(r['text'] for r in rows).count('Column'), 1)

    def test_table_internal_br_and_empty_cells(self):
        d, _, _ = self.run_html('<table><thead><tr><th>年<br>份</th><th>B</th></tr></thead><tbody><tr><td></td><td>x<br>y</td></tr></tbody></table>')
        self.assertEqual(d.text, '年\n份\tB\n\tx\ny\n')
        self.assertEqual(len(next(iter(d.tables.values()))['rows']), 2)

    def test_table_caption_footer_order(self):
        d, _, _ = self.run_html('<table><caption>表名</caption><thead><tr><th>列</th></tr></thead><tbody><tr><td>一</td></tr></tbody><tfoot><tr><td>合计</td></tr></tfoot></table>')
        self.assertEqual(d.text, '表名\n列\n一\n合计\n')

    def test_rowspan_colspan_mapping(self):
        html = '<table><tbody><tr><td rowspan="2">A</td><td colspan="2">B</td></tr><tr><td>C</td><td>D</td></tr></tbody></table>'
        d, _, _ = self.run_html(html)
        rows = next(iter(d.tables.values()))['rows']
        self.assertEqual([c['column'] for c in rows[1]['cells']], [1, 2])
        self.assertEqual(rows[0]['cells'][1]['colspan'], 2)
        self.assertEqual(d.text, 'A\tB\nC\tD\n')

    def test_rowspan_context_keeps_inherited_value(self):
        html = '<table><thead><tr><th>类型</th><th>内容</th></tr></thead><tbody><tr><td rowspan="2">种类A</td><td>' + '甲' * 40 + '</td></tr><tr><td>' + '乙' * 40 + '</td></tr></tbody></table>'
        d, rows, _ = self.run_html(html, target_chars=80, context_mode='structure')
        second = next(r for r in rows if '乙' in r['text'])
        self.assertIn('种类A', second['contextual_text'])
        self.assertTrue(any(s['kind'] == 'rowspan' for s in second['context_sources']))
        self.assertEqual(''.join(r['text'] for r in rows).count('种类A'), 1)

    def test_unsupported_html_fails_explicitly(self):
        for html in ('<script>bad</script><p>x</p>', '<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>', '<table><tr><td rowspan="0">x</td></tr></table>'):
            with self.subTest(html=html), self.assertRaises(PipelineError):
                normalize_html(html, 'x')

    def test_protected_paragraph_is_standalone(self):
        d, rows, report = self.run_html('<p>' + '甲' * 50 + '</p><p>随后</p>',
                                      target_chars=32, budget_policy='protect', absolute_chars=96)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['text'], '甲' * 50 + '\n')
        self.assertEqual(rows[0]['over_target_reason'], 'protected_unit')
        self.assertEqual(report['stats']['split_atoms'], 0)

    def test_multiple_fitting_paragraphs_cannot_overfill(self):
        _, rows, _ = self.run_html('<p>' + '甲' * 20 + '</p><p>' + '乙' * 20 + '</p>',
                                  target_chars=32, budget_policy='protect', absolute_chars=96)
        self.assertEqual(len(rows), 2)
        self.assertFalse(any(r['over_target'] for r in rows))

    def test_heading_binding_can_exceed_target(self):
        _, rows, _ = self.run_html('<h2>标题</h2><p>' + '甲' * 25 + '</p><p>下一段</p>',
                                  target_chars=24, budget_policy='protect', absolute_chars=64)
        self.assertEqual(rows[0]['text'], '标题\n' + '甲' * 25 + '\n')
        self.assertEqual(rows[0]['over_target_reason'], 'heading_binding')
        self.assertNotIn('下一段', rows[0]['text'])

    def test_heading_binding_must_not_force_paragraph_split(self):
        _, rows, report = self.run_html('<h2>' + '题' * 30 + '</h2><p>' + '文' * 45 + '</p>',
                                       target_chars=32, budget_policy='protect', absolute_chars=64)
        self.assertTrue(any(r['text'] == '文' * 45 + '\n' for r in rows))
        self.assertEqual(report['stats']['split_atoms'], 0)
        self.assertGreater(report['stats']['orphan_headings'], 0)

    def test_metadata_separation_does_not_move_content(self):
        d, rows, report = self.run_html('<h1>T</h1><div class="metadata"><p>M</p></div><p>B</p>', metadata_mode='separate')
        self.assertEqual([r['text'] for r in rows], ['T\n', 'M\n', 'B\n'])
        self.assertEqual(d.text, 'T\nM\nB\n')
        self.assertEqual(rows[0]['orphan_headings'][0]['reason'], 'metadata_separation')

    def test_long_note_degrades_at_child_boundaries(self):
        html = '<div class="note"><p>' + '甲' * 30 + '</p><p>' + '乙' * 30 + '</p><p>' + '丙' * 30 + '</p></div>'
        _, rows, _ = self.run_html(html, target_chars=32, budget_policy='protect', absolute_chars=64)
        self.assertEqual(rows[0]['text'], '甲' * 30 + '\n' + '乙' * 30 + '\n')
        self.assertTrue(rows[1]['continued'])
        self.assertEqual(rows[1]['parent_id'], rows[0]['chunk_id'])

    def test_hard_wrap_keeps_final_separator(self):
        _, rows, _ = self.run_html('<p>' + '字' * 64 + '</p>', target_chars=32)
        self.assertTrue(all(r['text'].strip() for r in rows))
        self.assertEqual(rows[-1]['text'], '字\n')
        self.assertTrue(any(r['split_reason'] == 'hard_wrap' for r in rows))

    def test_closing_quote_stays_with_sentence_when_it_fits(self):
        _, rows, _ = self.run_html('<p>“这是第一句。”这是第二句。第三句。</p>', target_chars=12)
        self.assertTrue(rows[0]['text'].endswith('。”'))

    def test_sections_and_pseudo_headings(self):
        html = '<h1>文档</h1><p>开头</p><h2>章节</h2><p>内容</p><p><strong>一、子节</strong></p><p>子节内容</p>'
        _, rows, report = self.run_html(html)
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[2]['text'].startswith('一、子节\n子节内容'))
        self.assertEqual(report['stats']['orphan_headings'], 0)
        _, merged, _ = self.run_html(html, merge_short_sections=True)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]['cross_section'])

    def test_context_deduplicates_owned_title(self):
        _, rows, _ = self.run_html('<h1>文档</h1><p>内容</p><h2>章节</h2><p>细节</p>', context_mode='structure')
        self.assertEqual(rows[0]['contextual_text'].count('文档'), 1)
        self.assertEqual(rows[1]['contextual_text'].count('章节'), 1)
        self.assertTrue(rows[1]['contextual_text'].startswith('文档\n'))

    def test_title_trim_does_not_delete_source(self):
        html = '<h1>文档标题非常长</h1><h2>另一个很长标题</h2><p>' + '内容' * 60 + '</p>'
        d, rows, report = self.run_html(html, target_chars=48, context_mode='structure', max_title_chars=4)
        self.assertIn('文档标题非常长', d.text)
        self.assertGreater(report['stats']['context_trimmed'], 0)

    def test_optional_long_title_cannot_block_small_budget(self):
        d, rows, report = self.run_html('<h1>' + '长题' * 20 + '</h1><p>正文在这里</p>',
                                       target_chars=16, context_mode='structure')
        self.assertEqual(''.join(r['text'] for r in rows), d.text)
        self.assertGreater(report['stats']['context_trimmed'], 0)

    def test_table_rows_and_headers_use_actual_dom(self):
        html = '<table><thead><tr><th>A<br>B</th><th>C</th></tr></thead><tbody>' + ''.join('<tr><td>值%d</td><td>原文|符号</td></tr>' % i for i in range(20)) + '</tbody></table>'
        d, rows, _ = self.run_html(html, target_chars=64, budget_policy='protect', absolute_chars=128, context_mode='structure')
        self.assertEqual(d.text.count('原文|符号'), 20)
        self.assertTrue(all(r['contextual_chars'] <= 64 for r in rows))
        self.assertTrue(all('表头：A B | C' in r['contextual_text'] for r in rows[1:]))

    def test_long_table_cell_never_repeats_neighbor_source(self):
        html = '<table><thead><tr><th>键</th><th>值</th></tr></thead><tbody><tr><td>唯一标识X</td><td>' + '长' * 180 + '</td></tr></tbody></table>'
        d, rows, _ = self.run_html(html, target_chars=80, context_mode='structure')
        self.assertEqual(''.join(r['text'] for r in rows).count('唯一标识X'), 1)
        self.assertEqual(''.join(r['text'] for r in rows).count('长'), 180)
        self.assertTrue(any(s['kind'] == 'table_fragment' for r in rows for s in r['context_sources']))

    def test_multiple_tables_do_not_share_headers(self):
        html = '<table><thead><tr><th>表甲</th></tr></thead><tbody><tr><td>A</td></tr></tbody></table><table><thead><tr><th>表乙</th></tr></thead><tbody><tr><td>B</td></tr></tbody></table>'
        _, rows, _ = self.run_html(html, context_mode='structure')
        self.assertEqual(len(rows), 2)
        self.assertNotIn('表甲', rows[1]['contextual_text'])

    def test_header_too_long_fails_without_silent_omission(self):
        with self.assertRaises(PipelineError):
            self.run_html('<table><thead><tr><th>' + '标题' * 40 + '</th></tr></thead><tbody><tr><td>值</td></tr></tbody></table>', target_chars=32, context_mode='structure')

    def test_recursive_has_no_dom_aware_table_split_reason(self):
        _, rows, _ = self.run_html('<table><tbody><tr><td>' + 'A' * 100 + '</td><td>B</td></tr></tbody></table>', target_chars=32, boundary_strategy='recursive')
        self.assertNotIn('table_cell', [r['split_reason'] for r in rows])

    def test_config_rejects_unsupported_or_implicit_options(self):
        for obj in ({'budget_policy': 'protect'}, {'counter': 'token'}, {'llm_context': True},
                    {'target_chars': True}, {'absolute_chars': 10}, {'boundary_strategy': 'semantic'},
                    {'target_chars': 0}, {'merge_short_sections': 'yes'}):
            with self.subTest(obj=obj), self.assertRaises(PipelineError):
                ChunkConfig.from_dict(obj)

    def test_token_guard_requires_real_counter_and_records_identity(self):
        doc = normalize_html('<p>abc</p>', 'x')
        cfg = ChunkConfig(max_embedding_tokens=3)
        with self.assertRaises(PipelineError):
            chunk_document(doc, cfg)
        with self.assertRaises(PipelineError):
            chunk_document(doc, cfg, token_counter=len, tokenizer_id='test_char_counter')
        rows, report = chunk_document(doc, ChunkConfig(max_embedding_tokens=4), token_counter=len, tokenizer_id='test_char_counter')
        self.assertEqual(rows[0]['embedding_tokens'], 4)
        self.assertTrue(report['token_limit_verified'])

    def test_llm_cache_context_is_separate_and_provenanced(self):
        d, rows, _ = self.run_html('<p>原文</p>')
        key = llm_cache_key(d, rows[0])
        cache = {key: {'text': '测试背景', 'model': 'test-fixture', 'prompt_version': 'v1', 'document_sha256': d.text_sha256}}
        views = retrieval_views(d, rows, cache)
        self.assertEqual(rows[0]['text'], '原文\n')
        self.assertEqual(views[0]['index_text'], '测试背景\n原文\n')
        self.assertEqual(views[0]['llm_context']['model'], 'test-fixture')
        self.assertEqual(views[0]['source_spans'], rows[0]['source_spans'])

    def test_llm_missing_stale_and_oversized_fail(self):
        d, rows, _ = self.run_html('<p>原文</p>')
        key = llm_cache_key(d, rows[0])
        for cache in ({}, {key: {'text': '背景', 'model': 'test', 'prompt_version': 'v1', 'document_sha256': 'stale'}},
                      {key: {'text': '大' * 300, 'model': 'test', 'prompt_version': 'v1', 'document_sha256': d.text_sha256}}):
            with self.subTest(cache=cache), self.assertRaises(PipelineError):
                retrieval_views(d, rows, cache)

    def test_same_text_different_positions_have_distinct_cache_keys(self):
        d, rows, _ = self.run_html('<p>' + '甲' * 20 + '</p><p>' + '甲' * 20 + '</p>', target_chars=24)
        self.assertEqual(rows[0]['text'], rows[1]['text'])
        self.assertNotEqual(llm_cache_key(d, rows[0]), llm_cache_key(d, rows[1]))

    def test_validation_catches_removed_separator_and_duplicate_source(self):
        d, rows, _ = self.run_html('<p>甲</p><p>乙</p>')
        rows[0]['text'] = rows[0]['text'].rstrip()
        with self.assertRaises(PipelineError):
            validate_records(d, rows, ChunkConfig())

    def test_determinism_and_atomic_no_overwrite(self):
        d, rows, report = self.run_html('<p>正文</p>')
        self.assertEqual(chunk_document(d, ChunkConfig()), (rows, report))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'out'
            write_run(path, d, rows, report, retrieval_views(d, rows))
            manifest = json.loads((path / 'manifest.json').read_text())
            for name, sha in manifest['files_sha256'].items():
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), sha)
            with self.assertRaises(PipelineError):
                write_run(path, d, rows, report, [])

    def test_cli_and_no_partial_output_on_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / 'input.html'
            source.write_text('<h1>标题</h1><p>正文</p>')
            cmd = [sys.executable, '-m', 'chunk_pipeline', '--html', str(source), '--doc-id', 'cli',
                   '--config', str(ROOT / 'configs/structure_hard.json'), '--out-dir', str(tmp / 'out')]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((tmp / 'out/normalized.txt').read_text(), '标题\n正文\n')
            cmd[cmd.index('--config') + 1] = str(ROOT / 'configs/structure_protect.json')
            cmd[-1] = str(tmp / 'invalid')
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((tmp / 'invalid').exists())

    def test_seeded_random_roundtrip_and_budgets(self):
        rng = random.Random(20260924)
        for case in range(40):
            parts = ['<h1>样本%d</h1>' % case]
            for _ in range(rng.randrange(1, 8)):
                value = ''.join(rng.choice('甲乙ABC😀。；，|') for _ in range(rng.randrange(1, 120)))
                parts.append('<p>' + value + '</p>')
            for policy in ('hard', 'protect'):
                with self.subTest(case=case, policy=policy):
                    self.run_html(''.join(parts), target_chars=40, budget_policy=policy,
                                  absolute_chars=80, context_mode='structure', max_title_chars=20)

    def test_legacy_baseline_unchanged(self):
        manifest = json.loads((ROOT / 'baselines/legacy_manifest.json').read_text())
        self.assertFalse(manifest['is_production_B0'])
        for relative, sha in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), sha, relative)


if __name__ == '__main__':
    unittest.main()
