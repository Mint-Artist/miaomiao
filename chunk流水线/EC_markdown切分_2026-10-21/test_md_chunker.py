"""md_chunker 单元测试：python3 -m unittest -v test_md_chunker"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from md_chunker import HARD, TARGET, chunk_markdown, clean_inline, parse_html_table

HERE = os.path.dirname(os.path.abspath(__file__))


class CleanInlineTest(unittest.TestCase):
    def test_links_keep_text_only(self):
        self.assertEqual(clean_inline('见[官网](https://a.cn/x)说明'), '见官网说明')
        self.assertEqual(clean_inline('[Python](https://zh.wikipedia.org/wiki/Python_(语言))。'), 'Python。')
        self.assertEqual(clean_inline('[带标题](https://a.cn "提示")'), '带标题')
        self.assertEqual(clean_inline('[引用式][ref1]链接'), '引用式链接')
        self.assertEqual(clean_inline('[分享](javascript:void(0))'), '分享')
        self.assertEqual(clean_inline('[](https://a.cn)空链接'), '空链接')

    def test_images_removed(self):
        self.assertEqual(clean_inline('前![图1](https://a.cn/1.png)后'), '前后')
        self.assertEqual(clean_inline('[![二维码](q.png)](https://a.cn)扫码'), '扫码')
        self.assertEqual(clean_inline('![图1](1.png)', keep_image_alt=True), '图1')
        self.assertEqual(clean_inline('![](data:image/png;base64,AAAA)正文'), '正文')

    def test_autolink_and_markup(self):
        self.assertEqual(clean_inline('详见<https://a.cn/x>。'), '详见。')
        self.assertEqual(clean_inline('**加粗**和*斜体*和~~删除~~和`代码`'), '加粗和斜体和删除和代码')
        self.assertEqual(clean_inline('3*4*5=60'), '3*4*5=60')
        self.assertEqual(clean_inline(r'\*不是强调\* \[不是链接\](x)'), '*不是强调* [不是链接](x)')
        self.assertEqual(clean_inline('A&amp;B<span style="x">文字</span><br>换行'), 'A&B文字\n换行')

    def test_plain_brackets_kept(self):
        self.assertEqual(clean_inline('参考文献[1]与（一）全角括号'), '参考文献[1]与（一）全角括号')


class TableTest(unittest.TestCase):
    def test_rowspan_colspan(self):
        markup = ('<table><tr><th>省</th><th>城市</th><th>风力</th></tr>'
                  '<tr><td rowspan="2">河北</td><td>石家庄</td><td>4级</td></tr>'
                  '<tr><td>张家口</td><td>6级</td></tr>'
                  '<tr><td colspan="2">北京</td><td>3级</td></tr></table>')
        caption, header, body = parse_html_table(markup)
        self.assertEqual(header, [['省', '城市', '风力']])
        self.assertEqual(body, [['河北', '石家庄', '4级'], ['河北', '张家口', '6级'], ['北京', '3级']])

    def test_links_in_cells(self):
        _, header, body = parse_html_table('<table><tr><td>网址</td><td><a href="https://a.cn">官网</a> '
                                           '[入口](https://b.cn)</td></tr></table>')
        self.assertEqual(header + body, [['网址', '官网 入口']])

    def test_long_table_repeats_header(self):
        rows = ''.join('<tr><td>第%d行</td><td>%s</td></tr>' % (i, '内容' * 20) for i in range(20))
        md = '## 表\n\n<table><tr><th>序号</th><th>内容</th></tr>%s</table>' % rows
        chunks = chunk_markdown(md)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(chunks[0].startswith('表\n序号 | 内容'))
        for c in chunks[1:]:
            self.assertTrue(c.startswith('序号 | 内容\n'), c[:30])
        self.assertTrue(all(len(c) <= TARGET for c in chunks))
        # 每一行都完整出现在某一块里
        for i in range(20):
            self.assertTrue(any('第%d行 | %s' % (i, '内容' * 20) in c for c in chunks))

    def test_single_column_table_is_text(self):
        md = '正文。\n\n<table><tr><td>排版用的单列表格</td></tr><tr><td>第二行</td></tr></table>\n\n后文。'
        self.assertEqual(chunk_markdown(md), ['正文。\n排版用的单列表格\n第二行\n后文。'])

    def test_table_inline_with_text(self):
        chunks = chunk_markdown('下表：<table><tr><td>a</td><td>b</td></tr></table>表后文字')
        self.assertEqual(chunks, ['下表：\na | b', '表后文字'])


class StructureTest(unittest.TestCase):
    def test_heading_bound_and_sections_not_merged(self):
        md = '# 标题\n\n## 一\n\n第一节内容。\n\n## 二\n\n第二节内容。'
        self.assertEqual(chunk_markdown(md), ['标题\n一\n第一节内容。', '二\n第二节内容。'])

    def test_heading_before_table_not_orphaned(self):
        md = '## 预报\n\n<table><tr><th>城市</th></tr><tr><td>北京</td></tr></table>'
        self.assertEqual(chunk_markdown(md), ['预报\n城市\n北京'])

    def test_greedy_merge_within_section(self):
        paras = ['第%d段。' % i + '字' * 60 for i in range(6)]
        chunks = chunk_markdown('\n\n'.join(paras))
        self.assertTrue(all(len(c) <= TARGET for c in chunks))
        self.assertEqual(''.join(chunks).replace('\n', ''), ''.join(paras))
        self.assertLess(len(chunks), len(paras))

    def test_protected_paragraph_kept_whole(self):
        para = '这是一个较长的段落，' * 40 + '结束。'  # 403 字
        chunks = chunk_markdown(para)
        self.assertEqual(chunks, [para])

    def test_oversized_paragraph_split_at_sentences(self):
        para = '第一句话很长很长，' * 10 + '。' + '第二句。' * 150
        chunks = chunk_markdown(para)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= TARGET for c in chunks))
        self.assertEqual(''.join(chunks), para)

    def test_colon_lead_binds_list(self):
        md = '措施如下：\n\n- 第一条\n- 第二条\n\n## 下一节\n\n内容'
        self.assertEqual(chunk_markdown(md), ['措施如下：\n第一条\n第二条', '下一节\n内容'])

    def test_lists_and_quotes(self):
        md = '1. 甲\n2. 乙\n   续行\n\n> 引用**内容**\n> 第二行\n\n```\ncode\n```'
        self.assertEqual(chunk_markdown(md), ['1. 甲\n2. 乙\n续行\n引用内容\n第二行\ncode'])

    def test_setext_and_hr(self):
        md = '大标题\n===\n\n正文\n\n---\n\n小标题\n---\n\n正文二'
        self.assertEqual(chunk_markdown(md), ['大标题\n正文', '小标题\n正文二'])

    def test_heading_context(self):
        md = '# 总标题\n\n## 一\n\n' + '内容。' * 60 + '\n\n' + '补充。' * 60
        chunks = chunk_markdown(md, heading_context=True)
        self.assertTrue(chunks[0].startswith('总标题\n一\n'))
        self.assertTrue(all(c.startswith('总标题\n一\n') for c in chunks))
        self.assertTrue(all(len(c) <= HARD for c in chunks))

    def test_empty(self):
        self.assertEqual(chunk_markdown(''), [])
        self.assertEqual(chunk_markdown(None), [])
        self.assertEqual(chunk_markdown('![](a.png)\n\n---'), [])

    def test_hard_cap(self):
        md = '<table><tr><td>%s</td></tr></table>\n\n%s' % ('超长单元格' * 300, '无标点' * 400)
        self.assertTrue(all(len(c) <= HARD for c in chunk_markdown(md)))


class CliTest(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, 'in.jsonl'), os.path.join(d, 'out.jsonl')
            with open(src, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'id': 1, 'markdown': '# 标题\n\n正文[链接](https://a.cn)', 'extra': [1]},
                                   ensure_ascii=False) + '\n')
                f.write(json.dumps({'id': 2}) + '\n')
            subprocess.run([sys.executable, os.path.join(HERE, 'md_chunker.py'), '--input', src, '--output', dst],
                           check=True, capture_output=True)
            with open(dst, encoding='utf-8') as f:
                out = [json.loads(line) for line in f]
            self.assertEqual(out[0], {'id': 1, 'markdown': '# 标题\n\n正文[链接](https://a.cn)', 'extra': [1],
                                      'chunks': ['标题\n正文链接']})
            self.assertEqual(out[1], {'id': 2, 'chunks': []})


if __name__ == '__main__':
    unittest.main()
