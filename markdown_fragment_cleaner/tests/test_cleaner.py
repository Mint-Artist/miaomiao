from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cleaner.config import (
    AssemblyConfig,
    ChunkConfig,
    CleanerConfig,
    ContentPolicy,
    InputConfig,
    NormalizationConfig,
    OutputConfig,
    RuleConfig,
    TemplateConfig,
)
from cleaner.markdown_parser import MarkdownDocumentParser
from cleaner.models import Fragment
from cleaner.normalizer import DocumentNormalizer
from cleaner.pipeline import clean_jsonl
from cleaner.rules import RuleEngine
from cleaner.tokenization import ApproxTokenCounter


class MarkdownParserTest(unittest.TestCase):
    def test_ast_parser_removes_headings_links_images_and_keeps_nested_list(self) -> None:
        markdown = """---
author: test
---
文档标题
========

正文包含[可见链接](https://example.com/x)，图片![无关截图](image.png)不会进入正文。

## 方法

步骤如下：

- 第一项包含 `inline_code`
  - 子项包含进一步说明
- 第二项包含进一步说明
"""
        document = MarkdownDocumentParser().parse(markdown, "d1")
        self.assertEqual(document.title, "文档标题")
        self.assertEqual([block.type for block in document.blocks], ["paragraph", "paragraph", "list"])
        self.assertNotIn("https://", document.blocks[0].text)
        self.assertNotIn("无关截图", document.blocks[0].text)
        self.assertIn("可见链接", document.blocks[0].text)
        self.assertIn("  - 子项", document.blocks[2].text)
        self.assertEqual(document.blocks[2].heading_path, ["文档标题", "方法"])

    def test_table_and_blockquote_are_rendered_but_fenced_code_is_typed(self) -> None:
        markdown = """# 标题

> 需要保留的引用说明。

| 名称 | 说明 |
| --- | --- |
| A | 第一项 |

```python
print('not prose')
```
"""
        document = MarkdownDocumentParser().parse(markdown, "d2")
        self.assertEqual([block.type for block in document.blocks], ["blockquote", "table", "code"])
        self.assertIn("| 名称 | 说明 |", document.blocks[1].text)

    def test_softbreak_only_inserts_space_inside_ascii_context(self) -> None:
        markdown = """中文正文连接到
Qwen3模型，然后重新连接到
中文内容。

search
engine handles hello,
world correctly.
"""
        document = MarkdownDocumentParser().parse(markdown, "d-softbreak")

        self.assertEqual(document.blocks[0].text, "中文正文连接到Qwen3模型，然后重新连接到中文内容。")
        self.assertEqual(document.blocks[1].text, "search engine handles hello, world correctly.")

    def test_smart_softbreak_preserves_structured_lines(self) -> None:
        markdown = """姓名：张三
电话：13800000000
地址：上海

第一句话已经完整结束。
第二句话也提供了完整信息。
"""
        document = MarkdownDocumentParser().parse(markdown, "d-structured-softbreak")

        self.assertEqual(
            document.blocks[0].text,
            "姓名：张三\n电话：13800000000\n地址：上海",
        )
        self.assertEqual(
            document.blocks[1].text,
            "第一句话已经完整结束。\n第二句话也提供了完整信息。",
        )

    def test_smart_softbreak_unwraps_visual_line_wrapping(self) -> None:
        markdown = """系统首先对用户查询进行规范化，然后从索引中
生成候选文档，接着计算相关性特征并完成排序。
"""
        document = MarkdownDocumentParser().parse(markdown, "d-visual-wrap")

        self.assertEqual(
            document.blocks[0].text,
            "系统首先对用户查询进行规范化，然后从索引中生成候选文档，接着计算相关性特征并完成排序。",
        )


class NormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = MarkdownDocumentParser()
        self.normalizer = DocumentNormalizer(NormalizationConfig())

    def test_boundary_artifacts_are_removed_but_internal_marker_is_preserved(self) -> None:
        markdown = """-->

\ufeff\u200b-->这是文章开头的完整正文，包含足够明确的信息。

技术文档会说明 HTML 注释以 --> 结束，这里的符号属于正文内容。
"""
        result = self.normalizer.normalize(self.parser.parse(markdown, "d-boundary"))

        self.assertEqual(len(result.document.blocks), 2)
        self.assertTrue(result.document.blocks[0].text.startswith("这是文章开头"))
        self.assertIn("以 --> 结束", result.document.blocks[1].text)
        self.assertEqual(result.stats.empty_blocks_removed, 1)
        self.assertGreaterEqual(result.stats.boundary_artifacts_removed, 3)

    def test_adjacent_repeated_sentence_sequence_keeps_one_copy(self) -> None:
        first = "检索系统首先规范化用户查询，并根据索引统计生成一批候选文档。"
        second = "随后排序模块结合相关性特征计算得分，并输出稳定且可复现的结果。"
        final = "最终结果还会经过独立校验，以便及时发现数据或配置异常。"
        markdown = first + second + first + second + final

        result = self.normalizer.normalize(self.parser.parse(markdown, "d-sentences"))
        block = result.document.blocks[0]

        self.assertEqual(block.text, first + second + final)
        self.assertEqual(result.stats.duplicate_sentence_sequences_removed, 1)
        self.assertEqual(result.stats.duplicate_sentences_removed, 2)
        repairs = block.metadata.get("repairs", [])
        self.assertTrue(
            any(
                repair.get("code", "") == "repeated_sentence_sequences_removed"
                for repair in repairs
            )
        )

    def test_adjacent_duplicate_prose_blocks_keep_one_copy(self) -> None:
        paragraph = "该段落完整介绍索引构建流程，包括数据读取、字段规范化和结果校验。"
        document = self.parser.parse(paragraph + "\n\n" + paragraph, "d-blocks")
        result = self.normalizer.normalize(document)

        self.assertEqual(len(result.document.blocks), 1)
        self.assertEqual(result.stats.duplicate_blocks_removed, 1)
        repairs = result.document.blocks[0].metadata.get("repairs", [])
        self.assertEqual(repairs[-1].get("code", ""), "adjacent_duplicate_block")

    def test_near_duplicate_and_short_emphasis_are_not_removed(self) -> None:
        first = "系统支持全文搜索，用户可以按照时间筛选最终结果。"
        second = "系统支持全文搜索，用户还可以按照时间筛选最终结果。"
        markdown = first + second + "注意！注意！"
        result = self.normalizer.normalize(self.parser.parse(markdown, "d-near"))

        self.assertEqual(result.document.blocks[0].text, markdown)
        self.assertEqual(result.stats.duplicate_sentences_removed, 0)

    def test_prose_spacing_is_normalized_without_changing_mixed_single_spaces(self) -> None:
        markdown = "检 索  系统 ，可以使用 Qwen3 模型；（ 正文 ）"
        result = self.normalizer.normalize(self.parser.parse(markdown, "d-spacing"))
        block = result.document.blocks[0]

        self.assertEqual(block.text, "检索系统，可以使用 Qwen3 模型；（正文）")
        self.assertEqual(result.stats.spacing_blocks_repaired, 1)
        self.assertEqual(result.stats.extra_spaces_removed, 1)
        self.assertGreaterEqual(result.stats.cjk_spaces_removed, 5)
        repairs = block.metadata.get("repairs", [])
        self.assertEqual(repairs[-1].get("code", ""), "prose_spacing_normalized")

    def test_list_and_table_spacing_are_not_rewritten(self) -> None:
        markdown = """- 中 文列表项

| 名称 | 说明 |
| --- | --- |
| 中 文 | 保 留 |
"""
        result = self.normalizer.normalize(self.parser.parse(markdown, "d-structured-spacing"))

        self.assertIn("中 文列表项", result.document.blocks[0].text)
        self.assertIn("中 文", result.document.blocks[1].text)
        self.assertEqual(result.stats.spacing_blocks_repaired, 0)


class PipelineTest(unittest.TestCase):
    def test_default_minimum_is_300_approximate_tokens_for_all_content_types(self) -> None:
        self.assertEqual(ChunkConfig().target_min_tokens, 300)
        self.assertEqual(RuleConfig().hard_min_tokens, 300)
        self.assertEqual(RuleConfig().structured_hard_min_tokens, 300)
        self.assertEqual(RuleConfig().soft_min_tokens, 300)

        counter = ApproxTokenCounter()
        rules = RuleEngine(RuleConfig(), token_counter=counter)
        below = _fragment_with_content(" ".join("word%d" % index for index in range(298)) + "!", counter)
        boundary = _fragment_with_content(" ".join("word%d" % index for index in range(299)) + "!", counter)

        self.assertEqual(below.token_count, 299)
        self.assertEqual(rules.evaluate_fragment(below).decision, "rejected")
        self.assertEqual(boundary.token_count, 300)
        self.assertEqual(rules.evaluate_fragment(boundary).decision, "accepted")

    def test_pipeline_separates_accepted_review_rejected_and_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            rows = []
            for index in range(4):
                rows.append(
                    {
                        "doc_id": "d%d" % index,
                        "url": "https://example.com/%d" % index,
                        "payload": {
                            "markdown": """# 标题%d

本文完整介绍检索系统中的第%d个具体模块。该模块首先读取经过规范化的输入，然后生成候选并计算相关特征。完成初步处理后，系统使用独立验证集合检查结果，最后记录可以复现的实验结论。

进一步分析表明，候选数量和排序开销之间存在明确权衡。工程实现需要同时监控召回率、延迟和错误率，不能根据单一离线指标决定最终配置。

版权所有 © 2026 Example
""" % (index, index)
                        },
                    }
                )
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n{bad json\n",
                encoding="utf-8",
            )
            output = OutputConfig(
                accepted_path=str(root / "accepted.jsonl"),
                review_path=str(root / "review.jsonl"),
                rejected_path=str(root / "rejected.jsonl"),
                templates_path=str(root / "templates.json"),
                statistics_path=str(root / "statistics.json"),
                preview_path=str(root / "preview.md"),
                preview_fragments=10,
            )
            config = CleanerConfig(
                input_path=str(input_path),
                input=InputConfig(markdown_key="payload.markdown", id_key="doc_id", url_key="url"),
                output=output,
                content=ContentPolicy(keep_code_blocks=False),
                chunk=ChunkConfig(target_min_tokens=30, target_max_tokens=200, hard_max_tokens=300),
                rules=RuleConfig(
                    hard_min_tokens=20,
                    structured_hard_min_tokens=10,
                    soft_min_tokens=30,
                    hard_max_tokens=300,
                ),
                templates=TemplateConfig(
                    enabled=True,
                    min_host_documents=3,
                    min_template_documents=3,
                    min_document_fraction=0.5,
                    min_edge_ratio=0.5,
                    ubiquitous_document_fraction=0.75,
                ),
            )
            summary = clean_jsonl(config)
            accepted = _read_jsonl(output.accepted_path)
            rejected = _read_jsonl(output.rejected_path)

            self.assertEqual(summary.invalid_json_rows, 1)
            self.assertEqual(summary.template_blocks_rejected, 4)
            self.assertEqual(len(accepted), 4)
            self.assertTrue(
                all("版权所有" not in value.get("content", "") for value in accepted)
            )
            self.assertTrue(any(value.get("reason") == "invalid_json" for value in rejected))
            self.assertTrue(Path(output.preview_path).exists())

    def test_context_dependent_fragment_goes_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "doc_id": "d",
                        "content": "# 标题\n\n因此，该方法必须结合上一节的配置才能工作。随后系统执行检查并输出最终结果。",
                    },
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            output = OutputConfig(
                accepted_path=str(root / "a.jsonl"),
                review_path=str(root / "v.jsonl"),
                rejected_path=str(root / "r.jsonl"),
                templates_path=str(root / "t.json"),
                statistics_path=str(root / "s.json"),
                preview_path=str(root / "p.md"),
            )
            config = CleanerConfig(
                input_path=str(source),
                output=output,
                chunk=ChunkConfig(target_min_tokens=10, target_max_tokens=100, hard_max_tokens=150),
                rules=RuleConfig(
                    hard_min_tokens=5,
                    structured_hard_min_tokens=5,
                    soft_min_tokens=10,
                    hard_max_tokens=150,
                ),
                templates=TemplateConfig(enabled=False),
            )
            clean_jsonl(config)
            review = _read_jsonl(output.review_path)
            self.assertEqual(len(review), 1)
            flags = review[0].get("flags", [])
            self.assertIn(
                "context_dependent_start",
                {flag.get("code", "") for flag in flags},
            )

    def test_pipeline_repairs_repeated_sentences_before_chunking_and_reports_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = "检索系统首先规范化用户查询，并根据索引统计生成一批候选文档。"
            second = "随后排序模块结合相关性特征计算得分，并输出稳定且可复现的结果。"
            final = "最终结果还会经过独立校验，以便及时发现数据或配置异常。"
            source = root / "input.jsonl"
            source.write_text(
                json.dumps(
                    {"doc_id": "d", "content": first + second + first + second + final},
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            output = OutputConfig(
                accepted_path=str(root / "a.jsonl"),
                review_path=str(root / "v.jsonl"),
                rejected_path=str(root / "r.jsonl"),
                templates_path=str(root / "t.json"),
                statistics_path=str(root / "s.json"),
                preview_path=str(root / "p.md"),
            )
            config = CleanerConfig(
                input_path=str(source),
                output=output,
                chunk=ChunkConfig(target_min_tokens=20, target_max_tokens=200, hard_max_tokens=300),
                rules=RuleConfig(
                    hard_min_tokens=20,
                    structured_hard_min_tokens=20,
                    soft_min_tokens=20,
                    hard_max_tokens=300,
                ),
                templates=TemplateConfig(enabled=False),
            )

            summary = clean_jsonl(config)
            accepted = _read_jsonl(output.accepted_path)

            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0].get("content", ""), first + second + final)
            self.assertEqual(summary.duplicate_sentence_sequences_removed, 1)
            self.assertEqual(summary.duplicate_sentences_removed, 2)
            metadata = accepted[0].get("metadata", {})
            repairs = metadata.get("repairs", [])
            self.assertEqual(
                repairs[0].get("code", ""),
                "repeated_sentence_sequences_removed",
            )

    def test_both_mode_keeps_fragment_schema_and_emits_one_cleaned_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "doc_id": "d-both",
                        "content": """# 用户信息

姓名：张三
电话：13800000000
地址：上海

## 检索流程

检索系统先规范化查询，再生成候选文档，最后依据相关性得分输出完整结果。

上一篇
""",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = _output_config(root)
            config = CleanerConfig(
                input_path=str(source),
                output=output,
                chunk=ChunkConfig(target_min_tokens=1, target_max_tokens=500, hard_max_tokens=1000),
                rules=RuleConfig(
                    hard_min_tokens=1,
                    structured_hard_min_tokens=1,
                    soft_min_tokens=1,
                    hard_max_tokens=1000,
                ),
                templates=TemplateConfig(enabled=False),
                assembly=AssemblyConfig(
                    output_mode="both",
                    document_min_tokens=1,
                    document_max_tokens=1000,
                ),
            )

            summary = clean_jsonl(config)
            fragments = _read_jsonl(output.accepted_path) + _read_jsonl(output.review_path)
            documents = (
                _read_jsonl(output.document_accepted_path)
                + _read_jsonl(output.document_review_path)
            )

            self.assertGreaterEqual(len(fragments), 1)
            self.assertTrue(all("fragment_id" in fragment for fragment in fragments))
            self.assertTrue(all("record_type" not in fragment for fragment in fragments))
            self.assertEqual(len(documents), 1)
            whole = documents[0]
            self.assertEqual(whole.get("record_type", ""), "whole_document")
            content = whole.get("content", "")
            self.assertIn("姓名：张三\n电话：13800000000\n地址：上海", content)
            self.assertIn("\n\n", content)
            self.assertNotIn("用户信息", content)
            self.assertNotIn("检索流程", content)
            self.assertNotIn("上一篇", content)
            sections = whole.get("sections", [])
            self.assertEqual([section.get("heading_path", []) for section in sections], [
                ["用户信息"],
                ["用户信息", "检索流程"],
            ])
            self.assertTrue(
                all(
                    section.get("start_char", 0) < section.get("end_char", 0)
                    for section in sections
                )
            )
            removed_blocks = whole.get("removed_blocks", [])
            self.assertEqual(removed_blocks[0].get("reason", ""), "hard_rule")
            self.assertEqual(summary.rule_blocks_rejected, 1)
            self.assertEqual(summary.accepted_documents + summary.review_documents, 1)

    def test_document_mode_routes_over_max_to_review_without_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            prose = (
                "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
                "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu."
            )
            source.write_text(
                json.dumps({"doc_id": "d-long", "content": prose}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = _output_config(root)
            config = CleanerConfig(
                input_path=str(source),
                output=output,
                templates=TemplateConfig(enabled=False),
                assembly=AssemblyConfig(
                    output_mode="document",
                    document_min_tokens=5,
                    document_max_tokens=20,
                    document_over_max_policy="review",
                ),
            )

            summary = clean_jsonl(config)
            review = _read_jsonl(output.document_review_path)

            self.assertEqual(len(review), 1)
            review_record = review[0]
            self.assertEqual(review_record.get("content", ""), prose)
            self.assertGreater(review_record.get("token_count", 0), 20)
            flags = review_record.get("flags", [])
            self.assertIn(
                "whole_document_too_long",
                {flag.get("code", "") for flag in flags},
            )
            self.assertEqual(summary.review_documents, 1)
            self.assertFalse(Path(output.accepted_path).exists())
            self.assertFalse(Path(output.review_path).exists())
            self.assertFalse(Path(output.rejected_path).exists())

    def test_document_mode_rejects_document_below_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps({"doc_id": "d-short", "content": "这是一段完整但很短的正文。"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            output = _output_config(root)
            config = CleanerConfig(
                input_path=str(source),
                output=output,
                templates=TemplateConfig(enabled=False),
                assembly=AssemblyConfig(
                    output_mode="document",
                    document_min_tokens=50,
                    document_max_tokens=100,
                ),
            )

            summary = clean_jsonl(config)
            rejected = _read_jsonl(output.document_rejected_path)

            self.assertEqual(len(rejected), 1)
            rejected_record = rejected[0]
            self.assertEqual(rejected_record.get("kind", ""), "whole_document")
            document = rejected_record.get("document", {})
            self.assertEqual(document.get("content", ""), "这是一段完整但很短的正文。")
            self.assertIn(
                "too_short",
                {flag.get("code", "") for flag in document.get("flags", [])},
            )
            self.assertEqual(summary.rejected_documents, 1)


def _read_jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _output_config(root: Path) -> OutputConfig:
    return OutputConfig(
        accepted_path=str(root / "fragments" / "accepted.jsonl"),
        review_path=str(root / "fragments" / "review.jsonl"),
        rejected_path=str(root / "fragments" / "rejected.jsonl"),
        templates_path=str(root / "templates.json"),
        statistics_path=str(root / "statistics.json"),
        preview_path=str(root / "fragments" / "preview.md"),
        document_accepted_path=str(root / "documents" / "accepted.jsonl"),
        document_review_path=str(root / "documents" / "review.jsonl"),
        document_rejected_path=str(root / "documents" / "rejected.jsonl"),
        document_preview_path=str(root / "documents" / "preview.md"),
    )


def _fragment_with_content(content: str, counter: ApproxTokenCounter) -> Fragment:
    return Fragment(
        fragment_id="fragment",
        doc_id="document",
        url="",
        doc_title="",
        heading_path=[],
        content=content,
        block_ids=["block"],
        block_types=["paragraph"],
        token_count=counter.count(content),
        start_line=1,
        end_line=1,
        source_row=1,
    )


if __name__ == "__main__":
    unittest.main()
