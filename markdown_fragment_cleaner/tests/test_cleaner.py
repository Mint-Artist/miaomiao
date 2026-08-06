from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cleaner.config import (
    ChunkConfig,
    CleanerConfig,
    ContentPolicy,
    InputConfig,
    OutputConfig,
    RuleConfig,
    TemplateConfig,
)
from cleaner.markdown_parser import MarkdownDocumentParser
from cleaner.pipeline import clean_jsonl


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


class PipelineTest(unittest.TestCase):
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
            self.assertTrue(all("版权所有" not in value["content"] for value in accepted))
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
            self.assertIn("context_dependent_start", {flag["code"] for flag in review[0]["flags"]})


def _read_jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
