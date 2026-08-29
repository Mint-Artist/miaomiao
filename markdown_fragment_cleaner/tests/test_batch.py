from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cleaner.batch import BatchConfig, clean_jsonl_shards
from cleaner.config import (
    AssemblyConfig,
    ChunkConfig,
    CleanerConfig,
    InputConfig,
    OutputConfig,
    RuleConfig,
    TemplateConfig,
)


class BatchCleaningTest(unittest.TestCase):
    def test_parallel_shards_preserve_paths_and_skip_completed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            (inputs / "news").mkdir(parents=True)
            first = inputs / "part-000.jsonl"
            second = inputs / "news" / "part-001.jsonl"
            _write_document(first, "第一篇文章完整介绍查询改写、候选召回和排序校验流程。")
            _write_document(second, "第二篇文章完整介绍索引构建、字段规范化和结果验证流程。")

            config = _base_config("document")
            batch = BatchConfig(
                input_dir=str(inputs),
                input_glob="**/*.jsonl",
                output_dir=str(outputs),
                max_workers=2,
                skip_completed_shards=True,
                write_shard_previews=False,
            )

            initial = clean_jsonl_shards(config, batch)

            self.assertEqual(initial.discovered_shards, 2)
            self.assertEqual(initial.completed_shards, 2)
            self.assertEqual(initial.failed_shards, 0)
            first_output = outputs / "documents" / "accepted" / "part-000.jsonl"
            second_output = outputs / "documents" / "accepted" / "news" / "part-001.jsonl"
            self.assertTrue(first_output.is_file())
            self.assertTrue(second_output.is_file())
            self.assertFalse((outputs / "fragments").exists())
            self.assertFalse((outputs / "metadata" / "templates").exists())
            self.assertFalse((outputs / "metadata" / "previews").exists())
            self.assertEqual(_read_jsonl(first_output)[0].get("doc_id", ""), "part-000:row-1")
            self.assertEqual(
                _read_jsonl(second_output)[0].get("doc_id", ""),
                "news/part-001:row-1",
            )
            self.assertTrue(
                (outputs / "metadata" / "completed" / "part-000.done.json").is_file()
            )
            self.assertTrue(
                (
                    outputs
                    / "metadata"
                    / "completed"
                    / "news"
                    / "part-001.done.json"
                ).is_file()
            )

            repeated = clean_jsonl_shards(config, batch)
            self.assertEqual(repeated.completed_shards, 0)
            self.assertEqual(repeated.skipped_shards, 2)
            self.assertEqual(repeated.aggregate.documents, 2)

            _write_document(
                second,
                "修改后的第二篇文章完整介绍索引构建、字段规范化和结果验证流程。",
            )
            changed = clean_jsonl_shards(config, batch)
            self.assertEqual(changed.completed_shards, 1)
            self.assertEqual(changed.skipped_shards, 1)
            self.assertIn(
                "修改后的第二篇",
                _read_jsonl(second_output)[0].get("content", ""),
            )

    def test_both_mode_writes_fragment_and_document_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            _write_document(
                inputs / "part.jsonl",
                "这篇文章完整说明如何规范化查询、生成候选文档并验证最终结果。",
            )

            summary = clean_jsonl_shards(
                _base_config("both"),
                BatchConfig(
                    input_dir=str(inputs),
                    output_dir=str(outputs),
                    max_workers=1,
                ),
            )

            self.assertEqual(summary.completed_shards, 1)
            self.assertTrue((outputs / "fragments" / "accepted" / "part.jsonl").is_file())
            self.assertTrue((outputs / "documents" / "accepted" / "part.jsonl").is_file())
            batch_summary = json.loads(
                (outputs / "metadata" / "batch_summary.json").read_text(encoding="utf-8")
            )
            summary_value = batch_summary.get("summary", {})
            self.assertEqual(summary_value.get("completed_shards", 0), 1)
            aggregate = summary_value.get("aggregate", {})
            self.assertEqual(aggregate.get("documents", 0), 1)


def _base_config(output_mode: str) -> CleanerConfig:
    return CleanerConfig(
        input_path="__batch__",
        input=InputConfig(
            markdown_key="content",
            id_key="doc_id",
            url_key="url",
            title_key="title",
        ),
        output=OutputConfig(
            accepted_path="unused/fragment-accepted.jsonl",
            review_path="unused/fragment-review.jsonl",
            rejected_path="unused/fragment-rejected.jsonl",
            templates_path="unused/templates.json",
            statistics_path="unused/statistics.json",
            preview_path="unused/fragment-preview.md",
            document_accepted_path="unused/document-accepted.jsonl",
            document_review_path="unused/document-review.jsonl",
            document_rejected_path="unused/document-rejected.jsonl",
            document_preview_path="unused/document-preview.md",
        ),
        chunk=ChunkConfig(target_min_tokens=1, target_max_tokens=500, hard_max_tokens=1000),
        rules=RuleConfig(
            hard_min_tokens=1,
            structured_hard_min_tokens=1,
            soft_min_tokens=1,
            hard_max_tokens=1000,
        ),
        templates=TemplateConfig(enabled=False),
        assembly=AssemblyConfig(
            output_mode=output_mode,
            document_min_tokens=1,
            document_max_tokens=1000,
        ),
    )


def _write_document(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": content}, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
