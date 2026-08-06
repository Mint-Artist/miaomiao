from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .chunker import SemanticChunker
from .config import CleanerConfig
from .markdown_parser import MarkdownDocumentParser
from .models import Document, Fragment, PreparedBlock
from .rules import RuleEngine
from .templates import TemplateIndex
from .tokenization import ApproxTokenCounter, HuggingFaceTokenCounter, TokenCounter


@dataclass
class CleaningSummary:
    input_rows: int = 0
    invalid_json_rows: int = 0
    missing_markdown_rows: int = 0
    parse_error_rows: int = 0
    documents: int = 0
    parsed_blocks: int = 0
    template_blocks_rejected: int = 0
    rule_blocks_rejected: int = 0
    accepted_fragments: int = 0
    review_fragments: int = 0
    rejected_fragments: int = 0
    accepted_tokens: int = 0
    review_tokens: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    review_reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["rejection_reasons"] = dict(self.rejection_reasons.most_common())
        value["review_reasons"] = dict(self.review_reasons.most_common())
        return value


def clean_jsonl(config: CleanerConfig) -> CleaningSummary:
    config.validate()
    _validate_output_paths(config)
    counter: TokenCounter
    if config.tokenizer_name_or_path:
        counter = HuggingFaceTokenCounter(config.tokenizer_name_or_path)
    else:
        counter = ApproxTokenCounter()

    parser = MarkdownDocumentParser(config.content)
    rules = RuleEngine(config.rules, config.content, counter)
    chunker = SemanticChunker(config.chunk, counter)

    templates = TemplateIndex(config.templates)
    templates.fit(_iter_valid_documents(config, parser))
    templates.save(config.output.templates_path)

    summary = CleaningSummary()
    preview: List[Fragment] = []
    with (
        _AtomicJsonlWriter(config.output.accepted_path) as accepted_writer,
        _AtomicJsonlWriter(config.output.review_path) as review_writer,
        _AtomicJsonlWriter(config.output.rejected_path) as rejected_writer,
    ):
        for row_number, row, error, raw_line in _iter_rows(config.input_path):
            summary.input_rows += 1
            if error is not None:
                summary.invalid_json_rows += 1
                summary.rejection_reasons["invalid_json"] += 1
                rejected_writer.write(
                    {
                        "kind": "input_row",
                        "decision": "rejected",
                        "reason": "invalid_json",
                        "source_row": row_number,
                        "error": error,
                        "raw_line_preview": raw_line[:1000],
                    }
                )
                continue

            markdown = _optional_path(row, config.input.markdown_key)
            if not isinstance(markdown, str) or not markdown.strip():
                summary.missing_markdown_rows += 1
                summary.rejection_reasons["missing_markdown"] += 1
                rejected_writer.write(
                    {
                        "kind": "input_row",
                        "decision": "rejected",
                        "reason": "missing_markdown",
                        "source_row": row_number,
                    }
                )
                continue

            try:
                document = _document_from_row(config, parser, row, row_number, markdown)
            except Exception as exc:
                summary.parse_error_rows += 1
                summary.rejection_reasons["markdown_parse_error"] += 1
                rejected_writer.write(
                    {
                        "kind": "input_row",
                        "decision": "rejected",
                        "reason": "markdown_parse_error",
                        "source_row": row_number,
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    }
                )
                continue

            summary.documents += 1
            summary.parsed_blocks += len(document.blocks)
            prepared: List[Optional[PreparedBlock]] = []
            for block in document.blocks:
                template = templates.match(document.url, block)
                if template is not None:
                    summary.template_blocks_rejected += 1
                    summary.rejection_reasons["site_template"] += 1
                    rejected_writer.write(
                        {
                            "kind": "block",
                            "decision": "rejected",
                            "reason": "site_template",
                            "doc_id": document.doc_id,
                            "url": document.url,
                            "source_row": document.source_row,
                            "block": block.to_dict(),
                            "template": template.to_dict(),
                        }
                    )
                    prepared.append(None)
                    continue

                block_result = rules.evaluate_block(block)
                if block_result.decision == "rejected":
                    summary.rule_blocks_rejected += 1
                    hard_codes = [flag.code for flag in block_result.flags if flag.severity == "hard"]
                    summary.rejection_reasons.update(hard_codes)
                    rejected_writer.write(
                        {
                            "kind": "block",
                            "decision": "rejected",
                            "reason": "hard_rule",
                            "doc_id": document.doc_id,
                            "url": document.url,
                            "source_row": document.source_row,
                            "block": block.to_dict(),
                            "flags": [flag.to_dict() for flag in block_result.flags],
                        }
                    )
                    prepared.append(None)
                else:
                    prepared.append(PreparedBlock(block, block_result.flags))

            for fragment in chunker.chunk_document(document, prepared):
                result = rules.evaluate_fragment(fragment)
                fragment.flags = result.flags
                fragment.decision = result.decision
                if result.decision == "accepted":
                    summary.accepted_fragments += 1
                    summary.accepted_tokens += fragment.token_count
                    accepted_writer.write(fragment.to_dict())
                elif result.decision == "review":
                    summary.review_fragments += 1
                    summary.review_tokens += fragment.token_count
                    summary.review_reasons.update(
                        flag.code for flag in result.flags if flag.severity == "soft"
                    )
                    review_writer.write(fragment.to_dict())
                else:
                    summary.rejected_fragments += 1
                    summary.rejection_reasons.update(
                        flag.code for flag in result.flags if flag.severity == "hard"
                    )
                    rejected_writer.write(
                        {
                            "kind": "fragment",
                            "decision": "rejected",
                            "reason": "hard_rule",
                            "fragment": fragment.to_dict(),
                        }
                    )
                if (
                    result.decision in {"accepted", "review"}
                    and len(preview) < config.output.preview_fragments
                ):
                    preview.append(fragment)

    _write_statistics(config, summary, len(templates.entries))
    _write_preview(config.output.preview_path, preview, summary)
    return summary


def _iter_valid_documents(
    config: CleanerConfig,
    parser: MarkdownDocumentParser,
) -> Iterable[Document]:
    for row_number, row, error, _ in _iter_rows(config.input_path):
        if error is not None:
            continue
        markdown = _optional_path(row, config.input.markdown_key)
        if not isinstance(markdown, str) or not markdown.strip():
            continue
        try:
            yield _document_from_row(config, parser, row, row_number, markdown)
        except Exception:
            continue


def _document_from_row(
    config: CleanerConfig,
    parser: MarkdownDocumentParser,
    row: Mapping[str, Any],
    row_number: int,
    markdown: str,
) -> Document:
    doc_id = _optional_path(row, config.input.id_key) if config.input.id_key else None
    url = _optional_path(row, config.input.url_key) if config.input.url_key else None
    title = _optional_path(row, config.input.title_key) if config.input.title_key else None
    return parser.parse(
        markdown=markdown,
        doc_id=str(doc_id) if doc_id not in {None, ""} else "row-%d" % row_number,
        url=str(url) if url not in {None, ""} else "",
        title=str(title) if title not in {None, ""} else "",
        source_row=row_number,
    )


def _iter_rows(path: str) -> Iterator[Tuple[int, Mapping[str, Any], Optional[str], str]]:
    with open(path, "r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("JSON value is not an object")
            except Exception as exc:
                yield row_number, {}, "%s: %s" % (type(exc).__name__, exc), line.rstrip("\n")
                continue
            yield row_number, value, None, line.rstrip("\n")


def _optional_path(value: Mapping[str, Any], dotted_path: Optional[str]) -> Any:
    if not dotted_path:
        return None
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


class _AtomicJsonlWriter:
    def __init__(self, path: str) -> None:
        self.target = Path(path)
        self.temporary = ""
        self.handle = None

    def __enter__(self) -> "_AtomicJsonlWriter":
        self.target.parent.mkdir(parents=True, exist_ok=True)
        fd, self.temporary = tempfile.mkstemp(
            prefix=self.target.name + ".", suffix=".tmp", dir=str(self.target.parent)
        )
        self.handle = os.fdopen(fd, "w", encoding="utf-8")
        return self

    def write(self, value: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("writer is not open")
        self.handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if exc_type is None:
            os.replace(self.temporary, self.target)
        else:
            try:
                os.unlink(self.temporary)
            except FileNotFoundError:
                pass


def _write_statistics(config: CleanerConfig, summary: CleaningSummary, template_entries: int) -> None:
    target = Path(config.output.statistics_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary.to_dict(),
        "template_entries": template_entries,
        "config": asdict(config),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_preview(path: str, fragments: List[Fragment], summary: CleaningSummary) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = [
        "# Markdown 清洗预览",
        "",
        "- 自动接收片段：`%d`" % summary.accepted_fragments,
        "- 灰区片段：`%d`" % summary.review_fragments,
        "- 拒绝片段：`%d`" % summary.rejected_fragments,
        "",
    ]
    for index, fragment in enumerate(fragments, 1):
        flags = ", ".join(flag.code for flag in fragment.flags) or "none"
        heading = " > ".join(fragment.heading_path) or "（无标题路径）"
        output.extend(
            [
                "## %d. %s" % (index, _one_line(fragment.doc_title)),
                "",
                "- 状态：`%s`" % fragment.decision,
                "- 标题路径：`%s`" % _one_line(heading),
                "- Token：`%d`" % fragment.token_count,
                "- 源行：`%d-%d`" % (fragment.start_line, fragment.end_line),
                "- 标记：`%s`" % _one_line(flags),
                "",
                fragment.content,
                "",
                "---",
                "",
            ]
        )
    target.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def _one_line(value: Any) -> str:
    return str(value).replace("`", "'").replace("\n", " ").strip()


def _validate_output_paths(config: CleanerConfig) -> None:
    input_path = Path(config.input_path).expanduser().resolve()
    paths = [
        config.output.accepted_path,
        config.output.review_path,
        config.output.rejected_path,
        config.output.templates_path,
        config.output.statistics_path,
        config.output.preview_path,
    ]
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if input_path in resolved:
        raise ValueError("an output path must not overwrite the input JSONL")
    if len(set(resolved)) != len(resolved):
        raise ValueError("all output paths must be distinct")
