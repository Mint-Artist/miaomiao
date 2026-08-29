from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from cleaner.assemblers import FragmentAssembler, WholeDocumentAssembler
from cleaner.chunker import SemanticChunker
from cleaner.config import CleanerConfig
from cleaner.markdown_parser import MarkdownDocumentParser
from cleaner.models import Document, Fragment, PreparedBlock, WholeDocumentRecord
from cleaner.normalizer import DocumentNormalizer, NormalizationStats
from cleaner.rules import RuleEngine
from cleaner.templates import TemplateIndex
from cleaner.tokenization import (
    ApproxTokenCounter,
    HuggingFaceTokenCounter,
    TokenCounter,
)


@dataclass
class CleaningSummary:
    input_rows: int = 0
    invalid_json_rows: int = 0
    missing_markdown_rows: int = 0
    parse_error_rows: int = 0
    documents: int = 0
    parsed_blocks: int = 0
    documents_repaired: int = 0
    blocks_repaired: int = 0
    boundary_artifacts_removed: int = 0
    empty_blocks_removed: int = 0
    spacing_blocks_repaired: int = 0
    extra_spaces_removed: int = 0
    cjk_spaces_removed: int = 0
    duplicate_blocks_removed: int = 0
    duplicate_sentence_sequences_removed: int = 0
    duplicate_sentences_removed: int = 0
    duplicate_chars_removed: int = 0
    template_blocks_rejected: int = 0
    rule_blocks_rejected: int = 0
    accepted_fragments: int = 0
    review_fragments: int = 0
    rejected_fragments: int = 0
    accepted_tokens: int = 0
    review_tokens: int = 0
    accepted_documents: int = 0
    review_documents: int = 0
    rejected_documents: int = 0
    accepted_document_tokens: int = 0
    review_document_tokens: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    review_reasons: Counter[str] = field(default_factory=Counter)
    document_rejection_reasons: Counter[str] = field(default_factory=Counter)
    document_review_reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["rejection_reasons"] = dict(self.rejection_reasons.most_common())
        value["review_reasons"] = dict(self.review_reasons.most_common())
        value["document_rejection_reasons"] = dict(self.document_rejection_reasons.most_common())
        value["document_review_reasons"] = dict(self.document_review_reasons.most_common())
        return value


def clean_jsonl(config: CleanerConfig) -> CleaningSummary:
    config.validate()
    _validate_output_paths(config)
    counter: TokenCounter
    if config.tokenizer_name_or_path:
        counter = HuggingFaceTokenCounter(config.tokenizer_name_or_path)
    else:
        counter = ApproxTokenCounter()

    parser = MarkdownDocumentParser(config.content, config.normalization)
    normalizer = DocumentNormalizer(config.normalization)
    rules = RuleEngine(config.rules, config.content, counter)
    chunker = SemanticChunker(config.chunk, counter)
    fragment_assembler = FragmentAssembler(chunker)
    document_assembler = WholeDocumentAssembler(counter)
    emit_fragments = config.assembly.output_mode in {"fragment", "both"}
    emit_documents = config.assembly.output_mode in {"document", "both"}

    templates = TemplateIndex(config.templates)
    templates.fit(_iter_valid_documents(config, parser, normalizer))
    if config.templates.enabled:
        templates.save(config.output.templates_path)

    summary = CleaningSummary()
    fragment_preview: List[Fragment] = []
    document_preview: List[WholeDocumentRecord] = []
    with ExitStack() as stack:
        fragment_writers: Optional[Tuple[_AtomicJsonlWriter, _AtomicJsonlWriter, _AtomicJsonlWriter]] = None
        document_writers: Optional[Tuple[_AtomicJsonlWriter, _AtomicJsonlWriter, _AtomicJsonlWriter]] = None
        if emit_fragments:
            fragment_writers = (
                stack.enter_context(_AtomicJsonlWriter(config.output.accepted_path)),
                stack.enter_context(_AtomicJsonlWriter(config.output.review_path)),
                stack.enter_context(_AtomicJsonlWriter(config.output.rejected_path)),
            )
        if emit_documents:
            document_writers = (
                stack.enter_context(_AtomicJsonlWriter(config.output.document_accepted_path)),
                stack.enter_context(_AtomicJsonlWriter(config.output.document_review_path)),
                stack.enter_context(_AtomicJsonlWriter(config.output.document_rejected_path)),
            )

        for row_number, row, error, raw_line in _iter_rows(config.input_path):
            summary.input_rows += 1
            if error is not None:
                summary.invalid_json_rows += 1
                summary.rejection_reasons["invalid_json"] += 1
                payload = {
                    "kind": "input_row",
                    "decision": "rejected",
                    "reason": "invalid_json",
                    "source_row": row_number,
                    "error": error,
                    "raw_line_preview": raw_line[:1000],
                }
                _write_rejection_to_active_modes(payload, fragment_writers, document_writers)
                continue

            markdown = _optional_path(row, config.input.markdown_key)
            if not isinstance(markdown, str) or not markdown.strip():
                summary.missing_markdown_rows += 1
                summary.rejection_reasons["missing_markdown"] += 1
                payload = {
                    "kind": "input_row",
                    "decision": "rejected",
                    "reason": "missing_markdown",
                    "source_row": row_number,
                }
                _write_rejection_to_active_modes(payload, fragment_writers, document_writers)
                continue

            try:
                document = _document_from_row(config, parser, row, row_number, markdown)
                parsed_block_count = len(document.blocks)
                normalization = normalizer.normalize(document)
                document = normalization.document
            except Exception as exc:
                summary.parse_error_rows += 1
                summary.rejection_reasons["markdown_parse_error"] += 1
                payload = {
                    "kind": "input_row",
                    "decision": "rejected",
                    "reason": "markdown_parse_error",
                    "source_row": row_number,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
                _write_rejection_to_active_modes(payload, fragment_writers, document_writers)
                continue

            _record_normalization_stats(summary, normalization.stats)
            summary.documents += 1
            summary.parsed_blocks += parsed_block_count
            prepared: List[Optional[PreparedBlock]] = []
            removed_blocks: List[Dict[str, Any]] = []
            for block in document.blocks:
                template = templates.match(document.url, block)
                if template is not None:
                    summary.template_blocks_rejected += 1
                    summary.rejection_reasons["site_template"] += 1
                    if fragment_writers is not None:
                        fragment_writers[2].write(
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
                    removed_blocks.append({"block_id": block.block_id, "reason": "site_template"})
                    prepared.append(None)
                    continue

                block_result = rules.evaluate_block(block)
                if block_result.decision == "rejected":
                    summary.rule_blocks_rejected += 1
                    hard_codes = [flag.code for flag in block_result.flags if flag.severity == "hard"]
                    summary.rejection_reasons.update(hard_codes)
                    if fragment_writers is not None:
                        fragment_writers[2].write(
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
                    removed_blocks.append(
                        {"block_id": block.block_id, "reason": "hard_rule", "flags": hard_codes}
                    )
                    prepared.append(None)
                else:
                    prepared.append(PreparedBlock(block, block_result.flags))

            if fragment_writers is not None:
                for fragment in fragment_assembler.assemble(document, prepared):
                    _route_fragment(
                        fragment,
                        rules,
                        fragment_writers,
                        summary,
                        fragment_preview,
                        config.output.preview_fragments,
                    )

            if document_writers is not None:
                whole = document_assembler.assemble(document, prepared, removed_blocks)
                if whole is None:
                    summary.rejected_documents += 1
                    summary.document_rejection_reasons["empty_after_cleaning"] += 1
                    document_writers[2].write(
                        {
                            "kind": "whole_document",
                            "decision": "rejected",
                            "reason": "empty_after_cleaning",
                            "doc_id": document.doc_id,
                            "url": document.url,
                            "source_row": document.source_row,
                            "removed_blocks": removed_blocks,
                        }
                    )
                else:
                    _route_document(
                        whole,
                        rules,
                        config,
                        document_writers,
                        summary,
                        document_preview,
                        config.output.preview_documents,
                    )

    _write_statistics(config, summary, len(templates.entries))
    if emit_fragments and config.output.write_fragment_preview:
        _write_preview(config.output.preview_path, fragment_preview, summary)
    if emit_documents and config.output.write_document_preview:
        _write_document_preview(config.output.document_preview_path, document_preview, summary)
    return summary


def _write_rejection_to_active_modes(
    payload: Mapping[str, Any],
    fragment_writers: Optional[Tuple[_AtomicJsonlWriter, _AtomicJsonlWriter, _AtomicJsonlWriter]],
    document_writers: Optional[Tuple[_AtomicJsonlWriter, _AtomicJsonlWriter, _AtomicJsonlWriter]],
) -> None:
    if fragment_writers is not None:
        fragment_writers[2].write(payload)
    if document_writers is not None:
        document_writers[2].write(payload)


def _route_fragment(
    fragment: Fragment,
    rules: RuleEngine,
    writers: Tuple[_AtomicJsonlWriter, _AtomicJsonlWriter, _AtomicJsonlWriter],
    summary: CleaningSummary,
    preview: List[Fragment],
    preview_limit: int,
) -> None:
    result = rules.evaluate_fragment(fragment)
    fragment.flags = result.flags
    fragment.decision = result.decision
    if result.decision == "accepted":
        summary.accepted_fragments += 1
        summary.accepted_tokens += fragment.token_count
        writers[0].write(fragment.to_dict())
    elif result.decision == "review":
        summary.review_fragments += 1
        summary.review_tokens += fragment.token_count
        summary.review_reasons.update(flag.code for flag in result.flags if flag.severity == "soft")
        writers[1].write(fragment.to_dict())
    else:
        summary.rejected_fragments += 1
        summary.rejection_reasons.update(flag.code for flag in result.flags if flag.severity == "hard")
        writers[2].write(
            {
                "kind": "fragment",
                "decision": "rejected",
                "reason": "hard_rule",
                "fragment": fragment.to_dict(),
            }
        )
    if result.decision in {"accepted", "review"} and len(preview) < preview_limit:
        preview.append(fragment)


def _route_document(
    document: WholeDocumentRecord,
    rules: RuleEngine,
    config: CleanerConfig,
    writers: Tuple[_AtomicJsonlWriter, _AtomicJsonlWriter, _AtomicJsonlWriter],
    summary: CleaningSummary,
    preview: List[WholeDocumentRecord],
    preview_limit: int,
) -> None:
    result = rules.evaluate_document(document, config.assembly)
    document.flags = result.flags
    document.decision = result.decision
    if result.decision == "accepted":
        summary.accepted_documents += 1
        summary.accepted_document_tokens += document.token_count
        writers[0].write(document.to_dict())
    elif result.decision == "review":
        summary.review_documents += 1
        summary.review_document_tokens += document.token_count
        summary.document_review_reasons.update(
            flag.code for flag in result.flags if flag.severity == "soft"
        )
        writers[1].write(document.to_dict())
    else:
        summary.rejected_documents += 1
        summary.document_rejection_reasons.update(
            flag.code for flag in result.flags if flag.severity == "hard"
        )
        writers[2].write(
            {
                "kind": "whole_document",
                "decision": "rejected",
                "reason": "hard_rule",
                "document": document.to_dict(),
            }
        )
    if result.decision in {"accepted", "review"} and len(preview) < preview_limit:
        preview.append(document)


def _iter_valid_documents(
    config: CleanerConfig,
    parser: MarkdownDocumentParser,
    normalizer: DocumentNormalizer,
) -> Iterable[Document]:
    for row_number, row, error, _ in _iter_rows(config.input_path):
        if error is not None:
            continue
        markdown = _optional_path(row, config.input.markdown_key)
        if not isinstance(markdown, str) or not markdown.strip():
            continue
        try:
            document = _document_from_row(config, parser, row, row_number, markdown)
            yield normalizer.normalize(document).document
        except Exception:
            continue


def _record_normalization_stats(
    summary: CleaningSummary,
    stats: NormalizationStats,
) -> None:
    for name in (
        "documents_repaired",
        "blocks_repaired",
        "boundary_artifacts_removed",
        "empty_blocks_removed",
        "spacing_blocks_repaired",
        "extra_spaces_removed",
        "cjk_spaces_removed",
        "duplicate_blocks_removed",
        "duplicate_sentence_sequences_removed",
        "duplicate_sentences_removed",
        "duplicate_chars_removed",
    ):
        setattr(summary, name, getattr(summary, name) + getattr(stats, name))


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
    fallback_id = "row-%d" % row_number
    if config.input.fallback_id_prefix:
        fallback_id = "%s:row-%d" % (config.input.fallback_id_prefix, row_number)
    return parser.parse(
        markdown=markdown,
        doc_id=str(doc_id) if doc_id not in {None, ""} else fallback_id,
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
        if not isinstance(current, Mapping):
            return None
        if part not in current:
            return None
        current = current.get(part)
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


def _write_preview(
    path: str,
    fragments: Sequence[Fragment],
    summary: CleaningSummary,
) -> None:
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


def _write_document_preview(
    path: str,
    documents: Sequence[WholeDocumentRecord],
    summary: CleaningSummary,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = [
        "# Markdown 全文清洗预览",
        "",
        "- 自动接收全文：`%d`" % summary.accepted_documents,
        "- 灰区全文：`%d`" % summary.review_documents,
        "- 拒绝全文：`%d`" % summary.rejected_documents,
        "",
    ]
    for index, document in enumerate(documents, 1):
        flags = ", ".join(flag.code for flag in document.flags) or "none"
        output.extend(
            [
                "## %d. %s" % (index, _one_line(document.doc_title)),
                "",
                "- 状态：`%s`" % document.decision,
                "- Token：`%d`" % document.token_count,
                "- 章节数：`%d`" % len(document.sections),
                "- 删除 block：`%d`" % len(document.removed_blocks),
                "- 源行：`%d-%d`" % (document.start_line, document.end_line),
                "- 标记：`%s`" % _one_line(flags),
                "",
                document.content,
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
    paths = [config.output.statistics_path]
    if config.templates.enabled:
        paths.append(config.output.templates_path)
    if config.assembly.output_mode in {"fragment", "both"}:
        paths.extend(
            [config.output.accepted_path, config.output.review_path, config.output.rejected_path]
        )
        if config.output.write_fragment_preview:
            paths.append(config.output.preview_path)
    if config.assembly.output_mode in {"document", "both"}:
        paths.extend(
            [
                config.output.document_accepted_path,
                config.output.document_review_path,
                config.output.document_rejected_path,
            ]
        )
        if config.output.write_document_preview:
            paths.append(config.output.document_preview_path)
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if input_path in resolved:
        raise ValueError("an output path must not overwrite the input JSONL")
    if len(set(resolved)) != len(resolved):
        raise ValueError("all output paths must be distinct")
