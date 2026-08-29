from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, Mapping, Optional, Sequence

from sequence_BIO.alignment import AlignmentResult, extract_refined_text
from sequence_BIO.constants import (
    ALIGNMENT_ALGORITHM,
    DEFAULT_LABEL_TO_ID,
    DEFAULT_MAX_ADJUST_CHARS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_MATCH_CHARS,
    IGNORE_INDEX,
    INDEX_CONVENTION,
)
from sequence_BIO.labeling import TokenLabelResult, label_aligned_pair

LOGGER = logging.getLogger(__name__)
SUCCESS_EXIT_CODE = 0
PROCESSING_ERROR_EXIT_CODE = 2
COUNT_KEYS = (
    "processed",
    "accepted",
    "aligned",
    "adjusted",
    "unaligned",
    "truncated",
    "errors",
)


@dataclass(frozen=True)
class OutputPaths:
    accepted: Path
    sft: Path
    rejected: Path
    manifest: Path

    def prepare(self) -> None:
        _validate_distinct_paths(
            self.accepted,
            self.sft,
            self.rejected,
            self.manifest,
        )
        for path in (self.accepted, self.sft, self.rejected, self.manifest):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OutputStreams:
    accepted: IO[str]
    sft: IO[str]
    rejected: IO[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sequence-BIO",
        description=(
            "Build SELECT-style accepted audit, SFT, rejected, and manifest "
            "files from source/teacher JSONL pairs."
        ),
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument(
        "--output",
        required=True,
        help="Accepted audit JSONL path; related output paths are derived by default",
    )
    parser.add_argument("--sft-output", help="Minimal SFT JSONL path")
    parser.add_argument("--rejected-output", help="Rejected/error audit JSONL path")
    parser.add_argument("--manifest-output", help="Dataset manifest JSON path")
    parser.add_argument(
        "--tokenizer", required=True, help="Hugging Face tokenizer name or path"
    )
    parser.add_argument("--source-field", default="source_text")
    parser.add_argument("--refined-field", default="refined_text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--parse-teacher-response",
        action="store_true",
        help="Parse paper-style refined_text: [doc]...[/doc] from the refined field",
    )
    parser.add_argument(
        "--min-match-chars",
        "--min-span-chars",
        dest="min_match_chars",
        type=int,
        default=DEFAULT_MIN_MATCH_CHARS,
        help="Minimum exact character match length (paper default: 20)",
    )
    parser.add_argument(
        "--max-adjust-chars",
        type=int,
        default=DEFAULT_MAX_ADJUST_CHARS,
        help="Maximum source/target internal-gap length difference (paper default: 5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length; truncated samples are routed to rejected output",
    )
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--omit-match-text",
        action="store_true",
        help="Omit duplicated exact-match text from accepted audit records",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    if args.max_length < 1:
        raise ValueError("max_length must be at least 1")

    input_path = Path(args.input)
    output_paths = _resolve_output_paths(args)
    output_paths.prepare()
    tokenizer = _load_tokenizer(args.tokenizer, args.trust_remote_code)
    counts = {key: 0 for key in COUNT_KEYS}

    with ExitStack() as stack:
        input_stream = stack.enter_context(input_path.open("r", encoding="utf-8"))
        streams = OutputStreams(
            accepted=stack.enter_context(
                output_paths.accepted.open("w", encoding="utf-8")
            ),
            sft=stack.enter_context(output_paths.sft.open("w", encoding="utf-8")),
            rejected=stack.enter_context(
                output_paths.rejected.open("w", encoding="utf-8")
            ),
        )
        _process_input_stream(input_stream, streams, tokenizer, args, counts)

    _write_manifest(input_path, output_paths, args, counts)
    _log_summary(output_paths, counts)
    return (
        SUCCESS_EXIT_CODE
        if counts.get("errors", 0) == 0
        else PROCESSING_ERROR_EXIT_CODE
    )


def _process_input_stream(
    input_stream: IO[str],
    streams: OutputStreams,
    tokenizer: Any,
    args: argparse.Namespace,
    counts: Dict[str, int],
) -> None:
    for line_number, line in enumerate(input_stream, start=1):
        if not line.strip():
            continue
        _increment(counts, "processed")
        record: Any = None
        try:
            record = _parse_record(line)
            _process_record(record, streams, tokenizer, args, counts)
        except Exception as exc:  # Keep processing after malformed rows.
            _increment(counts, "errors")
            _write_processing_error(streams.rejected, record, line_number, exc)


def _parse_record(line: str) -> Mapping[str, Any]:
    record = json.loads(line)
    if not isinstance(record, Mapping):
        raise TypeError("each JSONL row must be an object")
    return record


def _process_record(
    record: Mapping[str, Any],
    streams: OutputStreams,
    tokenizer: Any,
    args: argparse.Namespace,
    counts: Dict[str, int],
) -> None:
    source_text = _required_string(record, args.source_field)
    refined_value = _required_string(record, args.refined_field)
    teacher_text = (
        extract_refined_text(refined_value)
        if args.parse_teacher_response
        else refined_value
    )
    alignment, token_result = label_aligned_pair(
        source_text,
        teacher_text,
        tokenizer,
        min_match_chars=args.min_match_chars,
        max_adjust_chars=args.max_adjust_chars,
        max_length=args.max_length,
        add_special_tokens=not args.no_special_tokens,
    )
    audit_record = _build_audit_record(
        record,
        source_text=source_text,
        teacher_refined_text=teacher_text,
        tokenizer_name=args.tokenizer,
        alignment=alignment,
        token_result=token_result,
        include_match_text=not args.omit_match_text,
    )
    _increment(counts, alignment.status)

    if not alignment.is_accepted:
        _write_rejected(streams.rejected, audit_record, "unaligned")
        return
    if token_result is None:
        raise RuntimeError("accepted alignment did not produce token labels")
    if token_result.truncated:
        _increment(counts, "truncated")
        _write_rejected(
            streams.rejected,
            audit_record,
            "tokenization_truncated",
        )
        return

    _increment(counts, "accepted")
    audit_record.update({"dataset_status": "accepted", "rejection_reason": None})
    _write_jsonl(streams.accepted, audit_record)
    _write_jsonl(
        streams.sft,
        token_result.to_sft_dict(record.get(args.id_field)),
    )


def _write_rejected(stream: IO[str], audit_record: Dict[str, Any], reason: str) -> None:
    audit_record.update({"dataset_status": "rejected", "rejection_reason": reason})
    _write_jsonl(stream, audit_record)


def _write_processing_error(
    stream: IO[str],
    record: Any,
    line_number: int,
    exception: Exception,
) -> None:
    error_record = dict(record) if isinstance(record, Mapping) else {}
    error_record.update(
        {
            "line_number": line_number,
            "dataset_status": "rejected",
            "rejection_reason": "processing_error",
            "error": f"{type(exception).__name__}: {exception}",
        }
    )
    _write_jsonl(stream, error_record)


def _build_audit_record(
    record: Mapping[str, Any],
    *,
    source_text: str,
    teacher_refined_text: str,
    tokenizer_name: str,
    alignment: AlignmentResult,
    token_result: Optional[TokenLabelResult],
    include_match_text: bool,
) -> Dict[str, Any]:
    value = dict(record)
    value.update(
        {
            "source_text": source_text,
            "teacher_refined_text": teacher_refined_text,
            "adjusted_refined_text": alignment.adjusted_refined_text,
            "alignment": alignment.to_dict(
                include_text=False,
                include_match_text=include_match_text,
            ),
        }
    )
    if token_result is None:
        value.update({"tokenization": None, "supervision": None})
        return value

    tokenization = token_result.to_dict()
    bio_tags = tokenization.pop("bio_tags", None)
    labels = tokenization.pop("labels", None)
    if not isinstance(bio_tags, list) or not isinstance(labels, list):
        raise RuntimeError("token result is missing BIO supervision")
    tokenization.update({"tokenizer": tokenizer_name})
    value.update(
        {
            "tokenization": tokenization,
            "supervision": {
                "label2id": dict(DEFAULT_LABEL_TO_ID),
                "bio_tags": bio_tags,
                "labels": labels,
                "ignore_index": IGNORE_INDEX,
            },
        }
    )
    return value


def _resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    accepted = Path(args.output)
    return OutputPaths(
        accepted=accepted,
        sft=(
            Path(args.sft_output) if args.sft_output else _related_path(accepted, "sft")
        ),
        rejected=(
            Path(args.rejected_output)
            if args.rejected_output
            else _related_path(accepted, "rejected")
        ),
        manifest=(
            Path(args.manifest_output)
            if args.manifest_output
            else _related_path(accepted, "meta", suffix=".json")
        ),
    )


def _write_manifest(
    input_path: Path,
    output_paths: OutputPaths,
    args: argparse.Namespace,
    counts: Mapping[str, int],
) -> None:
    manifest = {
        "format": "select-sequence-bio-v1",
        "input": str(input_path),
        "outputs": {
            "accepted_audit": str(output_paths.accepted),
            "sft": str(output_paths.sft),
            "rejected": str(output_paths.rejected),
        },
        "tokenizer": args.tokenizer,
        "label2id": DEFAULT_LABEL_TO_ID,
        "id2label": {str(value): key for key, value in DEFAULT_LABEL_TO_ID.items()},
        "ignore_index": IGNORE_INDEX,
        "alignment": {
            "algorithm": ALIGNMENT_ALGORITHM,
            "min_match_chars": args.min_match_chars,
            "max_adjust_chars": args.max_adjust_chars,
            "index_convention": INDEX_CONVENTION,
        },
        "max_length": args.max_length,
        "counts": dict(counts),
    }
    output_paths.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _log_summary(output_paths: OutputPaths, counts: Mapping[str, int]) -> None:
    LOGGER.info(
        "processed=%d accepted=%d aligned=%d adjusted=%d unaligned=%d "
        "truncated=%d errors=%d",
        counts.get("processed", 0),
        counts.get("accepted", 0),
        counts.get("aligned", 0),
        counts.get("adjusted", 0),
        counts.get("unaligned", 0),
        counts.get("truncated", 0),
        counts.get("errors", 0),
    )
    LOGGER.info(
        "accepted=%s sft=%s rejected=%s manifest=%s",
        output_paths.accepted,
        output_paths.sft,
        output_paths.rejected,
        output_paths.manifest,
    )


def _load_tokenizer(name_or_path: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install with: "
            "pip install 'transformers>=4.51,<5'"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for return_offsets_mapping")
    return tokenizer


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if value is None:
        raise KeyError(f"missing required field {field!r}")
    if not isinstance(value, str):
        raise TypeError(f"field {field!r} must be a string")
    return value


def _related_path(path: Path, label: str, suffix: Optional[str] = None) -> Path:
    output_suffix = suffix if suffix is not None else (path.suffix or ".jsonl")
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.{label}{output_suffix}")


def _validate_distinct_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("accepted, SFT, rejected, and manifest paths must be distinct")


def _write_jsonl(stream: IO[str], value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _increment(counts: Dict[str, int], key: str) -> None:
    counts.update({key: counts.get(key, 0) + 1})


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    raise SystemExit(main())
