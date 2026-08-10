from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .alignment import extract_refined_text
from .labeling import DEFAULT_LABEL_TO_ID, label_aligned_pair


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
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or path")
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
        default=20,
        help="Minimum exact character match length (paper default: 20)",
    )
    parser.add_argument(
        "--max-adjust-chars",
        type=int,
        default=5,
        help="Maximum source/target internal-gap length difference (paper default: 5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
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
    args = build_parser().parse_args(argv)
    if args.max_length < 1:
        raise ValueError("max_length must be at least 1")

    tokenizer = _load_tokenizer(args.tokenizer, args.trust_remote_code)
    input_path = Path(args.input)
    accepted_path = Path(args.output)
    sft_path = Path(args.sft_output) if args.sft_output else _related_path(accepted_path, "sft")
    rejected_path = (
        Path(args.rejected_output)
        if args.rejected_output
        else _related_path(accepted_path, "rejected")
    )
    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output
        else _related_path(accepted_path, "meta", suffix=".json")
    )
    _validate_distinct_paths(accepted_path, sft_path, rejected_path, manifest_path)
    for path in (accepted_path, sft_path, rejected_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {
        "processed": 0,
        "accepted": 0,
        "aligned": 0,
        "adjusted": 0,
        "unaligned": 0,
        "truncated": 0,
        "errors": 0,
    }

    with ExitStack() as stack:
        input_stream = stack.enter_context(input_path.open("r", encoding="utf-8"))
        accepted_stream = stack.enter_context(accepted_path.open("w", encoding="utf-8"))
        sft_stream = stack.enter_context(sft_path.open("w", encoding="utf-8"))
        rejected_stream = stack.enter_context(rejected_path.open("w", encoding="utf-8"))

        for line_number, line in enumerate(input_stream, start=1):
            if not line.strip():
                continue
            counts["processed"] += 1
            record: Any = None
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise TypeError("each JSONL row must be an object")
                source_text = _required_string(record, args.source_field)
                refined_value = _required_string(record, args.refined_field)
                teacher_refined_text = (
                    extract_refined_text(refined_value)
                    if args.parse_teacher_response
                    else refined_value
                )
                alignment, token_result = label_aligned_pair(
                    source_text,
                    teacher_refined_text,
                    tokenizer,
                    min_match_chars=args.min_match_chars,
                    max_adjust_chars=args.max_adjust_chars,
                    max_length=args.max_length,
                    add_special_tokens=not args.no_special_tokens,
                )

                audit_record = _build_audit_record(
                    record,
                    source_text=source_text,
                    teacher_refined_text=teacher_refined_text,
                    tokenizer_name=args.tokenizer,
                    alignment=alignment,
                    token_result=token_result,
                    include_match_text=not args.omit_match_text,
                )

                counts[alignment.status] += 1
                if not alignment.is_accepted:
                    audit_record["dataset_status"] = "rejected"
                    audit_record["rejection_reason"] = "unaligned"
                    _write_jsonl(rejected_stream, audit_record)
                    continue

                assert token_result is not None
                if token_result.truncated:
                    counts["truncated"] += 1
                    audit_record["dataset_status"] = "rejected"
                    audit_record["rejection_reason"] = "tokenization_truncated"
                    _write_jsonl(rejected_stream, audit_record)
                    continue

                counts["accepted"] += 1
                audit_record["dataset_status"] = "accepted"
                audit_record["rejection_reason"] = None
                _write_jsonl(accepted_stream, audit_record)
                sample_id = record.get(args.id_field)
                _write_jsonl(sft_stream, token_result.to_sft_dict(sample_id))
            except Exception as exc:  # Preserve progress across malformed rows.
                counts["errors"] += 1
                error_record = dict(record) if isinstance(record, Mapping) else {}
                error_record.update(
                    {
                        "line_number": line_number,
                        "dataset_status": "rejected",
                        "rejection_reason": "processing_error",
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    }
                )
                _write_jsonl(rejected_stream, error_record)

    manifest = {
        "format": "select-sequence-bio-v1",
        "input": str(input_path),
        "outputs": {
            "accepted_audit": str(accepted_path),
            "sft": str(sft_path),
            "rejected": str(rejected_path),
        },
        "tokenizer": args.tokenizer,
        "label2id": DEFAULT_LABEL_TO_ID,
        "id2label": {str(value): key for key, value in DEFAULT_LABEL_TO_ID.items()},
        "ignore_index": -100,
        "alignment": {
            "algorithm": "select-longest-match-v1",
            "min_match_chars": args.min_match_chars,
            "max_adjust_chars": args.max_adjust_chars,
            "index_convention": "zero_based_half_open",
        },
        "max_length": args.max_length,
        "counts": counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        (
            "processed={processed} accepted={accepted} aligned={aligned} "
            "adjusted={adjusted} unaligned={unaligned} truncated={truncated} "
            "errors={errors}"
        ).format(**counts),
        file=sys.stderr,
    )
    print(
        "accepted=%s sft=%s rejected=%s manifest=%s"
        % (accepted_path, sft_path, rejected_path, manifest_path),
        file=sys.stderr,
    )
    return 0 if counts["errors"] == 0 else 2


def _build_audit_record(
    record: Mapping[str, Any],
    *,
    source_text: str,
    teacher_refined_text: str,
    tokenizer_name: str,
    alignment: Any,
    token_result: Any,
    include_match_text: bool,
) -> Dict[str, Any]:
    value = dict(record)
    value["source_text"] = source_text
    value["teacher_refined_text"] = teacher_refined_text
    value["adjusted_refined_text"] = alignment.adjusted_refined_text
    value["alignment"] = alignment.to_dict(
        include_text=False,
        include_match_text=include_match_text,
    )
    if token_result is None:
        value["tokenization"] = None
        value["supervision"] = None
        return value

    tokenization = token_result.to_dict()
    bio_tags = tokenization.pop("bio_tags")
    labels = tokenization.pop("labels")
    tokenization["tokenizer"] = tokenizer_name
    value["tokenization"] = tokenization
    value["supervision"] = {
        "label2id": dict(DEFAULT_LABEL_TO_ID),
        "bio_tags": bio_tags,
        "labels": labels,
        "ignore_index": -100,
    }
    return value


def _load_tokenizer(name_or_path: str, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install with: pip install 'transformers>=4.51,<5'"
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
    if field not in record:
        raise KeyError("missing required field %r" % field)
    value = record[field]
    if not isinstance(value, str):
        raise TypeError("field %r must be a string" % field)
    return value


def _related_path(path: Path, label: str, suffix: Optional[str] = None) -> Path:
    output_suffix = suffix if suffix is not None else (path.suffix or ".jsonl")
    stem = path.stem if path.suffix else path.name
    return path.with_name("%s.%s%s" % (stem, label, output_suffix))


def _validate_distinct_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("accepted, SFT, rejected, and manifest paths must be distinct")


def _write_jsonl(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
