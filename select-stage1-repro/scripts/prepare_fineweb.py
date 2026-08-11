#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def resolve_tokenizer_reference(value: str) -> str | Path:
    """Return a canonical Path for local tokenizers, or preserve a Hub model ID."""
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.exists() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream FineWeb and materialize a deterministic packed uint32 token file."
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "artifacts/models/Qwen3-0.6B-Base"),
        help="Local Qwen3 tokenizer directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts/data/fineweb_select_100m"),
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--source-jsonl",
        help="Optional local JSONL override for offline tests. Rows must contain a text field.",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--train-tokens", type=int, default=99_999_744)
    parser.add_argument("--validation-tokens", type=int, default=1_048_576)
    parser.add_argument("--batch-documents", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard an incomplete token stream and start again.",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_args(args: argparse.Namespace) -> None:
    if args.sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    for name in ("train_tokens", "validation_tokens"):
        value = getattr(args, name)
        if value <= 0 or value % args.sequence_length:
            raise ValueError(
                f"--{name.replace('_', '-')} must be a positive multiple of "
                f"--sequence-length ({args.sequence_length})"
            )


def finalize_dataset(
    *,
    args: argparse.Namespace,
    expected: dict[str, Any],
    token_path: Path,
    metadata_path: Path,
    progress_path: Path,
    documents_seen: int,
    source_revision: Any = None,
) -> None:
    digest = sha256(token_path)
    metadata = {
        **expected,
        "source_revision": str(source_revision) if source_revision is not None else None,
        "documents_consumed": documents_seen,
        "token_file": token_path.name,
        "token_dtype": "uint32",
        "token_file_sha256": digest,
        "train_offset_tokens": 0,
        "validation_offset_tokens": args.train_tokens,
        "train_sequences": args.train_tokens // args.sequence_length,
        "validation_sequences": args.validation_tokens // args.sequence_length,
        "packing": "contiguous documents separated by eos_token_id",
        "cross_document_attention": True,
        "packing_note": (
            "SELECT does not disclose document-boundary masking. This reproduction uses "
            "the common memory-efficient EOS-separated contiguous stream assumption."
        ),
    }
    atomic_json(metadata_path, metadata)
    atomic_json(
        progress_path,
        {**metadata, "tokens_written": expected["total_tokens"], "complete": True},
    )
    print(f"Wrote {expected['total_tokens']:,} tokens to {token_path}")
    print(f"SHA256: {digest}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "tokens.bin"
    progress_path = output_dir / "progress.json"
    metadata_path = output_dir / "metadata.json"

    if metadata_path.exists() and not args.restart:
        print(f"Dataset is already complete: {metadata_path}")
        return
    if args.restart:
        for path in (token_path, progress_path, metadata_path):
            path.unlink(missing_ok=True)

    tokenizer_reference = resolve_tokenizer_reference(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_reference, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")

    total_tokens = args.train_tokens + args.validation_tokens
    expected = {
        "format_version": 1,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "source_jsonl": str(Path(args.source_jsonl).resolve()) if args.source_jsonl else None,
        "text_column": args.text_column,
        "sequence_length": args.sequence_length,
        "train_tokens": args.train_tokens,
        "validation_tokens": args.validation_tokens,
        "total_tokens": total_tokens,
        "tokenizer": str(tokenizer_reference),
        "tokenizer_vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
    }

    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as stream:
            progress = json.load(stream)
        for key, value in expected.items():
            if progress.get(key) != value:
                raise ValueError(
                    f"resume metadata mismatch for {key}: {progress.get(key)!r} != {value!r}"
                )
        written = int(progress["tokens_written"])
        documents_seen = int(progress["documents_seen"])
        if token_path.stat().st_size != total_tokens * np.dtype(np.uint32).itemsize:
            raise ValueError("existing token file has an unexpected size")
        mode = "r+"
        print(f"Resuming after {documents_seen:,} documents and {written:,} tokens")
    else:
        written = 0
        documents_seen = 0
        with token_path.open("wb") as stream:
            stream.truncate(total_tokens * np.dtype(np.uint32).itemsize)
        progress = {
            **expected,
            "tokens_written": 0,
            "documents_seen": 0,
            "complete": False,
        }
        atomic_json(progress_path, progress)
        mode = "r+"

    # A previous process can be interrupted after its final flush but before hashing.
    if written == total_tokens:
        finalize_dataset(
            args=args,
            expected=expected,
            token_path=token_path,
            metadata_path=metadata_path,
            progress_path=progress_path,
            documents_seen=documents_seen,
        )
        return

    if args.source_jsonl:
        dataset = load_dataset(
            "json", data_files=args.source_jsonl, split="train", streaming=True
        )
        source_revision = None
    else:
        dataset = load_dataset(
            args.dataset,
            name=args.dataset_config,
            split=args.split,
            streaming=True,
        )
        source_revision = getattr(getattr(dataset, "info", None), "version", None)
    if documents_seen:
        dataset = dataset.skip(documents_seen)

    tokens = np.memmap(token_path, dtype=np.uint32, mode=mode, shape=(total_tokens,))
    started = time.perf_counter()
    for batch_index, batch in enumerate(dataset.iter(batch_size=args.batch_documents), start=1):
        raw_texts = batch[args.text_column]
        texts = [text if isinstance(text, str) else "" for text in raw_texts]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        for token_ids in encoded:
            documents_seen += 1
            if not token_ids:
                continue
            document = np.asarray([*token_ids, tokenizer.eos_token_id], dtype=np.uint32)
            take = min(document.size, total_tokens - written)
            tokens[written : written + take] = document[:take]
            written += take
            if written == total_tokens:
                break

        if batch_index % args.progress_every == 0 or written == total_tokens:
            tokens.flush()
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress = {
                **expected,
                "tokens_written": written,
                "documents_seen": documents_seen,
                "complete": False,
            }
            atomic_json(progress_path, progress)
            print(
                f"tokens={written:,}/{total_tokens:,} "
                f"documents={documents_seen:,} rate={written / elapsed:,.0f} token/s",
                flush=True,
            )
        if written == total_tokens:
            break
    else:
        raise RuntimeError(
            f"source exhausted at {written:,} tokens; requested {total_tokens:,}"
        )

    tokens.flush()
    del tokens
    finalize_dataset(
        args=args,
        expected=expected,
        token_path=token_path,
        metadata_path=metadata_path,
        progress_path=progress_path,
        documents_seen=documents_seen,
        source_revision=source_revision,
    )


if __name__ == "__main__":
    main()
