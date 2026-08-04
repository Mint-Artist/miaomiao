#!/usr/bin/env python3
"""Validate a JSONL/JSONL.GZ input file before running ProX inference."""

from __future__ import annotations

import argparse
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield handle


def get_by_path(record: dict, dotted_path: str):
    value = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 ProX 第一阶段 JSONL 输入。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--text-key",
        default="content",
        help="文本字段；嵌套字段可写成 data.cleaned_text。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = {
        "total": 0,
        "valid_text": 0,
        "empty_text": 0,
        "missing_text_key": 0,
        "non_string_text": 0,
        "with_newline": 0,
        "invalid_json": 0,
        "max_chars": 0,
    }

    with open_text(args.input) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats["total"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_json"] += 1
                print(f"第 {line_number} 行不是合法 JSON：{exc}")
                continue

            if not isinstance(record, dict):
                stats["invalid_json"] += 1
                print(f"第 {line_number} 行的 JSON 顶层不是对象。")
                continue

            try:
                text = get_by_path(record, args.text_key)
            except KeyError:
                stats["missing_text_key"] += 1
                continue

            if not isinstance(text, str):
                stats["non_string_text"] += 1
                continue

            if not text.strip():
                stats["empty_text"] += 1
                continue

            stats["valid_text"] += 1
            stats["max_chars"] = max(stats["max_chars"], len(text))
            if "\n" in text:
                stats["with_newline"] += 1

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    invalid = (
        stats["invalid_json"]
        + stats["missing_text_key"]
        + stats["non_string_text"]
    )
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
