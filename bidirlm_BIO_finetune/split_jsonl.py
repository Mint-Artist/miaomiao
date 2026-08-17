from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministically split BIO JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_ratio <= 0 or args.validation_ratio <= 0:
        raise ValueError("train and validation ratios must be positive")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    lines = [
        line
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 3:
        raise ValueError("at least three records are required")
    for line_number, line in enumerate(lines, start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"line {line_number}: JSON value must be an object")
    random.Random(args.seed).shuffle(lines)
    train_end = int(len(lines) * args.train_ratio)
    validation_end = train_end + int(len(lines) * args.validation_ratio)
    splits = {
        "train.jsonl": lines[:train_end],
        "validation.jsonl": lines[train_end:validation_end],
        "test.jsonl": lines[validation_end:],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (output_dir / name).write_text("\n".join(values) + "\n", encoding="utf-8")
    print(json.dumps({name: len(values) for name, values in splits.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
