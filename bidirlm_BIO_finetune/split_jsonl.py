from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)
MINIMUM_SPLIT_RECORDS = 3
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    _validate_ratios(args.train_ratio, args.validation_ratio)
    lines = _load_validated_lines(Path(args.input))
    random.Random(args.seed).shuffle(lines)
    splits = _split_lines(lines, args.train_ratio, args.validation_ratio)
    _write_splits(Path(args.output_dir), splits)
    LOGGER.info(
        "split sizes: %s",
        json.dumps(
            {name: len(values) for name, values in splits.items()},
            ensure_ascii=False,
        ),
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministically split BIO JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument(
        "--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def _validate_ratios(train_ratio: float, validation_ratio: float) -> None:
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train and validation ratios must be positive")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")


def _load_validated_lines(path: Path) -> List[str]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(lines) < MINIMUM_SPLIT_RECORDS:
        raise ValueError("at least three records are required")
    for line_number, line in enumerate(lines, start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"line {line_number}: JSON value must be an object")
    return lines


def _split_lines(
    lines: Sequence[str], train_ratio: float, validation_ratio: float
) -> Dict[str, List[str]]:
    train_end = int(len(lines) * train_ratio)
    validation_end = train_end + int(len(lines) * validation_ratio)
    return {
        "train.jsonl": lines[:train_end],
        "validation.jsonl": lines[train_end:validation_end],
        "test.jsonl": lines[validation_end:],
    }


def _write_splits(output_dir: Path, splits: Dict[str, List[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (output_dir / name).write_text("\n".join(values) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
