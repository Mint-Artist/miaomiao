from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset


class BioJsonlDataset(Dataset):
    """Load the minimal JSONL emitted by ``sequence_BIO``.

    Each non-empty row must contain equally sized ``input_ids``,
    ``attention_mask`` and ``labels`` lists. Long samples are rejected instead
    of silently truncated because truncation can split a retained BIO span.
    """

    def __init__(self, path: str | Path, *, max_length: Optional[int] = 8192):
        self.path = Path(path)
        self.records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                self.records.append(
                    validate_record(value, line_number=line_number, max_length=max_length)
                )
        if not self.records:
            raise ValueError(f"no training records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def validate_record(
    value: Any,
    *,
    line_number: int,
    max_length: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"line {line_number}: each JSONL row must be an object")
    result = dict(value)
    for field in ("input_ids", "attention_mask", "labels"):
        if field not in result or not isinstance(result[field], list):
            raise TypeError(f"line {line_number}: {field!r} must be a list")
    lengths = {len(result[field]) for field in ("input_ids", "attention_mask", "labels")}
    if len(lengths) != 1:
        raise ValueError(
            f"line {line_number}: input_ids, attention_mask and labels lengths differ"
        )
    sequence_length = len(result["input_ids"])
    if sequence_length == 0:
        raise ValueError(f"line {line_number}: empty token sequence")
    if max_length is not None and sequence_length > max_length:
        raise ValueError(
            f"line {line_number}: sequence length {sequence_length} exceeds "
            f"max_length={max_length}; rebuild or filter the BIO data instead of truncating"
        )
    invalid_labels = sorted({int(item) for item in result["labels"]} - {-100, 0, 1, 2})
    if invalid_labels:
        raise ValueError(f"line {line_number}: unsupported labels {invalid_labels}")
    if not any(int(item) != -100 for item in result["labels"]):
        raise ValueError(f"line {line_number}: all labels are ignored")
    result["input_ids"] = [int(item) for item in result["input_ids"]]
    result["attention_mask"] = [int(item) for item in result["attention_mask"]]
    result["labels"] = [int(item) for item in result["labels"]]
    return result


class BioDataCollator:
    """Dynamically pad pre-tokenized BIO examples to the batch maximum."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: Optional[int] = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        max_length = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        ids: List[List[int]] = []
        masks: List[List[int]] = []
        labels: List[List[int]] = []
        sample_ids: List[Any] = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            ids.append(list(item["input_ids"]) + [self.pad_token_id] * padding)
            masks.append(list(item["attention_mask"]) + [0] * padding)
            labels.append(list(item["labels"]) + [-100] * padding)
            sample_ids.append(item.get("id"))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ids": sample_ids,
        }
