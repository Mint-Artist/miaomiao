from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from bidirlm_BIO_finetune.constants import (
    ATTENTION_MASK_KEY,
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    LABELS_KEY,
    SUPPORTED_LABEL_IDS,
)

REQUIRED_SEQUENCE_FIELDS = (INPUT_IDS_KEY, ATTENTION_MASK_KEY, LABELS_KEY)
DEFAULT_PADDING_MULTIPLE = 8


class BioJsonlDataset(Dataset):
    """Load validated, pre-tokenized BIO examples from JSONL."""

    def __init__(
        self, path: str | Path, *, max_length: Optional[int] = DEFAULT_MAX_LENGTH
    ):
        self.path = Path(path)
        self.records = _load_records(self.path, max_length=max_length)
        if not self.records:
            raise ValueError(f"no training records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def _load_records(path: Path, *, max_length: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            records.append(
                validate_record(
                    json.loads(line),
                    line_number=line_number,
                    max_length=max_length,
                )
            )
    return records


def validate_record(
    value: Any,
    *,
    line_number: int,
    max_length: Optional[int],
) -> Dict[str, Any]:
    """Validate one dataset row and normalize sequence values to integers."""

    if not isinstance(value, Mapping):
        raise TypeError(f"line {line_number}: each JSONL row must be an object")

    result = dict(value)
    sequences = {
        field: _require_list(result, field, line_number)
        for field in REQUIRED_SEQUENCE_FIELDS
    }
    sequence_lengths = {len(sequence) for sequence in sequences.values()}
    if len(sequence_lengths) != 1:
        raise ValueError(
            f"line {line_number}: input_ids, attention_mask and labels lengths differ"
        )

    sequence_length = len(sequences.get(INPUT_IDS_KEY, []))
    _validate_sequence_length(sequence_length, max_length, line_number)
    normalized_labels = [int(item) for item in sequences.get(LABELS_KEY, [])]
    _validate_labels(normalized_labels, line_number)

    result.update(
        {
            INPUT_IDS_KEY: [int(item) for item in sequences.get(INPUT_IDS_KEY, [])],
            ATTENTION_MASK_KEY: [
                int(item) for item in sequences.get(ATTENTION_MASK_KEY, [])
            ],
            LABELS_KEY: normalized_labels,
        }
    )
    return result


def _require_list(record: Mapping[str, Any], field: str, line_number: int) -> List[Any]:
    value = record.get(field)
    if not isinstance(value, list):
        raise TypeError(f"line {line_number}: {field!r} must be a list")
    return value


def _validate_sequence_length(
    sequence_length: int,
    max_length: Optional[int],
    line_number: int,
) -> None:
    if sequence_length == 0:
        raise ValueError(f"line {line_number}: empty token sequence")
    if max_length is not None and sequence_length > max_length:
        raise ValueError(
            f"line {line_number}: sequence length {sequence_length} exceeds "
            f"max_length={max_length}; rebuild or filter the BIO data instead "
            "of truncating"
        )


def _validate_labels(labels: Sequence[int], line_number: int) -> None:
    invalid_labels = sorted(set(labels) - SUPPORTED_LABEL_IDS)
    if invalid_labels:
        raise ValueError(f"line {line_number}: unsupported labels {invalid_labels}")
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"line {line_number}: all labels are ignored")


class BioDataCollator:
    """Dynamically pad pre-tokenized BIO examples to the batch maximum."""

    def __init__(
        self,
        pad_token_id: int,
        *,
        pad_to_multiple_of: Optional[int] = DEFAULT_PADDING_MULTIPLE,
    ):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")

        sequences = [_extract_feature_sequences(feature) for feature in features]
        max_length = _padded_batch_length(
            sequences,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = [
            _pad(sequence.get(INPUT_IDS_KEY, []), max_length, self.pad_token_id)
            for sequence in sequences
        ]
        attention_masks = [
            _pad(sequence.get(ATTENTION_MASK_KEY, []), max_length, 0)
            for sequence in sequences
        ]
        labels = [
            _pad(sequence.get(LABELS_KEY, []), max_length, IGNORE_INDEX)
            for sequence in sequences
        ]
        return {
            INPUT_IDS_KEY: torch.tensor(input_ids, dtype=torch.long),
            ATTENTION_MASK_KEY: torch.tensor(attention_masks, dtype=torch.long),
            LABELS_KEY: torch.tensor(labels, dtype=torch.long),
            "ids": [feature.get("id") for feature in features],
        }


def _extract_feature_sequences(feature: Mapping[str, Any]) -> Dict[str, List[int]]:
    return {
        field: [int(item) for item in _require_feature_list(feature, field)]
        for field in REQUIRED_SEQUENCE_FIELDS
    }


def _require_feature_list(feature: Mapping[str, Any], field: str) -> List[Any]:
    value = feature.get(field)
    if not isinstance(value, list):
        raise TypeError(f"collator field {field!r} must be a list")
    return value


def _padded_batch_length(
    sequences: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_to_multiple_of: Optional[int],
) -> int:
    max_length = max(len(sequence.get(INPUT_IDS_KEY, [])) for sequence in sequences)
    if not pad_to_multiple_of:
        return max_length
    return (
        (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of
    ) * pad_to_multiple_of


def _pad(values: Sequence[int], target_length: int, pad_value: int) -> List[int]:
    return list(values) + [pad_value] * (target_length - len(values))
