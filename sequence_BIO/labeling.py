from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .alignment import AlignmentResult, CharSpan, align_deletion_only


# Numeric IDs are arbitrary as long as the model config and dataset agree.
DEFAULT_LABEL_TO_ID: Dict[str, int] = {"O": 0, "B": 1, "I": 2}


class OffsetTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class TokenLabelResult:
    input_ids: Tuple[int, ...]
    attention_mask: Tuple[int, ...]
    token_texts: Tuple[str, ...]
    offset_mapping: Tuple[Tuple[int, int], ...]
    bio_tags: Tuple[str, ...]
    labels: Tuple[int, ...]
    token_spans: Tuple[Tuple[int, int], ...]
    boundary_adjustments: Tuple[Dict[str, int], ...]
    truncated: bool
    tokenized_char_end: int

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    # Compatibility aliases for callers of the prototype API.
    @property
    def tokens(self) -> Tuple[str, ...]:
        return self.token_texts

    @property
    def label_ids(self) -> Tuple[int, ...]:
        return self.labels

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["input_ids"] = list(self.input_ids)
        value["attention_mask"] = list(self.attention_mask)
        value["token_texts"] = list(self.token_texts)
        value["offset_mapping"] = [list(offset) for offset in self.offset_mapping]
        value["bio_tags"] = list(self.bio_tags)
        value["labels"] = list(self.labels)
        value["token_spans"] = [list(span) for span in self.token_spans]
        value["boundary_adjustments"] = [dict(item) for item in self.boundary_adjustments]
        value["sequence_length"] = self.sequence_length
        return value

    def to_sft_dict(self, sample_id: Optional[object] = None) -> Dict[str, object]:
        value: Dict[str, object] = {}
        if sample_id is not None:
            value["id"] = sample_id
        value["input_ids"] = list(self.input_ids)
        value["attention_mask"] = list(self.attention_mask)
        value["labels"] = list(self.labels)
        return value


def spans_to_bio_labels(
    offset_mapping: Sequence[Sequence[int]],
    character_spans: Sequence[CharSpan],
    *,
    label_to_id: Mapping[str, int] = DEFAULT_LABEL_TO_ID,
    ignore_index: int = -100,
) -> Tuple[List[str], List[int], List[Tuple[int, int]], List[Dict[str, int]]]:
    """Convert retained source-character spans to token-level BIO labels."""

    offsets = [(int(item[0]), int(item[1])) for item in offset_mapping]
    bio_tags = ["O" if start != end else "IGN" for start, end in offsets]
    token_ranges: List[Tuple[int, int]] = []
    adjustments: List[Dict[str, int]] = []

    for char_start, char_end in character_spans:
        if char_start < 0 or char_end <= char_start:
            raise ValueError("invalid character span: %r" % ((char_start, char_end),))
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start != token_end
            and token_end > char_start
            and token_start < char_end
        ]
        if not overlapping:
            continue

        token_start_index = overlapping[0]
        token_end_index = overlapping[-1] + 1
        token_ranges.append((token_start_index, token_end_index))

        expanded_start = offsets[token_start_index][0]
        expanded_end = offsets[token_end_index - 1][1]
        if expanded_start != char_start or expanded_end != char_end:
            adjustments.append(
                {
                    "char_start": char_start,
                    "char_end": char_end,
                    "expanded_start": expanded_start,
                    "expanded_end": expanded_end,
                    "left_expansion": max(0, char_start - expanded_start),
                    "right_expansion": max(0, expanded_end - char_end),
                }
            )

    merged_ranges = _merge_overlapping_token_ranges(token_ranges)
    for token_start, token_end in merged_ranges:
        bio_tags[token_start] = "B"
        for index in range(token_start + 1, token_end):
            bio_tags[index] = "I"

    labels = [
        ignore_index if tag == "IGN" else int(label_to_id[tag])
        for tag in bio_tags
    ]
    return bio_tags, labels, merged_ranges, adjustments


def label_aligned_pair(
    source_text: str,
    refined_text: str,
    tokenizer: OffsetTokenizer,
    *,
    min_match_chars: int = 20,
    max_adjust_chars: int = 5,
    max_length: Optional[int] = 32768,
    add_special_tokens: bool = True,
    label_to_id: Mapping[str, int] = DEFAULT_LABEL_TO_ID,
    ignore_index: int = -100,
) -> Tuple[AlignmentResult, Optional[TokenLabelResult]]:
    """Align, verify, tokenize, and BIO-label one source/teacher pair."""

    alignment = align_deletion_only(
        source_text,
        refined_text,
        min_match_chars=min_match_chars,
        max_adjust_chars=max_adjust_chars,
    )
    if not alignment.is_accepted:
        return alignment, None

    tokenizer_kwargs: Dict[str, object] = {
        "return_offsets_mapping": True,
        "return_attention_mask": True,
        "add_special_tokens": add_special_tokens,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length
    encoded = tokenizer(source_text, **tokenizer_kwargs)

    input_ids = _flatten_input_ids(encoded["input_ids"])
    offsets = _normalize_offsets(encoded["offset_mapping"])
    raw_attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
    attention_mask = _flatten_input_ids(raw_attention_mask)
    if not (len(input_ids) == len(offsets) == len(attention_mask)):
        raise ValueError(
            "tokenizer returned different input_ids, attention_mask, and "
            "offset_mapping lengths"
        )

    bio_tags, labels, token_spans, adjustments = spans_to_bio_labels(
        offsets,
        alignment.character_spans,
        label_to_id=label_to_id,
        ignore_index=ignore_index,
    )
    token_texts = tuple(
        "<SPECIAL>" if start == end else source_text[start:end]
        for start, end in offsets
    )

    tokenized_char_end = max((end for start, end in offsets if start != end), default=0)
    truncated = tokenized_char_end < len(source_text)
    return alignment, TokenLabelResult(
        input_ids=tuple(int(item) for item in input_ids),
        attention_mask=tuple(int(item) for item in attention_mask),
        token_texts=token_texts,
        offset_mapping=tuple(offsets),
        bio_tags=tuple(bio_tags),
        labels=tuple(labels),
        token_spans=tuple(token_spans),
        boundary_adjustments=tuple(adjustments),
        truncated=truncated,
        tokenized_char_end=tokenized_char_end,
    )


def derive_transition_supervision(
    labels: Sequence[int],
    *,
    ignore_index: int = -100,
) -> Tuple[List[int], List[int], List[bool]]:
    """Derive the paper's conditional ``u -> v`` targets from BIO labels.

    This intentionally does not flatten transitions into a 9-class label.  The
    model should reshape transition logits to ``[..., 3, 3]``, select the row
    indexed by ``from_labels``, and apply cross-entropy over ``to_labels``.
    """

    from_labels = [int(item) for item in labels[:-1]]
    to_labels = [int(item) for item in labels[1:]]
    mask = [
        left != ignore_index and right != ignore_index
        for left, right in zip(from_labels, to_labels)
    ]
    return from_labels, to_labels, mask


def _merge_overlapping_token_ranges(
    ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: List[Tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start < previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _flatten_input_ids(value: Any) -> List[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError("tokenizer sequence field must be a list for one input")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("tokenizer returned a batch for one input")
        return list(value[0])
    return list(value)


def _normalize_offsets(value: Any) -> List[Tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError("tokenizer offset_mapping must be a list for one input")
    if (
        len(value) == 1
        and isinstance(value[0], list)
        and (not value[0] or isinstance(value[0][0], (list, tuple)))
    ):
        value = value[0]

    offsets: List[Tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each tokenizer offset must contain exactly two integers")
        offsets.append((int(item[0]), int(item[1])))
    return offsets
