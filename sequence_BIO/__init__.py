"""SELECT-style deletion-only alignment and BIO labeling utilities."""

from .alignment import (
    AlignmentResult,
    GapAdjustment,
    MatchSpan,
    align_deletion_only,
    extract_refined_text,
    find_longest_match_segments,
)
from .labeling import (
    DEFAULT_LABEL_TO_ID,
    TokenLabelResult,
    derive_transition_supervision,
    label_aligned_pair,
    spans_to_bio_labels,
)

__all__ = [
    "AlignmentResult",
    "DEFAULT_LABEL_TO_ID",
    "GapAdjustment",
    "MatchSpan",
    "TokenLabelResult",
    "align_deletion_only",
    "derive_transition_supervision",
    "extract_refined_text",
    "find_longest_match_segments",
    "label_aligned_pair",
    "spans_to_bio_labels",
]
