"""Shared constants for SELECT-style alignment and BIO supervision."""

from __future__ import annotations

IGNORE_INDEX = -100

OUTSIDE_TAG = "O"
BEGIN_TAG = "B"
INSIDE_TAG = "I"
IGNORED_TAG = "IGN"

DEFAULT_LABEL_TO_ID = {OUTSIDE_TAG: 0, BEGIN_TAG: 1, INSIDE_TAG: 2}
DEFAULT_MIN_MATCH_CHARS = 20
DEFAULT_MAX_ADJUST_CHARS = 5
DEFAULT_MAX_LENGTH = 32768

ALIGNMENT_ALGORITHM = "select-longest-match-v1"
INDEX_CONVENTION = "zero_based_half_open"
