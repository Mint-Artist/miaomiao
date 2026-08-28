"""Shared constants for SELECT-style BIO fine-tuning and inference."""

from __future__ import annotations

IGNORE_INDEX = -100

OUTSIDE_LABEL_ID = 0
BEGIN_LABEL_ID = 1
INSIDE_LABEL_ID = 2
NUM_BIO_LABELS = 3
NUM_BIO_TRANSITIONS = NUM_BIO_LABELS * NUM_BIO_LABELS

SUPPORTED_LABEL_IDS = frozenset(
    {IGNORE_INDEX, OUTSIDE_LABEL_ID, BEGIN_LABEL_ID, INSIDE_LABEL_ID}
)
LABEL_TO_TAG = {
    IGNORE_INDEX: "IGN",
    OUTSIDE_LABEL_ID: "O",
    BEGIN_LABEL_ID: "B",
    INSIDE_LABEL_ID: "I",
}
TAG_TO_LABEL = {tag: label for label, tag in LABEL_TO_TAG.items() if label >= 0}

INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
LOSS_KEY = "loss"
CLASSIFICATION_LOSS_KEY = "classification_loss"
TRANSITION_LOSS_KEY = "transition_loss"
CLASSIFICATION_LOGITS_KEY = "classification_logits"
TRANSITION_LOGITS_KEY = "transition_logits"

DEFAULT_MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 1
DEFAULT_ATTENTION_IMPLEMENTATION = "eager"
