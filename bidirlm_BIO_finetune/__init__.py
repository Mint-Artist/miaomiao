"""SELECT-style BIO fine-tuning utilities for BidirLM."""

from .data import BioDataCollator, BioJsonlDataset
from .decoding import (
    compute_bio_metrics,
    compute_bio_metrics_from_sequences,
    pad_and_cat,
    viterbi_decode_batch,
)

__all__ = [
    "BioDataCollator",
    "BioJsonlDataset",
    "compute_bio_metrics",
    "compute_bio_metrics_from_sequences",
    "pad_and_cat",
    "viterbi_decode_batch",
]
