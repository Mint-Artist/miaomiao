from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .constants import (
    BEGIN_LABEL_ID,
    IGNORE_INDEX,
    INSIDE_LABEL_ID,
    NUM_BIO_LABELS,
    OUTSIDE_LABEL_ID,
)


def viterbi_decode_batch(
    classification_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Decode labels using the SELECT classification and transition scores."""

    _validate_decode_shapes(classification_logits, transition_logits, valid_mask)
    classification_log_probs = classification_logits.float().log_softmax(dim=-1)
    transition_log_probs = transition_logits.float().log_softmax(dim=-1)
    decoded = torch.full(
        valid_mask.shape,
        IGNORE_INDEX,
        dtype=torch.long,
        device=classification_logits.device,
    )
    for batch_index in range(classification_logits.shape[0]):
        valid_indices = (
            torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten().tolist()
        )
        for run in _contiguous_runs(valid_indices):
            decoded[batch_index, run] = torch.tensor(
                _decode_run(
                    classification_log_probs[batch_index],
                    transition_log_probs[batch_index],
                    run,
                ),
                dtype=torch.long,
                device=decoded.device,
            )
    return decoded


def _validate_decode_shapes(
    classification_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if (
        classification_logits.ndim != 3
        or classification_logits.shape[-1] != NUM_BIO_LABELS
    ):
        raise ValueError("classification_logits must have shape [batch, length, 3]")
    expected = (
        *classification_logits.shape[:2],
        NUM_BIO_LABELS,
        NUM_BIO_LABELS,
    )
    if tuple(transition_logits.shape) != expected:
        raise ValueError(f"transition_logits must have shape {expected}")
    if tuple(valid_mask.shape) != tuple(classification_logits.shape[:2]):
        raise ValueError("valid_mask shape must match batch and sequence dimensions")


def _decode_run(
    classification_log_probs: torch.Tensor,
    transition_log_probs: torch.Tensor,
    indices: Sequence[int],
) -> List[int]:
    if not indices:
        return []
    score = classification_log_probs[indices[0]]
    backpointers: List[torch.Tensor] = []
    for previous_index, current_index in zip(indices, indices[1:], strict=False):
        candidates = score[:, None] + transition_log_probs[previous_index]
        best_score, best_previous = candidates.max(dim=0)
        score = best_score + classification_log_probs[current_index]
        backpointers.append(best_previous)
    return _backtrack(score, backpointers)


def _backtrack(
    final_score: torch.Tensor, backpointers: Sequence[torch.Tensor]
) -> List[int]:
    current = int(final_score.argmax().item())
    path = [current]
    for pointer in reversed(backpointers):
        current = int(pointer[current].item())
        path.append(current)
    path.reverse()
    return path


def _contiguous_runs(indices: Sequence[int]) -> Iterable[List[int]]:
    if not indices:
        return
    run = [indices[0]]
    for index in indices[1:]:
        if index == run[-1] + 1:
            run.append(index)
            continue
        yield run
        run = [index]
    yield run


def compute_bio_metrics(
    predictions: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Compute token, retained-content and exact BIO-span metrics."""

    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    predicted_sequences: List[List[int]] = []
    gold_sequences: List[List[int]] = []
    for prediction, gold in zip(predictions.cpu(), labels.cpu(), strict=True):
        mask = gold != IGNORE_INDEX
        predicted_sequences.append(prediction[mask].tolist())
        gold_sequences.append(gold[mask].tolist())
    return compute_bio_metrics_from_sequences(predicted_sequences, gold_sequences)


def compute_bio_metrics_from_sequences(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
) -> Dict[str, float]:
    if len(predicted_sequences) != len(gold_sequences):
        raise ValueError("prediction and gold sequence counts differ")

    confusion = _build_confusion_matrix(predicted_sequences, gold_sequences)
    total_tokens = int(confusion.sum())
    content_counts = _content_classification_counts(predicted_sequences, gold_sequences)
    predicted_spans = _indexed_span_set(predicted_sequences)
    gold_spans = _indexed_span_set(gold_sequences)
    span_true_positives = len(predicted_spans & gold_spans)
    return {
        "token_accuracy": (
            float(confusion.diag().sum()) / total_tokens if total_tokens else 0.0
        ),
        "token_macro_f1": _macro_f1(confusion),
        "content_f1": _f1(*content_counts),
        "span_f1": _f1(
            span_true_positives,
            len(predicted_spans - gold_spans),
            len(gold_spans - predicted_spans),
        ),
        "tokens": float(total_tokens),
        "gold_spans": float(len(gold_spans)),
    }


def _build_confusion_matrix(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
) -> torch.Tensor:
    confusion = torch.zeros((NUM_BIO_LABELS, NUM_BIO_LABELS), dtype=torch.long)
    for prediction, gold in zip(predicted_sequences, gold_sequences, strict=True):
        if len(prediction) != len(gold):
            raise ValueError("prediction and gold sequence lengths differ")
        for gold_label, predicted_label in zip(gold, prediction, strict=True):
            confusion[int(gold_label), int(predicted_label)] += 1
    return confusion


def _macro_f1(confusion: torch.Tensor) -> float:
    scores = []
    for label in range(NUM_BIO_LABELS):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum()) - true_positive
        false_negative = int(confusion[label, :].sum()) - true_positive
        scores.append(_f1(true_positive, false_positive, false_negative))
    return sum(scores) / NUM_BIO_LABELS


def _content_classification_counts(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
) -> Tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    retained_labels = {BEGIN_LABEL_ID, INSIDE_LABEL_ID}
    for prediction, gold in zip(predicted_sequences, gold_sequences, strict=True):
        for predicted_label, gold_label in zip(prediction, gold, strict=True):
            predicted_content = predicted_label in retained_labels
            gold_content = gold_label in retained_labels
            true_positive += int(predicted_content and gold_content)
            false_positive += int(predicted_content and not gold_content)
            false_negative += int(not predicted_content and gold_content)
    return true_positive, false_positive, false_negative


def _indexed_span_set(
    sequences: Sequence[Sequence[int]],
) -> set[Tuple[int, int, int]]:
    return {
        (sample_index, start, end)
        for sample_index, sequence in enumerate(sequences)
        for start, end in bio_spans(sequence)
    }


def pad_and_cat(
    tensors: Sequence[torch.Tensor], pad_value: int = IGNORE_INDEX
) -> torch.Tensor:
    """Right-pad variable-width ``[batch, length]`` tensors and concatenate."""

    if not tensors:
        raise ValueError("at least one tensor is required")
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("all tensors must have shape [batch, length]")
    max_length = max(tensor.shape[1] for tensor in tensors)
    padded = [
        F.pad(tensor, (0, max_length - tensor.shape[1]), value=pad_value)
        if tensor.shape[1] < max_length
        else tensor
        for tensor in tensors
    ]
    return torch.cat(padded, dim=0)


def bio_spans(labels: Sequence[int]) -> List[Tuple[int, int]]:
    """Return half-open retained spans; a stray I starts a new span."""

    spans: List[Tuple[int, int]] = []
    start = None
    for index, label in enumerate(list(labels) + [OUTSIDE_LABEL_ID]):
        starts_span = label == BEGIN_LABEL_ID or (
            label == INSIDE_LABEL_ID and start is None
        )
        if starts_span:
            if start is not None:
                spans.append((start, index))
            start = index
        elif label == OUTSIDE_LABEL_ID and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0
