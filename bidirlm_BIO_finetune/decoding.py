from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def viterbi_decode_batch(
    classification_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Decode labels using log P_cls + log P_tr from the SELECT paper.

    ``transition_logits[:, i, u, v]`` scores the transition from label ``u``
    at position ``i`` to label ``v`` at position ``i + 1``. Masked positions
    are returned as ``-100``. Disconnected valid runs are decoded separately.
    """

    if classification_logits.ndim != 3 or classification_logits.shape[-1] != 3:
        raise ValueError("classification_logits must have shape [batch, length, 3]")
    expected = (*classification_logits.shape[:2], 3, 3)
    if tuple(transition_logits.shape) != expected:
        raise ValueError(f"transition_logits must have shape {expected}")
    if tuple(valid_mask.shape) != tuple(classification_logits.shape[:2]):
        raise ValueError("valid_mask shape must match batch and sequence dimensions")

    cls_log_probs = classification_logits.float().log_softmax(dim=-1)
    tr_log_probs = transition_logits.float().log_softmax(dim=-1)
    decoded = torch.full(
        valid_mask.shape,
        IGNORE_INDEX,
        dtype=torch.long,
        device=classification_logits.device,
    )
    for batch_index in range(classification_logits.shape[0]):
        indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten().tolist()
        for run in _contiguous_runs(indices):
            run_result = _decode_run(
                cls_log_probs[batch_index], tr_log_probs[batch_index], run
            )
            decoded[batch_index, run] = torch.tensor(
                run_result, dtype=torch.long, device=decoded.device
            )
    return decoded


def _decode_run(
    cls_log_probs: torch.Tensor,
    tr_log_probs: torch.Tensor,
    indices: Sequence[int],
) -> List[int]:
    if not indices:
        return []
    score = cls_log_probs[indices[0]]
    backpointers: List[torch.Tensor] = []
    for previous_index, current_index in zip(indices, indices[1:]):
        candidates = score[:, None] + tr_log_probs[previous_index]
        best_score, best_previous = candidates.max(dim=0)
        score = best_score + cls_log_probs[current_index]
        backpointers.append(best_previous)
    current = int(score.argmax().item())
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
        else:
            yield run
            run = [index]
    yield run


def compute_bio_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    boundary_tolerance: int = 2,
) -> Dict[str, float]:
    """Compute token, retained-content and exact BIO-span metrics."""

    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    predicted_sequences: List[List[int]] = []
    gold_sequences: List[List[int]] = []
    for prediction, gold in zip(predictions.cpu(), labels.cpu()):
        mask = gold != IGNORE_INDEX
        predicted_sequences.append(prediction[mask].tolist())
        gold_sequences.append(gold[mask].tolist())
    return compute_bio_metrics_from_sequences(
        predicted_sequences, gold_sequences, boundary_tolerance=boundary_tolerance
    )


def compute_bio_metrics_from_sequences(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
    *,
    boundary_tolerance: int = 2,
) -> Dict[str, float]:
    if len(predicted_sequences) != len(gold_sequences):
        raise ValueError("prediction and gold sequence counts differ")
    confusion = torch.zeros((3, 3), dtype=torch.long)
    for prediction, gold in zip(predicted_sequences, gold_sequences):
        if len(prediction) != len(gold):
            raise ValueError("prediction and gold sequence lengths differ")
        pred_values = [int(item) for item in prediction]
        gold_values = [int(item) for item in gold]
        for gold_label, predicted_label in zip(gold_values, pred_values):
            confusion[gold_label, predicted_label] += 1

    total = int(confusion.sum())
    token_accuracy = float(confusion.diag().sum()) / total if total else 0.0
    per_label_f1 = []
    per_label_counts = []
    for label in range(3):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum()) - tp
        fn = int(confusion[label, :].sum()) - tp
        per_label_counts.append((tp, fp, fn))
        per_label_f1.append(_f1(tp, fp, fn))
    b_tp, b_fp, b_fn = per_label_counts[1]
    b_precision = b_tp / (b_tp + b_fp) if b_tp + b_fp else 0.0
    b_recall = b_tp / (b_tp + b_fn) if b_tp + b_fn else 0.0

    content_tp = content_fp = content_fn = 0
    for prediction, gold in zip(predicted_sequences, gold_sequences):
        for pred_label, gold_label in zip(prediction, gold):
            pred_content = pred_label in (1, 2)
            gold_content = gold_label in (1, 2)
            content_tp += int(pred_content and gold_content)
            content_fp += int(pred_content and not gold_content)
            content_fn += int(not pred_content and gold_content)

    predicted_spans = {
        (sample_index, start, end)
        for sample_index, sequence in enumerate(predicted_sequences)
        for start, end in bio_spans(sequence)
    }
    gold_spans = {
        (sample_index, start, end)
        for sample_index, sequence in enumerate(gold_sequences)
        for start, end in bio_spans(sequence)
    }
    span_tp = len(predicted_spans & gold_spans)

    tolerant_tp = 0
    for sample_index in range(len(predicted_sequences)):
        tolerant_tp += _tolerant_span_matches(
            bio_spans(predicted_sequences[sample_index]),
            bio_spans(gold_sequences[sample_index]),
            boundary_tolerance,
        )
    return {
        "token_accuracy": token_accuracy,
        "token_macro_f1": sum(per_label_f1) / len(per_label_f1),
        "token_f1_O": per_label_f1[0],
        "token_f1_B": per_label_f1[1],
        "token_f1_I": per_label_f1[2],
        "b_precision": b_precision,
        "b_recall": b_recall,
        "content_f1": _f1(content_tp, content_fp, content_fn),
        "span_f1": _f1(
            span_tp,
            len(predicted_spans - gold_spans),
            len(gold_spans - predicted_spans),
        ),
        "span_f1_tolerant": _f1(
            tolerant_tp,
            len(predicted_spans) - tolerant_tp,
            len(gold_spans) - tolerant_tp,
        ),
        "boundary_tolerance": float(boundary_tolerance),
        "tokens": float(total),
        "gold_spans": float(len(gold_spans)),
    }


def _tolerant_span_matches(
    predicted_spans: Sequence[Tuple[int, int]],
    gold_spans: Sequence[Tuple[int, int]],
    tolerance: int,
) -> int:
    """Greedily one-to-one match spans whose boundaries differ by <= tolerance."""

    matched = 0
    used = [False] * len(predicted_spans)
    for gold_start, gold_end in gold_spans:
        for index, (predicted_start, predicted_end) in enumerate(predicted_spans):
            if used[index]:
                continue
            if (
                abs(predicted_start - gold_start) <= tolerance
                and abs(predicted_end - gold_end) <= tolerance
            ):
                used[index] = True
                matched += 1
                break
    return matched


def pad_and_cat(tensors: Sequence[torch.Tensor], pad_value: int = IGNORE_INDEX) -> torch.Tensor:
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
    for index, label in enumerate(list(labels) + [0]):
        if label == 1 or (label == 2 and start is None):
            if start is not None:
                spans.append((start, index))
            start = index
        elif label == 0 and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2.0 * tp / denominator) if denominator else 0.0
