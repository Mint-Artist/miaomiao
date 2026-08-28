from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .constants import (
    ATTENTION_MASK_KEY,
    BEGIN_LABEL_ID,
    CLASSIFICATION_LOGITS_KEY,
    DEFAULT_ATTENTION_IMPLEMENTATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    INSIDE_LABEL_ID,
    LABEL_TO_TAG,
    LABELS_KEY,
    OUTSIDE_LABEL_ID,
    TRANSITION_LOGITS_KEY,
)
from .data import BioDataCollator, validate_record
from .decoding import compute_bio_metrics, pad_and_cat, viterbi_decode_batch
from .modeling import SelectBidirLM

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference or evaluation with a SELECT BidirLM checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model-name-or-path")
    parser.add_argument(
        "--input-format",
        choices=("auto", "raw", "sft", "audit"),
        default="auto",
        help="auto detects raw text, minimal SFT, or accepted audit JSONL",
    )
    parser.add_argument(
        "--text-field",
        default="source_text",
        help="text field used by raw JSONL input (default: source_text)",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="ID field used by raw JSONL input (default: id)",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--attention-implementation", default=DEFAULT_ATTENTION_IMPLEMENTATION
    )
    return parser


class PredictionJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        input_format: str = "auto",
        max_length: int = DEFAULT_MAX_LENGTH,
        tokenizer: Any = None,
        text_field: str = "source_text",
        id_field: str = "id",
    ):
        self.path = Path(path)
        self.records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                self.records.append(
                    normalize_prediction_record(
                        value,
                        line_number=line_number,
                        input_format=input_format,
                        max_length=max_length,
                        tokenizer=tokenizer,
                        text_field=text_field,
                        id_field=id_field,
                    )
                )
        if not self.records:
            raise ValueError(f"no prediction records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


class PredictionCollator:
    def __init__(self, pad_token_id: int):
        self.base_collator = BioDataCollator(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(features)
        batch.update(
            {"metadata": [feature.get("prediction_metadata") for feature in features]}
        )
        return batch


def normalize_prediction_record(
    value: Any,
    *,
    line_number: int,
    input_format: str,
    max_length: int,
    tokenizer: Any = None,
    text_field: str = "source_text",
    id_field: str = "id",
) -> Dict[str, Any]:
    detected_format = _detect_input_format(value, input_format, line_number)

    if detected_format != "raw" and not isinstance(value, Mapping):
        raise TypeError(f"line {line_number}: {detected_format} row must be an object")

    if detected_format == "audit":
        record, metadata = _normalize_audit_record(
            value, line_number=line_number, max_length=max_length
        )
    elif detected_format == "sft":
        record, metadata = _normalize_sft_record(
            value, line_number=line_number, max_length=max_length
        )
    elif detected_format == "raw":
        record, metadata = tokenize_raw_prediction_record(
            value,
            line_number=line_number,
            max_length=max_length,
            tokenizer=tokenizer,
            text_field=text_field,
            id_field=id_field,
        )
    else:
        raise ValueError(f"unsupported input format: {detected_format}")
    record.update({"prediction_metadata": metadata})
    return record


def _detect_input_format(value: Any, requested: str, line_number: int) -> str:
    if requested != "auto":
        return requested
    if isinstance(value, str):
        return "raw"
    if not isinstance(value, Mapping):
        raise TypeError(
            f"line {line_number}: JSONL row must be an object or a JSON string"
        )
    if "tokenization" in value or "supervision" in value:
        return "audit"
    sequence_fields = (INPUT_IDS_KEY, ATTENTION_MASK_KEY, LABELS_KEY)
    return "sft" if any(field in value for field in sequence_fields) else "raw"


def _normalize_audit_record(
    value: Mapping[str, Any], *, line_number: int, max_length: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tokenization = value.get("tokenization")
    supervision = value.get("supervision")
    if not isinstance(tokenization, Mapping) or not isinstance(supervision, Mapping):
        raise TypeError(
            f"line {line_number}: audit row requires tokenization and "
            "supervision objects"
        )
    record = validate_record(
        {
            "id": value.get("id"),
            INPUT_IDS_KEY: tokenization.get(INPUT_IDS_KEY),
            ATTENTION_MASK_KEY: tokenization.get(ATTENTION_MASK_KEY),
            LABELS_KEY: supervision.get(LABELS_KEY),
        },
        line_number=line_number,
        max_length=max_length,
    )
    source_text = value.get("source_text")
    if not isinstance(source_text, str):
        raise TypeError(f"line {line_number}: audit source_text must be a string")
    offsets = _normalize_offsets(
        tokenization.get("offset_mapping"),
        expected_length=len(record.get(INPUT_IDS_KEY, [])),
        source_text=source_text,
        line_number=line_number,
    )
    metadata = {
        "input_format": "audit",
        "sequence_length": len(record.get(INPUT_IDS_KEY, [])),
        "source_text": source_text,
        "offset_mapping": offsets,
        "teacher_refined_text": value.get(
            "teacher_refined_text", value.get("refined_text")
        ),
        "adjusted_refined_text": value.get("adjusted_refined_text"),
        "has_gold": True,
    }
    return record, metadata


def _normalize_sft_record(
    value: Mapping[str, Any], *, line_number: int, max_length: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    record = validate_record(value, line_number=line_number, max_length=max_length)
    metadata = {
        "input_format": "sft",
        "sequence_length": len(record.get(INPUT_IDS_KEY, [])),
        "source_text": value.get("source_text"),
        "offset_mapping": value.get("offset_mapping"),
        "teacher_refined_text": value.get("teacher_refined_text"),
        "adjusted_refined_text": value.get("adjusted_refined_text"),
        "has_gold": True,
    }
    return record, metadata


def tokenize_raw_prediction_record(
    value: Any,
    *,
    line_number: int,
    max_length: int,
    tokenizer: Any,
    text_field: str,
    id_field: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if tokenizer is None:
        raise ValueError("raw prediction input requires a tokenizer")
    sample_id, source_text = _extract_raw_text(
        value,
        line_number=line_number,
        text_field=text_field,
        id_field=id_field,
    )
    encoded = _tokenize_raw_text(tokenizer, source_text)
    input_ids = _integer_list(encoded.get(INPUT_IDS_KEY), INPUT_IDS_KEY)
    attention_mask = _integer_list(encoded.get(ATTENTION_MASK_KEY), ATTENTION_MASK_KEY)
    if len(attention_mask) != len(input_ids):
        raise RuntimeError("tokenizer attention_mask length differs from input_ids")
    offset_mapping = _normalize_offsets(
        encoded.get("offset_mapping"),
        expected_length=len(input_ids),
        source_text=source_text,
        line_number=line_number,
    )
    placeholder_labels = _raw_validity_labels(attention_mask, offset_mapping)
    record = validate_record(
        {
            "id": sample_id,
            INPUT_IDS_KEY: input_ids,
            ATTENTION_MASK_KEY: attention_mask,
            LABELS_KEY: placeholder_labels,
        },
        line_number=line_number,
        max_length=max_length,
    )
    metadata = {
        "input_format": "raw",
        "sequence_length": len(input_ids),
        "source_text": source_text,
        "offset_mapping": offset_mapping,
        "teacher_refined_text": None,
        "adjusted_refined_text": None,
        "has_gold": False,
    }
    return record, metadata


def _extract_raw_text(
    value: Any,
    *,
    line_number: int,
    text_field: str,
    id_field: str,
) -> Tuple[Any, str]:
    if isinstance(value, str):
        source_text = value
        sample_id: Any = f"line-{line_number}"
    elif isinstance(value, Mapping):
        source_text = value.get(text_field)
        sample_id = value.get(id_field) or f"line-{line_number}"
        if not isinstance(source_text, str):
            raise TypeError(
                f"line {line_number}: raw text field {text_field!r} must be a string"
            )
    else:
        raise TypeError(
            f"line {line_number}: raw JSONL row must be an object or a JSON string"
        )
    if not source_text:
        raise ValueError(f"line {line_number}: raw text must not be empty")
    return sample_id, source_text


def _raw_validity_labels(
    attention_mask: Sequence[int], offset_mapping: Sequence[Sequence[int]]
) -> List[int]:
    """Build an internal valid-token mask; these are not Gold labels."""

    return [
        OUTSIDE_LABEL_ID if mask and end > start else IGNORE_INDEX
        for mask, (start, end) in zip(attention_mask, offset_mapping, strict=True)
    ]


def _tokenize_raw_text(tokenizer: Any, source_text: str) -> Mapping[str, Any]:
    try:
        encoded = tokenizer(
            source_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            "raw prediction requires a fast tokenizer with offset_mapping support"
        ) from exc
    if not isinstance(encoded, Mapping):
        raise RuntimeError("tokenizer output must be a mapping")
    return encoded


def _integer_list(value: Any, field: str) -> List[int]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"tokenizer did not return a {field} sequence")
    return [int(item) for item in value]


def _normalize_offsets(
    value: Any,
    *,
    expected_length: int,
    source_text: str,
    line_number: int,
) -> List[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError(
            f"line {line_number}: offset_mapping must match input_ids length"
        )
    offsets = [_normalize_offset(offset, source_text, line_number) for offset in value]
    return offsets


def _normalize_offset(offset: Any, source_text: str, line_number: int) -> List[int]:
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        raise ValueError(f"line {line_number}: each offset must contain two integers")
    start, end = int(offset[0]), int(offset[1])
    if start < 0 or end < start or end > len(source_text):
        raise ValueError(
            f"line {line_number}: offset [{start}, {end}] is outside source_text"
        )
    return [start, end]


@torch.no_grad()
def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    model, tokenizer = _load_inference_artifacts(args, device)
    dataset, loader = _build_prediction_loader(args, tokenizer)
    predictions, labels = _write_prediction_file(
        args.output,
        loader,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    summary = _prediction_summary(dataset, predictions, labels)
    LOGGER.info(
        "prediction summary: %s",
        json.dumps(summary, ensure_ascii=False),
    )
    return 0


def _load_inference_artifacts(
    args: argparse.Namespace, device: torch.device
) -> Tuple[SelectBidirLM, Any]:
    model = SelectBidirLM.from_checkpoint(
        args.checkpoint,
        base_model_name_or_path=args.base_model_name_or_path,
        attention_implementation=args.attention_implementation,
    ).to(device)
    model.eval()
    tokenizer_path = Path(args.checkpoint) / "tokenizer"
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install bidirlm_BIO_finetune/requirements.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return model, tokenizer


def _build_prediction_loader(
    args: argparse.Namespace, tokenizer: Any
) -> Tuple[PredictionJsonlDataset, DataLoader]:
    dataset = PredictionJsonlDataset(
        args.input,
        input_format=args.input_format,
        max_length=args.max_length,
        tokenizer=tokenizer,
        text_field=args.text_field,
        id_field=args.id_field,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PredictionCollator(tokenizer.pad_token_id),
    )
    return dataset, loader


def _write_prediction_file(
    output: str,
    loader: DataLoader,
    *,
    model: SelectBidirLM,
    tokenizer: Any,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    all_predictions: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for batch in loader:
            predictions, labels = _predict_batch(batch, model, device)
            _write_batch_outputs(
                stream,
                batch,
                predictions,
                labels,
                tokenizer,
                all_predictions,
                all_labels,
            )
    return all_predictions, all_labels


def _predict_batch(
    batch: Mapping[str, Any],
    model: SelectBidirLM,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = _require_tensor(batch, INPUT_IDS_KEY).to(device)
    attention_mask = _require_tensor(batch, ATTENTION_MASK_KEY).to(device)
    labels = _require_tensor(batch, LABELS_KEY).to(device)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        result = model(input_ids=input_ids, attention_mask=attention_mask)
    valid_mask = (labels != IGNORE_INDEX) & attention_mask.bool()
    predictions = viterbi_decode_batch(
        _require_tensor(result, CLASSIFICATION_LOGITS_KEY),
        _require_tensor(result, TRANSITION_LOGITS_KEY),
        valid_mask,
    )
    return predictions.cpu(), labels.cpu()


def _write_batch_outputs(
    stream: Any,
    batch: Mapping[str, Any],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    all_predictions: List[torch.Tensor],
    all_labels: List[torch.Tensor],
) -> None:
    for sample_id, predicted, gold, ids, metadata in zip(
        _require_sequence(batch, "ids"),
        predictions,
        labels,
        _require_tensor(batch, INPUT_IDS_KEY),
        _require_sequence(batch, "metadata"),
        strict=True,
    ):
        if not isinstance(metadata, Mapping):
            raise TypeError("prediction metadata must be a mapping")
        length = int(metadata.get("sequence_length", 0))
        gold_labels = (
            gold[:length].tolist() if metadata.get("has_gold", False) else None
        )
        if gold_labels is not None:
            all_predictions.append(predicted[:length].unsqueeze(0))
            all_labels.append(gold[:length].unsqueeze(0))
        output_record = build_prediction_output(
            sample_id=sample_id,
            input_ids=ids[:length].tolist(),
            predicted_labels=predicted[:length].tolist(),
            gold_labels=gold_labels,
            metadata=metadata,
            tokenizer=tokenizer,
        )
        stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def _prediction_summary(
    dataset: PredictionJsonlDataset,
    predictions: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    if labels:
        summary: Dict[str, Any] = compute_bio_metrics(
            pad_and_cat(predictions), pad_and_cat(labels)
        )
        summary.update({"evaluated_samples": len(labels)})
        return summary
    return {
        "prediction_samples": len(dataset),
        "evaluated_samples": 0,
        "message": "raw inference completed; no Gold labels, metrics skipped",
    }


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"required tensor field {key!r} is missing")
    return value


def _require_sequence(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = mapping.get(key)
    if not isinstance(value, Sequence):
        raise TypeError(f"required sequence field {key!r} is missing")
    return value


def build_prediction_output(
    *,
    sample_id: Any,
    input_ids: Sequence[int],
    predicted_labels: Sequence[int],
    gold_labels: Optional[Sequence[int]],
    metadata: Mapping[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    predicted_spans = labels_to_token_spans(predicted_labels)
    gold_spans = labels_to_token_spans(gold_labels) if gold_labels is not None else []
    output = _predicted_output_fields(
        sample_id=sample_id,
        input_ids=input_ids,
        labels=predicted_labels,
        spans=predicted_spans,
        input_format=metadata.get("input_format"),
        tokenizer=tokenizer,
    )
    if gold_labels is not None:
        output.update(
            _gold_output_fields(input_ids, gold_labels, gold_spans, tokenizer)
        )

    source_text = metadata.get("source_text")
    offset_mapping = metadata.get("offset_mapping")
    if isinstance(source_text, str) and isinstance(offset_mapping, list):
        output.update(
            _source_prediction_fields(
                source_text,
                offset_mapping,
                predicted_labels,
                predicted_spans,
            )
        )
        if gold_labels is not None:
            output.update(
                _source_gold_fields(
                    source_text,
                    offset_mapping,
                    gold_labels,
                    gold_spans,
                    metadata,
                )
            )
    return output


def _predicted_output_fields(
    *,
    sample_id: Any,
    input_ids: Sequence[int],
    labels: Sequence[int],
    spans: Sequence[Tuple[int, int]],
    input_format: Any,
    tokenizer: Any,
) -> Dict[str, Any]:
    decoded_segments = _decode_segments(input_ids, spans, tokenizer)
    return {
        "id": sample_id,
        "input_format": input_format,
        "predicted_labels": list(labels),
        "predicted_bio_tags": [_tag_for_label(item) for item in labels],
        "predicted_token_spans": [list(span) for span in spans],
        "decoded_input_text": tokenizer.decode(input_ids, skip_special_tokens=True),
        "predicted_decoded_segments": decoded_segments,
        "predicted_decoded_text": "".join(decoded_segments),
    }


def _gold_output_fields(
    input_ids: Sequence[int],
    labels: Sequence[int],
    spans: Sequence[Tuple[int, int]],
    tokenizer: Any,
) -> Dict[str, Any]:
    decoded_segments = _decode_segments(input_ids, spans, tokenizer)
    return {
        "gold_labels": list(labels),
        "gold_bio_tags": [_tag_for_label(item) for item in labels],
        "gold_token_spans": [list(span) for span in spans],
        "gold_decoded_segments": decoded_segments,
        "gold_decoded_text": "".join(decoded_segments),
    }


def _source_prediction_fields(
    source_text: str,
    offset_mapping: Sequence[Sequence[int]],
    labels: Sequence[int],
    spans: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    segments = token_spans_to_text_segments(source_text, offset_mapping, spans, labels)
    return {
        "source_text": source_text,
        "predicted_retained_segments": segments,
        "predicted_refined_text": _join_segment_text(segments),
    }


def _source_gold_fields(
    source_text: str,
    offset_mapping: Sequence[Sequence[int]],
    labels: Sequence[int],
    spans: Sequence[Tuple[int, int]],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    segments = token_spans_to_text_segments(source_text, offset_mapping, spans, labels)
    return {
        "gold_retained_segments": segments,
        "gold_refined_text_from_labels": _join_segment_text(segments),
        "teacher_refined_text": metadata.get("teacher_refined_text"),
        "adjusted_refined_text": metadata.get("adjusted_refined_text"),
    }


def _decode_segments(
    input_ids: Sequence[int],
    spans: Sequence[Tuple[int, int]],
    tokenizer: Any,
) -> List[str]:
    return [
        tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
        for start, end in spans
    ]


def _join_segment_text(segments: Sequence[Mapping[str, Any]]) -> str:
    return "".join(str(segment.get("text", "")) for segment in segments)


def _tag_for_label(label: int) -> str:
    tag = LABEL_TO_TAG.get(int(label))
    if tag is None:
        raise ValueError(f"unsupported predicted label: {label}")
    return tag


def labels_to_token_spans(labels: Sequence[int]) -> List[Tuple[int, int]]:
    """Return retained spans without changing an I that follows O.

    A stray I starts a retained span for text inspection, while its original
    predicted label remains I in the JSONL output.
    """

    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, label in enumerate(list(labels) + [OUTSIDE_LABEL_ID]):
        label = int(label)
        if label == BEGIN_LABEL_ID:
            if start is not None:
                spans.append((start, index))
            start = index
        elif label == INSIDE_LABEL_ID:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    return spans


def token_spans_to_text_segments(
    source_text: str,
    offset_mapping: Sequence[Sequence[int]],
    token_spans: Sequence[Tuple[int, int]],
    labels: Sequence[int],
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for token_start, token_end in token_spans:
        real_offsets = [
            (int(offset[0]), int(offset[1]))
            for offset in offset_mapping[token_start:token_end]
            if int(offset[1]) > int(offset[0])
        ]
        if not real_offsets:
            continue
        char_start = real_offsets[0][0]
        char_end = real_offsets[-1][1]
        segments.append(
            {
                "token_span": [token_start, token_end],
                "char_span": [char_start, char_end],
                "starts_with_tag": _tag_for_label(labels[token_start]),
                "text": source_text[char_start:char_end],
            }
        )
    return segments


if __name__ == "__main__":
    raise SystemExit(main())
