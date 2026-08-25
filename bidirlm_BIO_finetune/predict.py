from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .data import BioDataCollator, validate_record
from .decoding import compute_bio_metrics, pad_and_cat, viterbi_decode_batch
from .modeling import SelectBidirLM


LABEL_TO_TAG = {-100: "IGN", 0: "O", 1: "B", 2: "I"}


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
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attention-implementation", default="eager")
    return parser


class PredictionJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        input_format: str = "auto",
        max_length: int = 8192,
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
        batch["metadata"] = [feature["prediction_metadata"] for feature in features]
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
    detected_format = input_format
    if input_format == "auto":
        if isinstance(value, str):
            detected_format = "raw"
        elif not isinstance(value, Mapping):
            raise TypeError(
                f"line {line_number}: JSONL row must be an object or a JSON string"
            )
        elif "tokenization" in value or "supervision" in value:
            detected_format = "audit"
        elif any(field in value for field in ("input_ids", "attention_mask", "labels")):
            detected_format = "sft"
        else:
            detected_format = "raw"

    if detected_format != "raw" and not isinstance(value, Mapping):
        raise TypeError(f"line {line_number}: {detected_format} row must be an object")

    if detected_format == "audit":
        tokenization = value.get("tokenization")
        supervision = value.get("supervision")
        if not isinstance(tokenization, Mapping) or not isinstance(
            supervision, Mapping
        ):
            raise TypeError(
                f"line {line_number}: audit row requires tokenization and supervision objects"
            )
        flat = {
            "id": value.get("id"),
            "input_ids": tokenization.get("input_ids"),
            "attention_mask": tokenization.get("attention_mask"),
            "labels": supervision.get("labels"),
        }
        record = validate_record(
            flat, line_number=line_number, max_length=max_length
        )
        offset_mapping = tokenization.get("offset_mapping")
        if not isinstance(offset_mapping, list) or len(offset_mapping) != len(
            record["input_ids"]
        ):
            raise ValueError(
                f"line {line_number}: audit offset_mapping must match input_ids length"
            )
        normalized_offsets = []
        for offset in offset_mapping:
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(
                    f"line {line_number}: each offset must contain two integers"
                )
            normalized_offsets.append([int(offset[0]), int(offset[1])])
        source_text = value.get("source_text")
        if not isinstance(source_text, str):
            raise TypeError(f"line {line_number}: audit source_text must be a string")
        for start, end in normalized_offsets:
            if start < 0 or end < start or end > len(source_text):
                raise ValueError(
                    f"line {line_number}: offset [{start}, {end}] is outside source_text"
                )
        metadata = {
            "input_format": "audit",
            "sequence_length": len(record["input_ids"]),
            "source_text": source_text,
            "offset_mapping": normalized_offsets,
            "teacher_refined_text": value.get(
                "teacher_refined_text", value.get("refined_text")
            ),
            "adjusted_refined_text": value.get("adjusted_refined_text"),
            "has_gold": True,
        }
    elif detected_format == "sft":
        record = validate_record(
            value, line_number=line_number, max_length=max_length
        )
        metadata = {
            "input_format": "sft",
            "sequence_length": len(record["input_ids"]),
            "source_text": value.get("source_text"),
            "offset_mapping": value.get("offset_mapping"),
            "teacher_refined_text": value.get("teacher_refined_text"),
            "adjusted_refined_text": value.get("adjusted_refined_text"),
            "has_gold": True,
        }
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
    record["prediction_metadata"] = metadata
    return record


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
    input_ids = [int(item) for item in encoded["input_ids"]]
    attention_mask = [int(item) for item in encoded["attention_mask"]]
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(raw_offsets, (list, tuple)):
        raise RuntimeError(
            "raw prediction tokenizer did not return an offset_mapping sequence"
        )
    offset_mapping: List[List[int]] = []
    for offset in raw_offsets:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise RuntimeError("tokenizer returned an invalid offset_mapping item")
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start or end > len(source_text):
            raise RuntimeError(
                f"tokenizer offset [{start}, {end}] is outside raw source text"
            )
        offset_mapping.append([start, end])
    if len(offset_mapping) != len(input_ids):
        raise RuntimeError("tokenizer offset_mapping length differs from input_ids")

    # Labels are used only as a valid-token mask by the existing collator and
    # Viterbi path. They are never emitted as Gold or included in metrics.
    placeholder_labels = [
        0 if mask and end > start else -100
        for mask, (start, end) in zip(attention_mask, offset_mapping)
    ]
    record = validate_record(
        {
            "id": sample_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": placeholder_labels,
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


@torch.no_grad()
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
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
    all_predictions = []
    all_labels = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = model(input_ids=input_ids, attention_mask=attention_mask)
            valid_mask = (labels != -100) & attention_mask.bool()
            predictions = viterbi_decode_batch(
                result["classification_logits"], result["transition_logits"], valid_mask
            )
            for sample_id, predicted, gold, ids, metadata in zip(
                batch["ids"],
                predictions.cpu(),
                labels.cpu(),
                batch["input_ids"],
                batch["metadata"],
            ):
                length = int(metadata["sequence_length"])
                predicted_labels = predicted[:length].tolist()
                gold_labels = gold[:length].tolist() if metadata["has_gold"] else None
                if gold_labels is not None:
                    all_predictions.append(predicted[:length].unsqueeze(0).cpu())
                    all_labels.append(gold[:length].unsqueeze(0).cpu())
                output_record = build_prediction_output(
                    sample_id=sample_id,
                    input_ids=ids[:length].tolist(),
                    predicted_labels=predicted_labels,
                    gold_labels=gold_labels,
                    metadata=metadata,
                    tokenizer=tokenizer,
                )
                stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    if all_labels:
        summary = compute_bio_metrics(
            pad_and_cat(all_predictions), pad_and_cat(all_labels)
        )
        summary["evaluated_samples"] = len(all_labels)
    else:
        summary = {
            "prediction_samples": len(dataset),
            "evaluated_samples": 0,
            "message": "raw inference completed; no Gold labels, metrics skipped",
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


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
    predicted_decoded_segments = [
        tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
        for start, end in predicted_spans
    ]
    gold_decoded_segments = [
        tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
        for start, end in gold_spans
    ]
    output: Dict[str, Any] = {
        "id": sample_id,
        "input_format": metadata.get("input_format"),
        "predicted_labels": list(predicted_labels),
        "predicted_bio_tags": [LABEL_TO_TAG[int(item)] for item in predicted_labels],
        "predicted_token_spans": [list(span) for span in predicted_spans],
        "decoded_input_text": tokenizer.decode(
            input_ids, skip_special_tokens=True
        ),
        "predicted_decoded_segments": predicted_decoded_segments,
        "predicted_decoded_text": "".join(predicted_decoded_segments),
    }
    if gold_labels is not None:
        output.update(
            {
                "gold_labels": list(gold_labels),
                "gold_bio_tags": [LABEL_TO_TAG[int(item)] for item in gold_labels],
                "gold_token_spans": [list(span) for span in gold_spans],
                "gold_decoded_segments": gold_decoded_segments,
                "gold_decoded_text": "".join(gold_decoded_segments),
            }
        )

    source_text = metadata.get("source_text")
    offset_mapping = metadata.get("offset_mapping")
    if isinstance(source_text, str) and isinstance(offset_mapping, list):
        predicted_text_segments = token_spans_to_text_segments(
            source_text, offset_mapping, predicted_spans, predicted_labels
        )
        output.update(
            {
                "source_text": source_text,
                "predicted_retained_segments": predicted_text_segments,
                "predicted_refined_text": "".join(
                    segment["text"] for segment in predicted_text_segments
                ),
            }
        )
        if gold_labels is not None:
            gold_text_segments = token_spans_to_text_segments(
                source_text, offset_mapping, gold_spans, gold_labels
            )
            output.update(
                {
                    "gold_retained_segments": gold_text_segments,
                    "gold_refined_text_from_labels": "".join(
                        segment["text"] for segment in gold_text_segments
                    ),
                    "teacher_refined_text": metadata.get("teacher_refined_text"),
                    "adjusted_refined_text": metadata.get("adjusted_refined_text"),
                }
            )
    return output


def labels_to_token_spans(labels: Sequence[int]) -> List[Tuple[int, int]]:
    """Return retained spans without changing an I that follows O.

    A stray I starts a retained span for text inspection, while its original
    predicted label remains I in the JSONL output.
    """

    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, label in enumerate(list(labels) + [0]):
        label = int(label)
        if label == 1:
            if start is not None:
                spans.append((start, index))
            start = index
        elif label == 2:
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
                "starts_with_tag": LABEL_TO_TAG[int(labels[token_start])],
                "text": source_text[char_start:char_end],
            }
        )
    return segments


if __name__ == "__main__":
    raise SystemExit(main())
