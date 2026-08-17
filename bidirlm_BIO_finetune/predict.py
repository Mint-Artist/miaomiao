from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .data import BioDataCollator, BioJsonlDataset
from .decoding import compute_bio_metrics, pad_and_cat, viterbi_decode_batch
from .modeling import SelectBidirLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a SELECT BidirLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model-name-or-path")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attention-implementation", default="eager")
    return parser


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
    dataset = BioJsonlDataset(args.input, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=BioDataCollator(tokenizer.pad_token_id),
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
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            for sample_id, predicted, gold, mask in zip(
                batch["ids"], predictions.cpu(), labels.cpu(), attention_mask.cpu()
            ):
                length = int(mask.sum())
                predicted_values = predicted[:length].tolist()
                record = {
                    "id": sample_id,
                    "predicted_labels": predicted_values,
                    "predicted_bio_tags": [
                        {-100: "IGN", 0: "O", 1: "B", 2: "I"}[item]
                        for item in predicted_values
                    ],
                    "gold_labels": gold[:length].tolist(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = compute_bio_metrics(
        pad_and_cat(all_predictions), pad_and_cat(all_labels)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
