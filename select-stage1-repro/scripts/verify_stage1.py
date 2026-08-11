#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from select_repro.modeling_bidirectional_qwen3 import Qwen3ForBidirectionalMaskedLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a SELECT stage-1 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--skip-data-hash", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint = resolve(args.checkpoint)
    with (checkpoint / "stage1_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest["bidirectional_attention"] or manifest["objective"] != "masked_language_modeling":
        raise ValueError("checkpoint manifest does not describe SELECT stage-1")

    weight_path = checkpoint / "model.safetensors"
    with safe_open(weight_path, framework="pt", device="cpu") as weights:
        weight_keys = list(weights.keys())
    if not weight_keys or "model.embed_tokens.weight" not in weight_keys:
        raise ValueError("checkpoint safetensors is incomplete")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.mask_token != manifest["mask_token"]:
        raise ValueError("saved tokenizer mask token does not match the manifest")
    if tokenizer.mask_token_id != manifest["mask_token_id"]:
        raise ValueError("saved tokenizer mask token ID does not match the manifest")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen3ForBidirectionalMaskedLM.from_pretrained(
        checkpoint, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    input_a = torch.tensor([[10, 11, tokenizer.mask_token_id, 12]], device=device)
    input_b = input_a.clone()
    input_b[0, -1] = 13
    labels = torch.tensor([[-100, -100, 42, -100]], device=device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output_a = model(input_ids=input_a, return_full_logits=True, use_cache=False)
        output_b = model(input_ids=input_b, return_full_logits=True, use_cache=False)
        masked_output = model(
            input_ids=input_a,
            labels=labels,
            return_full_logits=False,
            use_cache=False,
        )
    right_context_delta = float((output_a.logits[0, 0] - output_b.logits[0, 0]).abs().max())
    if right_context_delta <= 0.0:
        raise ValueError("future token did not affect a left-position logit; attention is not bidirectional")
    if masked_output.logits.shape != (1, model.config.vocab_size):
        raise ValueError("masked-only logits path returned an unexpected shape")
    if not math.isfinite(float(masked_output.loss)):
        raise ValueError("MLM loss is not finite")

    data_verified = None
    if args.data_dir:
        data_dir = resolve(args.data_dir)
        with (data_dir / "metadata.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata["sequence_length"] != manifest["sequence_length"]:
            raise ValueError("dataset and checkpoint sequence lengths do not match")
        if metadata["token_file_sha256"] != manifest["dataset_token_file_sha256"]:
            raise ValueError("dataset and checkpoint fingerprints do not match")
        if not args.skip_data_hash:
            actual_hash = sha256(data_dir / metadata["token_file"])
            if actual_hash != metadata["token_file_sha256"]:
                raise ValueError("dataset token file hash verification failed")
        data_verified = True

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "weights_bytes": weight_path.stat().st_size,
                "weight_tensors": len(weight_keys),
                "mask_token_id": tokenizer.mask_token_id,
                "right_context_logit_delta": right_context_delta,
                "masked_loss": float(masked_output.loss),
                "masked_logits_shape": list(masked_output.logits.shape),
                "data_verified": data_verified,
                "status": "ok",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
