#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from select_repro.data import MLMCollator, PackedTokenDataset, load_dataset_metadata
from select_repro.modeling_bidirectional_qwen3 import Qwen3ForBidirectionalMaskedLM
from select_repro.training import cosine_learning_rate, validate_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SELECT stage-1 bidirectional MLM continued pre-training."
    )
    parser.add_argument("--config", required=True, help="JSON training configuration")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_training_config(config)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_dtypes(config: dict[str, Any]) -> tuple[torch.dtype, torch.dtype]:
    """Keep parameter storage independent from the autocast compute dtype."""
    compute_dtype = torch.bfloat16 if config["bf16"] else torch.float32
    parameter_dtype = (
        torch.float32
        if config.get("parameter_dtype", "float32") == "float32"
        else torch.bfloat16
    )
    return parameter_dtype, compute_dtype


def batch_order(num_sequences: int, micro_batch_size: int, seed: int, shuffle: bool) -> list[list[int]]:
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(num_sequences, generator=generator).tolist()
    else:
        order = list(range(num_sequences))
    return [order[start : start + micro_batch_size] for start in range(0, len(order), micro_batch_size)]


def build_resume_invariants(
    *,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    return {
        "base_model_path": str(resolve_path(config["base_model"])),
        "dataset_token_file_sha256": data_metadata["token_file_sha256"],
        "sequence_length": config["sequence_length"],
        "train_tokens": config["train_tokens"],
        "micro_batch_size": config["micro_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "mlm_probability": config["mlm_probability"],
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "tokenizer_size": len(tokenizer),
        "learning_rate": config["learning_rate"],
        "min_learning_rate": config["min_learning_rate"],
        "warmup_steps": config["warmup_steps"],
        "weight_decay": config["weight_decay"],
        "adam_beta1": config["adam_beta1"],
        "adam_beta2": config["adam_beta2"],
        "adam_epsilon": config["adam_epsilon"],
        "max_grad_norm": config["max_grad_norm"],
        "seed": config["seed"],
        "shuffle": config["shuffle"],
        "bf16": config["bf16"],
        "parameter_dtype": config.get("parameter_dtype", "float32"),
        "gradient_checkpointing": config["gradient_checkpointing"],
        "fused_optimizer": config["fused_optimizer"],
        "max_optimizer_steps": config.get("max_optimizer_steps"),
    }


def validate_resume_checkpoint(
    checkpoint_dir: Path, expected: dict[str, Any]
) -> None:
    manifest_path = checkpoint_dir / "stage1_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"resume checkpoint has no manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    saved = manifest.get("resume_invariants")
    if saved is None:
        raise ValueError(
            "resume checkpoint predates strict invariant validation; start a new run "
            "with this code version rather than resuming it"
        )
    mismatches = {
        key: {"checkpoint": saved.get(key), "current": value}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    unexpected = sorted(set(saved) - set(expected))
    if unexpected:
        mismatches["unexpected_checkpoint_keys"] = unexpected
    if mismatches:
        raise ValueError(
            "resume invariant mismatch:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Qwen3ForBidirectionalMaskedLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    collator: MLMCollator,
    global_step: int,
    batch_position: int,
    tokens_consumed: int,
    total_optimizer_steps: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "batch_position": batch_position,
        "tokens_consumed": tokens_consumed,
        "collator": collator.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    manifest = {
        "format_version": 1,
        "paper": "SELECting over Tokens (ACL 2026), stage-1 scaled reproduction",
        "base_model": config["base_model"],
        "objective": "masked_language_modeling",
        "bidirectional_attention": True,
        "attention_implementation": "sdpa",
        "parameter_dtype": str(next(model.parameters()).dtype),
        "compute_dtype": "torch.bfloat16" if config["bf16"] else "torch.float32",
        "sequence_length": config["sequence_length"],
        "mlm_probability": config["mlm_probability"],
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "model_vocab_size": model.config.vocab_size,
        "tokenizer_size": len(tokenizer),
        "global_step": global_step,
        "total_optimizer_steps": total_optimizer_steps,
        "tokens_consumed": tokens_consumed,
        "target_train_tokens": config["train_tokens"],
        "dataset_token_file_sha256": data_metadata["token_file_sha256"],
        "dataset_packing": data_metadata["packing"],
        "cross_document_attention": data_metadata["cross_document_attention"],
        "resume_invariants": build_resume_invariants(
            config=config, data_metadata=data_metadata, tokenizer=tokenizer
        ),
        "paper_disclosed_hyperparameters": {
            "base_model": "Qwen3-0.6B-Base",
            "source": "FineWeb",
            "objective": "MLM",
            "sequence_length": 32768,
            "batch_size": 512,
            "peak_learning_rate": 5e-5,
            "min_learning_rate": 1e-6,
            "tokens": 100_000_000_000,
        },
        "reproduction_choices_not_disclosed_by_paper": {
            "masking_policy": "15% targets with BERT 80/10/10 replacement",
            "optimizer": "AdamW",
            "adam_beta1": config["adam_beta1"],
            "adam_beta2": config["adam_beta2"],
            "weight_decay": config["weight_decay"],
            "warmup_steps": config["warmup_steps"],
        },
    }
    with (checkpoint_dir / "stage1_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 0.6B stage-1 training recipe")
    if config["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU/PyTorch build does not support bfloat16")

    base_model_dir = resolve_path(config["base_model"])
    data_dir = resolve_path(config["data_dir"])
    output_dir = resolve_path(config["output_dir"])
    resume_from = resolve_path(config["resume_from"]) if config.get("resume_from") else None
    if output_dir.exists() and any(output_dir.iterdir()) and resume_from is None:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new directory or resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    parameter_dtype, compute_dtype = resolve_training_dtypes(config)

    data_metadata = load_dataset_metadata(data_dir)
    if data_metadata["sequence_length"] != config["sequence_length"]:
        raise ValueError("data and training sequence lengths do not match")
    if data_metadata["train_tokens"] < config["train_tokens"]:
        raise ValueError("the prepared dataset contains fewer train tokens than requested")
    token_path = data_dir / data_metadata["token_file"]
    dataset = PackedTokenDataset(
        token_path,
        sequence_length=config["sequence_length"],
        offset_tokens=data_metadata["train_offset_tokens"],
        num_tokens=config["train_tokens"],
    )

    model_source = resume_from if resume_from is not None else base_model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": config["mask_token"]})
    elif tokenizer.mask_token != config["mask_token"]:
        raise ValueError(
            f"checkpoint mask token {tokenizer.mask_token!r} does not match config"
        )
    if resume_from is not None:
        validate_resume_checkpoint(
            resume_from,
            build_resume_invariants(
                config=config, data_metadata=data_metadata, tokenizer=tokenizer
            ),
        )

    model = Qwen3ForBidirectionalMaskedLM.from_pretrained(
        model_source,
        torch_dtype=parameter_dtype,
        attn_implementation="sdpa",
    )
    embedding_rows = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_rows:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    elif tokenizer.mask_token_id >= embedding_rows:
        raise ValueError("mask token ID is outside the model embedding table")
    model.config.use_cache = False
    model.config.bidirectional_attention = True
    model.config.stage1_objective = "mlm"
    model.config.mask_token_id = tokenizer.mask_token_id
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train().to(device)

    collator = MLMCollator(
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=len(tokenizer),
        special_token_ids=tokenizer.all_special_ids,
        mlm_probability=config["mlm_probability"],
        seed=int(config["seed"]) + 1,
    )
    use_fused = bool(config["fused_optimizer"] and device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
        fused=use_fused,
    )

    batches = batch_order(
        len(dataset),
        config["micro_batch_size"],
        int(config["seed"]),
        bool(config["shuffle"]),
    )
    natural_steps = math.ceil(len(batches) / config["gradient_accumulation_steps"])
    max_steps = config.get("max_optimizer_steps")
    total_optimizer_steps = natural_steps if max_steps is None else min(natural_steps, int(max_steps))
    if config["warmup_steps"] >= total_optimizer_steps:
        raise ValueError("warmup_steps must be smaller than the executed optimizer step count")

    global_step = 0
    batch_position = 0
    tokens_consumed = 0
    if resume_from is not None:
        state = torch.load(resume_from / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        batch_position = int(state["batch_position"])
        tokens_consumed = int(state["tokens_consumed"])
        collator.load_state_dict(state["collator"])
        torch.set_rng_state(state["torch_rng_state"])
        if state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "sequence_length": config["sequence_length"],
                "train_sequences": len(dataset),
                "optimizer_steps": total_optimizer_steps,
                "tokens_target": min(
                    config["train_tokens"],
                    total_optimizer_steps
                    * config["gradient_accumulation_steps"]
                    * config["micro_batch_size"]
                    * config["sequence_length"],
                ),
                "parameter_dtype": str(parameter_dtype),
                "compute_dtype": str(compute_dtype),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    log_path = output_dir / "train_log.jsonl"
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    while global_step < total_optimizer_steps and batch_position < len(batches):
        group_size = min(
            config["gradient_accumulation_steps"], len(batches) - batch_position
        )
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        step_tokens = 0
        step_started = time.perf_counter()

        for _ in range(group_size):
            indexes = batches[batch_position]
            examples = [dataset[index] for index in indexes]
            batch = collator(examples)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=compute_dtype, enabled=bool(config["bf16"])
            ):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                    return_full_logits=False,
                )
                raw_loss = outputs.loss
                loss = raw_loss / group_size
            loss.backward()
            losses.append(float(raw_loss.detach()))
            consumed = len(indexes) * config["sequence_length"]
            step_tokens += consumed
            tokens_consumed += consumed
            batch_position += 1

        learning_rate = cosine_learning_rate(
            global_step,
            total_steps=total_optimizer_steps,
            peak_lr=config["learning_rate"],
            min_lr=config["min_learning_rate"],
            warmup_steps=config["warmup_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        global_step += 1

        elapsed = max(time.perf_counter() - step_started, 1e-9)
        record = {
            "step": global_step,
            "loss": sum(losses) / len(losses),
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm),
            "step_tokens": step_tokens,
            "tokens_consumed": tokens_consumed,
            "tokens_per_second": step_tokens / elapsed,
            "elapsed_seconds": time.perf_counter() - run_started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if global_step % config["log_every_steps"] == 0 or global_step == 1:
            print(json.dumps(record, ensure_ascii=False), flush=True)

        save_every = int(config["save_every_steps"])
        if save_every and global_step % save_every == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"step-{global_step:06d}",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                config=config,
                data_metadata=data_metadata,
                collator=collator,
                global_step=global_step,
                batch_position=batch_position,
                tokens_consumed=tokens_consumed,
                total_optimizer_steps=total_optimizer_steps,
            )

    final_dir = output_dir / "final"
    save_checkpoint(
        final_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=config,
        data_metadata=data_metadata,
        collator=collator,
        global_step=global_step,
        batch_position=batch_position,
        tokens_consumed=tokens_consumed,
        total_optimizer_steps=total_optimizer_steps,
    )
    print(f"Training complete. Final checkpoint: {final_dir}", flush=True)


if __name__ == "__main__":
    main()
