from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .data import BioDataCollator, BioJsonlDataset
from .decoding import compute_bio_metrics_from_sequences, viterbi_decode_batch
from .modeling import SelectBidirLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune BidirLM with SELECT BIO labels")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--finetuning-mode", choices=("lora", "full", "heads_only"), default="lora"
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for BidirLM fine-tuning")
    if not args.no_fp16 and not torch.cuda.get_device_capability(device)[0] >= 7:
        raise RuntimeError("FP16 training requires a CUDA GPU with compute capability >= 7.0")
    seed_everything(args.seed + rank)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install bidirlm_BIO_finetune/requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = BioJsonlDataset(args.train_file, max_length=args.max_length)
    validation_dataset = BioJsonlDataset(args.validation_file, max_length=args.max_length)
    collator = BioDataCollator(tokenizer.pad_token_id)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=(
            DistributedEvalSampler(validation_dataset, rank, world_size)
            if distributed
            else None
        ),
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = _build_model(args)
    model.to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    unwrapped = unwrap(model)
    learning_rate = args.learning_rate or (
        1e-4 if args.finetuning_mode == "lora" else 1e-5
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        make_lr_lambda(
            total_updates, warmup_updates, args.min_learning_rate_ratio
        ),
    )
    fp16 = not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    patience = 0
    if args.resume_from_checkpoint:
        state = torch.load(
            Path(args.resume_from_checkpoint) / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        patience = int(state.get("patience", 0))

    if is_main:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = vars(args).copy()
        run_config.update(
            {
                "world_size": world_size,
                "effective_global_batch_size": (
                    args.per_device_train_batch_size
                    * world_size
                    * args.gradient_accumulation_steps
                ),
                "learning_rate_resolved": learning_rate,
                "train_examples": len(train_dataset),
                "validation_examples": len(validation_dataset),
                "parameters": unwrapped.trainable_parameter_summary(),
            }
        )
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_cls = 0.0
        running_tr = 0.0
        for batch_index, batch in enumerate(train_loader):
            tensors = move_batch(batch, device)
            group_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps, len(train_loader) - group_start
            )
            should_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            sync_context = (
                nullcontext()
                if should_update or not distributed
                else model.no_sync()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=fp16, dtype=torch.float16):
                    output = model(**tensors)
                    loss = output["loss"] / group_size
                scaler.scale(loss).backward()
            running_loss += float(output["loss"].detach())
            running_cls += float(output["classification_loss"])
            running_tr += float(output["transition_loss"])
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_stats = torch.tensor(
            [running_loss, running_cls, running_tr, float(len(train_loader))],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)

        local_validation = evaluate_local(unwrapped, validation_loader, device, fp16)
        if distributed:
            gathered_validation: Optional[List[Dict[str, Any]]] = (
                [None] * world_size if is_main else None  # type: ignore[list-item]
            )
            dist.gather_object(local_validation, gathered_validation, dst=0)
        else:
            gathered_validation = [local_validation]

        if is_main:
            assert gathered_validation is not None
            validation = merge_evaluation_parts(gathered_validation)
            validation_loss = validation["loss"]
            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                patience = 0
                unwrapped.save_artifacts(Path(args.output_dir) / "best", tokenizer)
            else:
                patience += 1
            averages = (train_stats[:3] / train_stats[3]).tolist()
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_loss": averages[0],
                "train_classification_loss": averages[1],
                "train_transition_loss": averages[2],
                "validation": validation,
                "best_validation_loss": best_validation_loss,
                "early_stopping_patience": patience,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with (Path(args.output_dir) / "metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            last_dir = Path(args.output_dir) / "last"
            unwrapped.save_artifacts(last_dir, tokenizer)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_validation_loss": best_validation_loss,
                    "patience": patience,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                last_dir / "trainer_state.pt",
            )
            should_stop = patience >= args.early_stopping_patience
        else:
            should_stop = False
        stop_tensor = torch.tensor(int(should_stop), device=device)
        if distributed:
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
        if bool(stop_tensor.item()):
            break

    if distributed:
        dist.destroy_process_group()
    return 0


def _build_model(args: argparse.Namespace) -> SelectBidirLM:
    if args.resume_from_checkpoint:
        return SelectBidirLM.from_checkpoint(
            args.resume_from_checkpoint,
            base_model_name_or_path=args.model_name_or_path,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            attention_implementation=args.attention_implementation,
        )
    return SelectBidirLM(
        args.model_name_or_path,
        finetuning_mode=args.finetuning_mode,
        dropout=args.dropout,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[
            item.strip() for item in args.lora_target_modules.split(",") if item.strip()
        ],
        gradient_checkpointing=not args.no_gradient_checkpointing,
        attention_implementation=args.attention_implementation,
    )


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "epochs",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
        raise ValueError("--min-learning-rate-ratio must be in [0, 1]")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")


@torch.no_grad()
def evaluate_local(
    model: SelectBidirLM,
    data_loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Dict[str, Any]:
    model.eval()
    total_loss = total_cls = total_tr = 0.0
    predicted_sequences: List[List[int]] = []
    gold_sequences: List[List[int]] = []
    for batch in data_loader:
        tensors = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=fp16, dtype=torch.float16):
            output = model(**tensors)
        total_loss += float(output["loss"])
        total_cls += float(output["classification_loss"])
        total_tr += float(output["transition_loss"])
        valid_mask = (tensors["labels"] != -100) & tensors["attention_mask"].bool()
        decoded = viterbi_decode_batch(
            output["classification_logits"], output["transition_logits"], valid_mask
        )
        for prediction, gold in zip(decoded.cpu(), tensors["labels"].cpu()):
            mask = gold != -100
            predicted_sequences.append(prediction[mask].tolist())
            gold_sequences.append(gold[mask].tolist())
    return {
        "loss_sum": total_loss,
        "classification_loss_sum": total_cls,
        "transition_loss_sum": total_tr,
        "batches": len(data_loader),
        "predictions": predicted_sequences,
        "labels": gold_sequences,
    }


def merge_evaluation_parts(parts: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    batches = sum(int(part["batches"]) for part in parts)
    if batches == 0:
        raise ValueError("validation dataset produced no batches")
    predictions = [
        sequence for part in parts for sequence in part["predictions"]
    ]
    labels = [sequence for part in parts for sequence in part["labels"]]
    metrics = compute_bio_metrics_from_sequences(predictions, labels)
    metrics.update(
        {
            "loss": sum(float(part["loss_sum"]) for part in parts) / batches,
            "classification_loss": sum(
                float(part["classification_loss_sum"]) for part in parts
            )
            / batches,
            "transition_loss": sum(
                float(part["transition_loss_sum"]) for part in parts
            )
            / batches,
        }
    )
    return metrics


def setup_distributed() -> tuple[bool, int, int, int]:
    if not torch.cuda.is_available():
        return False, 0, 0, 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(hours=2)
        )
    return distributed, rank, local_rank, world_size


def unwrap(model: torch.nn.Module) -> SelectBidirLM:
    return model.module if isinstance(model, DistributedDataParallel) else model


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lr_lambda(total_steps: int, warmup_steps: int, min_ratio: float):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return schedule


class DistributedEvalSampler(Sampler[int]):
    """Shard validation data without DistributedSampler's duplicate padding."""

    def __init__(self, dataset: BioJsonlDataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


if __name__ == "__main__":
    raise SystemExit(main())
