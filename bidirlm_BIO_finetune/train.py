from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .constants import (
    ATTENTION_MASK_KEY,
    CLASSIFICATION_LOGITS_KEY,
    CLASSIFICATION_LOSS_KEY,
    DEFAULT_ATTENTION_IMPLEMENTATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
    LABELS_KEY,
    LOSS_KEY,
    TRANSITION_LOGITS_KEY,
    TRANSITION_LOSS_KEY,
)
from .data import BioDataCollator, BioJsonlDataset
from .decoding import compute_bio_metrics_from_sequences, viterbi_decode_batch
from .modeling import SelectBidirLM

LOGGER = logging.getLogger(__name__)
MAIN_PROCESS_RANK = 0
MINIMUM_FP16_COMPUTE_CAPABILITY = 7
DEFAULT_LORA_LEARNING_RATE = 1e-4
DEFAULT_LEARNING_RATE = 1e-5
CHECKPOINT_PREFIX = "checkpoint-step-"
TRAINER_STATE_FILE = "trainer_state.pt"
RUN_CONFIG_FILE = "run_config.json"
METRICS_FILE = "metrics.jsonl"
DISTRIBUTED_TIMEOUT_HOURS = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_EPOCHS = 8
DEFAULT_MIN_LEARNING_RATE_RATIO = 0.1
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_DROPOUT = 0.1
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"
DEFAULT_NUM_WORKERS = 2
DEFAULT_RANDOM_SEED = 42
DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_SAVE_STEPS = 100


@dataclass
class TrainingProgress:
    start_epoch: int = 0
    resume_batch_index: int = 0
    resume_running_stats: Tuple[float, float, float, int] = (0.0, 0.0, 0.0, 0)
    global_step: int = 0
    best_validation_loss: float = float("inf")
    patience: int = 0


@dataclass
class RunningStats:
    loss: float = 0.0
    classification_loss: float = 0.0
    transition_loss: float = 0.0
    batches: int = 0

    @classmethod
    def from_tuple(cls, values: Tuple[float, float, float, int]) -> "RunningStats":
        return cls(*values)

    def update(self, output: Mapping[str, torch.Tensor]) -> None:
        self.loss += float(_require_tensor(output, LOSS_KEY).detach())
        self.classification_loss += float(
            _require_tensor(output, CLASSIFICATION_LOSS_KEY)
        )
        self.transition_loss += float(_require_tensor(output, TRANSITION_LOSS_KEY))
        self.batches += 1

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                self.loss,
                self.classification_loss,
                self.transition_loss,
                float(self.batches),
            ],
            dtype=torch.float64,
            device=device,
        )


@dataclass(frozen=True)
class DataBundle:
    train_dataset: BioJsonlDataset
    validation_dataset: BioJsonlDataset
    train_sampler: DistributedSampler
    train_loader: DataLoader
    validation_loader: DataLoader


@dataclass(frozen=True)
class TrainingComponents:
    model: torch.nn.Module
    unwrapped_model: SelectBidirLM
    optimizer: torch.optim.Optimizer
    scheduler: LambdaLR
    scaler: torch.cuda.amp.GradScaler
    trainable_parameters: Sequence[torch.nn.Parameter]
    learning_rate: float
    fp16: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune BidirLM with SELECT BIO labels"
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--finetuning-mode", choices=("lora", "full", "heads_only"), default="lora"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--per-device-train-batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--per-device-eval-batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--min-learning-rate-ratio",
        type=float,
        default=DEFAULT_MIN_LEARNING_RATE_RATIO,
    )
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--attention-implementation", default=DEFAULT_ATTENTION_IMPLEMENTATION
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help="Save checkpoint-step-N every N optimizer updates; 0 disables it",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="Maximum step checkpoints to retain; 0 keeps all",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    validate_args(args)
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == MAIN_PROCESS_RANK
    device = torch.device(f"cuda:{local_rank}")
    _validate_cuda(device, fp16=not args.no_fp16)
    seed_everything(args.seed + rank)

    try:
        tokenizer = _load_tokenizer(args.model_name_or_path)
        data = _build_data_bundle(
            args,
            tokenizer,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        components = _build_training_components(
            args,
            data.train_loader,
            device=device,
            distributed=distributed,
            local_rank=local_rank,
        )
        progress = _restore_training_progress(
            args.resume_from_checkpoint,
            components,
            world_size=world_size,
        )
        if is_main:
            _write_run_config(
                args,
                data,
                components,
                world_size=world_size,
            )
        _run_training_epochs(
            args=args,
            data=data,
            components=components,
            progress=progress,
            tokenizer=tokenizer,
            device=device,
            distributed=distributed,
            is_main=is_main,
            world_size=world_size,
        )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
    return 0


def _validate_cuda(device: torch.device, *, fp16: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for BidirLM fine-tuning")
    major_capability = torch.cuda.get_device_capability(device)[0]
    if fp16 and major_capability < MINIMUM_FP16_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "FP16 training requires a CUDA GPU with compute capability >= 7.0"
        )


def _load_tokenizer(model_name_or_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install bidirlm_BIO_finetune/requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_data_bundle(
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataBundle:
    train_dataset = BioJsonlDataset(args.train_file, max_length=args.max_length)
    validation_dataset = BioJsonlDataset(
        args.validation_file, max_length=args.max_length
    )
    collator = BioDataCollator(tokenizer.pad_token_id)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False,
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
    return DataBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_sampler=train_sampler,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )


def _build_training_components(
    args: argparse.Namespace,
    train_loader: DataLoader,
    *,
    device: torch.device,
    distributed: bool,
    local_rank: int,
) -> TrainingComponents:
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
        DEFAULT_LORA_LEARNING_RATE
        if args.finetuning_mode == "lora"
        else DEFAULT_LEARNING_RATE
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        make_lr_lambda(total_updates, warmup_updates, args.min_learning_rate_ratio),
    )
    fp16 = not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    return TrainingComponents(
        model=model,
        unwrapped_model=unwrapped,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainable_parameters=trainable_parameters,
        learning_rate=learning_rate,
        fp16=fp16,
    )


def _restore_training_progress(
    checkpoint: Optional[str],
    components: TrainingComponents,
    *,
    world_size: int,
) -> TrainingProgress:
    progress = TrainingProgress()
    if not checkpoint:
        return progress

    state = torch.load(
        Path(checkpoint) / TRAINER_STATE_FILE,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, Mapping):
        raise TypeError("trainer state must be a mapping")
    components.optimizer.load_state_dict(_required_state(state, "optimizer"))
    components.scheduler.load_state_dict(_required_state(state, "scheduler"))
    components.scaler.load_state_dict(_required_state(state, "scaler"))

    saved_world_size = int(state.get("world_size", world_size))
    if state.get("resume_epoch") is not None:
        progress.start_epoch = int(state.get("resume_epoch", 0))
        progress.resume_batch_index = int(state.get("resume_batch_index", 0))
        _validate_resume_world_size(
            progress.resume_batch_index,
            saved_world_size=saved_world_size,
            current_world_size=world_size,
        )
        if world_size == 1 and saved_world_size == 1:
            progress.resume_running_stats = (
                float(state.get("running_loss", 0.0)),
                float(state.get("running_classification_loss", 0.0)),
                float(state.get("running_transition_loss", 0.0)),
                int(state.get("running_batches", 0)),
            )
    else:
        progress.start_epoch = int(state.get("epoch", -1)) + 1
    progress.global_step = int(state.get("global_step", 0))
    progress.best_validation_loss = float(
        state.get("best_validation_loss", float("inf"))
    )
    progress.patience = int(state.get("patience", 0))
    return progress


def _required_state(state: Mapping[str, Any], key: str) -> Any:
    value = state.get(key)
    if value is None:
        raise KeyError(f"trainer state is missing {key!r}")
    return value


def _validate_resume_world_size(
    resume_batch_index: int,
    *,
    saved_world_size: int,
    current_world_size: int,
) -> None:
    if resume_batch_index and saved_world_size != current_world_size:
        raise ValueError(
            "a mid-epoch checkpoint must resume with the same world_size: "
            f"saved={saved_world_size}, current={current_world_size}"
        )


def _write_run_config(
    args: argparse.Namespace,
    data: DataBundle,
    components: TrainingComponents,
    *,
    world_size: int,
) -> None:
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
            "learning_rate_resolved": components.learning_rate,
            "train_examples": len(data.train_dataset),
            "validation_examples": len(data.validation_dataset),
            "parameters": components.unwrapped_model.trainable_parameter_summary(),
        }
    )
    (output_dir / RUN_CONFIG_FILE).write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("run configuration: %s", json.dumps(run_config, ensure_ascii=False))


def _run_training_epochs(
    *,
    args: argparse.Namespace,
    data: DataBundle,
    components: TrainingComponents,
    progress: TrainingProgress,
    tokenizer: Any,
    device: torch.device,
    distributed: bool,
    is_main: bool,
    world_size: int,
) -> None:
    for epoch in range(progress.start_epoch, args.epochs):
        train_stats = _train_one_epoch(
            epoch=epoch,
            args=args,
            data=data,
            components=components,
            progress=progress,
            tokenizer=tokenizer,
            device=device,
            distributed=distributed,
            is_main=is_main,
            world_size=world_size,
        )
        validation_parts = _gather_validation(
            components.unwrapped_model,
            data.validation_loader,
            device=device,
            fp16=components.fp16,
            distributed=distributed,
            is_main=is_main,
            world_size=world_size,
        )
        should_stop = False
        if is_main:
            if validation_parts is None:
                raise RuntimeError("main process did not receive validation results")
            should_stop = _finish_epoch(
                epoch=epoch,
                args=args,
                train_stats=train_stats,
                validation_parts=validation_parts,
                components=components,
                progress=progress,
                tokenizer=tokenizer,
                world_size=world_size,
            )
        if _synchronize_stop(should_stop, device, distributed):
            break


def _train_one_epoch(
    *,
    epoch: int,
    args: argparse.Namespace,
    data: DataBundle,
    components: TrainingComponents,
    progress: TrainingProgress,
    tokenizer: Any,
    device: torch.device,
    distributed: bool,
    is_main: bool,
    world_size: int,
) -> torch.Tensor:
    data.train_sampler.set_epoch(epoch)
    components.model.train()
    components.optimizer.zero_grad(set_to_none=True)
    running = _initial_running_stats(epoch, progress)
    for batch_index, batch in enumerate(data.train_loader):
        if _should_skip_resumed_batch(epoch, batch_index, progress):
            continue
        should_update = _is_optimizer_update(
            batch_index,
            len(data.train_loader),
            args.gradient_accumulation_steps,
        )
        output = _forward_backward_batch(
            batch=batch,
            batch_index=batch_index,
            train_loader_length=len(data.train_loader),
            accumulation_steps=args.gradient_accumulation_steps,
            components=components,
            device=device,
            distributed=distributed,
            should_update=should_update,
        )
        running.update(output)
        if not should_update:
            continue
        _optimizer_step(components, args.max_grad_norm)
        progress.global_step += 1
        _maybe_save_step_checkpoint(
            epoch=epoch,
            batch_index=batch_index,
            args=args,
            data=data,
            components=components,
            progress=progress,
            running=running,
            tokenizer=tokenizer,
            distributed=distributed,
            is_main=is_main,
            world_size=world_size,
        )

    train_stats = running.as_tensor(device)
    if distributed:
        dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
    return train_stats


def _initial_running_stats(epoch: int, progress: TrainingProgress) -> RunningStats:
    if epoch == progress.start_epoch and progress.resume_batch_index:
        return RunningStats.from_tuple(progress.resume_running_stats)
    return RunningStats()


def _should_skip_resumed_batch(
    epoch: int, batch_index: int, progress: TrainingProgress
) -> bool:
    return epoch == progress.start_epoch and batch_index < progress.resume_batch_index


def _is_optimizer_update(
    batch_index: int, train_loader_length: int, accumulation_steps: int
) -> bool:
    return (
        batch_index + 1
    ) % accumulation_steps == 0 or batch_index + 1 == train_loader_length


def _forward_backward_batch(
    *,
    batch: Mapping[str, Any],
    batch_index: int,
    train_loader_length: int,
    accumulation_steps: int,
    components: TrainingComponents,
    device: torch.device,
    distributed: bool,
    should_update: bool,
) -> Mapping[str, torch.Tensor]:
    tensors = move_batch(dict(batch), device)
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    group_size = min(accumulation_steps, train_loader_length - group_start)
    sync_context = (
        nullcontext()
        if should_update or not distributed
        else components.model.no_sync()
    )
    with sync_context:
        with torch.cuda.amp.autocast(
            enabled=components.fp16,
            dtype=torch.float16,
        ):
            output = components.model(**tensors)
            loss = _require_tensor(output, LOSS_KEY) / group_size
        components.scaler.scale(loss).backward()
    return output


def _optimizer_step(components: TrainingComponents, max_grad_norm: float) -> None:
    components.scaler.unscale_(components.optimizer)
    torch.nn.utils.clip_grad_norm_(
        components.trainable_parameters,
        max_grad_norm,
    )
    components.scaler.step(components.optimizer)
    components.scaler.update()
    components.scheduler.step()
    components.optimizer.zero_grad(set_to_none=True)


def _maybe_save_step_checkpoint(
    *,
    epoch: int,
    batch_index: int,
    args: argparse.Namespace,
    data: DataBundle,
    components: TrainingComponents,
    progress: TrainingProgress,
    running: RunningStats,
    tokenizer: Any,
    distributed: bool,
    is_main: bool,
    world_size: int,
) -> None:
    should_save = args.save_steps > 0 and progress.global_step % args.save_steps == 0
    if not should_save:
        return
    if distributed:
        dist.barrier()
    if is_main:
        next_epoch, next_batch_index = next_training_position(
            epoch, batch_index, len(data.train_loader)
        )
        checkpoint_dir = (
            Path(args.output_dir) / f"{CHECKPOINT_PREFIX}{progress.global_step:08d}"
        )
        save_training_checkpoint(
            checkpoint_dir,
            model=components.unwrapped_model,
            tokenizer=tokenizer,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            scaler=components.scaler,
            epoch=epoch,
            global_step=progress.global_step,
            best_validation_loss=progress.best_validation_loss,
            patience=progress.patience,
            resume_epoch=next_epoch,
            resume_batch_index=next_batch_index,
            running_loss=running.loss,
            running_classification_loss=running.classification_loss,
            running_transition_loss=running.transition_loss,
            running_batches=running.batches,
            world_size=world_size,
        )
        prune_step_checkpoints(Path(args.output_dir), args.save_total_limit)
    if distributed:
        dist.barrier()


def _gather_validation(
    model: SelectBidirLM,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    fp16: bool,
    distributed: bool,
    is_main: bool,
    world_size: int,
) -> Optional[List[Dict[str, Any]]]:
    local_validation = evaluate_local(model, validation_loader, device, fp16)
    if not distributed:
        return [local_validation]
    gathered: Optional[List[Dict[str, Any]]] = (
        [None] * world_size if is_main else None  # type: ignore[list-item]
    )
    dist.gather_object(local_validation, gathered, dst=MAIN_PROCESS_RANK)
    return gathered


def _finish_epoch(
    *,
    epoch: int,
    args: argparse.Namespace,
    train_stats: torch.Tensor,
    validation_parts: Sequence[Dict[str, Any]],
    components: TrainingComponents,
    progress: TrainingProgress,
    tokenizer: Any,
    world_size: int,
) -> bool:
    validation = merge_evaluation_parts(validation_parts)
    validation_loss = float(validation.get("loss", float("inf")))
    if validation_loss < progress.best_validation_loss:
        progress.best_validation_loss = validation_loss
        progress.patience = 0
        components.unwrapped_model.save_artifacts(
            Path(args.output_dir) / "best", tokenizer
        )
    else:
        progress.patience += 1

    record = _epoch_record(
        epoch,
        train_stats,
        validation,
        components,
        progress,
    )
    LOGGER.info("epoch metrics: %s", json.dumps(record, ensure_ascii=False))
    with (Path(args.output_dir) / METRICS_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    _save_last_checkpoint(
        epoch=epoch,
        args=args,
        components=components,
        progress=progress,
        tokenizer=tokenizer,
        world_size=world_size,
    )
    return progress.patience >= args.early_stopping_patience


def _epoch_record(
    epoch: int,
    train_stats: torch.Tensor,
    validation: Mapping[str, float],
    components: TrainingComponents,
    progress: TrainingProgress,
) -> Dict[str, Any]:
    if float(train_stats[3]) <= 0:
        raise ValueError("training epoch produced no batches")
    averages = (train_stats[:3] / train_stats[3]).tolist()
    return {
        "epoch": epoch + 1,
        "global_step": progress.global_step,
        "learning_rate": components.scheduler.get_last_lr()[0],
        "train_loss": averages[0],
        "train_classification_loss": averages[1],
        "train_transition_loss": averages[2],
        "validation": dict(validation),
        "best_validation_loss": progress.best_validation_loss,
        "early_stopping_patience": progress.patience,
    }


def _save_last_checkpoint(
    *,
    epoch: int,
    args: argparse.Namespace,
    components: TrainingComponents,
    progress: TrainingProgress,
    tokenizer: Any,
    world_size: int,
) -> None:
    save_training_checkpoint(
        Path(args.output_dir) / "last",
        model=components.unwrapped_model,
        tokenizer=tokenizer,
        optimizer=components.optimizer,
        scheduler=components.scheduler,
        scaler=components.scaler,
        epoch=epoch,
        global_step=progress.global_step,
        best_validation_loss=progress.best_validation_loss,
        patience=progress.patience,
        resume_epoch=epoch + 1,
        resume_batch_index=0,
        running_loss=0.0,
        running_classification_loss=0.0,
        running_transition_loss=0.0,
        running_batches=0,
        world_size=world_size,
    )


def _synchronize_stop(
    should_stop: bool, device: torch.device, distributed: bool
) -> bool:
    stop_tensor = torch.tensor(int(should_stop), device=device)
    if distributed:
        dist.broadcast(stop_tensor, src=MAIN_PROCESS_RANK)
        dist.barrier()
    return bool(stop_tensor.item())


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
    if args.save_steps < 0:
        raise ValueError("--save-steps must be non-negative")
    if args.save_total_limit < 0:
        raise ValueError("--save-total-limit must be non-negative")


def next_training_position(
    epoch: int, batch_index: int, batches_in_epoch: int
) -> tuple[int, int]:
    if batch_index + 1 >= batches_in_epoch:
        return epoch + 1, 0
    return epoch, batch_index + 1


def save_training_checkpoint(
    checkpoint_dir: Path,
    *,
    model: SelectBidirLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    patience: int,
    resume_epoch: int,
    resume_batch_index: int,
    running_loss: float,
    running_classification_loss: float,
    running_transition_loss: float,
    running_batches: int,
    world_size: int,
) -> None:
    model.save_artifacts(checkpoint_dir, tokenizer)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "patience": patience,
            "resume_epoch": resume_epoch,
            "resume_batch_index": resume_batch_index,
            "running_loss": running_loss,
            "running_classification_loss": running_classification_loss,
            "running_transition_loss": running_transition_loss,
            "running_batches": running_batches,
            "world_size": world_size,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_dir / TRAINER_STATE_FILE,
    )


def prune_step_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit == 0:
        return
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob(f"{CHECKPOINT_PREFIX}*")
            if path.is_dir() and path.name.removeprefix(CHECKPOINT_PREFIX).isdigit()
        ),
        key=lambda path: int(path.name.removeprefix(CHECKPOINT_PREFIX)),
    )
    for checkpoint in checkpoints[:-save_total_limit]:
        shutil.rmtree(checkpoint)


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
        total_loss += float(_require_tensor(output, LOSS_KEY))
        total_cls += float(_require_tensor(output, CLASSIFICATION_LOSS_KEY))
        total_tr += float(_require_tensor(output, TRANSITION_LOSS_KEY))
        labels = _require_tensor(tensors, LABELS_KEY)
        attention_mask = _require_tensor(tensors, ATTENTION_MASK_KEY)
        valid_mask = (labels != IGNORE_INDEX) & attention_mask.bool()
        decoded = viterbi_decode_batch(
            _require_tensor(output, CLASSIFICATION_LOGITS_KEY),
            _require_tensor(output, TRANSITION_LOGITS_KEY),
            valid_mask,
        )
        for prediction, gold in zip(decoded.cpu(), labels.cpu(), strict=True):
            mask = gold != IGNORE_INDEX
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
    batches = sum(int(part.get("batches", 0)) for part in parts)
    if batches == 0:
        raise ValueError("validation dataset produced no batches")
    predictions = [
        sequence for part in parts for sequence in _require_list(part, "predictions")
    ]
    labels = [sequence for part in parts for sequence in _require_list(part, "labels")]
    metrics = compute_bio_metrics_from_sequences(predictions, labels)
    metrics.update(
        {
            "loss": sum(float(part.get("loss_sum", 0.0)) for part in parts) / batches,
            "classification_loss": sum(
                float(part.get("classification_loss_sum", 0.0)) for part in parts
            )
            / batches,
            "transition_loss": sum(
                float(part.get("transition_loss_sum", 0.0)) for part in parts
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
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=DISTRIBUTED_TIMEOUT_HOURS),
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


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"required tensor {key!r} is missing")
    return value


def _require_list(mapping: Mapping[str, Any], key: str) -> List[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise KeyError(f"required list {key!r} is missing")
    return value


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
