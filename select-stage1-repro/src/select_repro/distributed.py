from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from .data import MLMCollator, PackedTokenDataset, load_dataset_metadata
from .modeling_bidirectional_qwen3 import Qwen3ForBidirectionalMaskedLM
from .training import cosine_learning_rate, validate_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Platform = Literal["cuda", "npu"]
_ENV_PATTERN = re.compile(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([^}]+)\})")


@dataclass(frozen=True)
class DistributedContext:
    platform: Platform
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    device_name: str
    process_group_initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class RankDataPlan:
    total_sequences: int
    usable_sequences: int
    dropped_sequences: int
    sequences_per_rank: int
    micro_batches_per_rank: int
    optimizer_steps: int
    effective_train_tokens: int
    effective_executed_tokens: int


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    references = _ENV_PATTERN.findall(value)
    unavailable = [
        first or braced
        for first, braced in references
        if not os.environ.get(first or braced)
    ]
    if unavailable:
        raise ValueError(
            "configuration references unset environment variables: "
            + ", ".join(sorted(set(unavailable)))
        )
    expanded = os.path.expanduser(os.path.expandvars(value))
    missing = [first or braced for first, braced in _ENV_PATTERN.findall(expanded)]
    if missing:
        raise ValueError(
            "configuration references unset environment variables: "
            + ", ".join(sorted(set(missing)))
        )
    return expanded


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _tokenizer_sha256(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        serialized = backend.to_str()
    else:
        serialized = json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_distributed_config(
    config_path: str | Path,
    *,
    platform: Platform,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw_config = json.load(stream)
    config = _expand_environment(raw_config)

    resume_override = os.environ.get("SELECT_RESUME_FROM")
    if resume_override:
        config["resume_from"] = os.path.expanduser(resume_override)
    max_steps_override = os.environ.get("SELECT_MAX_STEPS")
    if max_steps_override:
        config["max_optimizer_steps"] = int(max_steps_override)

    aliases = {
        "precision": "fp16" if platform == "cuda" else "bf16",
        "distributed_backend": "nccl" if platform == "cuda" else "hccl",
        "distributed_strategy": "ddp",
        "use_grad_scaler": platform == "cuda",
        "parameter_dtype": "float32",
        "fused_optimizer": False,
        "max_optimizer_steps": None,
        "resume_from": None,
        "save_every_steps": 0,
        "process_group_timeout_minutes": 60,
        "max_consecutive_overflows": 8,
    }
    for key, default in aliases.items():
        config.setdefault(key, default)

    # Keep the original single-GPU validator usable without duplicating its rules.
    config["bf16"] = config["precision"] == "bf16"
    validate_training_config(config)
    validate_distributed_config(config, platform=platform)

    for key in ("base_model", "data_dir", "output_dir"):
        config[key] = str(_resolve_project_path(config[key]))
    if config.get("resume_from"):
        config["resume_from"] = str(_resolve_project_path(config["resume_from"]))
    config["config_path"] = str(path)
    return config


def validate_distributed_config(
    config: dict[str, Any],
    *,
    platform: Platform,
) -> None:
    expected_backend = "nccl" if platform == "cuda" else "hccl"
    expected_precision = "fp16" if platform == "cuda" else "bf16"
    if config.get("platform", platform) != platform:
        raise ValueError(
            f"config platform {config.get('platform')!r} does not match {platform!r}"
        )
    if config["distributed_backend"] != expected_backend:
        raise ValueError(
            f"{platform} requires distributed_backend={expected_backend!r}"
        )
    if config["distributed_strategy"] != "ddp":
        raise ValueError("this trainer currently supports distributed_strategy='ddp'")
    if config["precision"] != expected_precision:
        raise ValueError(f"{platform} requires precision={expected_precision!r}")
    if bool(config["use_grad_scaler"]) != (platform == "cuda"):
        raise ValueError(
            "V100/CUDA requires GradScaler; Ascend BF16 must leave it disabled"
        )
    if config.get("parameter_dtype", "float32") != "float32":
        raise ValueError(
            "distributed recipes keep FP32 parameters and optimizer state for stability"
        )
    if int(config["save_every_steps"]) < 0:
        raise ValueError("save_every_steps cannot be negative")
    if int(config["process_group_timeout_minutes"]) <= 0:
        raise ValueError("process_group_timeout_minutes must be positive")
    if int(config["max_consecutive_overflows"]) <= 0:
        raise ValueError("max_consecutive_overflows must be positive")
    max_steps = config.get("max_optimizer_steps")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError("max_optimizer_steps must be null or a positive integer")


def build_rank_data_plan(
    *,
    train_tokens: int,
    sequence_length: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    max_optimizer_steps: int | None = None,
) -> RankDataPlan:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if train_tokens % sequence_length:
        raise ValueError("train_tokens must be divisible by sequence_length")
    total_sequences = train_tokens // sequence_length
    sequence_multiple = world_size * micro_batch_size
    usable_sequences = total_sequences - total_sequences % sequence_multiple
    if usable_sequences == 0:
        raise ValueError(
            "not enough sequences for one full distributed micro-batch; "
            "reduce world_size or micro_batch_size"
        )
    sequences_per_rank = usable_sequences // world_size
    micro_batches = sequences_per_rank // micro_batch_size
    natural_steps = math.ceil(micro_batches / gradient_accumulation_steps)
    optimizer_steps = (
        natural_steps
        if max_optimizer_steps is None
        else min(natural_steps, max_optimizer_steps)
    )
    executed_micro_batches = min(
        micro_batches, optimizer_steps * gradient_accumulation_steps
    )
    return RankDataPlan(
        total_sequences=total_sequences,
        usable_sequences=usable_sequences,
        dropped_sequences=total_sequences - usable_sequences,
        sequences_per_rank=sequences_per_rank,
        micro_batches_per_rank=micro_batches,
        optimizer_steps=optimizer_steps,
        effective_train_tokens=usable_sequences * sequence_length,
        effective_executed_tokens=(
            executed_micro_batches
            * micro_batch_size
            * sequence_length
            * world_size
        ),
    )


def make_rank_sequence_order(
    *,
    total_sequences: int,
    usable_sequences: int,
    rank: int,
    world_size: int,
    seed: int,
    shuffle: bool,
) -> torch.Tensor:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be inside world_size")
    if not 0 < usable_sequences <= total_sequences:
        raise ValueError("usable_sequences must be in (0, total_sequences]")
    if usable_sequences % world_size:
        raise ValueError("usable_sequences must be divisible by world_size")
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        global_order = torch.randperm(total_sequences, generator=generator)
    else:
        global_order = torch.arange(total_sequences)
    # Trim globally and then stride by rank: equal work and no duplicated examples.
    return global_order[:usable_sequences][rank::world_size].clone()


def compare_resume_invariants(
    saved: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {
        key: {"checkpoint": saved.get(key), "current": value}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    unexpected = sorted(set(saved) - set(expected))
    if unexpected:
        mismatches["unexpected_checkpoint_keys"] = unexpected
    return mismatches


def _initialize_distributed(
    platform: Platform,
    config: dict[str, Any],
) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid RANK/WORLD_SIZE torchrun environment")

    if platform == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run this recipe on NVIDIA V100 nodes")
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA devices are visible"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
    else:
        try:
            import torch_npu  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "torch_npu is unavailable; use an official CANN/TorchNPU server image"
            ) from exc
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable to TorchNPU")
        device_count = npu.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} NPUs are visible"
            )
        npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
        get_name = getattr(npu, "get_device_name", None)
        device_name = str(get_name(local_rank)) if get_name else f"Ascend NPU {local_rank}"

    initialized = False
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
        backend = str(config["distributed_backend"])
        if not dist.is_backend_available(backend):
            raise RuntimeError(f"torch.distributed backend {backend!r} is unavailable")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=int(config["process_group_timeout_minutes"])),
        )
        initialized = True

    return DistributedContext(
        platform=platform,
        backend=str(config["distributed_backend"]),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        device_name=device_name,
        process_group_initialized=initialized,
    )


def _barrier(context: DistributedContext) -> None:
    if context.process_group_initialized:
        dist.barrier()


def _raise_if_any_rank_failed(
    context: DistributedContext,
    *,
    phase: str,
    local_error: str | None,
) -> None:
    """Turn rank-local I/O failures into the same exception on every rank."""
    if not context.process_group_initialized:
        if local_error:
            raise RuntimeError(f"{phase} failed: {local_error}")
        return
    failed_rank = torch.tensor(
        context.rank + 1 if local_error else 0,
        dtype=torch.int32,
        device=context.device,
    )
    dist.all_reduce(failed_rank, op=dist.ReduceOp.MAX)
    if failed_rank.item():
        source_rank = int(failed_rank.item()) - 1
        detail = local_error or f"see rank {source_rank} stderr for the local exception"
        raise RuntimeError(
            f"{phase} failed on rank {source_rank}: {detail}"
        )


def _set_seed(seed: int, context: DistributedContext) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if context.platform == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        torch.npu.manual_seed_all(seed)  # type: ignore[attr-defined]


def _device_rng_state(context: DistributedContext) -> torch.Tensor:
    if context.platform == "cuda":
        return torch.cuda.get_rng_state(context.device)
    return torch.npu.get_rng_state(context.device)  # type: ignore[attr-defined]


def _set_device_rng_state(context: DistributedContext, state: torch.Tensor) -> None:
    if context.platform == "cuda":
        torch.cuda.set_rng_state(state, context.device)
    else:
        torch.npu.set_rng_state(state, context.device)  # type: ignore[attr-defined]


def _create_grad_scaler(context: DistributedContext, enabled: bool) -> Any | None:
    if not enabled:
        return None
    if context.platform != "cuda":
        raise ValueError("GradScaler is only used by the CUDA/FP16 recipe")
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:  # PyTorch 2.3 compatibility.
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(context: DistributedContext, precision: str):
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=context.platform, dtype=dtype, enabled=True)


def _reset_peak_memory(context: DistributedContext) -> None:
    if context.platform == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    else:
        reset = getattr(torch.npu, "reset_peak_memory_stats", None)  # type: ignore[attr-defined]
        if reset:
            reset(context.device)


def _peak_memory_gib(context: DistributedContext) -> tuple[float | None, float | None]:
    namespace = (
        torch.cuda
        if context.platform == "cuda"
        else torch.npu  # type: ignore[attr-defined]
    )
    allocated_fn = getattr(namespace, "max_memory_allocated", None)
    reserved_fn = getattr(namespace, "max_memory_reserved", None)
    allocated = (
        float(allocated_fn(context.device)) / 2**30 if allocated_fn else None
    )
    reserved = float(reserved_fn(context.device)) / 2**30 if reserved_fn else None
    return allocated, reserved


def _build_resume_invariants(
    *,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    tokenizer: Any,
    context: DistributedContext,
    plan: RankDataPlan,
) -> dict[str, Any]:
    return {
        "dataset_token_file_sha256": data_metadata["token_file_sha256"],
        "sequence_length": config["sequence_length"],
        "requested_train_tokens": config["train_tokens"],
        "effective_train_tokens": plan.effective_train_tokens,
        "micro_batch_size": config["micro_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "mlm_probability": config["mlm_probability"],
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "tokenizer_size": len(tokenizer),
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
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
        "precision": config["precision"],
        "parameter_dtype": config["parameter_dtype"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "fused_optimizer": config["fused_optimizer"],
        "max_optimizer_steps": config.get("max_optimizer_steps"),
        "max_consecutive_overflows": config["max_consecutive_overflows"],
        "platform": context.platform,
        "backend": context.backend,
        "world_size": context.world_size,
    }


def _validate_resume_checkpoint(
    checkpoint_dir: Path,
    expected: dict[str, Any],
) -> None:
    manifest_path = checkpoint_dir / "stage1_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"resume checkpoint has no manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    saved = manifest.get("resume_invariants")
    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint has no distributed resume invariants")
    mismatches = compare_resume_invariants(saved, expected)
    if mismatches:
        raise ValueError(
            "resume invariant mismatch:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _save_distributed_checkpoint(
    checkpoint_dir: Path,
    *,
    ddp_model: DistributedDataParallel | Qwen3ForBidirectionalMaskedLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    collator: MLMCollator,
    context: DistributedContext,
    plan: RankDataPlan,
    global_step: int,
    batch_position: int,
    tokens_consumed: int,
) -> None:
    temporary_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.incomplete")
    preparation_error: str | None = None
    if context.is_main:
        try:
            checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_dir.exists():
                raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
            if temporary_dir.exists():
                raise FileExistsError(
                    f"incomplete checkpoint already exists: {temporary_dir}; "
                    "inspect it before retrying"
                )
            temporary_dir.mkdir()
        except Exception as exc:  # Broadcast the failure so peers do not wait forever.
            preparation_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase="checkpoint directory preparation",
        local_error=preparation_error,
    )

    rank_state = {
        "format_version": 1,
        "rank": context.rank,
        "world_size": context.world_size,
        "batch_position": batch_position,
        "collator": collator.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "device_rng_state": _device_rng_state(context),
    }
    rank_state_error: str | None = None
    try:
        torch.save(
            rank_state,
            temporary_dir / f"rank-{context.rank:05d}-state.pt",
        )
    except Exception as exc:
        rank_state_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase="rank-local checkpoint state write",
        local_error=rank_state_error,
    )

    main_state_error: str | None = None
    if context.is_main:
        try:
            model = (
                ddp_model.module
                if isinstance(ddp_model, DistributedDataParallel)
                else ddp_model
            )
            model.save_pretrained(
                temporary_dir,
                safe_serialization=True,
                max_shard_size="4GB",
            )
            tokenizer.save_pretrained(temporary_dir)
            torch.save(
                {
                    "format_version": 1,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "global_step": global_step,
                    "tokens_consumed": tokens_consumed,
                },
                temporary_dir / "trainer_state.pt",
            )
            manifest = {
                "format_version": 2,
                "paper": "SELECting over Tokens (ACL 2026), stage-1 scaled reproduction",
                "objective": "masked_language_modeling",
                "bidirectional_attention": True,
                "attention_implementation": "sdpa",
                "distributed_strategy": "ddp",
                "platform": context.platform,
                "backend": context.backend,
                "world_size": context.world_size,
                "sequence_length": config["sequence_length"],
                "requested_train_tokens": config["train_tokens"],
                "effective_train_tokens": plan.effective_train_tokens,
                "tokens_consumed": tokens_consumed,
                "global_step": global_step,
                "optimizer_steps": plan.optimizer_steps,
                "precision": config["precision"],
                "parameter_dtype": config["parameter_dtype"],
                "mask_token": tokenizer.mask_token,
                "mask_token_id": tokenizer.mask_token_id,
                "dataset_token_file_sha256": data_metadata["token_file_sha256"],
                "resume_invariants": _build_resume_invariants(
                    config=config,
                    data_metadata=data_metadata,
                    tokenizer=tokenizer,
                    context=context,
                    plan=plan,
                ),
                "reproduction_choices_not_disclosed_by_paper": {
                    "masking_policy": "15% targets with BERT 80/10/10 replacement",
                    "optimizer": "AdamW",
                    "adam_beta1": config["adam_beta1"],
                    "adam_beta2": config["adam_beta2"],
                    "weight_decay": config["weight_decay"],
                    "warmup_steps": config["warmup_steps"],
                },
            }
            _write_json(temporary_dir / "stage1_manifest.json", manifest)
        except Exception as exc:
            main_state_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase="rank-0 model and optimizer checkpoint write",
        local_error=main_state_error,
    )

    rename_error: str | None = None
    if context.is_main:
        try:
            temporary_dir.replace(checkpoint_dir)
        except Exception as exc:
            rename_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase="checkpoint commit",
        local_error=rename_error,
    )


def _load_rank_state(
    checkpoint_dir: Path,
    *,
    context: DistributedContext,
    collator: MLMCollator,
) -> int:
    rank_path = checkpoint_dir / f"rank-{context.rank:05d}-state.pt"
    if not rank_path.exists():
        raise FileNotFoundError(
            f"resume checkpoint has no state for rank {context.rank}: {rank_path}"
        )
    state = torch.load(rank_path, map_location="cpu", weights_only=False)
    if int(state["world_size"]) != context.world_size:
        raise ValueError("checkpoint world size differs from current torchrun world size")
    collator.load_state_dict(state["collator"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    _set_device_rng_state(context, state["device_rng_state"])
    return int(state["batch_position"])


def _reduce_step_metrics(
    *,
    context: DistributedContext,
    loss_sum: float,
    loss_count: int,
    elapsed_seconds: float,
) -> tuple[float, float]:
    if not context.process_group_initialized:
        return loss_sum / loss_count, elapsed_seconds
    loss_stats = torch.tensor(
        [loss_sum, float(loss_count)],
        dtype=torch.float32,
        device=context.device,
    )
    elapsed = torch.tensor(
        elapsed_seconds,
        dtype=torch.float32,
        device=context.device,
    )
    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float((loss_stats[0] / loss_stats[1]).item()), float(elapsed.item())


def _all_ranks_finite(
    context: DistributedContext,
    value: torch.Tensor,
) -> bool:
    finite = torch.isfinite(value).to(dtype=torch.int32, device=context.device)
    if context.process_group_initialized:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def _verify_equal_fingerprint(
    context: DistributedContext,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    if not context.process_group_initialized:
        return
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFF_FFFF
        for offset in range(0, 16, 4)
    ]
    minimum = torch.tensor(words, dtype=torch.int32, device=context.device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise ValueError(
            f"{label} differs across ranks; deploy identical configs, tokenizer, "
            "and packed-token metadata on every node"
        )


def _verify_equal_rank_position(
    context: DistributedContext,
    batch_position: int,
) -> None:
    if not context.process_group_initialized:
        return
    minimum = torch.tensor(batch_position, dtype=torch.int32, device=context.device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if minimum.item() != maximum.item():
        raise ValueError("rank-local checkpoint batch positions do not match")


def _verify_shared_directory(
    directory: Path,
    *,
    context: DistributedContext,
    label: str,
) -> None:
    if not context.process_group_initialized:
        return

    marker_value = torch.tensor(
        time.time_ns() % 2_000_000_000 if context.is_main else 0,
        dtype=torch.int32,
        device=context.device,
    )
    dist.broadcast(marker_value, src=0)
    marker_token = str(int(marker_value.item()))
    marker_path = directory / f".select-shared-filesystem-{marker_token}"

    marker_write_error: str | None = None
    if context.is_main:
        try:
            marker_path.write_text(marker_token, encoding="utf-8")
        except Exception as exc:
            marker_write_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase=f"{label} shared-filesystem marker write",
        local_error=marker_write_error,
    )

    visibility_error: str | None = None
    try:
        if marker_path.read_text(encoding="utf-8") != marker_token:
            raise RuntimeError("shared marker content differs")
    except Exception as exc:
        visibility_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase=f"{label} shared-filesystem visibility verification",
        local_error=visibility_error,
    )

    marker_cleanup_error: str | None = None
    if context.is_main:
        try:
            marker_path.unlink()
        except Exception as exc:
            marker_cleanup_error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase=f"{label} shared-filesystem marker cleanup",
        local_error=marker_cleanup_error,
    )


def _prepare_output_directory(
    output_dir: Path,
    *,
    resume_from: Path | None,
    context: DistributedContext,
) -> None:
    error: str | None = None
    if context.is_main:
        try:
            if output_dir.exists() and any(output_dir.iterdir()) and resume_from is None:
                raise FileExistsError(
                    f"output directory is not empty: {output_dir}; "
                    "set SELECT_RESUME_FROM or choose a new directory"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(
        context,
        phase="output directory preparation",
        local_error=error,
    )
    _verify_shared_directory(
        output_dir,
        context=context,
        label="SELECT_OUTPUT_DIR",
    )


def run_training(*, platform: Platform, config_path: str | Path) -> None:
    """Run SELECT stage-1 under torchrun on CUDA/NCCL or Ascend/HCCL."""
    config = load_distributed_config(config_path, platform=platform)
    context: DistributedContext | None = None
    dataset: PackedTokenDataset | None = None
    try:
        context = _initialize_distributed(platform, config)
        resume_from = Path(config["resume_from"]) if config.get("resume_from") else None
        output_dir = Path(config["output_dir"])
        _prepare_output_directory(
            output_dir,
            resume_from=resume_from,
            context=context,
        )
        if resume_from is not None:
            resume_probe_error: str | None = None
            try:
                if not resume_from.is_dir():
                    raise FileNotFoundError(
                        f"resume checkpoint directory not found: {resume_from}"
                    )
                for required_name in (
                    "stage1_manifest.json",
                    "trainer_state.pt",
                    "config.json",
                ):
                    if not (resume_from / required_name).is_file():
                        raise FileNotFoundError(
                            f"resume checkpoint is missing {required_name}: {resume_from}"
                        )
            except Exception as exc:
                resume_probe_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                context,
                phase="resume checkpoint path validation",
                local_error=resume_probe_error,
            )
            _verify_shared_directory(
                resume_from,
                context=context,
                label="SELECT_RESUME_FROM",
            )

            resume_manifest: dict[str, Any] | None = None
            resume_manifest_error: str | None = None
            try:
                with (resume_from / "stage1_manifest.json").open(
                    "r", encoding="utf-8"
                ) as stream:
                    loaded_manifest = json.load(stream)
                if not isinstance(loaded_manifest, dict):
                    raise ValueError("stage1_manifest.json must contain an object")
                resume_manifest = loaded_manifest
            except Exception as exc:
                resume_manifest_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                context,
                phase="resume manifest read",
                local_error=resume_manifest_error,
            )
            assert resume_manifest is not None
            _verify_equal_fingerprint(
                context,
                resume_manifest,
                label="resume manifest",
            )

        if platform == "cuda":
            # V100 has no TF32 path. Explicit settings also keep behavior predictable
            # if this recipe is smoke-tested on a newer CUDA GPU.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # All ranks load identical pretrained weights, then use distinct RNG streams
        # for dropout and masking after DDP synchronizes the parameters.
        _set_seed(int(config["seed"]), context)
        data_dir = Path(config["data_dir"])
        data_metadata: dict[str, Any] | None = None
        plan: RankDataPlan | None = None
        data_error: str | None = None
        try:
            loaded_metadata = load_dataset_metadata(data_dir)
            if int(loaded_metadata["sequence_length"]) != int(
                config["sequence_length"]
            ):
                raise ValueError("data and training sequence lengths do not match")
            if int(loaded_metadata["train_tokens"]) < int(config["train_tokens"]):
                raise ValueError(
                    "prepared dataset contains fewer train tokens than requested; "
                    "build the 10B packed dataset before launching this recipe"
                )
            built_plan = build_rank_data_plan(
                train_tokens=int(config["train_tokens"]),
                sequence_length=int(config["sequence_length"]),
                micro_batch_size=int(config["micro_batch_size"]),
                gradient_accumulation_steps=int(
                    config["gradient_accumulation_steps"]
                ),
                world_size=context.world_size,
                max_optimizer_steps=config.get("max_optimizer_steps"),
            )
            if int(config["warmup_steps"]) >= built_plan.optimizer_steps:
                raise ValueError("warmup_steps must be smaller than optimizer_steps")
            loaded_dataset = PackedTokenDataset(
                data_dir / loaded_metadata["token_file"],
                sequence_length=int(config["sequence_length"]),
                offset_tokens=int(loaded_metadata["train_offset_tokens"]),
                num_tokens=int(config["train_tokens"]),
            )
            data_metadata = loaded_metadata
            plan = built_plan
            dataset = loaded_dataset
        except Exception as exc:
            data_error = f"{type(exc).__name__}: {exc}"
        _raise_if_any_rank_failed(
            context,
            phase="packed training data initialization",
            local_error=data_error,
        )
        assert data_metadata is not None
        assert plan is not None
        assert dataset is not None
        _verify_equal_fingerprint(
            context,
            {
                "token_file_sha256": data_metadata["token_file_sha256"],
                "sequence_length": data_metadata["sequence_length"],
                "requested_train_tokens": config["train_tokens"],
                "effective_train_tokens": plan.effective_train_tokens,
            },
            label="packed training data identity",
        )
        rank_order = make_rank_sequence_order(
            total_sequences=len(dataset),
            usable_sequences=plan.usable_sequences,
            rank=context.rank,
            world_size=context.world_size,
            seed=int(config["seed"]),
            shuffle=bool(config["shuffle"]),
        )

        model_source = resume_from if resume_from is not None else Path(config["base_model"])
        tokenizer: Any | None = None
        tokenizer_error: str | None = None
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
            if loaded_tokenizer.mask_token is None:
                loaded_tokenizer.add_special_tokens(
                    {"mask_token": config["mask_token"]}
                )
            elif loaded_tokenizer.mask_token != config["mask_token"]:
                raise ValueError(
                    f"checkpoint mask token {loaded_tokenizer.mask_token!r} "
                    "does not match config"
                )
            tokenizer = loaded_tokenizer
        except Exception as exc:
            tokenizer_error = f"{type(exc).__name__}: {exc}"
        _raise_if_any_rank_failed(
            context,
            phase="tokenizer initialization",
            local_error=tokenizer_error,
        )
        assert tokenizer is not None

        model: Qwen3ForBidirectionalMaskedLM | None = None
        model_error: str | None = None
        try:
            loaded_model = Qwen3ForBidirectionalMaskedLM.from_pretrained(
                model_source,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )
            embedding_rows = loaded_model.get_input_embeddings().num_embeddings
            if len(tokenizer) > embedding_rows:
                loaded_model.resize_token_embeddings(
                    len(tokenizer), mean_resizing=False
                )
            elif tokenizer.mask_token_id >= embedding_rows:
                raise ValueError("mask token ID is outside the model embedding table")
            loaded_model.config.use_cache = False
            loaded_model.config.bidirectional_attention = True
            loaded_model.config.stage1_objective = "mlm"
            loaded_model.config.mask_token_id = tokenizer.mask_token_id
            if bool(config["gradient_checkpointing"]):
                loaded_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            loaded_model.train().to(context.device)
            model = loaded_model
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
        _raise_if_any_rank_failed(
            context,
            phase="model initialization",
            local_error=model_error,
        )
        assert model is not None

        ddp_model: DistributedDataParallel | Qwen3ForBidirectionalMaskedLM
        if context.process_group_initialized:
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            ddp_model = model

        collator = MLMCollator(
            mask_token_id=int(tokenizer.mask_token_id),
            vocab_size=len(tokenizer),
            special_token_ids=tokenizer.all_special_ids,
            mlm_probability=float(config["mlm_probability"]),
            seed=int(config["seed"]) + 10_000 + context.rank,
        )
        optimizer_kwargs: dict[str, Any] = {
            "lr": float(config["learning_rate"]),
            "betas": (float(config["adam_beta1"]), float(config["adam_beta2"])),
            "eps": float(config["adam_epsilon"]),
            "weight_decay": float(config["weight_decay"]),
        }
        if platform == "cuda" and bool(config["fused_optimizer"]):
            optimizer_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(ddp_model.parameters(), **optimizer_kwargs)
        scaler = _create_grad_scaler(context, bool(config["use_grad_scaler"]))

        expected_invariants = _build_resume_invariants(
            config=config,
            data_metadata=data_metadata,
            tokenizer=tokenizer,
            context=context,
            plan=plan,
        )
        _verify_equal_fingerprint(
            context,
            expected_invariants,
            label="training identity",
        )
        global_step = 0
        batch_position = 0
        tokens_consumed = 0
        if resume_from is not None:
            resume_validation_error: str | None = None
            try:
                _validate_resume_checkpoint(resume_from, expected_invariants)
            except Exception as exc:
                resume_validation_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                context,
                phase="resume invariant validation",
                local_error=resume_validation_error,
            )

            trainer_state: dict[str, Any] | None = None
            trainer_state_error: str | None = None
            try:
                loaded_trainer_state = torch.load(
                    resume_from / "trainer_state.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if not isinstance(loaded_trainer_state, dict):
                    raise ValueError("trainer_state.pt must contain a dictionary")
                optimizer.load_state_dict(loaded_trainer_state["optimizer"])
                if scaler is not None:
                    if loaded_trainer_state.get("scaler") is None:
                        raise ValueError("FP16 resume checkpoint has no GradScaler state")
                    scaler.load_state_dict(loaded_trainer_state["scaler"])
                trainer_state = loaded_trainer_state
            except Exception as exc:
                trainer_state_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                context,
                phase="optimizer and scaler resume",
                local_error=trainer_state_error,
            )
            assert trainer_state is not None
            global_step = int(trainer_state["global_step"])
            tokens_consumed = int(trainer_state["tokens_consumed"])

            loaded_batch_position: int | None = None
            rank_state_error: str | None = None
            try:
                loaded_batch_position = _load_rank_state(
                    resume_from,
                    context=context,
                    collator=collator,
                )
            except Exception as exc:
                rank_state_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                context,
                phase="rank-local resume state",
                local_error=rank_state_error,
            )
            assert loaded_batch_position is not None
            batch_position = loaded_batch_position
            _verify_equal_rank_position(context, batch_position)
            if global_step > plan.optimizer_steps:
                raise ValueError("checkpoint step exceeds configured optimizer schedule")
            if batch_position > plan.micro_batches_per_rank:
                raise ValueError("checkpoint data position exceeds this rank's data plan")
            expected_batch_position = min(
                global_step * int(config["gradient_accumulation_steps"]),
                plan.micro_batches_per_rank,
            )
            if batch_position != expected_batch_position:
                raise ValueError(
                    "checkpoint batch position is inconsistent with global_step"
                )
            expected_tokens = (
                batch_position
                * int(config["micro_batch_size"])
                * int(config["sequence_length"])
                * context.world_size
            )
            if tokens_consumed != expected_tokens:
                raise ValueError(
                    "checkpoint tokens_consumed is inconsistent with rank data position"
                )
        else:
            # Parameters are already identical; rank-specific device RNG avoids
            # correlated dropout while the collator owns an independent CPU RNG.
            _set_seed(int(config["seed"]) + context.rank, context)

        if context.is_main:
            startup = {
                "platform": platform,
                "backend": context.backend,
                "device": context.device_name,
                "world_size": context.world_size,
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "sequence_length": config["sequence_length"],
                "requested_train_tokens": config["train_tokens"],
                "effective_train_tokens": plan.effective_train_tokens,
                "dropped_tail_tokens": plan.dropped_sequences * config["sequence_length"],
                "optimizer_steps": plan.optimizer_steps,
                "effective_executed_tokens": plan.effective_executed_tokens,
                "precision": config["precision"],
                "parameter_dtype": config["parameter_dtype"],
            }
            print(json.dumps(startup, ensure_ascii=False), flush=True)
            _write_json(output_dir / "run_manifest.json", startup | {
                "config": config,
                "resume_invariants": expected_invariants,
            })

        micro_batch_size = int(config["micro_batch_size"])
        accumulation_steps = int(config["gradient_accumulation_steps"])
        sequence_length = int(config["sequence_length"])
        log_path = output_dir / "train_log.jsonl"
        _reset_peak_memory(context)
        run_started = time.perf_counter()
        consecutive_overflows = 0

        while global_step < plan.optimizer_steps and batch_position < plan.micro_batches_per_rank:
            retry_batch_position = batch_position
            retry_collator_state = collator.state_dict() if scaler is not None else None
            retry_device_rng_state = _device_rng_state(context) if scaler is not None else None
            group_size = min(
                accumulation_steps,
                plan.micro_batches_per_rank - batch_position,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            local_step_tokens = 0
            step_started = time.perf_counter()

            for micro_step in range(group_size):
                start = batch_position * micro_batch_size
                indexes = rank_order[start : start + micro_batch_size].tolist()
                examples = [dataset[index] for index in indexes]
                batch = collator(examples)
                input_ids = batch["input_ids"].to(context.device, non_blocking=True)
                labels = batch["labels"].to(context.device, non_blocking=True)
                sync_context = (
                    nullcontext()
                    if micro_step == group_size - 1
                    else ddp_model.no_sync()
                    if isinstance(ddp_model, DistributedDataParallel)
                    else nullcontext()
                )
                with sync_context:
                    with _autocast_context(context, str(config["precision"])):
                        outputs = ddp_model(
                            input_ids=input_ids,
                            labels=labels,
                            use_cache=False,
                            return_full_logits=False,
                        )
                        raw_loss = outputs.loss
                        loss = raw_loss / group_size
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                loss_sum += float(raw_loss.detach())
                batch_position += 1
                local_step_tokens += len(indexes) * sequence_length

            learning_rate = cosine_learning_rate(
                global_step,
                total_steps=plan.optimizer_steps,
                peak_lr=float(config["learning_rate"]),
                min_lr=float(config["min_learning_rate"]),
                warmup_steps=int(config["warmup_steps"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(), float(config["max_grad_norm"])
            )
            if not _all_ranks_finite(context, grad_norm):
                if scaler is None:
                    raise FloatingPointError(
                        "non-finite BF16 gradient norm detected on at least one rank"
                    )
                scaler.update()
                assert retry_collator_state is not None
                assert retry_device_rng_state is not None
                batch_position = retry_batch_position
                collator.load_state_dict(retry_collator_state)
                _set_device_rng_state(context, retry_device_rng_state)
                optimizer.zero_grad(set_to_none=True)
                consecutive_overflows += 1
                if context.is_main:
                    print(
                        json.dumps(
                            {
                                "event": "fp16_overflow_retry",
                                "step": global_step + 1,
                                "consecutive_overflows": consecutive_overflows,
                                "new_grad_scale": float(scaler.get_scale()),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if consecutive_overflows >= int(config["max_consecutive_overflows"]):
                    raise FloatingPointError(
                        "FP16 gradients remained non-finite after "
                        f"{consecutive_overflows} retries"
                    )
                continue
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            consecutive_overflows = 0
            global_step += 1
            global_step_tokens = local_step_tokens * context.world_size
            tokens_consumed += global_step_tokens

            mean_loss, elapsed = _reduce_step_metrics(
                context=context,
                loss_sum=loss_sum,
                loss_count=group_size,
                elapsed_seconds=max(time.perf_counter() - step_started, 1e-9),
            )
            peak_allocated, peak_reserved = _peak_memory_gib(context)
            record = {
                "step": global_step,
                "loss": mean_loss,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "step_tokens": global_step_tokens,
                "tokens_consumed": tokens_consumed,
                "tokens_per_second": global_step_tokens / elapsed,
                "elapsed_seconds": time.perf_counter() - run_started,
                "peak_device_memory_gib": peak_allocated,
                "peak_device_reserved_gib": peak_reserved,
            }
            if scaler is not None:
                record["grad_scale"] = float(scaler.get_scale())
            if context.is_main:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if global_step == 1 or global_step % int(config["log_every_steps"]) == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)

            save_every = int(config["save_every_steps"])
            if save_every and global_step % save_every == 0:
                _save_distributed_checkpoint(
                    output_dir / "checkpoints" / f"step-{global_step:08d}",
                    ddp_model=ddp_model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    data_metadata=data_metadata,
                    collator=collator,
                    context=context,
                    plan=plan,
                    global_step=global_step,
                    batch_position=batch_position,
                    tokens_consumed=tokens_consumed,
                )

        final_dir = output_dir / "final"
        _save_distributed_checkpoint(
            final_dir,
            ddp_model=ddp_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            data_metadata=data_metadata,
            collator=collator,
            context=context,
            plan=plan,
            global_step=global_step,
            batch_position=batch_position,
            tokens_consumed=tokens_consumed,
        )
        if context.is_main:
            print(f"Training complete. Final checkpoint: {final_dir}", flush=True)
    finally:
        if dataset is not None:
            dataset.close()
        if context is not None and context.process_group_initialized and dist.is_initialized():
            dist.destroy_process_group()
