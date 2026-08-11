from __future__ import annotations

import math
from typing import Any


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Warm up linearly, then decay from peak_lr to min_lr with cosine."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= step < total_steps:
        raise ValueError("step must be within the training schedule")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0.0 <= min_lr <= peak_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= peak_lr")
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    if decay_steps == 1:
        return min_lr
    progress = decay_step / (decay_steps - 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def validate_training_config(config: dict[str, Any]) -> None:
    positive_ints = (
        "sequence_length",
        "train_tokens",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "log_every_steps",
    )
    for key in positive_ints:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["train_tokens"] % config["sequence_length"]:
        raise ValueError("train_tokens must be a multiple of sequence_length")
    if not 0.0 < float(config["mlm_probability"]) < 1.0:
        raise ValueError("mlm_probability must be between zero and one")
    if float(config["min_learning_rate"]) > float(config["learning_rate"]):
        raise ValueError("min_learning_rate cannot exceed learning_rate")
    if int(config["warmup_steps"]) < 0:
        raise ValueError("warmup_steps cannot be negative")
    if config.get("parameter_dtype", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("parameter_dtype must be 'float32' or 'bfloat16'")
