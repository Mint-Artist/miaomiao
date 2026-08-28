from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .constants import (
    CLASSIFICATION_LOGITS_KEY,
    CLASSIFICATION_LOSS_KEY,
    IGNORE_INDEX,
    LOSS_KEY,
    NUM_BIO_LABELS,
    NUM_BIO_TRANSITIONS,
    TAG_TO_LABEL,
    TRANSITION_LOGITS_KEY,
    TRANSITION_LOSS_KEY,
)

DEFAULT_DROPOUT = 0.1
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SUPPORTED_FINETUNING_MODES = frozenset({"lora", "full", "heads_only"})


class SelectBidirLM(nn.Module):
    """BidirLM encoder with SELECT classification and transition heads."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        finetuning_mode: str = "lora",
        dropout: float = DEFAULT_DROPOUT,
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
        gradient_checkpointing: bool = True,
        attention_implementation: Optional[str] = "eager",
        adapter_path: Optional[str] = None,
    ):
        super().__init__()
        if finetuning_mode not in SUPPORTED_FINETUNING_MODES:
            raise ValueError("finetuning_mode must be lora, full, or heads_only")
        self.base_model_name_or_path = str(model_name_or_path)
        self.finetuning_mode = finetuning_mode
        self.dropout_probability = float(dropout)
        self.lora_settings = {
            "rank": int(lora_rank),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "target_modules": list(lora_target_modules),
        }

        self.backbone = _load_backbone(model_name_or_path, attention_implementation)
        self.backbone = _configure_finetuning(
            self.backbone,
            finetuning_mode=finetuning_mode,
            adapter_path=adapter_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        if gradient_checkpointing:
            _enable_gradient_checkpointing(self.backbone, finetuning_mode)

        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(hidden_size, NUM_BIO_LABELS)
        self.transition_head = nn.Linear(hidden_size, NUM_BIO_TRANSITIONS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        if hidden_states.shape[:2] != input_ids.shape:
            raise RuntimeError(
                "BidirLM output length does not match input length: "
                f"{tuple(hidden_states.shape[:2])} vs {tuple(input_ids.shape)}"
            )
        hidden_states = self.dropout(hidden_states)
        classification_logits = self.classification_head(hidden_states)
        transition_logits = self.transition_head(hidden_states).view(
            *hidden_states.shape[:2], NUM_BIO_LABELS, NUM_BIO_LABELS
        )
        result = {
            CLASSIFICATION_LOGITS_KEY: classification_logits,
            TRANSITION_LOGITS_KEY: transition_logits,
        }
        if labels is not None:
            losses = select_loss(
                classification_logits, transition_logits, labels, attention_mask
            )
            result.update(losses)
        return result

    def trainable_parameter_summary(self) -> Dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "trainable_percent": 100.0 * trainable / total,
        }

    def save_artifacts(self, output_dir: str | Path, tokenizer: Any = None) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "select-bidirlm-v1",
            "base_model_name_or_path": self.base_model_name_or_path,
            "finetuning_mode": self.finetuning_mode,
            "dropout": self.dropout_probability,
            "lora": self.lora_settings,
            "label2id": TAG_TO_LABEL,
            "ignore_index": IGNORE_INDEX,
        }
        (output_path / "select_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "classification_head": self.classification_head.state_dict(),
                "transition_head": self.transition_head.state_dict(),
            },
            output_path / "select_heads.pt",
        )
        if self.finetuning_mode == "full":
            self.backbone.save_pretrained(
                output_path / "backbone", safe_serialization=True
            )
        elif self.finetuning_mode == "lora":
            self.backbone.save_pretrained(
                output_path / "adapter", safe_serialization=True
            )
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path / "tokenizer")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        base_model_name_or_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        attention_implementation: Optional[str] = "eager",
    ) -> "SelectBidirLM":
        checkpoint_path = Path(checkpoint_dir)
        metadata = json.loads(
            (checkpoint_path / "select_config.json").read_text(encoding="utf-8")
        )
        if not isinstance(metadata, Mapping):
            raise TypeError("select_config.json must contain a JSON object")
        mode = _required_string(metadata, "finetuning_mode")
        configured_base_path = _required_string(metadata, "base_model_name_or_path")
        base_path = base_model_name_or_path or configured_base_path
        adapter_path = None
        if mode == "full":
            base_path = str(checkpoint_path / "backbone")
        elif mode == "lora":
            adapter_path = str(checkpoint_path / "adapter")
        lora_value = metadata.get("lora", {})
        lora = lora_value if isinstance(lora_value, Mapping) else {}
        model = cls(
            str(base_path),
            finetuning_mode=mode,
            dropout=float(metadata.get("dropout", DEFAULT_DROPOUT)),
            lora_rank=int(lora.get("rank", DEFAULT_LORA_RANK)),
            lora_alpha=int(lora.get("alpha", DEFAULT_LORA_ALPHA)),
            lora_dropout=float(lora.get("dropout", DEFAULT_LORA_DROPOUT)),
            lora_target_modules=lora.get("target_modules", DEFAULT_LORA_TARGET_MODULES),
            gradient_checkpointing=gradient_checkpointing,
            attention_implementation=attention_implementation,
            adapter_path=adapter_path,
        )
        heads = torch.load(
            checkpoint_path / "select_heads.pt", map_location="cpu", weights_only=True
        )
        if not isinstance(heads, Mapping):
            raise TypeError("select_heads.pt must contain a state dictionary")
        classification_state = heads.get("classification_head")
        transition_state = heads.get("transition_head")
        if classification_state is None or transition_state is None:
            raise KeyError("select_heads.pt is missing a required head state")
        model.classification_head.load_state_dict(classification_state)
        model.transition_head.load_state_dict(transition_state)
        return model


def select_loss(
    classification_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    valid_tokens = (labels != IGNORE_INDEX) & attention_mask.bool()
    if not valid_tokens.any():
        raise ValueError("batch contains no supervised tokens")
    classification_loss = F.cross_entropy(
        classification_logits[valid_tokens].float(),
        labels[valid_tokens],
        reduction="mean",
    )

    from_labels = labels[:, :-1]
    to_labels = labels[:, 1:]
    transition_mask = (
        (from_labels != IGNORE_INDEX)
        & (to_labels != IGNORE_INDEX)
        & attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
    )
    logits = transition_logits[:, :-1]
    safe_from_labels = from_labels.clamp(min=0)
    selected_rows = logits.gather(
        2,
        safe_from_labels[..., None, None].expand(-1, -1, 1, NUM_BIO_LABELS),
    ).squeeze(2)
    if transition_mask.any():
        transition_loss = F.cross_entropy(
            selected_rows[transition_mask].float(),
            to_labels[transition_mask],
            reduction="mean",
        )
    else:
        transition_loss = selected_rows.sum() * 0.0
    return {
        LOSS_KEY: classification_loss + transition_loss,
        CLASSIFICATION_LOSS_KEY: classification_loss.detach(),
        TRANSITION_LOSS_KEY: transition_loss.detach(),
    }


def _load_backbone(
    model_name_or_path: str,
    attention_implementation: Optional[str],
) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install bidirlm_BIO_finetune/requirements.txt"
        ) from exc
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if attention_implementation:
        load_kwargs.update({"attn_implementation": attention_implementation})
    backbone = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    return backbone


def _configure_finetuning(
    backbone: nn.Module,
    *,
    finetuning_mode: str,
    adapter_path: Optional[str],
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
) -> nn.Module:
    if finetuning_mode == "heads_only":
        backbone.requires_grad_(False)
        return backbone
    if finetuning_mode != "lora":
        return backbone
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for --finetuning-mode lora") from exc
    if adapter_path:
        return PeftModel.from_pretrained(backbone, adapter_path, is_trainable=True)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
        bias="none",
    )
    return get_peft_model(backbone, peft_config)


def _enable_gradient_checkpointing(backbone: nn.Module, mode: str) -> None:
    if not hasattr(backbone, "gradient_checkpointing_enable"):
        raise RuntimeError(
            "the loaded backbone does not support gradient checkpointing"
        )
    try:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        backbone.gradient_checkpointing_enable()
    if mode == "lora" and hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise KeyError(f"missing or invalid required field {key!r}")
    return value
