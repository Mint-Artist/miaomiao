from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class SelectBidirLM(nn.Module):
    """BidirLM encoder with SELECT classification and transition heads."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        finetuning_mode: str = "lora",
        dropout: float = 0.1,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        gradient_checkpointing: bool = True,
        attention_implementation: Optional[str] = "eager",
        adapter_path: Optional[str] = None,
    ):
        super().__init__()
        if finetuning_mode not in {"lora", "full", "heads_only"}:
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

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required; install bidirlm_BIO_finetune/requirements.txt"
            ) from exc

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if attention_implementation:
            load_kwargs["attn_implementation"] = attention_implementation
        self.backbone = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        if finetuning_mode == "lora":
            try:
                from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required for --finetuning-mode lora"
                ) from exc
            if adapter_path:
                self.backbone = PeftModel.from_pretrained(
                    self.backbone, adapter_path, is_trainable=True
                )
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(lora_target_modules),
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, peft_config)
        elif finetuning_mode == "heads_only":
            self.backbone.requires_grad_(False)

        if gradient_checkpointing:
            if not hasattr(self.backbone, "gradient_checkpointing_enable"):
                raise RuntimeError("the loaded backbone does not support gradient checkpointing")
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
            if finetuning_mode == "lora" and hasattr(
                self.backbone, "enable_input_require_grads"
            ):
                self.backbone.enable_input_require_grads()

        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(hidden_size, 3)
        self.transition_head = nn.Linear(hidden_size, 9)

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
            *hidden_states.shape[:2], 3, 3
        )
        result = {
            "classification_logits": classification_logits,
            "transition_logits": transition_logits,
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
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
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
            "label2id": {"O": 0, "B": 1, "I": 2},
            "ignore_index": -100,
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
            self.backbone.save_pretrained(output_path / "backbone", safe_serialization=True)
        elif self.finetuning_mode == "lora":
            self.backbone.save_pretrained(output_path / "adapter", safe_serialization=True)
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
        mode = metadata["finetuning_mode"]
        base_path = base_model_name_or_path or metadata["base_model_name_or_path"]
        adapter_path = None
        if mode == "full":
            base_path = str(checkpoint_path / "backbone")
        elif mode == "lora":
            adapter_path = str(checkpoint_path / "adapter")
        lora = metadata.get("lora", {})
        model = cls(
            str(base_path),
            finetuning_mode=mode,
            dropout=float(metadata.get("dropout", 0.1)),
            lora_rank=int(lora.get("rank", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            lora_target_modules=lora.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            gradient_checkpointing=gradient_checkpointing,
            attention_implementation=attention_implementation,
            adapter_path=adapter_path,
        )
        heads = torch.load(
            checkpoint_path / "select_heads.pt", map_location="cpu", weights_only=True
        )
        model.classification_head.load_state_dict(heads["classification_head"])
        model.transition_head.load_state_dict(heads["transition_head"])
        return model


def select_loss(
    classification_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    valid_tokens = (labels != -100) & attention_mask.bool()
    if not valid_tokens.any():
        raise ValueError("batch contains no supervised tokens")
    classification_loss = F.cross_entropy(
        classification_logits[valid_tokens].float(), labels[valid_tokens], reduction="mean"
    )

    from_labels = labels[:, :-1]
    to_labels = labels[:, 1:]
    transition_mask = (
        (from_labels != -100)
        & (to_labels != -100)
        & attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
    )
    logits = transition_logits[:, :-1]
    safe_from_labels = from_labels.clamp(min=0)
    selected_rows = logits.gather(
        2,
        safe_from_labels[..., None, None].expand(-1, -1, 1, 3),
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
        "loss": classification_loss + transition_loss,
        "classification_loss": classification_loss.detach(),
        "transition_loss": transition_loss.detach(),
    }
