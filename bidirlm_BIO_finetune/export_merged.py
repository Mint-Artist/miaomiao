"""Merge a SELECT BidirLM checkpoint into a self-contained safetensors export.

The export can be loaded by ``predict.py`` (as ``finetuning_mode=full``) and
by ``standalone_inference.py``, which needs only torch, transformers and
safetensors.  Layout::

    <output>/backbone/                   merged weights (model.safetensors),
                                         config.json, configuration_bidirlm.py,
                                         modeling_bidirlm.py
    <output>/tokenizer/
    <output>/select_heads.safetensors    classification + transition heads
    <output>/select_heads.pt             same heads, for predict.py
    <output>/select_config.json

BidirLM is not a stock Qwen3: its ``model_type`` is ``bidirlm`` and its
attention is bidirectional, so the custom modeling files must travel with the
weights and the model must be loaded with ``trust_remote_code=True``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from safetensors.torch import save_file

from .modeling import SelectBidirLM


CUSTOM_CODE_FILES = ("configuration_bidirlm.py", "modeling_bidirlm.py")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge LoRA (if any) and export a SELECT BidirLM as safetensors"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-model-name-or-path",
        help="base BidirLM path; defaults to the one recorded in select_config.json",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--attention-implementation", default="eager")
    return parser


def heads_state_dict(model: Any) -> Dict[str, torch.Tensor]:
    """Flatten both heads into safetensors-friendly keys."""

    flat: Dict[str, torch.Tensor] = {}
    for head_name in ("classification_head", "transition_head"):
        head = getattr(model, head_name)
        for key, value in head.state_dict().items():
            flat[f"{head_name}.{key}"] = value.detach().to("cpu", torch.float32).contiguous()
    return flat


def write_heads(output_dir: Path, heads: Dict[str, torch.Tensor]) -> None:
    save_file(heads, str(output_dir / "select_heads.safetensors"))
    nested: Dict[str, Dict[str, torch.Tensor]] = {
        "classification_head": {},
        "transition_head": {},
    }
    for key, value in heads.items():
        head_name, parameter = key.split(".", 1)
        nested[head_name][parameter] = value
    torch.save(nested, output_dir / "select_heads.pt")


def copy_custom_code(source_dir: Optional[Path], backbone_dir: Path) -> list[str]:
    """Copy BidirLM's trust_remote_code files next to the merged weights."""

    copied = []
    if source_dir is None or not source_dir.is_dir():
        return copied
    for name in CUSTOM_CODE_FILES:
        source = source_dir / name
        target = backbone_dir / name
        if source.is_file() and not target.exists():
            shutil.copyfile(source, target)
            copied.append(name)
    return copied


def write_metadata(
    output_dir: Path,
    *,
    source_checkpoint: str,
    source_mode: str,
    base_model_name_or_path: str,
    dropout: float,
    dtype: str,
) -> None:
    metadata = {
        "format": "select-bidirlm-v1",
        "exported_from": source_checkpoint,
        "exported_from_mode": source_mode,
        "base_model_name_or_path": base_model_name_or_path,
        "finetuning_mode": "full",
        "dropout": dropout,
        "lora": {},
        "label2id": {"O": 0, "B": 1, "I": 2},
        "ignore_index": -100,
        "backbone_dtype": dtype,
        "heads_dtype": "float32",
        "heads_files": ["select_heads.safetensors", "select_heads.pt"],
        "notes": (
            "load backbone with AutoModel.from_pretrained(<output>/backbone, "
            "trust_remote_code=True); apply heads to last_hidden_state; decode "
            "with Viterbi (see standalone_inference.py)"
        ),
    }
    (output_dir / "select_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory {output_dir} is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SelectBidirLM.from_checkpoint(
        checkpoint,
        base_model_name_or_path=args.base_model_name_or_path,
        gradient_checkpointing=False,
        attention_implementation=args.attention_implementation,
    )
    model.eval()
    source_mode = model.finetuning_mode
    backbone = model.backbone
    if source_mode == "lora":
        backbone = backbone.merge_and_unload()
    backbone = backbone.to(DTYPES[args.dtype])

    backbone_dir = output_dir / "backbone"
    backbone.save_pretrained(backbone_dir, safe_serialization=True)

    metadata = json.loads((checkpoint / "select_config.json").read_text(encoding="utf-8"))
    base_path = args.base_model_name_or_path or metadata["base_model_name_or_path"]
    code_sources = [Path(base_path)]
    if source_mode == "full":
        code_sources.insert(0, checkpoint / "backbone")
    copied: list[str] = []
    for source in code_sources:
        copied += copy_custom_code(source, backbone_dir)
    missing = [
        name for name in CUSTOM_CODE_FILES if not (backbone_dir / name).exists()
    ]

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install bidirlm_BIO_finetune/requirements.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "tokenizer", trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")

    write_heads(output_dir, heads_state_dict(model))
    write_metadata(
        output_dir,
        source_checkpoint=str(checkpoint),
        source_mode=source_mode,
        base_model_name_or_path=str(base_path),
        dropout=float(model.dropout_probability),
        dtype=args.dtype,
    )

    summary = {
        "output": str(output_dir),
        "merged_lora": source_mode == "lora",
        "backbone_dtype": args.dtype,
        "custom_code_copied": copied,
        "custom_code_missing": missing,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        print(
            "WARNING: copy the missing BidirLM files into backbone/ manually, "
            "otherwise trust_remote_code loading will fail on another machine"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
