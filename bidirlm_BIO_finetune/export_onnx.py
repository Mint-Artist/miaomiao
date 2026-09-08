"""Export a SELECT BidirLM checkpoint to ONNX (backbone + both heads in one graph).

The graph takes ``input_ids`` and ``attention_mask`` and returns
``classification_logits[B, L, 3]`` and ``transition_logits[B, L, 3, 3]``.
Batch and sequence axes are dynamic.  Everything after the logits
(log_softmax, Viterbi, char mapping) stays outside the graph, in float32.

Declaring dynamic axes is not proof that they survived tracing: model code
that slices a cache with a Python int bakes the export length into the graph
without raising.  This script therefore re-runs the exported graph at several
*different* lengths and compares against PyTorch, and also checks that
right-padding with ``attention_mask=0`` leaves the real positions unchanged --
the property that bucketed, padded serving depends on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .modeling import SelectBidirLM


DTYPES = {"float32": torch.float32, "float16": torch.float16}
INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAMES = ("classification_logits", "transition_logits")


class SelectOnnxModule(nn.Module):
    """Backbone plus both heads, with ONNX-traceable shape handling."""

    def __init__(self, backbone: nn.Module, classification_head: nn.Module, transition_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classification_head = classification_head
        self.transition_head = transition_head

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        classification_logits = self.classification_head(hidden_states)
        transition_logits = self.transition_head(hidden_states)
        # -1 keeps the sequence axis symbolic; a literal length would bake in.
        transition_logits = transition_logits.reshape(
            transition_logits.shape[0], -1, 3, 3
        )
        return classification_logits, transition_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a merged SELECT BidirLM checkpoint to ONNX"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint or export_merged output")
    parser.add_argument("--output", required=True, help="path to the .onnx file to write")
    parser.add_argument("--base-model-name-or-path")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float32",
        help=(
            "graph precision; float32 is the safe default and lets Ascend ATC "
            "pick its own precision mode, float16 halves the file"
        ),
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--export-length", type=int, default=256)
    parser.add_argument(
        "--verify-lengths",
        default="128,384,777",
        help="comma-separated lengths re-checked against PyTorch after export",
    )
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--attention-implementation", default="eager")
    return parser


def load_export_module(
    checkpoint: Path,
    *,
    base_model_name_or_path: Optional[str],
    attention_implementation: str,
    dtype: torch.dtype,
) -> Tuple[SelectOnnxModule, Dict[str, Any]]:
    model = SelectBidirLM.from_checkpoint(
        checkpoint,
        base_model_name_or_path=base_model_name_or_path,
        gradient_checkpointing=False,
        attention_implementation=attention_implementation,
    )
    model.eval()
    backbone = model.backbone
    source_mode = model.finetuning_mode
    if source_mode == "lora":
        backbone = backbone.merge_and_unload()
    module = SelectOnnxModule(backbone, model.classification_head, model.transition_head)
    module = module.to(dtype).eval()
    info = {
        "source_mode": source_mode,
        "merged_lora": source_mode == "lora",
        "vocab_size": int(backbone.config.vocab_size),
        "hidden_size": int(backbone.config.hidden_size),
    }
    return module, info


def export(module: SelectOnnxModule, output: Path, *, length: int, vocab_size: int, opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    input_ids = torch.randint(0, vocab_size, (1, length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "classification_logits": {0: "batch", 1: "sequence"},
        "transition_logits": {0: "batch", 1: "sequence"},
    }
    with torch.no_grad():
        torch.onnx.export(
            module,
            (input_ids, attention_mask),
            str(output),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )


def exported_files(output: Path) -> List[Dict[str, Any]]:
    """List the .onnx file plus any external-data siblings it created."""

    directory = output.parent
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and (path == output or path.name.startswith(output.name)):
            files.append({"file": path.name, "bytes": path.stat().st_size})
    return files


def verify(
    module: SelectOnnxModule,
    output: Path,
    *,
    lengths: Sequence[int],
    vocab_size: int,
    tolerance: float,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Compare ONNX against PyTorch at new lengths and under right-padding."""

    import numpy as np
    import onnxruntime as ort

    from .standalone_inference import viterbi

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    checks: List[Dict[str, Any]] = []
    worst = 0.0
    labels_match = True

    def run_onnx(ids: torch.Tensor, mask: torch.Tensor):
        outputs = session.run(
            list(OUTPUT_NAMES),
            {"input_ids": ids.numpy(), "attention_mask": mask.numpy()},
        )
        return [torch.from_numpy(np.asarray(item)).float() for item in outputs]

    generator = torch.Generator().manual_seed(0)
    for length in lengths:
        ids = torch.randint(0, vocab_size, (1, length), generator=generator)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            torch_cls, torch_tr = module(ids, mask)
        onnx_cls, onnx_tr = run_onnx(ids, mask)
        difference = max(
            float((torch_cls.float() - onnx_cls).abs().max()),
            float((torch_tr.float() - onnx_tr).abs().max()),
        )
        worst = max(worst, difference)
        torch_labels = viterbi(
            torch_cls[0].float().log_softmax(-1),
            torch_tr[0].float().log_softmax(-1),
            [True] * length,
        )
        onnx_labels = viterbi(
            onnx_cls[0].log_softmax(-1), onnx_tr[0].log_softmax(-1), [True] * length
        )
        same = torch_labels == onnx_labels
        labels_match = labels_match and same
        checks.append(
            {
                "check": "length",
                "length": length,
                "max_abs_diff": difference,
                "labels_identical": same,
            }
        )

    # Right-padding must not disturb the real positions; bucketed serving
    # depends on this, and it is cheap to prove here rather than in production.
    base_length = int(lengths[0])
    padded_length = base_length + 37
    ids = torch.randint(0, vocab_size, (1, base_length), generator=generator)
    mask = torch.ones_like(ids)
    padded_ids = torch.zeros((1, padded_length), dtype=torch.long)
    padded_ids[:, :base_length] = ids
    padded_mask = torch.zeros((1, padded_length), dtype=torch.long)
    padded_mask[:, :base_length] = 1
    plain_cls, plain_tr = run_onnx(ids, mask)
    padded_cls, padded_tr = run_onnx(padded_ids, padded_mask)
    padding_difference = max(
        float((plain_cls - padded_cls[:, :base_length]).abs().max()),
        float((plain_tr - padded_tr[:, :base_length]).abs().max()),
    )
    worst = max(worst, padding_difference)
    checks.append(
        {
            "check": "right_padding_invariance",
            "length": base_length,
            "padded_to": padded_length,
            "max_abs_diff": padding_difference,
        }
    )

    tolerance = tolerance if dtype is torch.float32 else max(tolerance, 2e-2)
    return {
        "checks": checks,
        "max_abs_diff": worst,
        "tolerance": tolerance,
        "labels_identical": labels_match,
        "passed": worst <= tolerance and labels_match,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dtype = DTYPES[args.dtype]
    output = Path(args.output)
    module, info = load_export_module(
        Path(args.checkpoint),
        base_model_name_or_path=args.base_model_name_or_path,
        attention_implementation=args.attention_implementation,
        dtype=dtype,
    )
    export(
        module,
        output,
        length=args.export_length,
        vocab_size=info["vocab_size"],
        opset=args.opset,
    )

    summary: Dict[str, Any] = {
        "output": str(output),
        "files": exported_files(output),
        "dtype": args.dtype,
        "opset": args.opset,
        "inputs": list(INPUT_NAMES),
        "outputs": list(OUTPUT_NAMES),
        "dynamic_axes": ["batch", "sequence"],
        **info,
    }
    if args.skip_verify:
        summary["verification"] = "skipped"
    else:
        lengths = [int(item) for item in args.verify_lengths.split(",") if item.strip()]
        try:
            summary["verification"] = verify(
                module,
                output,
                lengths=lengths,
                vocab_size=info["vocab_size"],
                tolerance=args.tolerance,
                dtype=dtype,
            )
        except ImportError:
            summary["verification"] = "onnxruntime not installed; install it and re-run"
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    verification = summary.get("verification")
    if isinstance(verification, dict) and not verification["passed"]:
        print(
            "VERIFICATION FAILED: the graph does not reproduce PyTorch at other "
            "lengths. The export length is likely baked into the graph; do not "
            "deploy it.",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
