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
    """Backbone plus both heads, with ONNX-traceable shape handling.

    ``mask_mode="4d"`` converts the 2D padding mask into a 4D additive mask
    inside this wrapper.  transformers' ``_preprocess_mask_arguments`` returns
    a 4D mask as-is, which skips its mask-construction path -- that path
    expands index tensors using Python ints (``expand(batch_size, -1,
    q_length, kv_length)``) and bakes the export length into the graph.  The
    conversion here is pure broadcasting, so no shape is ever read.

    The graph input stays 2D either way: callers pass ``[B, L]`` with 1 for
    real tokens and 0 for padding.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classification_head: nn.Module,
        transition_head: nn.Module,
        *,
        mask_mode: str = "4d",
    ):
        super().__init__()
        if mask_mode not in {"2d", "4d"}:
            raise ValueError("mask_mode must be 2d or 4d")
        self.backbone = backbone
        self.classification_head = classification_head
        self.transition_head = transition_head
        self.mask_mode = mask_mode

    def _expand_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # [B, L] -> [B, 1, 1, L] additive mask: 0 keeps a key, min blocks it.
        # Only keys are masked; masking query rows too would make a padded
        # row all -inf and turn its softmax into NaN.
        keep = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - keep) * torch.finfo(dtype).min

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask
        if self.mask_mode == "4d":
            mask = self._expand_mask(attention_mask, self.classification_head.weight.dtype)
        hidden_states = self.backbone(
            input_ids=input_ids, attention_mask=mask, return_dict=True
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
    parser.add_argument(
        "--mask-mode",
        choices=("4d", "2d"),
        default="4d",
        help=(
            "4d converts the padding mask inside the wrapper so transformers "
            "skips its mask builder, whose Python-int expand bakes in the "
            "export length; 2d passes the mask through unchanged"
        ),
    )
    parser.add_argument(
        "--exporter",
        choices=("tracing", "dynamo"),
        default="tracing",
        help="dynamo uses torch.export symbolic shapes; try it if tracing bakes shapes",
    )
    parser.add_argument("--export-length", type=int, default=256)
    parser.add_argument(
        "--verify-lengths",
        default="128,384,777",
        help="comma-separated lengths re-checked against PyTorch after export",
    )
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help=(
            "keep torch's one-file-per-tensor external data instead of merging "
            "it into a single <model>.onnx.data sibling"
        ),
    )
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--attention-implementation", default="eager")
    return parser


def load_export_module(
    checkpoint: Path,
    *,
    base_model_name_or_path: Optional[str],
    attention_implementation: str,
    dtype: torch.dtype,
    mask_mode: str = "4d",
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
    module = SelectOnnxModule(
        backbone, model.classification_head, model.transition_head, mask_mode=mask_mode
    )
    module = module.to(dtype).eval()
    info = {
        "source_mode": source_mode,
        "merged_lora": source_mode == "lora",
        "vocab_size": int(backbone.config.vocab_size),
        "hidden_size": int(backbone.config.hidden_size),
    }
    return module, info


def export(
    module: SelectOnnxModule,
    output: Path,
    *,
    length: int,
    vocab_size: int,
    opset: int,
    exporter: str = "tracing",
) -> None:
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
        if exporter == "dynamo":
            from torch.export import Dim

            batch = Dim("batch", min=1, max=64)
            sequence = Dim("sequence", min=2, max=65536)
            torch.onnx.export(
                module,
                (input_ids, attention_mask),
                str(output),
                input_names=list(INPUT_NAMES),
                output_names=list(OUTPUT_NAMES),
                dynamic_shapes={
                    "input_ids": {0: batch, 1: sequence},
                    "attention_mask": {0: batch, 1: sequence},
                },
                opset_version=opset,
                dynamo=True,
            )
            return
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
    """List every file in the output directory; all of them must be copied.

    Weights above the 2GB protobuf limit live in external-data files whose
    names come from the tensors, not from the model, so matching on the model
    filename would under-report what deployment needs.
    """

    return [
        {"file": path.name, "bytes": path.stat().st_size}
        for path in sorted(output.parent.iterdir())
        if path.is_file()
    ]


def inline_external_tensors(model: Any, base_dir: Path) -> int:
    """Read external tensor data with plain file I/O and inline it.

    ``onnx.load`` routes external data through the C++ checker, which rejects
    some locations outright ("kernel rejected path"). The files are ours and
    sit next to the model, so read them directly instead; this also tolerates
    an absolute or directory-qualified location by falling back to the
    basename inside the model directory.
    """

    import onnx
    from onnx.external_data_helper import ExternalDataInfo, _get_all_tensors

    inlined = 0
    for tensor in _get_all_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = ExternalDataInfo(tensor)
        candidate = Path(info.location)
        path = candidate if candidate.is_absolute() else base_dir / candidate
        if not path.is_file():
            path = base_dir / candidate.name
        offset = int(getattr(info, "offset", 0) or 0)
        length = int(getattr(info, "length", 0) or 0)
        with open(path, "rb") as stream:
            if offset:
                stream.seek(offset)
            tensor.raw_data = stream.read(length) if length else stream.read()
        tensor.data_location = onnx.TensorProto.DEFAULT
        del tensor.external_data[:]
        inlined += 1
    return inlined


def consolidate_external_data(output: Path) -> Dict[str, Any]:
    """Rewrite per-tensor external-data files into a single ``.data`` sibling.

    torch.onnx.export writes one file per tensor once the model exceeds the
    2GB protobuf limit, which leaves hundreds of files that must all travel
    together. Consolidating keeps that to two.
    """

    import onnx

    location = output.name + ".data"
    stale = [
        path
        for path in output.parent.iterdir()
        if path.is_file() and path.name not in {output.name, location}
    ]
    if not stale:
        return {"consolidated": False, "reason": "weights already fit in the model file"}

    model = onnx.load(str(output), load_external_data=False)
    inlined = inline_external_tensors(model, output.parent)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=1024,
        convert_attribute=False,
    )
    for path in stale:
        path.unlink()
    return {
        "consolidated": True,
        "location": location,
        "inlined_tensors": inlined,
        "removed_files": len(stale),
    }


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
        mask_mode=args.mask_mode,
    )
    export(
        module,
        output,
        length=args.export_length,
        vocab_size=info["vocab_size"],
        opset=args.opset,
        exporter=args.exporter,
    )

    consolidation: Dict[str, Any] = {"consolidated": False, "reason": "disabled"}
    if not args.no_consolidate:
        # Merging external data is a convenience; a failure here must not lose
        # the export or skip verification.
        try:
            consolidation = consolidate_external_data(output)
        except Exception as exc:  # noqa: BLE001
            consolidation = {
                "consolidated": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "hint": (
                    "the export itself is fine; copy every file in the output "
                    "directory, or re-export with --dtype float16 to stay under "
                    "the 2GB limit and get a single file"
                ),
            }

    summary: Dict[str, Any] = {
        "output": str(output),
        "external_data": consolidation,
        "files": exported_files(output),
        "dtype": args.dtype,
        "opset": args.opset,
        "inputs": list(INPUT_NAMES),
        "outputs": list(OUTPUT_NAMES),
        "dynamic_axes": ["batch", "sequence"],
        "mask_mode": args.mask_mode,
        "exporter": args.exporter,
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
            "deploy it. Try --mask-mode 4d (default) and then --exporter dynamo.",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
