"""Standalone inference for an exported SELECT BidirLM (see export_merged.py).

Depends only on torch, transformers and safetensors; it does not import the
rest of this package, so it can be copied to a serving platform as-is.

How it differs from autoregressive LLM inference:

- one bidirectional forward pass per document, no ``generate`` loop, no KV
  cache, no sampling;
- the outputs are per-token O/B/I logits plus per-position 3x3 transition
  logits; Viterbi picks the best label path;
- B/I tokens are mapped back to character offsets of the *input* text, so the
  refined text is always a verbatim subsequence of the source.

Documents longer than ``--window`` tokens are processed with overlapping
windows; each position keeps the logits from the window where it is farthest
from an edge, then a single Viterbi runs over the stitched sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

LABEL_O, LABEL_B, LABEL_I = 0, 1, 2
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# forward_fn(input_ids[1, L]) -> (classification_logits[L, 3], transition_logits[L, 3, 3])
ForwardFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class SelectHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.classification_head = nn.Linear(hidden_size, 3)
        self.transition_head = nn.Linear(hidden_size, 9)

    @classmethod
    def from_safetensors(cls, path: Path) -> "SelectHeads":
        from safetensors.torch import load_file

        state = load_file(str(path))
        hidden_size = int(state["classification_head.weight"].shape[1])
        heads = cls(hidden_size)
        heads.load_state_dict(state)
        return heads.eval()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.float()
        classification = self.classification_head(hidden_states)
        transition = self.transition_head(hidden_states).view(
            *hidden_states.shape[:-1], 3, 3
        )
        return classification, transition


def load_select_model(
    model_dir: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "float16",
    attention_implementation: Optional[str] = None,
):
    """Return (backbone, heads, tokenizer) from an export_merged.py directory."""

    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": DTYPES[dtype],
    }
    if attention_implementation:
        load_kwargs["attn_implementation"] = attention_implementation
    backbone = AutoModel.from_pretrained(model_dir / "backbone", **load_kwargs)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    backbone = backbone.to(device).eval()
    heads = SelectHeads.from_safetensors(model_dir / "select_heads.safetensors").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", trust_remote_code=True)
    return backbone, heads, tokenizer


def make_forward_fn(backbone: nn.Module, heads: SelectHeads, device: str) -> ForwardFn:
    @torch.no_grad()
    def forward(input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids.to(device)
        attention_mask = torch.ones_like(ids)
        hidden = backbone(
            input_ids=ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        classification, transition = heads(hidden)
        return classification[0].cpu(), transition[0].cpu()

    return forward


def window_plan(length: int, window: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """Return (start, end, keep_start, keep_end) windows covering [0, length)."""

    if window < 1 or stride < 1 or stride > window:
        raise ValueError("require 1 <= stride <= window")
    if length <= window:
        return [(0, length, 0, length)]
    margin = (window - stride) // 2
    starts: List[int] = list(range(0, length - window, stride)) + [length - window]
    plan = []
    for index, start in enumerate(starts):
        end = start + window
        keep_start = 0 if index == 0 else start + margin
        keep_end = length if index == len(starts) - 1 else end - margin
        plan.append((start, end, keep_start, keep_end))
    return plan


def stitched_log_probs(
    forward_fn: ForwardFn,
    input_ids: Sequence[int],
    *,
    window: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    length = len(input_ids)
    ids = torch.tensor([list(input_ids)], dtype=torch.long)
    cls_log_probs = torch.empty((length, 3), dtype=torch.float32)
    tr_log_probs = torch.empty((length, 3, 3), dtype=torch.float32)
    for start, end, keep_start, keep_end in window_plan(length, window, stride):
        classification, transition = forward_fn(ids[:, start:end])
        cls_lp = classification.float().log_softmax(dim=-1)
        tr_lp = transition.float().log_softmax(dim=-1)
        cls_log_probs[keep_start:keep_end] = cls_lp[keep_start - start : keep_end - start]
        tr_log_probs[keep_start:keep_end] = tr_lp[keep_start - start : keep_end - start]
    return cls_log_probs, tr_log_probs


def viterbi(
    cls_log_probs: torch.Tensor,
    tr_log_probs: torch.Tensor,
    valid: Sequence[bool],
) -> List[int]:
    """Decode labels; invalid positions get -100 and split the sequence into runs."""

    labels = [-100] * len(valid)
    for run in _contiguous_runs([index for index, flag in enumerate(valid) if flag]):
        score = cls_log_probs[run[0]]
        backpointers: List[torch.Tensor] = []
        for previous_index, current_index in zip(run, run[1:]):
            candidates = score[:, None] + tr_log_probs[previous_index]
            best_score, best_previous = candidates.max(dim=0)
            score = best_score + cls_log_probs[current_index]
            backpointers.append(best_previous)
        current = int(score.argmax().item())
        path = [current]
        for pointer in reversed(backpointers):
            current = int(pointer[current].item())
            path.append(current)
        path.reverse()
        for index, label in zip(run, path):
            labels[index] = label
    return labels


def _contiguous_runs(indices: Sequence[int]) -> Iterable[List[int]]:
    if not indices:
        return
    run = [indices[0]]
    for index in indices[1:]:
        if index == run[-1] + 1:
            run.append(index)
        else:
            yield run
            run = [index]
    yield run


def labels_to_token_spans(labels: Sequence[int]) -> List[Tuple[int, int]]:
    """Half-open retained token spans; B starts a span, a stray I starts one too."""

    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, label in enumerate(list(labels) + [LABEL_O]):
        if label == LABEL_B:
            if start is not None:
                spans.append((start, index))
            start = index
        elif label == LABEL_I:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    return spans


def token_spans_to_char_spans(
    offsets: Sequence[Sequence[int]], token_spans: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    char_spans = []
    for token_start, token_end in token_spans:
        real = [
            (int(item[0]), int(item[1]))
            for item in offsets[token_start:token_end]
            if int(item[1]) > int(item[0])
        ]
        if real:
            char_spans.append((real[0][0], real[-1][1]))
    return char_spans


def refine_text(
    text: str,
    tokenizer: Any,
    forward_fn: ForwardFn,
    *,
    window: int = 8192,
    stride: int = 6144,
) -> Dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = [int(item) for item in encoded["input_ids"]]
    offsets = [(int(item[0]), int(item[1])) for item in encoded["offset_mapping"]]
    valid = [end > start for start, end in offsets]
    cls_lp, tr_lp = stitched_log_probs(
        forward_fn, input_ids, window=window, stride=stride
    )
    labels = viterbi(cls_lp, tr_lp, valid)
    char_spans = token_spans_to_char_spans(offsets, labels_to_token_spans(labels))
    segments = [text[start:end] for start, end in char_spans]
    return {
        "labels": labels,
        "char_spans": [list(span) for span in char_spans],
        "segments": segments,
        "refined_text": "".join(segments),
        "num_tokens": len(input_ids),
        "num_windows": len(window_plan(len(input_ids), window, stride)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine raw text JSONL with an exported SELECT BidirLM"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="JSONL with a text field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="source_text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=6144)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-implementation")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="apply bidirlm_BIO_finetune.postprocess boundary rules if importable",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    backbone, heads, tokenizer = load_select_model(
        args.model_dir,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
    )
    forward_fn = make_forward_fn(backbone, heads, args.device)

    postprocess_fn = None
    if args.postprocess:
        try:
            from bidirlm_BIO_finetune.postprocess import postprocess_char_spans

            postprocess_fn = postprocess_char_spans
        except ImportError:
            print("postprocess module not importable; skipping", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(args.input).open("r", encoding="utf-8") as stream, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record[args.text_field]
            result = refine_text(
                text, tokenizer, forward_fn, window=args.window, stride=args.stride
            )
            output: Dict[str, Any] = {
                "id": record.get(args.id_field, f"line-{line_number}"),
                "source_text": text,
                **result,
            }
            if postprocess_fn is not None:
                spans = postprocess_fn(text, result["char_spans"])
                texts = [text[start:end] for start, end in spans]
                output["postprocessed_char_spans"] = [list(span) for span in spans]
                output["postprocessed_segments"] = texts
                output["postprocessed_refined_text"] = "".join(texts)
            out.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"documents": count, "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
