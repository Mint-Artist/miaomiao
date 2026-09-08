"""Reference ONNX inference for SELECT BidirLM: numpy + onnxruntime only.

No torch. This file exists to be read and ported: everything the graph does
not do is here in plain numpy, and the only framework-specific part is
``OnnxRunner`` -- swap that one class to target ACL/OM, TensorRT, or an
in-house engine, and the rest carries over unchanged.

Pipeline::

    text
      -> tokenizer            input_ids, attention_mask, offsets
      -> graph (OnnxRunner)   classification_logits, transition_logits
      -> log_softmax + Viterbi   O/B/I per token
      -> offsets              character spans -> refined text

Long documents are split into overlapping windows; each position keeps the
logits from the window where it sits farthest from an edge, so no position is
ever predicted at a window boundary that is not a real document boundary.

Fixed-shape backends: pass ``--buckets`` to right-pad each window up to the
next bucket size with ``attention_mask`` 0. Padding on the right leaves the
real positions' logits unchanged (export_onnx.py verifies this), and the
padded tail is dropped before decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LABEL_O, LABEL_B, LABEL_I = 0, 1, 2
INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAMES = ("classification_logits", "transition_logits")


# --------------------------------------------------------------------------
# The only framework-specific piece. Replace with your engine's equivalent.
# --------------------------------------------------------------------------
class OnnxRunner:
    """Run the exported graph. Must return float32 logits for one batch item.

    ``run`` takes ``input_ids`` and ``attention_mask`` of shape ``[1, L]``
    (int64) and returns ``(classification_logits[L, 3], transition_logits[L,
    3, 3])``. A port only has to preserve that contract.
    """

    def __init__(self, model_path: str | Path, providers: Optional[Sequence[str]] = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=list(providers or ["CPUExecutionProvider"])
        )

    def run(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        classification, transition = self.session.run(
            list(OUTPUT_NAMES),
            {
                INPUT_NAMES[0]: input_ids.astype(np.int64),
                INPUT_NAMES[1]: attention_mask.astype(np.int64),
            },
        )
        return (
            np.asarray(classification, dtype=np.float32)[0],
            np.asarray(transition, dtype=np.float32)[0],
        )


def load_tokenizer(tokenizer_path: str | Path):
    """Prefer transformers; fall back to the standalone tokenizers library."""

    tokenizer_path = Path(tokenizer_path)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    except ImportError:
        from tokenizers import Tokenizer

        path = (
            tokenizer_path
            if tokenizer_path.is_file()
            else tokenizer_path / "tokenizer.json"
        )
        return Tokenizer.from_file(str(path))


def encode(tokenizer: Any, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return token ids and character offsets, whichever tokenizer we have.

    Offsets must match the ones used at training time exactly; a mismatch
    silently shifts every character span.
    """

    if hasattr(tokenizer, "encode") and not callable(getattr(tokenizer, "__call__", None)):
        encoded = tokenizer.encode(text)
        return list(encoded.ids), [tuple(item) for item in encoded.offsets]
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        return (
            [int(item) for item in encoded["input_ids"]],
            [(int(start), int(end)) for start, end in encoded["offset_mapping"]],
        )
    except TypeError:  # tokenizers.Tokenizer is callable in some versions
        encoded = tokenizer.encode(text)
        return list(encoded.ids), [tuple(item) for item in encoded.offsets]


def log_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def window_plan(length: int, window: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """(start, end, keep_start, keep_end) windows covering [0, length)."""

    if window < 1 or stride < 1 or stride > window:
        raise ValueError("require 1 <= stride <= window")
    if length <= window:
        return [(0, length, 0, length)]
    margin = (window - stride) // 2
    starts = list(range(0, length - window, stride)) + [length - window]
    plan = []
    for index, start in enumerate(starts):
        end = start + window
        keep_start = 0 if index == 0 else start + margin
        keep_end = length if index == len(starts) - 1 else end - margin
        plan.append((start, end, keep_start, keep_end))
    return plan


def pad_to_bucket(
    input_ids: np.ndarray, buckets: Optional[Sequence[int]], pad_token_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Right-pad to the next bucket; returns ids, mask and the real length."""

    length = int(input_ids.shape[1])
    if not buckets:
        return input_ids, np.ones_like(input_ids), length
    candidates = [size for size in sorted(buckets) if size >= length]
    if not candidates:
        raise ValueError(
            f"sequence length {length} exceeds the largest bucket {max(buckets)}; "
            "lower --window or add a bigger bucket"
        )
    target = candidates[0]
    padded_ids = np.full((1, target), pad_token_id, dtype=input_ids.dtype)
    padded_ids[:, :length] = input_ids
    mask = np.zeros((1, target), dtype=input_ids.dtype)
    mask[:, :length] = 1
    return padded_ids, mask, length


def stitched_log_probs(
    runner: Any,
    input_ids: Sequence[int],
    *,
    window: int,
    stride: int,
    buckets: Optional[Sequence[int]] = None,
    pad_token_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    length = len(input_ids)
    ids = np.asarray([list(input_ids)], dtype=np.int64)
    cls_log_probs = np.empty((length, 3), dtype=np.float32)
    tr_log_probs = np.empty((length, 3, 3), dtype=np.float32)
    for start, end, keep_start, keep_end in window_plan(length, window, stride):
        chunk = ids[:, start:end]
        padded, mask, real_length = pad_to_bucket(chunk, buckets, pad_token_id)
        classification, transition = runner.run(padded, mask)
        # Drop the padded tail before it can reach the decoder.
        classification = classification[:real_length]
        transition = transition[:real_length]
        cls_lp = log_softmax(classification, axis=-1)
        tr_lp = log_softmax(transition, axis=-1)
        cls_log_probs[keep_start:keep_end] = cls_lp[keep_start - start : keep_end - start]
        tr_log_probs[keep_start:keep_end] = tr_lp[keep_start - start : keep_end - start]
    return cls_log_probs, tr_log_probs


def viterbi(
    cls_log_probs: np.ndarray, tr_log_probs: np.ndarray, valid: Sequence[bool]
) -> List[int]:
    """Best label path; invalid positions get -100 and split the sequence."""

    labels = [-100] * len(valid)
    for run in _contiguous_runs([index for index, flag in enumerate(valid) if flag]):
        score = cls_log_probs[run[0]]
        backpointers: List[np.ndarray] = []
        for previous_index, current_index in zip(run, run[1:]):
            candidates = score[:, None] + tr_log_probs[previous_index]
            backpointers.append(candidates.argmax(axis=0))
            score = candidates.max(axis=0) + cls_log_probs[current_index]
        current = int(score.argmax())
        path = [current]
        for pointer in reversed(backpointers):
            current = int(pointer[current])
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
    """Half-open retained spans; B opens one, and a stray I opens one too."""

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
    offsets: Sequence[Tuple[int, int]], token_spans: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    char_spans = []
    for token_start, token_end in token_spans:
        real = [item for item in offsets[token_start:token_end] if item[1] > item[0]]
        if real:
            char_spans.append((real[0][0], real[-1][1]))
    return char_spans


def refine_text(
    text: str,
    tokenizer: Any,
    runner: Any,
    *,
    window: int = 8192,
    stride: int = 6144,
    buckets: Optional[Sequence[int]] = None,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    input_ids, offsets = encode(tokenizer, text)
    valid = [end > start for start, end in offsets]
    cls_lp, tr_lp = stitched_log_probs(
        runner,
        input_ids,
        window=window,
        stride=stride,
        buckets=buckets,
        pad_token_id=pad_token_id,
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
        description="Refine raw text JSONL with an exported SELECT BidirLM ONNX graph"
    )
    parser.add_argument("--model", required=True, help="path to model.onnx")
    parser.add_argument(
        "--tokenizer", required=True, help="tokenizer directory or tokenizer.json"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="source_text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=6144)
    parser.add_argument(
        "--buckets",
        help=(
            "comma-separated fixed shapes for backends that need them, e.g. "
            "512,1024,2048,4096,8192; omit for fully dynamic backends"
        ),
    )
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--providers", help="comma-separated onnxruntime providers")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="apply boundary snap/trim rules from postprocess.py if importable",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    buckets = (
        [int(item) for item in args.buckets.split(",") if item.strip()]
        if args.buckets
        else None
    )
    providers = (
        [item.strip() for item in args.providers.split(",") if item.strip()]
        if args.providers
        else None
    )
    runner = OnnxRunner(args.model, providers=providers)
    tokenizer = load_tokenizer(args.tokenizer)

    postprocess_fn = None
    if args.postprocess:
        try:
            from .postprocess import postprocess_char_spans

            postprocess_fn = postprocess_char_spans
        except ImportError:
            try:
                from postprocess import postprocess_char_spans  # type: ignore

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
                text,
                tokenizer,
                runner,
                window=args.window,
                stride=args.stride,
                buckets=buckets,
                pad_token_id=args.pad_token_id,
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
