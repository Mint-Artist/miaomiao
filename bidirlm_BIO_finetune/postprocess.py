"""Boundary post-processing for predicted retained segments.

Gold spans inherit two boundary artifacts: teacher refinements that start or
stop mid-sentence, and char-to-token expansion pulling in punctuation merged
into an edge token (for example an opening bracket kept at a segment end).
These rules repair predicted character spans without touching the model:

- snap: move a boundary to the nearest sentence boundary within a small
  window, preferring the direction that recovers dropped source text.
- trim: strip edge whitespace, unpaired opening brackets/quotes at the end,
  and unpaired closing brackets/quotes at the start.

Run as a module to add post-processed fields to an existing prediction JSONL
produced by ``bidirlm_BIO_finetune.predict`` without re-running inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CharSpan = Tuple[int, int]

# Directional pairs only; ASCII straight quotes are ambiguous and left alone.
OPEN_TO_CLOSE: Dict[str, str] = {
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
    "《": "》",
    "〈": "〉",
    "「": "」",
    "『": "』",
    "｛": "｝",
    "{": "}",
    "“": "”",
    "‘": "’",
}
CLOSE_TO_OPEN: Dict[str, str] = {value: key for key, value in OPEN_TO_CLOSE.items()}

# ASCII "." is excluded because of decimals, URLs and file names.
DEFAULT_BOUNDARY_CHARS = "。！？；…\n!?;"


def snap_span(
    text: str,
    start: int,
    end: int,
    *,
    window: int = 5,
    boundary_chars: str = DEFAULT_BOUNDARY_CHARS,
) -> CharSpan:
    """Move span edges to the nearest sentence boundary within ``window``.

    Ties prefer moving the start left and the end right, because the known
    failure modes lose source text at the edges rather than keep extra text.
    """

    if window > 0:
        if not _good_start(text, start, boundary_chars):
            for distance in range(1, window + 1):
                for candidate in (start - distance, start + distance):
                    if 0 <= candidate < end and _good_start(
                        text, candidate, boundary_chars
                    ):
                        start = candidate
                        break
                else:
                    continue
                break
        if not _good_end(text, end, boundary_chars):
            for distance in range(1, window + 1):
                for candidate in (end + distance, end - distance):
                    if start < candidate <= len(text) and _good_end(
                        text, candidate, boundary_chars
                    ):
                        end = candidate
                        break
                else:
                    continue
                break
    return start, end


def trim_span(text: str, start: int, end: int) -> CharSpan:
    """Strip edge whitespace and unpaired directional punctuation."""

    changed = True
    while changed and start < end:
        changed = False
        if text[start].isspace():
            start += 1
            changed = True
            continue
        if text[end - 1].isspace():
            end -= 1
            changed = True
            continue
        segment = text[start:end]
        last = text[end - 1]
        if last in OPEN_TO_CLOSE and segment.count(last) > segment.count(
            OPEN_TO_CLOSE[last]
        ):
            end -= 1
            changed = True
            continue
        first = text[start]
        if first in CLOSE_TO_OPEN and segment.count(first) > segment.count(
            CLOSE_TO_OPEN[first]
        ):
            start += 1
            changed = True
    return start, end


def postprocess_char_spans(
    text: str,
    spans: Sequence[Sequence[int]],
    *,
    snap_window: int = 5,
    boundary_chars: str = DEFAULT_BOUNDARY_CHARS,
    min_chars: int = 0,
) -> List[CharSpan]:
    """Snap, trim, drop empty/short spans, and merge overlapping results."""

    processed: List[CharSpan] = []
    for raw_start, raw_end in spans:
        start, end = int(raw_start), int(raw_end)
        if not 0 <= start < end <= len(text):
            raise ValueError(f"span [{start}, {end}) is outside the source text")
        start, end = snap_span(
            text, start, end, window=snap_window, boundary_chars=boundary_chars
        )
        start, end = trim_span(text, start, end)
        if end - start > max(0, min_chars):
            processed.append((start, end))

    processed.sort()
    merged: List[CharSpan] = []
    for start, end in processed:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _good_start(text: str, position: int, boundary_chars: str) -> bool:
    return position == 0 or text[position - 1] in boundary_chars


def _good_end(text: str, position: int, boundary_chars: str) -> bool:
    if position == len(text):
        return True
    return text[position - 1] in boundary_chars or text[position] == "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add post-processed retained segments to a prediction JSONL from "
            "bidirlm_BIO_finetune.predict (requires source_text and "
            "predicted_retained_segments)"
        ),
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snap-window", type=int, default=5)
    parser.add_argument("--boundary-chars", default=DEFAULT_BOUNDARY_CHARS)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Drop post-processed segments this many characters or shorter",
    )
    parser.add_argument(
        "--separator",
        default="",
        help="Separator joining segments into postprocessed_refined_text",
    )
    return parser


def postprocess_record(
    record: Dict[str, Any],
    *,
    snap_window: int,
    boundary_chars: str,
    min_chars: int,
    separator: str,
) -> Dict[str, Any]:
    source_text = record.get("source_text")
    segments = record.get("predicted_retained_segments")
    if not isinstance(source_text, str) or not isinstance(segments, list):
        raise ValueError(
            "record requires source_text and predicted_retained_segments; "
            "run predict on raw or audit input to produce them"
        )
    spans = [segment["char_span"] for segment in segments]
    processed = postprocess_char_spans(
        source_text,
        spans,
        snap_window=snap_window,
        boundary_chars=boundary_chars,
        min_chars=min_chars,
    )
    texts = [source_text[start:end] for start, end in processed]
    record["postprocessed_char_spans"] = [list(span) for span in processed]
    record["postprocessed_segments"] = texts
    record["postprocessed_refined_text"] = separator.join(texts)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.snap_window < 0:
        raise ValueError("--snap-window must be non-negative")
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = segments_before = segments_after = 0
    with input_path.open("r", encoding="utf-8") as input_stream, output_path.open(
        "w", encoding="utf-8"
    ) as output_stream:
        for line in input_stream:
            if not line.strip():
                continue
            record = json.loads(line)
            segments_before += len(record.get("predicted_retained_segments", []))
            record = postprocess_record(
                record,
                snap_window=args.snap_window,
                boundary_chars=args.boundary_chars,
                min_chars=args.min_chars,
                separator=args.separator,
            )
            segments_after += len(record["postprocessed_segments"])
            records += 1
            output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"records={records} segments_before={segments_before} "
        f"segments_after={segments_after}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
