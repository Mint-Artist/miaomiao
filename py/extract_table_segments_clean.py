#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


BREAK_RE = re.compile(r"(?is)(?:<\s*br\s*/?\s*>|</p>\s*<p>|\\r\\n|\\n|\\r|\r\n|\n|\r)+")
BR_RE = re.compile(r"(?is)<\s*br\s*/?\s*>")
TAG_RE = re.compile(r"(?is)<[^>]+>")
ENTITY_RE = re.compile(r"&[#a-zA-Z0-9]+;")
CONTENT_RE = re.compile(r"""(?P<q>["'])content(?P=q)\s*:""")
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}


@dataclass
class NormalizedText:
    raw: str
    text: str
    mapping: list[int]


@dataclass
class MatchLocation:
    start: int
    end: int
    score: float
    mode: str


@dataclass
class TableMatch:
    table_idx: int
    peer_content: str
    peer_body: str
    matched: bool
    extracted_table: str
    similarity: float
    match_score: float
    match_mode: str
    full_span: list[int] | None
    body_span: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_idx": self.table_idx,
            "peer_content": self.peer_content,
            "peer_body": self.peer_body,
            "matched": self.matched,
            "extracted_table": self.extracted_table,
            "similarity": round(self.similarity, 6),
            "match_score": round(self.match_score, 6),
            "match_mode": self.match_mode,
            "full_span": self.full_span,
            "body_span": self.body_span,
        }


def normalize_piece(text: str) -> str:
    chars: list[str] = []
    for char in unicodedata.normalize("NFKC", text).lower():
        if char in ZERO_WIDTH or char.isspace():
            continue
        if unicodedata.category(char) in {"Cc", "Cf"}:
            continue
        chars.append(char)
    return "".join(chars)


def normalize_with_mapping(text: Any) -> NormalizedText:
    raw = "" if text is None else str(text)
    if not raw:
        return NormalizedText(raw="", text="", mapping=[])

    chars: list[str] = []
    mapping: list[int] = []
    i = 0

    while i < len(raw):
        br_match = BR_RE.match(raw, i)
        if br_match:
            i = br_match.end()
            continue

        tag_match = TAG_RE.match(raw, i)
        if tag_match:
            i = tag_match.end()
            continue

        entity_match = ENTITY_RE.match(raw, i)
        if entity_match:
            decoded = html.unescape(entity_match.group(0))
            piece = normalize_piece(decoded)
            chars.extend(piece)
            mapping.extend([i] * len(piece))
            i = entity_match.end()
            continue

        piece = normalize_piece(raw[i])
        chars.extend(piece)
        mapping.extend([i] * len(piece))
        i += 1

    return NormalizedText(raw=raw, text="".join(chars), mapping=mapping)


def normalize_text(text: Any) -> str:
    return normalize_with_mapping(text).text


def parse_table_info(raw_value: Any) -> list[dict[str, Any]]:
    if raw_value is None:
        return []
    if isinstance(raw_value, dict):
        return [raw_value]
    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw_value.strip())
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def parse_loose_quoted_string(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    i = start + 1
    chars: list[str] = []
    escape_map = {
        "\\": "\\",
        "'": "'",
        '"': '"',
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
        "/": "/",
    }

    while i < len(text):
        char = text[i]
        if char == "\\":
            if i + 1 >= len(text):
                chars.append("\\")
                i += 1
                continue
            nxt = text[i + 1]
            if nxt == "u" and i + 5 < len(text):
                code = text[i + 2 : i + 6]
                try:
                    chars.append(chr(int(code, 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            chars.append(escape_map.get(nxt, nxt))
            i += 2
            continue
        if char == quote:
            return "".join(chars), i + 1
        chars.append(char)
        i += 1

    return "".join(chars), i


def extract_contents_loose(text: str) -> list[str]:
    contents: list[str] = []
    start = 0
    while True:
        match = CONTENT_RE.search(text, start)
        if not match:
            return contents

        i = match.end()
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            return contents
        if text.startswith("None", i) or text.startswith("null", i):
            start = i + 4
            continue
        if text[i] not in {"'", '"'}:
            start = i + 1
            continue

        value, start = parse_loose_quoted_string(text, i)
        contents.append(value)


def extract_peer_contents(raw_value: Any) -> list[str]:
    contents: list[str] = []
    parsed_items = parse_table_info(raw_value)
    if parsed_items:
        candidates = [item.get("content") for item in parsed_items]
    elif isinstance(raw_value, str):
        candidates = extract_contents_loose(raw_value)
    else:
        candidates = []

    for content in candidates:
        if content is None:
            continue
        content = str(content)
        if normalize_text(content):
            contents.append(content)
    return contents


def split_peer_content(content: str) -> tuple[str, int | None]:
    content = content.strip()
    if not content:
        return "", None

    match = BREAK_RE.search(content)
    if not match:
        return content, None

    body = content[match.end() :].strip()
    if not normalize_text(body):
        return content, None

    header = content[: match.start()]
    return body, len(normalize_text(header))


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def estimate_body_start(span_text: str) -> int | None:
    for match in BREAK_RE.finditer(span_text):
        left = normalize_text(span_text[: match.start()])
        right = normalize_text(span_text[match.end() :])
        if left and right:
            return match.end()
    return None


def iter_candidate_windows(source: str, query: str) -> list[tuple[int, int]]:
    if not source or not query:
        return []

    query_len = len(query)
    anchor_len = min(48, max(8, query_len // 4))
    padding = max(32, query_len // 3)
    anchor_starts = {0, max(0, query_len // 2 - anchor_len // 2), max(0, query_len - anchor_len)}
    windows: set[tuple[int, int]] = set()

    for anchor_start in anchor_starts:
        anchor = query[anchor_start : anchor_start + anchor_len]
        if not anchor:
            continue
        hits = 0
        pos = 0
        while hits < 20:
            pos = source.find(anchor, pos)
            if pos == -1:
                break
            start = max(0, pos - anchor_start - padding)
            end = min(len(source), pos - anchor_start + query_len + padding)
            windows.add((start, end))
            pos += 1
            hits += 1

    return sorted(windows)


def locate_query(source: str, query: str) -> MatchLocation | None:
    if not source or not query:
        return None

    exact_start = source.find(query)
    if exact_start != -1:
        return MatchLocation(exact_start, exact_start + len(query), 1.0, "exact")

    best: MatchLocation | None = None
    for win_start, win_end in iter_candidate_windows(source, query):
        window = source[win_start:win_end]
        matcher = SequenceMatcher(None, query, window, autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        if not blocks:
            continue

        matched_chars = sum(block.size for block in blocks)
        coverage = matched_chars / len(query)
        if coverage < 0.6:
            continue

        span_start = min(block.b for block in blocks)
        span_end = max(block.b + block.size for block in blocks)
        compactness = matched_chars / max(1, span_end - span_start)
        score = coverage * 0.8 + compactness * 0.2
        current = MatchLocation(win_start + span_start, win_start + span_end, score, "fuzzy")
        if best is None or current.score > best.score:
            best = current
    return best


def similarity_ratio(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def build_unmatched(table_idx: int, peer_content: str, peer_body: str) -> TableMatch:
    return TableMatch(
        table_idx=table_idx,
        peer_content=peer_content,
        peer_body=peer_body,
        matched=False,
        extracted_table="",
        similarity=0.0,
        match_score=0.0,
        match_mode="unmatched",
        full_span=None,
        body_span=None,
    )


def infer_peer_body(
    peer_content: str,
    peer_body: str,
    peer_norm: NormalizedText,
    raw_text: str,
    raw_full_start: int,
    raw_body_start: int,
    match_mode: str,
    header_norm_len: int | None,
) -> str:
    if header_norm_len is not None and normalize_text(peer_body):
        return peer_body
    if match_mode != "exact" or raw_body_start <= raw_full_start or not peer_norm.mapping:
        return peer_body

    prefix_norm_len = len(normalize_text(raw_text[raw_full_start:raw_body_start]))
    if 0 < prefix_norm_len < len(peer_norm.mapping):
        return peer_content[peer_norm.mapping[prefix_norm_len] :].strip()
    return peer_body


def extract_single_table(
    source: NormalizedText,
    peer_content: str,
    table_idx: int,
) -> TableMatch:
    peer_body, header_norm_len = split_peer_content(peer_content)
    peer_norm = normalize_with_mapping(peer_content)
    location = locate_query(source.text, peer_norm.text)
    if not location or not source.mapping:
        return build_unmatched(table_idx, peer_content, peer_body)

    if location.start >= len(source.mapping) or location.end <= 0:
        return build_unmatched(table_idx, peer_content, peer_body)

    raw_full_start = source.mapping[location.start]
    raw_full_end = source.mapping[location.end - 1] + 1
    raw_full_start, raw_full_end = trim_span(source.raw, raw_full_start, raw_full_end)
    if raw_full_start >= raw_full_end:
        return build_unmatched(table_idx, peer_content, peer_body)

    raw_body_start = raw_full_start
    if location.mode == "exact" and header_norm_len is not None and location.start + header_norm_len < location.end:
        raw_body_start = source.mapping[location.start + header_norm_len]
    else:
        offset = estimate_body_start(source.raw[raw_full_start:raw_full_end])
        if offset is not None:
            raw_body_start = raw_full_start + offset

    raw_body_start, raw_full_end = trim_span(source.raw, raw_body_start, raw_full_end)
    extracted_table = source.raw[raw_body_start:raw_full_end]
    peer_body = infer_peer_body(
        peer_content=peer_content,
        peer_body=peer_body,
        peer_norm=peer_norm,
        raw_text=source.raw,
        raw_full_start=raw_full_start,
        raw_body_start=raw_body_start,
        match_mode=location.mode,
        header_norm_len=header_norm_len,
    )
    target = peer_body if normalize_text(peer_body) else peer_content

    return TableMatch(
        table_idx=table_idx,
        peer_content=peer_content,
        peer_body=peer_body,
        matched=True,
        extracted_table=extracted_table,
        similarity=similarity_ratio(extracted_table, target),
        match_score=location.score,
        match_mode=location.mode,
        full_span=[raw_full_start, raw_full_end],
        body_span=[raw_body_start, raw_full_end],
    )


def resolve_overlaps(matches: list[TableMatch]) -> list[TableMatch]:
    selected: list[TableMatch] = []
    ordered = sorted(
        (match for match in matches if match.body_span),
        key=lambda match: (match.body_span[0], match.body_span[1]),  # type: ignore[index]
    )

    for match in ordered:
        if not selected:
            selected.append(match)
            continue

        prev = selected[-1]
        if match.body_span[0] >= prev.body_span[1]:  # type: ignore[index]
            selected.append(match)
            continue

        prev_len = prev.body_span[1] - prev.body_span[0]  # type: ignore[index]
        curr_len = match.body_span[1] - match.body_span[0]  # type: ignore[index]
        prev_key = (prev.match_score, prev.similarity, prev_len)
        curr_key = (match.match_score, match.similarity, curr_len)
        if curr_key > prev_key:
            selected[-1] = match

    return selected


def build_text_segments(raw_text: str, matches: list[TableMatch]) -> tuple[list[str], list[int]]:
    text_list: list[str] = []
    type_list: list[int] = []
    cursor = 0

    for match in resolve_overlaps(matches):
        start, end = match.body_span  # type: ignore[misc]
        if start > cursor:
            text_list.append(raw_text[cursor:start])
            type_list.append(0)
        text_list.append(raw_text[start:end])
        type_list.append(1)
        cursor = max(cursor, end)

    if cursor < len(raw_text):
        text_list.append(raw_text[cursor:])
        type_list.append(0)

    if not text_list:
        return [raw_text], [0]
    return text_list, type_list


def process_record(record: dict[str, Any]) -> tuple[list[str], list[int], list[TableMatch]]:
    raw_text = "" if record.get("ctt") is None else str(record.get("ctt"))
    source = normalize_with_mapping(raw_text)
    peer_contents = extract_peer_contents(record.get("tabel_info"))
    matches = [extract_single_table(source, content, idx) for idx, content in enumerate(peer_contents)]
    text_list, type_list = build_text_segments(raw_text, matches)
    return text_list, type_list, matches


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def process_file(input_path: Path, output_path: Path, flat_output_path: Path) -> None:
    total_records = 0
    total_tables = 0
    matched_tables = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as record_writer, flat_output_path.open("w", encoding="utf-8") as flat_writer:
        for line_no, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            total_records += 1
            record = json.loads(line)
            text_list, type_list, matches = process_record(record)

            output_record = dict(record)
            output_record["text_list"] = text_list
            output_record["type_list"] = type_list
            output_record["table_metrics"] = [match.to_dict() for match in matches]
            record_writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            for match in matches:
                total_tables += 1
                matched_tables += int(match.matched)
                flat_writer.write(
                    json.dumps(
                        {
                            "record_index": total_records - 1,
                            "line_no": line_no,
                            **match.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_records": str(output_path),
                "output_flat": str(flat_output_path),
                "total_records": total_records,
                "total_tables": total_tables,
                "matched_tables": matched_tables,
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取网页正文中的表格正文，并输出切分结果和相似度。")
    parser.add_argument("input", type=Path, help="输入 jsonl 文件")
    parser.add_argument("-o", "--output", type=Path, help="逐条记录输出文件")
    parser.add_argument("--flat-output", type=Path, help="打平后的表格结果文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input, ".table_segments.jsonl")
    flat_output_path = args.flat_output or default_output_path(args.input, ".table_metrics.flat.jsonl")
    process_file(args.input, output_path, flat_output_path)


if __name__ == "__main__":
    main()
