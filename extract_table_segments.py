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


BREAK_TOKEN_RE = re.compile(r"(?is)(?:<\s*br\s*/?\s*>|</p>\s*<p>|\\r\\n|\\n|\\r|\r\n|\n|\r)+")
BR_RE = re.compile(r"(?is)<\s*br\s*/?\s*>")
TAG_RE = re.compile(r"(?is)<[^>]+>")
ENTITY_RE = re.compile(r"&[#a-zA-Z0-9]+;")
CONTENT_KEY_RE = re.compile(r"""(?P<quote>["'])content(?P=quote)\s*:""")
ZERO_WIDTH_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
    "\u2060",
}


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


def normalize_fragment(fragment: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", fragment).lower()
    result: list[str] = []
    for char in normalized:
        if char in ZERO_WIDTH_CHARS:
            continue
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cf"}:
            continue
        result.append(char)
    return result


def normalize_with_mapping(text: str | None) -> tuple[str, list[int]]:
    if not text:
        return "", []

    normalized_chars: list[str] = []
    mapping: list[int] = []
    index = 0

    while index < len(text):
        br_match = BR_RE.match(text, index)
        if br_match:
            index = br_match.end()
            continue

        tag_match = TAG_RE.match(text, index)
        if tag_match:
            index = tag_match.end()
            continue

        entity_match = ENTITY_RE.match(text, index)
        if entity_match:
            fragment = html.unescape(entity_match.group(0))
            for char in normalize_fragment(fragment):
                normalized_chars.append(char)
                mapping.append(index)
            index = entity_match.end()
            continue

        for char in normalize_fragment(text[index]):
            normalized_chars.append(char)
            mapping.append(index)
        index += 1

    return "".join(normalized_chars), mapping


def normalize_text(text: str | None) -> str:
    return normalize_with_mapping(text)[0]


def parse_table_info(raw_value: Any) -> list[dict[str, Any]]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]

    if isinstance(raw_value, dict):
        return [raw_value]

    if not isinstance(raw_value, str):
        return []

    text = raw_value.strip()
    if not text:
        return []

    parsed: Any | None = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            break
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def extract_peer_contents(raw_value: Any) -> list[str]:
    contents: list[str] = []
    parsed_items = parse_table_info(raw_value)
    if parsed_items:
        for item in parsed_items:
            content = item.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                content = str(content)
            if normalize_text(content):
                contents.append(content)
        return contents

    if isinstance(raw_value, str):
        for content in extract_contents_loose(raw_value):
            if normalize_text(content):
                contents.append(content)
    return contents


def extract_contents_loose(text: str) -> list[str]:
    contents: list[str] = []
    search_from = 0
    while True:
        match = CONTENT_KEY_RE.search(text, search_from)
        if not match:
            break

        value_start = match.end()
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1

        if value_start >= len(text):
            break

        if text.startswith("None", value_start) or text.startswith("null", value_start):
            search_from = value_start + 4
            continue

        quote = text[value_start]
        if quote not in {"'", '"'}:
            search_from = value_start + 1
            continue

        value, value_end = parse_loose_quoted_string(text, value_start)
        contents.append(value)
        search_from = value_end

    return contents


def parse_loose_quoted_string(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    index = start + 1
    chars: list[str] = []

    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                chars.append("\\")
                index += 1
                continue

            escaped = text[index + 1]
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

            if escaped == "u" and index + 5 < len(text):
                hex_code = text[index + 2 : index + 6]
                try:
                    chars.append(chr(int(hex_code, 16)))
                    index += 6
                    continue
                except ValueError:
                    pass

            chars.append(escape_map.get(escaped, escaped))
            index += 2
            continue

        if char == quote:
            return "".join(chars), index + 1

        chars.append(char)
        index += 1

    return "".join(chars), index


def split_peer_content(content: str) -> tuple[str, int | None]:
    stripped = content.strip()
    if not stripped:
        return "", None

    match = BREAK_TOKEN_RE.search(content)
    if not match:
        return stripped, None

    body = content[match.end() :].strip()
    if not normalize_text(body):
        return stripped, None

    header = content[: match.start()]
    header_norm_len = len(normalize_text(header))
    return body, header_norm_len


def estimate_body_start_from_span(span_text: str) -> int | None:
    for match in BREAK_TOKEN_RE.finditer(span_text):
        left_norm = normalize_text(span_text[: match.start()])
        right_norm = normalize_text(span_text[match.end() :])
        if left_norm and right_norm:
            return match.end()
    return None


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_candidate_windows(source_norm: str, query_norm: str) -> list[tuple[int, int]]:
    if not source_norm or not query_norm:
        return []

    query_len = len(query_norm)
    anchor_len = min(48, max(8, query_len // 4))
    query_len = len(query_norm)
    padding = max(32, query_len // 3)
    candidates: set[tuple[int, int]] = set()

    anchor_starts = {
        0,
        max(0, query_len // 2 - anchor_len // 2),
        max(0, query_len - anchor_len),
    }

    for anchor_start in anchor_starts:
        anchor = query_norm[anchor_start : anchor_start + anchor_len]
        if not anchor:
            continue
        search_from = 0
        hit_count = 0
        while True:
            found_at = source_norm.find(anchor, search_from)
            if found_at == -1:
                break
            start = max(0, found_at - anchor_start - padding)
            end = min(len(source_norm), found_at - anchor_start + query_len + padding)
            candidates.add((start, end))
            search_from = found_at + 1
            hit_count += 1
            if hit_count >= 20:
                break

    return sorted(candidates)


def locate_in_norm_text(source_norm: str, query_norm: str) -> tuple[int, int, float, str] | None:
    if not source_norm or not query_norm:
        return None

    exact_at = source_norm.find(query_norm)
    if exact_at != -1:
        return exact_at, exact_at + len(query_norm), 1.0, "exact"

    best_result: tuple[int, int, float, str] | None = None
    for window_start, window_end in build_candidate_windows(source_norm, query_norm):
        window = source_norm[window_start:window_end]
        matcher = SequenceMatcher(None, query_norm, window, autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        if not blocks:
            continue

        matched_chars = sum(block.size for block in blocks)
        coverage = matched_chars / max(1, len(query_norm))
        span_start = min(block.b for block in blocks)
        span_end = max(block.b + block.size for block in blocks)
        compactness = matched_chars / max(1, span_end - span_start)
        score = coverage * 0.8 + compactness * 0.2

        if coverage < 0.6:
            continue

        candidate = (
            window_start + span_start,
            window_start + span_end,
            score,
            "fuzzy",
        )
        if best_result is None or candidate[2] > best_result[2]:
            best_result = candidate

    return best_result


def similarity_ratio(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def resolve_overlaps(matches: list[TableMatch]) -> list[TableMatch]:
    selected: list[TableMatch] = []
    for match in sorted(
        matches,
        key=lambda item: (
            item.body_span[0] if item.body_span else 10**18,
            item.body_span[1] if item.body_span else 10**18,
        ),
    ):
        if not match.body_span:
            continue
        if not selected:
            selected.append(match)
            continue

        previous = selected[-1]
        if not previous.body_span:
            selected.append(match)
            continue

        if match.body_span[0] >= previous.body_span[1]:
            selected.append(match)
            continue

        previous_length = previous.body_span[1] - previous.body_span[0]
        current_length = match.body_span[1] - match.body_span[0]
        previous_key = (previous.match_score, previous.similarity, previous_length)
        current_key = (match.match_score, match.similarity, current_length)
        if current_key > previous_key:
            selected[-1] = match

    return selected


def segment_text(full_text: str, table_matches: list[TableMatch]) -> tuple[list[str], list[int]]:
    text_list: list[str] = []
    type_list: list[int] = []
    cursor = 0

    for match in resolve_overlaps(table_matches):
        if not match.body_span:
            continue

        start, end = match.body_span
        if start > cursor:
            normal_text = full_text[cursor:start]
            if normal_text:
                text_list.append(normal_text)
                type_list.append(0)

        table_text = full_text[start:end]
        if table_text:
            text_list.append(table_text)
            type_list.append(1)
        cursor = max(cursor, end)

    if cursor < len(full_text):
        tail_text = full_text[cursor:]
        if tail_text:
            text_list.append(tail_text)
            type_list.append(0)

    if not text_list:
        return [full_text], [0]
    return text_list, type_list


def extract_tables_from_record(record: dict[str, Any]) -> tuple[list[str], list[int], list[TableMatch]]:
    ctt = record.get("ctt")
    if not isinstance(ctt, str):
        ctt = "" if ctt is None else str(ctt)

    source_norm, source_map = normalize_with_mapping(ctt)
    peer_contents = extract_peer_contents(record.get("tabel_info"))
    matches: list[TableMatch] = []

    for table_idx, peer_content in enumerate(peer_contents):
        peer_body, peer_body_offset_norm = split_peer_content(peer_content)
        query_norm, peer_map = normalize_with_mapping(peer_content)

        located = locate_in_norm_text(source_norm, query_norm)
        if not located or not source_map:
            matches.append(
                TableMatch(
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
            )
            continue

        norm_start, norm_end, match_score, match_mode = located
        if norm_start >= len(source_map) or norm_end <= 0:
            matches.append(
                TableMatch(
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
            )
            continue

        raw_full_start = source_map[norm_start]
        raw_full_end = source_map[norm_end - 1] + 1
        raw_full_start, raw_full_end = trim_span(ctt, raw_full_start, raw_full_end)
        if raw_full_start >= raw_full_end:
            matches.append(
                TableMatch(
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
            )
            continue

        raw_body_start = raw_full_start
        if match_mode == "exact" and peer_body_offset_norm is not None and norm_start + peer_body_offset_norm < norm_end:
            raw_body_start = source_map[norm_start + peer_body_offset_norm]
        else:
            raw_span = ctt[raw_full_start:raw_full_end]
            body_start_in_span = estimate_body_start_from_span(raw_span)
            if body_start_in_span is not None:
                raw_body_start = raw_full_start + body_start_in_span

        raw_body_start, raw_full_end = trim_span(ctt, raw_body_start, raw_full_end)
        extracted_table = ctt[raw_body_start:raw_full_end]
        if match_mode == "exact" and peer_body_offset_norm is None and raw_body_start > raw_full_start and peer_map:
            inferred_peer_body_offset = len(normalize_text(ctt[raw_full_start:raw_body_start]))
            if 0 < inferred_peer_body_offset < len(peer_map):
                peer_body = peer_content[peer_map[inferred_peer_body_offset] :].strip()

        similarity_target = peer_body if normalize_text(peer_body) else peer_content
        similarity = similarity_ratio(extracted_table, similarity_target)

        matches.append(
            TableMatch(
                table_idx=table_idx,
                peer_content=peer_content,
                peer_body=peer_body,
                matched=True,
                extracted_table=extracted_table,
                similarity=similarity,
                match_score=match_score,
                match_mode=match_mode,
                full_span=[raw_full_start, raw_full_end],
                body_span=[raw_body_start, raw_full_end],
            )
        )

    text_list, type_list = segment_text(ctt, matches)
    return text_list, type_list, matches


def derive_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def process_file(input_path: Path, output_path: Path, flat_output_path: Path) -> None:
    total_records = 0
    total_tables = 0
    matched_tables = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as record_writer, flat_output_path.open("w", encoding="utf-8") as flat_writer:
        for line_no, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total_records += 1
            record = json.loads(stripped)
            text_list, type_list, table_matches = extract_tables_from_record(record)

            output_record = dict(record)
            output_record["text_list"] = text_list
            output_record["type_list"] = type_list
            output_record["table_metrics"] = [match.to_dict() for match in table_matches]
            record_writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            for match in table_matches:
                total_tables += 1
                if match.matched:
                    matched_tables += 1

                flat_record = {
                    "record_index": total_records - 1,
                    "line_no": line_no,
                    **match.to_dict(),
                }
                flat_writer.write(json.dumps(flat_record, ensure_ascii=False) + "\n")

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
    parser = argparse.ArgumentParser(
        description=(
            "从 jsonl 数据中读取网页正文 ctt 和 tabel_info，"
            "抽取表格正文片段，并输出文本切分结果与表格相似度。"
        )
    )
    parser.add_argument("input", type=Path, help="输入 jsonl 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="按原记录输出的 jsonl 路径，默认自动生成",
    )
    parser.add_argument(
        "--flat-output",
        type=Path,
        help="打平后的表格指标 jsonl 路径，默认自动生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or derive_output_path(input_path, ".table_segments.jsonl")
    flat_output_path: Path = args.flat_output or derive_output_path(
        input_path, ".table_metrics.flat.jsonl"
    )
    process_file(input_path, output_path, flat_output_path)


if __name__ == "__main__":
    main()
