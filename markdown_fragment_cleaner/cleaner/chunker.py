from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import ChunkConfig
from .models import Block, Document, Fragment, PreparedBlock, RuleFlag
from .tokenization import ApproxTokenCounter, TokenCounter


_INTRO_END_RE = re.compile(r"(?:[:：]|如下(?:所示)?[：:]?|包括(?:以下)?[：:]?|分为(?:以下)?[：:]?)\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])")


@dataclass
class _Unit:
    blocks: List[PreparedBlock]
    token_count: int
    atomic: bool


class SemanticChunker:
    """Pack complete Markdown blocks without crossing structural gaps."""

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.config = config or ChunkConfig()
        self.config.validate()
        self.counter = token_counter or ApproxTokenCounter()

    def chunk_document(
        self,
        document: Document,
        prepared: Sequence[Optional[PreparedBlock]],
    ) -> List[Fragment]:
        fragments: List[Fragment] = []
        for run in _contiguous_runs(prepared):
            for group in self._pack_run(run):
                fragments.append(self._make_fragment(document, group))
        self._add_context(document, fragments, prepared)
        return fragments

    def _pack_run(self, run: Sequence[PreparedBlock]) -> List[List[PreparedBlock]]:
        expanded: List[PreparedBlock] = []
        for item in run:
            expanded.extend(self._split_oversized_prose(item))
        units = self._make_units(expanded)

        packed: List[List[PreparedBlock]] = []
        current: List[PreparedBlock] = []
        for unit in units:
            current_tokens = self._count(current)
            if not current:
                current = list(unit.blocks)
                if unit.token_count > self.config.target_max_tokens:
                    packed.append(current)
                    current = []
                continue

            projected = self._count(current + unit.blocks)
            if projected <= self.config.target_max_tokens:
                current.extend(unit.blocks)
            elif current_tokens < self.config.target_min_tokens and projected <= self.config.hard_max_tokens:
                current.extend(unit.blocks)
            else:
                packed.append(current)
                current = list(unit.blocks)
                if unit.token_count > self.config.target_max_tokens:
                    packed.append(current)
                    current = []

        if current:
            packed.append(current)

        # Merge a short prose tail backward. Structured units are intentionally
        # not merged here because their formatting is part of their semantics.
        if len(packed) >= 2:
            tail = packed[-1]
            previous = packed[-2]
            tail_is_structured = any(item.block.is_structured for item in tail)
            previous_is_structured = any(item.block.is_structured for item in previous)
            if (
                not tail_is_structured
                and not previous_is_structured
                and self._count(tail) < self.config.target_min_tokens
                and self._count(previous + tail) <= self.config.hard_max_tokens
            ):
                previous.extend(tail)
                packed.pop()
        return packed

    def _make_units(self, blocks: Sequence[PreparedBlock]) -> List[_Unit]:
        units: List[_Unit] = []
        index = 0
        while index < len(blocks):
            item = blocks[index]
            if (
                self.config.attach_intro_to_structured_block
                and item.block.type in {"paragraph", "blockquote"}
                and index + 1 < len(blocks)
                and blocks[index + 1].block.is_structured
                and _INTRO_END_RE.search(item.block.text.rstrip())
            ):
                pair = [item, blocks[index + 1]]
                units.append(_Unit(pair, self._count(pair), True))
                index += 2
                continue
            units.append(_Unit([item], self._count([item]), item.block.is_structured))
            index += 1
        return units

    def _split_oversized_prose(self, item: PreparedBlock) -> List[PreparedBlock]:
        block = item.block
        if block.type not in {"paragraph", "blockquote", "html"}:
            return [item]
        if self.counter.count(block.text) <= self.config.hard_max_tokens:
            return [item]

        sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(block.text) if part.strip()]
        if len(sentences) <= 1:
            return [item]

        pieces: List[str] = []
        current = ""
        for sentence in sentences:
            separator = _join_separator(current, sentence)
            proposed = sentence if not current else current + separator + sentence
            if current and self.counter.count(proposed) > self.config.target_max_tokens:
                pieces.append(current)
                current = sentence
            else:
                current = proposed
        if current:
            pieces.append(current)
        if len(pieces) <= 1:
            return [item]

        result: List[PreparedBlock] = []
        for ordinal, piece in enumerate(pieces, 1):
            metadata = dict(block.metadata)
            metadata["split_from_block_id"] = block.block_id
            split_block = Block(
                block_id="%s-s%03d" % (block.block_id, ordinal),
                type=block.type,
                text=piece,
                raw_text=piece,
                heading_path=list(block.heading_path),
                start_line=block.start_line,
                end_line=block.end_line,
                metadata=metadata,
            )
            flags = list(item.flags) + [
                RuleFlag("sentence_split", "soft", "oversized prose split at sentence boundaries")
            ]
            result.append(PreparedBlock(split_block, flags))
        return result

    def _make_fragment(self, document: Document, blocks: Sequence[PreparedBlock]) -> Fragment:
        content = _render(blocks)
        block_ids = [item.block.block_id for item in blocks]
        raw_id = "%s\x1f%d\x1f%s\x1f%s" % (
            document.doc_id, document.source_row, ",".join(block_ids), content
        )
        digest = hashlib.blake2b(raw_id.encode("utf-8"), digest_size=12).hexdigest()
        flags: List[RuleFlag] = []
        seen = set()
        for item in blocks:
            for flag in item.flags:
                key = (flag.code, flag.severity, flag.detail)
                if key not in seen:
                    flags.append(flag)
                    seen.add(key)
        repairs = [
            repair
            for item in blocks
            for repair in item.block.metadata.get("repairs", [])
            if isinstance(repair, dict)
        ]
        metadata = {"chunker_version": "markdown-ast-section-v1"}
        if repairs:
            metadata["repairs"] = repairs
        return Fragment(
            fragment_id="frag-" + digest,
            doc_id=document.doc_id,
            url=document.url,
            doc_title=document.title,
            heading_path=list(blocks[0].block.heading_path),
            content=content,
            block_ids=block_ids,
            block_types=[item.block.type for item in blocks],
            token_count=self.counter.count(content),
            start_line=min(item.block.start_line for item in blocks),
            end_line=max(item.block.end_line for item in blocks),
            source_row=document.source_row,
            flags=flags,
            metadata=metadata,
        )

    def _add_context(
        self,
        document: Document,
        fragments: Sequence[Fragment],
        prepared: Sequence[Optional[PreparedBlock]],
    ) -> None:
        if not self.config.context_chars:
            return
        kept_blocks = [item.block for item in prepared if item is not None]
        for fragment in fragments:
            before = next(
                (block.text for block in reversed(kept_blocks) if block.end_line < fragment.start_line),
                "",
            )
            after = next(
                (block.text for block in kept_blocks if block.start_line > fragment.end_line),
                "",
            )
            fragment.context_before = before[-self.config.context_chars:]
            fragment.context_after = after[:self.config.context_chars]

    def _count(self, blocks: Sequence[PreparedBlock]) -> int:
        return self.counter.count(_render(blocks)) if blocks else 0


def _contiguous_runs(items: Sequence[Optional[PreparedBlock]]) -> Iterable[List[PreparedBlock]]:
    current: List[PreparedBlock] = []
    current_path: Optional[Tuple[str, ...]] = None
    for item in items:
        if item is None:
            if current:
                yield current
            current = []
            current_path = None
            continue
        path = tuple(item.block.heading_path)
        if current and path != current_path:
            yield current
            current = []
        if not current:
            current_path = path
        current.append(item)
    if current:
        yield current


def _render(blocks: Sequence[PreparedBlock]) -> str:
    return "\n\n".join(item.block.text.strip() for item in blocks if item.block.text.strip()).strip()


def _join_separator(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if left[-1].isascii() and right[0].isascii():
        return " "
    return ""
