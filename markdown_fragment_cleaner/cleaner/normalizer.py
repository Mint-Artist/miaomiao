from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from .config import NormalizationConfig
from .models import Block, Document


_PROSE_BLOCK_TYPES = {"paragraph", "html", "blockquote"}
_EDGE_FORMAT_CHARS_RE = re.compile(r"^[\ufeff\u200b\u2060]+|[\ufeff\u200b\u2060]+$")
_LEADING_HTML_ARTIFACT_RE = re.compile(r"^(?:-->|<!--|<!\s*--\s*>)[ \t]*")
_TRAILING_HTML_ARTIFACT_RE = re.compile(r"[ \t]*(?:-->|<!--|<!\s*--\s*>)$")
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[。！？；!?])(?=\s*[^。！？；!?\s])|(?<=\.)(?=\s+[A-Z0-9\"'“‘])"
)
_CJK_CHARACTER_CLASS = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_PUNCTUATION = "，。！？；：、（）【】《》「」『』“”‘’"
_MULTIPLE_HORIZONTAL_SPACES_RE = re.compile(r"[ \t]{2,}")
_CJK_INTERNAL_SPACE_RE = re.compile(
    rf"(?<=[{_CJK_CHARACTER_CLASS}])[ \t]+(?=[{_CJK_CHARACTER_CLASS}])"
)
_CJK_BEFORE_PUNCTUATION_SPACE_RE = re.compile(
    rf"(?<=[{_CJK_CHARACTER_CLASS}])[ \t]+(?=[{re.escape(_CJK_PUNCTUATION)}])"
)
_PUNCTUATION_BEFORE_CJK_SPACE_RE = re.compile(
    rf"(?<=[{re.escape(_CJK_PUNCTUATION)}])[ \t]+(?=[{_CJK_CHARACTER_CLASS}])"
)


@dataclass
class NormalizationStats:
    documents_repaired: int = 0
    blocks_repaired: int = 0
    boundary_artifacts_removed: int = 0
    empty_blocks_removed: int = 0
    spacing_blocks_repaired: int = 0
    extra_spaces_removed: int = 0
    cjk_spaces_removed: int = 0
    duplicate_blocks_removed: int = 0
    duplicate_sentence_sequences_removed: int = 0
    duplicate_sentences_removed: int = 0
    duplicate_chars_removed: int = 0

    @property
    def repaired(self) -> bool:
        return any(
            (
                self.blocks_repaired,
                self.boundary_artifacts_removed,
                self.empty_blocks_removed,
                self.spacing_blocks_repaired,
                self.extra_spaces_removed,
                self.cjk_spaces_removed,
                self.duplicate_blocks_removed,
                self.duplicate_sentence_sequences_removed,
                self.duplicate_sentences_removed,
                self.duplicate_chars_removed,
            )
        )

    def add(self, other: "NormalizationStats") -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class NormalizationResult:
    document: Document
    stats: NormalizationStats


class DocumentNormalizer:
    """Apply narrow, auditable repairs without changing document semantics."""

    def __init__(self, config: NormalizationConfig) -> None:
        self.config = config
        self.config.validate()

    def normalize(self, document: Document) -> NormalizationResult:
        if not self.config.enabled:
            return NormalizationResult(document, NormalizationStats())

        stats = NormalizationStats()
        normalized_blocks: List[Block] = []
        for block in document.blocks:
            normalized, block_stats = self._normalize_block(block)
            stats.add(block_stats)
            if normalized is None:
                continue

            if (
                self.config.deduplicate_adjacent_prose_blocks
                and normalized_blocks
                and _are_duplicate_adjacent_blocks(normalized_blocks[-1], normalized, self.config)
            ):
                previous = normalized_blocks[-1]
                removed_chars = _meaningful_char_count(normalized.text)
                normalized_blocks[-1] = _append_repair(
                    previous,
                    {
                        "code": "adjacent_duplicate_block",
                        "removed_block_id": normalized.block_id,
                        "removed_chars": removed_chars,
                    },
                )
                stats.blocks_repaired += 1
                stats.duplicate_blocks_removed += 1
                stats.duplicate_chars_removed += removed_chars
                continue

            normalized_blocks.append(normalized)

        if stats.repaired:
            stats.documents_repaired = 1
        metadata = dict(document.metadata)
        if stats.repaired:
            metadata["normalization"] = stats.to_dict()
        return NormalizationResult(
            replace(document, blocks=normalized_blocks, metadata=metadata),
            stats,
        )

    def _normalize_block(self, block: Block) -> Tuple[Optional[Block], NormalizationStats]:
        stats = NormalizationStats()
        text = block.text
        repairs: List[Dict[str, Any]] = []

        if self.config.strip_boundary_artifacts:
            text, artifacts = _strip_boundary_artifacts(text)
            if artifacts:
                stats.blocks_repaired = 1
                stats.boundary_artifacts_removed = len(artifacts)
                repairs.append(
                    {
                        "code": "boundary_artifacts_removed",
                        "artifacts": artifacts,
                    }
                )

        if not text.strip():
            stats.blocks_repaired = max(1, stats.blocks_repaired)
            stats.empty_blocks_removed = 1
            return None, stats

        if self.config.normalize_prose_spacing and block.type in _PROSE_BLOCK_TYPES:
            text, spacing = _normalize_prose_spacing(text)
            if spacing["extra_spaces_removed"] or spacing["cjk_spaces_removed"]:
                stats.blocks_repaired = 1
                stats.spacing_blocks_repaired = 1
                stats.extra_spaces_removed = spacing["extra_spaces_removed"]
                stats.cjk_spaces_removed = spacing["cjk_spaces_removed"]
                repairs.append({"code": "prose_spacing_normalized", **spacing})

        if (
            self.config.deduplicate_repeated_sentence_sequences
            and block.type in _PROSE_BLOCK_TYPES
        ):
            text, deduplication = _deduplicate_repeated_sentence_sequences(text, self.config)
            if deduplication["removed_sequences"]:
                stats.blocks_repaired = 1
                stats.duplicate_sentence_sequences_removed = deduplication["removed_sequences"]
                stats.duplicate_sentences_removed = deduplication["removed_sentences"]
                stats.duplicate_chars_removed = deduplication["removed_chars"]
                repairs.append(
                    {
                        "code": "repeated_sentence_sequences_removed",
                        **deduplication,
                    }
                )

        metadata = dict(block.metadata)
        if repairs:
            existing = metadata.get("repairs", [])
            metadata["repairs"] = list(existing) + repairs if isinstance(existing, list) else repairs
        return replace(block, text=text.strip(), metadata=metadata), stats


def _strip_boundary_artifacts(text: str) -> Tuple[str, List[str]]:
    value = text
    artifacts: List[str] = []
    while True:
        updated = _EDGE_FORMAT_CHARS_RE.sub("", value)
        if updated != value:
            artifacts.append("unicode_boundary_format_character")
            value = updated
            continue

        leading = _LEADING_HTML_ARTIFACT_RE.match(value)
        if leading is not None:
            artifacts.append(leading.group(0).strip())
            value = value[leading.end():]
            continue

        trailing = _TRAILING_HTML_ARTIFACT_RE.search(value)
        if trailing is not None:
            artifacts.append(trailing.group(0).strip())
            value = value[:trailing.start()]
            continue
        break
    return value.strip(), artifacts


def _normalize_prose_spacing(text: str) -> Tuple[str, Dict[str, int]]:
    extra_spaces_removed = 0
    cjk_spaces_removed = 0

    def collapse_horizontal_spaces(match: re.Match[str]) -> str:
        nonlocal extra_spaces_removed
        extra_spaces_removed += len(match.group(0)) - 1
        return " "

    value = _MULTIPLE_HORIZONTAL_SPACES_RE.sub(collapse_horizontal_spaces, text)

    for pattern in (
        _CJK_INTERNAL_SPACE_RE,
        _CJK_BEFORE_PUNCTUATION_SPACE_RE,
        _PUNCTUATION_BEFORE_CJK_SPACE_RE,
    ):
        value, replacements = pattern.subn("", value)
        cjk_spaces_removed += replacements

    return value, {
        "extra_spaces_removed": extra_spaces_removed,
        "cjk_spaces_removed": cjk_spaces_removed,
    }


def _deduplicate_repeated_sentence_sequences(
    text: str,
    config: NormalizationConfig,
) -> Tuple[str, Dict[str, int]]:
    sentences = [part for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) < 2:
        return text, {"removed_sequences": 0, "removed_sentences": 0, "removed_chars": 0}

    keys = [_canonical_text(sentence) for sentence in sentences]
    output: List[str] = []
    removed_sequences = 0
    removed_sentences = 0
    removed_chars = 0
    index = 0

    while index < len(sentences):
        max_width = min(config.max_duplicate_sequence_sentences, (len(sentences) - index) // 2)
        repeated_width = 0
        for width in range(max_width, 0, -1):
            first = keys[index:index + width]
            second = keys[index + width:index + width * 2]
            if first != second or not all(first):
                continue
            copy_chars = sum(_meaningful_char_count(value) for value in sentences[index:index + width])
            minimum = (
                config.min_duplicate_sentence_chars
                if width == 1
                else config.min_duplicate_sequence_chars
            )
            if copy_chars >= minimum:
                repeated_width = width
                break

        if not repeated_width:
            output.append(sentences[index])
            index += 1
            continue

        output.extend(sentences[index:index + repeated_width])
        repeated_key = keys[index:index + repeated_width]
        cursor = index + repeated_width
        copies_removed = 0
        while (
            cursor + repeated_width <= len(sentences)
            and keys[cursor:cursor + repeated_width] == repeated_key
        ):
            removed = sentences[cursor:cursor + repeated_width]
            removed_sentences += repeated_width
            removed_chars += sum(_meaningful_char_count(value) for value in removed)
            copies_removed += 1
            cursor += repeated_width
        if copies_removed:
            removed_sequences += 1
        index = cursor

    return "".join(output).strip(), {
        "removed_sequences": removed_sequences,
        "removed_sentences": removed_sentences,
        "removed_chars": removed_chars,
    }


def _are_duplicate_adjacent_blocks(
    left: Block,
    right: Block,
    config: NormalizationConfig,
) -> bool:
    if left.type not in _PROSE_BLOCK_TYPES or right.type != left.type:
        return False
    if left.heading_path != right.heading_path:
        return False
    if _meaningful_char_count(right.text) < config.min_duplicate_sentence_chars:
        return False
    return _canonical_text(left.text) == _canonical_text(right.text)


def _append_repair(block: Block, repair: Dict[str, Any]) -> Block:
    metadata = dict(block.metadata)
    existing = metadata.get("repairs", [])
    metadata["repairs"] = list(existing) + [repair] if isinstance(existing, list) else [repair]
    return replace(block, metadata=metadata)


def _canonical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", normalized)


def _meaningful_char_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace())
