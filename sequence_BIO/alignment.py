from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    ALIGNMENT_ALGORITHM,
    DEFAULT_MAX_ADJUST_CHARS,
    DEFAULT_MIN_MATCH_CHARS,
    INDEX_CONVENTION,
)

CharSpan = Tuple[int, int]


_REFINED_TEXT_RE = re.compile(
    r"refined_text\s*:\s*\[doc\](.*?)\[/doc\]",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class MatchSpan:
    """One exact character match, using zero-based half-open offsets."""

    source_start: int
    source_end: int
    target_start: int
    target_end: int

    @property
    def length(self) -> int:
        return self.source_end - self.source_start

    def to_dict(self, source_text: Optional[str] = None) -> Dict[str, object]:
        value: Dict[str, object] = {
            "source_span": [self.source_start, self.source_end],
            "target_span": [self.target_start, self.target_end],
            "length": self.length,
        }
        if source_text is not None:
            value["text"] = source_text[self.source_start : self.source_end]
        return value


@dataclass(frozen=True)
class GapAdjustment:
    """A small teacher modification repaired with the original source gap."""

    source_start: int
    source_end: int
    target_start: int
    target_end: int
    source_text: str
    target_text: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_span": [self.source_start, self.source_end],
            "target_span": [self.target_start, self.target_end],
            "source_gap_length": self.source_end - self.source_start,
            "target_gap_length": self.target_end - self.target_start,
            "source_text": self.source_text,
            "target_text": self.target_text,
        }


@dataclass(frozen=True)
class AlignmentResult:
    """Paper-style alignment and verification result."""

    status: str
    source_text: str
    refined_text: str
    adjusted_refined_text: Optional[str]
    matched_spans: Tuple[MatchSpan, ...]
    character_spans: Tuple[CharSpan, ...]
    adjustments: Tuple[GapAdjustment, ...]
    matched_characters: int
    target_coverage: float
    min_match_chars: int
    max_adjust_chars: int
    error: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        return self.status in {"aligned", "adjusted"}

    @property
    def is_aligned(self) -> bool:
        """Backward-compatible alias meaning accepted for BIO assignment."""

        return self.is_accepted

    def to_dict(
        self,
        *,
        include_text: bool = False,
        include_match_text: bool = True,
    ) -> Dict[str, object]:
        value: Dict[str, object] = {
            "status": self.status,
            "algorithm": ALIGNMENT_ALGORITHM,
            "index_convention": INDEX_CONVENTION,
            "min_match_chars": self.min_match_chars,
            "max_adjust_chars": self.max_adjust_chars,
            "target_coverage": self.target_coverage,
            "matched_characters": self.matched_characters,
            "matched_spans": [
                item.to_dict(self.source_text if include_match_text else None)
                for item in self.matched_spans
            ],
            "retained_source_spans": [list(span) for span in self.character_spans],
            "adjustments": [item.to_dict() for item in self.adjustments],
            "error": self.error,
        }
        if include_text:
            value["source_text"] = self.source_text
            value["teacher_refined_text"] = self.refined_text
            value["adjusted_refined_text"] = self.adjusted_refined_text
        return value


def extract_refined_text(response: str) -> str:
    """Extract the paper-style ``refined_text: [doc]...[/doc]`` payload."""

    matches = list(_REFINED_TEXT_RE.finditer(response))
    if not matches:
        raise ValueError("teacher response does not contain a refined_text [doc] field")
    if len(matches) > 1:
        raise ValueError("teacher response contains multiple refined_text [doc] fields")
    return matches[0].group(1)


def find_longest_match_segments(
    source_text: str,
    target_text: str,
    *,
    min_match_chars: int = DEFAULT_MIN_MATCH_CHARS,
) -> Tuple[MatchSpan, ...]:
    """Implement SELECT Appendix B, Algorithm 1.

    The paper enumerates every source position whose first character matches the
    current target character, then chooses the longest exact continuation.  A
    qualifying continuation must begin with ``min_match_chars`` exact
    characters, so ``str.find`` over that prefix is an equivalent but much
    faster way to enumerate only candidates that can be retained.
    """

    if min_match_chars < 1:
        raise ValueError("min_match_chars must be at least 1")

    matches: List[MatchSpan] = []
    target_cursor = 0
    source_cursor = 0

    while target_cursor < len(target_text) and source_cursor < len(source_text):
        if len(target_text) - target_cursor < min_match_chars:
            break

        best_start, best_length = _find_best_match(
            source_text,
            target_text,
            source_cursor=source_cursor,
            target_cursor=target_cursor,
            min_match_chars=min_match_chars,
        )

        if best_length >= min_match_chars:
            matches.append(
                MatchSpan(
                    best_start,
                    best_start + best_length,
                    target_cursor,
                    target_cursor + best_length,
                )
            )
            target_cursor += best_length
            source_cursor = best_start + best_length
        else:
            target_cursor += 1

    return tuple(matches)


def _find_best_match(
    source_text: str,
    target_text: str,
    *,
    source_cursor: int,
    target_cursor: int,
    min_match_chars: int,
) -> Tuple[int, int]:
    prefix = target_text[target_cursor : target_cursor + min_match_chars]
    candidate = source_text.find(prefix, source_cursor)
    best_start = -1
    best_length = 0
    remaining_target_length = len(target_text) - target_cursor
    while candidate >= 0:
        match_length = _continuation_length(
            source_text,
            target_text,
            source_start=candidate,
            target_start=target_cursor,
            initial_length=min_match_chars,
        )
        if match_length > best_length:
            best_start, best_length = candidate, match_length
            if best_length == remaining_target_length:
                break
        candidate = source_text.find(prefix, candidate + 1)
    return best_start, best_length


def _continuation_length(
    source_text: str,
    target_text: str,
    *,
    source_start: int,
    target_start: int,
    initial_length: int,
) -> int:
    match_length = initial_length
    maximum = min(
        len(source_text) - source_start,
        len(target_text) - target_start,
    )
    while (
        match_length < maximum
        and source_text[source_start + match_length]
        == target_text[target_start + match_length]
    ):
        match_length += 1
    return match_length


def align_deletion_only(
    source_text: str,
    refined_text: str,
    *,
    min_match_chars: int = DEFAULT_MIN_MATCH_CHARS,
    max_adjust_chars: int = DEFAULT_MAX_ADJUST_CHARS,
) -> AlignmentResult:
    """Classify one pair as ``aligned``, ``adjusted``, or ``unaligned``.

    ``aligned`` samples have matches that completely cover the teacher target.
    ``adjusted`` samples contain only internal unmatched target gaps whose
    length differs from the corresponding source gap by at most
    ``max_adjust_chars``; those gaps are replaced with the source text and the
    surrounding retained spans are merged.  All other samples are rejected.
    """

    if min_match_chars < 1:
        raise ValueError("min_match_chars must be at least 1")
    if max_adjust_chars < 0:
        raise ValueError("max_adjust_chars must be non-negative")

    if refined_text == "":
        return AlignmentResult(
            status="aligned",
            source_text=source_text,
            refined_text=refined_text,
            adjusted_refined_text="",
            matched_spans=(),
            character_spans=(),
            adjustments=(),
            matched_characters=0,
            target_coverage=1.0,
            min_match_chars=min_match_chars,
            max_adjust_chars=max_adjust_chars,
        )

    matches = find_longest_match_segments(
        source_text,
        refined_text,
        min_match_chars=min_match_chars,
    )
    matched_characters = sum(item.length for item in matches)
    target_coverage = matched_characters / len(refined_text)

    if _matches_fully_cover_target(matches, len(refined_text)):
        character_spans = tuple(
            _merge_adjacent_spans(
                (item.source_start, item.source_end) for item in matches
            )
        )
        return AlignmentResult(
            status="aligned",
            source_text=source_text,
            refined_text=refined_text,
            adjusted_refined_text=refined_text,
            matched_spans=matches,
            character_spans=character_spans,
            adjustments=(),
            matched_characters=matched_characters,
            target_coverage=target_coverage,
            min_match_chars=min_match_chars,
            max_adjust_chars=max_adjust_chars,
        )

    adjusted = _adjust_internal_gaps(
        source_text,
        refined_text,
        matches,
        max_adjust_chars=max_adjust_chars,
    )
    if adjusted is not None:
        character_spans, adjusted_text, adjustments = adjusted
        return AlignmentResult(
            status="adjusted",
            source_text=source_text,
            refined_text=refined_text,
            adjusted_refined_text=adjusted_text,
            matched_spans=matches,
            character_spans=character_spans,
            adjustments=adjustments,
            matched_characters=matched_characters,
            target_coverage=target_coverage,
            min_match_chars=min_match_chars,
            max_adjust_chars=max_adjust_chars,
        )

    return AlignmentResult(
        status="unaligned",
        source_text=source_text,
        refined_text=refined_text,
        adjusted_refined_text=None,
        matched_spans=matches,
        character_spans=(),
        adjustments=(),
        matched_characters=matched_characters,
        target_coverage=target_coverage,
        min_match_chars=min_match_chars,
        max_adjust_chars=max_adjust_chars,
        error=(
            "matches neither fully cover the teacher target nor satisfy the "
            "paper's internal-gap adjustment rule"
        ),
    )


def _matches_fully_cover_target(
    matches: Sequence[MatchSpan],
    target_length: int,
) -> bool:
    if target_length == 0:
        return True
    if not matches:
        return False
    if matches[0].target_start != 0 or matches[-1].target_end != target_length:
        return False
    return all(
        left.target_end == right.target_start
        for left, right in zip(matches, matches[1:], strict=False)
    )


def _adjust_internal_gaps(
    source: str,
    target: str,
    matches: Sequence[MatchSpan],
    *,
    max_adjust_chars: int,
) -> Optional[Tuple[Tuple[CharSpan, ...], str, Tuple[GapAdjustment, ...]]]:
    # Appendix 2.2 only defines repair for gaps between adjacent matches.  An
    # unmatched target prefix or suffix therefore remains unaligned.
    if not matches:
        return None
    if matches[0].target_start != 0 or matches[-1].target_end != len(target):
        return None

    retained: List[CharSpan] = []
    adjustments: List[GapAdjustment] = []
    group_start = matches[0].source_start
    group_end = matches[0].source_end

    for previous, current in zip(matches, matches[1:], strict=False):
        valid, split_group, adjustment = _classify_internal_gap(
            source,
            target,
            previous,
            current,
            max_adjust_chars=max_adjust_chars,
        )
        if not valid:
            return None
        if split_group:
            retained.append((group_start, group_end))
            group_start = current.source_start
        group_end = current.source_end
        if adjustment is not None:
            adjustments.append(adjustment)

    retained.append((group_start, group_end))
    if not adjustments:
        return None

    character_spans = tuple(retained)
    adjusted_text = "".join(source[start:end] for start, end in character_spans)
    return character_spans, adjusted_text, tuple(adjustments)


def _classify_internal_gap(
    source: str,
    target: str,
    previous: MatchSpan,
    current: MatchSpan,
    *,
    max_adjust_chars: int,
) -> Tuple[bool, bool, Optional[GapAdjustment]]:
    source_start, source_end = previous.source_end, current.source_start
    target_start, target_end = previous.target_end, current.target_start
    source_length = source_end - source_start
    target_length = target_end - target_start
    if source_length < 0 or target_length < 0:
        return False, False, None
    if target_length == 0:
        return True, source_length > 0, None
    if abs(source_length - target_length) > max_adjust_chars:
        return False, False, None
    return (
        True,
        False,
        GapAdjustment(
            source_start,
            source_end,
            target_start,
            target_end,
            source[source_start:source_end],
            target[target_start:target_end],
        ),
    )


def _merge_adjacent_spans(spans: Iterable[CharSpan]) -> Iterable[CharSpan]:
    iterator = iter(spans)
    try:
        start, end = next(iterator)
    except StopIteration:
        return
    for next_start, next_end in iterator:
        if next_start == end:
            end = next_end
        else:
            yield start, end
            start, end = next_start, next_end
    yield start, end
