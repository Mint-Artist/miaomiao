from __future__ import annotations

import re
from typing import List, Sequence

from cleaner.config import NormalizationConfig

_KEY_VALUE_RE = re.compile(r"^\s*([^：:\n]{1,20})[：:]\s*\S")
_INVALID_KEY_RE = re.compile(r"[，。！？；,.!?/\\]")
_STEP_RE = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百\d]+(?:步|章|节|项)[：:]?"
    r"|[（(]?[一二三四五六七八九十百\d]+[）).、]"
    r")\s*\S"
)
_INTRO_RE = re.compile(
    r"(?:如下(?:所示)?|包括(?:以下)?|分为(?:以下)?|主要有|步骤为|内容为)[：:]\s*$"
)
_TERMINAL_RE = re.compile(r"[。！？!?；;…）)】》」』”’\"']\s*$")
MINIMUM_STRUCTURED_LINE_RATIO = 0.60


class SoftbreakNormalizer:
    """Preserve semantic line structure while unwrapping visual line wrapping."""

    def __init__(self, config: NormalizationConfig) -> None:
        self.policy = config.softbreak_policy
        self.min_structured_lines = config.min_structured_softbreak_lines
        self.preserve_complete_sentence_lines = config.preserve_complete_sentence_lines

    def join(self, lines: Sequence[str]) -> str:
        if len(lines) <= 1:
            return lines[0].strip() if lines else ""

        cleaned = [line.strip(" \t") for line in lines]
        if self.policy == "preserve":
            return "\n".join(cleaned).strip()
        if self.policy == "smart" and self._looks_structured(cleaned):
            return "\n".join(cleaned).strip()
        return _join_unwrapped(cleaned)

    def _looks_structured(self, lines: Sequence[str]) -> bool:
        nonempty = [line for line in lines if line.strip()]
        if len(nonempty) < self.min_structured_lines:
            return False

        key_value_lines = sum(_looks_like_key_value(line) for line in nonempty)
        key_value_ratio = key_value_lines / len(nonempty)
        if (
            key_value_lines >= self.min_structured_lines
            and key_value_ratio >= MINIMUM_STRUCTURED_LINE_RATIO
        ):
            return True

        step_lines = sum(bool(_STEP_RE.match(line)) for line in nonempty)
        step_ratio = step_lines / len(nonempty)
        if (
            step_lines >= self.min_structured_lines
            and step_ratio >= MINIMUM_STRUCTURED_LINE_RATIO
        ):
            return True

        if _INTRO_RE.search(nonempty[0]) and len(nonempty) >= self.min_structured_lines:
            return True

        if self.preserve_complete_sentence_lines and all(_TERMINAL_RE.search(line) for line in nonempty):
            return True
        return False


def _looks_like_key_value(line: str) -> bool:
    match = _KEY_VALUE_RE.match(line)
    if match is None:
        return False
    key = match.group(1).strip()
    return bool(key) and _INVALID_KEY_RE.search(key) is None


def _join_unwrapped(lines: Sequence[str]) -> str:
    nonempty: List[str] = [line for line in lines if line]
    if not nonempty:
        return ""
    value = nonempty[0]
    for line in nonempty[1:]:
        previous = _last_visible_character(value)
        following = _first_visible_character(line)
        separator = " " if previous and following and previous.isascii() and following.isascii() else ""
        value += separator + line
    return value.strip()


def _last_visible_character(value: str) -> str:
    return next((character for character in reversed(value) if not character.isspace()), "")


def _first_visible_character(value: str) -> str:
    return next((character for character in value if not character.isspace()), "")
