from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from cleaner.config import AssemblyConfig, ContentPolicy, RuleConfig
from cleaner.models import Block, Fragment, RuleFlag, WholeDocumentRecord
from cleaner.tokenization import ApproxTokenCounter, TokenCounter

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]}>]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_MARKDOWN_ARTIFACT_RE = re.compile(r"(?:<[^>]{1,200}>|\]\([^)]{0,500}\)|!\[[^]]*\]|`{3,}|#{2,})")
_TERMINAL_RE = re.compile(r"[。！？!?；;…）)】》」』”’\"']\s*$")
_INCOMPLETE_END_RE = re.compile(r"(?:[:：,，、]|如下(?:所示)?[：:]?|分别为[：:]?|包括(?:以下)?[：:]?)\s*$")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_HEADING_PUNCTUATION_RE = re.compile(r"[，,：:]")
MINIMUM_NAVIGATION_LINKS = 3
MINIMUM_NAVIGATION_LINK_RATIO = 0.6
MAXIMUM_NAVIGATION_LINK_CHARS = 50
MINIMUM_VISIBLE_HTML_CHARS = 10
MINIMUM_TERMINAL_PROSE_CHARS = 40
MINIMUM_LANGUAGE_CONTENT_CHARS = 30
MINIMUM_DUPLICATE_LINES = 3
UNCLOSED_BRACKET_DIFFERENCE = 2
MAXIMUM_STANDALONE_HEADING_TOKENS = 30
_BOILERPLATE_PATTERNS = (
    re.compile(r"^(?:登录|注册|退出登录|返回首页|网站地图|联系我们|加入收藏|设为首页)(?:\s*[|·/｜]\s*.*)?$"),
    re.compile(r"^(?:上一篇|下一篇|上一页|下一页|返回顶部|相关推荐|相关(?:文章|内容|链接))(?:\s*[:：].*)?$"),
    re.compile(r"^(?:分享至?|分享到|打印本页|关闭窗口|字体(?:大小)?[：:]?).{0,100}$"),
    re.compile(r"^(?:更多|相关)(?:资料|内容|信息|文章)(?:请)?(?:访问|参见|点击).{0,160}$"),
    re.compile(r"^(?:copyright\s*)?[©ⓒ]\s*\d{0,4}.*(?:all rights reserved|版权所有).*$", re.IGNORECASE),
    re.compile(r"^(?:copyright\b|版权所有\b|[©ⓒ]\s*(?:19|20)\d{2}).{0,250}$", re.IGNORECASE),
    re.compile(r"^(?:cookie|隐私政策|使用条款|免责声明)(?:\s*(?:policy|设置|同意|接受))?$", re.IGNORECASE),
    re.compile(r"^(?:作者|来源|发布时间|发布日期|编辑)[：:]?\s*.{0,80}$"),
)


@dataclass
class RuleResult:
    decision: str
    flags: List[RuleFlag]


class RuleEngine:
    """High-precision rejects plus conservative review flags."""

    def __init__(
        self,
        config: Optional[RuleConfig] = None,
        content_policy: Optional[ContentPolicy] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.config = config or RuleConfig()
        self.policy = content_policy or ContentPolicy()
        self.counter = token_counter or ApproxTokenCounter()

    def evaluate_block(self, block: Block) -> RuleResult:
        flags = self._text_flags(block.text, block.raw_text)
        if not self._type_is_enabled(block.type):
            flags.append(RuleFlag("excluded_block_type", "hard", "block type disabled by content policy"))

        normalized = re.sub(r"\s+", " ", block.text).strip()
        if any(pattern.fullmatch(normalized) for pattern in _BOILERPLATE_PATTERNS):
            flags.append(RuleFlag("known_boilerplate", "hard", "matched a conservative boilerplate pattern"))

        link_count = int(block.metadata.get("link_count", 0))
        line_count = max(1, len([line for line in block.text.splitlines() if line.strip()]))
        if _is_navigation_link_list(link_count, line_count, len(normalized)):
            flags.append(RuleFlag("navigation_link_list", "hard", "short link-dense block resembles navigation"))

        if block.type == "html" and len(normalized) < MINIMUM_VISIBLE_HTML_CHARS:
            flags.append(RuleFlag("empty_html", "hard", "HTML block has almost no visible content"))
        return RuleResult(_decision(flags), _deduplicate(flags))

    def evaluate_fragment(self, fragment: Fragment) -> RuleResult:
        flags = list(fragment.flags)
        flags.extend(self._text_flags(fragment.content, fragment.content))
        tokens = fragment.token_count or self.counter.count(fragment.content)
        structured_only = all(kind in {"list", "table", "blockquote"} for kind in fragment.block_types)
        hard_minimum = self.config.structured_hard_min_tokens if structured_only else self.config.hard_min_tokens

        if tokens < hard_minimum:
            flags.append(RuleFlag("too_short", "hard", "below hard token minimum", float(tokens)))
        elif tokens < self.config.soft_min_tokens:
            flags.append(RuleFlag("short_fragment", "soft", "below preferred token minimum", float(tokens)))
        if tokens > self.config.hard_max_tokens:
            flags.append(RuleFlag("too_long", "hard", "above hard token maximum", float(tokens)))

        stripped = fragment.content.strip()
        if stripped.startswith(self.config.context_dependent_prefixes):
            flags.append(RuleFlag("context_dependent_start", "soft", "likely depends on preceding context"))
        if _INCOMPLETE_END_RE.search(stripped):
            flags.append(RuleFlag("incomplete_end", "soft", "ends with an introducer or non-terminal punctuation"))
        elif _has_non_terminal_prose_end(stripped, fragment.block_types):
            flags.append(RuleFlag("non_terminal_end", "soft", "prose does not end with terminal punctuation"))
        if _has_obvious_unclosed_brackets(stripped):
            flags.append(RuleFlag("unclosed_brackets", "soft", "bracket counts suggest truncation"))
        if _looks_like_standalone_heading(stripped, tokens, fragment.block_types):
            flags.append(RuleFlag("heading_like_fragment", "soft", "short non-sentential fragment resembles a title"))
        return RuleResult(_decision(flags), _deduplicate(flags))

    def evaluate_document(
        self,
        document: WholeDocumentRecord,
        assembly: AssemblyConfig,
    ) -> RuleResult:
        flags = list(document.flags)
        flags.extend(self._text_flags(document.content, document.content))
        tokens = document.token_count or self.counter.count(document.content)

        if tokens < assembly.document_min_tokens:
            flags.append(
                RuleFlag("too_short", "hard", "whole document is below token minimum", float(tokens))
            )
        if tokens > assembly.document_max_tokens:
            if assembly.document_over_max_policy == "review":
                flags.append(
                    RuleFlag(
                        "whole_document_too_long",
                        "soft",
                        "whole document exceeds preferred token maximum and was not truncated",
                        float(tokens),
                    )
                )
            elif assembly.document_over_max_policy == "reject":
                flags.append(
                    RuleFlag(
                        "whole_document_too_long",
                        "hard",
                        "whole document exceeds hard token maximum",
                        float(tokens),
                    )
                )

        stripped = document.content.strip()
        if stripped.startswith(self.config.context_dependent_prefixes):
            flags.append(RuleFlag("context_dependent_start", "soft", "likely depends on preceding context"))
        if _INCOMPLETE_END_RE.search(stripped):
            flags.append(RuleFlag("incomplete_end", "soft", "ends with an introducer or non-terminal punctuation"))
        elif _has_non_terminal_prose_end(stripped, document.block_types):
            flags.append(RuleFlag("non_terminal_end", "soft", "prose does not end with terminal punctuation"))
        if _has_obvious_unclosed_brackets(stripped):
            flags.append(RuleFlag("unclosed_brackets", "soft", "bracket counts suggest truncation"))
        return RuleResult(_decision(flags), _deduplicate(flags))

    def _type_is_enabled(self, block_type: str) -> bool:
        mapping = {
            "paragraph": self.policy.keep_paragraphs,
            "list": self.policy.keep_lists,
            "blockquote": self.policy.keep_blockquotes,
            "table": self.policy.keep_tables,
            "code": self.policy.keep_code_blocks,
            "html": self.policy.keep_html_visible_text,
            "other": False,
        }
        return mapping.get(block_type, False)

    def _text_flags(self, text: str, raw_text: str) -> List[RuleFlag]:
        stripped = text.strip()
        if not stripped:
            return [RuleFlag("empty", "hard", "empty visible text")]
        flags: List[RuleFlag] = []
        length = max(1, len(stripped))

        replacement_count = stripped.count("\ufffd")
        replacement_ratio = replacement_count / length
        if replacement_count > self.config.max_replacement_chars or replacement_ratio > self.config.max_replacement_ratio:
            flags.append(RuleFlag("replacement_characters", "hard", "too many U+FFFD characters", replacement_ratio))

        control_count = _count_disallowed_control_characters(stripped)
        control_ratio = control_count / length
        if control_ratio > self.config.max_control_ratio:
            flags.append(RuleFlag("control_characters", "hard", "too many control characters", control_ratio))

        # Link destinations are intentionally removed by the AST renderer.
        # Only URLs surviving in visible output should influence quality.
        url_chars = sum(len(match.group(0)) for match in _URL_RE.finditer(stripped))
        url_ratio = url_chars / length
        if url_ratio > self.config.max_url_ratio:
            flags.append(RuleFlag("url_heavy", "hard", "URLs dominate source text", url_ratio))

        # Valid Markdown syntax is removed by the AST renderer. Only syntax
        # that survives into output indicates an actual parsing artifact.
        artifact_chars = sum(len(match.group(0)) for match in _MARKDOWN_ARTIFACT_RE.finditer(stripped))
        artifact_ratio = artifact_chars / length
        if artifact_ratio > self.config.max_markdown_artifact_ratio:
            flags.append(RuleFlag("markdown_artifact_heavy", "hard", "unparsed markup artifacts dominate output", artifact_ratio))

        duplicate_ratio = _duplicate_line_ratio(stripped)
        if duplicate_ratio > self.config.max_duplicate_line_ratio:
            flags.append(RuleFlag("duplicate_lines", "hard", "duplicate lines dominate text", duplicate_ratio))

        if re.search(r"(.)\1{%d,}" % (self.config.repeated_character_run - 1), stripped):
            flags.append(RuleFlag("repeated_characters", "hard", "long repeated-character run"))

        repeated_ngram_ratio = _repeated_ngram_ratio(stripped)
        if repeated_ngram_ratio > self.config.max_repeated_ngram_ratio:
            flags.append(RuleFlag("repeated_ngrams", "hard", "repeated five-grams dominate text", repeated_ngram_ratio))

        visible = sum(1 for char in stripped if not char.isspace())
        cjk_count = len(_CJK_RE.findall(stripped))
        if _lacks_language_content(stripped, visible, cjk_count):
            flags.append(RuleFlag("no_language_content", "hard", "no meaningful CJK or Latin word content"))
        return flags


def _decision(flags: Sequence[RuleFlag]) -> str:
    if any(flag.severity == "hard" for flag in flags):
        return "rejected"
    if any(flag.severity == "soft" for flag in flags):
        return "review"
    return "accepted"


def _count_disallowed_control_characters(text: str) -> int:
    count = 0
    for char in text:
        if unicodedata.category(char) != "Cc":
            continue
        if char in {"\n", "\r", "\t"}:
            continue
        count += 1
    return count


def _is_navigation_link_list(
    link_count: int,
    line_count: int,
    text_length: int,
) -> bool:
    if link_count < MINIMUM_NAVIGATION_LINKS:
        return False
    if link_count < line_count * MINIMUM_NAVIGATION_LINK_RATIO:
        return False
    return text_length / link_count < MAXIMUM_NAVIGATION_LINK_CHARS


def _has_non_terminal_prose_end(text: str, block_types: Sequence[str]) -> bool:
    if not block_types:
        return False
    if block_types[-1] not in {"paragraph", "html"}:
        return False
    if len(text) < MINIMUM_TERMINAL_PROSE_CHARS:
        return False
    return _TERMINAL_RE.search(text) is None


def _lacks_language_content(text: str, visible: int, cjk_count: int) -> bool:
    if visible < MINIMUM_LANGUAGE_CONTENT_CHARS:
        return False
    if cjk_count:
        return False
    return _LATIN_WORD_RE.search(text) is None


def _deduplicate(flags: Iterable[RuleFlag]) -> List[RuleFlag]:
    result: List[RuleFlag] = []
    seen = set()
    for flag in flags:
        key = (flag.code, flag.severity, flag.detail)
        if key not in seen:
            result.append(flag)
            seen.add(key)
    return result


def _duplicate_line_ratio(text: str) -> float:
    lines = [re.sub(r"\s+", " ", line).strip().lower() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < MINIMUM_DUPLICATE_LINES:
        return 0.0
    return (len(lines) - len(set(lines))) / len(lines)


def _repeated_ngram_ratio(text: str, width: int = 5) -> float:
    characters = [char for char in text if not char.isspace()]
    if len(characters) < width * 4:
        return 0.0
    ngrams = ["".join(characters[index:index + width]) for index in range(len(characters) - width + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def _has_obvious_unclosed_brackets(text: str) -> bool:
    pairs: Tuple[Tuple[str, str], ...] = (
        ("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"), ("《", "》"), ("“", "”"),
    )
    return any(
        abs(text.count(left) - text.count(right)) >= UNCLOSED_BRACKET_DIFFERENCE
        for left, right in pairs
    )


def _looks_like_standalone_heading(text: str, tokens: int, block_types: Sequence[str]) -> bool:
    if tokens > MAXIMUM_STANDALONE_HEADING_TOKENS:
        return False
    if "\n" in text:
        return False
    if any(kind in {"list", "table"} for kind in block_types):
        return False
    if _TERMINAL_RE.search(text):
        return False
    return _HEADING_PUNCTUATION_RE.search(text) is None
