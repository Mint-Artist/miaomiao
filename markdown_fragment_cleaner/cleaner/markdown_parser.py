from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .config import ContentPolicy
from .models import Block, Document


_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]}>]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CLOSING_PUNCTUATION = set("，。！？；：、,.!?;:)]}）》」』】")


@dataclass
class _Node:
    token: Optional[Token]
    children: List["_Node"] = field(default_factory=list)


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


class MarkdownDocumentParser:
    """Parse CommonMark/GFM blocks while retaining source line spans."""

    def __init__(self, policy: Optional[ContentPolicy] = None) -> None:
        self.policy = policy or ContentPolicy()
        self.md = MarkdownIt("commonmark", {"html": True})
        self.md.enable("table")
        self.md.enable("strikethrough")

    def parse(
        self,
        markdown: str,
        doc_id: str,
        url: str = "",
        title: str = "",
        source_row: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        source = markdown.replace("\r\n", "\n").replace("\r", "\n")
        parse_source = _mask_front_matter(source)
        lines = source.split("\n")
        root = _build_tree(self.md.parse(parse_source))

        blocks: List[Block] = []
        heading_stack: List[Tuple[int, str]] = []
        first_heading = ""
        first_h1 = ""

        for node in root.children:
            token_type = node.token.type if node.token else ""
            if token_type == "heading_open":
                level = int(node.token.tag[1:])
                heading_text = _render_inline_descendants(node, self.policy).strip()
                if heading_text:
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, heading_text))
                    if not first_heading:
                        first_heading = heading_text
                    if level == 1 and not first_h1:
                        first_h1 = heading_text
                continue

            converted = self._convert_block_node(
                node=node,
                lines=lines,
                doc_id=doc_id,
                ordinal=len(blocks) + 1,
                heading_path=[value for _, value in heading_stack],
            )
            if converted is not None and converted.text.strip():
                blocks.append(converted)

        return Document(
            doc_id=doc_id,
            url=url,
            title=title.strip() or first_h1 or first_heading or doc_id,
            blocks=blocks,
            source_row=source_row,
            metadata=dict(metadata or {}),
        )

    def _convert_block_node(
        self,
        node: _Node,
        lines: Sequence[str],
        doc_id: str,
        ordinal: int,
        heading_path: List[str],
    ) -> Optional[Block]:
        token = node.token
        if token is None:
            return None
        token_type = token.type
        block_type = ""
        text = ""

        if token_type == "paragraph_open":
            block_type = "paragraph"
            text = _render_inline_descendants(node, self.policy)
        elif token_type in {"bullet_list_open", "ordered_list_open"}:
            block_type = "list"
            text = _render_list(node, self.policy)
        elif token_type == "blockquote_open":
            block_type = "blockquote"
            visible = _render_container(node, self.policy)
            text = "\n".join("> " + line if line else ">" for line in visible.splitlines())
        elif token_type == "table_open":
            block_type = "table"
            text = _render_table(node, self.policy)
        elif token_type in {"fence", "code_block"}:
            block_type = "code"
            info = (token.info or "").strip()
            fence = "```" + info
            text = "%s\n%s\n```" % (fence, token.content.rstrip("\n"))
        elif token_type == "html_block":
            block_type = "html"
            text = _visible_html(token.content)
        elif token_type in {"hr"}:
            return None
        else:
            # Unknown top-level extensions are kept only when they expose
            # meaningful inline text. This avoids silently losing content.
            text = _render_container(node, self.policy).strip()
            if not text:
                return None
            block_type = "other"

        start, end = _source_map(node)
        raw_text = "\n".join(lines[start:end]).strip() if end > start else token.content.strip()
        metadata = _collect_metadata(node, raw_text)
        metadata["ast_token_type"] = token_type
        return Block(
            block_id="%s-b%06d" % (doc_id, ordinal),
            type=block_type,
            text=_normalize_rendered_text(text),
            raw_text=raw_text,
            heading_path=list(heading_path),
            start_line=start + 1,
            end_line=max(start + 1, end),
            metadata=metadata,
        )


def _build_tree(tokens: Sequence[Token]) -> _Node:
    root = _Node(None)
    stack = [root]
    for token in tokens:
        if token.nesting == -1:
            if len(stack) > 1:
                stack.pop()
            continue
        node = _Node(token)
        stack[-1].children.append(node)
        if token.nesting == 1:
            stack.append(node)
    return root


def _mask_front_matter(source: str) -> str:
    lines = source.split("\n")
    if not lines or lines[0].strip() not in {"---", "+++"}:
        return source
    marker = lines[0].strip()
    closing = next((i for i in range(1, min(len(lines), 300)) if lines[i].strip() == marker), None)
    if closing is None:
        return source
    masked = list(lines)
    for index in range(closing + 1):
        masked[index] = ""
    return "\n".join(masked)


def _source_map(node: _Node) -> Tuple[int, int]:
    if node.token and node.token.map:
        return int(node.token.map[0]), int(node.token.map[1])
    maps = [_source_map(child) for child in node.children]
    maps = [value for value in maps if value[1] > value[0]]
    if not maps:
        return 0, 0
    return min(value[0] for value in maps), max(value[1] for value in maps)


def _render_inline_descendants(node: _Node, policy: ContentPolicy) -> str:
    inline_nodes = [child for child in _walk(node) if child.token and child.token.type == "inline"]
    return "\n".join(_render_inline(node.token, policy) for node in inline_nodes).strip()


def _render_inline(token: Token, policy: ContentPolicy) -> str:
    children = token.children or []
    output: List[str] = []
    for index, child in enumerate(children):
        child_type = child.type
        if child_type == "text":
            output.append(child.content)
        elif child_type == "code_inline":
            if policy.preserve_inline_code_markers:
                output.append("`%s`" % child.content.replace("`", "\\`"))
            else:
                output.append(child.content)
        elif child_type == "image":
            if policy.keep_image_alt_text and child.content.strip():
                output.append(child.content.strip())
        elif child_type == "softbreak":
            previous = _last_visible_character(output)
            following = _next_visible_character(children, index + 1)
            if previous and following and _CJK_RE.match(previous) and (
                _CJK_RE.match(following) or following in _CLOSING_PUNCTUATION
            ):
                output.append("")
            else:
                output.append(" ")
        elif child_type == "hardbreak":
            output.append("\n")
        elif child_type == "html_inline":
            output.append(_visible_html(child.content))
        # link/emphasis/strike open/close tokens deliberately emit nothing;
        # their visible child text is emitted normally.
    return html.unescape("".join(output)).strip()


def _render_list(node: _Node, policy: ContentPolicy, depth: int = 0) -> str:
    ordered = bool(node.token and node.token.type == "ordered_list_open")
    start = 1
    if node.token and ordered:
        start_value = node.token.attrGet("start")
        if start_value and start_value.isdigit():
            start = int(start_value)
    rendered: List[str] = []
    item_number = start
    for child in node.children:
        if not child.token or child.token.type != "list_item_open":
            continue
        direct_parts: List[str] = []
        nested_parts: List[str] = []
        for item_child in child.children:
            item_type = item_child.token.type if item_child.token else ""
            if item_type == "paragraph_open":
                value = _render_inline_descendants(item_child, policy)
                if value:
                    direct_parts.append(value)
            elif item_type in {"bullet_list_open", "ordered_list_open"}:
                value = _render_list(item_child, policy, depth + 1)
                if value:
                    nested_parts.append(value)
            else:
                value = _render_container(item_child, policy)
                if value:
                    direct_parts.append(value)
        prefix = "%d. " % item_number if ordered else "- "
        item_number += 1
        body = "\n\n".join(direct_parts).strip()
        body_lines = body.splitlines() or [""]
        rendered.append(prefix + body_lines[0])
        continuation_indent = " " * len(prefix)
        rendered.extend(continuation_indent + line for line in body_lines[1:])
        for nested in nested_parts:
            rendered.extend("  " + line for line in nested.splitlines())
    return "\n".join(rendered).strip()


def _render_table(node: _Node, policy: ContentPolicy) -> str:
    rows: List[List[str]] = []
    header_rows = 0
    for row in _find_nodes(node, "tr_open"):
        cells: List[str] = []
        is_header = False
        for cell in row.children:
            cell_type = cell.token.type if cell.token else ""
            if cell_type not in {"th_open", "td_open"}:
                continue
            is_header = is_header or cell_type == "th_open"
            value = _render_inline_descendants(cell, policy).replace("|", "\\|")
            cells.append(re.sub(r"\s*\n\s*", " ", value).strip())
        if cells:
            rows.append(cells)
            if is_header:
                header_rows += 1
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if header_rows == 0:
        rows.insert(0, ["字段%d" % (index + 1) for index in range(width)])
    output = ["| " + " | ".join(rows[0]) + " |"]
    output.append("| " + " | ".join(["---"] * width) + " |")
    output.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(output)


def _render_container(node: _Node, policy: ContentPolicy) -> str:
    token_type = node.token.type if node.token else ""
    if token_type == "paragraph_open":
        return _render_inline_descendants(node, policy)
    if token_type in {"bullet_list_open", "ordered_list_open"}:
        return _render_list(node, policy)
    if token_type == "table_open":
        return _render_table(node, policy)
    if token_type == "html_block" and node.token:
        return _visible_html(node.token.content)
    if token_type in {"fence", "code_block"} and node.token:
        return node.token.content.strip()
    parts = [_render_container(child, policy) for child in node.children]
    if node.token and node.token.type == "inline":
        return _render_inline(node.token, policy)
    return "\n\n".join(value for value in parts if value).strip()


def _visible_html(value: str) -> str:
    parser = _VisibleHTMLParser()
    try:
        parser.feed(value)
        parser.close()
        return parser.text()
    except Exception:
        return re.sub(r"<[^>]+>", "", value).strip()


def _normalize_rendered_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _walk(node: _Node) -> Iterable[_Node]:
    for child in node.children:
        yield child
        yield from _walk(child)


def _find_nodes(node: _Node, token_type: str) -> Iterable[_Node]:
    for child in _walk(node):
        if child.token and child.token.type == token_type:
            yield child


def _collect_metadata(node: _Node, raw_text: str) -> Dict[str, Any]:
    links = 0
    images = 0
    inline_code = 0
    for child in _walk(node):
        tokens: Iterable[Token]
        if child.token and child.token.type == "inline":
            tokens = child.token.children or []
        elif child.token:
            tokens = [child.token]
        else:
            tokens = []
        for token in tokens:
            links += int(token.type == "link_open")
            images += int(token.type == "image")
            inline_code += int(token.type == "code_inline")
    url_chars = sum(len(match.group(0)) for match in _URL_RE.finditer(raw_text))
    return {
        "link_count": links,
        "image_count": images,
        "inline_code_count": inline_code,
        "url_char_count": url_chars,
        "raw_char_count": len(raw_text),
    }


def _last_visible_character(parts: Sequence[str]) -> str:
    for part in reversed(parts):
        if part:
            return part[-1]
    return ""


def _next_visible_character(children: Sequence[Token], start: int) -> str:
    for child in children[start:]:
        if child.type in {"text", "code_inline"} and child.content:
            return child.content[0]
        if child.type == "image" and child.content:
            return child.content[0]
    return ""
