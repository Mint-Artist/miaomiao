#!/usr/bin/env python3
"""EC Markdown 正文 → chunk 列表（结构保护优先，256 字符软目标）。

输入：jsonl，每行一个对象，`markdown` 键是 EC 输出的 Markdown 正文。
      表格沿用 HTML（<table>，可含 rowspan/colspan）；也兼容 Markdown 管道表。
输出：原对象原样保留，新增一个键（默认 `chunks`），值是切好的纯文本字符串列表。

切分规则（与自研 structure_protect 一致的口径）：
  1. 解析成块：标题、段落、列表项、引用、代码块、表格。
  2. 清洗成纯文本：去掉 [文字](链接) 的链接只留文字，图片整体去掉，去掉强调/代码等标记和行内 HTML 标签。
  3. 结构单元不拆：段落、列表项等单元整体 ≤ 512（硬上限）就不切开；超过 512 才按 句末 → 分号 → 逗号 → 硬切 拆成 ≤ 256 的片段。
  4. 标题绑定：标题（含连续多级标题）并入其后第一个单元，不会单独落在块尾；以"："结尾的引导句并入下一个单元。
  5. 贪心装箱：同一小节内的文本单元依次合并，合并后不超过 256（软目标）；遇到新标题或小节变化就开新块，不跨章节。
  6. 表格：合并单元格展开成网格（rowspan 的值向下复制，colspan 只保留一次），一行一行输出，单元格用 " | " 分隔；
     按行装箱到 256，续块重复表头；表格不和正文混装（引导句除外）。
  7. 兜底：任何块都不超过 512。

用法：
  python3 md_chunker.py --input in.jsonl --output out.jsonl
  python3 md_chunker.py --input in.jsonl --output out.jsonl --heading-context   # 每块前补标题路径
只依赖 Python 标准库。
"""
import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser

TARGET = 256
HARD = 512
MAX_TITLE_CHARS = 100

# ---------------------------------------------------------------- 行内清洗

INLINE_TAGS = ('a|abbr|b|big|br|center|cite|code|del|div|em|figcaption|figure|font|i|img|input|ins|kbd|label|'
               'mark|p|q|s|samp|small|source|span|strike|strong|sub|sup|time|u|var|video|audio|iframe|wbr|hr|'
               'section|article|header|footer|nav|aside|ul|ol|li|h[1-6]|blockquote|pre|dl|dt|dd|tt|button')
HTML_TAG = re.compile(r'</?(?:%s)(?:\s[^<>]*)?/?>' % INLINE_TAGS, re.I)
HTML_COMMENT = re.compile(r'<!--.*?-->', re.S)
BR = re.compile(r'<br\s*/?>', re.I)
AUTOLINK = re.compile(r'<(?:https?|ftp|mailto):[^<>\s]*>|<[\w.+-]+@[\w-]+(?:\.[\w-]+)+>', re.I)
CODE_SPAN = re.compile(r'(`+)(.+?)\1', re.S)
EMPHASIS = [
    re.compile(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1', re.S),
    re.compile(r'(~~)(?=\S)(.+?)(?<=\S)\1', re.S),
    # 单星号强调；两侧是数字时不处理（避免把 3*4*5 当成强调）
    re.compile(r'(?<![*\d])(\*)(?=[^\s*\d])(.+?)(?<=[^\s*\d])\*(?![*\d])', re.S),
]
ESCAPE = re.compile(r'\\([!-/:-@\[-`{-~])')
PUA = 0xE000
ESCAPED = re.compile('[\uE021-\uE07E]')
SPACES = re.compile(r'[ \t\u00a0\u3000]+')


def _match_bracket(s, i):
    """s[i] == '['，返回配对的 ']' 位置；找不到返回 -1。"""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _match_paren(s, i):
    """s[i] == '('，返回配对的 ')' 位置（链接地址里可以有成对括号，如 Python_(语言)）；找不到返回 -1。"""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '\n':
            return -1
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def strip_links(s, keep_image_alt=False):
    """[文字](地址) → 文字；[文字][引用] → 文字；![说明](地址) → 去掉（或保留说明）；链接里套图片也能处理。"""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            out.append(s[i:i + 2])
            i += 2
            continue
        is_image = c == '!' and i + 1 < n and s[i + 1] == '['
        if c == '[' or is_image:
            b = i + 1 if is_image else i
            e = _match_bracket(s, b)
            if e != -1:
                end = None
                if s.startswith('(', e + 1):
                    p = _match_paren(s, e + 1)
                    if p != -1:
                        end = p
                elif s.startswith('[', e + 1):
                    q = s.find(']', e + 2)
                    if q != -1 and '\n' not in s[e + 2:q] and '[' not in s[e + 2:q]:
                        end = q
                if end is not None:
                    inner = s[b + 1:e]
                    if not is_image or keep_image_alt:
                        out.append(strip_links(inner, keep_image_alt))
                    i = end + 1
                    continue
        out.append(c)
        i += 1
    return ''.join(out)


def clean_inline(s, keep_image_alt=False):
    """Markdown 行内语法 → 纯文本。保留换行，去掉行首尾空白和空行。"""
    # 转义字符（\* \[ 等）先换成私用区占位符，避免被当成强调或链接语法，最后还原
    s = ESCAPE.sub(lambda m: chr(PUA + ord(m.group(1))), s)
    s = HTML_COMMENT.sub('', s)
    s = AUTOLINK.sub('', s)
    s = strip_links(s, keep_image_alt)
    s = BR.sub('\n', s)
    s = CODE_SPAN.sub(lambda m: m.group(2).strip(), s)
    s = HTML_TAG.sub('', s)
    for _ in range(2):  # 处理嵌套强调，如 ***粗斜体***
        for pattern in EMPHASIS:
            s = pattern.sub(r'\2', s)
    s = ESCAPED.sub(lambda m: chr(ord(m.group()) - PUA), s)
    s = html.unescape(s)
    s = SPACES.sub(' ', s)
    lines = (line.strip() for line in s.split('\n'))
    return '\n'.join(line for line in lines if line)


# ---------------------------------------------------------------- HTML 表格

class _TableParser(HTMLParser):
    """只解析最外层表格的行和单元格；嵌套表格的文字并入所在单元格。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.rows = []        # [{'cells': [...], 'head': bool}]
        self.caption = []
        self.cell = None
        self.in_thead = False
        self.in_caption = False
        self.skip = 0         # script/style 内

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
            return
        if tag == 'table':
            self.depth += 1
            return
        if self.depth != 1:
            if tag == 'br' and self.cell is not None:
                self.cell['text'].append('\n')
            elif tag in ('tr', 'p', 'div', 'li') and self.cell is not None:
                self.cell['text'].append(' ')
            return
        attrs = dict(attrs)
        if tag == 'thead':
            self.in_thead = True
        elif tag == 'caption':
            self.in_caption = True
        elif tag == 'tr':
            self._close_cell()
            self.rows.append({'cells': [], 'head': self.in_thead})
        elif tag in ('td', 'th'):
            self._close_cell()
            if not self.rows:
                self.rows.append({'cells': [], 'head': self.in_thead})
            self.cell = {'text': [], 'th': tag == 'th',
                         'rowspan': _span(attrs.get('rowspan')), 'colspan': _span(attrs.get('colspan'))}
        elif tag == 'br' and self.cell is not None:
            self.cell['text'].append('\n')
        elif tag in ('p', 'div', 'li') and self.cell is not None:
            self.cell['text'].append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
            return
        if tag == 'table':
            if self.depth == 1:
                self._close_cell()
            self.depth = max(0, self.depth - 1)
            return
        if self.depth != 1:
            return
        if tag == 'thead':
            self.in_thead = False
        elif tag == 'caption':
            self.in_caption = False
        elif tag in ('td', 'th', 'tr'):
            self._close_cell()

    def handle_data(self, data):
        if self.skip or self.depth < 1:
            return
        if self.cell is not None:
            self.cell['text'].append(data)
        elif self.in_caption:
            self.caption.append(data)

    def _close_cell(self):
        if self.cell is not None:
            self.rows[-1]['cells'].append(self.cell)
            self.cell = None


def _span(value):
    try:
        return max(1, min(int(str(value).strip() or 1), 1000))
    except (TypeError, ValueError):
        return 1


def parse_html_table(markup, keep_image_alt=False):
    """返回 (caption, 表头行列表, 数据行列表)，每行是展开合并单元格后的单元格文字列表。"""
    parser = _TableParser()
    parser.feed(markup)
    parser.close()
    parser._close_cell()
    grid = {}          # (行, 列) -> (文字, 是否 colspan 的延续)
    is_head = []
    for r, row in enumerate(parser.rows):
        col = 0
        for cell in row['cells']:
            while (r, col) in grid:
                col += 1
            text = clean_inline(''.join(cell['text']), keep_image_alt).replace('\n', ' ')
            for rr in range(r, min(r + cell['rowspan'], len(parser.rows))):
                for cc in range(col, col + cell['colspan']):
                    grid.setdefault((rr, cc), (text, cc != col))
            col += cell['colspan']
        cells = row['cells']
        is_head.append(row['head'] or (bool(cells) and all(c['th'] for c in cells)))
    rows = []
    for r in range(len(parser.rows)):
        cols = sorted(c for (rr, c) in grid if rr == r)
        values = [grid[r, c][0] for c in cols if not grid[r, c][1]]
        rows.append(values)
    # 表头：thead 或开头全是 th 的行；都没有时用第一行（只在表格跨块时用于续块）
    n_head = 0
    while n_head < len(rows) and is_head[n_head]:
        n_head += 1
    keep = [i for i, values in enumerate(rows) if any(values)]
    header = [rows[i] for i in keep if i < n_head]
    body = [rows[i] for i in keep if i >= n_head]
    if not header and len(body) > 1:
        header, body = body[:1], body[1:]
    caption = clean_inline(''.join(parser.caption), keep_image_alt).replace('\n', ' ')
    return caption, header, body


def parse_pipe_table(lines, keep_image_alt=False):
    """Markdown 管道表 → (caption, 表头行, 数据行)。lines 第二行是分隔行。"""
    def cells(line):
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|') and not line.endswith('\\|'):
            line = line[:-1]
        return [clean_inline(c.replace('\\|', '|'), keep_image_alt).replace('\n', ' ')
                for c in re.split(r'(?<!\\)\|', line)]
    header = [cells(lines[0])]
    body = [cells(line) for line in lines[2:]]
    return '', header, [b for b in body if any(b)]


def row_text(values):
    return ' | '.join(v for v in values)


# ---------------------------------------------------------------- 块解析

@dataclass
class Block:
    kind: str                 # heading / text / table
    text: str = ''
    level: int = 0
    header: list = field(default_factory=list)   # 表格：表头行文字
    rows: list = field(default_factory=list)     # 表格：数据行文字


ATX = re.compile(r'^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$')
SETEXT = re.compile(r'^ {0,3}(=+|-+)[ \t]*$')
THEMATIC = re.compile(r'^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$')
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})')
LIST_ITEM = re.compile(r'^([ \t]*)([-*+]|\d{1,9}[.)])(?:[ \t]+(.*))?$')
QUOTE = re.compile(r'^ {0,3}>[ ]?(.*)$')
LINK_DEF = re.compile(r'^ {0,3}\[[^\]]+\]:\s*\S+.*$')
PIPE_SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')
TABLE_OPEN = re.compile(r'<table\b', re.I)
TABLE_TAG = re.compile(r'<(/?)table\b[^>]*>', re.I)


def _is_block_start(line):
    return bool(ATX.match(line) or FENCE.match(line) or THEMATIC.match(line) or QUOTE.match(line)
                or LIST_ITEM.match(line) or TABLE_OPEN.search(line))


def parse_blocks(markdown, keep_image_alt=False):
    lines = markdown.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    blocks = []
    para = []

    def flush():
        if para:
            text = clean_inline('\n'.join(para), keep_image_alt)
            if text:
                blocks.append(Block('text', text))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        # HTML 表格：可跨多行；同一行表格前后的文字分别处理
        m = TABLE_OPEN.search(line)
        if m:
            before = line[:m.start()]
            if before.strip():
                para.append(before)
            flush()
            buf, depth, j, rest = [], 0, i, ''
            first = line[m.start():]
            current = first
            while True:
                end = None
                for t in TABLE_TAG.finditer(current):
                    depth += -1 if t.group(1) else 1
                    if depth == 0:
                        end = t.end()
                        break
                if end is not None:
                    buf.append(current[:end])
                    rest = current[end:]
                    break
                buf.append(current)
                j += 1
                if j >= len(lines):
                    break
                current = lines[j]
            caption, header, body = parse_html_table('\n'.join(buf), keep_image_alt)
            if all(len(r) <= 1 for r in header + body):
                # 单列表格多为排版用途，按普通段落处理
                text = '\n'.join(t for t in [caption] + [r[0] for r in header + body if r] if t)
                if text:
                    blocks.append(Block('text', text))
            elif header or body:
                blocks.append(Block('table', caption, header=[row_text(r) for r in header],
                                    rows=[row_text(r) for r in body]))
            if rest.strip():
                lines[j] = rest
                i = j
            else:
                i = j + 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        m = FENCE.match(line)
        if m:
            flush()
            fence = m.group(1)
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            i += 1
            text = '\n'.join(c.rstrip() for c in code).strip('\n')
            if text.strip():
                blocks.append(Block('text', text))
            continue
        m = ATX.match(line)
        if m:
            flush()
            text = clean_inline(m.group(2) or '', keep_image_alt).replace('\n', ' ')
            if text:
                blocks.append(Block('heading', text, level=len(m.group(1))))
            i += 1
            continue
        m = SETEXT.match(line)
        if m and para:
            text = clean_inline(' '.join(para), keep_image_alt).replace('\n', ' ')
            para.clear()
            if text:
                blocks.append(Block('heading', text, level=1 if m.group(1)[0] == '=' else 2))
            i += 1
            continue
        if THEMATIC.match(line) or LINK_DEF.match(line):
            flush()
            i += 1
            continue
        if '|' in line and i + 1 < len(lines) and PIPE_SEP.match(lines[i + 1]) and '|' in lines[i + 1]:
            flush()
            j = i + 2
            while j < len(lines) and lines[j].strip() and '|' in lines[j]:
                j += 1
            caption, header, body = parse_pipe_table(lines[i:j], keep_image_alt)
            blocks.append(Block('table', caption, header=[row_text(r) for r in header],
                                rows=[row_text(r) for r in body]))
            i = j
            continue
        m = QUOTE.match(line)
        if m:
            flush()
            inner = []
            while i < len(lines):
                q = QUOTE.match(lines[i])
                if q:
                    inner.append(q.group(1))
                elif lines[i].strip() and inner and inner[-1].strip() and not _is_block_start(lines[i]):
                    inner.append(lines[i])  # 惰性续行
                else:
                    break
                i += 1
            blocks.extend(parse_blocks('\n'.join(inner), keep_image_alt))
            continue
        m = LIST_ITEM.match(line)
        if m:
            flush()
            indent = len(m.group(1).expandtabs(4))
            marker = m.group(2)
            item = [m.group(3) or '']
            content_indent = indent + len(marker) + 1
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    # 空行后缩进的内容仍属于本项（松散列表）
                    k = i
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    if (k < len(lines) and len(lines[k].expandtabs(4)) - len(lines[k].expandtabs(4).lstrip())
                            >= content_indent and not LIST_ITEM.match(lines[k])):
                        item.append('')
                        i = k
                        continue
                    break
                if LIST_ITEM.match(nxt) or (_is_block_start(nxt) and not nxt.startswith((' ', '\t'))):
                    break
                if TABLE_OPEN.search(nxt) or FENCE.match(nxt.strip()):
                    break
                item.append(nxt.strip())
                i += 1
            text = clean_inline('\n'.join(item), keep_image_alt)
            if text:
                if marker[0].isdigit():
                    text = marker + ' ' + text
                blocks.append(Block('text', text))
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


# ---------------------------------------------------------------- 长文本拆分

SENTENCE = re.compile(r'[^\n。！？!?；;]*(?:[。！？!?；;]+[”’」』）)\]】》"\']*|\n|$)')
CLAUSE = re.compile(r'[^\n，,、：:]*(?:[，,、：:]+[”’」』）)\]】》"\']*|$)')


def _pieces(text, pattern):
    return [p for p in pattern.findall(text) if p]


def split_text(text, target=TARGET):
    """把超长文本拆成 ≤ target 的片段：优先在换行/句末/分号处，其次逗号/顿号/冒号，最后硬切。"""
    out, cur = [], ''
    for sent in _pieces(text, SENTENCE):
        if len(sent.strip()) > target:
            parts = []
            for clause in _pieces(sent, CLAUSE):
                while len(clause) > target:
                    parts.append(clause[:target])
                    clause = clause[target:]
                parts.append(clause)
        else:
            parts = [sent]
        for part in parts:
            if cur and len((cur + part).strip()) > target:
                out.append(cur)
                cur = ''
            cur += part
    if cur:
        out.append(cur)
    return [p.strip() for p in out if p.strip()]


# ---------------------------------------------------------------- 结构单元与装箱

@dataclass
class Unit:
    kind: str                  # text / table
    path: tuple                # 标题路径
    lead: list                 # 绑定到本单元开头的标题
    block: Block
    bind_next: bool = False    # 以冒号结尾的引导句，与下一个单元绑定


def build_units(blocks):
    units, stack, pending = [], [], []
    for b in blocks:
        if b.kind == 'heading':
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            pending.append(b.text)
            continue
        u = Unit('table' if b.kind == 'table' else 'text', tuple(t for _, t in stack), pending, b)
        if u.kind == 'text' and b.text.rstrip().endswith(('：', ':')):
            u.bind_next = True
        units.append(u)
        pending = []
    if pending:  # 文末只有标题、没有正文
        units.append(Unit('text', tuple(t for _, t in stack), [], Block('text', '\n'.join(pending))))
    return units


@dataclass
class Chunk:
    text: str
    path: tuple
    family: str
    lead: list
    bind_next: bool = False


def _join(*parts):
    return '\n'.join(p for p in parts if p)


def text_pieces(u, target, hard):
    lead = _join(*u.lead)
    full = _join(lead, u.block.text)
    if len(full) <= hard:
        return [full]  # 结构单元整体保留（可超过 target，不超过 hard）
    pieces = split_text(u.block.text, max(target - len(lead) - 1, target // 2) if lead else target)
    pieces[0] = _join(lead, pieces[0])
    return pieces


def table_pieces(u, target, hard):
    """表格按行装箱：首块 = 标题/caption + 表头 + 行；续块重复表头。单行整体不拆（≤ hard），更长才拆。"""
    b = u.block
    header = '\n'.join(b.header)
    lead = _join(*u.lead, b.text)
    repeat = header if len(header) <= target // 2 else ''
    first = _join(lead, header)
    if not b.rows:
        return [first] if len(first) <= hard else split_text(first, target)
    out, cur, has_rows = [], first, False
    for row in b.rows:
        candidate = _join(cur, row)
        # 已有数据行时按 target 装箱；块里还没有数据行时，允许一行整体放到 hard
        if len(candidate) <= (target if has_rows else hard):
            cur, has_rows = candidate, True
            continue
        if has_rows or (cur and cur != repeat):
            out.append(cur)
        cur, has_rows = repeat, False
        candidate = _join(cur, row)
        if len(candidate) <= hard:
            cur, has_rows = candidate, True
            continue
        for part in split_text(row, max(target - len(cur) - 1, target // 2)):
            out.append(_join(cur, part))
        cur, has_rows = repeat, False
    if has_rows or (cur and cur != repeat):
        out.append(cur)
    return out


def pack(units, target=TARGET, hard=HARD, heading_context=False):
    chunks = []
    for u in units:
        t, h = target, hard
        if heading_context:
            reserve = len(_title_prefix(u.path, [])) + 1
            t, h = max(target - reserve, target // 2), max(hard - reserve, target)
        if u.kind == 'table':
            pieces, family = table_pieces(u, t, h), 'table'
        else:
            pieces, family = text_pieces(u, t, h), 'text'
        for k, piece in enumerate(pieces):
            prev = chunks[-1] if chunks else None
            starts_section = k == 0 and bool(u.lead)
            merge = False
            if prev is not None and prev.path == u.path and not starts_section:
                size = len(prev.text) + 1 + len(piece)
                if k == 0 and prev.bind_next and size <= h:
                    merge = True                       # 引导句 + 下一个单元
                elif prev.family == family == 'text' and size <= t:
                    merge = True                       # 同节文本贪心装箱
            if merge:
                prev.text = _join(prev.text, piece)
                prev.bind_next = u.bind_next and k == len(pieces) - 1
                if prev.family != family:
                    prev.family = 'table'
            else:
                chunks.append(Chunk(piece, u.path, family, u.lead if k == 0 else [],
                                    u.bind_next and k == len(pieces) - 1))
    return chunks


def _title_prefix(path, lead):
    """标题路径中不在本块开头的祖先标题；超过 MAX_TITLE_CHARS 时从最深一级开始丢。"""
    titles = list(path[:len(path) - len(lead)]) if lead else list(path)
    while titles and len('\n'.join(titles)) > MAX_TITLE_CHARS:
        titles.pop()
    return '\n'.join(titles)


def chunk_markdown(markdown, target=TARGET, hard=HARD, heading_context=False, keep_image_alt=False):
    """Markdown 文本 → chunk 字符串列表。"""
    if not isinstance(markdown, str) or not markdown.strip():
        return []
    units = build_units(parse_blocks(markdown, keep_image_alt))
    out = []
    for c in pack(units, target, hard, heading_context):
        text = c.text
        if heading_context:
            text = _join(_title_prefix(c.path, c.lead), text)
        if len(text) > hard:  # 兜底：保证不超过硬上限
            out.extend(split_text(text, target))
        elif text.strip():
            out.append(text.strip())
    return out


# ---------------------------------------------------------------- 命令行

def _percentile(values, q):
    if not values:
        return 0
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='输入 jsonl')
    ap.add_argument('--output', required=True, help='输出 jsonl（原字段保留，新增 chunk 列表）')
    ap.add_argument('--markdown-key', default='markdown', help='Markdown 正文所在的键（默认 markdown）')
    ap.add_argument('--chunks-key', default='chunks', help='新增的 chunk 列表键名（默认 chunks）')
    ap.add_argument('--target', type=int, default=TARGET, help='软目标字符数（默认 256）')
    ap.add_argument('--hard', type=int, default=HARD, help='硬上限字符数（默认 512）')
    ap.add_argument('--heading-context', action='store_true', help='每块前补上所在章节的标题路径（计入长度）')
    ap.add_argument('--keep-image-alt', action='store_true', help='图片保留说明文字（默认整体去掉）')
    args = ap.parse_args(argv)
    if args.hard < args.target:
        ap.error('--hard 不能小于 --target')
    n = errors = 0
    lengths = []
    with open(args.input, encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                chunks = chunk_markdown(record.get(args.markdown_key), args.target, args.hard,
                                        args.heading_context, args.keep_image_alt)
            except Exception as exc:  # noqa: BLE001  单条失败不影响整批：退化为纯文本拆分
                errors += 1
                print('第 %d 行切分失败，已退化为纯文本拆分：%r' % (lineno, exc), file=sys.stderr)
                chunks = split_text(clean_inline(str(record.get(args.markdown_key) or '')), args.target)
            record[args.chunks_key] = chunks
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            n += 1
            lengths.extend(len(c) for c in chunks)
    print(json.dumps({'records': n, 'fallback_records': errors, 'chunks': len(lengths),
                      'chunks_per_record': round(len(lengths) / max(n, 1), 2),
                      'len_p50': _percentile(lengths, .5), 'len_p90': _percentile(lengths, .9),
                      'len_max': max(lengths, default=0),
                      'over_target_pct': round(100 * sum(x > args.target for x in lengths) / max(len(lengths), 1), 1)},
                     ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
