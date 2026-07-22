# -*- coding: utf-8 -*-
"""
标题吸附 + section path 前缀的结构化切分 Demo
==============================================

针对现有 pipeline（按 <br> 切段 -> 纯按长度贪心凑 ~256）的两个缺陷：

1. 【标题孤儿】标题行被当普通段落，可能被甩在 chunk 尾部，与正文分离
   -> 引入"逻辑块"：标题与其正文是一个原子单元，合并/再切分都不可拆散
2. 【归属丢失】chunk 之间硬切，被切断的正文不知道自己属于哪个小节
   -> 每个 chunk 头部拼接 section path 前缀："文档标题 > 一级小节 > 二级小节"

长度口径与线上规则一致：中文字符按字计、英文按词计（256 ≈ 256 个中文字符）。

运行：python3 heading_aware_chunker.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ----------------------------------------------------------------------
# 长度口径
# ----------------------------------------------------------------------

_CJK_RE = re.compile(r'[一-鿿豈-﫿]')
_EN_WORD_RE = re.compile(r'[A-Za-z0-9]+')


def text_len(s: str) -> int:
    """中文字符数 + 英文单词数，与线上 256 规则同口径。"""
    return len(_CJK_RE.findall(s)) + len(_EN_WORD_RE.findall(s))


# ----------------------------------------------------------------------
# 标题识别
# ----------------------------------------------------------------------

# (层级, 正则)，层级数字越小层级越高；业务接入时按语料实际情况补充
_HEADING_PATTERNS: list[tuple[int, re.Pattern]] = [
    (1, re.compile(r'^第[一二三四五六七八九十百零\d]+[章节篇部分][、.：:]?\s*\S')),  # 第一章
    (1, re.compile(r'^[一二三四五六七八九十]+[、.]\s*\S')),                         # 一、二、
    (2, re.compile(r'^（[一二三四五六七八九十]+）\s*\S')),                           # （一）（二）
    (2, re.compile(r'^\d{1,2}[、.]\s*\S')),                                          # 1、2.
    (3, re.compile(r'^（\d{1,2}）\s*\S')),                                           # （1）（2）
    (3, re.compile(r'^\d{1,2}[）)]\s*\S')),                                          # 1) 2）
]

MAX_HEADING_LEN = 40                            # 超长按正文处理（防 "1、物理类在705分..." 标题正文同行误判）
_END_PUNCT_RE = re.compile(r'[。！？；!?;]$')      # 句末标点收尾的不是标题（防 "一、二等奖各奖励一名。" 误判）


def detect_heading(seg: str) -> tuple[int, str] | None:
    """识别独立标题行，命中返回 (level, title)，否则返回 None。

    注意：这是通用启发式。接业务语料时建议补充领域规则，例如
    "连续多条短编号行"应判为数据列表整体保留，而不是一串标题。
    """
    s = seg.strip()
    if not s or text_len(s) > MAX_HEADING_LEN or _END_PUNCT_RE.search(s):
        return None
    for level, pat in _HEADING_PATTERNS:
        if pat.match(s):
            return level, s
    return None


# ----------------------------------------------------------------------
# 逻辑块：标题 + 正文的原子单元
# ----------------------------------------------------------------------

@dataclass
class Block:
    """逻辑块：一个标题和它名下的全部正文。合并与再切分都以块为单位，

    保证"标题永远和其正文在同一块里"——这是消除标题孤儿的根基。
    """
    path: list[str]   # 完整归属路径（含文档标题），如 [文档, 一、..., 1、物理类]
    lines: list[str]  # 块内容，第一行是标题行本身

    @property
    def size(self) -> int:
        return sum(text_len(line) for line in self.lines)

    @property
    def is_heading_only(self) -> bool:
        return len(self.lines) == 1


def parse_blocks(doc_title: str, body: str) -> list[Block]:
    """把 <br> 分隔的正文解析成逻辑块序列，并维护标题栈得到 section path。"""
    segments = [s.strip() for s in body.split('<br>')]
    segments = [s for s in segments if s]

    stack: list[tuple[int, str]] = []   # 标题栈 (level, title)
    blocks: list[Block] = []
    cur = Block(path=[doc_title], lines=[])

    def path_now() -> list[str]:
        return [doc_title] + [t for _, t in stack]

    for seg in segments:
        heading = detect_heading(seg)
        if heading:
            level, _ = heading
            while stack and stack[-1][0] >= level:   # 同级或更深标题出栈
                stack.pop()
            stack.append((level, seg))
            if cur.lines:                            # 新标题 -> 旧块收尾
                blocks.append(cur)
            cur = Block(path=path_now(), lines=[seg])
        else:
            cur.lines.append(seg)
    if cur.lines:
        blocks.append(cur)

    # 纯标题块（父标题后紧跟子标题，自己没有正文）前向并入下一个块，
    # 保证任何块都不会是"孤零零一行标题"
    merged: list[Block] = []
    for b in blocks:
        if merged and merged[-1].is_heading_only and not b.is_heading_only:
            prev = merged.pop()
            b.lines = prev.lines + b.lines   # 父标题行保留在文本里，path 用子块的更精确
        merged.append(b)
    if len(merged) >= 2 and merged[-1].is_heading_only:
        # 文档末尾的悬挂标题：没有后文可绑，只能并入前块（此时靠前缀兜底归属）
        tail = merged.pop()
        merged[-1].lines.extend(tail.lines)
    return merged


# ----------------------------------------------------------------------
# 超长块的二次切分（句界对齐，标题吸附优先于长度上限）
# ----------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[。！？；!?;])')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[，、,])')


def split_sentences(text: str) -> list[str]:
    return [p for p in _SENT_SPLIT_RE.split(text) if p.strip()]


def hard_wrap(unit: str, hard_max: int) -> list[str]:
    """单句仍超 hard_max（一分一段表这类超长枚举句）：先按逗号/顿号折行，

    再不行按字符硬切。兜底逻辑，正常语料应尽量走不到这里。
    """
    if text_len(unit) <= hard_max:
        return [unit]
    out, cur, cur_len = [], '', 0
    for piece in (p for p in _CLAUSE_SPLIT_RE.split(unit) if p):
        plen = text_len(piece)
        if plen > hard_max:  # 逗号都救不回来，按字符硬切
            if cur:
                out.append(cur)
                cur, cur_len = '', 0
            for i in range(0, len(piece), hard_max):
                out.append(piece[i:i + hard_max])
            continue
        if cur and cur_len + plen > hard_max:
            out.append(cur)
            cur, cur_len = '', 0
        cur += piece
        cur_len += plen
    if cur:
        out.append(cur)
    return out


def split_block(block: Block, hard_max: int) -> list[list[str]]:
    """超长逻辑块按句界二次切分。每一片都以标题行开头或紧随其后，

    关键不变量：标题行绝不单独成一片（标题吸附是硬约束，长度是软约束）。
    """
    units = [block.lines[0]]                       # 第一行必是标题
    for line in block.lines[1:]:
        for sent in split_sentences(line):
            units.extend(hard_wrap(sent, hard_max))

    slices, cur, cur_len = [], [], 0
    for u in units:
        ulen = text_len(u)
        # cur 里只有标题行时强制继续装：宁可超长，不可让标题落单
        if len(cur) > 1 and cur_len + ulen > hard_max:
            slices.append(cur)
            cur, cur_len = [], 0
        cur.append(u)
        cur_len += ulen
    if cur:
        slices.append(cur)
    return slices


# ----------------------------------------------------------------------
# chunk 生成
# ----------------------------------------------------------------------

@dataclass
class Chunk:
    prefix_path: list[str]
    text_lines: list[str]

    @property
    def prefix(self) -> str:
        return ' > '.join(self.prefix_path)

    @property
    def content(self) -> str:
        return '\n'.join(self.text_lines)

    @property
    def content_len(self) -> int:
        return sum(text_len(line) for line in self.text_lines)

    def render(self) -> str:
        """建索引时写入的文本：section path 前缀 + 正文。"""
        return f'{self.prefix}\n{self.content}' if self.prefix else self.content


def _common_prefix(paths: list[list[str]]) -> list[str]:
    """多个块路径的最长公共前缀：chunk 跨小节时，前缀退化为公共祖先。"""
    out = []
    for elems in zip(*paths):
        if all(e == elems[0] for e in elems):
            out.append(elems[0])
        else:
            break
    return out


def chunk_blocks(
    blocks: list[Block],
    soft_target: int = 256,
    hard_max: int = 320,
    keep_section_pure: bool = False,
) -> list[Chunk]:
    """把逻辑块合并成 chunk。

    - soft_target/hard_max：弹性长度区间，给结构对齐留余地（替代"卡死 256"）
    - keep_section_pure=True 时只允许同 path 的块合并，chunk 绝不跨小节；
      对"物理类/历史类"这种兄弟小节易混淆的场景建议开启，代价是 chunk 更碎
    """
    grouped: list[list[Block]] = []
    cur: list[Block] = []
    cur_len = 0

    for b in blocks:
        if b.size > hard_max:                        # 超长块单独走句界再切分
            if cur:
                grouped.append(cur)
                cur, cur_len = [], 0
            for slice_lines in split_block(b, hard_max):
                grouped.append([Block(path=b.path, lines=slice_lines)])
            continue
        cross_section = cur and keep_section_pure and cur[-1].path != b.path
        if (cur and cur_len + b.size > hard_max) or cross_section:
            grouped.append(cur)
            cur, cur_len = [], 0
        cur.append(b)
        cur_len += b.size
        if cur_len >= soft_target:
            grouped.append(cur)
            cur, cur_len = [], 0
    if cur:
        grouped.append(cur)

    return [
        Chunk(prefix_path=_common_prefix([b.path for b in g]),
              text_lines=[line for b in g for line in b.lines])
        for g in grouped
    ]


def add_overlap(chunks: list[Chunk], n_sent: int = 1) -> list[Chunk]:
    """可选 overlap：把上一 chunk 末尾 n 句复制到下一 chunk 开头。

    关键设计：只在两个 chunk 的 section path 完全相同时才重叠。
    否则会把上一小节的句子带进新小节——和标题孤儿是同一种跨小节污染。
    """
    for i in range(1, len(chunks)):
        prev, cur = chunks[i - 1], chunks[i]
        if prev.prefix_path != cur.prefix_path:
            continue
        units = [s for line in prev.text_lines for s in split_sentences(line)]
        carry = [u for u in units if not detect_heading(u)][-n_sent:]
        if carry:
            cur.text_lines = carry + cur.text_lines
    return chunks


# ----------------------------------------------------------------------
# 质量指标：标题孤儿率（bad case 的量化形式，目标 = 0）
# ----------------------------------------------------------------------

def heading_orphan_rate(chunk_texts: list[str]) -> float:
    if not chunk_texts:
        return 0.0
    orphans = sum(1 for t in chunk_texts
                  if detect_heading(t.rstrip().split('\n')[-1]))
    return orphans / len(chunk_texts)


# ----------------------------------------------------------------------
# 旧规则复现（对照组）：按 <br> 切段后纯按长度贪心合并
# ----------------------------------------------------------------------

def legacy_chunk(body: str, target: int = 256) -> list[str]:
    segs = [s.strip() for s in body.split('<br>') if s.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in segs:
        if cur and cur_len + text_len(s) > target:
            chunks.append('\n'.join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += text_len(s)
    if cur:
        chunks.append('\n'.join(cur))
    return chunks


# ----------------------------------------------------------------------
# 高层入口
# ----------------------------------------------------------------------

def chunk_document(
    doc_title: str,
    body: str,
    soft_target: int = 256,
    hard_max: int = 320,
    keep_section_pure: bool = False,
    overlap_sentences: int = 0,
) -> list[Chunk]:
    blocks = parse_blocks(doc_title, body)
    chunks = chunk_blocks(blocks, soft_target, hard_max, keep_section_pure)
    if overlap_sentences > 0:
        chunks = add_overlap(chunks, overlap_sentences)
    return chunks


# ----------------------------------------------------------------------
# Demo：用 bad case 原文跑对照
# ----------------------------------------------------------------------

DOC_TITLE = '2026安徽高考一分一段表及位次对应大学'
BODY = (
    '2026安徽高考600分：历史类对应位次是3121，物理类对应位次是29273；'
    '500分：历史类对应位次是39038，物理类对应位次是136697。<br>'
    '安徽高考前1万名对应能上的大学有哈尔滨工业大学、大连理工大学、山东大学、武汉大学等。<br>'
    '一分一段表完整版及更多位次排名对应大学名单见下文。<br>'
    '一、安徽2026高考一分一段表<br>'
    '2026年安徽高考成绩排名表将于6月25日公布，届时本文同步教育考试院整理更新，可保持关注了解。'
    '下面参考安徽2025年高考一分一段表，介绍分数排名情况：<br>'
    '1、物理类<br>'
    '在705分及以上的有18人，600分对应29273名、590分对应36279名、580分对应44025名，'
    '570分对应52697名、560分对应62346名、550分对应73146名、540分对应84632名、530分对应96909名、'
    '520分对应109853名、510分对应123125名、500分对应136697名。<br>'
    '2、历史类<br>'
    '分数在671分及以上的考生有10人，600分对应3121名、590分对应4620名、580分对应6512名，'
    '570分对应8880名、560分对应11628名、550分对应14951名、540分对应18810名、530分对应23285名、'
    '520分对应28067名、510分对应33397名、500分对应39038名。<br>'
    '二、安徽高考位次排名对应大学<br>'
    '篇幅有限，本文仅为大家展示了2025年前1万名对应的大学名单（物理组）。'
)


def _show(title: str, chunk_texts: list[str]) -> None:
    print('=' * 70)
    print(title)
    print('=' * 70)
    for i, t in enumerate(chunk_texts, 1):
        print(f'--- chunk #{i} | 内容长度 {text_len(t)} ---')
        print(t)
        print()
    lens = [text_len(t) for t in chunk_texts]
    print(f'>> chunk 数={len(chunk_texts)}  长度 min/avg/max = '
          f'{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}  '
          f'标题孤儿率 = {heading_orphan_rate(chunk_texts):.0%}')
    print()


def main() -> None:
    # 0) 旧规则：复现 bad case（chunk 尾部挂着孤儿标题，物理/历史混在一个 chunk）
    _show('【对照组】旧规则：按 <br> 纯长度合并', legacy_chunk(BODY))

    # 1) 新方案默认档：标题吸附 + section path，允许跨小节合并
    chunks = chunk_document(DOC_TITLE, BODY)
    _show('【新方案】标题吸附 + section path 前缀', [c.render() for c in chunks])

    # 2) 小节纯度模式 + overlap：物理/历史这类兄弟小节坚决不合并
    pure = chunk_document(DOC_TITLE, BODY, keep_section_pure=True, overlap_sentences=1)
    _show('【新方案·纯度模式】keep_section_pure=True + overlap=1句',
          [c.render() for c in pure])

    # 演示路径结构
    print('各 chunk 的 section path：')
    for c in pure:
        print('  ', ' > '.join(c.prefix_path))


if __name__ == '__main__':
    main()
