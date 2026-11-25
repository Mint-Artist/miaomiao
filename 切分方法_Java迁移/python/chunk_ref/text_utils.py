"""文本工具：空白清洗、分句、超长段落切分、伪标题判断、Markdown 行内清洗。"""
import re
from typing import List
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

WS = re.compile(r"[ \t\r\f\v 　]+")


SENT_END = re.compile(r"[。！？!?；;…]+[”’\"）)】》」』]*|\n+")


END_PUNCT = re.compile(r"[。！？!?；;，,、…]$")


def clean_inline(s: str) -> str:
    """合并行内空白，保留换行。"""
    lines = [WS.sub(" ", x).strip() for x in s.split("\n")]
    return "\n".join(x for x in lines if x)


def split_sentences(text: str, max_len: int) -> List[str]:
    """按句末标点与换行分句；单句超过 max_len 时先在逗号处切，再硬切。"""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        if m.end() > start:
            out.append(text[start:m.end()])
            start = m.end()
    if start < len(text):
        out.append(text[start:])
    res = []
    for s in out:
        while len(s) > max_len:
            cut = max(s.rfind("，", 0, max_len), s.rfind(",", 0, max_len), s.rfind("、", 0, max_len))
            cut = cut + 1 if cut > 0 else max_len
            res.append(s[:cut])
            s = s[cut:]
        if s.strip():
            res.append(s)
    return res


def split_long(text: str, soft: int) -> List[str]:
    """超过硬上限的段落：按句贪心打包到软目标以内。"""
    pieces, cur = [], ""
    for s in split_sentences(text, soft):
        if cur and len(cur) + len(s) > soft:
            pieces.append(cur)
            cur = ""
        cur += s
    if cur.strip():
        pieces.append(cur)
    return [p.strip() for p in pieces if p.strip()]


MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


MD_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(.*)$")


MD_PIPE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


MD_EMPH = re.compile(r"(\*\*|__)(.+?)\1")


MD_BOLD_LINE = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*$")


HTML_TAG = re.compile(r"<[^>]+>")


def md_inline(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = MD_IMAGE.sub(r"\1", s)
    s = MD_LINK.sub(r"\1", s)
    s = MD_EMPH.sub(r"\2", s)
    s = HTML_TAG.sub("", s)
    return clean_inline(s)


def is_pseudo_heading(t: str) -> bool:
    return 0 < len(t) <= 40 and not END_PUNCT.search(t)
