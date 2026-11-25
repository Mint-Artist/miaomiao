"""5 种结构切分方法的独立参考实现（从 notebook 原版逐字抽取，不依赖 notebook 全局变量）。

用法：
    from chunk_ref import chunk, chunk_with_kind, METHOD_NAMES
    chunks = chunk("md_struct_secmerge_t512", markdown=md_text)
    chunks = chunk("html_container", html=html_text)
    pairs = chunk_with_kind(html_text)          # html_container：[(块文本, 块类型), ...]

方法与输入字段：
    md_struct_secmerge_t512         markdown
    md_struct_secmerge_t512_adh     markdown
    html_struct_secmerge_t512       html
    html_struct_secmerge_t512_adh   html
    html_container                  html
依赖：beautifulsoup4 + lxml（与离线评测相同；没有 lxml 时退回 html.parser，结果可能不同）。
"""
from .config import SOFT, HARD, MIN_LEN, TABLE_SOFT, ADH_MAX_LEN, ADH_MAX_LINES, ADH_LEAD_IN
from .atoms import BS_PARSER
from .parse_md import parse_markdown
from .parse_html import parse_html
from .section_chunker import (chunk_struct_sections_t, chunk_struct_sections_adh, merge_sections,
                              chunk_secmerge_t, chunk_secmerge_adh)
from .container_chunker import (CT_HARD, CT_SPLIT, CT_TINY, CT_HEAD_MAX, CT_LEAD, CT_LEAD_MAX, CT_LEAD_KINDS,
                                parse_html_blocks, chunk_container_recs, chunk_container)

# 与 notebook 里 METHODS 的注册完全相同（v2 第 7 段、chunk_adhere.py、chunk_container.py）
METHODS = {
    "md_struct_secmerge_t512": lambda md, html: chunk_secmerge_t(parse_markdown(md or ""), SOFT, HARD, TABLE_SOFT),
    "html_struct_secmerge_t512": lambda md, html: chunk_secmerge_t(parse_html(html or ""), SOFT, HARD, TABLE_SOFT),
    "md_struct_secmerge_t512_adh": lambda md, html: chunk_secmerge_adh(parse_markdown(md or ""), SOFT, HARD, TABLE_SOFT),
    "html_struct_secmerge_t512_adh": lambda md, html: chunk_secmerge_adh(parse_html(html or ""), SOFT, HARD, TABLE_SOFT),
    "html_container": lambda md, html: chunk_container(parse_html_blocks(html or "")),
}
METHOD_NAMES = list(METHODS)
INPUT_FIELD = {m: ("markdown" if m.startswith("md_") else "html") for m in METHODS}


def chunk(method, markdown="", html=""):
    """按方法名切分，返回块文本列表。"""
    if method not in METHODS:
        raise KeyError(f"未知方法 {method}，可选：{METHOD_NAMES}")
    return METHODS[method](markdown, html)


def chunk_with_kind(html):
    """html_container 的块及其类型（para / list / table / note / metadata / quote / code / figure），过滤规则与 chunk 相同。"""
    groups = chunk_container_recs(parse_html_blocks(html or ""))
    return [(g["text"], g["kind"]) for g in groups if len(g["text"].replace("\n", "")) >= MIN_LEN]
