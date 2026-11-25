"""html → 结构单元（原结构切分 / 吸附版用）。"""
from typing import List
from bs4 import BeautifulSoup, NavigableString, Tag
from .text_utils import *
from .atoms import *
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

def parse_html(html: str) -> List[Atom]:
    atoms: List[Atom] = []
    if not html:
        return atoms
    soup = BeautifulSoup(html, BS_PARSER)
    for br in soup.find_all("br"):
        br.replace_with("\n")
    tid = [0]
    buf: List[str] = []

    def flush():
        t = clean_inline("".join(buf))
        buf.clear()
        if t:
            atoms.append(Atom("para", t))

    def own_text(el, exclude=("ul", "ol", "table")):
        parts = []
        for c in el.children:
            if isinstance(c, NavigableString):
                parts.append(str(c))
            elif isinstance(c, Tag) and c.name not in exclude and c.name not in SKIP_TAGS:
                parts.append(c.get_text())
        return clean_inline("".join(parts))

    def handle(c, depth):
        name = c.name
        if depth > 200:  # 极深嵌套直接取文字
            buf.append(c.get_text())
        elif name in H_TAGS:
            flush()
            t = clean_inline(c.get_text())
            if t:
                atoms.append(Atom("heading", t, level=H_TAGS[name]))
        elif name == "p":
            flush()
            t = clean_inline(c.get_text())
            kids = [k for k in c.children if not (isinstance(k, NavigableString) and not k.strip())]
            if t and len(kids) == 1 and isinstance(kids[0], Tag) and kids[0].name in ("strong", "b") and is_pseudo_heading(t):
                atoms.append(Atom("heading", t, level=7))  # 整段加粗的短行视为小标题
            elif t:
                atoms.append(Atom("para", t))
        elif name in ("ul", "ol"):
            flush()
            for li in c.find_all("li", recursive=False):
                t = own_text(li)
                if t:
                    atoms.append(Atom("li", t))
                for sub in li.find_all(["ul", "ol"], recursive=False):
                    handle(sub, depth + 1)
        elif name == "table":  # 嵌套表格的文字已包含在外层单元格里，不再单独处理
            flush()
            rows, header = table_rows_from_soup(c)
            for r in rows:
                atoms.append(Atom("row", r, table_id=tid[0], header=header))
            tid[0] += 1
        elif name == "pre":
            flush()
            t = c.get_text()
            if t.strip():
                atoms.append(Atom("code", t.strip("\n")))
        elif name in INLINE_TAGS:
            buf.append(c.get_text())
        else:  # div / section / blockquote 等容器：作为段落边界
            flush()
            walk(c, depth + 1)
            flush()

    def walk(node, depth=0):
        for c in node.children:
            if isinstance(c, NavigableString):
                buf.append(str(c))
            elif isinstance(c, Tag) and c.name not in SKIP_TAGS:
                handle(c, depth)

    walk(soup)
    flush()
    return atoms
