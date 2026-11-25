"""Markdown → 结构单元。"""
from typing import List
from bs4 import BeautifulSoup
from .text_utils import *
from .atoms import Atom, table_rows_from_soup, BS_PARSER
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

def parse_markdown(md: str) -> List[Atom]:
    atoms: List[Atom] = []
    if not md:
        return atoms
    lines = md.replace("\r\n", "\n").split("\n")
    i, tid = 0, 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # HTML 表格（可能跨多行）
        if s.lower().startswith("<table"):
            buf = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
            i += 1
            soup = BeautifulSoup("\n".join(buf), BS_PARSER)
            for table in soup.find_all("table"):
                rows, header = table_rows_from_soup(table)
                for r in rows:
                    atoms.append(Atom("row", r, table_id=tid, header=header))
                tid += 1
            continue
        # 代码块
        if s.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            if "".join(buf).strip():
                atoms.append(Atom("code", "\n".join(buf)))
            continue
        # 管道表格
        if s.startswith("|") and i + 1 < len(lines) and MD_PIPE_SEP.match(lines[i + 1]):
            header = " | ".join(c.strip() for c in s.strip("|").split("|"))
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = " | ".join(md_inline(c) for c in lines[i].strip().strip("|").split("|"))
                atoms.append(Atom("row", row, table_id=tid, header=md_inline(header)))
                i += 1
            tid += 1
            continue
        m = MD_HEADING.match(s)
        if m:
            t = md_inline(m.group(2))
            if t:
                atoms.append(Atom("heading", t, level=len(m.group(1))))
            i += 1
            continue
        m = MD_BOLD_LINE.match(s)
        if m and is_pseudo_heading(md_inline(m.group(2))):
            atoms.append(Atom("heading", md_inline(m.group(2)), level=7))
            i += 1
            continue
        m = MD_LIST.match(line)
        if m:
            t = md_inline(m.group(1))
            if t:
                atoms.append(Atom("li", t))
            i += 1
            continue
        t = md_inline(s)
        if t:
            atoms.append(Atom("para", t))
        i += 1
    return atoms
