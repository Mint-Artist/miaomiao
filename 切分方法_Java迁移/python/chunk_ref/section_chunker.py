"""小节打包 + 相邻小节合并：md/html_struct_secmerge_t512 及其标题吸附版（_adh）。"""
from collections import Counter
from typing import List
from .config import *
from .text_utils import split_long
from .atoms import Atom
from .title_detect import _tl_info
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

def chunk_struct_sections_t(atoms: List[Atom], soft: int = SOFT, hard: int = HARD, table_soft: int = TABLE_SOFT):
    """返回 [[块文本, 上级标题路径（标题在 atoms 里的序号）, 是否以标题开头], ...]；不丢短块。"""
    recs = []
    cur, cur_family, cur_parent, cur_head = [], None, (), False
    path, pending = [], []  # path: [(level, 序号)]；pending: [(序号, 标题文字)]

    def flush():
        nonlocal cur, cur_family
        if cur:
            recs.append(["\n".join(cur), cur_parent, cur_head])
        cur, cur_family = [], None

    for i, a in enumerate(atoms):
        if a.kind == "heading":
            flush()
            while path and path[-1][0] >= a.level:
                path.pop()
            path.append((a.level, i))
            pending.append((i, a.text))
            continue
        family = ("row", a.table_id) if a.kind == "row" else ("li",) if a.kind == "li" else ("text",)
        lim = table_soft if a.kind == "row" else soft
        pieces = [a.text] if len(a.text) <= hard else split_long(a.text, lim)
        for piece in pieces:
            size = sum(len(x) for x in cur) + len(cur)
            if cur and (family != cur_family or size + len(piece) > lim):
                flush()
            add = piece
            if not cur:
                if a.kind == "row" and a.header:  # 表格的每一块都带表头
                    add = a.header + "\n" + add
                own = {j for j, _ in pending}
                cur_parent = tuple(j for _, j in path if j not in own)
                cur_head = bool(pending)
                if pending:  # 标题与紧随的第一段放在同一块
                    add = "\n".join(t for _, t in pending) + "\n" + add
                cur_family = family
                pending = []
            cur.append(add)
    if pending:  # 页面以标题结尾
        own = {j for j, _ in pending}
        cur_parent = tuple(j for _, j in path if j not in own)
        cur_head = True
        cur.append("\n".join(t for _, t in pending))
    flush()
    return recs


def merge_sections(recs, target: int):
    """相邻小块贪心合并：合并后不超过 target 字，且不跨越更高一级的标题。
    组的范围 = 组内第一块的上级标题路径（第一块是某一节的后半段时，退到这一节的上一级）；后面的块的上级标题路径以它开头才合并。"""
    out = []
    for text, parent, head in recs:
        if out:
            g = out[-1]
            if parent[: len(g["scope"])] == g["scope"] and len(g["text"]) + 1 + len(text) <= target:
                g["text"] += "\n" + text
                g["n"] += 1
                continue
        out.append({"text": text, "scope": parent if head else parent[:-1], "n": 1})
    return out


def chunk_secmerge_t(atoms: List[Atom], soft: int = SOFT, hard: int = HARD, table_soft: int = TABLE_SOFT) -> List[str]:
    groups = merge_sections(chunk_struct_sections_t(atoms, soft, hard, table_soft), soft)
    return [g["text"] for g in groups if len(g["text"].replace("\n", "")) >= MIN_LEN]


def chunk_struct_sections_adh(atoms, soft=SOFT, hard=HARD, table_soft=TABLE_SOFT, adhere=True, stats=None):
    """与 chunk_struct_sections_t 相同（adhere=False 时逐字一致），另加标题吸附。
    返回 [[块文本, 上级标题路径, 是否以标题开头], ...]；stats（Counter）记录吸附 / 跳过次数。"""
    st = stats if stats is not None else Counter()
    recs = []
    cur, cur_tl, cur_family, cur_parent, cur_head = [], [], None, (), False
    path, pending = [], []

    def flush():
        nonlocal cur, cur_tl, cur_family
        if cur:
            recs.append(["\n".join(cur), cur_parent, cur_head])
        cur, cur_tl, cur_family = [], [], None

    for i, a in enumerate(atoms):
        if a.kind == "heading":
            flush()
            while path and path[-1][0] >= a.level:
                path.pop()
            path.append((a.level, i))
            pending.append((i, a.text))
            continue
        family = ("row", a.table_id) if a.kind == "row" else ("li",) if a.kind == "li" else ("text",)
        lim = table_soft if a.kind == "row" else soft
        whole = len(a.text) <= hard
        pieces = [a.text] if whole else split_long(a.text, lim)
        for piece in pieces:
            piece_tl = _tl_info(piece) if a.kind == "para" and whole else None
            size = sum(len(x) for x in cur) + len(cur)
            carry, inherit = [], None
            if cur and (family != cur_family or size + len(piece) > lim):
                k = 0
                if adhere:
                    while k < len(cur_tl) and cur_tl[-1 - k]:
                        k += 1
                    styles = Counter(s for _, s in cur_tl[len(cur_tl) - k:] if s) if k else Counter()
                    if k > ADH_MAX_LINES:
                        st["跳过：块尾连续类标题行超过上限"] += 1
                        k = 0
                    elif styles and max(styles.values()) >= 2:
                        st["跳过：块尾同一编号样式出现多次（像列表）"] += 1
                        k = 0
                    elif k and piece_tl and piece_tl[0] == "编号标题" and cur_tl[-1][0] == "编号标题":
                        st["跳过：下一段也是编号标题"] += 1
                        k = 0
                    if k:  # 挪过去后下一块（标题行 + 表头 + 这一段）不能超过硬上限
                        moved = cur if sum(len(x) for x in cur[:-k]) < MIN_LEN else cur[-k:]
                        head = len(a.header) + 1 if a.kind == "row" and a.header else 0
                        if sum(len(x) + 1 for x in moved) + head + len(piece) > hard:
                            st["跳过：挪过去会超过硬上限"] += 1
                            k = 0
                if k:
                    st["吸附涉及行数"] += k
                    if sum(len(x) for x in cur[:-k]) < MIN_LEN:  # 剩下的太短：整块并入下一块
                        carry, inherit = cur[:], (cur_parent, cur_head)
                        cur, cur_tl, cur_family = [], [], None
                        st["吸附：整块并入下一块"] += 1
                    else:
                        carry = cur[-k:]
                        cur, cur_tl = cur[:-k], cur_tl[:-k]
                        flush()
                        st["吸附：挪走块尾类标题行"] += 1
                    if family != ("text",):
                        st["其中并入表格/列表块"] += 1
                else:
                    flush()
            add = piece
            if not cur:
                if a.kind == "row" and a.header:  # 表格的每一块都带表头
                    add = a.header + "\n" + add
                own = {j for j, _ in pending}
                cur_parent = tuple(j for _, j in path if j not in own)
                cur_head = bool(pending)
                if pending:  # 标题与紧随的第一段放在同一块
                    add = "\n".join(t for _, t in pending) + "\n" + add
                if carry:  # 吸附过来的类标题行放在最前（表格块里在表头之前）
                    add = "\n".join(carry) + "\n" + add
                    if inherit:
                        cur_parent, cur_head = inherit
                cur_family = family
                pending = []
            cur.append(add)
            cur_tl.append(piece_tl)
    if pending:  # 页面以标题结尾
        own = {j for j, _ in pending}
        cur_parent = tuple(j for _, j in path if j not in own)
        cur_head = True
        cur.append("\n".join(t for _, t in pending))
    flush()
    return recs


def chunk_secmerge_adh(atoms, soft=SOFT, hard=HARD, table_soft=TABLE_SOFT, stats=None):
    groups = merge_sections(chunk_struct_sections_adh(atoms, soft, hard, table_soft, True, stats), soft)
    return [g["text"] for g in groups if len(g["text"].replace("\n", "")) >= MIN_LEN]
