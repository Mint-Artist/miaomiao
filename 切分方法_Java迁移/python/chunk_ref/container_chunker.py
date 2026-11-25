"""容器级结构切分：html_container（html 解析为容器单元 + 装块 + 合并）。"""
from collections import Counter
from dataclasses import dataclass, field
from typing import List
from bs4 import BeautifulSoup, NavigableString, Tag
from .config import *
from .text_utils import clean_inline, split_long, is_pseudo_heading
from .atoms import table_rows_from_soup, BS_PARSER, H_TAGS, SKIP_TAGS, INLINE_TAGS
from .title_detect import _tl_info
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

CT_HARD = 1024


CT_SPLIT = 512


CT_TINY = MIN_LEN


CT_HEAD_MAX = 100


CT_LEAD = True


CT_LEAD_MAX = 256


CT_LEAD_KINDS = ("table", "list", "code")


NOTE_CLASSES = {"note"}


META_CLASSES = {"metadata"}


CONTAINER_KINDS = ("list", "table", "note", "metadata", "quote", "code", "figure")


BLOCK_TAGS = {"p", "div", "section", "article", "main", "aside", "header", "footer", "nav", "li", "ul", "ol",
              "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "figure", "figcaption", "table", "thead",
              "tbody", "tfoot", "tr", "caption", "dl", "dt", "dd", "details", "summary", "hr"}


@dataclass
class Unit:
    """html 的一个结构单元：heading（level 1–7）/ para / 容器（list、table、note、metadata、quote、code、figure）。
    items：para 只有 1 条；容器为子结构文本（列表项、表格行、提示框里的每行等）。"""
    kind: str
    items: List[str] = field(default_factory=list)
    level: int = 0
    header: str = ""
    caption: str = ""


def block_text(el):
    """元素文字：块级子元素之间换行，行内元素连在一起（嵌套段落、子列表各占一行）。"""
    lines = [[]]

    def rec(n):
        for c in n.children:
            if isinstance(c, NavigableString):
                lines[-1].append(str(c))
            elif isinstance(c, Tag) and c.name not in SKIP_TAGS:
                if c.name in BLOCK_TAGS:
                    lines.append([])
                    rec(c)
                    lines.append([])
                else:
                    rec(c)

    rec(el)
    return clean_inline("\n".join("".join(x) for x in lines))


def is_layout_table(table, header):
    """排版用的表格：没有表头，且每一行最多 1 个非空单元格（论坛把整条帖子 / 评论放在一个单元格里）。
    这种表格不是数据表，按段落处理，才能和前后的段落合并；≥ 2 列或带表头的才算真正的表格容器。"""
    if header:
        return False
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = [td for td in tr.find_all(["td", "th"], recursive=False) if clean_inline(td.get_text()).strip()]
        if len(cells) > 1:
            return False
    return True


def parse_html_blocks(html):
    units = []
    if not html:
        return units
    soup = BeautifulSoup(html, BS_PARSER)
    for br in soup.find_all("br"):
        br.replace_with("\n")
    buf = []

    def flush():
        t = clean_inline("".join(buf))
        buf.clear()
        if t:
            units.append(Unit("para", [t]))

    def lines_of(el):
        return [x for x in block_text(el).split("\n") if x]

    def handle(c, depth):
        name = c.name
        classes = set(c.get("class") or [])
        if depth > 200:
            buf.append(c.get_text())
        elif name in H_TAGS:
            flush()
            t = block_text(c)
            if t and len(t) > CT_HEAD_MAX:  # 标签没闭合好时"标题"会吞进大段正文，按段落处理
                units.append(Unit("para", [t]))
            elif t:
                units.append(Unit("heading", [t], level=H_TAGS[name]))
        elif name == "p":
            flush()
            t = block_text(c)
            kids = [k for k in c.children if not (isinstance(k, NavigableString) and not k.strip())]
            if t and len(kids) == 1 and isinstance(kids[0], Tag) and kids[0].name in ("strong", "b") and is_pseudo_heading(t):
                units.append(Unit("heading", [t], level=7))
            elif t:
                units.append(Unit("para", [t]))
        elif name in ("ul", "ol"):
            flush()
            lis = c.find_all("li", recursive=False)
            items = [x for x in (block_text(li) for li in lis) if x] if lis else lines_of(c)
            if items:
                units.append(Unit("list", items))
        elif name == "table":
            flush()
            rows, header = table_rows_from_soup(c)
            cap = c.find("caption")
            cap = clean_inline(cap.get_text()).replace("\n", " ") if cap else ""
            if is_layout_table(c, header):  # 排版用的单列表格（论坛帖子、评论常见）：每行按段落处理
                for tr in c.find_all("tr"):
                    if tr.find_parent("table") is c:
                        t = block_text(tr)
                        if t:
                            units.append(Unit("para", [t]))
            elif rows or header:
                units.append(Unit("table", rows, header=header, caption=cap))
        elif name == "pre":
            flush()
            t = c.get_text().strip("\n")
            if t.strip():
                units.append(Unit("code", t.split("\n")))
        elif name in ("blockquote", "figure"):
            flush()
            items = lines_of(c)
            if items:
                units.append(Unit("quote" if name == "blockquote" else "figure", items))
        elif name == "div" and classes & (NOTE_CLASSES | META_CLASSES):
            flush()
            items = lines_of(c)
            if items:
                units.append(Unit("metadata" if classes & META_CLASSES else "note", items))
        elif name in INLINE_TAGS:
            buf.append(c.get_text())
        else:  # 其他 div / section 等外壳：作为段落边界，向内展开
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
    return units


def chunk_container_recs(units, soft=SOFT, hard=CT_HARD, split_to=CT_SPLIT, stats=None):
    """返回合并后的块：[{"text", "scope", "kind"}, ...]（未过滤 MIN_LEN）。kind = para 或容器类型。"""
    st = stats if stats is not None else Counter()
    recs = []                      # [文本, 上级标题路径, 是否以标题开头, kind]
    path, pending = [], []         # 标题栈 [(level, 序号)]；待吸附的标题 [(序号, 文本)]
    run, run_tl = [], []           # 段落组当前块的各段，及每段的类标题信息
    run_desc = []                  # 每段若是完整段落则记其字数（可作为表格前的描述段），否则 None
    rp = {"parent": (), "head": False}

    def take_pending():
        nonlocal pending
        own = {j for j, _ in pending}
        parent = tuple(j for _, j in path if j not in own)
        pre, head = [t for _, t in pending], bool(pending)
        pending = []
        return pre, parent, head

    def flush_run():
        nonlocal run, run_tl, run_desc
        if run:
            recs.append(["\n".join(run), rp["parent"], rp["head"], "para", True])
        run, run_tl, run_desc = [], [], []

    def trailing_k():
        """段落块末尾可挪走的类标题行数（规则同 chunk_adhere：≤ ADH_MAX_LINES 行，同一编号样式不能出现两次）。"""
        k = 0
        while k < len(run_tl) and run_tl[-1 - k]:
            k += 1
        if not k:
            return 0
        styles = Counter(s for _, s in run_tl[len(run_tl) - k:] if s)
        if k > ADH_MAX_LINES:
            st["跳过：类标题行超过上限"] += 1
            return 0
        if styles and max(styles.values()) >= 2:
            st["跳过：同一编号样式出现多次（像列表）"] += 1
            return 0
        return k

    def tail_fits(k, extra):
        """取走段落块末尾 k 段（剩下不足 MIN_LEN 时整块）后，与对方块其余 extra 字合起来是否不超过 hard。"""
        whole = sum(len(x) for x in run[:-k]) < MIN_LEN
        moved = run if whole else run[-k:]
        return sum(len(x) + 1 for x in moved) + extra <= hard

    def take_tail(k, extra):
        """从段落块末尾取走 k 行（剩下不足 MIN_LEN 时整块取走）；extra = 并入后对方块的其余长度，超过 hard 则不取。"""
        nonlocal run, run_tl, run_desc
        if not tail_fits(k, extra):
            st["跳过：并入后超过硬上限"] += 1
            return [], None
        whole = sum(len(x) for x in run[:-k]) < MIN_LEN
        moved = run[:] if whole else run[-k:]
        inherit = (rp["parent"], rp["head"]) if whole else None
        if whole:
            run, run_tl, run_desc = [], [], []
        else:
            run, run_tl, run_desc = run[:-k], run_tl[:-k], run_desc[:-k]
        return moved, inherit

    def add_para_piece(piece, tl, desc=None):
        nonlocal run, run_tl, run_desc
        carry, inherit = [], None
        if run and sum(len(x) for x in run) + len(run) + len(piece) > soft:
            k = trailing_k()
            if k and tl and tl[0] == "编号标题" and run_tl[-1][0] == "编号标题":
                st["跳过：下一段也是编号标题"] += 1
                k = 0
            if k:
                carry, inherit = take_tail(k, len(piece))
                if carry:
                    st["段落吸附：类标题行挪到下一块"] += 1
            flush_run()
        if not run:
            pre, parent, head = take_pending()
            rp["parent"], rp["head"] = inherit if inherit else (parent, head)
            run.append("\n".join(pre + carry + [piece]))
        else:
            run.append(piece)
        run_tl.append(tl)
        run_desc.append(desc)

    def emit_container(u):
        lead, inherit = [], None
        k = trailing_k()
        body_len = len("\n".join(([u.caption] if u.caption else []) + ([u.header] if u.header else []) + u.items))
        # 容器前的描述段：容器能整块保留（≤ hard）时，把紧挨着的一个完整段落（≤ CT_LEAD_MAX 字）一起挪到容器开头；
        # 容器要拆分时不带。类标题行（编号标题）照常一起挪。
        if CT_LEAD and u.kind in CT_LEAD_KINDS and body_len <= hard:
            j = len(run) - k - 1
            if j >= 0 and run_desc[j] is not None and run_desc[j] <= CT_LEAD_MAX and not run_tl[j]:
                if tail_fits(k + 1, body_len):
                    lead, inherit = take_tail(k + 1, body_len)
                    st[f"前文：描述段并入{u.kind}"] += 1
                else:
                    st["前文：并入后超过上限，不带"] += 1
        if not lead and k:
            extra = body_len if body_len <= hard else len(u.header) + len(u.caption) + 2 + (len(u.items[0]) if u.items else 0)
            lead, inherit = take_tail(k, extra)
            if lead:
                st["引导绑定：类标题行并入容器开头"] += 1
        flush_run()
        pre, parent, head = take_pending()
        if inherit:
            parent, head = inherit
        first = pre + lead + ([u.caption] if u.caption else [])
        rep = [u.header] if u.kind == "table" and u.header else []
        whole = "\n".join(first + rep + u.items)
        if len(whole) <= hard:
            recs.append([whole, parent, head, u.kind, True])
            st[f"{u.kind}：整块"] += 1
            return
        st[f"{u.kind}：超过 {hard} 字，按子结构拆"] += 1
        pieces = []
        for it in u.items:
            pieces += [it] if len(it) <= split_to else split_long(it, split_to)
        cur, has, n_out = first + rep, False, 0
        for p in pieces:
            if has and sum(len(x) for x in cur) + len(cur) + len(p) > split_to:
                recs.append(["\n".join(cur), parent, head and n_out == 0, u.kind, False])  # 拆开的容器片段不参与合并
                n_out += 1
                cur, has = list(rep), False
            cur.append(p)
            has = True
        if has or n_out == 0:
            recs.append(["\n".join(cur), parent, head and n_out == 0, u.kind, n_out == 0])

    for i, u in enumerate(units):
        if u.kind == "heading":
            flush_run()
            while path and path[-1][0] >= u.level:
                path.pop()
            path.append((u.level, i))
            pending.append((i, u.items[0]))
        elif u.kind == "para":
            t = u.items[0]
            if len(t) <= hard:
                add_para_piece(t, _tl_info(t), len(t))
            else:
                for p in split_long(t, split_to):
                    add_para_piece(p, None)
        elif sum(len(x) for x in u.items) + len(u.header) + len(u.caption) < CT_TINY:
            # 极小容器（如两条短提示）单独成块会被 MIN_LEN 过滤掉，按段落处理，可与相邻段落合并
            st[f"{u.kind}：极小，按段落处理"] += 1
            add_para_piece("\n".join(([u.caption] if u.caption else []) + ([u.header] if u.header else []) + u.items), None)
        else:
            emit_container(u)
    flush_run()
    if pending:  # 页面以标题结尾
        pre, parent, _ = take_pending()
        recs.append(["\n".join(pre), parent, True, "para", True])

    # 相邻块合并：同一小节范围内、合并后不超过 soft；段落块可以和整块的容器合并，但一块里最多一个容器
    # （容器与容器之间从不合并，避免提示框和步骤混装）；拆开的容器片段不参与合并
    out = []
    for text, parent, head, kind, mergeable in recs:
        if out:
            g = out[-1]
            if (mergeable and g["mergeable"] and not (g["has_cont"] and kind != "para")
                    and parent[: len(g["scope"])] == g["scope"] and len(g["text"]) + 1 + len(text) <= soft):
                g["text"] += "\n" + text
                if kind != "para":
                    g["has_cont"], g["kind"] = True, kind
                st["合并：段落块与容器块合并" if (g["has_cont"]) else "合并：段落块合并"] += 1
                continue
        out.append({"text": text, "scope": parent if head else parent[:-1], "kind": kind,
                    "has_cont": kind != "para", "mergeable": mergeable})
    return out


def chunk_container(units, soft=SOFT, hard=CT_HARD, split_to=CT_SPLIT, stats=None):
    groups = chunk_container_recs(units, soft, hard, split_to, stats)
    return [g["text"] for g in groups if len(g["text"].replace("\n", "")) >= MIN_LEN]
