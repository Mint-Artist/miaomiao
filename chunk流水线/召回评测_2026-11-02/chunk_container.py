# %% [markdown]
# # 容器级结构切分（只做 html）：以完整的结构容器为切分单位，不再按 256 字贪心切断列表 / 表格 / 提示框
#
# 运行顺序：recall_pipeline_v2.py 第 1–7 段 → chunk_adhere.py 的"定义"段（类标题行识别与它共用）→ 本文件。
#
# 为什么要做：现有结构切分（*_struct / *_secmerge_t512）先把 html 摊平成 段落 / 列表项 / 表格行 这类单元，
# 结构只决定"哪里必须断开"（标题、单元类型切换），同一类单元内部仍按 256 字贪心切 —— 长列表、长表格、长小节内部
# 的切点和 content_br 一样由长度决定；而且所有列表项是同一类，提示框和步骤列表会被混装；硬上限只用于单个超长段落。
#
# 规则：
#   A 容器识别：同一小节里连续的段落（段落组）；一个完整的 ul / ol（含嵌套段落、子列表）；div.note；div.metadata；
#     blockquote；pre；figure；一张完整的表格（caption + 表头 + 全部行）。其他 div / section 等只是外壳，向内展开。
#   B 容器整体 ≤ CT_HARD（1024 字）→ 整块成为一个 chunk，不受 256 限制（接受向量模型截断）。
#   C 容器 > CT_HARD → 在子结构边界切（列表按项、表格按行且每块重复表头、其他按行），每块贪心装到 CT_SPLIT（512 字）；
#     单个子结构仍超过 CT_SPLIT 才按句切。
#   D 只有段落可以合并：段落组内按 256 字贪心装箱（单个段落 ≤ 1024 不拆），同一小节内相邻的小段落块合并到 256 以内；
#     列表、提示框、表格、代码等容器不与任何其他内容合并（两个相邻的不同列表也不合并）。
#   E 标题吸附：h1–h6、整段加粗的伪标题 → 并到其后内容的第一块；段落组里的编号小标题（规则同 chunk_adhere）
#     在段落组内切断时挪到下一块，紧挨容器时并到容器开头（如"二、价格表" + 表格）。
#     不带编号的引导句（"价格如下："）默认不参与（chunk_adhere 里 ADH_LEAD_IN = False）。
#   F 上下文前缀：暂不加（现网有基于大模型的上下文 chunk）。h1 按普通标题处理，metadata 作为独立容器保留在正文里。
#   G 极小容器（< MIN_LEN 字，如两条很短的提示）按段落处理，可并入相邻段落 —— 否则单独成块会被 MIN_LEN 过滤掉。
#   H 排版用的表格（没有表头、每行最多 1 个非空单元格，论坛帖子 / 评论常见）不算表格容器，每行按段落处理。
#   I 表格前文：表格能整块保留（≤ CT_HARD）时，同一小节里紧挨在表格前的一个完整段落（≤ CT_TABLE_LEAD_MAX 字）
#     挪到表格块开头（挪动，不复制），作为表格的说明；加上它会超过 CT_HARD 就不带；表格要拆分时也不带。
# 最后过滤 < MIN_LEN 字的块（与其他方法一致）。
#
# 新方法：html_container

# %% 定义
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import numpy as np
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm.auto import tqdm

if "title_like" not in globals():
    raise NameError("请先运行 chunk_adhere.py 的“定义”段（类标题行识别规则与它共用）")

CT_HARD = 1024        # 容器整块保留的上限（字）
CT_SPLIT = 512        # 超大容器拆开后每块的目标长度（字）
CT_TINY = MIN_LEN     # 不足这么多字的容器按段落处理（否则单独成块会被 MIN_LEN 过滤掉，内容丢失）
CT_HEAD_MAX = 100     # 超过这么多字的 h1–h6 不当标题（多为标签没闭合好吞进了正文），按段落处理
CT_TABLE_LEAD = True  # 表格能整块保留时，把紧挨在前面的一个描述段落挪到表格开头（表格要拆分时不带）
CT_TABLE_LEAD_MAX = 256  # 描述段落最长多少字才带（更长的多半是普通正文，不是表格说明）
NOTE_CLASSES = {"note"}          # 视为"提示框"容器的 div class
META_CLASSES = {"metadata"}      # 视为"元信息"容器的 div class
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
            recs.append(["\n".join(run), rp["parent"], rp["head"], "para"])
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
        # 表格前的描述段：表格能整块保留（≤ hard）时，把紧挨着的一个完整段落（≤ CT_TABLE_LEAD_MAX 字）一起挪到表格开头；
        # 表格要拆分时不带。类标题行（编号标题）照常一起挪。
        if CT_TABLE_LEAD and u.kind == "table" and body_len <= hard:
            j = len(run) - k - 1
            if j >= 0 and run_desc[j] is not None and run_desc[j] <= CT_TABLE_LEAD_MAX and not run_tl[j]:
                if tail_fits(k + 1, body_len):
                    lead, inherit = take_tail(k + 1, body_len)
                    st["表格前文：描述段并入表格"] += 1
                else:
                    st["表格前文：并入后超过上限，不带"] += 1
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
            recs.append([whole, parent, head, u.kind])
            st[f"{u.kind}：整块"] += 1
            return
        st[f"{u.kind}：超过 {hard} 字，按子结构拆"] += 1
        pieces = []
        for it in u.items:
            pieces += [it] if len(it) <= split_to else split_long(it, split_to)
        cur, has, n_out = first + rep, False, 0
        for p in pieces:
            if has and sum(len(x) for x in cur) + len(cur) + len(p) > split_to:
                recs.append(["\n".join(cur), parent, head and n_out == 0, u.kind])
                n_out += 1
                cur, has = list(rep), False
            cur.append(p)
            has = True
        if has or n_out == 0:
            recs.append(["\n".join(cur), parent, head and n_out == 0, u.kind])

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
        recs.append(["\n".join(pre), parent, True, "para"])

    # 只有段落块之间合并：同一小节范围内、合并后不超过 soft
    out = []
    for text, parent, head, kind in recs:
        if out:
            g = out[-1]
            if (kind == "para" and g["kind"] == "para" and parent[: len(g["scope"])] == g["scope"]
                    and len(g["text"]) + 1 + len(text) <= soft):
                g["text"] += "\n" + text
                continue
        out.append({"text": text, "scope": parent if head else parent[:-1], "kind": kind})
    return out


def chunk_container(units, soft=SOFT, hard=CT_HARD, split_to=CT_SPLIT, stats=None):
    groups = chunk_container_recs(units, soft, hard, split_to, stats)
    return [g["text"] for g in groups if len(g["text"].replace("\n", "")) >= MIN_LEN]


METHODS["html_container"] = lambda d: chunk_container(parse_html_blocks(d.rec.get("html") or ""))
CT_METHODS = ["html_container"]


# ---------- 评估：块长分布 + 结构完整率（不需要推理） ----------
def _nows(s):
    return "".join(s.split())


def structure_metrics(units, chunks):
    """按子结构文本在块里的出现情况统计（去掉全部空白后做子串匹配）：
    混装块：同一块里出现 ≥ 2 个不同容器（列表 / 表格 / 提示框等）的内容；
    容器完整：容器所有子结构都出现在同一块里（只统计 ≤ CT_HARD 字、且所有子结构都能在该方法的块里找到的容器）；
    找到率：容器子结构能在任意块里找到的比例（content_br 的正文来自另一套抽取，找到率低时它的完整率不可靠）。"""
    cs = [_nows(c) for c in chunks]
    conts = [(i, [x for x in (_nows(it) for it in u.items) if len(x) >= 4])
             for i, u in enumerate(units) if u.kind in CONTAINER_KINDS and u.kind != "metadata"]
    conts = [(i, its) for i, its in conts if its]
    r = Counter()
    for c in cs:
        ids = {i for i, its in conts if any(it in c for it in its)}
        r["混装块"] += len(ids) >= 2
    for i, its in conts:
        found = [any(it in c for c in cs) for it in its]
        r["子结构"] += len(its)
        r["子结构找到"] += sum(found)
        if not all(found) or len("".join(its)) > CT_HARD:
            continue
        r["可统计容器"] += 1
        r["完整容器"] += any(all(it in c for it in its) for c in cs)
    r["块"] += len(chunks)
    return r


def container_effect(n_docs=20000, n_parts=40, seed=0, pattern=None, n_examples=100):
    """抽样网页（只统计 html 非空的网页），对比 content_br、html_struct_secmerge_t512、html_container：
    块数、块长分布、超长占比、混装块占比、容器完整率。返回 (各方法的统计, 容器切分记录, 样例)。"""
    parts = list(list_parts(pattern or DATA_GLOB))
    random.Random(seed).shuffle(parts)
    parts = parts[: max(1, min(n_parts, len(parts)))]
    per = math.ceil(n_docs / len(parts))
    names = ["content_br", "html_struct_secmerge_t512", "html_container"]
    lens = {m: [] for m in names}
    sm = {m: Counter() for m in names}
    st, examples, seen = Counter(), [], 0
    for p in tqdm(parts, desc="抽样统计"):
        for i, rec in enumerate(iter_records(p)):
            if i >= per or seen >= n_docs:
                break
            html = rec.get("html") or ""
            if not html.strip():
                continue
            seen += 1
            units = parse_html_blocks(html)
            out = {"content_br": split(rec.get("content") or ""),
                   "html_struct_secmerge_t512": chunk_secmerge_t(parse_html(html), SOFT, HARD, TABLE_SOFT),
                   "html_container": chunk_container(units, stats=st)}
            for m, cs in out.items():
                lens[m] += [len(c.replace("\n", "")) for c in cs]
                sm[m].update(structure_metrics(units, cs))
            if len(examples) < n_examples and any(u.kind in ("list", "table", "note") for u in units):
                examples.append({"url": rec.get("url"), **out})
    print(f"抽样网页 {seen} 个（html 非空）\n")
    rows = []
    for m in names:
        a = np.array(lens[m] or [0])
        s = sm[m]
        rows.append({"方法": m, "块数": len(lens[m]), "块/页": round(len(lens[m]) / max(seen, 1), 2),
                     "平均字数": round(a.mean(), 1), "中位数": int(np.median(a)), "p90": int(np.percentile(a, 90)),
                     ">256%": round(100 * (a > 256).mean(), 2), ">512%": round(100 * (a > 512).mean(), 2),
                     ">1024%": round(100 * (a > 1024).mean(), 2), "<64字%": round(100 * (a < 64).mean(), 2),
                     "混装块%": round(100 * s["混装块"] / max(s["块"], 1), 2),
                     "容器完整率%": round(100 * s["完整容器"] / max(s["可统计容器"], 1), 2),
                     "可统计容器": s["可统计容器"],
                     "子结构找到率%": round(100 * s["子结构找到"] / max(s["子结构"], 1), 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nhtml_container 切分记录：", dict(st))
    print("注：content_br 的正文来自另一套抽取，子结构找到率明显偏低时，它的混装 / 完整率只作参考。")
    return df, st, examples


def show_page(rec, full=True, width=40):
    """单个网页：content_br、html_struct_secmerge_t512、html_container 三种切法逐块对照。"""
    html = rec.get("html") or ""
    units = parse_html_blocks(html)
    print("结构单元：", Counter(u.kind for u in units))
    st = Counter()
    out = [("content_br（线上，<br> 段落裸拼接）", split(rec.get("content") or "")),
           ("html_struct_secmerge_t512（现有结构切分）", chunk_secmerge_t(parse_html(html), SOFT, HARD, TABLE_SOFT)),
           ("html_container（容器级）", chunk_container(units, stats=st))]
    for name, cs in out:
        print(f"\n==================== {name}：{len(cs)} 块 ====================")
        for j, c in enumerate(cs):
            n = len(c.replace("\n", ""))
            if full:
                print(f"\n----- 第 {j} 块，{n} 字 -----")
                print(c)
            else:
                print(f"  {j:3d} {n:4d}字  {c[:width]!r}")
    print("\nhtml_container 切分记录：", dict(st))


import pandas as pd  # noqa: E402  （评估表格用）

# %% 统计（抽样 2 万个网页，几分钟；想看谷歌网页时传 pattern=DATA_SOURCES[2]["glob"]）
CT_DF, CT_STATS, CT_EXAMPLES = container_effect(n_docs=20000)

# %% 样例：随机看几个含列表 / 表格 / 提示框的网页（逐块对照三种切法）
for ex in CT_EXAMPLES[:3]:
    print("\n\n######", ex["url"])
    for m in ("content_br", "html_struct_secmerge_t512", "html_container"):
        print(f"  [{m}] {len(ex[m])} 块，块长 {[len(c.replace(chr(10), '')) for c in ex[m]]}")

# %% 单个网页（填 url；知道所在分片时 find_record(URL, parts=[分片路径]) 会快很多；find_record 来自 chunk_adhere.py）
URL = "这里填那个网页的 url"
show_page(find_record(URL), full=True)

# %% 切分：确认统计和样例没问题后再运行（三批数据都切，按 content_br 对齐文档；已切完的分片自动跳过）
# 需要先运行 v2 第 9 段的定义（第 9 段末尾的 run_all_sources() 对已切完的方法会直接跳过）。
# 切完后用 8 卡脚本推理 html_container 的向量；评测时把它加进 v2 第 1 段的 RUN_METHODS。
RUN_METHODS = RUN_METHODS + [m for m in CT_METHODS if m not in RUN_METHODS]
run_all_sources()
