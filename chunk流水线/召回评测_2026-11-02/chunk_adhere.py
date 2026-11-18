# %% [markdown]
# # 标题吸附：没被识别成标题的"1、xxx""具体步骤如下："不要留在块尾，挪到下一块开头（兜底规则）
#
# 接在 recall_pipeline_v2.py 第 1–7 段之后运行（用到 Atom、split_long、merge_sections、chunk_secmerge_t、
# parse_markdown、parse_html、METHODS、SOFT、HARD、TABLE_SOFT、MIN_LEN 等）。
#
# 问题：网页作者用普通 <p> 写"1、xxx"这类小标题（不是 <h*>，也没整段加粗），解析后是普通段落，不会强制断开；
#       打包到软上限时切点正好落在它后面，标题留在上一块末尾，它的正文进了下一块。
# 做法：在 *_secmerge_t512 的打包阶段，因"超软上限"或"类型切换（段落→表格/列表）"要切断时，
#       若当前块末尾是连续的"类标题行"，把它们挪到下一块开头（和真标题"与紧随的第一段放在同一块"同理）。
#       不强制断开、不改变小节范围（上级标题路径），只调整切点位置；merge_sections 与 MIN_LEN 过滤不变。
#
# 类标题行（para 单元，去首尾空白后 ≤ 40 字）：
#   - 编号标题：以编号开头且不以句末标点结尾。编号：1、 1. 1． 1) 1）（"1." 后紧跟数字不算，如 1.5 亿）、
#               (1) （1）、一、 （一）、第一章 / 第二节 / 第3步 / 第三部分、①–⑳、Step 1 / 步骤 1
#               以冒号结尾但带编号的（"1、准备工作："）也算编号标题
#   - 引导句：不带编号、以 ： 或 : 结尾，如"具体操作步骤如下：" —— 实测效果不好，默认关闭（ADH_LEAD_IN = False）
# 保护条件（满足任一条就不挪）：
#   - 末尾连续类标题行超过 4 行
#   - 末尾这几行里同一种编号样式出现 ≥ 2 次（"8、条目8 / 9、条目9"更像列表）；
#     不同样式叠在一起（"一、xxx / （一）yyy / 1、物理类"）是标题链，照常挪
#   - 下一段本身也是编号标题，且块尾最后一行也是编号标题（短条目列表，切在哪都一样）；块尾是引导句时照常吸附
#   - 挪过去后下一块（类标题行 + 表头 + 下一段）会超过硬上限 HARD
#   - 挪走后上一块剩下不到 MIN_LEN 字 → 不拆，整块并入下一块（否则剩下的部分会被 MIN_LEN 过滤掉）
# 并入表格块时顺序为：类标题行 → 表头 → 表格行。content_br 基线不受影响。
#
# 新方法：md_struct_secmerge_t512_adh、html_struct_secmerge_t512_adh

# %% 定义
import math
import random
import re
from collections import Counter

import numpy as np
from tqdm.auto import tqdm

ADH_MAX_LEN = 40      # 类标题行最长字数
ADH_MAX_LINES = 4     # 块尾最多挪几行（标题链如"一、xxx / （一）yyy / 1、物理类"是 3 行）
ADH_LEAD_IN = False   # 不带编号、以冒号结尾的引导句（"具体步骤如下："）是否参与吸附；实测效果不好，默认关闭

# 编号样式：同一样式在块尾连续出现 ≥ 2 次 → 更像列表（1、苹果 / 2、香蕉），不挪；不同样式叠在一起是标题链，照常挪
_STYLES = [
    ("数字", re.compile(r"^\d{1,2}(?:[、)）]|[.．](?!\d))")),
    ("括号数字", re.compile(r"^[（(]\d{1,2}[)）]")),
    ("中文数字", re.compile(r"^[一二三四五六七八九十]{1,3}[、.．]")),
    ("括号中文数字", re.compile(r"^[（(][一二三四五六七八九十]{1,3}[)）]")),
    ("第X", re.compile(r"^第[一二三四五六七八九十百零\d]{1,3}[章节部分条步篇讲课]")),
    ("圈号", re.compile(r"^[①-⑳]")),
    ("步骤", re.compile(r"^(?:step|步骤)\s*\d{1,2}", re.I)),
]
_NOT_TITLE_END = set("。！？!?；;，,、…")


def num_style(text):
    t = text.strip()
    return next((name for name, p in _STYLES if p.match(t)), None)


def title_like(text):
    """类标题行返回 "编号标题" / "引导句"，否则 None。
    以冒号结尾的行：带编号（如"1、准备工作："）仍算编号标题；不带编号的引导句只在 ADH_LEAD_IN=True 时算。"""
    t = text.strip()
    if not t or len(t) > ADH_MAX_LEN:
        return None
    if t[-1] in "：:":
        if num_style(t):
            return "编号标题"
        return "引导句" if ADH_LEAD_IN else None
    if t[-1] in _NOT_TITLE_END:
        return None
    return "编号标题" if num_style(t) else None


def _tl_info(text):
    """(类型, 编号样式) 或 None；打包时逐段记录。"""
    k = title_like(text)
    return (k, num_style(text) if k == "编号标题" else None) if k else None


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


METHODS["md_struct_secmerge_t512_adh"] = lambda d: chunk_secmerge_adh(d.md_atoms, SOFT, HARD, TABLE_SOFT)
METHODS["html_struct_secmerge_t512_adh"] = lambda d: chunk_secmerge_adh(d.html_atoms, SOFT, HARD, TABLE_SOFT)
ADH_METHODS = ["md_struct_secmerge_t512_adh", "html_struct_secmerge_t512_adh"]


def dangling(chunks):
    """块尾悬空的类标题行个数（不算页面最后一块：页面以标题结尾时无处可挪）。"""
    return sum(1 for c in chunks[:-1] if title_like(c.rsplit("\n", 1)[-1]))


def adhere_effect(n_docs=20000, n_parts=40, seed=0, pattern=None, n_examples=200):
    """抽样网页，对比 *_secmerge_t512 与 *_adh：总块数、块尾悬空类标题行数、被改动的网页数、吸附 / 跳过次数；
    并核对 adhere=False 时与原打包逐字一致。返回 (stats, examples)。"""
    parts = list(list_parts(pattern or DATA_GLOB))
    random.Random(seed).shuffle(parts)
    parts = parts[: max(1, min(n_parts, len(parts)))]
    per = math.ceil(n_docs / len(parts))
    st = {w: Counter() for w in ("md", "html")}
    agg = {w: Counter() for w in ("md", "html")}
    examples, seen, mismatch = [], 0, 0
    for p in tqdm(parts, desc="抽样统计"):
        for i, rec in enumerate(iter_records(p)):
            if i >= per or seen >= n_docs:
                break
            seen += 1
            for w, atoms in (("md", parse_markdown(rec.get("markdown") or "")),
                             ("html", parse_html(rec.get("html") or ""))):
                if chunk_struct_sections_adh(atoms, adhere=False) != chunk_struct_sections_t(atoms, SOFT, HARD, TABLE_SOFT):
                    mismatch += 1
                old = chunk_secmerge_t(atoms, SOFT, HARD, TABLE_SOFT)
                new = chunk_secmerge_adh(atoms, stats=st[w])
                g = agg[w]
                g["网页"] += 1
                g["原块数"] += len(old)
                g["新块数"] += len(new)
                g["原悬空标题"] += dangling(old)
                g["新悬空标题"] += dangling(new)
                g["原有悬空标题的网页"] += dangling(old) > 0
                if old != new:
                    g["被改动的网页"] += 1
                    if len(examples) < n_examples:
                        examples.append({"url": rec.get("url"), "which": w, "old": old, "new": new})
    print(f"抽样网页 {seen} 个；adhere=False 时与原打包不一致：{mismatch}（应为 0）\n")
    for w in ("md", "html"):
        g = agg[w]
        n = max(g["网页"], 1)
        print(f"[{w}] 块数 {g['原块数']} → {g['新块数']}（{100 * (g['新块数'] / max(g['原块数'], 1) - 1):+.2f}%）；"
              f"块尾悬空类标题行 {g['原悬空标题']} → {g['新悬空标题']}；"
              f"有悬空标题的网页 {100 * g['原有悬空标题的网页'] / n:.2f}%；被改动的网页 {100 * g['被改动的网页'] / n:.2f}%")
        print("      ", dict(st[w]))
    return st, examples


def _boundary_view(chunks, title, n_tail=2, n_head=3, width=60):
    """找到包含 title 这一行的块，打印它前一块的末尾几行和它自己的开头几行。"""
    for j, c in enumerate(chunks):
        lines = c.split("\n")
        if title in lines:
            k = lines.index(title)
            prev = chunks[j - 1].split("\n")[-n_tail:] if k == 0 and j > 0 else lines[max(0, k - n_tail):k]
            for x in prev:
                print(f"      … {x[:width]}")
            print("      ────── 切点 ──────" if k == 0 else "      （同一块内）")
            for x in lines[k:k + n_head]:
                print(f"        {x[:width]}")
            return


def show_adhere_examples(examples, n=20, width=60):
    """逐个打印吸附前后的切点：原来块尾悬空的类标题行，现在落在哪里。"""
    shown = 0
    for ex in examples:
        old = ex["old"]
        hit = next((c.rsplit("\n", 1)[-1] for c in old[:-1] if title_like(c.rsplit("\n", 1)[-1])), None)
        if hit is None:
            continue
        print(f"\n[{ex['which']}] {ex['url']}\n    类标题行：{hit!r}（{title_like(hit)}）")
        print("    原来：")
        _boundary_view(old, hit, width=width)
        print("    吸附后：")
        _boundary_view(ex["new"], hit, width=width)
        shown += 1
        if shown >= n:
            break


def find_record(url, parts=None):
    """按顺序扫分片找到这条 url 的原始记录；知道所在分片时传 parts=[分片路径] 会快很多。"""
    for p in parts or list_parts(DATA_GLOB):
        for r in iter_records(p):
            if r.get("url") == url:
                return r
    raise KeyError(f"没找到：{url}")


def _content_tail_title(c, titles):
    """content_br 的块是 <br> 段落裸拼接（没有换行），用"块是否以某个类标题段落结尾"判断悬空。"""
    return next((t for t in titles if c.endswith(t)), None)


def compare_page(rec, which="html", full=False, only_changed=False, with_content=True):
    """单个网页：线上 content_br 切法 + 结构切分吸附前后，块数、块长，标出块尾悬空的类标题行。
    full=True：打印每块完整内容（保留换行）；默认只打印每块前 30 字的一行摘要。
    only_changed=True：结构切分只打印吸附前后内容不同的块（content_br 不受影响，总是全部打印）。
    with_content=False：不打印 content_br（同一网页先看 html 再看 md 时，第二次可以关掉）。"""
    def show(title, chunks, tail_title, skip=()):
        n_dang = sum(1 for c in chunks[:-1] if tail_title(c))
        print(f"\n{title}：{len(chunks)} 块，悬空类标题行 {n_dang} 个")
        for j, c in enumerate(chunks):
            if c in skip:
                continue
            t = tail_title(c) if j < len(chunks) - 1 else None
            mark = f"  ← 块尾悬空：{t[:30]!r}" if t else ""
            n = len(c.replace("\n", ""))
            if full:
                print(f"\n----- 第 {j} 块，{n} 字{mark} -----")
                print(c)
            else:
                print(f"  {j:3d} {n:4d}字  {c[:30]!r}{mark}")

    if with_content:
        ctt = rec.get("content") or ""
        titles = sorted({s.strip() for s in ctt.split("<br>") if title_like(s)}, key=len, reverse=True)
        show("[content_br] 线上切法（<br> 段落裸拼接，块内没有换行）", split(ctt),
             lambda c: _content_tail_title(c, titles))

    atoms = parse_html(rec.get("html") or "") if which == "html" else parse_markdown(rec.get("markdown") or "")
    st = Counter()
    old = chunk_secmerge_t(atoms, SOFT, HARD, TABLE_SOFT)
    new = chunk_secmerge_adh(atoms, stats=st)
    last_line = lambda c: c.rsplit("\n", 1)[-1] if title_like(c.rsplit("\n", 1)[-1]) else None
    show(f"[{which}] 结构切分·吸附前", old, last_line, set(new) if only_changed else ())
    show(f"[{which}] 结构切分·吸附后", new, last_line, set(old) if only_changed else ())
    print("\n吸附记录：", dict(st))


# %% 统计（抽样 2 万个网页，几分钟；想看谷歌网页时传 pattern=DATA_SOURCES[2]["glob"]）
ADH_STATS, ADH_EXAMPLES = adhere_effect(n_docs=20000)

# %% 样例：吸附前后的切点对比（人工核对规则是否合理）
show_adhere_examples(ADH_EXAMPLES, n=20)

# %% 单个网页（填 url；知道所在分片时 find_record(URL, parts=[分片路径]) 会快很多）
URL = "这里填那个网页的 url"
rec = find_record(URL)
compare_page(rec, "html")                                      # 摘要：content_br + html 吸附前后，每块一行
compare_page(rec, "html", full=True)                           # 完整内容：content_br 与 html 吸附前后的全部块
# compare_page(rec, "html", full=True, only_changed=True)      # 完整内容：html 只看吸附前后有变化的块
compare_page(rec, "md", with_content=False)                    # md 吸附前后（content_br 上面已经打印过）

# %% 切分：确认样例没问题后再运行（三批数据都切，按 content_br 对齐文档；已切完的分片自动跳过）
# 需要先运行 v2 第 9 段的定义（第 9 段末尾的 run_all_sources() 对已切完的方法会直接跳过）。
# 切完后用 8 卡脚本推理这两个方法的向量；评测时把 ADH_METHODS 加进 v2 第 1 段的 RUN_METHODS。
RUN_METHODS = RUN_METHODS + [m for m in ADH_METHODS if m not in RUN_METHODS]
run_all_sources()
