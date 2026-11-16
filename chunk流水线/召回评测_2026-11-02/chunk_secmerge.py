# %% [markdown]
# # 新切分方法：相邻小节合并（md_struct_secmerge / html_struct_secmerge）
#
# 背景：chunk_struct 遇到标题（含"整段加粗的短行"伪标题）或单元类型切换（段落 ↔ 列表 ↔ 表格）就强制断开，
# 卡片式、问答式页面会因此切出一串相邻的几十字小块（诊断见 why_small.py）。
# 新方法先按 chunk_struct 的原规则切块，再把相邻的小块从前往后贪心合并：
#   - 合并后的长度（含换行）不超过软上限才合并；所以只有小块会被合并，正常大小的块不动；
#   - 不跨越更高一级的标题：块只能并入"同一个上级标题之下"的组。例如 h3 小节之间、h3 小节与它下面的 h4 小节可以合并，
#     但遇到新的 h2（上一组属于前一个 h2）就另起一组；
#   - 不同类型的块（正文、列表、表格）可以合并；块里的标题、表头原样保留在合并后的文字里；
#   - 合并完成后再去掉不足 20 字（MIN_LEN）的块。原规则在打包时就丢掉这些碎块，新方法先参与合并，所以会多保留一点内容。
# 不合并时，切分结果与 md_struct / html_struct 完全相同。原有函数、方法都不修改。
#
# 用法（notebook）：
#   1. 运行 recall_pipeline.py 第 1–7 段之后，运行本文件的"定义"段（注册两个新方法）。
#   2. "示例"段：填一个 url，看这个网页合并前后各切出多少块、每块多长。
#   3. 可选："统计"段：抽样看全库块数、块长、短块占比的变化（与 content_br 对比）。
#   4. "切分"段：只切两个新方法，文档集合与 content_br 对齐；随机库、inj-、ggl- 三批各运行一次（需要第 9 段的定义）。
#   5. 推理向量：bash run_embed_job_multigpu.sh md_struct_secmerge html_struct_secmerge

# %% 定义
import math
import os
import random

import numpy as np
from tqdm.auto import tqdm


def chunk_struct_sections(atoms, soft: int = SOFT, hard: int = HARD):
    """与 chunk_struct（with_context=False）相同的打包规则，但不丢短块，并给每块记下：
    [块文本, 上级标题路径（标题在 atoms 里的序号）, 是否以标题开头]。"""
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
        pieces = [a.text] if len(a.text) <= hard else split_long(a.text, soft)
        for piece in pieces:
            size = sum(len(x) for x in cur) + len(cur)
            if cur and (family != cur_family or size + len(piece) > soft):
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
    """相邻小块从前往后贪心合并：合并后不超过 target 字，且不跨越更高一级的标题。
    组的范围（scope）= 组内第一块的上级标题路径；第一块不以标题开头（某一节的后半段）时，范围退到这一节的上一级。
    后面的块的上级标题路径以这个范围开头，才算在同一个上级标题之下。"""
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


def chunk_struct_secmerge(atoms, soft: int = SOFT, hard: int = HARD, target: int = None):
    """相邻小节合并后的结构切分。target：合并后的长度上限，默认等于软上限。"""
    groups = merge_sections(chunk_struct_sections(atoms, soft, hard), target or soft)
    return [g["text"] for g in groups if len(g["text"].replace("\n", "")) >= MIN_LEN]


METHODS["md_struct_secmerge"] = lambda d: chunk_struct_secmerge(d.md_atoms, SOFT, HARD)
METHODS["html_struct_secmerge"] = lambda d: chunk_struct_secmerge(d.html_atoms, SOFT, HARD)


def register_secmerge(soft, hard=None):
    """软上限不是 SOFT 时另行注册：md_struct_secmerge_s{soft} / html_struct_secmerge_s{soft}。返回两个方法名。"""
    hard = hard or HARD
    names = (f"md_struct_secmerge_s{soft}", f"html_struct_secmerge_s{soft}")
    METHODS[names[0]] = lambda d, s=soft, h=hard: chunk_struct_secmerge(d.md_atoms, s, h)
    METHODS[names[1]] = lambda d, s=soft, h=hard: chunk_struct_secmerge(d.html_atoms, s, h)
    print("已注册：", ", ".join(names))
    return list(names)


def find_record(url, parts=None):
    """按顺序扫分片找到这条 url 的原始记录（可能要一会儿）；知道所在分片时传 parts=[分片路径] 会快很多。"""
    for p in parts or list_parts(DATA_GLOB):
        for r in iter_records(p):
            if r.get("url") == url:
                return r
    raise KeyError(f"没找到：{url}")


def compare_page(rec, soft: int = SOFT, hard: int = HARD, show_text: bool = True, width: int = 100):
    """打印一个网页在 content_br、原结构切分、相邻小节合并下各切出多少块，以及合并后每块的内容。"""
    _n = lambda c: len(c.replace("\n", ""))
    cb = split(rec.get("content") or "")
    print(f"网页：{rec.get('url')}")
    print(f"content_br：{len(cb)} 块，块长 {[_n(c) for c in cb]}")
    for which, atoms in (("md", parse_markdown(rec.get("markdown") or "")), ("html", parse_html(rec.get("html") or ""))):
        old = chunk_struct(atoms, soft, hard)
        groups = merge_sections(chunk_struct_sections(atoms, soft, hard), soft)
        new = [g for g in groups if _n(g["text"]) >= MIN_LEN]
        print(f"\n===== {which}_struct：原规则 {len(old)} 块 → 相邻小节合并 {len(new)} 块 =====")
        print(f"原规则块长：{[_n(c) for c in old]}")
        print(f"合并后块长：{[_n(g['text']) for g in new]}")
        if show_text:
            for g in new:
                tag = f"（由 {g['n']} 块合并）" if g["n"] > 1 else ""
                print(f"  [{_n(g['text'])}字]{tag} " + g["text"].replace("\n", " / ")[:width])


def secmerge_effect(n_docs: int = 20000, n_parts: int = 40, seed: int = 0, short: int = 64):
    """抽样网页，对比 content_br、原结构切分、相邻小节合并的总块数、平均块长、短块占比；并核对不合并时与原规则一致。"""
    parts = list(list_parts(DATA_GLOB))
    random.Random(seed).shuffle(parts)
    parts = parts[: max(1, min(n_parts, len(parts)))]
    per = math.ceil(n_docs / len(parts))
    lens = {k: [] for k in ("content_br", "md_struct", "md_struct_secmerge", "html_struct", "html_struct_secmerge")}
    mismatch = seen = 0
    _n = lambda c: len(c.replace("\n", ""))
    for p in tqdm(parts, desc="抽样统计"):
        for i, rec in enumerate(iter_records(p)):
            if i >= per or seen >= n_docs:
                break
            seen += 1
            lens["content_br"] += [_n(c) for c in split(rec.get("content") or "")]
            for which, atoms in (("md", parse_markdown(rec.get("markdown") or "")), ("html", parse_html(rec.get("html") or ""))):
                old = chunk_struct(atoms, SOFT, HARD)
                recs = chunk_struct_sections(atoms, SOFT, HARD)
                if [t for t, _, _ in recs if _n(t) >= MIN_LEN] != old:
                    mismatch += 1
                new = [g["text"] for g in merge_sections(recs, SOFT) if _n(g["text"]) >= MIN_LEN]
                lens[f"{which}_struct"] += [_n(c) for c in old]
                lens[f"{which}_struct_secmerge"] += [_n(c) for c in new]
    base = max(len(lens["content_br"]), 1)
    print(f"抽样网页 {seen} 个；不合并时与原规则不一致的网页：{mismatch}（应为 0）")
    print(f"{'方法':22s}{'总块数':>8s}{'块数/content_br':>16s}{'平均块长':>9s}{'块长中位数':>11s}{f'短块(<{short}字)':>13s}{'每页总字数':>11s}")
    for k, v in lens.items():
        a = np.array(v or [0])
        print(f"{k:22s}{len(v):8d}{len(v) / base:16.3f}{a.mean():9.1f}{np.median(a):11.0f}"
              f"{100 * (a < short).mean():12.1f}%{a.sum() / max(seen, 1):11.0f}")


def align_drop(parts, ref: str = "content_br"):
    """让新方法切出的文档集合、顺序与 ref 方法完全一致（与 chunk_merge.py 中的同名函数相同）：
    按 ref 的 *.urls.txt 逐条对齐原始记录，ref 当时跳过的记录记入 DROP，chunk_part 会同样跳过。"""
    global DROP
    drop, n_skip, missing = {}, 0, []
    for path in tqdm(parts, desc=f"按 {ref} 对齐文档"):
        name = part_name(path)
        f = os.path.join(OUT_ROOT, ref, "chunks", name + ".urls.txt")
        if not os.path.exists(f[: -len(".urls.txt")] + ".done"):
            missing.append(name)
            continue
        with open(f, encoding="utf-8") as fu:
            want = fu.read().split("\n")
        want = want[:-1] if want and want[-1] == "" else want
        p = 0
        for i, rec in enumerate(iter_records(path)):
            url = (rec.get("url") or "").replace("\n", " ")
            if p < len(want) and url == want[p]:
                p += 1
            else:
                drop.setdefault(name, set()).add(i)
                n_skip += 1
        if p != len(want):
            raise ValueError(f"{name}：{ref} 有 {len(want)} 条文档，按原始数据只对齐到 {p} 条，数据可能已变化")
    if missing:
        raise ValueError(f"{len(missing)} 个分片在 {ref} 里还没切完（如 {missing[:3]}），先确认 DATA_GLOB / PART_PREFIX 与当时一致")
    DROP = drop
    print(f"已按 {ref} 对齐 {len(parts)} 个分片，跳过 {n_skip} 条（{ref} 当时去重掉的记录）")


# %% 示例：看一个网页合并前后各切出多少块
URL = "这里填网页的 url"
PAGE = find_record(URL)  # 知道所在分片时：find_record(URL, parts=["分片路径"])
compare_page(PAGE)

# %% 统计（可选）：抽样看全库的变化，不写任何文件
secmerge_effect(20000)

# %% 切分：只切两个新方法（随机库、inj-、ggl- 三批各运行一次；每批先改第 1 段的 DATA_GLOB / PART_PREFIX，
#        重新运行第 1、2 段，再运行这里；需要第 9 段的定义）
RUN_METHODS = ["md_struct_secmerge", "html_struct_secmerge"]
DEDUP_URLS = False  # 用 align_drop 对齐，不再按 url 重新去重
_parts = list_parts(DATA_GLOB)
align_drop(_parts, ref="content_br")
run_chunking(_parts)
# 切完后推理向量：bash run_embed_job_multigpu.sh md_struct_secmerge html_struct_secmerge
# 评测前把 RUN_METHODS 改回全部方法，例如：
# RUN_METHODS = ["content_br", "md_struct", "html_struct", "md_struct_secmerge", "html_struct_secmerge"]
