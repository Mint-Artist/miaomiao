# %% [markdown]
# # 切分效果对比导出：抽网页（或按 url 列表取网页）→ 多种切法各切一遍 → 写 jsonl，用 chunk_viewer.html 查看
#
# 运行顺序：recall_pipeline_v2.py 第 1–7 段 → chunk_adhere.py "定义"段 → chunk_container.py "定义"段 → 本文件。
# 不需要 GPU、不需要向量；只读原始分片（DATA_SOURCES 里的三批数据）。
#
# 两种取网页的方式（见文末"运行"段）：
#   1. 随机抽样：export_sample(n=300)，可限定只要含列表 / 表格 / 提示框的网页
#   2. 指定 url：export_urls(["https://...", ...]) 或 export_urls("urls.txt")（每行一个 url），多进程扫描全部分片
# 输出 jsonl（.gz 结尾则压缩）：
#   第 1 行 {"type": "meta", ...}：方法列表、参数、生成时间
#   之后每行一个网页 {"type": "page", "url", "source", "structure": [结构单元], "methods": {方法: {"chunks", "stats"}}, "html"}
# 每个方法的 stats：块数、平均 / 最长字数、>512 字块数、<64 字块数、混装块、容器完整（同 chunk_container 的口径）、块尾悬空标题数。

# %% 定义
import gzip
import json
import math
import os
import random
import time
from collections import Counter
import multiprocessing as mp

import numpy as np
from tqdm.auto import tqdm

for _need in ("title_like", "parse_html_blocks"):
    if _need not in globals():
        raise NameError("请先运行 chunk_adhere.py 和 chunk_container.py 的“定义”段")

COMPARE_METHODS = ["content_br", "html_struct_secmerge_t512", "html_container"]
# 想多看几种时加进来，例如 "html_struct_secmerge_t512_adh"、"md_struct_secmerge_t512"、"md_struct_secmerge_t512_adh"
HTML_MAX = 200_000    # 每个网页导出的原始 html 最多多少字（查看器里看源码 / 渲染用）；0 = 不导出 html


def _chunk_dicts(m, rec, units):
    """一个方法切一个网页：[{text, len, kind?, dangling?}]。"""
    if m == "html_container":
        groups = [g for g in chunk_container_recs(units) if len(g["text"].replace("\n", "")) >= MIN_LEN]
        texts, kinds = [g["text"] for g in groups], [g["kind"] for g in groups]
    else:
        texts, kinds = METHODS[m](Doc(rec)), None
    if m == "content_br":  # 段落裸拼接、块内没有换行：看块是否以某个类标题段落结尾
        titles = sorted({s.strip() for s in (rec.get("content") or "").split("<br>") if title_like(s)}, key=len, reverse=True)
        tail = lambda c: next((t for t in titles if c.endswith(t)), None)
    else:
        tail = lambda c: c.rsplit("\n", 1)[-1] if title_like(c.rsplit("\n", 1)[-1]) else None
    out = []
    for j, c in enumerate(texts):
        d = {"text": c, "len": len(c.replace("\n", ""))}
        if kinds:
            d["kind"] = kinds[j]
        t = tail(c) if j < len(texts) - 1 else None
        if t:
            d["dangling"] = t
        out.append(d)
    return out


def build_page(rec, methods=None, html_max=HTML_MAX, source=""):
    methods = methods or COMPARE_METHODS
    html = rec.get("html") or ""
    units = parse_html_blocks(html)
    page = {"type": "page", "url": rec.get("url"), "source": source,
            "html_len": len(html), "content_len": len((rec.get("content") or "").replace("<br>", "")),
            "structure": [{"kind": u.kind, "level": u.level, "n": len(u.items),
                           "len": sum(len(x) for x in u.items) + len(u.header) + len(u.caption),
                           "text": (u.items[0] if u.items else "")[:80]} for u in units],
            "methods": {}}
    for m in methods:
        try:
            cs = _chunk_dicts(m, rec, units)
        except Exception as e:  # 单个方法出错不影响其他方法
            page["methods"][m] = {"error": repr(e), "chunks": [], "stats": {}}
            continue
        texts = [c["text"] for c in cs]
        lens = [c["len"] for c in cs] or [0]
        sm = structure_metrics(units, texts)
        page["methods"][m] = {"chunks": cs, "stats": {
            "n": len(cs), "avg": round(float(np.mean(lens)), 1), "max": int(max(lens)),
            "over512": sum(x > 512 for x in lens if cs), "short64": sum(x < 64 for x in lens if cs),
            "mixed": sm["混装块"], "complete": sm["完整容器"], "eligible": sm["可统计容器"],
            "dangling": sum("dangling" in c for c in cs)}}
    if html_max:
        page["html"] = html[:html_max]
        page["html_truncated"] = len(html) > html_max
    return page


def _open_out(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w", encoding="utf-8")


def _write(path, pages, methods, extra):
    with _open_out(path) as f:
        meta = {"type": "meta", "methods": methods, "baseline": "content_br" if "content_br" in methods else methods[0],
                "n_pages": len(pages), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": {"SOFT": SOFT, "HARD(原结构切分)": HARD, "TABLE_SOFT": TABLE_SOFT, "CT_HARD": CT_HARD,
                           "CT_SPLIT": CT_SPLIT, "MIN_LEN": MIN_LEN, "ADH_LEAD_IN": ADH_LEAD_IN,
                           "ADH_MAX_LINES": ADH_MAX_LINES}, **extra}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"已导出 {len(pages)} 个网页 → {path}")


def _all_parts(sources=None):
    out = []
    for s in sources or DATA_SOURCES:
        out += [(s["name"], p) for p in list_parts(s["glob"])]
    return out


def _has_container(rec):
    return any(u.kind in ("list", "table", "note") for u in parse_html_blocks(rec.get("html") or ""))


def export_sample(path, n=300, methods=None, sources=None, require="html", n_parts=60, seed=0, html_max=HTML_MAX):
    """随机抽 n 个网页导出。sources：DATA_SOURCES 的子集（默认三批都抽）；
    require："html" = 只要 html 非空的网页；"container" = 只要含列表 / 表格 / 提示框的网页；None = 不限。"""
    methods = methods or COMPARE_METHODS
    parts = _all_parts(sources)
    random.Random(seed).shuffle(parts)
    parts = parts[: max(1, min(n_parts, len(parts)))]
    per = math.ceil(n / len(parts))
    pages = []
    for name, p in tqdm(parts, desc="抽样"):
        got = 0
        for rec in iter_records(p):
            if got >= per or len(pages) >= n:
                break
            if require in ("html", "container") and not (rec.get("html") or "").strip():
                continue
            if require == "container" and not _has_container(rec):
                continue
            pages.append(build_page(rec, methods, html_max, f"{name} / {os.path.basename(p)}"))
            got += 1
        if len(pages) >= n:
            break
    _write(path, pages, methods, {"mode": "sample", "require": require, "seed": seed})
    return pages


_WANT = set()


def _scan_part(item):
    name, p = item
    hits = []
    for rec in iter_records(p):
        if norm_url(rec.get("url") or "") in _WANT:
            hits.append((name, p, rec))
    return hits


def export_urls(urls, path, methods=None, sources=None, n_proc=None, html_max=HTML_MAX):
    """按 url 取网页导出。urls：url 列表，或一个文本文件路径（每行一个 url，# 开头的行忽略）。
    多进程扫描全部分片（按 norm_url 匹配，同一 url 在多批出现时取第一次出现的）；导出顺序与输入顺序一致。"""
    global _WANT
    methods = methods or COMPARE_METHODS
    if isinstance(urls, str):
        with open(urls, encoding="utf-8") as f:
            urls = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    order = [norm_url(u) for u in urls]
    _WANT = set(order)
    parts = _all_parts(sources)
    found = {}
    with mp.get_context("fork").Pool(n_proc or N_PROC) as pool:  # 用 fork：子进程继承 notebook 里的函数和 _WANT
        for hits in tqdm(pool.imap_unordered(_scan_part, parts), total=len(parts), desc="扫描分片"):
            for name, p, rec in hits:
                found.setdefault(norm_url(rec.get("url") or ""), (name, p, rec))
    pages, missing = [], []
    for u, nu in zip(urls, order):
        if nu in found:
            name, p, rec = found[nu]
            pages.append(build_page(rec, methods, html_max, f"{name} / {os.path.basename(p)}"))
        else:
            missing.append(u)
    if missing:
        print(f"没找到 {len(missing)} 个 url，例如：{missing[:5]}")
    _write(path, pages, methods, {"mode": "urls", "n_requested": len(urls), "n_missing": len(missing)})
    return pages


# %% 运行 1：随机抽样（输出路径按实际修改；.gz 结尾会压缩）
CMP_FILE = os.path.join(OUT_ROOT, "chunk_compare_sample.jsonl.gz")
export_sample(CMP_FILE, n=300, require="container")     # require="html" 看全部有 html 的网页

# %% 运行 2：指定 url（列表或文件，二选一）
# export_urls(["https://example.com/a", "https://example.com/b"], os.path.join(OUT_ROOT, "chunk_compare_urls.jsonl.gz"))
# export_urls("/path/to/urls.txt", os.path.join(OUT_ROOT, "chunk_compare_urls.jsonl.gz"))
