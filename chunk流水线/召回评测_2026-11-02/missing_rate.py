# %% [markdown]
# # 各切分方法的缺失率：谷歌网页上，基线 / md / html 各有多少网页一块都没切出来（只读，不写任何文件）
#
# 接在 recall_pipeline_v2.py 第 1–2 段之后运行（用到 OUT_ROOT、DATA_SOURCES、QUERY_FILES、norm_url、iter_records、list_parts）。
# 不需要 GPU、不需要加载模型；读的是已切好的 chunks/*.urls.txt 和 emb/*.offsets.npy（每个网页切出几块）。
#
# "缺失" = 网页在库里，但该方法切出 0 块（所以检索时这个网页不可能被该方法召回）。
# 输出四部分：
#   【1】谷歌网页这一批（ggl- 分片）：各方法缺失率；并按原因拆开：
#        原始字段为空（content / markdown / html 本身就没有）vs 字段有内容但切出 0 块（解析失败或全是 < MIN_LEN 的碎片）
#   【2】谷歌标准答案网页（在库里的，可能落在任意一批）：各方法缺失率，以及按谷歌名次 1–10 拆开
#   【3】缺失的交叉情况：html 缺失时 md / 基线能不能补上（决定"html 优先、缺了退回 md"的组合方案能挽回多少）
#   【4】query 层面：标准答案网页全部缺失的 query 有多少条（这些 query 对该方法来说必然 0 分）

# %% 定义
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

# 方法名 → 它用的原始字段（按方法名前缀判断）
FIELD_OF = {"content": "content", "md": "markdown", "html": "html"}


def field_of(method):
    return next((f for p, f in FIELD_OF.items() if method.startswith(p)), None)


def _shard_names(method, shard_glob):
    cdir = os.path.join(OUT_ROOT, method, "chunks")
    return sorted(os.path.basename(p)[: -len(".urls.txt")] for p in glob.glob(os.path.join(cdir, shard_glob + ".urls.txt")))


def _counts_of_shard(method, name):
    """该分片每个文档切出几块：优先读 emb/*.offsets.npy（快），没有推理过时读 chunks/*.jsonl。"""
    po = os.path.join(OUT_ROOT, method, "emb", name + ".offsets.npy")
    if os.path.exists(po):
        return np.diff(np.load(po))
    with open(os.path.join(OUT_ROOT, method, "chunks", name + ".jsonl"), encoding="utf-8") as f:
        return np.array([len(json.loads(l)["chunks"]) for l in f], dtype=np.int64)


def chunk_counts(methods, shard_glob="*", keep=None, ref=None):
    """{方法: {归一化 url: 块数}}。keep 给定时只保留这些 url（全库扫描时省内存）。
    各方法的 urls.txt 逐行一致时（align_drop 保证），url 只归一化一次。"""
    ref = ref or methods[0]
    out = {m: {} for m in methods}
    for name in _shard_names(ref, shard_glob):
        with open(os.path.join(OUT_ROOT, ref, "chunks", name + ".urls.txt"), encoding="utf-8") as f:
            raw_ref = f.read()
        nus = None
        for m in methods:
            fu = os.path.join(OUT_ROOT, m, "chunks", name + ".urls.txt")
            if not os.path.exists(fu):
                print(f"  [{m}] 缺分片 {name}，跳过")
                continue
            raw = raw_ref if m == ref else open(fu, encoding="utf-8").read()
            cnt = _counts_of_shard(m, name)
            if raw == raw_ref and nus is not None:
                cur = nus
            else:
                cur = [norm_url(u) for u in raw.split("\n")[: len(cnt)]]
                if raw == raw_ref:
                    nus = cur
            d = out[m]
            for nu, c in zip(cur, cnt):
                if keep is None or nu in keep:
                    d[nu] = max(d.get(nu, 0), int(c))
    return out


def raw_fields(pattern, keep=None):
    """原始记录里三个字段是否非空：{归一化 url: (content, markdown, html)}。"""
    info = {}
    for p in list_parts(pattern):
        for r in iter_records(p):
            nu = norm_url(r.get("url") or "")
            if keep is None or nu in keep:
                info[nu] = tuple(bool((r.get(k) or "").strip()) for k in ("content", "markdown", "html"))
    return info


def _pct(a, b):
    return round(100 * a / b, 2) if b else float("nan")


def batch_missing(methods, prefix="ggl-", raw_glob=None):
    """【1】某一批网页（按分片前缀）各方法的缺失率，并按原因拆开。"""
    cc = chunk_counts(methods, prefix + "*")
    urls = set(cc[methods[0]])
    raw = raw_fields(raw_glob, keep=urls) if raw_glob else {}
    idx = {"content": 0, "markdown": 1, "html": 2}
    rows = []
    for m in methods:
        miss = [u for u in urls if cc[m].get(u, 0) == 0]
        f = field_of(m)
        row = {"方法": m, "所用字段": f, "网页数": len(urls), "缺失": len(miss), "缺失率%": _pct(len(miss), len(urls))}
        if raw and f:
            empty = sum(1 for u in miss if u in raw and not raw[u][idx[f]])
            unknown = sum(1 for u in miss if u not in raw)
            row["其中原始字段为空"] = empty
            row["其中字段有内容但切出0块"] = len(miss) - empty - unknown
            row["其中原始记录没找到"] = unknown
        rows.append(row)
    if raw:
        n = len(urls)
        print(f"原始记录：content 为空 {_pct(sum(not v[0] for v in raw.values()), n)}%，"
              f"markdown 为空 {_pct(sum(not v[1] for v in raw.values()), n)}%，"
              f"html 为空 {_pct(sum(not v[2] for v in raw.values()), n)}%（分母为该批网页 {n} 个）")
    return pd.DataFrame(rows), cc


def load_gold(path, topn=10):
    """[(qid, [归一化 url, ...按名次])]；支持 relevant_urls（有序列表）或 relevance（字典）。"""
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            us = d.get("relevant_urls") or [u for u, g in (d.get("relevance") or {}).items() if g > 0]
            out.append((d.get("qid"), [norm_url(u) for u in us[:topn]]))
    return out


def gold_missing(methods, query_path, topn=10):
    """【2】【4】标准答案网页（在库里的）各方法缺失率、按名次拆开；以及答案网页全部缺失的 query 数。"""
    gold = load_gold(query_path, topn)
    keep = {u for _, us in gold for u in us}
    cc = chunk_counts(methods, "*", keep=keep)
    lib = set(cc[methods[0]])  # 各方法文档集合相同，看第一个方法即可
    n_all = len(keep)
    print(f"标准答案网页 {n_all} 个（去重），其中在库里 {len(lib)} 个（{_pct(len(lib), n_all)}%）")

    rows = []
    for m in methods:
        miss = sum(1 for u in lib if cc[m].get(u, 0) == 0)
        rows.append({"方法": m, "在库答案网页": len(lib), "缺失": miss, "缺失率%": _pct(miss, len(lib))})
    t_all = pd.DataFrame(rows)

    # 按名次（同一个 url 在不同 query 里名次不同，这里按 (query, 名次) 计数）
    by_rank = {m: Counter() for m in methods}
    tot_rank = Counter()
    for _, us in gold:
        for r, u in enumerate(us, 1):
            if u in lib:
                tot_rank[r] += 1
                for m in methods:
                    by_rank[m][r] += cc[m].get(u, 0) == 0
    t_rank = pd.DataFrame({m: {f"第{r}名": _pct(by_rank[m][r], tot_rank[r]) for r in sorted(tot_rank)} for m in methods})
    t_rank["在库答案数"] = pd.Series({f"第{r}名": tot_rank[r] for r in sorted(tot_rank)})

    # query 层面
    rows = []
    qs = [(qid, [u for u in us if u in lib]) for qid, us in gold]
    qs = [(qid, us) for qid, us in qs if us]
    for m in methods:
        all_miss = sum(1 for _, us in qs if all(cc[m].get(u, 0) == 0 for u in us))
        frac = np.mean([np.mean([cc[m].get(u, 0) == 0 for u in us]) for _, us in qs]) if qs else float("nan")
        rows.append({"方法": m, "可评测 query": len(qs), "答案网页全部缺失的 query": all_miss,
                     "占比%": _pct(all_miss, len(qs)), "平均每条 query 答案缺失比例%": round(100 * frac, 2)})
    return t_all, t_rank, pd.DataFrame(rows), cc, lib


def cross_missing(cc, urls, html_m, md_m, base_m):
    """【3】html 缺失时 md / 基线能否补上。"""
    h = {u for u in urls if cc[html_m].get(u, 0) == 0}
    md_ok = sum(1 for u in h if cc[md_m].get(u, 0) > 0)
    base_ok = sum(1 for u in h if cc[base_m].get(u, 0) > 0)
    all_miss = sum(1 for u in h if cc[md_m].get(u, 0) == 0 and cc[base_m].get(u, 0) == 0)
    n = len(urls)
    print(f"  {html_m} 缺失 {len(h)} 个（{_pct(len(h), n)}%）：其中 {md_m} 有块 {md_ok} 个，{base_m} 有块 {base_ok} 个，三者都缺 {all_miss} 个")
    print(f"  → 组合方案（html 优先，缺了退回 md）的缺失率：{_pct(len(h) - md_ok, n)}%")


# %% 运行
CHECK_METHODS = [m for m in ["content_br", "md_struct", "html_struct", "md_struct_secmerge_t512", "html_struct_secmerge_t512"]
                 if os.path.isdir(os.path.join(OUT_ROOT, m, "chunks"))]
GOOGLE_SRC = next(s for s in DATA_SOURCES if s["prefix"] == "ggl-")
GOOGLE_QUERY_FILE = QUERY_FILES["谷歌前10"]
pd.set_option("display.width", 200)
print("参与统计的方法：", CHECK_METHODS)

print("\n【1】谷歌网页这一批（ggl- 分片）")
t1, cc_ggl = batch_missing(CHECK_METHODS, "ggl-", raw_glob=GOOGLE_SRC["glob"])
print(t1.to_string(index=False))

print("\n【2】谷歌标准答案网页（在库里的）")
t2, t2_rank, t4, cc_gold, lib_gold = gold_missing(CHECK_METHODS, GOOGLE_QUERY_FILE)
print(t2.to_string(index=False))
print("\n按谷歌名次的缺失率%：")
print(t2_rank.to_string())

print("\n【3】html 缺失时能否由 md / 基线补上")
for hm, mm in [("html_struct", "md_struct"), ("html_struct_secmerge_t512", "md_struct_secmerge_t512")]:
    if hm in CHECK_METHODS and mm in CHECK_METHODS:
        print(" 谷歌网页这一批：")
        cross_missing(cc_ggl, set(cc_ggl[CHECK_METHODS[0]]), hm, mm, "content_br")
        print(" 谷歌标准答案网页：")
        cross_missing(cc_gold, lib_gold, hm, mm, "content_br")

print("\n【4】query 层面")
print(t4.to_string(index=False))
