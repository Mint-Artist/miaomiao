# %% [markdown]
# # 抽样 query（接在 recall_pipeline.py 第 14 段之后运行）
#
# 用途：从当前评测的 query 里抽一批（默认 1000 条），送去友商搜索引擎取精排结果；同时检查这批抽样 query 上的评测结果
# 和全量是否一致（抽样是否有代表性）。直接用第 14 段留在内存里的 queries、hits，不重新检索。
#
# 只在 html 有内容的网页上评测（ONLY_HTML = True，口径同诊断 7）：
#   - "有 html" = html_struct 至少切出 1 块的网页（诊断 6：切不出块的网页 98.6% 是 html 字段为空）；
#   - 标准答案只保留有 html 的网页，一个都不剩的 query 去掉，抽样只从剩下的 query 里抽；
#   - 各方法的检索结果里也去掉没有 html 的网页（相当于在"有 html 的网页库"里检索），三种方法在同一个网页集合上比较。
#   之后换成友商结果作为标准答案时，同样先用 restrict_to_html 过滤一遍。
#
# 产出（都在 OUT_ROOT 下，文件名带 _html 表示是只看 html 网页的那一批）：
#   queries_sample1000_html.jsonl           原格式的 query 记录（qid、query、relevant_urls / relevance），之后可直接作为 QUERY_FILE
#   queries_sample1000_html_external.tsv    只有 qid 和 query 两列，送给友商检索用（不含我们的标准答案）
#   queries_sample1000_html.idx.json        抽中的 qid、随机种子、抽样时间，用于复现；以后不要重抽，固定用这一批
#   viewer_sample1000.jsonl.gz              这批 query 的查看器数据（可选，需要先运行 viewer_export.py 的定义段）
#
# 注意：1000 条 query 的置信区间约为全量（3377 条）的 1.8 倍宽。全量上 +2.9 [+1.7, +4.1] 这样的差异，
# 在 1000 条上区间大约是 ±2.2 个百分点，接近显著的边缘。看结果时以差值方向是否一致为主，不要只看是否显著。

# %% 定义
import json
import random
import time

import numpy as np


def html_ok_urls(hits, method="html_struct"):
    """库里 html_struct 至少切出 1 块的网页（归一化 url 集合），作为"有 html"的判定。"""
    idx = hits[method][0]
    ok = set()
    for p, name in enumerate(idx.names):
        offs = idx.offsets[p]
        with open(os.path.join(idx.cdir, name + ".urls.txt"), encoding="utf-8") as f:
            for d, u in enumerate(f.read().split("\n")[: len(offs) - 1]):
                if offs[d + 1] > offs[d]:
                    ok.add(norm_url(u))
    return ok


def restrict_to_html(hits, queries, ok):
    """只看有 html 的网页：标准答案去掉没有 html 的网页，一个都不剩的 query 去掉；
    各方法的检索结果里去掉没有 html 的网页（后面的结果前移，末尾补 -1）。返回新的 (hits, queries)，原变量不变。"""
    keep, out_q, n_rel, n_rel_ok = [], [], 0, 0
    for i, q in enumerate(queries):
        rel = {u: g for u, g in q["rel"].items() if u in ok}
        n_rel += len(q["rel"])
        n_rel_ok += len(rel)
        if rel:
            keep.append(i)
            out_q.append({**q, "rel": rel})
    out_h = {}
    for m, (idx, scores, gids) in hits.items():
        g2 = np.full((len(keep), gids.shape[1]), -1, dtype=gids.dtype)
        s2 = np.full((len(keep), gids.shape[1]), -np.inf, dtype=np.float32) if scores is not None else None
        dropped = 0
        for r, i in enumerate(keep):
            row = [(g, scores[i][c] if scores is not None else 0.0) for c, g in enumerate(gids[i])
                   if g >= 0 and norm_url(idx.url(int(g))) in ok]
            dropped += int((gids[i] >= 0).sum()) - len(row)
            for c, (g, s) in enumerate(row):
                g2[r, c] = g
                if s2 is not None:
                    s2[r, c] = s
        out_h[m] = (idx, s2, g2)
        print(f"  {m:12s} 检索结果中去掉没有 html 的网页的块：平均每条 query {dropped / max(len(keep), 1):.1f} 个（共 {gids.shape[1]} 个）")
    print(f"有 html 的网页 {len(ok)} 个；标准答案网页 {n_rel} → {n_rel_ok} 个（去掉 {n_rel - n_rel_ok} 个没有 html 的）；"
          f"query {len(queries)} → {len(out_q)} 条")
    return out_h, out_q


def sample_queries(queries, n=1000, seed=0):
    """简单随机抽样（不放回），返回 queries 里的下标（升序）。"""
    n = min(n, len(queries))
    return sorted(random.Random(seed).sample(range(len(queries)), n))


def _per_query_metrics(hits, queries, sel):
    """每种方法、每条 query 的指标（用第 14 段的检索结果直接算，不重新检索）。"""
    out = {}
    for m, (idx, _scores, gids) in hits.items():
        out[m] = {i: url_metrics(idx, gids[i], queries[i]["rel"]) for i in sel}
    return out


def subset_check(hits, queries, sel, baseline=None, keys=None, kind_of_url=None):
    """对比"全量"和"抽样"两组 query：各方法的平均指标、相对基线的差值（配对 bootstrap 95% 置信区间），
    以及 query 本身的分布（标准答案网页数、query 长度、标准答案网页的结构类型）。"""
    import pandas as pd
    baseline = baseline or BASELINE_METHOD
    keys = keys or ["MRR", "hit@10", "recall@100", f"hit@{CHAR_BUDGETS[1]}字"]
    full = list(range(len(queries)))
    pq = _per_query_metrics(hits, queries, full)
    rows = []
    for name, s in (("全量", full), ("抽样", sel)):
        for m in hits:
            r = {"集合": name, "方法": m, "query 数": len(s)}
            for k in keys:
                r[k] = round(float(np.mean([pq[m][i][k] for i in s])), 4)
            for k in keys:
                if m == baseline:
                    r["Δ" + k] = "—"
                    continue
                d, lo, hi = paired_bootstrap([pq[m][i][k] for i in s], [pq[baseline][i][k] for i in s])
                r["Δ" + k] = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            rows.append(r)
    df = pd.DataFrame(rows)
    print("===== 指标：全量 vs 抽样（Δ 为相对", baseline, "的百分点差，方括号为 95% 置信区间）=====")
    print(df.to_string(index=False))

    def dist(s):
        n_gold = [len(queries[i]["rel"]) for i in s]
        q_len = [len(queries[i]["query"]) for i in s]
        d = {"query 数": len(s), "标准答案网页数/条 均值": round(float(np.mean(n_gold)), 2),
             "query 长度 中位数": float(np.median(q_len))}
        if kind_of_url:
            cnt = {}
            for i in s:
                for u in queries[i]["rel"]:
                    k = kind_of_url.get(u, "纯文本")
                    cnt[k] = cnt.get(k, 0) + 1
            tot = sum(cnt.values()) or 1
            for k in ("表格", "标题列表", "纯文本"):
                d[f"{k}网页占比"] = f"{100 * cnt.get(k, 0) / tot:.1f}%"
        return d

    print("\n===== query 分布：全量 vs 抽样 =====")
    print(pd.DataFrame([{"集合": "全量", **dist(full)}, {"集合": "抽样", **dist(sel)}]).to_string(index=False))
    return df


def write_query_files(queries, sel, out_dir, name, seed):
    """写出抽样结果：原格式 jsonl（去掉内部字段 rel）、只含 qid 和 query 的 tsv、抽样记录。返回三个路径。"""
    os.makedirs(out_dir, exist_ok=True)
    p_jsonl = os.path.join(out_dir, f"{name}.jsonl")
    p_tsv = os.path.join(out_dir, f"{name}_external.tsv")
    p_idx = os.path.join(out_dir, f"{name}.idx.json")
    with open(p_jsonl, "w", encoding="utf-8") as f:
        for i in sel:
            f.write(json.dumps({k: v for k, v in queries[i].items() if k != "rel"}, ensure_ascii=False) + "\n")
    with open(p_tsv, "w", encoding="utf-8") as f:
        f.write("qid\tquery\n")
        for i in sel:
            q = " ".join(str(queries[i]["query"]).split())  # 去掉 query 里的换行和制表符，避免破坏 tsv
            f.write(f"{queries[i]['qid']}\t{q}\n")
    with open(p_idx, "w", encoding="utf-8") as f:
        json.dump({"n": len(sel), "seed": seed, "from_n_queries": len(queries), "query_file": QUERY_FILE,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S"), "index": sel,
                   "qids": [queries[i]["qid"] for i in sel]}, f, ensure_ascii=False)
    print(f"已写出 {len(sel)} 条：\n  {p_jsonl}\n  {p_tsv}\n  {p_idx}")
    return p_jsonl, p_tsv, p_idx


# %% 运行：抽样 + 检查（SAMPLE_SEED 固定后不要再改，保证以后拿到的是同一批 query）
SAMPLE_N = 1000
SAMPLE_SEED = 20261113
ONLY_HTML = True  # 只在 html 有内容的网页上评测；False 时用全部网页
SAMPLE_NAME = f"queries_sample{SAMPLE_N}" + ("_html" if ONLY_HTML else "")

if ONLY_HTML:
    HTML_OK = html_ok_urls(hits)
    hits_eval, queries_eval = restrict_to_html(hits, queries, HTML_OK)
else:
    hits_eval, queries_eval = hits, queries

_idx_file = os.path.join(OUT_ROOT, f"{SAMPLE_NAME}.idx.json")
if os.path.exists(_idx_file):  # 已经抽过：沿用当时那一批，不重抽
    _saved = json.load(open(_idx_file, encoding="utf-8"))
    _pos = {q["qid"]: i for i, q in enumerate(queries_eval)}
    SAMPLE_IDX = sorted(_pos[qid] for qid in _saved["qids"] if qid in _pos)
    print(f"沿用已有抽样 {_idx_file}：{len(SAMPLE_IDX)} / {len(_saved['qids'])} 条在当前 queries 里")
else:
    SAMPLE_IDX = sample_queries(queries_eval, SAMPLE_N, SAMPLE_SEED)
    write_query_files(queries_eval, SAMPLE_IDX, OUT_ROOT, SAMPLE_NAME, SAMPLE_SEED)

subset_check(hits_eval, queries_eval, SAMPLE_IDX, kind_of_url=globals().get("kind"))

# %% 可选：导出这批 query 的查看器数据（先运行 viewer_export.py 的"定义"段）
# export_viewer_data(hits_eval, queries_eval, os.path.join(OUT_ROOT, "viewer_sample1000.jsonl.gz"), topn=20,
#                    qids=[queries_eval[i]["qid"] for i in SAMPLE_IDX], kind_of_url=globals().get("kind"))
