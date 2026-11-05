# %% [markdown]
# # 导出召回结果，给 recall_viewer.html 查看（接在 recall_pipeline.py 第 14 段之后运行）
#
# 直接用第 14 段留在内存里的 hits、queries，不重新检索。产出一个 .jsonl（或 .jsonl.gz）文件：
#   第 1 行 {"type": "meta", ...}：方法、每种方法导出的 chunk 数、生成时间等
#   之后每行一条 query：query 原文、标准答案网页（含每种方法下第一次命中的名次）、每种方法的指标与前 topn 个 chunk（原文、url、分数、是否命中）
# 用 recall_viewer.html 打开：页面上"选择文件"选中这个文件即可；换一份文件就换一套数据。
# 大小参考：3000 多条 query × 3 种方法 × 前 20 个 chunk，未压缩约 130MB，.gz 约 30MB。要看更深的名次，用 sample 只导出一部分 query。
# 若运行过诊断 3（diagnostics.py），会自动带上标准答案网页的结构类型（表格 / 标题列表 / 纯文本），查看器里可以按类型筛选。

# %% 定义
import gzip
import json
import random
import time

from tqdm.auto import tqdm


def export_viewer_data(hits, queries, path, topn=20, sample=None, qids=None, kind_of_url=None, seed=0):
    """hits / queries：第 14 段的结果；topn：每种方法导出前几个 chunk；
    sample：随机抽多少条 query（None = 全部）；qids：只导出这些 qid；kind_of_url：诊断 3 的 kind（网页 → 结构类型）。"""
    methods = list(hits.keys())
    sel = list(range(len(queries)))
    if qids is not None:
        want = set(qids)
        sel = [i for i in sel if queries[i]["qid"] in want]
    if sample is not None and sample < len(sel):
        sel = sorted(random.Random(seed).sample(sel, sample))
    kind_of_url = kind_of_url or {}
    order = {"表格": 0, "标题列表": 1, "纯文本": 2}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "methods": methods, "topn": topn, "topk": int(hits[methods[0]][2].shape[1]),
                            "n_queries": len(sel), "baseline": BASELINE_METHOD, "out_root": OUT_ROOT,
                            "query_file": QUERY_FILE, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                           ensure_ascii=False) + "\n")
        for i in tqdm(sel, desc="导出"):
            q = queries[i]
            rel = q["rel"]
            raw = list((q.get("relevance") or {}).keys()) or list(q.get("relevant_urls") or [])
            raw_of = {}
            for u in raw:  # 标准答案的原始 url（可点击）；rel 里是归一化后的
                if isinstance(u, str):
                    raw_of.setdefault(norm_url(u), u)
            gold = {nu: {"url": raw_of.get(nu, nu), "grade": g, "kind": kind_of_url.get(nu), "rank": {}}
                    for nu, g in rel.items()}
            out = {"type": "query", "qid": q["qid"], "query": q["query"], "methods": {}}
            kinds = [g["kind"] for g in gold.values() if g["kind"]]
            out["kind"] = min(kinds, key=order.get) if kinds else None
            for m in methods:
                idx, scores, gids = hits[m]
                row, sc = gids[i], scores[i]
                for r, g in enumerate(row):  # 每个标准答案网页在 top-K 里第一次出现的名次（K = 检索深度，不止导出的 topn）
                    if g < 0:
                        break
                    nu = norm_url(idx.url(int(g)))
                    if nu in gold and m not in gold[nu]["rank"]:
                        gold[nu]["rank"][m] = r + 1
                metrics = {k: round(float(v), 4) for k, v in url_metrics(idx, row, rel).items()} if rel else {}
                chunks = []
                for r, g in enumerate(row[:topn]):
                    if g < 0:
                        break
                    g = int(g)
                    p, d, c = idx.locate(g)
                    u = idx.url(g)
                    chunks.append({"rank": r + 1, "score": round(float(sc[r]), 4), "url": u, "len": idx.chunk_len(g),
                                   "gold": rel.get(norm_url(u), 0), "pos": c + 1,
                                   "of": int(idx.offsets[p][d + 1] - idx.offsets[p][d]), "text": idx.text(g)})
                out["methods"][m] = {"metrics": metrics, "chunks": chunks}
            out["gold"] = list(gold.values())
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"已导出 {len(sel)} 条 query → {path}")


# %% 运行：导出（改 sample / topn 按需取舍）
VIEWER_FILE = os.path.join(OUT_ROOT, "viewer_top20.jsonl.gz")
export_viewer_data(hits, queries, VIEWER_FILE, topn=20, sample=None, kind_of_url=globals().get("kind"))
