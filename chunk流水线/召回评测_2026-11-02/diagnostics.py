# %% [markdown]
# # 评测结果诊断（接在 recall_pipeline.py 第 14 段之后运行）
#
# 直接用第 14 段留在内存里的 `hits`、`queries` 以及 notebook 里定义的函数，不重新检索。
#   诊断 1：按网页去重后的召回 —— 排除"切得越细，一个网页占越多名额"对 recall@K 的影响
#   诊断 2：标准答案网页在各方法下被切成什么样 —— 看有没有丢正文、0 块网页、块数分布是否异常
#   诊断 3：按标准答案网页的结构类型（表格 / 标题列表 / 纯文本）分组的配对比较 —— 区分"结构保护"与"块更短"的收益
# 用法：把下面各段贴进 notebook 的新单元格（或在 notebook 里 %run -i diagnostics.py）。

# %% 诊断 1：按网页去重后的召回
import numpy as np
import pandas as pd

rows = []
for m, (idx, scores, gids) in hits.items():
    for i, q in enumerate(queries):
        seen, pages = set(), []
        for g in gids[i]:
            if g >= 0:
                u = norm_url(idx.url(int(g)))
                if u not in seen:
                    seen.add(u)
                    pages.append(u)
        r = {"method": m, "top-K块里的不同网页数": len(pages)}
        for n in (1, 5, 10, 20):
            r[f"网页recall@{n}"] = len(set(pages[:n]) & set(q["rel"])) / len(q["rel"])
        rows.append(r)
print(pd.DataFrame(rows).groupby("method").mean().round(4).to_string())
# 注意：网页 recall@20 只能在 top-K 个 chunk 里数网页；某方法"不同网页数"明显少于 20 时，需把 TOPK 调到 300 重新检索再比

# %% 诊断 2：标准答案网页在各方法下的切分情况
gold = {u for q in queries for u in q["rel"]}
stat = {}
for m, (idx, _, _) in hits.items():
    found = {}
    for p, name in enumerate(idx.names):
        with open(os.path.join(idx.cdir, name + ".urls.txt"), encoding="utf-8") as f:
            for d, u in enumerate(f.read().split("\n")):
                nu = norm_url(u)
                if nu in gold and nu not in found:
                    found[nu] = (p, d)
    nc, ch = [], []
    for p, d in found.values():
        a, b = int(idx.offsets[p][d]), int(idx.offsets[p][d + 1])
        nc.append(b - a)
        ch.append(int(idx.lens[p][a:b].sum()))
    nc, ch = np.array(nc), np.array(ch)
    stat[m] = {"找到的标准答案网页": len(found), "块数/页 均值": nc.mean(), "块数/页 中位数": np.median(nc),
               "0块网页占比": (nc == 0).mean(), "正文字数/页 中位数": np.median(ch), "平均块长": ch.sum() / max(nc.sum(), 1)}
print(pd.DataFrame(stat).T.round(3).to_string())
for m in RUN_METHODS:  # 全库的切分统计（含随机库）
    print(m, json.load(open(os.path.join(OUT_ROOT, m, "chunk_stats.json"))))

# %% 诊断 3：按标准答案网页的结构类型分组比较
import re

GOLD_SOURCES = ["/path/to/injected/part-*"]  # 补入数据的目录；若有标准答案网页在随机库里，把随机库目录也加进来
gold = {u for q in queries for u in q["rel"]}
kind = {}  # 标准答案网页 → 表格 / 标题列表 / 纯文本（看 markdown 字段）
for pat in GOLD_SOURCES:
    for p in list_parts(pat):
        for rec in iter_records(p):
            u = norm_url(rec.get("url") or "")
            if u in gold and u not in kind:
                md = rec.get("markdown") or ""
                if "<table" in md or re.search(r"(?m)^\|.*\|\s*$", md):
                    kind[u] = "表格"
                elif re.search(r"(?m)^#{1,6} ", md) or re.search(r"(?m)^\s*([-*]|\d+[.)])\s", md):
                    kind[u] = "标题列表"
                else:
                    kind[u] = "纯文本"
print(f"标准答案网页 {len(gold)} 个，在 GOLD_SOURCES 里找到 {len(kind)} 个（没找到的按纯文本算）："
      + "，".join(f"{k} {v}" for k, v in pd.Series(list(kind.values())).value_counts().items()))
order = {"表格": 0, "标题列表": 1, "纯文本": 2}
qkind = [min((kind.get(u, "纯文本") for u in q["rel"]), key=order.get) for q in queries]  # 标准答案里有一个网页带表格就算"表格"
per = {m: [url_metrics(idx, gids[i], q["rel"]) for i, q in enumerate(queries)] for m, (idx, _, gids) in hits.items()}
for g in ("表格", "标题列表", "纯文本"):
    sel = [i for i, k in enumerate(qkind) if k == g]
    print(f"\n【{g}】query {len(sel)} 条")
    if not sel:
        continue
    for m in RUN_METHODS:
        out = []
        for k in ("MRR", "hit@10", "hit@3000字"):
            a = [per[m][i][k] for i in sel]
            b = [per[BASELINE_METHOD][i][k] for i in sel]
            d, lo, hi = paired_bootstrap(a, b)
            out.append(f"{k} {np.mean(a):.4f} Δ{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        print(f"  {m:12s} " + " | ".join(out))
