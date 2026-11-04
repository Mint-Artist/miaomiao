# %% [markdown]
# # 评测结果诊断（接在 recall_pipeline.py 第 14 段之后运行）
#
# 直接用第 14 段留在内存里的 `hits`、`queries` 以及 notebook 里定义的函数，不重新检索。
#   诊断 1：按网页去重后的召回 —— 排除"切得越细，一个网页占越多名额"对 recall@K 的影响
#   诊断 2：标准答案网页在各方法下被切成什么样 —— 看有没有丢正文、0 块网页、块数分布是否异常
#   诊断 3：按标准答案网页的结构类型（表格 / 标题列表 / 纯文本）分组的配对比较 —— 区分"结构保护"与"块更短"的收益
#           （按 query 分组：标准答案里有一个网页带结构就算带结构，比较粗）
#   诊断 4：更严格的标准答案（只取精排前 3 / 前 5）—— 看标准答案噪声对结论的影响
#   诊断 5：按每个标准答案网页自己的结构类型分组 —— 比诊断 3 更准
#   诊断 6：html 切不出块的标准答案网页，html 字段是什么样 —— 区分"数据缺失"与"解析问题"
#   诊断 7：只看 html 有内容的标准答案网页，html_struct 分别对比 content_br 和 md_struct（块长接近）—— 排除 html 数据缺失后，
#           检验 html 在有结构网页上的提升是否来自结构
# 诊断 4–7 依赖诊断 3 定义的 GOLD_SOURCES、gold、kind，须先运行诊断 3；诊断 7 还依赖诊断 5 的 rk、bd 和诊断 6 的 zero_html。
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

# %% 诊断 4：更严格的标准答案（只取精排前 3 / 前 5）
# 前提：query 文件里的 url 按精排名次从高到低排列（relevant_urls 的顺序 / relevance 的键顺序）
for k in (3, 5):
    qk = [{**q, "rel": dict(list(q["rel"].items())[:k])} for q in queries]
    per = {m: [url_metrics(idx, gids[i], q["rel"]) for i, q in enumerate(qk)] for m, (idx, _, gids) in hits.items()}
    print(f"\n标准答案 = 精排前 {k}")
    for m in RUN_METHODS:
        out = []
        for key in ("MRR", "hit@10", "hit@3000字"):
            a = [r[key] for r in per[m]]
            b = [r[key] for r in per[BASELINE_METHOD]]
            d, lo, hi = paired_bootstrap(a, b)
            out.append(f"{key} {np.mean(a):.4f} Δ{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        print(f"  {m:12s} " + " | ".join(out))

# %% 诊断 5：按每个标准答案网页自己的结构类型分组（3000 字预算内 / 前 10 个网页内能召回该类网页的比例）


def pages_of(idx, row, budget=None):
    """检索结果按网页去重后的顺序；给 budget 时只看累计字数在预算内的 chunk（至少一个，与 hit@N字 一致）。"""
    seen, out, tot = set(), [], 0
    for g in row:
        if g < 0:
            break
        tot += idx.chunk_len(int(g))
        if budget and tot > budget and out:
            break
        u = norm_url(idx.url(int(g)))
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


rk = {m: [pages_of(idx, gids[i]) for i in range(len(queries))] for m, (idx, _, gids) in hits.items()}
bd = {m: [set(pages_of(idx, gids[i], 3000)) for i in range(len(queries))] for m, (idx, _, gids) in hits.items()}
for g in ("表格", "标题列表", "纯文本"):
    res = {m: {"3000字内召回": [], "前10网页召回": []} for m in RUN_METHODS}
    for i, q in enumerate(queries):
        us = {u for u in q["rel"] if kind.get(u, "纯文本") == g}
        if us:  # 每条 query 只算它自己这一类标准答案网页的召回比例，再按 query 配对比较
            for m in RUN_METHODS:
                res[m]["3000字内召回"].append(len(us & bd[m][i]) / len(us))
                res[m]["前10网页召回"].append(len(us & set(rk[m][i][:10])) / len(us))
    print(f"\n【{g}网页】涉及 query {len(res[RUN_METHODS[0]]['3000字内召回'])} 条")
    if not res[RUN_METHODS[0]]["3000字内召回"]:
        continue
    for m in RUN_METHODS:
        out = []
        for key in ("3000字内召回", "前10网页召回"):
            d, lo, hi = paired_bootstrap(res[m][key], res[BASELINE_METHOD][key])
            out.append(f"{key} {np.mean(res[m][key]):.4f} Δ{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        print(f"  {m:12s} " + " | ".join(out))

# %% 诊断 6：html 切不出块的标准答案网页，html 字段是什么样
import collections

zero_html = set()
idx = hits["html_struct"][0]
for p, name in enumerate(idx.names):
    offs = idx.offsets[p]
    with open(os.path.join(idx.cdir, name + ".urls.txt"), encoding="utf-8") as f:
        for d, u in enumerate(f.read().split("\n")[: len(offs) - 1]):
            if offs[d + 1] == offs[d] and norm_url(u) in gold:
                zero_html.add(norm_url(u))
cnt, samples, seen6 = collections.Counter(), [], set()
for pat in GOLD_SOURCES:
    for p in list_parts(pat):
        for rec in iter_records(p):
            u = norm_url(rec.get("url") or "")
            if u in zero_html and u not in seen6:  # 同一网页可能在多个来源里出现，只数一次
                seen6.add(u)
                h = rec.get("html") or ""
                cnt["html 为空" if not h.strip() else "html 很短（<200 字符）" if len(h) < 200 else "html 有内容"] += 1
                if h.strip() and len(samples) < 3:
                    samples.append((rec["url"], len(h), h[:300]))
print(f"html 切不出块的标准答案网页 {len(zero_html)} 个：", dict(cnt))
for s in samples:  # 大多数是"html 有内容"时，说明是解析问题，把这几条样例发出来看
    print(s)

# %% 诊断 7：只看 html 有内容的标准答案网页，html_struct 直接对比 md_struct（两者块长接近）
for base in ("content_br", "md_struct"):
    print(f"\n===== 基线：{base}（排除 html 切不出块的 {len(zero_html)} 个网页）=====")
    for g in ("表格", "标题列表", "纯文本"):
        res = {m: {"3000字内召回": [], "前10网页召回": []} for m in RUN_METHODS}
        for i, q in enumerate(queries):
            us = {u for u in q["rel"] if kind.get(u, "纯文本") == g and u not in zero_html}
            if us:
                for m in RUN_METHODS:
                    res[m]["3000字内召回"].append(len(us & bd[m][i]) / len(us))
                    res[m]["前10网页召回"].append(len(us & set(rk[m][i][:10])) / len(us))
        print(f"【{g}网页】涉及 query {len(res[base]['3000字内召回'])} 条")
        if not res[base]["3000字内召回"]:
            continue
        for m in RUN_METHODS:
            if m == base:
                continue
            out = []
            for key in ("3000字内召回", "前10网页召回"):
                d, lo, hi = paired_bootstrap(res[m][key], res[base][key])
                out.append(f"{key} Δ{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            print(f"  {m:12s} vs {base:10s} " + " | ".join(out))
