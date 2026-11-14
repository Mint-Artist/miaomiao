# %% [markdown]
# # 用谷歌 top10 作为标准答案评测（接在 recall_pipeline.py 第 1–2 段之后运行）
#
# 输入两份文件：
#   A. 网页文件（jsonl）：每行 {"url", "content", "markdown", "html"}，是谷歌结果网页的正文
#   B. 排名文件：每行一条 (query, rank, url)；支持 jsonl（字段 query / rank / url）、tsv、csv（有无表头都行，无表头时按 query, rank, url 的列序）
# 两份文件不需要互相拼接：
#   - B 和抽样 query 文件（queries_sample1000_html.jsonl）按 query 文本对上 qid，生成新的 query 文件（标准答案 = 谷歌 top10，按名次排序）；
#   - A 作为新一批网页补入网页库（和之前补入 query 相关网页的流程完全相同），原有网页的块和向量不动。
#
# 流程（详见文末"运行"各段）：
#   1. 预处理：生成 query 文件、把网页文件切成分片、统计覆盖率
#   2. 补入网页库：DATA_GLOB 指向分片目录，PART_PREFIX="ggl-"，DEDUP_URLS=True，运行第 9 段；任务队列增量推理向量
#   3. 评测：QUERY_FILE 指向新 query 文件，运行第 13–14 段，再运行本文件"评测"段（只看 html 有内容的网页）
# 第一轮用原有三种方法；之后改好的方法（如 chunk_merge.py 的 *_merge）用新方法名另切另推理，四批网页（随机库、inj-、ggl-）
# 都要切，再和原方法放在一起评测。

# %% 定义
import csv
import gzip
import json
import os
from collections import defaultdict


def _norm_q(q):
    return " ".join(str(q).split())


def _open(path, mode="rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def read_rank_file(path):
    """读排名文件，返回 [(query, rank, url)]。jsonl 按字段名读；tsv / csv 自动判断分隔符和表头。"""
    rows, bad = [], 0
    if path.endswith((".jsonl", ".jsonl.gz", ".json")):
        with _open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    q = d.get("query") or d.get("q")
                    r = d.get("rank", d.get("position", d.get("pos")))
                    u = d.get("url") or d.get("link")
                    rows.append((q, int(r), u))
                except Exception:
                    bad += 1
    else:
        with _open(path) as f:
            sample = f.read(65536)
            f.seek(0)
            delim = "\t" if sample.count("\t") >= sample.count(",") else ","
            reader = csv.reader(f, delimiter=delim)
            first = next(reader, None)
            cols = [c.strip().lower() for c in first] if first else []
            if {"query", "rank", "url"} <= set(cols):
                iq, ir, iu = cols.index("query"), cols.index("rank"), cols.index("url")
            else:
                iq, ir, iu = 0, 1, 2
                reader = [first] + list(reader) if first else []
            for row in reader:
                try:
                    rows.append((row[iq], int(float(row[ir])), row[iu]))
                except Exception:
                    bad += 1
    rows = [(q, r, u.strip()) for q, r, u in rows if q and u and str(u).strip()]
    print(f"排名文件 {path}：有效 {len(rows)} 行，跳过 {bad} 行")
    return rows


def build_google_queries(rank_path, sample_path, out_path, topn=10):
    """排名文件 + 抽样 query 文件 → 新 query 文件：{"qid", "query", "relevant_urls"（谷歌名次顺序，去重后取前 topn）,
    "rerank_urls"（原来的精排标准答案，仅供对照）}。按 query 文本（合并空白后）对上 qid。"""
    sample = [json.loads(l) for l in _open(sample_path) if l.strip()]
    by_text = defaultdict(list)
    for d in sample:
        by_text[_norm_q(d["query"])].append(d)
    dup = sum(1 for v in by_text.values() if len(v) > 1)
    groups = defaultdict(list)
    for q, r, u in read_rank_file(rank_path):
        groups[_norm_q(q)].append((r, u))
    n_out, unmatched, short = 0, [], 0
    with open(out_path, "w", encoding="utf-8") as f:
        for qt, items in groups.items():
            if qt not in by_text:
                unmatched.append(qt)
                continue
            urls, seen = [], set()
            for r, u in sorted(items, key=lambda x: x[0]):
                nu = norm_url(u)
                if nu not in seen:
                    seen.add(nu)
                    urls.append(u)
            urls = urls[:topn]
            short += len(urls) < topn
            for d in by_text[qt]:  # 文本完全相同的 query 共用同一组谷歌结果
                f.write(json.dumps({"qid": d["qid"], "query": d["query"], "relevant_urls": urls,
                                    "rerank_urls": d.get("relevant_urls") or list((d.get("relevance") or {}).keys())},
                                   ensure_ascii=False) + "\n")
                n_out += 1
    missing = len(set(by_text) - set(groups))
    print(f"抽样 query {len(sample)} 条（文本重复 {dup} 组）；排名文件 query {len(groups)} 条")
    print(f"写出 {n_out} 条 → {out_path}；排名文件里对不上抽样 query 的 {len(unmatched)} 条，抽样里没有谷歌结果的 {missing} 条，"
          f"谷歌结果不足 {topn} 个的 {short} 条")
    for qt in unmatched[:5]:
        print("  对不上：", qt[:60])
    return out_path


def prepare_google_pages(pages_path, out_dir, shard_size=50000):
    """网页文件 → 分片（out_dir/part-xxxxx.jsonl），供第 9 段补入。去掉无 url、文件内重复 url、三个正文字段全空的记录。"""
    os.makedirs(out_dir, exist_ok=True)
    if any(n.startswith("part-") for n in os.listdir(out_dir)):
        raise ValueError(f"{out_dir} 里已有分片；为避免混入旧数据，请换一个空目录")
    st = {"读入": 0, "解析失败": 0, "无url": 0, "文件内重复": 0, "正文全空": 0, "写出": 0, "其中html为空": 0}
    seen, shard, n_in_shard, fout = set(), 0, 0, None
    with _open(pages_path) as f:
        for line in f:
            if not line.strip():
                continue
            st["读入"] += 1
            try:
                d = json.loads(line)
            except Exception:
                st["解析失败"] += 1
                continue
            u = d.get("url") or ""
            if not u.strip():
                st["无url"] += 1
                continue
            nu = norm_url(u)
            if nu in seen:
                st["文件内重复"] += 1
                continue
            seen.add(nu)
            if not any((d.get(k) or "").strip() for k in ("content", "markdown", "html")):
                st["正文全空"] += 1
                continue
            st["其中html为空"] += not (d.get("html") or "").strip()
            if fout is None or n_in_shard >= shard_size:
                if fout:
                    fout.close()
                fout = open(os.path.join(out_dir, f"part-{shard:05d}.jsonl"), "w", encoding="utf-8")
                shard, n_in_shard = shard + 1, 0
            fout.write(json.dumps({k: d.get(k) for k in ("url", "content", "markdown", "html")}, ensure_ascii=False) + "\n")
            n_in_shard += 1
            st["写出"] += 1
    if fout:
        fout.close()
    print("，".join(f"{k} {v}" for k, v in st.items()), f"；分片 {shard} 个 → {out_dir}")
    return os.path.join(out_dir, "part-*")


def coverage_report(query_path, pages_glob, ref_method="content_br"):
    """谷歌标准答案网页的覆盖情况：有没有正文、有没有 html、是否已在原网页库里。补入前后都可以跑。"""
    have_lib = set()
    for fu in glob.glob(os.path.join(OUT_ROOT, ref_method, "chunks", "*.urls.txt")):
        with open(fu, encoding="utf-8") as f:
            have_lib.update(norm_url(u) for u in f.read().split("\n") if u)
    page_html = {}
    for p in sorted(glob.glob(pages_glob)):
        for d in iter_records(p):
            page_html[norm_url(d.get("url") or "")] = bool((d.get("html") or "").strip())
    qs = [json.loads(l) for l in open(query_path, encoding="utf-8") if l.strip()]
    tot = {"标准答案网页": 0, "已在原网页库": 0, "在谷歌网页文件里": 0, "有html（谷歌网页文件）": 0, "两处都没有": 0}
    per_q_html = []
    for q in qs:
        n_html = 0
        for u in q["relevant_urls"]:
            nu = norm_url(u)
            tot["标准答案网页"] += 1
            tot["已在原网页库"] += nu in have_lib
            tot["在谷歌网页文件里"] += nu in page_html
            tot["有html（谷歌网页文件）"] += page_html.get(nu, False)
            tot["两处都没有"] += nu not in have_lib and nu not in page_html
            n_html += page_html.get(nu, False)
        per_q_html.append(n_html)
    t = max(tot["标准答案网页"], 1)
    print("，".join(f"{k} {v}（{100 * v / t:.1f}%）" if k != "标准答案网页" else f"{k} {v}" for k, v in tot.items()))
    print(f"每条 query 有 html 的谷歌网页数：平均 {sum(per_q_html) / max(len(qs), 1):.2f}；一个都没有的 query {sum(n == 0 for n in per_q_html)} 条")


def strict_queries(queries, k):
    """更严格的标准答案：只保留谷歌前 k 名（按 relevant_urls 的顺序；先过滤再取前 k，所以是"前 k 个仍可评测的网页"）。"""
    out = []
    for q in queries:
        order = [norm_url(u) for u in q.get("relevant_urls", [])]
        keep = [u for u in order if u in q["rel"]][:k]
        if keep:
            out.append({**q, "rel": {u: q["rel"][u] for u in keep}})
    return out


def report_html_only(hits, queries, baseline=None, keys=None, strict=(3, 5)):
    """只看 html 有内容的网页（口径同 query_sample.py：标准答案和检索结果都去掉没有 html 的网页），
    输出各方法指标和相对基线的差值；另外给出只认谷歌前 3 / 前 5 的严格口径。需要先运行 query_sample.py 的"定义"段。"""
    import pandas as pd
    baseline = baseline or BASELINE_METHOD
    keys = keys or ["MRR", "hit@1", "hit@10", "recall@100", f"hit@{CHAR_BUDGETS[1]}字", "nDCG@10"]
    hits_h, qs_h = restrict_to_html(hits, queries, html_ok_urls(hits))
    pos = {q["qid"]: i for i, q in enumerate(qs_h)}
    for name, qs in [("谷歌前 10", qs_h)] + [(f"谷歌前 {k}", strict_queries(qs_h, k)) for k in strict]:
        sel = [pos[q["qid"]] for q in qs]
        rows = []
        for m, (idx, _s, gids) in hits_h.items():
            per = [url_metrics(idx, gids[i], q["rel"]) for i, q in zip(sel, qs)]
            rows.append((m, per))
        base = dict(rows)[baseline]
        tab = []
        for m, per in rows:
            r = {"方法": m}
            for k in keys:
                r[k] = round(sum(p[k] for p in per) / max(len(per), 1), 4)
            for k in ("MRR", "hit@10", f"hit@{CHAR_BUDGETS[1]}字"):
                if m == baseline:
                    r["Δ" + k] = "—"
                else:
                    d, lo, hi = paired_bootstrap([p[k] for p in per], [b[k] for b in base])
                    r["Δ" + k] = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            tab.append(r)
        print(f"\n===== 标准答案 = {name}（只看有 html 的网页），query {len(qs)} 条 =====")
        print(pd.DataFrame(tab).to_string(index=False))


# %% 运行 1：预处理（只需运行一次；路径按实际修改）
GOOGLE_RANK_FILE = "/path/to/google_rank.tsv"                       # 排名文件：query, rank, url
GOOGLE_PAGES_FILE = "/path/to/google_pages.jsonl"                   # 网页文件：url, content, markdown, html
SAMPLE_QUERY_FILE = os.path.join(OUT_ROOT, "queries_sample1000_html.jsonl")
GOOGLE_QUERY_FILE = os.path.join(OUT_ROOT, "queries_google_top10.jsonl")
GOOGLE_PAGES_DIR = "/path/to/google_pages_parts"                    # 新的空目录，存放切好的网页分片

build_google_queries(GOOGLE_RANK_FILE, SAMPLE_QUERY_FILE, GOOGLE_QUERY_FILE, topn=10)
GOOGLE_DATA_GLOB = prepare_google_pages(GOOGLE_PAGES_FILE, GOOGLE_PAGES_DIR)
coverage_report(GOOGLE_QUERY_FILE, GOOGLE_DATA_GLOB)

# %% 运行 2：补入网页库 —— 不在这里运行。改 recall_pipeline.py 第 1 段后重启 kernel、从第 1 段开始依次运行到第 9 段：
#   DATA_GLOB = GOOGLE_DATA_GLOB 的值（如 "/path/to/google_pages_parts/part-*"）
#   PART_PREFIX = "ggl-"
#   DEDUP_URLS = True          # 已在库里的 url 跳过，沿用原来的块和向量
#   RUN_METHODS = ["content_br", "md_struct", "html_struct"]
# 然后用任务队列增量推理：bash run_embed_job_multigpu.sh content_br md_struct html_struct（已完成的分片自动跳过）

# %% 运行 3：评测（第 1 段改 QUERY_FILE = GOOGLE_QUERY_FILE 的值，运行第 10、12、13、14 段，再运行 query_sample.py 的"定义"段，然后运行这里）
report_html_only(hits, queries)
