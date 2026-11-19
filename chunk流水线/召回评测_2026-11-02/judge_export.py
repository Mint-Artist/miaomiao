# %% [markdown]
# # 截断与完整性评测：数据抽取（LLM 判断与汇总交给 code agent，见同目录 LLM判断评测_指南.md）
#
# 两份数据：
#   1. query 视角：抽 300 条 query，各方法召回结果的前 10 块 → 让 LLM 判断"能否回答、答案有没有被切断、答案是否落在向量模型看不到的部分"
#   2. 页面视角：抽一批网页，各方法的完整切分 → 让 LLM 判断每个切点是否切坏、每块能否独立读懂
# 两份数据里每块都带"向量模型截断"信息：按推理时同样的方式（DOC_PREFIX + 文本，含特殊 token 共 MAX_LENGTH）
# 算出 token 数和模型实际能看到的字数（visible_chars），不需要 LLM。
#
# 运行顺序：
#   query 视角：v2 第 1–7、10、12、13、14 段跑完（RESULTS 在内存里）→ 本文件"定义"段 → "运行 1"
#   页面视角：v2 第 1–7 段 → chunk_adhere.py / chunk_container.py / chunk_compare_export.py 的"定义"段 → 本文件"定义"段 → "运行 2"
# 只读、不改任何已有结果；输出 .jsonl.gz。

# %% 定义
import gzip
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

BOUNDARY_WIN = 150    # 页面视角：每个切点取前一块末尾、后一块开头各多少字给 LLM 看
_TOK, _N_SPECIAL = None, None


def get_tok():
    global _TOK, _N_SPECIAL
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=TRUST_REMOTE_CODE)
        _N_SPECIAL = len(_TOK("", add_special_tokens=True)["input_ids"])
        print(f"tokenizer：{type(_TOK).__name__}（fast={getattr(_TOK, 'is_fast', False)}），特殊 token {_N_SPECIAL} 个，"
              f"正文可用 {MAX_LENGTH - _N_SPECIAL} 个 token（MAX_LENGTH={MAX_LENGTH}，DOC_PREFIX={DOC_PREFIX!r}）")
    return _TOK


def trunc_info(text, prefix=None, max_length=None):
    """与向量推理相同的截断：prefix + text，含特殊 token 共 max_length。返回 (正文 token 数, 模型能看到的正文字数)。"""
    tok = get_tok()
    prefix = DOC_PREFIX if prefix is None else prefix
    budget = (max_length or MAX_LENGTH) - _N_SPECIAL
    full = prefix + text
    if getattr(tok, "is_fast", False):
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if len(ids) <= budget:
            return len(ids), len(text)
        cut = offs[budget - 1][1]
    else:  # 慢速 tokenizer 没有字符偏移：二分找能放下的最长前缀
        ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return len(ids), len(text)
        lo, hi = 0, len(full)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tok(full[:mid], add_special_tokens=False)["input_ids"]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        cut = lo
    return len(ids), max(0, min(len(text), cut - len(prefix)))


def _chunk_info(text):
    n_tok, vis = trunc_info(text)
    return {"chunk_id": hashlib.md5(text.encode("utf-8")).hexdigest()[:16], "text": text,
            "chars": len(text.replace("\n", "")), "tokens": n_tok, "visible_chars": vis, "truncated": vis < len(text)}


def _open_out(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w", encoding="utf-8")


def _meta(kind, methods, **extra):
    return {"type": "meta", "kind": kind, "methods": methods, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path_name": os.path.basename(str(MODEL_PATH).rstrip("/")), "max_length": MAX_LENGTH,
            "doc_prefix": DOC_PREFIX, **extra}


# ---------------- 1. query 视角 ----------------
def export_query_judge(path, set_name="谷歌前10", n=300, methods=None, topk=10, html_only=True, seed=0):
    """从 RESULTS[set_name] 抽 n 条 query，导出各方法前 topk 块（含截断信息）。
    html_only=True：与第 14 段"只看有 html 的网页"同口径（去掉没有 html 的网页，各方法在同一批网页上比较）。
    每行一条 query：{"type": "query", qid, query, gold_urls, methods: {方法: [{rank, url, is_gold, pos, of, chunk_id, text, chars, tokens, visible_chars, truncated}]}}"""
    queries, _summary, hits = RESULTS[set_name]
    methods = methods or list(hits)
    if html_only:
        hits, queries = restrict_to_html(hits, queries, html_ok_urls(hits))
    cand = [i for i, q in enumerate(queries) if q["rel"]]
    sel = sorted(random.Random(seed).sample(cand, min(n, len(cand))))
    stat = defaultdict(lambda: defaultdict(int))
    with _open_out(path) as f:
        f.write(json.dumps(_meta("query", methods, set_name=set_name, n=len(sel), topk=topk, html_only=html_only,
                                 seed=seed), ensure_ascii=False) + "\n")
        for i in tqdm(sel, desc="导出 query"):
            q = queries[i]
            out = {"type": "query", "qid": q["qid"], "query": q["query"], "gold_urls": sorted(q["rel"]), "methods": {}}
            for m in methods:
                idx, _s, gids = hits[m]
                rows = []
                for g in gids[i]:
                    if g < 0 or len(rows) >= topk:
                        continue
                    g = int(g)
                    p, d, c = idx.locate(g)
                    u = idx.url(g)
                    ci = _chunk_info(idx.text(g))
                    rows.append({"rank": len(rows) + 1, "url": u, "is_gold": q["rel"].get(norm_url(u), 0), "pos": c + 1,
                                 "of": int(idx.offsets[p][d + 1] - idx.offsets[p][d]), **ci})
                    s = stat[m]
                    s["块"] += 1
                    s["截断块"] += ci["truncated"]
                    s["字"] += len(ci["text"])
                    s["看不到的字"] += len(ci["text"]) - ci["visible_chars"]
                    s["标准答案块"] += bool(rows[-1]["is_gold"])
                    s["被截断的标准答案块"] += bool(rows[-1]["is_gold"]) and ci["truncated"]
                out["methods"][m] = rows
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"已导出 {len(sel)} 条 query → {path}")
    print(f"\n各方法前 {topk} 块的向量截断情况（不需要 LLM）：")
    for m in methods:
        s = stat[m]
        print(f"  {m:28s} 截断块 {100 * s['截断块'] / max(s['块'], 1):5.1f}%  模型看不到的字 {100 * s['看不到的字'] / max(s['字'], 1):5.1f}%  "
              f"标准答案块中被截断 {100 * s['被截断的标准答案块'] / max(s['标准答案块'], 1):5.1f}%（{s['被截断的标准答案块']}/{s['标准答案块']}）")


# ---------------- 2. 页面视角 ----------------
def _page_methods_with_trunc(page, win=BOUNDARY_WIN):
    out = {}
    for m, d in page["methods"].items():
        cs = [{**_chunk_info(c["text"]), "i": j + 1, **({"kind": c["kind"]} if c.get("kind") else {})}
              for j, c in enumerate(d.get("chunks", []))]
        bs = [{"i": j + 1, "tail": cs[j]["text"][-win:], "head": cs[j + 1]["text"][:win]} for j in range(len(cs) - 1)]
        out[m] = {"chunks": cs, "boundaries": bs, "stats": d.get("stats", {})}
    return out


def export_page_judge(path, n=100, urls=None, methods=None, require="container", seed=0):
    """抽 n 个网页（或按 urls 取），各方法完整切分 + 截断信息 + 相邻块切点（前一块末尾 / 后一块开头各 BOUNDARY_WIN 字）。
    同时在同目录写一份 chunk_viewer.html 能打开的对比文件（文件名加 .viewer），方便人工对照。
    每行一个网页：{"type": "page", url, source, methods: {方法: {chunks: [...], boundaries: [...], stats}}}"""
    if "build_page" not in globals():
        raise NameError("请先运行 chunk_adhere.py、chunk_container.py、chunk_compare_export.py 的“定义”段")
    methods = methods or COMPARE_METHODS
    vpath = path.replace(".jsonl", ".viewer.jsonl")
    pages = export_urls(urls, vpath, methods) if urls else export_sample(vpath, n=n, methods=methods, require=require, seed=seed)
    stat = defaultdict(lambda: defaultdict(int))
    with _open_out(path) as f:
        f.write(json.dumps(_meta("page", methods, n=len(pages), require=require, seed=seed,
                                 boundary_win=BOUNDARY_WIN), ensure_ascii=False) + "\n")
        for pg in tqdm(pages, desc="计算截断"):
            ms = _page_methods_with_trunc(pg)
            for m, d in ms.items():
                s = stat[m]
                s["网页"] += 1
                s["块"] += len(d["chunks"])
                s["截断块"] += sum(c["truncated"] for c in d["chunks"])
                s["字"] += sum(len(c["text"]) for c in d["chunks"])
                s["看不到的字"] += sum(len(c["text"]) - c["visible_chars"] for c in d["chunks"])
                s["切点"] += len(d["boundaries"])
            f.write(json.dumps({"type": "page", "url": pg["url"], "source": pg.get("source", ""), "methods": ms},
                               ensure_ascii=False) + "\n")
    print(f"已导出 {len(pages)} 个网页 → {path}（查看器文件：{vpath}）")
    print("\n各方法的向量截断情况（不需要 LLM）：")
    for m in methods:
        s = stat[m]
        print(f"  {m:28s} 块/页 {s['块'] / max(s['网页'], 1):5.1f}  切点 {s['切点']:5d}  截断块 {100 * s['截断块'] / max(s['块'], 1):5.1f}%  "
              f"模型看不到的字 {100 * s['看不到的字'] / max(s['字'], 1):5.1f}%")


# %% 运行 1：query 视角（需要 RESULTS；默认谷歌答案、只看有 html 的网页、300 条、前 10 块）
QJ_FILE = os.path.join(OUT_ROOT, "judge", "judge_query_top10.jsonl.gz")
export_query_judge(QJ_FILE, set_name="谷歌前10", n=300, topk=10)

# %% 运行 2：页面视角（默认随机 100 个含列表 / 表格 / 提示框的网页；也可以 urls="/path/to/urls.txt"）
COMPARE_METHODS = ["content_br", "html_struct_secmerge_t512", "html_container"]
PJ_FILE = os.path.join(OUT_ROOT, "judge", "judge_pages.jsonl.gz")
export_page_judge(PJ_FILE, n=100, require="container")
