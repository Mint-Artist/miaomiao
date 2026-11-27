# %% [markdown]
# # 从离线评测已有的切分结果里抽样（Jupyter 版，不依赖 chunk_ref）
#
# 直接取当初向量召回实验落盘的切分结果（OUT_ROOT/方法/chunks/），配上原始数据，导出一份 Java 对照用的标准答案。
# 不重新切分、不需要 chunk_ref / bs4 / lxml，只需要原始分片和 OUT_ROOT。
#
# 与 sample_real_jupyter.py 的区别：
#   - 这里的 expected 就是离线评测实际用的切分结果，最权威；
#   - 只能覆盖离线跑过的方法；_adh 两种方法当初没跑，这里没有，需要时用 sample_real_jupyter.py 现场切。
#
# 用法：在同一个 notebook 里先运行 recall_pipeline_v2.py 第 1 段（配置），再把本文件各段贴进去运行。
# 本文件第 1 段沿用那里的 DATA_SOURCES、OUT_ROOT，只需要改 SAMPLE_OUT；变量、函数都带 SAMPLE_ / smp_ 前缀，不会覆盖 v2 的。
#
# 每行输出：{"type": "page", "id", "url", "source", "part", "line", "content", "markdown", "html", "extra",
#           "expected": {方法: [块, ...]}, "content_br": [块, ...]}
# 第 1 行是 meta。可以直接给 Java 的 GoldenRunner / mvn test -Dgolden.path=... 用。
# 注意：输出里是真实网页数据，只在内网使用，不要提交到公开仓库。

# %% 1. 配置（按需修改）
if "DATA_SOURCES" not in globals():   # 没运行过 v2 第 1 段时用这里的值（前缀、路径必须与离线评测一致）
    DATA_SOURCES = [
        {"name": "随机网页库",                 "glob": "/path/to/random/part-*",              "prefix": ""},
        {"name": "自己引擎的精排与向量召回网页", "glob": "/path/to/inj/part-*",                 "prefix": "inj-"},
        {"name": "谷歌精排网页",               "glob": "/path/to/google_pages_parts/part-*", "prefix": "ggl-"},
    ]
if "OUT_ROOT" not in globals():
    OUT_ROOT = "/path/to/output/chunk_eval"

# 要导出的方法：OUT_ROOT 下的目录名（= v2 里 RUN_METHODS 的方法名）。目录不存在的方法自动跳过。
SAMPLE_METHODS = ["md_struct_secmerge_t512", "html_struct_secmerge_t512", "html_container"]
SAMPLE_BASELINE = "content_br"   # 顺带导出线上基线的切分结果（不参与 Java 对照）；不需要设为 None

SAMPLE_N = 10000          # 目标页数（各批平分）
SAMPLE_PER_PART = 200     # 每个分片最多抽几页；越小覆盖的分片越多、读得越久
SAMPLE_SEED = 20261127    # 固定种子，重跑结果相同
SAMPLE_OUT = "/path/to/golden_offline.jsonl.gz"   # 输出文件（.gz 结尾则压缩）

import os
print("数据：", [(s["name"], s["prefix"], s["glob"]) for s in DATA_SOURCES])
print("离线结果：", OUT_ROOT)
for _m in SAMPLE_METHODS + ([SAMPLE_BASELINE] if SAMPLE_BASELINE else []):
    print(f"  {_m:32s}", "有" if os.path.isdir(os.path.join(OUT_ROOT, _m, "chunks")) else "没有这个目录（将跳过）")
print("输出：", SAMPLE_OUT)

# %% 2. 定义
import glob
import gzip
import json
import math
import random
import time
from collections import Counter


def smp_list_parts(pattern):
    return sorted(p for p in glob.glob(pattern) if not p.endswith((".crc", "_SUCCESS")) and os.path.isfile(p))


def smp_part_name(path, prefix):
    base = os.path.basename(path)
    for ext in (".gz", ".jsonl", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return prefix + base


def smp_iter_records(path):
    """与 v2 的 iter_records 相同：跳过空行、坏行、非 dict 行（序号按跳过后的记录数计）。"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                yield d


def smp_align(urls, urls_txt):
    """与 v2 的 align_drop 相同的贪心对齐：原始第 i 条记录 → chunks jsonl 第 p 行（当初去重跳过的记录没有）。"""
    with open(urls_txt, encoding="utf-8") as fu:
        want = fu.read().split("\n")
    want = want[:-1] if want and want[-1] == "" else want
    pos, p = {}, 0
    for i, u in enumerate(urls):
        if p < len(want) and u == want[p]:
            pos[i] = p
            p += 1
    return pos if p == len(want) else None


def smp_read_lines(jsonl, line_nos):
    """读 chunks jsonl 的指定行：有 lineidx.npy 就按字节偏移直接跳，否则顺序扫。"""
    out = {}
    idx_path = jsonl[: -len(".jsonl")] + ".lineidx.npy"
    with open(jsonl, "rb") as f:
        if os.path.exists(idx_path):
            import numpy as np
            lineidx = np.load(idx_path)
            for p in sorted(line_nos):
                f.seek(int(lineidx[p]))
                out[p] = json.loads(f.readline())
        else:
            need = set(line_nos)
            for p, line in enumerate(f):
                if p in need:
                    out[p] = json.loads(line)
    return out


def smp_one_part(path, prefix, methods, k, rng):
    """一个分片：先对齐各方法的离线结果，只在"所有方法都有结果、且有 markdown 或 html"的记录里蓄水池抽 k 条。"""
    name = smp_part_name(path, prefix)
    recs_urls = [(rec.get("url") or "").replace("\n", " ") for rec in smp_iter_records(path)]
    pos = {}
    for m in methods:
        base = os.path.join(OUT_ROOT, m, "chunks", name)
        if not (os.path.exists(base + ".done") and os.path.exists(base + ".urls.txt")):
            return name, [], f"{m} 没有这个分片的结果"
        al = smp_align(recs_urls, base + ".urls.txt")
        if al is None:
            return name, [], f"{m} 与原始分片对不上（数据可能已变化）"
        pos[m] = al
    ok = set(range(len(recs_urls)))
    for m in methods:
        ok &= set(pos[m])
    # 第二遍读原始数据：在可用记录里蓄水池抽样
    pool, seen = [], 0
    for i, rec in enumerate(smp_iter_records(path)):
        if i not in ok or not (rec.get("markdown") or rec.get("html")):
            continue
        seen += 1
        if len(pool) < k:
            pool.append((i, rec))
        else:
            j = rng.randrange(seen)
            if j < k:
                pool[j] = (i, rec)
    pool.sort(key=lambda x: x[0])
    chunks = {}
    for m in methods:
        lines = smp_read_lines(os.path.join(OUT_ROOT, m, "chunks", name + ".jsonl"), [pos[m][i] for i, _ in pool])
        chunks[m] = {i: lines[pos[m][i]] for i, _ in pool}
    out = []
    for i, rec in pool:
        bad = [m for m in methods if chunks[m][i].get("url") != recs_urls[i]]
        if bad:
            continue  # 理论上不会发生；url 对不上的记录不导出
        out.append((i, rec, {m: chunks[m][i].get("chunks") or [] for m in methods}))
    return name, out, None


def smp_run(sources=None, methods=None, baseline=None, n=None, per_part=None, seed=None, out_path=None):
    sources = sources if sources is not None else DATA_SOURCES
    methods = [m for m in (methods or SAMPLE_METHODS) if os.path.isdir(os.path.join(OUT_ROOT, m, "chunks"))]
    baseline = baseline if baseline is not None else SAMPLE_BASELINE
    if baseline and not os.path.isdir(os.path.join(OUT_ROOT, baseline, "chunks")):
        baseline = None
    n, per_part = n or SAMPLE_N, per_part or SAMPLE_PER_PART
    seed = seed if seed is not None else SAMPLE_SEED
    out_path = out_path or SAMPLE_OUT
    if not methods:
        raise ValueError(f"{OUT_ROOT} 下没有 SAMPLE_METHODS 里的任何方法")
    all_methods = methods + ([baseline] if baseline else [])

    quota = math.ceil(n / len(sources))
    t0, n_pages, per_source, skipped = time.time(), 0, Counter(), []
    stats = {m: Counter() for m in all_methods}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as fout:
        meta = {"type": "meta", "kind": "offline_sample", "out_root": os.path.basename(OUT_ROOT.rstrip("/")),
                "methods": methods, "baseline": baseline, "seed": seed, "n_target": n,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for s in sources:
            prefix = s["prefix"]
            parts = smp_list_parts(s["glob"])
            if not parts:
                raise ValueError(f"{s['glob']} 下没有分片")
            rng = random.Random(f"{seed}:{prefix}")
            n_parts = min(len(parts), max(1, math.ceil(quota / per_part)))
            chosen = sorted(rng.sample(parts, n_parts))
            k = math.ceil(quota / n_parts)
            print(f"== {s['name']}（前缀 {prefix!r}）：共 {len(parts)} 个分片，抽 {n_parts} 个分片 × 每片 {k} 页", flush=True)
            for path in chosen:
                if per_source[prefix] >= quota:
                    break
                name, got, err = smp_one_part(path, prefix, all_methods, min(k, quota - per_source[prefix]),
                                              random.Random(f"{seed}:{smp_part_name(path, prefix)}"))
                if err:
                    skipped.append(f"{name}：{err}")
                    print(f"  {name}：跳过（{err}）", flush=True)
                    continue
                for i, rec, ch in got:
                    page = {"type": "page", "id": f"{name}:{i}", "url": rec.get("url"), "source": prefix, "part": name,
                            "line": i, "content": rec.get("content"), "markdown": rec.get("markdown"),
                            "html": rec.get("html"), "expected": {m: ch[m] for m in methods}}
                    if baseline:
                        page[baseline] = ch[baseline]
                    extra = {kk: v for kk, v in rec.items() if kk not in ("url", "content", "markdown", "html")}
                    if extra:
                        page["extra"] = extra
                    fout.write(json.dumps(page, ensure_ascii=False) + "\n")
                    for m in all_methods:
                        st = stats[m]
                        st["pages"] += 1
                        st["chunks"] += len(ch[m])
                        st["chars"] += sum(len(c) for c in ch[m])
                        st["empty"] += not ch[m]
                    n_pages += 1
                    per_source[prefix] += 1
                print(f"  {name}：抽到 {len(got)} 页（本批累计 {per_source[prefix]}/{quota}，总计 {n_pages}，"
                      f"{time.time() - t0:.0f}s）", flush=True)

    print(f"\n写入 {n_pages} 页 → {out_path}（{os.path.getsize(out_path) / 2**20:.1f} MB，{time.time() - t0:.0f}s）")
    for s in sources:
        print(f"  {s['name']}：{per_source[s['prefix']]} 页")
    if skipped:
        print(f"跳过 {len(skipped)} 个分片，例如：", skipped[:3])
    print("\n各方法统计（抽样页上）：")
    for m in all_methods:
        st = stats[m]
        print(f"  {m:32s} 块/页 {st['chunks'] / max(st['pages'], 1):6.2f}  平均块长 {st['chars'] / max(st['chunks'], 1):7.1f}"
              f"  无块页 {st['empty']}")
    return {"pages": n_pages, "per_source": dict(per_source), "skipped": skipped}


# %% 3. 运行：抽样并导出
SAMPLE_SUMMARY = smp_run()
