# %% [markdown]
# # 抽样真实网页（Jupyter 版）
#
# 与 sample_real.py 功能相同，只是不用命令行参数：在第 1 段改配置，然后依次运行各段。
# 每页导出：原始数据（url、content、markdown、html）、5 种方法的现场切分结果 expected、
# html_container 的块类型 container_kinds、离线评测时落盘的切分结果 offline（含 content_br）。
# 输出格式与 make_golden.py 相同，可以直接给 Java 的 GoldenRunner / mvn test 用。
#
# 注意：
#   - 用离线评测同一个 Python 环境（同一 lxml 版本）运行；
#   - 输出里是真实网页数据，只在内网使用，不要提交到公开仓库。

# %% 1. 配置（按需修改）
# 本迁移包 python/ 目录的路径（里面有 chunk_ref/、make_golden.py、sample_real.py）
PKG_DIR = "/path/to/切分方法_Java迁移/python"

# 三批数据：前缀、路径必须与离线评测 recall_pipeline_v2.py 第 1 段 DATA_SOURCES 一致
DATA_SOURCES = [
    {"name": "随机网页库",                 "glob": "/path/to/random/part-*",              "prefix": ""},
    {"name": "自己引擎的精排与向量召回网页", "glob": "/path/to/inj/part-*",                 "prefix": "inj-"},
    {"name": "谷歌精排网页",               "glob": "/path/to/google_pages_parts/part-*", "prefix": "ggl-"},
]

# 离线评测的 OUT_ROOT：用来读取已落盘的切分结果并核对；设为 None 则不导出 offline、不核对
OUT_ROOT = "/path/to/output/chunk_eval"

N_PAGES = 10000      # 目标页数（各批平分）
PER_PART = 200       # 每个分片最多抽几页；越小覆盖的分片越多、读得越久
SEED = 20261127      # 固定种子，重跑结果相同
N_PROC = 16          # 切分用的进程数
OUT_PATH = "/path/to/golden_real.jsonl.gz"   # 输出文件（.gz 结尾则压缩）

# %% 2. 定义：抽样主流程（复用 sample_real.py 里的函数）
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
import chunk_ref  # noqa: E402
import sample_real as sr  # noqa: E402


def run_sample(sources=None, out_root=None, n=None, per_part=None, seed=None, n_proc=None, out_path=None):
    """按配置抽样并写出 jsonl；返回汇总（每批页数、出错页、现场切分与离线结果的核对）。参数不给则用第 1 段的配置。"""
    sources = sources if sources is not None else DATA_SOURCES
    out_root = out_root if out_root is not None else OUT_ROOT
    n = n or N_PAGES
    per_part = per_part or PER_PART
    seed = seed if seed is not None else SEED
    n_proc = n_proc or N_PROC
    out_path = out_path or OUT_PATH

    srcs = []
    for s in sources:
        parts = sr.list_parts(s["glob"])
        if not parts:
            raise ValueError(f"{s['glob']} 下没有分片")
        srcs.append((s["name"], s["prefix"], parts))

    import bs4
    try:
        import lxml
        lxml_ver = lxml.__version__
    except ImportError:
        lxml_ver = None
    if chunk_ref.BS_PARSER != "lxml":
        print("警告：没有安装 lxml，用的是 html.parser，结果与离线评测、Java 版都会不一致")

    quota = math.ceil(n / len(srcs))
    t0 = time.time()
    n_pages, errors = 0, []
    agree = {m: Counter() for m in chunk_ref.METHOD_NAMES}
    diff_examples = {m: [] for m in chunk_ref.METHOD_NAMES}
    per_source = Counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as fout, mp.get_context("fork").Pool(n_proc) as pool:
        meta = {"type": "meta", "kind": "real_sample", "bs_parser": chunk_ref.BS_PARSER, "bs4": bs4.__version__,
                "lxml": lxml_ver, "methods": chunk_ref.METHOD_NAMES, "seed": seed, "n_target": n,
                "sources": [{"name": nm, "prefix": p, "parts": len(ps)} for nm, p, ps in srcs],
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for src_name, prefix, parts in srcs:
            rng = random.Random(f"{seed}:{prefix}")
            n_parts = min(len(parts), max(1, math.ceil(quota / per_part)))
            chosen = sorted(rng.sample(parts, n_parts))
            k = math.ceil(quota / n_parts)
            print(f"== {src_name}（前缀 {prefix!r}）：共 {len(parts)} 个分片，抽 {n_parts} 个分片 × 每片 {k} 页", flush=True)
            for path in chosen:
                if per_source[prefix] >= quota:
                    break
                name = sr.part_name(path, prefix)
                picked, urls = sr.sample_part(path, min(k, quota - per_source[prefix]), random.Random(f"{seed}:{name}"))
                idxs = [i for i, _ in picked]
                offline = {}
                if out_root:
                    for m in sr.OFFLINE_METHODS:
                        r = sr.read_offline(out_root, m, name, urls, idxs)
                        if r is not None:
                            offline[m] = r
                items = [(f"{name}:{i}", rec.get("markdown"), rec.get("html")) for i, rec in picked]
                results = pool.map(sr._work, items, chunksize=4)
                for (i, rec), (page, err) in zip(picked, results):
                    if err:
                        errors.append(err)
                        continue
                    page["source"] = prefix
                    page["part"] = name
                    page["line"] = i
                    page["url"] = rec.get("url")
                    page["content"] = rec.get("content")
                    extra = {kk: v for kk, v in rec.items() if kk not in sr.RAW_KEYS}
                    if extra:
                        page["extra"] = extra
                    if out_root:
                        page["offline"] = {m: offline[m].get(i) for m in offline}
                        for m, want in page["expected"].items():
                            got = offline.get(m, {}).get(i)
                            if got is None:
                                continue
                            same = got == want
                            agree[m]["same" if same else "diff"] += 1
                            if not same and len(diff_examples[m]) < 5:
                                diff_examples[m].append(page["id"])
                    fout.write(json.dumps(page, ensure_ascii=False) + "\n")
                    n_pages += 1
                    per_source[prefix] += 1
                print(f"  {name}：抽到 {len(picked)} 页（本批累计 {per_source[prefix]}/{quota}，"
                      f"总计 {n_pages}，{time.time() - t0:.0f}s）", flush=True)

    print(f"\n写入 {n_pages} 页 → {out_path}（{os.path.getsize(out_path) / 2**20:.1f} MB，{time.time() - t0:.0f}s，"
          f"bs4 {bs4.__version__}，lxml {lxml_ver}）")
    for src_name, prefix, _ in srcs:
        print(f"  {src_name}：{per_source[prefix]} 页")
    if errors:
        print(f"切分出错 {len(errors)} 页（未写入），例如：")
        for e in errors[:5]:
            print("  ", e)
    check = {}
    if out_root:
        print("\n现场切分（chunk_ref）vs 离线评测落盘结果：")
        for m in chunk_ref.METHOD_NAMES:
            c = agree[m]
            tot = c["same"] + c["diff"]
            check[m] = {"checked": tot, "diff": c["diff"], "examples": diff_examples[m]}
            if tot == 0:
                print(f"  {m:32s} 离线没有这个方法的结果（未核对）")
                continue
            print(f"  {m:32s} 核对 {tot:6d} 页  不一致 {c['diff']:5d}  一致率 {100.0 * c['same'] / tot:.2f}%"
                  + (f"  例如 {diff_examples[m]}" if diff_examples[m] else ""))
        print("不一致说明服务器上离线评测用的代码或环境与 chunk_ref 不同（例如 notebook 被改过、lxml 版本不同），"
              "请先用第 4 段查看样例、查清楚再用这份数据验证 Java。")
    return {"pages": n_pages, "per_source": dict(per_source), "errors": errors, "check": check}


def show_diff(page_id, method, path=None, width=200):
    """查看某页某方法的现场切分结果与离线结果的差别（第一个不同的块起，各打印几块）。"""
    path = path or OUT_PATH
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("id") != page_id:
                continue
            want = p["expected"].get(method)
            got = (p.get("offline") or {}).get(method)
            print(p.get("url"))
            if want is None or got is None:
                print(f"没有 {method} 的{'现场' if want is None else '离线'}结果")
                return
            i = 0
            while i < len(want) and i < len(got) and want[i] == got[i]:
                i += 1
            print(f"块数：现场 {len(want)} / 离线 {len(got)}，第 {i + 1} 块起不同")
            for tag, cs in (("现场", want), ("离线", got)):
                for j in range(i, min(i + 3, len(cs))):
                    print(f"  [{tag} {j + 1}] ({len(cs[j])}字) {cs[j][:width]!r}")
            return
    print(f"没找到 {page_id}")


# %% 3. 运行：抽样并导出
SUMMARY = run_sample()

# %% 4. 运行（可选）：查看核对不一致的样例
for _m, _c in SUMMARY["check"].items():
    for _pid in _c["examples"][:2]:
        print(f"\n===== {_m}  {_pid}")
        show_diff(_pid, _m)
