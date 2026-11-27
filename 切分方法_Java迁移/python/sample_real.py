"""在服务器上从离线评测的原始分片里随机抽一批真实网页（默认约 1 万页），连同切分结果一起导出，用于验证 Java 迁移。

每页输出：
  - 原始数据：url、content、markdown、html（其余字段放在 extra 里）
  - expected：用 chunk_ref（与离线评测代码逐字一致的参考实现）现场切出的 5 种方法的块
  - container_kinds：html_container 每块的类型
  - offline：离线评测时落盘的切分结果（OUT_ROOT/方法/chunks/），有哪个方法就带哪个，包括 content_br；
    没有对应文档（例如被去重跳过）时为 null
输出格式与 make_golden.py 相同（第 1 行 meta，之后每行一页），可以直接给 Java 的 GoldenRunner / mvn test 用。

抽样方式：每批数据（--source）分到相同的配额，随机选若干分片，在每个分片内做均匀的蓄水池抽样；固定种子，可复现。
另外会核对 expected 和 offline 是否一致：不一致说明服务器上离线评测用的代码或环境与 chunk_ref 不同，需要先查清楚。

用法（与离线评测用同一个 Python 环境，保证 lxml 版本相同）：
  python sample_real.py \
      --source "=/path/to/random/part-*" \
      --source "inj-=/path/to/inj/part-*" \
      --source "ggl-=/path/to/google_pages_parts/part-*" \
      --out-root /path/to/output/chunk_eval \
      --n 10000 --out sample_real.jsonl.gz --proc 16

  --source 写成 "前缀=分片路径通配"，前缀与离线评测 DATA_SOURCES 里的 prefix 一致（随机网页库前缀为空，写成 "=..."）
  --out-root 是离线评测的 OUT_ROOT，用来读取已落盘的切分结果；不给则不导出 offline、不做核对

注意：输出里是真实网页数据，只在内网使用，不要提交到公开仓库。
"""
import argparse
import glob
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_ref  # noqa: E402
from make_golden import golden_page  # noqa: E402

RAW_KEYS = ("url", "content", "markdown", "html")
OFFLINE_METHODS = ["content_br"] + chunk_ref.METHOD_NAMES


# ---------- 读原始分片（与 recall_pipeline_v2 的 list_parts / part_name / iter_records 语义相同） ----------
def list_parts(pattern):
    return sorted(p for p in glob.glob(pattern) if not p.endswith((".crc", "_SUCCESS")) and os.path.isfile(p))


def part_name(path, prefix):
    base = os.path.basename(path)
    for ext in (".gz", ".jsonl", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return prefix + base


def iter_records(path):
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


def usable(rec):
    return bool(rec.get("markdown")) or bool(rec.get("html"))


def sample_part(path, k, rng):
    """分片内蓄水池抽样 k 条有 markdown 或 html 的记录；同时记下所有记录的 url（对齐离线结果用）。"""
    urls, pool, seen = [], [], 0
    for i, rec in enumerate(iter_records(path)):
        urls.append((rec.get("url") or "").replace("\n", " "))
        if not usable(rec):
            continue
        seen += 1
        if len(pool) < k:
            pool.append((i, rec))
        else:
            j = rng.randrange(seen)
            if j < k:
                pool[j] = (i, rec)
    pool.sort(key=lambda x: x[0])
    return pool, urls


# ---------- 对齐离线评测落盘的切分结果 ----------
def align_lines(urls, urls_txt):
    """与 recall_pipeline_v2.align_drop 相同的贪心对齐：原始第 i 条记录 → chunks jsonl 的第 p 行（被跳过的记录没有）。"""
    with open(urls_txt, encoding="utf-8") as fu:
        want = fu.read().split("\n")
    want = want[:-1] if want and want[-1] == "" else want
    pos, p = {}, 0
    for i, u in enumerate(urls):
        if p < len(want) and u == want[p]:
            pos[i] = p
            p += 1
    if p != len(want):
        return None  # 原始数据与落盘结果对不上，不导出这个分片的离线结果
    return pos


def read_offline(out_root, method, name, urls, idxs):
    """读出某方法某分片里指定原始记录的离线切分结果：{原始序号: chunks}；没有落盘结果时返回 None。"""
    base = os.path.join(out_root, method, "chunks", name)
    if not (os.path.exists(base + ".done") and os.path.exists(base + ".urls.txt")):
        return None
    pos = align_lines(urls, base + ".urls.txt")
    if pos is None:
        print(f"  警告：{method}/{name} 与原始分片对不上，跳过该方法的离线结果", flush=True)
        return None
    need = {pos[i]: i for i in idxs if i in pos}
    out = {}
    lineidx = None
    if os.path.exists(base + ".lineidx.npy"):
        import numpy as np
        lineidx = np.load(base + ".lineidx.npy")
    with open(base + ".jsonl", "rb") as f:
        if lineidx is not None:
            for p in sorted(need):
                f.seek(int(lineidx[p]))
                out[need[p]] = json.loads(f.readline())
        else:
            for p, line in enumerate(f):
                if p in need:
                    out[need[p]] = json.loads(line)
    res = {}
    for i, d in out.items():
        if d.get("url") == urls[i]:
            res[i] = d.get("chunks") or []
    return res


# ---------- 切分（多进程） ----------
def _work(item):
    pid, md, html = item
    try:
        return golden_page(pid, md, html), None
    except Exception as e:  # noqa: BLE001  单页出错不影响整批
        return None, f"{pid}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help='"前缀=分片路径通配"，可重复')
    ap.add_argument("--out-root", help="离线评测的 OUT_ROOT（读取已落盘的切分结果并核对）")
    ap.add_argument("--n", type=int, default=10000, help="目标页数（各批平分）")
    ap.add_argument("--per-part", type=int, default=200, help="每个分片最多抽几页；越小覆盖的分片越多、读得越久")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sources = []
    for s in a.source:
        if "=" not in s:
            ap.error(f'--source 格式应为 "前缀=分片路径通配"：{s}')
        prefix, pattern = s.split("=", 1)
        parts = list_parts(pattern)
        if not parts:
            ap.error(f"{pattern} 下没有分片")
        sources.append((prefix, parts))

    import bs4
    try:
        import lxml
        lxml_ver = lxml.__version__
    except ImportError:
        lxml_ver = None
    if chunk_ref.BS_PARSER != "lxml":
        print("警告：没有安装 lxml，用的是 html.parser，结果与离线评测、Java 版都会不一致", flush=True)

    quota = math.ceil(a.n / len(sources))
    t0 = time.time()
    n_pages, errors = 0, []
    agree = {m: Counter() for m in chunk_ref.METHOD_NAMES}  # {方法: {"same": x, "diff": y}}
    diff_examples = {m: [] for m in chunk_ref.METHOD_NAMES}
    per_source = Counter()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    opener = gzip.open if a.out.endswith(".gz") else open
    with opener(a.out, "wt", encoding="utf-8") as fout, mp.get_context("fork").Pool(a.proc) as pool:
        meta = {"type": "meta", "kind": "real_sample", "bs_parser": chunk_ref.BS_PARSER, "bs4": bs4.__version__,
                "lxml": lxml_ver, "methods": chunk_ref.METHOD_NAMES, "seed": a.seed, "n_target": a.n,
                "sources": [{"prefix": p, "parts": len(ps)} for p, ps in sources],
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for prefix, parts in sources:
            rng = random.Random(f"{a.seed}:{prefix}")
            n_parts = min(len(parts), max(1, math.ceil(quota / a.per_part)))
            chosen = sorted(rng.sample(parts, n_parts))
            k = math.ceil(quota / n_parts)
            label = prefix or "(无前缀)"
            print(f"== 批次 {label}：共 {len(parts)} 个分片，抽 {n_parts} 个分片 × 每片 {k} 页", flush=True)
            for path in chosen:
                if per_source[prefix] >= quota:
                    break
                name = part_name(path, prefix)
                picked, urls = sample_part(path, min(k, quota - per_source[prefix]), random.Random(f"{a.seed}:{name}"))
                idxs = [i for i, _ in picked]
                offline = {}
                if a.out_root:
                    for m in OFFLINE_METHODS:
                        r = read_offline(a.out_root, m, name, urls, idxs)
                        if r is not None:
                            offline[m] = r
                items = [(f"{name}:{i}", rec.get("markdown"), rec.get("html")) for i, rec in picked]
                results = pool.map(_work, items, chunksize=4)
                for (i, rec), (page, err) in zip(picked, results):
                    if err:
                        errors.append(err)
                        continue
                    page["source"] = prefix
                    page["part"] = name
                    page["line"] = i
                    page["url"] = rec.get("url")
                    page["content"] = rec.get("content")
                    extra = {kk: v for kk, v in rec.items() if kk not in RAW_KEYS}
                    if extra:
                        page["extra"] = extra
                    if a.out_root:
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

    print(f"\n写入 {n_pages} 页 → {a.out}（{os.path.getsize(a.out) / 2**20:.1f} MB，{time.time() - t0:.0f}s，"
          f"bs4 {bs4.__version__}，lxml {lxml_ver}）")
    for prefix, _ in sources:
        print(f"  批次 {prefix or '(无前缀)'}：{per_source[prefix]} 页")
    if errors:
        print(f"切分出错 {len(errors)} 页（未写入），例如：")
        for e in errors[:5]:
            print("  ", e)
    if a.out_root:
        print("\nchunk_ref 现场切分 vs 离线评测落盘结果：")
        for m in chunk_ref.METHOD_NAMES:
            c = agree[m]
            tot = c["same"] + c["diff"]
            if tot == 0:
                print(f"  {m:32s} 离线没有这个方法的结果（未核对）")
                continue
            print(f"  {m:32s} 核对 {tot:6d} 页  不一致 {c['diff']:5d}  一致率 {100.0 * c['same'] / tot:.2f}%"
                  + (f"  例如 {diff_examples[m]}" if diff_examples[m] else ""))
        print("不一致说明服务器上离线评测用的代码或环境与 chunk_ref 不同（例如 notebook 被改过、lxml 版本不同），"
              "请先查清楚再用这份数据验证 Java。")


if __name__ == "__main__":
    main()
