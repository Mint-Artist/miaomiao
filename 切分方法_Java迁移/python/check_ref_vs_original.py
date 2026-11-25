"""验证 chunk_ref（独立参考实现）与 notebook 原版逐字一致。

原版的加载方式与 notebook 相同：依次执行 original/recall_pipeline_v2.py 第 1–7 段、
original/chunk_adhere.py 和 original/chunk_container.py 的"定义"段（不读数据、不推理）。
然后在 构造用例 + 随机 html / markdown 页 上比较 5 种方法的输出。

    python check_ref_vs_original.py --cases ../testdata/cases --fuzz 1000
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chunk_ref  # noqa: E402
from fuzz_pages import gen_html, gen_markdown  # noqa: E402


def load_original():
    g = {"__name__": "original"}
    src = open(os.path.join(HERE, "original", "recall_pipeline_v2.py"), encoding="utf-8").read()
    cells, cur, name = {}, [], None
    for line in src.split("\n"):
        if line.startswith("# %% "):
            if name:
                cells[name] = "\n".join(cur)
            name, cur = line[5:].split(".")[0].strip(), []
        cur.append(line)
    cells[name] = "\n".join(cur)
    for k in ["1", "2", "3", "4", "5", "6", "7"]:
        exec(cells[k], g)
    for fn in ["chunk_adhere.py", "chunk_container.py"]:
        exec(open(os.path.join(HERE, "original", fn), encoding="utf-8").read().split("# %% 统计")[0], g)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "..", "testdata", "cases"))
    ap.add_argument("--fuzz", type=int, default=1000)
    a = ap.parse_args()
    g = load_original()
    Doc, METHODS = g["Doc"], g["METHODS"]
    pages = []
    for fn in sorted(os.listdir(a.cases)):
        with open(os.path.join(a.cases, fn), encoding="utf-8", newline="") as f:
            t = f.read()
        pages.append((fn, t if fn.endswith(".md") else "", t if fn.endswith(".html") else ""))
    for i in range(a.fuzz):
        pages.append((f"fuzz:{i}", gen_markdown(random.Random(2_000_000 + i)), gen_html(random.Random(1_000_000 + i))))
    bad = {m: 0 for m in chunk_ref.METHOD_NAMES}
    for pid, md, html in pages:
        d = Doc({"markdown": md, "html": html})
        for m in chunk_ref.METHOD_NAMES:
            if METHODS[m](d) != chunk_ref.chunk(m, markdown=md, html=html):
                bad[m] += 1
                if bad[m] <= 3:
                    print(f"不一致：{m} {pid}")
    print(f"共 {len(pages)} 页（构造用例 + 随机 {a.fuzz} 页 html 与 markdown）")
    for m, b in bad.items():
        print(f"  {m:32s} 不一致 {b} 页")
    sys.exit(1 if any(bad.values()) else 0)


if __name__ == "__main__":
    main()
