#!/usr/bin/env python3
"""为 npv_py 的合成样例生成合成标签（偏序对 + 绝对档位），只用于跑通评估流程。

  python make_sample_labels.py --input ../npv_py/sample/input.tsv --out labels --pairs 3000 --grades 1500

"真实权威度"由 make_sample.py 里站点的隐藏等级加上页面类型惩罚再加噪声得到；
它与基线模型相关但不相同，所以基线在这套标签上不会是满分，可以看到改进空间。真实数据请用大模型或人工标注替换。
"""
import argparse
import os
import random
import sys

import lab  # noqa: F401
from lab import NPV_PY_DIR
from lab.cli_common import iter_rows

sys.path.insert(0, NPV_PY_DIR)
from make_sample import SITES  # noqa: E402

SITE_TRUTH = {s: (level * 1.0 + sr / 100.0 + spr / 10.0) for s, sr, spr, _dr, level in SITES}


def true_authority(url: str, rnd: random.Random) -> float:
    host = url.split("://", 1)[-1].split("/", 1)[0]
    base = SITE_TRUTH.get(host, 0.5)
    path = url.split(host, 1)[-1]
    if "/u/" in path or "/user/" in path:
        base -= 1.0
    if "/list" in path or "/forum" in path:
        base -= 0.6
    if "/article/" in path:
        base += 0.3
    return base + rnd.gauss(0, 0.4)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="labels")
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--grades", type=int, default=1500)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args(argv)
    rnd = random.Random(a.seed)
    urls = [x.url for x, _ in iter_rows(a.input)]
    truth = {u: true_authority(u, rnd) for u in urls}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("url_a\turl_b\tpref\tgroup\n")
        for _ in range(a.pairs):
            ua, ub = rnd.sample(urls, 2)
            d = truth[ua] - truth[ub]
            if abs(d) < 0.3:
                pref = "tie"
            else:
                pref = "a" if d > 0 else "b"
                if rnd.random() < 0.05:  # 5% 标注噪声
                    pref = "b" if pref == "a" else "a"
            f.write(f"{ua}\t{ub}\t{pref}\t{rnd.choice(['head', 'tail'])}\n")
    lo, hi = min(truth.values()), max(truth.values())
    with open(os.path.join(a.out, "grades.tsv"), "w", encoding="utf-8") as f:
        f.write("url\tgrade\tgroup\n")
        for u in rnd.sample(urls, min(a.grades, len(urls))):
            g = round(4 * (truth[u] - lo) / (hi - lo))
            f.write(f"{u}\t{max(0, min(4, g))}\tall\n")
    print(f"wrote {a.pairs} pairs and {min(a.grades, len(urls))} grades to {a.out}/")


if __name__ == "__main__":
    main()
