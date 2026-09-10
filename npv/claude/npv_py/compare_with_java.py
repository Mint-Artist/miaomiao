#!/usr/bin/env python3
"""把 Python 版输出与 Java 作业输出按 url 对齐比较，用于在原环境验证复刻是否忠实。

  python compare_with_java.py --java /path/to/output/npv_ori --py out/npv_ori.tsv [--tol 1e-4]

--java 可以是目录（读取其中所有 part-* 文件）或单个文件。两边都是无表头 TSV：
url, npv_ori, [npv], npv_fea。分数按 npv_ori 比较，若两边都有 npv 列也比较等级。
"""
import argparse
import glob
import os
import sys


def read_rows(path):
    files = sorted(glob.glob(os.path.join(path, "part-*"))) if os.path.isdir(path) else [path]
    rows = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 3:
                    continue
                level = int(cols[2]) if len(cols) >= 4 and cols[2].lstrip("-").isdigit() else None
                rows[cols[0]] = (float(cols[1]), level)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()

    j, p = read_rows(a.java), read_rows(a.py)
    common = set(j) & set(p)
    print(f"java rows={len(j)} py rows={len(p)} common={len(common)} "
          f"only_java={len(set(j) - common)} only_py={len(set(p) - common)}")
    if not common:
        sys.exit(1)
    diffs = sorted(((abs(j[u][0] - p[u][0]), u) for u in common), reverse=True)
    over = [d for d in diffs if d[0] > a.tol]
    print(f"score: max_abs_diff={diffs[0][0]:.6g} mean_abs_diff={sum(d for d, _ in diffs) / len(diffs):.6g} "
          f"rows_over_tol={len(over)}")
    for d, u in diffs[:a.show]:
        if d > a.tol:
            print(f"  {d:.6g}\tjava={j[u][0]!r}\tpy={p[u][0]!r}\t{u}")
    lv = [(u, j[u][1], p[u][1]) for u in common if j[u][1] is not None and p[u][1] is not None]
    if lv:
        mism = [t for t in lv if t[1] != t[2]]
        print(f"level: compared={len(lv)} mismatched={len(mism)}")
        for u, a_, b_ in mism[:a.show]:
            print(f"  java={a_}\tpy={b_}\t{u}")
    sys.exit(0 if not over else 2)


if __name__ == "__main__":
    main()
