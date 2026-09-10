#!/usr/bin/env python3
"""用实验特征（lab/features.py）与配置文件打分，输出格式与 npv_py/run_npv.py 一致。

  python run_lab.py --input ../npv_py/sample/input.tsv ... --output runs/baseline --config configs/baseline.json
"""
import argparse
import os
import sys
from collections import Counter

import lab  # noqa: F401  (加入 npv_py 路径)
from lab.cli_common import add_table_args, build_scorer
from npv import assign_levels
from npv.io import BadRow, features_json, format_row, parse_line

HQW_FLAGS = {"1", "2", "5"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_table_args(p)
    p.add_argument("--output", required=True)
    p.add_argument("--scroll", type=int, default=0)
    a = p.parse_args(argv)
    scorer, site_list = build_scorer(a)
    os.makedirs(a.output, exist_ok=True)
    stats = Counter()
    hqw = []
    checked = False
    with open(a.input, encoding="utf-8") as fin, \
            open(os.path.join(a.output, "npv_ori.tsv"), "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                x, flag = parse_line(line)
            except BadRow:
                stats["bad_rows"] += 1
                continue
            r = scorer.score(x, site_list)
            if not checked:
                scorer.check_weights(r.features)
                checked = True
            stats["rows"] += 1
            fea = features_json(r.features)
            if a.scroll == 1 and flag in HQW_FLAGS:
                hqw.append((x.url, r.score, fea))
            else:
                fout.write(format_row(x.url, r.score, fea) + "\n")
    if a.scroll == 1:
        c = scorer.cfg
        ranked = assign_levels(hqw, score_of=lambda t: t[1], tie_key=lambda t: t[0],
                               z_start=c.norm_z_start, z_end=c.norm_z_end, buckets=c.norm_buckets)
        with open(os.path.join(a.output, "npv.tsv"), "w", encoding="utf-8") as fout:
            for (url, score, fea), _rank, level in ranked:
                fout.write(format_row(url, score, fea, level) + "\n")
        stats["npv_rows"] = len(ranked)
    stats["now_seconds"] = scorer.now
    for k, v in stats.items():
        print(f"{k}\t{v}", file=sys.stderr)


if __name__ == "__main__":
    main()
