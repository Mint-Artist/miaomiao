#!/usr/bin/env python3
"""评估打分输出。

  python eval_scores.py --scores runs/baseline/npv_ori.tsv --pairs labels/pairs.tsv --grades labels/grades.tsv
  python eval_scores.py --scores runs/exp1/npv_ori.tsv --compare runs/baseline/npv_ori.tsv --pairs labels/pairs.tsv
"""
import argparse
import json

import lab  # noqa: F401
from lab.evaluate import compare_pairwise, evaluate, format_report, read_pairs, read_scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--pairs", default=None)
    p.add_argument("--grades", default=None)
    p.add_argument("--compare", default=None, help="另一份打分输出（通常是基线），做配对比较")
    p.add_argument("--gap-edges", default="0,2,5,10,20", help="分差分桶边界")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    edges = tuple(float(v) for v in a.gap_edges.split(",")) + (float("inf"),)
    r = evaluate(a.scores, a.pairs, a.grades, edges)
    print(format_report(r))
    if a.compare and a.pairs:
        c = compare_pairwise(read_scores(a.scores), read_scores(a.compare), read_pairs(a.pairs))
        r["compare"] = c
        print("\n## 与对照的配对比较\n")
        print(f"- 本版 {c['accuracy_a']:.1%} 对照 {c['accuracy_b']:.1%}")
        print(f"- 本版对/对照错 {c['a_right_b_wrong']}，本版错/对照对 {c['b_right_a_wrong']}，"
              f"符号检验 p = {c['sign_test_p']:.3g}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
