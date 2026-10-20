#!/usr/bin/env python3
"""train 错例分析：当前最好版本在 train 偏序对上判错的对，按"错得最有把握"排序。只看 train，不看 dev。

  python harness/errors.py --top 30
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HARNESS_DIR, ROOT, read_pairs, read_scores, site_of  # noqa: E402

BEST = os.path.join(ROOT, "runs", "best", "npv_ori.tsv")
TRAIN_PAIRS = os.path.join(HARNESS_DIR, "labels", "train", "pairs.tsv")


def path_kind(url: str) -> str:
    path = url.split("://", 1)[-1].split("/", 1)[-1] if "/" in url.split("://", 1)[-1] else ""
    low = "/" + path.lower()
    if low in ("/", ""):
        return "root"
    if "/u/" in low or "/user/" in low:
        return "user"
    if "/list" in low or "/tag" in low or "?page=" in low or "/forum" in low:
        return "list"
    return "content"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--features", type=int, default=5, help="对前几条错例打印特征差")
    a = p.parse_args(argv)
    if not os.path.exists(BEST):
        print("还没有基线打分结果，先调用一次 run_eval。")
        return 1
    scores = read_scores(BEST, with_features=True)
    pairs = [x for x in read_pairs(TRAIN_PAIRS) if x["pref"] in ("a", "b")]
    wrong, ties = [], 0
    kinds, same_site, groups, total = Counter(), 0, Counter(), 0
    for pr in pairs:
        sa, sb = scores.get(pr["url_a"]), scores.get(pr["url_b"])
        if sa is None or sb is None:
            continue
        total += 1
        hi, lo = (pr["url_a"], pr["url_b"]) if pr["pref"] == "a" else (pr["url_b"], pr["url_a"])
        s_hi, s_lo = scores[hi][0], scores[lo][0]
        if s_hi == s_lo:
            ties += 1
            continue
        if s_hi < s_lo:
            wrong.append((s_lo - s_hi, hi, lo, s_hi, s_lo, pr))
            kinds[(path_kind(hi), path_kind(lo))] += 1
            same_site += site_of(hi) == site_of(lo)
            groups[pr["group"]] += 1
    wrong.sort(key=lambda t: -t[0])
    print(f"train 对 {total}，判错 {len(wrong)}（{len(wrong) / max(1, total):.1%}），同分 {ties}")
    print(f"判错中同站点对 {same_site}；按分组 {dict(groups)}")
    print("判错中（应高页类型, 实际高页类型）分布：", kinds.most_common(8))
    print()
    print("gap\tscore_should_high\tscore_actual_high\tgroup\tshould_be_higher\tactually_higher")
    for gap, hi, lo, s_hi, s_lo, pr in wrong[: a.top]:
        print(f"{gap:.2f}\t{s_hi:.2f}\t{s_lo:.2f}\t{pr['group']}\t{hi}\t{lo}")
    print()
    for gap, hi, lo, s_hi, s_lo, pr in wrong[: a.features]:
        fh, fl = scores[hi][1] or {}, scores[lo][1] or {}
        diff = {k: round(fh.get(k, 0) - fl.get(k, 0), 3) for k in sorted(set(fh) | set(fl)) if abs(fh.get(k, 0) - fl.get(k, 0)) > 1e-9}
        print(f"- 应高 {hi}\n  实高 {lo}\n  特征差(应高-实高): {diff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
