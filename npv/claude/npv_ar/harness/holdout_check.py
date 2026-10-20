#!/usr/bin/env python3
"""中环：在仓库之外的 holdout 上复核当前最好版本是否值得晋级。由人或外层驱动运行，不给 agent。

  python harness/holdout_check.py            # 只报告 PASS / FAIL
  python harness/holdout_check.py --promote  # PASS 时打 git 标签 promoted/N，记录到 state.json

只输出"过/不过"和粗档，不输出精确数字。每版 holdout 最多查询 max_queries 次。
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HARNESS_DIR, ROOT, load_harness_config, read_json, read_pairs, read_scores, write_json  # noqa: E402
from metrics import paired_compare, strict_pairwise  # noqa: E402
import gate  # noqa: E402

STATE = os.path.join(HARNESS_DIR, "state.json")
BEST = os.path.join(ROOT, "runs", "best", "npv_ori.tsv")
PROMOTED_DIR = os.path.join(ROOT, "runs", "promoted")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--private-dir", default=None)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--max-queries", type=int, default=20)
    a = p.parse_args(argv)
    hcfg = load_harness_config()
    priv = os.path.abspath(a.private_dir or os.path.join(ROOT, hcfg["private_dir"]))
    hold = os.path.join(priv, "holdout", "pairs.tsv")
    if not os.path.exists(hold):
        print(f"没有 holdout 标签: {hold}")
        return 2
    usage_path = os.path.join(priv, "holdout", "usage.json")
    usage = read_json(usage_path, {"queries": 0})
    if usage["queries"] >= a.max_queries:
        print(f"这版 holdout 已查询 {usage['queries']} 次，达到上限，请换新的 holdout")
        return 2
    state = read_json(STATE)
    if not state or not state.get("best") or not os.path.exists(BEST):
        print("还没有基线")
        return 2
    pairs = read_pairs(hold)
    cur = strict_pairwise(read_scores(BEST), pairs)
    promoted_scores = os.path.join(PROMOTED_DIR, "npv_ori.tsv")
    if not os.path.exists(promoted_scores):
        print("尚无晋级点：把当前最好版本记为 promoted/1（基线）")
        if a.promote:
            _promote(state, 1)
        return 0
    prev = strict_pairwise(read_scores(promoted_scores), pairs)
    comp = paired_compare(cur["per_pair"], prev["per_pair"])
    usage["queries"] += 1
    write_json(usage_path, usage)
    eps = gate.threshold(comp["se_cluster"], hcfg)
    one_sided_p = comp["p_sign"] / 2 if comp["wins"] > comp["losses"] else 1.0
    passed = comp["delta"] >= eps and one_sided_p < 0.05
    print(f"holdout: {'PASS' if passed else 'FAIL'}  档位={gate.coarse(comp['delta'], eps) if not passed else '显著更好'}  "
          f"查询 {usage['queries']}/{a.max_queries}")
    if passed and a.promote:
        n = int(state.get("promoted") or 1) + 1
        _promote(state, n)
    return 0 if passed else 1


def _promote(state, n):
    os.makedirs(PROMOTED_DIR, exist_ok=True)
    shutil.copy(BEST, os.path.join(PROMOTED_DIR, "npv_ori.tsv"))
    state["promoted"] = n
    write_json(STATE, state)
    subprocess.run(["git", "tag", "-f", f"promoted/{n}", state["best"].get("commit", "HEAD")], cwd=ROOT, check=False)
    print(f"已标记 promoted/{n}")


if __name__ == "__main__":
    sys.exit(main())
