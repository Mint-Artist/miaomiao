#!/usr/bin/env python3
"""跑一轮实验：用某个配置打分 -> 评估 -> 与基线配对比较 -> 追加一行到 results.tsv。

  python experiment.py --name exp1 --config configs/exp1.json --note "pr 权重 4->10" \
      --input ... [表参数] --pairs labels/pairs.tsv --grades labels/grades.tsv --baseline runs/baseline/npv_ori.tsv

自改进循环的骨架：每轮 agent 改 features.py 或配置，然后调用本脚本，读 results.tsv 决定保留还是回退。
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import lab  # noqa: F401
from lab import LAB_DIR
from lab.evaluate import compare_pairwise, evaluate, read_pairs, read_scores

TABLE_ARGS = ("spr", "dr_site", "dr_suffix", "ow", "pr_split", "ow_blacklist", "adc_whitelist")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--note", default="")
    p.add_argument("--input", required=True)
    for t in TABLE_ARGS:
        p.add_argument("--" + t.replace("_", "-"), required=True)
    p.add_argument("--region", default="")
    p.add_argument("--now", type=int, default=None)
    p.add_argument("--pairs", default=None)
    p.add_argument("--grades", default=None)
    p.add_argument("--baseline", default=None, help="基线的 npv_ori.tsv，用于配对比较")
    p.add_argument("--runs-dir", default=os.path.join(LAB_DIR, "runs"))
    p.add_argument("--results", default=os.path.join(LAB_DIR, "results.tsv"))
    a = p.parse_args(argv)

    out_dir = os.path.join(a.runs_dir, a.name)
    cmd = [sys.executable, os.path.join(LAB_DIR, "run_lab.py"), "--input", a.input, "--output", out_dir,
           "--region", a.region]
    for t in TABLE_ARGS:
        cmd += ["--" + t.replace("_", "-"), getattr(a, t)]
    if a.now is not None:
        cmd += ["--now", str(a.now)]
    if a.config:
        cmd += ["--config", a.config]
    subprocess.run(cmd, check=True)

    scores_path = os.path.join(out_dir, "npv_ori.tsv")
    r = evaluate(scores_path, a.pairs, a.grades)
    row = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": a.name, "config": a.config or "baseline", "note": a.note,
        "constraints_ok": r["constraints"]["passed"],
        "pairwise_acc": r.get("pairwise", {}).get("accuracy"),
        "hard_acc": r.get("pairwise", {}).get("hard_accuracy"),
        "n_pairs_used": r.get("pairwise", {}).get("n_used"),
        "spearman": r.get("grades", {}).get("spearman"),
        "buried_good_rate": r.get("grades", {}).get("buried_good_rate"),
        "top_bad_rate": r.get("grades", {}).get("top_bad_rate"),
        "vs_baseline_acc": None, "vs_baseline_p": None,
    }
    if a.baseline and a.pairs and os.path.exists(a.baseline):
        c = compare_pairwise(read_scores(scores_path), read_scores(a.baseline), read_pairs(a.pairs))
        row["vs_baseline_acc"] = c["accuracy_b"]
        row["vs_baseline_p"] = c["sign_test_p"]
        r["compare"] = c
    with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    header = list(row)
    new_file = not os.path.exists(a.results)
    with open(a.results, "a", encoding="utf-8") as f:
        if new_file:
            f.write("\t".join(header) + "\n")
        f.write("\t".join("" if row[k] is None else (f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]))
                          for k in header) + "\n")
    print("\t".join(f"{k}={row[k]}" for k in ("name", "pairwise_acc", "hard_acc", "spearman",
                                              "buried_good_rate", "vs_baseline_acc", "vs_baseline_p")))


if __name__ == "__main__":
    main()
