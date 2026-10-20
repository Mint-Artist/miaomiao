#!/usr/bin/env python3
"""用 npv_py 的合成样例搭一套可跑的数据：快照、train/dev 标签、仓库外的 holdout。只用于跑通流程。

  python tools/prepare_synthetic.py [--pairs 3000] [--grades 1500] [--seed 11]

真实数据：把 input.tsv 与六张表放进 harness/snapshot/，标签按 pairs.tsv 五列格式（url_a url_b pref group qid）
放进 harness/labels/{train,dev}/ 与 ../npv_ar_private/holdout/，然后删除 harness/state.json 重跑基线。
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
from common import CLAUDE_DIR, HARNESS_DIR, NPV_LAB_DIR, NPV_PY_DIR, TABLE_FILES, load_harness_config, sha256_file  # noqa: E402

sys.path.insert(0, NPV_LAB_DIR)
import make_sample_labels  # noqa: E402

SNAPSHOT = os.path.join(HARNESS_DIR, "snapshot")
LABELS = os.path.join(HARNESS_DIR, "labels")
NOW = 1757030400  # 与 npv_lab README 示例一致的时间基准


def write_pairs(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("url_a\turl_b\tpref\tgroup\tqid\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def write_grades(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("url\tgrade\tgroup\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--grades", type=int, default=1500)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--pairs-per-query", type=int, default=3)
    a = p.parse_args(argv)
    hcfg = load_harness_config()
    priv = os.path.abspath(os.path.join(ROOT, hcfg["private_dir"]))

    # 1. 快照
    os.makedirs(SNAPSHOT, exist_ok=True)
    src = os.path.join(NPV_PY_DIR, "sample")
    for fn in ["input.tsv"] + list(TABLE_FILES.values()):
        shutil.copy(os.path.join(src, fn), os.path.join(SNAPSHOT, fn))
    meta = {"region": hcfg["region"], "now": NOW, "source": "npv_py/sample 合成数据（仅用于跑通流程）",
            "sha256": {fn: sha256_file(os.path.join(SNAPSHOT, fn)) for fn in ["input.tsv"] + list(TABLE_FILES.values())}}
    with open(os.path.join(SNAPSHOT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 2. 合成标签 -> 加 qid -> 按 qid 切分
    tmp = tempfile.mkdtemp()
    make_sample_labels.main(["--input", os.path.join(SNAPSHOT, "input.tsv"), "--out", tmp,
                             "--pairs", str(a.pairs), "--grades", str(a.grades), "--seed", str(a.seed)])
    pairs = []
    with open(os.path.join(tmp, "pairs.tsv"), encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            pairs.append(cols[:4] + [f"q{i // a.pairs_per_query:05d}"])
    qids = sorted({r[4] for r in pairs})
    rnd = random.Random(a.seed)
    rnd.shuffle(qids)
    n = len(qids)
    split = {q: ("train" if i < 0.6 * n else "dev" if i < 0.85 * n else "holdout") for i, q in enumerate(qids)}
    parts = {"train": [], "dev": [], "holdout": []}
    for r in pairs:
        parts[split[r[4]]].append(r)
    write_pairs(os.path.join(LABELS, "train", "pairs.tsv"), parts["train"])
    write_pairs(os.path.join(LABELS, "dev", "pairs.tsv"), parts["dev"])
    write_pairs(os.path.join(priv, "holdout", "pairs.tsv"), parts["holdout"])
    grades = []
    with open(os.path.join(tmp, "grades.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            grades.append(line.rstrip("\n").split("\t")[:3])
    rnd.shuffle(grades)
    half = len(grades) // 2
    write_grades(os.path.join(LABELS, "train", "grades.tsv"), grades[:half])
    write_grades(os.path.join(LABELS, "dev", "grades.tsv"), grades[half:])
    shutil.rmtree(tmp)
    print(f"snapshot: {SNAPSHOT}\ntrain {len(parts['train'])} 对, dev {len(parts['dev'])} 对, holdout {len(parts['holdout'])} 对 -> {priv}/holdout")
    print(f"grades: train {half}, dev {len(grades) - half}")


if __name__ == "__main__":
    main()
