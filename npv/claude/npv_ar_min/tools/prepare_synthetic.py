#!/usr/bin/env python3
"""用 npv_py 的合成样例给薄版本准备 data/ 与 labels/，只为跑通流程。

  python tools/prepare_synthetic.py
真实数据：data/ 放 input.tsv 与六张表（URL 要覆盖标签里的全部页面），labels/train/pairs.tsv 用 tools/convert_labels.py 生成。
"""
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLAUDE = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(CLAUDE, "npv_ar", "harness"))
from common import NPV_LAB_DIR, NPV_PY_DIR, TABLE_FILES, sha256_file  # noqa: E402

sys.path.insert(0, NPV_LAB_DIR)
import make_sample_labels  # noqa: E402

DATA = os.path.join(ROOT, "data")
TRAIN = os.path.join(ROOT, "labels", "train")
TEST = os.path.join(CLAUDE, "npv_ar_private", "min_test")


def main():
    os.makedirs(DATA, exist_ok=True)
    src = os.path.join(NPV_PY_DIR, "sample")
    files = ["input.tsv"] + list(TABLE_FILES.values())
    for fn in files:
        shutil.copy(os.path.join(src, fn), os.path.join(DATA, fn))
    with open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"region": "zh", "now": 1757030400, "tie_band": {"unit": "raw", "value": 0},
                   "source": "npv_py/sample 合成数据", "sha256": {fn: sha256_file(os.path.join(DATA, fn)) for fn in files}},
                  f, ensure_ascii=False, indent=2)
    tmp = tempfile.mkdtemp()
    make_sample_labels.main(["--input", os.path.join(DATA, "input.tsv"), "--out", tmp, "--pairs", "3000", "--grades", "10"])
    rows = []
    with open(os.path.join(tmp, "pairs.tsv"), encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            rows.append(line.rstrip("\n").split("\t")[:4] + [f"q{i // 3:05d}"])
    qids = sorted({r[4] for r in rows})
    random.Random(11).shuffle(qids)
    train_q = set(qids[: int(0.8 * len(qids))])
    for d, sel in ((TRAIN, [r for r in rows if r[4] in train_q]), (TEST, [r for r in rows if r[4] not in train_q])):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "pairs.tsv"), "w", encoding="utf-8") as f:
            f.write("url_a\turl_b\tpref\tgroup\tqid\n")
            for r in sel:
                f.write("\t".join(r) + "\n")
        print(f"{len(sel)} 对 → {d}/pairs.tsv")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
