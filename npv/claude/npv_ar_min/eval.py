#!/usr/bin/env python3
"""薄版本的冻结评估：用 sandbox/ 的特征与配置给 data/ 打分，在 labels/train 上算严格偏序一致率，只打印一行。

  python eval.py            # agent 用：在 labels/train 上评估
  python eval.py --test     # 人用：在仓库外的测试集上评估（agent 的权限配置不允许带参数）

输出示例：
  acc=0.7542 hard=0.5312 ties=12 missing=0 n=11800 complexity=85 fit=off unit=level band=50

- acc：严格一致率 = (判对 + 0.5×同档) / 非"相当"标签的对数（含打分缺失的对，缺失算错）
- 同档：两页分差 ≤ band。band 的单位由 data/meta.json 的 tie_band 决定：
    {"unit": "level", "value": 50}  → 先用 data/level_map.tsv 把原始分映射到线上 1~1000 等级，再比等级差
    {"unit": "raw",   "value": 0}   → 直接比原始分（0 表示只有完全相等才算同档）
- hard：分差 ≤ 2×band（band=0 时取原始分 2 分）的"难对"上的一致率，防止只在容易的对上刷分
- fit=on 表示 config.json 的 fit 列表非空，权重在 labels/train 上拟合过，acc 是样本内数字，偏乐观
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AR_HARNESS = os.path.join(os.path.dirname(HERE), "npv_ar", "harness")
sys.path.insert(0, AR_HARNESS)
from common import build_scorer, iter_input, load_features_module, load_sandbox_config, read_pairs  # noqa: E402
from metrics import complexity  # noqa: E402
from score import fit_weights  # noqa: E402
from npv.level_map import LevelMap  # noqa: E402

DATA = os.path.join(HERE, "data")
SANDBOX = os.path.join(HERE, "sandbox")
TRAIN = os.path.join(HERE, "labels", "train", "pairs.tsv")
TEST = os.path.join(os.path.dirname(HERE), "npv_ar_private", "min_test", "pairs.tsv")


def banded_accuracy(scores, pairs, band, hard_gap):
    n = correct = ties = missing = 0
    hard = []
    for p in pairs:
        if p["pref"] not in ("a", "b"):
            continue
        n += 1
        a, b = scores.get(p["url_a"]), scores.get(p["url_b"])
        if a is None or b is None:
            missing += 1
            continue
        d = a - b
        if abs(d) <= band:
            ties += 1
            correct += 0.5
            hard.append(0.5)
            continue
        ok = (d > 0) == (p["pref"] == "a")
        correct += ok
        if abs(d) <= hard_gap:
            hard.append(1.0 if ok else 0.0)
    return {"n": n, "acc": correct / n if n else float("nan"), "ties": ties, "missing": missing,
            "hard": sum(hard) / len(hard) if hard else float("nan"), "n_hard": len(hard)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true", help="在仓库外的测试集上评估")
    a = p.parse_args(argv)
    with open(os.path.join(DATA, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    tie = meta.get("tie_band") or {"unit": "raw", "value": 0}
    band, unit = float(tie["value"]), tie["unit"]
    lm = None
    if unit == "level":
        lm_path = os.path.join(DATA, "level_map.tsv")
        if not os.path.exists(lm_path):
            sys.exit("meta.json 的 tie_band 单位是 level，但缺少 data/level_map.tsv（用 npv_py/fit_level_map.py 生成）")
        lm = LevelMap.load(lm_path)

    cfg, fit_names = load_sandbox_config(os.path.join(SANDBOX, "config.json"))
    fm = load_features_module(os.path.join(SANDBOX, "features.py"))
    scorer, site_list = build_scorer(DATA, cfg, fm, meta["region"], int(meta["now"]))
    rows = list(iter_input(os.path.join(DATA, "input.tsv")))
    pairs = read_pairs(TEST if a.test else TRAIN)
    if fit_names:
        feats = {x.url: scorer.gen_features(x, site_list) for x, _ in rows}
        total_abs = sum(abs(v) for v in cfg.fea_weight.values()) or 76.0
        w, _ = fit_weights(feats, read_pairs(TRAIN), cfg, fit_names, total_abs)  # 拟合永远只用 train
        scorer = scorer.with_weights(w)
    scores = {}
    for x, _ in rows:
        s = scorer.score(x, site_list).score
        scores[x.url] = float(lm.lookup(s)) if lm else s
    hard_gap = 2 * band if band > 0 else 2.0
    m = banded_accuracy(scores, pairs, band, hard_gap)
    cx = complexity(os.path.join(SANDBOX, "features.py"), scorer.cfg.fea_weight, len(cfg.pc_rules))
    print(f"{'test_' if a.test else ''}acc={m['acc']:.4f} hard={m['hard']:.4f} ties={m['ties']} missing={m['missing']} "
          f"n={m['n']} complexity={cx['total']} fit={'on' if fit_names else 'off'} unit={unit} band={band:g}")


if __name__ == "__main__":
    main()
