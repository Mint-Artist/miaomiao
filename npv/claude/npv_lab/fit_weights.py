#!/usr/bin/env python3
"""用偏序标签拟合基础分权重，生成一份新的配置文件。

  python fit_weights.py --scores runs/baseline/npv_ori.tsv --pairs labels/pairs.tsv \
      --features spr_sr,pr,dr,ow --base-config configs/baseline.json --out configs/fitted.json

--scores 的 fea 列提供每个 url 的特征值，所以要先用 run_lab.py 跑一遍（特征在 features.py 里有即可，不必有权重）。
"""
import argparse

import lab  # noqa: F401
from lab.config_io import load_config, save_config
from lab.evaluate import read_pairs, read_scores
from lab.fit import fit_pairwise_logistic, rescale_weights
from npv.config import ScoreConfig
import dataclasses


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--features", default="spr_sr,pr,dr,ow")
    p.add_argument("--base-config", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--total-abs", type=float, default=80.0, help="权重绝对值之和缩放到该值，0 表示不缩放")
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--use-ties", action="store_true", help="把标签为相当的对作为 0.5 目标加入训练")
    a = p.parse_args(argv)
    names = [s.strip() for s in a.features.split(",") if s.strip()]
    scores = read_scores(a.scores, with_features=True)
    diffs, targets = [], []
    for pr in read_pairs(a.pairs):
        fa, fb = scores.get(pr["url_a"]), scores.get(pr["url_b"])
        if not fa or not fb or fa[1] is None or fb[1] is None:
            continue
        if pr["pref"] == "a":
            t = 1.0
        elif pr["pref"] == "b":
            t = 0.0
        elif a.use_ties:
            t = 0.5
        else:
            continue
        diffs.append([fa[1].get(n, 0.0) - fb[1].get(n, 0.0) for n in names])
        targets.append(t)
    base = load_config(a.base_config) if a.base_config else ScoreConfig()
    init = [base.fea_weight.get(n, 0.0) / 10.0 for n in names]  # 从基线权重的缩小版出发
    w, info = fit_pairwise_logistic(diffs, targets, names, a.l2, a.epochs, a.lr, init)
    print("拟合权重（原始）:", {k: round(v, 4) for k, v in w.items()})
    print("训练指标:", {k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()})
    if a.total_abs > 0:
        w = rescale_weights(w, a.total_abs)
        print(f"缩放到 |w| 之和 = {a.total_abs}:", {k: round(v, 3) for k, v in w.items()})
    cfg = dataclasses.replace(base, fea_weight=w)
    save_config(cfg, a.out)
    print("已写入", a.out)


if __name__ == "__main__":
    main()
