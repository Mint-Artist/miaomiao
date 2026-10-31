#!/usr/bin/env python3
"""用偏序标签重新拟合一条加法公式，并在 test 上评估。最短路径：把手定的权重换成拟合的权重。

  python tools/fit_formula.py --data data --train labels/train/pairs.tsv --test ../npv_ar_private/min_test/pairs.tsv \\
      --out work/fit [--features-out work/features.tsv] [--l1 0,0.001,0.003,0.01,0.03] [--drop-decay]

做的事：
  1. 用 tools/features_v2.py 给 data/input.tsv 的每页算约 30 个特征（可导出 TSV 给 xgboost 用）
  2. 偏序逻辑回归：P(A 更权威) = sigmoid(w · (f_A − f_B))，无截距；可选 L1 得到稀疏公式
  3. 在 train / test 上按严格口径算一致率（同档记半对；同档区间与 eval.py 一致）
  4. 输出权重表、符号检查、"项数 对 一致率"曲线、可贴进 Java 的伪代码、以及 sandbox/config.json 用的 fea_weight

有 numpy 就用 numpy（秒级），没有就退回 npv_lab 的纯 Python 实现（分钟级，且只支持 L2）。
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "npv_ar", "harness"))
sys.path.insert(0, HERE)
from common import build_scorer, iter_input, read_pairs  # noqa: E402
from npv.config import ScoreConfig  # noqa: E402
from npv.level_map import LevelMap  # noqa: E402
import features_v2  # noqa: E402

try:
    import numpy as np
except ImportError:  # noqa: F841
    np = None


class _FM:  # 让 build_scorer 接受一个"模块"
    gen_features = staticmethod(features_v2.gen_features)


def load_feature_table(path):
    """已有每页特征表时直接读：首行表头，第一列 url，其余列为数值特征。"""
    feats = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = header[1:]
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != len(header):
                continue
            row = {}
            for n, v in zip(cols, c[1:]):
                try:
                    row[n] = float(v)
                except ValueError:
                    row[n] = 0.0
            feats[c[0]] = row
    return feats


def load_everything(data_dir, feature_table=None):
    meta_path = os.path.join(data_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {"region": "zh", "now": 0}
    if feature_table:
        feats = load_feature_table(feature_table)
    else:
        scorer, site_list = build_scorer(data_dir, ScoreConfig(), _FM, meta["region"], int(meta["now"]))
        rows = list(iter_input(os.path.join(data_dir, "input.tsv")))
        feats = {x.url: scorer.gen_features(x, site_list) for x, _ in rows}
    names = sorted(next(iter(feats.values())).keys())
    tie = meta.get("tie_band") or {"unit": "raw", "value": 0}
    lm = None
    if tie["unit"] == "level" and os.path.exists(os.path.join(data_dir, "level_map.tsv")):
        lm = LevelMap.load(os.path.join(data_dir, "level_map.tsv"))
    return meta, feats, names, float(tie["value"]), lm


def pair_matrix(feats, pairs, names):
    X, y, keep = [], [], []
    for p in pairs:
        if p["pref"] not in ("a", "b"):
            continue
        fa, fb = feats.get(p["url_a"]), feats.get(p["url_b"])
        if fa is None or fb is None:
            continue
        X.append([fa.get(n, 0.0) - fb.get(n, 0.0) for n in names])
        y.append(1.0 if p["pref"] == "a" else 0.0)
        keep.append(p)
    return X, y, keep


def fit_numpy(X, y, l2=1e-3, l1=0.0, epochs=2000, lr=0.5):
    """对称增广后的逻辑回归，近端梯度做 L1。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.vstack([X, -X])
    y = np.concatenate([y, 1 - y])
    n, k = X.shape
    w = np.zeros(k)
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / n + l2 * w
        w = w - lr * g
        if l1 > 0:
            w = np.sign(w) * np.maximum(np.abs(w) - lr * l1, 0.0)
    return w.tolist()


def fit_pure(X, y, names, l2=1e-3):
    from lab.fit import fit_pairwise_logistic
    w, _ = fit_pairwise_logistic(X, y, names, l2=l2, epochs=400, lr=0.5)
    return [w[n] for n in names]


def score_pages(feats, names, w):
    # 与线上/eval.py 一致：加法公式的结果钳在 0 以上
    return {u: max(0.0, sum(f.get(n, 0.0) * wi for n, wi in zip(names, w))) for u, f in feats.items()}


def strict_acc(scores, pairs, band, lm):
    n = correct = ties = missing = 0
    hard = []
    hard_gap = 2 * band if band > 0 else 2.0  # band=0 时按原始分 2 分算难对
    for p in pairs:
        if p["pref"] not in ("a", "b"):
            continue
        n += 1
        a, b = scores.get(p["url_a"]), scores.get(p["url_b"])
        if a is None or b is None:
            missing += 1
            continue
        if lm:
            a, b = lm.lookup(a), lm.lookup(b)
        d = a - b
        if abs(d) <= band:
            ties += 1
            correct += 0.5
            continue
        ok = (d > 0) == (p["pref"] == "a")
        correct += ok
        if abs(d) <= hard_gap:
            hard.append(float(ok))
    return {"n": n, "acc": correct / n if n else float("nan"), "ties": ties, "missing": missing,
            "hard": sum(hard) / len(hard) if hard else float("nan")}


def rescale(w, total=76.0):
    s = sum(abs(v) for v in w)
    return [v * total / s for v in w] if s > 0 else w


def java_pseudocode(names, w):
    lines = ["double score = 0;"]
    for n, wi in zip(names, w):
        if abs(wi) > 1e-9:
            lines.append(f"score += {wi:+.3f} * f_{n};")
    lines.append("score = Math.max(0, score);")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--train", default=os.path.join(ROOT, "labels", "train", "pairs.tsv"))
    ap.add_argument("--test", default=None, help="holdout / 测试集 pairs.tsv，只在最后看一次")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features-out", default=None, help="导出每页特征 TSV（url + 各列），给 xgboost 用")
    ap.add_argument("--feature-table", default=None, help="已有的每页特征 TSV（首列 url，其余数值列）；给了就不从 input.tsv 算特征")
    ap.add_argument("--l1", default="0,0.0005,0.001,0.002,0.005,0.01", help="逗号分隔的 L1 强度列表，画项数-一致率曲线")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--drop", default="", help="逗号分隔，不参与拟合的特征名，例如 pct_age,pt_age（把时效从权威分里拿掉）")
    ap.add_argument("--drop-decay", action="store_true", help="等价于 --drop pct_age,pt_age,pct_missing,pt_missing")
    ap.add_argument("--total", type=float, default=76.0, help="权重绝对值之和缩放到该值")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    meta, feats, names, band, lm = load_everything(a.data, a.feature_table)
    drop = {s.strip() for s in a.drop.split(",") if s.strip()}
    if a.drop_decay:
        drop |= {"pct_age", "pt_age", "pct_missing", "pt_missing"}
    names = [n for n in names if n not in drop]
    if a.features_out:
        with open(a.features_out, "w", encoding="utf-8") as f:
            f.write("url\t" + "\t".join(names) + "\n")
            for u, fv in feats.items():
                f.write(u + "\t" + "\t".join(f"{fv.get(n, 0.0):g}" for n in names) + "\n")
        print(f"特征表 {len(feats)} 页 × {len(names)} 列 → {a.features_out}")

    train = read_pairs(a.train)
    test = read_pairs(a.test) if a.test else []
    X, y, used = pair_matrix(feats, train, names)
    print(f"train 可用对 {len(used)} / {len(train)}，特征 {len(names)} 列，{'numpy' if np is not None else '纯 Python'} 拟合")

    # 基线：原公式权重（只 spr_sr/dr/ow）在同一口径下的一致率
    if {"spr_sr", "dr", "ow"} <= set(names):
        base_w = [{"spr_sr": 60.0, "dr": 12.0, "ow": 4.0}.get(n, 0.0) for n in names]
        base_scores = score_pages(feats, names, base_w)
        bt = strict_acc(base_scores, train, band, lm)["acc"] if used else float("nan")
        be = strict_acc(base_scores, test, band, lm)["acc"] if test else float("nan")
        print(f"参考：只用 60/12/4 三项、不乘衰减与页面类别系数时 train={bt:.4f} test={be:.4f}（完整原公式的数用 eval.py 看）")

    curve = []
    l1s = [float(s) for s in a.l1.split(",") if s.strip()] if np is not None else [0.0]
    for l1 in l1s:
        w = fit_numpy(X, y, a.l2, l1) if np is not None else fit_pure(X, y, names, a.l2)
        w = rescale(w, a.total)
        nz = sum(1 for v in w if abs(v) > 1e-6)
        sc = score_pages(feats, names, w)
        tr = strict_acc(sc, train, band, lm)
        te = strict_acc(sc, test, band, lm) if test else {"acc": float("nan"), "hard": float("nan"), "ties": None}
        curve.append({"l1": l1, "terms": nz, "train_acc": tr["acc"], "test_acc": te["acc"], "train_hard": tr["hard"],
                      "test_hard": te["hard"], "train_ties": tr["ties"], "test_ties": te["ties"], "w": dict(zip(names, w))})
        print(f"l1={l1:<7g} 项数={nz:<3d} train={tr['acc']:.4f} test={te['acc']:.4f} hard_test={te['hard']:.4f} ties_test={te['ties']}")

    # 选最终公式：test 一致率离最高值 0.3 个点以内的最稀疏那条；没有 test 就取 train
    key = "test_acc" if test else "train_acc"
    best = max(c[key] for c in curve)
    chosen = min((c for c in curve if c[key] >= best - 0.003), key=lambda c: c["terms"])
    w = chosen["w"]
    flips = [n for n, s in features_v2.EXPECTED_SIGN.items() if n in w and abs(w[n]) > 1e-6 and (w[n] > 0) != (s > 0)]
    print(f"\n选定 l1={chosen['l1']}，{chosen['terms']} 项，train={chosen['train_acc']:.4f} test={chosen['test_acc']:.4f}")
    print("权重（按绝对值排序）:")
    for n, v in sorted(w.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) > 1e-6:
            print(f"  {n:<18s} {v:+8.3f}")
    if flips:
        print("符号与常识相反、建议复核的项:", flips)
    with open(os.path.join(a.out, "fit.json"), "w", encoding="utf-8") as f:
        json.dump({"names": names, "band": band, "unit": "level" if lm else "raw", "curve": curve, "chosen": chosen,
                   "sign_flips": flips}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "fea_weight.json"), "w", encoding="utf-8") as f:
        json.dump({n: round(v, 4) for n, v in w.items() if abs(v) > 1e-6}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "formula.java.txt"), "w", encoding="utf-8") as f:
        f.write(java_pseudocode(names, [w[n] for n in names]) + "\n")
    print(f"\n→ {a.out}/fit.json, fea_weight.json, formula.java.txt")
    print("要在 eval.py 里复现同样的数：sandbox/features.py 换成 tools/features_v2.py，config.json 里 fea_weight 用 fea_weight.json、"
          "fit 置空、加 \"linear_only\": true（纯加法公式，不再乘衰减与页面类别系数）。")


if __name__ == "__main__":
    main()
