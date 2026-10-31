#!/usr/bin/env python3
"""打分子进程：只接触快照、sandbox 和 train 标签，看不到 dev 与 holdout。

  python score.py --sandbox DIR --snapshot DIR --train-pairs FILE --out DIR [--cv 5]

步骤：加载 sandbox 特征与配置 -> （若 fit 非空）在 train 上拟合权重 -> 按 qid 分组 K 折交叉验证 -> 全量打分。
输出：out/npv_ori.tsv（url, score, 特征 JSON）、out/weights.json、out/cv.json、out/stats.json。
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BASELINE_FEATURES, baseline_scorer, build_scorer, count_bad_rows, iter_input,  # noqa: E402
                    load_features_module, load_sandbox_config, read_pairs, read_sandbox_options, format_row,
                    features_json, write_json)
from metrics import strict_pairwise  # noqa: E402
from lab.fit import fit_pairwise_logistic, rescale_weights  # noqa: E402


def pair_diffs(feats, pairs, names):
    diffs, targets = [], []
    for p in pairs:
        fa, fb = feats.get(p["url_a"]), feats.get(p["url_b"])
        if fa is None or fb is None or p["pref"] not in ("a", "b"):
            continue
        diffs.append([fa.get(n, 0.0) - fb.get(n, 0.0) for n in names])
        targets.append(1.0 if p["pref"] == "a" else 0.0)
    return diffs, targets


def fit_weights(feats, pairs, cfg, fit_names, total_abs):
    names = list(cfg.fea_weight) + [n for n in fit_names if n not in cfg.fea_weight]
    diffs, targets = pair_diffs(feats, pairs, names)
    if not diffs:
        raise RuntimeError("train 上没有可用于拟合的对")
    init = [cfg.fea_weight.get(n, 0.0) / 10.0 for n in names]
    w, info = fit_pairwise_logistic(diffs, targets, names, l2=1e-3, epochs=300, lr=0.5, init=init)
    return rescale_weights(w, total_abs), info


def score_all(scorer, rows, site_list):
    out = {}
    for x, _flag in rows:
        r = scorer.score(x, site_list)
        out[x.url] = (r.score, r.features)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sandbox", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--train-pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args(argv)

    with open(os.path.join(a.snapshot, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg, fit_names = load_sandbox_config(os.path.join(a.sandbox, "config.json"))
    opts = read_sandbox_options(os.path.join(a.sandbox, "config.json"))
    fm = load_features_module(os.path.join(a.sandbox, "features.py"))
    scorer, site_list = build_scorer(a.snapshot, cfg, fm, meta["region"], int(meta["now"]), opts["linear_only"])
    input_path = os.path.join(a.snapshot, "input.tsv")
    rows = list(iter_input(input_path))
    os.makedirs(a.out, exist_ok=True)

    # 基线特征不变性检查：前 6 个特征必须与冻结基线逐条一致
    base, _ = baseline_scorer(a.snapshot, meta["region"], int(meta["now"]))
    mismatch = 0
    for x, _ in rows[:500]:
        fb = base.gen_features(x, site_list)
        fs = scorer.gen_features(x, site_list)
        if any(abs(fb[k] - fs.get(k, float("nan"))) > 1e-9 for k in BASELINE_FEATURES):
            mismatch += 1
    feats = {x.url: scorer.gen_features(x, site_list) for x, _ in rows}
    missing = [k for k in cfg.fea_weight if k not in next(iter(feats.values()), {})]
    if missing:
        raise KeyError(f"fea_weight 中的特征不存在于 features.py 输出: {missing}")
    for n in fit_names:
        if n not in next(iter(feats.values()), {}):
            raise KeyError(f"fit 列表中的特征不存在于 features.py 输出: {n}")

    pairs = read_pairs(a.train_pairs)
    total_abs = sum(abs(v) for v in cfg.fea_weight.values()) or 76.0
    fit_info = None
    if fit_names:
        w, fit_info = fit_weights(feats, pairs, cfg, fit_names, total_abs)
        scorer = scorer.with_weights(w)
    write_json(os.path.join(a.out, "weights.json"), {"fea_weight": scorer.cfg.fea_weight, "fit": fit_names, "info": fit_info})

    # 按 qid 分组的 K 折交叉验证（第一道关的依据）
    cv = {"folds": [], "mean": None}
    if a.cv and a.cv > 1 and pairs:
        groups = defaultdict(list)
        for pr in pairs:
            groups[pr["qid"]].append(pr)
        keys = list(groups)
        random.Random(a.seed).shuffle(keys)
        folds = [[] for _ in range(a.cv)]
        for i, k in enumerate(keys):
            folds[i % a.cv].extend(groups[k])
        for i in range(a.cv):
            held = folds[i]
            train = [pr for j in range(a.cv) if j != i for pr in folds[j]]
            if fit_names:
                w_i, _ = fit_weights(feats, train, cfg, fit_names, total_abs)
                sc_i = scorer.with_weights(w_i)
            else:
                sc_i = scorer
            urls = {pr["url_a"] for pr in held} | {pr["url_b"] for pr in held}
            by_url = {x.url: x for x, _ in rows if x.url in urls}
            s_i = {u: (sc_i.score(x, site_list).score, None) for u, x in by_url.items()}
            m = strict_pairwise(s_i, held)
            cv["folds"].append({"n": m["n_eligible"], "accuracy": m["accuracy"]})
        accs = [f["accuracy"] for f in cv["folds"] if f["n"]]
        cv["mean"] = sum(accs) / len(accs) if accs else None
        cv["std"] = (sum((x - cv["mean"]) ** 2 for x in accs) / max(1, len(accs) - 1)) ** 0.5 if accs else None
    write_json(os.path.join(a.out, "cv.json"), cv)

    scores = score_all(scorer, rows, site_list)
    with open(os.path.join(a.out, "npv_ori.tsv"), "w", encoding="utf-8") as f:
        for url, (s, fea) in scores.items():
            f.write(format_row(url, s, features_json(fea)) + "\n")
    train_m = strict_pairwise(scores, pairs)
    write_json(os.path.join(a.out, "stats.json"), {
        "rows": len(rows), "bad_rows": count_bad_rows(input_path),
        "baseline_feature_mismatch": mismatch, "train_accuracy": train_m["accuracy"],
        "train_n": train_m["n_eligible"], "feature_names": sorted(next(iter(feats.values()), {})),
    })
    print(f"scored {len(rows)} rows; train_acc={train_m['accuracy']:.4f}; cv_mean={cv['mean']}")


if __name__ == "__main__":
    main()
