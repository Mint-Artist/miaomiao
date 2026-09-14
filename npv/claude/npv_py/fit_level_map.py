#!/usr/bin/env python3
"""从线上均匀抽样的 (npv_ori[, npv]) 拟合"分数 -> 等级"阈值表。

  # 样本文件是线上 npv 输出（url, npv_ori, npv, npv_fea）：
  python fit_level_map.py --sample online_npv_sample.tsv --out level_map.tsv
  # 样本只有分数（url, npv_ori, ...）：
  python fit_level_map.py --sample online_npv_ori_sample.tsv --level-col -1 --method quantile --out level_map.tsv

之后给 run_npv.py / run_lab.py / npv_exact.py 传 --level-map level_map.tsv，样本数据就能算出与线上可比的 npv。
前提：样本必须是对参与归一化的页面（第 2 列为 1、2、5 的）均匀随机抽的，且 npv_ori 与 npv 来自同一次作业。
"""
import argparse
import sys

from npv.level_map import LevelMap, fit_observed, fit_quantile, validate


def read_sample(path, score_col, level_col, sep):
    scores, levels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split(sep)
            if len(cols) <= score_col:
                continue
            try:
                s = float(cols[score_col])
            except ValueError:
                continue  # 表头或坏行
            lv = None
            if level_col >= 0 and len(cols) > level_col:
                try:
                    lv = int(float(cols[level_col]))
                except ValueError:
                    lv = None
            scores.append(s)
            levels.append(lv)
    return scores, levels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, help="线上样本 TSV")
    p.add_argument("--score-col", type=int, default=1, help="npv_ori 所在列（0 起），默认 1")
    p.add_argument("--level-col", type=int, default=2, help="npv 所在列（0 起），默认 2；-1 表示样本没有 npv")
    p.add_argument("--sep", default="\t")
    p.add_argument("--method", choices=["auto", "quantile", "observed"], default="auto",
                   help="auto：有 npv 列用 observed，否则 quantile")
    p.add_argument("--z-start", type=float, default=-2.0)
    p.add_argument("--z-end", type=float, default=3.0)
    p.add_argument("--buckets", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    scores, levels = read_sample(a.sample, a.score_col, a.level_col, a.sep)
    has_levels = a.level_col >= 0 and any(lv is not None for lv in levels)
    method = a.method if a.method != "auto" else ("observed" if has_levels else "quantile")
    if method == "observed" and not has_levels:
        sys.exit("observed 方法需要 npv 列，请检查 --level-col")
    if method == "observed":
        pairs = [(s, lv) for s, lv in zip(scores, levels) if lv is not None]
        lm = fit_observed([s for s, _ in pairs], [lv for _, lv in pairs], a.z_start, a.z_end, a.buckets)
    else:
        lm = fit_quantile(scores, a.z_start, a.z_end, a.buckets)
    lm.meta.update({"source": a.sample, "z_range": f"[{a.z_start:g},{a.z_end:g}]"})
    lm.save(a.out)
    print(f"method={method}\tn_sample={len(scores)}\tlevels={lm.levels}\tout={a.out}", file=sys.stderr)
    print(f"min_score(level 1)={lm.thresholds[0]!r}\tmin_score(level 400)={lm.thresholds[399]!r}\t"
          f"min_score(level 800)={lm.thresholds[799]!r}\tmin_score(level 1000)={lm.thresholds[-1]!r}", file=sys.stderr)
    if has_levels:
        pairs = [(s, lv) for s, lv in zip(scores, levels) if lv is not None]
        v = validate(lm, [s for s, _ in pairs], [lv for _, lv in pairs])
        print("回代校验（样本自身）:\t" + "\t".join(f"{k}={v[k]:.4f}" if isinstance(v[k], float) else f"{k}={v[k]}"
                                            for k in v), file=sys.stderr)
        if method == "observed":
            lq = fit_quantile(scores, a.z_start, a.z_end, a.buckets)
            vq = validate(lq, [s for s, _ in pairs], [lv for _, lv in pairs])
            print("对照 quantile 方法:\t" + "\t".join(f"{k}={vq[k]:.4f}" if isinstance(vq[k], float) else f"{k}={vq[k]}"
                                                 for k in vq), file=sys.stderr)


if __name__ == "__main__":
    main()
