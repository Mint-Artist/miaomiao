#!/usr/bin/env python3
"""NPV 打分作业的 Python 版入口，对应 Java PageValueScoreMain。

用法示例：
  python run_npv.py --input sample/input.tsv --spr sample/spr.tsv --dr-site sample/dr_site.tsv \
      --dr-suffix sample/dr_suffix.tsv --ow sample/ow.tsv \
      --ow-blacklist sample/ow_blacklist.txt --adc-whitelist sample/adc_whitelist.txt \
      --output out --region zh --scroll 1 --now 1757030400

输出：
  {output}/npv_ori.tsv   url, npv_ori, npv_fea          （scroll=1 时只含非高质量网页）
  {output}/npv.tsv       url, npv_ori, npv, npv_fea     （scroll=1 时的高质量网页）
"""
import argparse
import os
import sys
import time
from collections import Counter

from npv import PageValueScore, assign_levels, load_tables
from npv.config import ScoreConfig, region_site_list
from npv.io import BadRow, features_json, format_row, parse_line

HQW_FLAGS = {"1", "2", "5"}  # Java main 中视为“高质量网页”的第 2 列取值


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--spr", required=True)
    p.add_argument("--dr-site", required=True)
    p.add_argument("--dr-suffix", required=True)
    p.add_argument("--ow", required=True)
    p.add_argument("--ow-blacklist", required=True)
    p.add_argument("--adc-whitelist", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--region", default="", help="'zh' 使用中文区官网白名单")
    p.add_argument("--scroll", type=int, default=0, help="1 = 拆分高质量网页并做排名归一化")
    p.add_argument("--now", type=int, default=None, help="时间衰减基准（秒级时间戳），缺省取当前时间")
    return p


def run(args, config: ScoreConfig = None) -> dict:
    now = args.now if args.now is not None else int(time.time())
    tables = load_tables(args.spr, args.dr_site, args.dr_suffix, args.ow,
                         args.ow_blacklist, args.adc_whitelist)
    scorer = PageValueScore(tables, now, config)
    site_list = region_site_list(args.region)
    os.makedirs(args.output, exist_ok=True)

    stats = Counter()
    hqw_rows = []  # scroll=1 时的高质量网页，需全部读入后排序
    ori_path = os.path.join(args.output, "npv_ori.tsv")
    with open(args.input, encoding="utf-8") as fin, open(ori_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                x, flag = parse_line(line)
            except BadRow:
                stats["bad_rows"] += 1
                continue
            stats["rows"] += 1
            r = scorer.score(x, site_list)
            fea = features_json(r.features)
            if args.scroll == 1 and flag in HQW_FLAGS:
                hqw_rows.append((x.url, r.score, fea))
            else:
                fout.write(format_row(x.url, r.score, fea) + "\n")
                stats["npv_ori_rows"] += 1

    if args.scroll == 1:
        c = config or ScoreConfig()
        ranked = assign_levels(hqw_rows, score_of=lambda t: t[1], tie_key=lambda t: t[0],
                               z_start=c.norm_z_start, z_end=c.norm_z_end, buckets=c.norm_buckets)
        npv_path = os.path.join(args.output, "npv.tsv")
        with open(npv_path, "w", encoding="utf-8") as fout:
            for (url, score, fea), _rank, level in ranked:
                fout.write(format_row(url, score, fea, level) + "\n")
        stats["npv_rows"] = len(ranked)

    stats["now_seconds"] = now
    return dict(stats)


def main(argv=None):
    args = build_parser().parse_args(argv)
    stats = run(args)
    for k, v in stats.items():
        print(f"{k}\t{v}", file=sys.stderr)


if __name__ == "__main__":
    main()
