"""各命令行脚本共用的参数与数据读取。"""
import argparse
import time
from typing import Iterator, Tuple

from npv import ScoreInput, load_tables
from npv.config import ScoreConfig, region_site_list
from npv.io import BadRow, parse_line

from .config_io import load_config
from .scorer import LabScorer


def add_table_args(p: argparse.ArgumentParser, with_input: bool = True) -> None:
    if with_input:
        p.add_argument("--input", required=True, help="输入 TSV（与 Java 作业同格式）")
    p.add_argument("--spr", required=True)
    p.add_argument("--dr-site", required=True)
    p.add_argument("--dr-suffix", required=True)
    p.add_argument("--ow", required=True)
    p.add_argument("--pr-split", required=True)
    p.add_argument("--ow-blacklist", required=True)
    p.add_argument("--adc-whitelist", required=True)
    p.add_argument("--region", default="", help="'zh' 使用中文区官网白名单")
    p.add_argument("--now", type=int, default=None, help="时间衰减基准（秒级时间戳）")
    p.add_argument("--config", default=None, help="configs/*.json，缺省用基线参数")


def build_scorer(args) -> Tuple[LabScorer, tuple]:
    tables = load_tables(args.spr, args.dr_site, args.dr_suffix, args.ow, args.pr_split,
                         args.ow_blacklist, args.adc_whitelist)
    cfg = load_config(args.config) if args.config else ScoreConfig()
    now = args.now if args.now is not None else int(time.time())
    return LabScorer(tables, now, cfg), tuple(region_site_list(args.region))


def iter_rows(path: str, limit: int = None) -> Iterator[Tuple[ScoreInput, str]]:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield parse_line(line)
            except BadRow:
                continue
            n += 1
            if limit and n >= limit:
                return
