#!/usr/bin/env python3
"""对一批数据做结构诊断，输出 Markdown 报告（stdout 或 --out）与 JSON（--json）。

  python diagnose_data.py --input ... [表参数] --out runs/diagnose.md
"""
import argparse
import json

import lab  # noqa: F401
from lab.cli_common import add_table_args, build_scorer, iter_rows
from lab.diagnose import diagnose, format_report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_table_args(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-site-rows", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    scorer, site_list = build_scorer(a)
    d = diagnose(scorer, iter_rows(a.input, a.limit), site_list, a.min_site_rows)
    report = format_report(d)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    else:
        print(report)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
