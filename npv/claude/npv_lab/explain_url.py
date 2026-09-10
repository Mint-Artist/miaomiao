#!/usr/bin/env python3
"""解释一条或几条记录的分数是怎么算出来的。

  python explain_url.py --input ... [表参数] --url https://www.gov.cn/article/1.html
  python explain_url.py --input ... [表参数] --grep gov.cn --limit 3
"""
import argparse

import lab  # noqa: F401
from lab.cli_common import add_table_args, build_scorer, iter_rows
from lab.explain import explain, format_explanation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_table_args(p)
    p.add_argument("--url", action="append", default=[], help="精确匹配的 url，可多次")
    p.add_argument("--grep", default=None, help="url 含该子串即解释")
    p.add_argument("--limit", type=int, default=5)
    a = p.parse_args(argv)
    scorer, site_list = build_scorer(a)
    wanted = set(a.url)
    shown = 0
    for x, _flag in iter_rows(a.input):
        if (wanted and x.url in wanted) or (a.grep and a.grep in x.url):
            print(format_explanation(explain(scorer, x, site_list)))
            print("-" * 60)
            shown += 1
            if shown >= a.limit:
                break
    if shown == 0:
        print("没有匹配的记录")


if __name__ == "__main__":
    main()
