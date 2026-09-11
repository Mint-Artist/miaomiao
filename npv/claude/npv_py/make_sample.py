#!/usr/bin/env python3
"""生成一套可直接运行的合成样例数据（输入 TSV + 七张表），用于本地跑通流程与做实验。

  python make_sample.py --out sample --rows 5000 --seed 7

数据是随机的，只保证格式与 Java 作业一致，不代表真实分布。
"""
import argparse
import json
import os
import random

SITES = [
    ("www.gov.cn", 95, 8.0, "3", 3), ("www.moe.edu.cn", 90, 6.5, "3", 3),
    ("baike.baidu.com", 92, 9.0, "2", 2), ("www.bendibao.com", 70, 3.0, "", 0),
    ("news.sina.com.cn", 85, 5.0, "1", 1), ("www.zhihu.com", 88, 7.0, "", 0),
    ("blog.csdn.net", 80, 4.0, "", 0), ("www.example.com", 40, 1.0, "", 0),
    ("shop.taobao.com", 86, 6.0, "", 0), ("tieba.baidu.com", 84, 5.5, "", 0),
    ("www.smallsite.net", 20, 0.3, "", 0), ("music.163.com", 82, 4.5, "", 0),
    ("spam.example.org", 5, 0.0, "", 0), ("www.hknews.cn", 60, 2.0, "", 0),
    ("edu.example.edu.com", 50, 1.5, "", 0), ("www.ucla.edu", 90, 7.5, "", 0),
]
PATHS = ["/", "/article/{n}.html", "/news/{n}", "/u/{n}", "/user/{n}/posts", "/list?page={n}",
         "/item/{n}", "/forum/thread-{n}", "/video/{n}"]


def write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # 表
    write(os.path.join(a.out, "spr.tsv"), [f"{s}\t{spr}" for s, _, spr, _, _ in SITES])
    write(os.path.join(a.out, "dr_site.tsv"),
          [f"{s}\t{dr}" for s, _, _, dr, _ in SITES if dr])
    write(os.path.join(a.out, "dr_suffix.tsv"), ["gov.cn\t3", "edu.cn\t3", ".edu\t3", ".com\t2", ".net\t1"])
    write(os.path.join(a.out, "ow.tsv"), ["www.zhihu.com\t1", "shop.taobao.com\t1", "www.example.com\t0"])
    write(os.path.join(a.out, "ow_blacklist.txt"), ["spam.example.org"])
    write(os.path.join(a.out, "adc_whitelist.txt"), ["moe.edu.cn"])

    # 输入
    now = 1757030400  # 2026-09-05 00:00:00 UTC
    lines = []
    for n in range(a.rows):
        site, sr, spr, _dr, level = rnd.choice(SITES)
        url = "https://" + site + rnd.choice(PATHS).format(n=n)
        pc = 0
        for bit in (10, 11, 13, 16, 18, 19, 20, 22, 24, 26, 29, 32, 33, 38):
            if rnd.random() < 0.06:
                pc |= 1 << bit
        r = rnd.random()
        pct = 0 if r < 0.1 else now - rnd.randint(0, 3000) * 86400
        if 0.1 <= r < 0.2:
            pct *= 1000  # 13 位毫秒
        pt = 0 if rnd.random() < 0.3 else now - rnd.randint(0, 4000) * 86400
        obj = {
            "adc": json.dumps({"level": level}) if level else "",
            "pcLong": pc, "pct": pct, "pt": pt,
            "pureTextLen": rnd.choice([50, 200, 400, 800, 3000]),
            "sr": -1 if rnd.random() < 0.05 else sr + rnd.randint(-5, 5),
            "spr": max(0.0, spr + rnd.uniform(-0.5, 0.5)),
        }
        if rnd.random() < 0.01:  # 故意混入坏行
            lines.append(f"{url}\tx\t0\tx\tx\tnot-json")
            continue
        flag = rnd.choice(["0", "1", "2", "3", "5", "5"])
        lines.append("\t".join([url, "x", flag, "x", "x", json.dumps(obj, ensure_ascii=False)]))
    write(os.path.join(a.out, "input.tsv"), lines)
    print(f"wrote {a.rows} rows to {a.out}/ (now={now})")


if __name__ == "__main__":
    main()
