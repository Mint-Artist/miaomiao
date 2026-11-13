"""从 query 的 jsonl 文件里只取出 query 文本，写成 txt：一行一条，顺序与原文件一致。

用法（命令行）：
    python extract_queries.py queries_sample1000_html.jsonl                 # 输出 queries_sample1000_html.txt
    python extract_queries.py queries.jsonl -o out.txt --dedup              # 指定输出文件，并去掉重复的 query
    python extract_queries.py queries.jsonl --field question                # query 字段名不是 "query" 时

用法（notebook）：
    from extract_queries import extract_queries
    extract_queries(os.path.join(OUT_ROOT, "queries_sample1000_html.jsonl"))

处理规则：
    - query 内部的换行、制表符、连续空白合并成一个空格，保证一条 query 只占一行；
    - 空 query、缺字段、"nan" / "none" / "null"、解析失败的行跳过，并在最后汇总报告；
    - 默认不去重（保持和 jsonl 一一对应）；需要时加 --dedup，保留第一次出现的那条；
    - 支持 .jsonl.gz 输入。
"""
import argparse
import gzip
import json
import os
import sys


def extract_queries(path, out_path=None, field="query", dedup=False):
    """读取 jsonl，把 field 字段写成 txt（一行一条），返回输出路径。"""
    if out_path is None:
        base = path[:-3] if path.endswith(".gz") else path
        out_path = os.path.splitext(base)[0] + ".txt"
    opener = gzip.open if path.endswith(".gz") else open
    stats = {"总行数": 0, "写出": 0, "空行": 0, "解析失败": 0, "缺字段或为空": 0, "重复已去掉": 0}
    seen = set()
    with opener(path, "rt", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                stats["空行"] += 1
                continue
            stats["总行数"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["解析失败"] += 1
                continue
            q = rec.get(field) if isinstance(rec, dict) else None
            q = " ".join(str(q).split()) if q is not None else ""
            if not q or q.lower() in ("nan", "none", "null"):
                stats["缺字段或为空"] += 1
                continue
            if dedup:
                if q in seen:
                    stats["重复已去掉"] += 1
                    continue
                seen.add(q)
            fout.write(q + "\n")
            stats["写出"] += 1
    print(f"{path} → {out_path}")
    print("，".join(f"{k} {v}" for k, v in stats.items() if v or k in ("总行数", "写出")))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="从 query jsonl 中提取 query 文本，输出 txt（一行一条）")
    ap.add_argument("path", help="输入 jsonl（支持 .gz）")
    ap.add_argument("-o", "--out", default=None, help="输出 txt 路径，默认与输入同名、扩展名改为 .txt")
    ap.add_argument("--field", default="query", help='query 所在字段名，默认 "query"')
    ap.add_argument("--dedup", action="store_true", help="去掉重复的 query（保留第一次出现的）")
    a = ap.parse_args()
    if not os.path.exists(a.path):
        sys.exit(f"找不到文件：{a.path}")
    extract_queries(a.path, a.out, a.field, a.dedup)
