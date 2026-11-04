"""query 文件检查与清洗：去掉 NaN / 空 url（常见于 pandas 导出时精排结果不足 10 条留下的空格子）。

notebook 里用：把本文件内容贴进一个单元格运行，然后
    check_queries(QUERY_FILE)                  # 看有多少行有问题
    QUERY_FILE = clean_queries(QUERY_FILE)     # 清洗后另存为 *_clean.jsonl，原文件不动
    # 之后重新运行第 13 段（load_queries 的默认路径在定义时固定），再运行第 14 段
命令行用：
    python query_tools.py /path/to/queries.jsonl            # 只检查
    python query_tools.py /path/to/queries.jsonl --clean    # 检查并清洗
"""
import json
import sys


def _ok(u) -> bool:
    # JSON 对象的键只能是字符串，NaN 作 relevance 的键会变成 "NaN"；pandas astype(str) 也会产生 "nan" / "None"
    return isinstance(u, str) and bool(u.strip()) and u.strip().lower() not in ("nan", "none", "null")


def _urls(d: dict) -> list:
    return list((d.get("relevance") or {}).keys()) or d.get("relevant_urls", [])


def check_queries(path: str, show: int = 5):
    """打印有问题的行：(行号, qid, 非法 url 数, url 总数)；query 本身为空的也算。"""
    bad, n = [], 0
    for i, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        n += 1
        d = json.loads(line)
        urls = _urls(d)
        n_bad = sum(not _ok(u) for u in urls)
        if n_bad or not _ok(d.get("query")):
            bad.append((i + 1, d.get("qid"), n_bad, len(urls)))
    print(f"共 {n} 条，有问题的行 {len(bad)} 条，例：{bad[:show]}   # (行号, qid, 非法 url 数, url 总数)")
    return bad


def clean_queries(path: str, out_path: str = None) -> str:
    """去掉非法 url；query 为空的整条丢掉；url 去光的 query 保留（评测时不计分）。返回清洗后的文件路径。"""
    out_path = out_path or (path[: -len(".jsonl")] if path.endswith(".jsonl") else path) + "_clean.jsonl"
    n_in = n_out = n_drop_url = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            n_in += 1
            d = json.loads(line)
            if not _ok(d.get("query")):
                continue
            if "relevance" in d:
                n_drop_url += sum(not _ok(u) for u in d["relevance"])
                d["relevance"] = {u: g for u, g in d["relevance"].items() if _ok(u)}
            if "relevant_urls" in d:
                n_drop_url += sum(not _ok(u) for u in d["relevant_urls"])
                d["relevant_urls"] = [u for u in d["relevant_urls"] if _ok(u)]
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"读入 {n_in} 条，写出 {n_out} 条，去掉非法 url {n_drop_url} 个 → {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：python query_tools.py <queries.jsonl> [--clean]")
    check_queries(sys.argv[1])
    if "--clean" in sys.argv[2:]:
        clean_queries(sys.argv[1])
