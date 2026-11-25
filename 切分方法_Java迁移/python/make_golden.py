"""生成"标准答案"（golden）：用 Python 参考实现切分，输出 jsonl，供 Java 版逐字对照。

三种输入来源（可以同时用）：
  --cases DIR       构造用例目录：*.html 作为 html 输入，*.md 作为 markdown 输入
  --fuzz-html N     随机 html 页 N 个（fuzz_pages.gen_html，固定种子）
  --fuzz-md N       随机 markdown 页 N 个
  --input FILE      真实网页 jsonl / jsonl.gz（每行至少有 markdown、html 之一；url 可选）——在服务器上用
  --limit N         --input 只取前 N 条

输出（--out，.gz 结尾则压缩）：
  第 1 行 {"type": "meta", "bs_parser", "bs4", "lxml", "methods", "created", ...}
  之后每行 {"type": "page", "id", "markdown", "html", "expected": {方法: [块, ...]}, "container_kinds": [...]}
  只对有对应输入的方法给出 expected（没有 markdown 的页不出现 md 方法）。

示例：
  python make_golden.py --cases ../testdata/cases --fuzz-html 300 --fuzz-md 300 --out ../testdata/golden_cases.jsonl.gz
  python make_golden.py --input /path/pages.jsonl.gz --limit 5000 --out golden_real.jsonl.gz
"""
import argparse
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_ref  # noqa: E402
from fuzz_pages import gen_html, gen_markdown  # noqa: E402


def _open(path, mode):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def golden_page(pid, markdown, html):
    exp = {}
    for m in chunk_ref.METHOD_NAMES:
        src = markdown if chunk_ref.INPUT_FIELD[m] == "markdown" else html
        if src is None:
            continue
        exp[m] = chunk_ref.chunk(m, markdown=markdown or "", html=html or "")
    out = {"type": "page", "id": pid, "markdown": markdown, "html": html, "expected": exp}
    if html is not None:
        out["container_kinds"] = [k for _, k in chunk_ref.chunk_with_kind(html)]
    return out


def iter_inputs(a):
    if a.cases:
        for fn in sorted(os.listdir(a.cases)):
            p = os.path.join(a.cases, fn)
            with open(p, encoding="utf-8", newline="") as f:  # newline="" 保留 \r\n
                text = f.read()
            if fn.endswith(".html"):
                yield "case:" + fn, None, text
            elif fn.endswith(".md"):
                yield "case:" + fn, text, None
    for i in range(a.fuzz_html):
        yield f"fuzz_html:{i}", None, gen_html(random.Random(1_000_000 + i))
    for i in range(a.fuzz_md):
        yield f"fuzz_md:{i}", gen_markdown(random.Random(2_000_000 + i)), None
    if a.input:
        with _open(a.input, "r") as f:
            for k, line in enumerate(f):
                if a.limit and k >= a.limit:
                    break
                if not line.strip():
                    continue
                r = json.loads(line)
                yield r.get("url") or f"input:{k}", r.get("markdown"), r.get("html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--fuzz-html", type=int, default=0)
    ap.add_argument("--fuzz-md", type=int, default=0)
    ap.add_argument("--input")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import bs4
    try:
        import lxml
        lxml_ver = lxml.__version__
    except ImportError:
        lxml_ver = None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    t0 = time.time()
    with _open(a.out, "w") as f:
        meta = {"type": "meta", "bs_parser": chunk_ref.BS_PARSER, "bs4": bs4.__version__, "lxml": lxml_ver,
                "methods": chunk_ref.METHOD_NAMES, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": {"SOFT": chunk_ref.SOFT, "HARD": chunk_ref.HARD, "MIN_LEN": chunk_ref.MIN_LEN,
                           "TABLE_SOFT": chunk_ref.TABLE_SOFT, "CT_HARD": chunk_ref.CT_HARD, "CT_SPLIT": chunk_ref.CT_SPLIT}}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for pid, md, html in iter_inputs(a):
            f.write(json.dumps(golden_page(pid, md, html), ensure_ascii=False) + "\n")
            n += 1
    print(f"写入 {n} 页 → {a.out}（{time.time() - t0:.1f}s，解析器 {chunk_ref.BS_PARSER}）")
    if chunk_ref.BS_PARSER != "lxml":
        print("警告：没有安装 lxml，用的是 html.parser，结果与离线评测、Java 版都可能不一致")


if __name__ == "__main__":
    main()
