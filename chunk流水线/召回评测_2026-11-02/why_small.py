# %% [markdown]
# # 诊断：为什么结构切分会切出一串相邻的小块（只读，不写任何文件）
#
# 接在 recall_pipeline.py 第 1–7 段定义之后运行。
# chunk_struct 只在"同一节内、同一类单元（正文段落 / 列表项 / 同一张表的行）"里把小单元拼到软上限；
# 遇到标题（含"整段加粗的短行"这类伪标题）、或单元类型切换（段落 ↔ 列表 ↔ 表格，以及换了一张表）都会强制断开。
# 本脚本逐个列出网页的结构单元，标出每一处强制断开的原因，并统计各类原因的次数和最终块长。

# %% 定义
from collections import Counter


def why_small(rec, which="html", n=80):
    """rec：一条原始记录；which："html" 或 "md"；n：最多打印多少个结构单元。"""
    atoms = parse_html(rec.get("html") or "") if which == "html" else parse_markdown(rec.get("markdown") or "")
    prev, cnt = None, Counter()
    for k, a in enumerate(atoms):
        fam = ("row", a.table_id) if a.kind == "row" else ("li",) if a.kind == "li" else ("text",)
        if a.kind == "heading":
            why = "伪标题（加粗短行）→ 断开" if a.level == 7 else f"h{a.level} 标题 → 断开"
            cnt["伪标题" if a.level == 7 else "标题"] += 1
            prev = None
        elif prev is not None and fam != prev:
            why = f"类型切换 {prev[0]}→{fam[0]} → 断开"
            cnt["类型切换"] += 1
            prev = fam
        else:
            why, prev = "", fam
        if k < n:
            print(f"{a.kind:7s} {len(a.text):4d}字  {a.text[:40]!r:44s} {why}")
    chunks = chunk_struct(atoms, SOFT, HARD)
    print(f"\n[{which}] 结构单元 {len(atoms)} 个；断开原因：{dict(cnt)}")
    print(f"切出 {len(chunks)} 块，块长：{[len(c.replace(chr(10), '')) for c in chunks]}")


def find_record(url, parts=None):
    """按顺序扫分片找到这条 url 的原始记录（可能要一会儿）；知道所在分片时传 parts=[分片路径] 会快很多。"""
    for p in parts or list_parts(DATA_GLOB):
        for r in iter_records(p):
            if r.get("url") == url:
                return r
    raise KeyError(f"没找到：{url}")


# %% 运行
URL = "这里填那个网页的 url"
rec = find_record(URL)
why_small(rec, "html")
why_small(rec, "md")  # 顺便对比 md 的切法
