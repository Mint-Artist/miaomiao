# %% [markdown]
# # 检查谷歌数据有没有给错（只读，不改任何文件）
#
# 在 notebook 里随时可以运行：需要先运行 recall_pipeline.py 第 1–2 段（用到 OUT_ROOT、RUN_METHODS、norm_url、iter_records），
# 且第 1 段的 QUERY_FILE 指向 google_gt.py 生成的 queries_google_top10.jsonl。
#
# 四项检查及判断标准：
#   【1】qid 对应：谷歌前 10 与原精排答案（rerank_urls）的重合个数分布。相当一部分 query 有 ≥1 个重合为正常；几乎全为 0 说明 query 与 qid 对错了。
#   【2】网页正文：content 字数分位数、疑似拦截页、html 为空的数量。字数中位数几百字以上、拦截页很少为正常。
#   【3】完成情况：各方法 ggl- 分片的切分完成数、向量完成数。应相同且等于分片数，否则有方法没切完或没推理完。
#   【4】抽几条 query，打印谷歌前 5 的网址和正文开头，人工看是否对得上；"网页文件里没有"很多说明谷歌网页抓取覆盖不全。

# %% 定义
import glob
import json
import os
import random
import re
from collections import Counter


def data_sanity_check(query_path, ggl_glob, methods=None, n_show=3, seed=0):
    methods = methods or RUN_METHODS
    G = [json.loads(l) for l in open(query_path, encoding="utf-8") if l.strip()]
    ov = [len({norm_url(u) for u in d["relevant_urls"]} & {norm_url(u) for u in d.get("rerank_urls", [])})
          for d in G if d.get("rerank_urls")]
    print("【1】与原精排答案的重合个数分布：", sorted(Counter(ov).items()), " 平均", round(sum(ov) / max(len(ov), 1), 2))

    parts = sorted(glob.glob(ggl_glob))
    recs = [r for p in parts for r in iter_records(p)]
    lens = sorted(len(r.get("content") or "") for r in recs)
    junk = re.compile("验证码|安全验证|访问受限|Access Denied|captcha|403 Forbidden|开启JavaScript|正在检查|请稍候")
    q = [lens[min(int(len(lens) * x), len(lens) - 1)] for x in (0.1, 0.5, 0.9)] if lens else []
    print(f"【2】谷歌网页 {len(recs)} 个（分片 {len(parts)} 个）；content 字数 10%/50%/90% 分位：{q}；"
          f"疑似拦截页 {sum(bool(junk.search((r.get('content') or '')[:300])) for r in recs)}；"
          f"html 为空 {sum(not (r.get('html') or '').strip() for r in recs)}")

    for m in methods:
        print(f"【3】{m:18s} ggl 分片：切分完成 {len(glob.glob(os.path.join(OUT_ROOT, m, 'chunks', 'ggl-*.done')))} 个，"
              f"向量完成 {len(glob.glob(os.path.join(OUT_ROOT, m, 'emb', 'ggl-*.done')))} 个（分片共 {len(parts)} 个）")

    page = {norm_url(r.get("url") or ""): r for r in recs}
    for d in random.Random(seed).sample(G, min(n_show, len(G))):
        print("\n【4】query：", d["query"])
        for u in d["relevant_urls"][:5]:
            r = page.get(norm_url(u))
            txt = (r.get("content") or "")[:60].replace("<br>", " ") if r else "（网页文件里没有）"
            print("   ", u[:70], "|", txt)


# %% 运行（路径按实际修改：谷歌网页分片目录，即 google_gt.py 里的 GOOGLE_PAGES_DIR）
data_sanity_check(QUERY_FILE, "/path/to/google_pages_parts/part-*")
