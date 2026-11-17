# %% [markdown]
# # 导出召回结果给 recall_viewer.html 查看（v2：按 QUERY_FILES 每套标准答案各导出一份）
#
# 运行顺序：
#   1. recall_pipeline_v2.py 跑完第 14 段（结果在 RESULTS 里，不需要重新检索）
#   2. 运行 viewer_export.py 的"定义"段（只要 export_viewer_data 函数；不要运行它的"运行"段，
#      那段用全局 hits / queries，在 v2 里指向最后一套答案，且记录的答案来源会写错）
#   3. 运行本文件
# 不要运行 query_sample.py 的"定义"段：它的旧版 html_ok_urls 写死 html_struct，会覆盖 v2 第 13 段修好的版本。
#
# 产出：OUT_ROOT/viewer_{答案名}_top{TOPN}.jsonl.gz，用浏览器打开 recall_viewer.html → "选择文件" 选中即可。
# 大小参考：3000 多条 query × 3 种方法 × 前 20 个 chunk，压缩后约 30MB；先抽 500 条看通常就够。

# %% 运行：每套答案各导出一份
if "export_viewer_data" not in globals():
    raise NameError("请先运行 viewer_export.py 的“定义”段")

TOPN = 20         # 每种方法导出前几个 chunk
SAMPLE = 500      # 每套答案随机抽多少条 query；None = 全部
SEED = 0          # 抽样种子，固定后每次抽到同一批 query

_qf_saved = QUERY_FILE
try:
    for _name, (queries, summary, hits) in RESULTS.items():
        QUERY_FILE = QUERY_FILES[_name]  # 只用于在导出文件里记录来源
        export_viewer_data(hits, queries, os.path.join(OUT_ROOT, f"viewer_{_name}_top{TOPN}.jsonl.gz"),
                           topn=TOPN, sample=SAMPLE, seed=SEED)
finally:
    QUERY_FILE = _qf_saved

# %% 可选：只导出两个方法某项指标结果不同的 query（逐条看新方法赢 / 输在哪里）
DIFF_SET = "谷歌前10"                       # RESULTS 里的答案名
DIFF_METHOD = "md_struct_secmerge_t512"     # 要看的方法，和 BASELINE_METHOD 比
DIFF_KEY = "hit@3000字"                     # 比较哪项指标

queries, summary, hits = RESULTS[DIFF_SET]
_m, _b = DIFF_METHOD, BASELINE_METHOD
_win, _lose = [], []
for i, q in enumerate(queries):
    if q["rel"]:
        a = url_metrics(hits[_m][0], hits[_m][2][i], q["rel"])[DIFF_KEY]
        c = url_metrics(hits[_b][0], hits[_b][2][i], q["rel"])[DIFF_KEY]
        if a > c:
            _win.append(q["qid"])
        elif a < c:
            _lose.append(q["qid"])
print(f"[{DIFF_SET}] {DIFF_KEY}：{_m} 赢 {len(_win)} 条，输 {len(_lose)} 条（相对 {_b}）")

_qf_saved = QUERY_FILE
try:
    QUERY_FILE = QUERY_FILES[DIFF_SET]
    for _tag, _qids in [("赢", _win), ("输", _lose)]:
        export_viewer_data(hits, queries, os.path.join(OUT_ROOT, f"viewer_{DIFF_SET}_{_m}_{_tag}.jsonl.gz"),
                           topn=TOPN, qids=_qids)
finally:
    QUERY_FILE = _qf_saved
