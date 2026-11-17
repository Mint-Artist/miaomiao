# %% [markdown]
# # 评测前检查：切分和向量是否齐全、各方法的向量配置是否一致（只读，不写任何文件）
#
# 接在 recall_pipeline_v2.py 第 1–2 段之后运行（需要 OUT_ROOT、RUN_METHODS、BASELINE_METHOD 和模型配置）。
# 检查项：
#   1. 每个方法的 chunks/*.done 与 emb/*.done 分片名完全一致（不只是个数相等）
#   2. 各方法的分片名集合与基线一致（三批数据都切完、推理完）
#   3. 每个分片 emb/*.npy 的行数 = chunks/*.done 里记录的 chunk 数（防止推理中断留下残缺文件）
#   4. emb/meta.json 在各方法之间一致，且与第 1 段的模型配置一致（否则新旧方法的向量不可比）
#   5. 抽查若干分片的向量里有没有 NaN / inf、范数是否接近 1（fp16 溢出或没归一化时会出问题）
# 全部通过时最后打印"可以开始评测"。

# %% 定义
import glob
import json
import os

import numpy as np

META_KEYS = ("model_path", "loader", "pooling", "max_length", "doc_prefix", "query_prefix", "dim")


def _names(d, suffix=".done"):
    return {os.path.basename(p)[: -len(suffix)] for p in glob.glob(os.path.join(d, "*" + suffix))}


def check_emb_ready(methods=None, baseline=None, sample_shards=3, sample_rows=2000):
    methods = list(methods or RUN_METHODS)
    baseline = baseline or BASELINE_METHOD
    problems = []
    ref_names = _names(os.path.join(OUT_ROOT, baseline, "chunks"))
    metas = {}
    for m in methods:
        cdir, edir = os.path.join(OUT_ROOT, m, "chunks"), os.path.join(OUT_ROOT, m, "emb")
        cn, en = _names(cdir), _names(edir)
        n_chunks = n_rows = 0
        bad_rows = []
        for n in sorted(cn & en):
            want = json.load(open(os.path.join(cdir, n + ".done")))["chunks"]
            p = os.path.join(edir, n + ".npy")
            got = np.load(p, mmap_mode="r").shape[0] if os.path.exists(p) else -1
            n_chunks, n_rows = n_chunks + want, n_rows + max(got, 0)
            if got != want:
                bad_rows.append((n, want, got))
        print(f"{m:28s} 切分 {len(cn):5d} 片  向量 {len(en):5d} 片  chunk {n_chunks:>12,d}  向量行 {n_rows:>12,d}")
        if cn - en:
            problems.append(f"[{m}] 有 {len(cn - en)} 个分片切了但没推理，例如 {sorted(cn - en)[:3]}")
        if en - cn:
            problems.append(f"[{m}] 有 {len(en - cn)} 个分片有向量但没有切分结果，例如 {sorted(en - cn)[:3]}")
        if m != baseline and cn != ref_names:
            problems.append(f"[{m}] 分片集合与基线 {baseline} 不同：少 {sorted(ref_names - cn)[:3]}，多 {sorted(cn - ref_names)[:3]}")
        if bad_rows:
            problems.append(f"[{m}] {len(bad_rows)} 个分片向量行数 ≠ chunk 数（分片, 应有, 实有）：{bad_rows[:3]}")

        mp = os.path.join(edir, "meta.json")
        metas[m] = json.load(open(mp)) if os.path.exists(mp) else None
        if metas[m] is None:
            problems.append(f"[{m}] 没有 emb/meta.json，无法确认它用的模型配置")

        # 抽查向量数值
        for n in sorted(en)[:: max(1, len(en) // sample_shards)][:sample_shards]:
            p = os.path.join(edir, n + ".npy")
            if not os.path.exists(p):
                continue
            v = np.asarray(np.load(p, mmap_mode="r")[:sample_rows], dtype=np.float32)
            if len(v) == 0:
                continue
            if not np.isfinite(v).all():
                problems.append(f"[{m}] 分片 {n} 里有 NaN/inf")
            norm = np.linalg.norm(v, axis=1)
            if abs(float(norm.mean()) - 1) > 0.01:
                problems.append(f"[{m}] 分片 {n} 向量平均范数 {norm.mean():.3f}，不像归一化过的")

    # meta 一致性
    print()
    for m in methods:
        print(f"{m:28s}", {k: (metas[m] or {}).get(k) for k in META_KEYS})
    known = {m: mt for m, mt in metas.items() if mt}
    for k in META_KEYS:
        vals = {m: mt.get(k) for m, mt in known.items()}
        if len(set(map(str, vals.values()))) > 1:
            problems.append(f"meta.json 的 {k} 在各方法间不一致：{vals}")
    cfg = {"model_path": MODEL_PATH, "loader": LOADER, "pooling": POOLING, "max_length": MAX_LENGTH,
           "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX}
    for m, mt in known.items():
        diff = {k: (mt.get(k), v) for k, v in cfg.items() if k in mt and mt.get(k) != v}
        if diff:
            problems.append(f"[{m}] meta.json 与第 1 段配置不一致（meta, 第 1 段）：{diff}")

    print()
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
    else:
        print("全部通过，可以开始评测（第 10、12、13、14 段）")
    return problems


# %% 运行
problems = check_emb_ready()
