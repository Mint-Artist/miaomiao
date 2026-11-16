# %% [markdown]
# # 探索 md_struct 的软上限（以 content_br 为基线；硬上限默认固定 512）
#
# 接在 recall_pipeline.py 第 1–7 段定义之后运行；要统计 token 数的话，再先运行第 10 段（加载 encoder，复用它的分词器）。
# 只读，不写任何文件；原有方法、切分结果、向量都不受影响。
#
# 两个参数各管什么（见 chunk_struct）：
#   软上限 soft：同一节内同类单元（段落 / 列表项 / 同一张表的行）贪心打包，加上下一个单元会超过 soft 就另起一块。
#               主要决定块数和平均块长。
#   硬上限 hard：单个单元（一段 / 一行 / 一个代码块）不超过 hard 就整段保留，超过才按句切到 soft 以内。
#               主要决定长段落、长表格行会不会被拆开，以及最长的块有多长（截断风险）。
#   注意：块开头挂的标题、表头不受这两个值约束，所以块长可能超过 hard。
#
# 步骤：
#   运行 1：抽样网页，对每档软上限（硬上限固定 512，也可传入多个硬上限一起比较）切一遍 md_struct，与 content_br 比较
#          总块数、块长分布、短块占比、超过模型长度（MAX_LENGTH token）的比例和截断丢掉的 token 占比，
#          并插值出"总块数与 content_br 相同"的软上限。
#   关于超过 512：硬上限只管单个单元，块开头挂的标题、表头不计入，所以会有少量块超过 512 字 / 512 token，
#          向量模型只截掉尾部。看"超过 512 token 的块占比"和"截断丢掉的 token 占比"，两者都很小（例如 ≤1–2%）就可以接受。
#   运行 2（可选）：看一个网页在某个组合下与 content_br 的实际切分结果。
#   运行 3：定好组合后注册新方法 md_struct_s{soft}_h{hard}，按 content_br 的文档集合切分（之后推理向量、评测）。

# %% 定义
import math
import os
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _clen(c):
    return len(c.replace("\n", ""))


def _sample_records(n_docs, parts=None, n_parts=40, seed=0):
    """从随机挑的 n_parts 个分片里各取前若干条，凑够 n_docs 条；只保留 content、markdown 两个字段以省内存。"""
    parts = list(parts or list_parts(DATA_GLOB))
    random.Random(seed).shuffle(parts)
    parts = parts[: max(1, min(n_parts, len(parts)))]
    per = math.ceil(n_docs / len(parts))
    out = []
    for p in tqdm(parts, desc="抽样网页"):
        for i, rec in enumerate(iter_records(p)):
            if i >= per:
                break
            out.append({"url": rec.get("url"), "content": rec.get("content") or "", "markdown": rec.get("markdown") or ""})
    return out[:n_docs]


def _load_tokenizer(tokenizer="auto"):
    """优先复用第 10 段的 encoder 的分词器；没有就按 MODEL_PATH 加载；都不行返回 None（只统计字数）。"""
    if tokenizer is None or tokenizer is False:
        return None
    if tokenizer != "auto":
        return tokenizer
    enc = globals().get("encoder")
    if enc is not None:
        if getattr(enc, "tok", None) is not None:
            return enc.tok
        if getattr(getattr(enc, "model", None), "tokenizer", None) is not None:
            return enc.model.tokenizer
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    except Exception as e:  # noqa: BLE001
        print("没有加载到分词器，只统计字数：", repr(e)[:200])
        return None


def _token_lens(tok, texts, batch=1000):
    out = []
    for s in range(0, len(texts), batch):
        enc = tok([DOC_PREFIX + t for t in texts[s:s + batch]], add_special_tokens=True, truncation=False)
        out += [len(x) for x in enc["input_ids"]]
    return out


def explore_limits(softs=(256, 300, 350, 400, 450, 500), hards=(512,),
                   n_docs=20000, n_parts=40, seed=0, tokenizer="auto", tok_docs=1000, short=64):
    """对 soft × hard 的每种组合切 md_struct，与 content_br 对比。返回 (明细 DataFrame, 抽样记录)。
    tok_docs：只在前这么多个网页的块上统计 token 数（分词比切分慢）；short：短块的字数阈值。"""
    global _EXPLORE_RECS
    recs = _sample_records(n_docs, n_parts=n_parts, seed=seed)
    _EXPLORE_RECS = recs
    cfgs = [(s, h) for h in hards for s in softs if h >= s]
    BASE = ("content_br", "—", "—")
    pages, texts = {}, {}
    for i, rec in enumerate(tqdm(recs, desc="切分")):
        res = {BASE: split(rec["content"])}
        atoms = parse_markdown(rec["markdown"])
        for s, h in cfgs:
            res[("md_struct", s, h)] = chunk_struct(atoms, s, h)
        for k, cs in res.items():
            pages.setdefault(k, []).append([_clen(c) for c in cs])
            if i < tok_docs:
                texts.setdefault(k, []).extend(cs)

    tok = _load_tokenizer(tokenizer)
    max_tok = globals().get("MAX_LENGTH", 512)
    md0 = pages[("md_struct",) + cfgs[0]]
    common = [bool(a) and bool(b) for a, b in zip(pages[BASE], md0)]
    base_total = sum(len(p) for p in pages[BASE])
    base_common = sum(len(p) for p, c in zip(pages[BASE], common) if c)

    rows = []
    for k, ps in pages.items():
        flat = np.array([x for p in ps for x in p] or [0])
        n_chunk = sum(len(p) for p in ps)
        n_common = sum(len(p) for p, c in zip(ps, common) if c)
        hard = k[2] if k[2] != "—" else SINGLE_CHUNK_MAX_LENGTH
        r = {"方法": k[0], "软上限": k[1], "硬上限": k[2],
             "总块数": n_chunk, "块数比(全部网页)": round(n_chunk / max(base_total, 1), 3),
             "块数比(两者都有块)": round(n_common / max(base_common, 1), 3),
             "有块网页": sum(bool(p) for p in ps),
             "每页总字数": round(sum(sum(p) for p in ps) / len(ps), 0),
             "平均块长": round(float(flat.mean()), 1), "块长中位数": float(np.median(flat)),
             "块长90分位": float(np.percentile(flat, 90)), "块长99分位": float(np.percentile(flat, 99)),
             f"短块(<{short}字)": f"{100 * (flat < short).mean():.1f}%",
             "超过硬上限": f"{100 * (flat > hard).mean():.1f}%",
             "超过512字": f"{100 * (flat > 512).mean():.1f}%"}
        if tok is not None and texts.get(k):
            tl = np.array(_token_lens(tok, texts[k]))
            r["token90分位"] = float(np.percentile(tl, 90))
            r[f"超过{max_tok}token(会截断)"] = f"{100 * (tl > max_tok).mean():.1f}%"
            r["截断丢掉的token占比"] = f"{100 * np.clip(tl - max_tok, 0, None).sum() / max(tl.sum(), 1):.2f}%"
            over = tl[tl > max_tok] - max_tok
            r["被截断块平均丢掉token"] = round(float(over.mean()), 1) if len(over) else 0
        rows.append(r)
    df = pd.DataFrame(rows)

    base = df[df["方法"] == "content_br"].iloc[0]
    print(f"抽样网页 {len(recs)} 个（{n_parts} 个分片）；content_br 与 md_struct 都有块的网页 {sum(common)} 个")
    print(f"content_br：总块数 {base['总块数']}，平均块长 {base['平均块长']}，块长90分位 {base['块长90分位']}，"
          f"短块 {base[f'短块(<{short}字)']}" + (f"，超过{max_tok}token {base[f'超过{max_tok}token(会截断)']}" if tok else ""))
    md = df[df["方法"] == "md_struct"].copy()
    md["软上限"] = md["软上限"].astype(int)
    md["硬上限"] = md["硬上限"].astype(int)
    show = [("块数比(全部网页)", "md_struct 总块数 / content_br 总块数（=1 即同一量级；索引规模看这个）"),
            ("块数比(两者都有块)", "只算两种方法都切得出块的网页"),
            ("平均块长", "平均块长（字，content_br 见上）"),
            ("块长90分位", "块长 90 分位（字）"),
            (f"短块(<{short}字)", f"短块（<{short}字）占比：结构切分块多，主要来自标题、表格、列表边界上切出的短块")]
    show.append(("超过512字", "超过 512 字的块占比"))
    if tok is not None:
        show += [(f"超过{max_tok}token(会截断)", f"超过模型长度 {max_tok} token、尾部会被截掉的块占比（前 {tok_docs} 个网页）"),
                 ("截断丢掉的token占比", "截断丢掉的 token 占全部 token 的比例"),
                 ("被截断块平均丢掉token", "被截断的块平均丢掉多少 token")]
    print(f"\n===== content_br 对照：超过512字 {base['超过512字']}" +
          (f"，超过{max_tok}token {base[f'超过{max_tok}token(会截断)']}，截断丢掉的token占比 {base['截断丢掉的token占比']}" if tok else "") + " =====")
    for col, title in show:
        print(f"\n===== {title} =====（行：软上限；列：硬上限）")
        print(md.pivot(index="软上限", columns="硬上限", values=col).to_string())

    print("\n===== 与 content_br 总块数相同的软上限（线性插值）=====")
    for scope in ("块数比(全部网页)", "块数比(两者都有块)"):
        for h in sorted(md["硬上限"].unique()):
            sub = md[md["硬上限"] == h].sort_values("软上限")
            xs, ys = sub["软上限"].tolist(), sub[scope].tolist()
            hit = None
            for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
                if (y0 - 1) * (y1 - 1) <= 0 and y0 != y1:
                    hit = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
                    break
            msg = f"约 {hit:.0f}" if hit is not None else f"扫描范围内对不齐（块数比 {ys[0]} → {ys[-1]}），需要扩大软上限范围"
            print(f"  {scope}，硬上限 {h}：{msg}")
    return df


def show_example(soft, hard, i=None, seed=0, width=120):
    """打印一个抽样网页在 content_br 与 md_struct(soft, hard) 下的切分结果，人工核对。"""
    recs = _EXPLORE_RECS
    if i is None:
        cand = [j for j, r in enumerate(recs) if r["markdown"] and r["content"]]
        i = random.Random(seed).choice(cand)
    rec = recs[i]
    print("网页：", rec["url"], f"（第 {i} 个）")
    for name, cs in (("content_br", split(rec["content"])),
                     (f"md_struct soft={soft} hard={hard}", chunk_struct(parse_markdown(rec["markdown"]), soft, hard))):
        print(f"\n--- {name}：{len(cs)} 块 ---")
        for c in cs:
            print(f"[{_clen(c)}字] " + c.replace("\n", " / ")[:width])


def register_md_limits(soft, hard=None):
    """注册新方法：规则与 md_struct 相同，只改软上限（及硬上限）。硬上限为默认 HARD 时名为 md_struct_s{soft}，
    否则为 md_struct_s{soft}_h{hard}。返回方法名。"""
    hard = hard or HARD
    name = f"md_struct_s{soft}" if hard == HARD else f"md_struct_s{soft}_h{hard}"
    METHODS[name] = lambda d, s=soft, h=hard: chunk_struct(d.md_atoms, s, h)
    print("已注册：", name)
    return name


def align_drop(parts, ref: str = "content_br"):
    """让新方法切出的文档集合、顺序与 ref 方法完全一致（与 chunk_merge.py 中的同名函数相同）：
    按 ref 的 *.urls.txt 逐条对齐原始记录，ref 当时跳过的记录记入 DROP，chunk_part 会同样跳过。"""
    global DROP
    drop, n_skip, missing = {}, 0, []
    for path in tqdm(parts, desc=f"按 {ref} 对齐文档"):
        name = part_name(path)
        f = os.path.join(OUT_ROOT, ref, "chunks", name + ".urls.txt")
        if not os.path.exists(f[: -len(".urls.txt")] + ".done"):
            missing.append(name)
            continue
        with open(f, encoding="utf-8") as fu:
            want = fu.read().split("\n")
        want = want[:-1] if want and want[-1] == "" else want
        p = 0
        for i, rec in enumerate(iter_records(path)):
            url = (rec.get("url") or "").replace("\n", " ")
            if p < len(want) and url == want[p]:
                p += 1
            else:
                drop.setdefault(name, set()).add(i)
                n_skip += 1
        if p != len(want):
            raise ValueError(f"{name}：{ref} 有 {len(want)} 条文档，按原始数据只对齐到 {p} 条，数据可能已变化")
    if missing:
        raise ValueError(f"{len(missing)} 个分片在 {ref} 里还没切完（如 {missing[:3]}），先确认 DATA_GLOB / PART_PREFIX 与当时一致")
    DROP = drop
    print(f"已按 {ref} 对齐 {len(parts)} 个分片，跳过 {n_skip} 条（{ref} 当时去重掉的记录）")


# %% 运行 1：扫描（第 1 段的 DATA_GLOB 指向随机库；只用 CPU，不写文件）
LIMIT_SCAN = explore_limits(softs=(256, 300, 350, 400, 450, 500), hards=(512,), n_docs=20000)
# 完整明细：LIMIT_SCAN.to_string() 或 LIMIT_SCAN.to_csv("md_limit_scan.csv", index=False)

# %% 运行 2（可选）：看一个网页的实际切分（i=None 时随机挑一个）
show_example(soft=400, hard=HARD)

# %% 运行 3：定好组合后切分（随机库、inj-、ggl- 三批各运行一次。每批先改第 1 段的 DATA_GLOB / PART_PREFIX，
#            重新运行第 1、2 段，再运行这里；需要第 9 段的定义）
NEW_SOFT, NEW_HARD = 400, HARD  # NEW_SOFT 改成运行 1 选定的值；硬上限保持 512
NEW_METHOD = register_md_limits(NEW_SOFT, NEW_HARD)
RUN_METHODS = [NEW_METHOD]
DEDUP_URLS = False  # 用 align_drop 对齐，不再按 url 重新去重
_parts = list_parts(DATA_GLOB)
align_drop(_parts, ref="content_br")
run_chunking(_parts)
# 三批都切完后推理向量：bash run_embed_job_multigpu.sh md_struct_s{NEW_SOFT}
# 评测前把 RUN_METHODS 改回全部方法，例如：
# RUN_METHODS = ["content_br", "md_struct", "html_struct", NEW_METHOD]
