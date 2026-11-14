# %% [markdown]
# # 新切分方法：短块并入相邻块（md_struct_merge / html_struct_merge）
#
# 背景：md_struct / html_struct 打包后，去掉换行不足 20 字（MIN_LEN）的块会被直接丢弃。
# 这会丢掉两类内容：
#   1. 很短的整节，例如"## 电话 / 12345"（7 字），整节消失；
#   2. 一节的尾巴，例如前一块已接近 256 字，最后一个 15 字的短段被挤成单独一块后丢弃。
# 新方法的切分规则与 md_struct / html_struct 完全相同，只把"丢弃"改为"并入相邻块"：
#   - 以标题开头的短块（一整节）：优先并入后一块（放在后一块开头），没有后一块时并入前一块；
#   - 不以标题开头的短块（一节的尾巴或类别切换处的碎片）：优先并入前一块（同一节），没有前一块时并入后一块；
#   - 若优先方向合并后超过硬上限 512 字、另一方向不超过，则改并另一方向；
#   - 合并后仍不足 20 字的继续合并；整篇文档合起来都不足 20 字时，与原规则一致，丢弃。
# 原有的 chunk_struct、md_struct、html_struct 不做任何修改。
#
# 用法（notebook）：
#   1. 运行 recall_pipeline.py 第 1–7 段之后，运行本文件的"定义"段（注册两个新方法）。
#   2. 可选：运行"统计"段，抽样看有多少网页受影响、找回多少字，再决定是否全量推理。
#   3. 运行"切分"段：只切两个新方法；文档集合、顺序与 content_br 完全对齐（随机库、inj- 补入分片都适用）。
#   4. 推理向量：bash run_embed_job_multigpu.sh md_struct_merge html_struct_merge
#   5. 评测：RUN_METHODS 加上两个新方法，重新运行第 13–14 段；诊断 7 / query_sample.py 的 html 口径照常适用。

# %% 定义（在第 7 段之后、第 9 段之前运行）
def chunk_struct_merge(atoms, soft: int = SOFT, hard: int = HARD, min_len: int = MIN_LEN):
    """与 chunk_struct（with_context=False）相同的打包规则；不足 min_len 字的块并入相邻块而不是丢弃。"""
    recs = []  # [块文本, 是否以标题开头]
    cur, cur_family, cur_head, pending = [], None, False, []

    def flush():
        nonlocal cur, cur_family, cur_head
        if cur:
            recs.append(["\n".join(cur), cur_head])
        cur, cur_family, cur_head = [], None, False

    for a in atoms:
        if a.kind == "heading":
            flush()
            pending.append(a.text)
            continue
        family = ("row", a.table_id) if a.kind == "row" else ("li",) if a.kind == "li" else ("text",)
        pieces = [a.text] if len(a.text) <= hard else split_long(a.text, soft)
        for piece in pieces:
            size = sum(len(x) for x in cur) + len(cur)
            if cur and (family != cur_family or size + len(piece) > soft):
                flush()
            add = piece
            if not cur:
                if a.kind == "row" and a.header:  # 表格的每一块都带表头
                    add = a.header + "\n" + add
                if pending:  # 标题与紧随的第一段放在同一块
                    add = "\n".join(pending) + "\n" + add
                    cur_head = True
                cur_family = family
                pending = []
            cur.append(add)
    if pending:  # 页面以标题结尾
        cur.append("\n".join(pending))
        cur_head = True
    flush()

    n = lambda s: len(s.replace("\n", ""))
    while len(recs) > 1:
        k = next((i for i, (b, _) in enumerate(recs) if n(b) < min_len), None)
        if k is None:
            break
        prev = k - 1 if k > 0 else None
        nxt = k + 1 if k + 1 < len(recs) else None
        order = [j for j in ((nxt, prev) if recs[k][1] else (prev, nxt)) if j is not None]
        fits = [j for j in order if n(recs[j][0]) + n(recs[k][0]) <= hard]
        j = (fits or order)[0]
        if j < k:
            recs[j][0] = recs[j][0] + "\n" + recs[k][0]
        else:
            recs[j][0] = recs[k][0] + "\n" + recs[j][0]
            recs[j][1] = recs[k][1] or recs[j][1]
        del recs[k]
    return [b for b, _ in recs if n(b) >= min_len]


METHODS["md_struct_merge"] = lambda d: chunk_struct_merge(d.md_atoms, SOFT, HARD)
METHODS["html_struct_merge"] = lambda d: chunk_struct_merge(d.html_atoms, SOFT, HARD)


def align_drop(parts, ref: str = "content_br"):
    """让新方法切出的文档集合、顺序与 ref 方法完全一致：按 ref 的 *.urls.txt 逐条对齐原始记录，
    ref 当时跳过的记录（补入时按 url 去重掉的）记入 DROP，chunk_part 会同样跳过。
    在 run_chunking 之前调用，并把 DEDUP_URLS 设为 False（否则 run_chunking 会用 prepare_dedup 覆盖 DROP）。"""
    global DROP
    drop, n_skip, missing = {}, 0, []
    for path in tqdm(parts, desc=f"按 {ref} 对齐文档"):
        name = part_name(path)
        f = os.path.join(OUT_ROOT, ref, "chunks", name + ".urls.txt")
        if not os.path.exists(f[: -len(".urls.txt")] + ".done"):
            missing.append(name)
            continue
        with open(f, encoding="utf-8") as fu:
            want = [u for u in fu.read().split("\n")]
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


# %% 统计（可选）：抽样看影响有多大，不写任何文件
def merge_effect(n_docs: int = 20000, parts=None):
    """在前若干条记录上对比旧方法（丢弃短块）与新方法（并入相邻块）。"""
    parts = parts or list_parts()
    res = {k: {"docs": 0, "受影响网页": 0, "旧块数": 0, "新块数": 0, "找回字数": 0, "旧总字数": 0} for k in ("md", "html")}
    examples, seen = [], 0
    for path in parts:
        for rec in iter_records(path):
            d = Doc(rec)
            for k, atoms in (("md", d.md_atoms), ("html", d.html_atoms)):
                old = chunk_struct(atoms, SOFT, HARD)
                new = chunk_struct_merge(atoms, SOFT, HARD)
                r = res[k]
                r["docs"] += 1
                r["旧块数"] += len(old)
                r["新块数"] += len(new)
                lo = sum(len(c.replace("\n", "")) for c in old)
                ln = sum(len(c.replace("\n", "")) for c in new)
                r["旧总字数"] += lo
                r["找回字数"] += ln - lo
                if ln > lo:
                    r["受影响网页"] += 1
                    if k == "md" and len(examples) < 3:
                        examples.append((rec.get("url"), [c for c in new if c not in old][:2]))
            seen += 1
            if seen >= n_docs:
                break
        if seen >= n_docs:
            break
    for k, r in res.items():
        print(f"{k}_struct：网页 {r['docs']}，受影响 {r['受影响网页']}（{100 * r['受影响网页'] / max(r['docs'], 1):.1f}%），"
              f"块数 {r['旧块数']} → {r['新块数']}，找回 {r['找回字数']} 字（占原总字数 {100 * r['找回字数'] / max(r['旧总字数'], 1):.2f}%）")
    for u, cs in examples:
        print("例：", u)
        for c in cs:
            print("   ", c.replace("\n", " / ")[:150])


merge_effect(20000)

# %% 切分：只切两个新方法（与 content_br 的文档完全对齐；随机库和 inj- 补入分片各跑一次，DATA_GLOB / PART_PREFIX 与当时一致）
RUN_METHODS = ["md_struct_merge", "html_struct_merge"]
DEDUP_URLS = False  # 用 align_drop 对齐，不再按 url 重新去重
align_drop(list_parts(), ref="content_br")
run_chunking()
# 切完后把 RUN_METHODS 改回包含全部方法，再推理向量、评测，例如：
# RUN_METHODS = ["content_br", "md_struct", "html_struct", "md_struct_merge", "html_struct_merge"]
