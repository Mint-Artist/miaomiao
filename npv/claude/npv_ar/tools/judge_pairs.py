#!/usr/bin/env python3
"""用大模型判官给"同一 query 下的头尾网页对"打权威度偏序标签，产出 harness 需要的 pairs.tsv。

纯标准库，内网可用。prompt 来自 claude/权威度偏序标注prompt设计.md。

流程（四个子命令，可分开跑，judge 可断点续跑）：

  1. sample   从 pages.jsonl 里每个 query 抽头尾对（默认 top1-10 抽一页、top11-20 抽一页），随机决定谁是 A
  2. judge    每对判 4 次：带域名 × (AB, BA)、遮域名 × (AB, BA)。原始 JSON 追加写入 judgments.jsonl
  3. build    顺序一致性过滤 + 相关性门槛 → pairs_with_host.tsv、pairs_blind.tsv（五列：url_a url_b pref group qid）+ stats.json
  4. split    按 qid 切成 train / dev / holdout，同一 qid 的两轮标签落在同一份里
  （grade     对单页打绝对档位，产出 grades.tsv；prompt  打印某一对的完整 prompt 供检查）

pages.jsonl 每行一个结果页：
  {"qid": "q0001", "query": "...", "rank": 3, "url": "https://...", "title": "...", "text": "正文纯文本",
   "date": "2024-05-01"(可无), "author": "..."(可无), "vertical": "医疗"(可无)}

典型用法：
  python judge_pairs.py sample --pages pages.jsonl --out work/pairs_todo.jsonl --per-query 1
  python judge_pairs.py prompt --todo work/pairs_todo.jsonl --index 0 --blind
  python judge_pairs.py judge  --todo work/pairs_todo.jsonl --out work/judgments.jsonl \\
         --base http://内网网关/v1 --key $KEY --model 判官模型 --workers 8 --limit 300   # 先小批量
  python judge_pairs.py build  --judgments work/judgments.jsonl --todo work/pairs_todo.jsonl --out work/labels
  python judge_pairs.py split  --labels work/labels --out-train harness/labels/train --out-dev harness/labels/dev \\
         --out-holdout ../npv_ar_private/holdout
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT_VERSION = "pair-v1"

SYSTEM_PAIR = """你是搜索质量评估员，任务是比较两个网页在给定查询主题下的权威度。

权威度指：如果两个网页都在回答这个查询，你更信任哪一个网页的信息来源。它由三方面构成：
1. 来源可信度：发布者是谁（政府、机构、企业官方、专业媒体、专家个人、普通用户、内容农场），是否有明确的署名、机构落款、联系方式、资质说明。
2. 内容专业性：是否展现出领域知识，论述是否具体、准确、有依据（数据、引用、法规条文、一手经验），有无明显错误或夸大。
3. 内容质量：是否原创、完整、结构清楚；广告、导流、模板文本、堆砌关键词的比例；是否为列表页、聚合页、用户主页等非正文页。

不要考虑的因素：
- 哪个网页与查询更相关。只需先判断两页是否都基本切题；胜负只比权威度。
- 哪个网页更新。发布时间不影响权威度判断。
- 网页篇幅长短本身。长不等于权威。
- 网页在搜索结果中的位置或任何排名信息。

注意：大型知名站点上也有大量低质量的用户生成内容、聚合页、营销页；小型或不知名站点上也可能有机构官网、专业人士的权威内容。请根据页面本身的证据判断，不要仅凭站点名气。

允许判为"相当"：当两页权威度差别不明显时，选择 tie，不要强行分出高下。

只输出 JSON，不要输出其他文字。"""

USER_PAIR = """查询：{query}

【网页 A】
站点：{host_a}
URL 路径：{path_a}
标题：{title_a}
发布时间：{date_a}
作者/机构：{author_a}
正文（截取）：
{text_a}

【网页 B】
站点：{host_b}
URL 路径：{path_b}
标题：{title_b}
发布时间：{date_b}
作者/机构：{author_b}
正文（截取）：
{text_b}

请按以下 JSON 格式输出：
{{
  "relevance": {{"a": 0-2, "b": 0-2}},
  "source_credibility": {{"a": 1-5, "b": 1-5}},
  "expertise": {{"a": 1-5, "b": 1-5}},
  "content_quality": {{"a": 1-5, "b": 1-5}},
  "page_type": {{"a": "正文页|列表页|用户主页|论坛帖|问答|商品页|其他", "b": "..."}},
  "risk": {{"a": ["垃圾|营销|误导|无"], "b": ["..."]}},
  "verdict": "a|b|tie",
  "confidence": 1-5,
  "reason": "不超过 60 字"
}}
relevance：0 不切题，1 部分切题，2 切题。"""

USER_GRADE = """查询：{query}

【网页】
站点：{host}
URL 路径：{path}
标题：{title}
发布时间：{date}
作者/机构：{author}
正文（截取）：
{text}

请按以下 JSON 格式输出：
{{
  "relevance": 0-2,
  "source_credibility": 1-5,
  "expertise": 1-5,
  "content_quality": 1-5,
  "page_type": "正文页|列表页|用户主页|论坛帖|问答|商品页|其他",
  "risk": ["垃圾|营销|误导|无"],
  "authority_grade": 0-4,
  "reason": "不超过 60 字"
}}
authority_grade：0 垃圾或误导；1 低质或不可信来源；2 普通正规来源；3 专业或知名机构来源；4 官方或权威机构来源。"""

HEAD_CHARS, TAIL_CHARS = 1500, 300


# ---------------- 工具函数 ----------------
def host_of(url):
    return url.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0].lower()


def path_of(url):
    rest = url.split("://", 1)[-1]
    return "/" + rest.split("/", 1)[1] if "/" in rest else "/"


def clip_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= HEAD_CHARS + TAIL_CHARS + 20:
        return text or "（无正文）"
    return text[:HEAD_CHARS] + " …… " + text[-TAIL_CHARS:]


def site_tokens(host):
    """站点名的几种写法，遮域名时要从正文里抹掉：完整 host、去 www 的 host、注册域主标签。"""
    toks = {host}
    h = host[4:] if host.startswith("www.") else host
    toks.add(h)
    parts = h.split(".")
    if len(parts) >= 2:
        label = parts[-2] if parts[-2] not in ("com", "net", "org", "gov", "edu", "co") or len(parts) < 3 else parts[-3]
        if len(label) >= 3:
            toks.add(label)
    return sorted(toks, key=len, reverse=True)


def mask_site(text, hosts):
    for h in hosts:
        for tok in site_tokens(h):
            text = re.sub(re.escape(tok), "〔站点名〕", text, flags=re.I)
    return text


def page_block(p, blind, hosts):
    host = "（已隐藏）" if blind else host_of(p["url"])
    text = p.get("text", "") or ""
    title = p.get("title") or "（无标题）"
    if blind:  # 先遮再截，否则截断处会留下半个域名
        text, title = mask_site(text, hosts), mask_site(title, hosts)
    text = clip_text(text)
    return {"host": host, "path": path_of(p["url"]), "title": title,
            "date": p.get("date") or "未知", "author": p.get("author") or "未知", "text": text}


def build_pair_messages(query, pa, pb, blind):
    hosts = [host_of(pa["url"]), host_of(pb["url"])]
    a, b = page_block(pa, blind, hosts), page_block(pb, blind, hosts)
    user = USER_PAIR.format(query=query, **{k + "_a": v for k, v in a.items()}, **{k + "_b": v for k, v in b.items()})
    return [{"role": "system", "content": SYSTEM_PAIR}, {"role": "user", "content": user}]


def build_grade_messages(query, p):
    blk = page_block(p, False, [])
    return [{"role": "system", "content": SYSTEM_PAIR}, {"role": "user", "content": USER_GRADE.format(query=query, **blk)}]


def parse_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    return json.loads(text)


class Client:
    def __init__(self, base, key, model, timeout=120, json_mode=False, max_tokens=600):
        self.base, self.key, self.model, self.timeout, self.json_mode, self.max_tokens = base.rstrip("/"), key, model, timeout, json_mode, max_tokens

    def chat(self, messages, retries=3):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body, ensure_ascii=False).encode()
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(self.base + "/chat/completions", data=data, method="POST")
            req.add_header("Authorization", f"Bearer {self.key}")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read().decode("utf-8"))
                return d["choices"][0]["message"]["content"] or "", d.get("usage") or {}
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
                if e.code in (400, 401, 403, 404, 422):
                    break
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(last)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


# ---------------- 子命令 ----------------
def cmd_sample(a):
    by_q = defaultdict(list)
    for p in read_jsonl(a.pages):
        by_q[p["qid"]].append(p)
    rnd = random.Random(a.seed)
    todo, skipped = [], Counter()
    for qid in sorted(by_q):
        pages = [p for p in by_q[qid] if p.get("url")]
        if a.require_text:
            pages = [p for p in pages if (p.get("text") or "").strip()]
        heads = [p for p in pages if a.head_min <= int(p["rank"]) <= a.head_max]
        tails = [p for p in pages if a.tail_min <= int(p["rank"]) <= a.tail_max]
        if not heads or not tails:
            skipped["no_head_or_tail"] += 1
            continue
        rnd.shuffle(heads)
        rnd.shuffle(tails)
        used = set()
        for i in range(a.per_query):
            cand = [(h, t) for h in heads for t in tails if h["url"] not in used and t["url"] not in used and h["url"] != t["url"]]
            if not cand:
                break
            h, t = cand[0]
            used.update((h["url"], t["url"]))
            a_page, b_page = (h, t) if rnd.random() < 0.5 else (t, h)  # 随机决定谁是 A，避免位置与头尾绑定
            query = h.get("query") or t.get("query") or ""
            group = (h.get("vertical") or t.get("vertical") or "all")
            todo.append({"pair_id": f"{qid}#{i}", "qid": qid, "query": query, "group": group,
                         "a": {k: h.get(k) if a_page is h else t.get(k) for k in ("url", "title", "text", "date", "author", "rank")},
                         "b": {k: t.get(k) if a_page is h else h.get(k) for k in ("url", "title", "text", "date", "author", "rank")}})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for t in todo:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"抽出 {len(todo)} 对，来自 {len({t['qid'] for t in todo})} 个 query；跳过 {dict(skipped)} → {a.out}")


def cmd_prompt(a):
    todo = list(read_jsonl(a.todo))
    t = todo[a.index]
    msgs = build_pair_messages(t["query"], t["a"], t["b"], a.blind)
    print("=== system ===\n" + msgs[0]["content"] + "\n\n=== user ===\n" + msgs[1]["content"])
    print(f"\n（字符数：system {len(msgs[0]['content'])}，user {len(msgs[1]['content'])}）")


def cmd_judge(a):
    client = Client(a.base, a.key, a.model, a.timeout, a.json_mode, a.max_tokens)
    todo = list(read_jsonl(a.todo))
    if a.limit:
        todo = todo[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for j in read_jsonl(a.out):
            done.add(j["key"])
    tasks = []
    for t in todo:
        for blind in (False, True):
            if a.no_blind and blind:
                continue
            for order in ("ab", "ba"):
                key = f"{t['pair_id']}|{'blind' if blind else 'host'}|{order}"
                if key not in done:
                    tasks.append((key, t, blind, order))
    print(f"待判 {len(tasks)} 次调用（已完成 {len(done)}），模型 {a.model}，prompt {PROMPT_VERSION}", flush=True)
    lock = threading.Lock()
    stats = Counter()
    t0 = time.time()

    def work(task):
        key, t, blind, order = task
        pa, pb = (t["a"], t["b"]) if order == "ab" else (t["b"], t["a"])
        msgs = build_pair_messages(t["query"], pa, pb, blind)
        rec = {"key": key, "pair_id": t["pair_id"], "qid": t["qid"], "blind": blind, "order": order,
               "model": a.model, "prompt_version": PROMPT_VERSION, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            text, usage = client.chat(msgs)
            rec["raw"] = text
            rec["usage"] = usage
            try:
                rec["parsed"] = parse_json(text)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"JSON 解析失败: {e}"
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
        return rec

    with open(a.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, task) for task in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                stats["error" if "error" in rec else "ok"] += 1
                if i % 50 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}  ok={stats['ok']} error={stats['error']}  {time.time() - t0:.0f}s", flush=True)
    print(f"完成：{dict(stats)} → {a.out}")


def _verdict_in_original_order(parsed, order):
    v = str(parsed.get("verdict", "")).strip().lower()
    if v not in ("a", "b", "tie"):
        return None
    if order == "ba" and v in ("a", "b"):
        v = "b" if v == "a" else "a"
    return v


def _relevance(parsed, order):
    r = parsed.get("relevance") or {}
    ra, rb = r.get("a"), r.get("b")
    if order == "ba":
        ra, rb = rb, ra
    try:
        return int(ra), int(rb)
    except (TypeError, ValueError):
        return None, None


def cmd_build(a):
    todo = {t["pair_id"]: t for t in read_jsonl(a.todo)}
    by_pair = defaultdict(dict)
    for j in read_jsonl(a.judgments):
        if "parsed" in j:
            by_pair[j["pair_id"]][(("blind" if j["blind"] else "host"), j["order"])] = j["parsed"]
    os.makedirs(a.out, exist_ok=True)
    stats = {"host": Counter(), "blind": Counter()}
    outs = {"host": open(os.path.join(a.out, "pairs_with_host.tsv"), "w", encoding="utf-8"),
            "blind": open(os.path.join(a.out, "pairs_blind.tsv"), "w", encoding="utf-8")}
    for f in outs.values():
        f.write("url_a\turl_b\tpref\tgroup\tqid\n")
    detail = open(os.path.join(a.out, "pairs_detail.jsonl"), "w", encoding="utf-8")
    for pid, t in todo.items():
        for rnd in ("host", "blind"):
            j1, j2 = by_pair[pid].get((rnd, "ab")), by_pair[pid].get((rnd, "ba"))
            st = stats[rnd]
            if j1 is None or j2 is None:
                st["missing_judgment"] += 1
                continue
            v1, v2 = _verdict_in_original_order(j1, "ab"), _verdict_in_original_order(j2, "ba")
            if v1 is None or v2 is None:
                st["bad_verdict"] += 1
                continue
            if {v1, v2} == {"a", "b"}:
                st["inconsistent"] += 1
                pref = None
            elif v1 == v2:
                pref = v1
            else:
                pref = "tie"  # 一次 tie 一次有偏向
            ra1, rb1 = _relevance(j1, "ab")
            ra2, rb2 = _relevance(j2, "ba")
            rel_a = min(x for x in (ra1, ra2) if x is not None) if any(x is not None for x in (ra1, ra2)) else 2
            rel_b = min(x for x in (rb1, rb2) if x is not None) if any(x is not None for x in (rb1, rb2)) else 2
            rel_gap = abs(rel_a - rel_b)
            if rel_a == 0 or rel_b == 0 or rel_gap >= 2:
                st["relevance_dropped"] += 1
                pref = None
            rec = {"pair_id": pid, "qid": t["qid"], "round": rnd, "url_a": t["a"]["url"], "url_b": t["b"]["url"],
                   "pref": pref, "v_ab": v1, "v_ba": v2, "rel_a": rel_a, "rel_b": rel_b, "rel_gap": rel_gap,
                   "rank_a": t["a"].get("rank"), "rank_b": t["b"].get("rank"), "j_ab": j1, "j_ba": j2}
            detail.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if pref is None:
                continue
            st["kept"] += 1
            st["tie" if pref == "tie" else "decided"] += 1
            group = t["group"] + ("|relgap1" if rel_gap == 1 else "")
            outs[rnd].write(f"{t['a']['url']}\t{t['b']['url']}\t{pref}\t{group}\t{t['qid']}\n")
    for f in outs.values():
        f.close()
    detail.close()
    summary = {k: dict(v) for k, v in stats.items()}
    for rnd, st in stats.items():
        tot = sum(v for k, v in st.items() if k in ("kept", "inconsistent", "relevance_dropped", "bad_verdict"))
        if tot:
            summary[rnd]["order_consistency"] = round(1 - st["inconsistent"] / tot, 4)
            summary[rnd]["tie_rate_among_kept"] = round(st["tie"] / max(1, st["kept"]), 4)
    with open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"→ {a.out}/pairs_with_host.tsv, pairs_blind.tsv, pairs_detail.jsonl, stats.json")


def cmd_split(a):
    def load(path):
        rows = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) >= 5 and cols[0] != "url_a":
                        rows.append(cols)
        return rows

    host = load(os.path.join(a.labels, "pairs_with_host.tsv"))
    blind = load(os.path.join(a.labels, "pairs_blind.tsv"))
    qids = sorted({r[4] for r in host} | {r[4] for r in blind})
    random.Random(a.seed).shuffle(qids)
    n = len(qids)
    part = {}
    for i, q in enumerate(qids):
        part[q] = "train" if i < a.train_frac * n else ("dev" if i < (a.train_frac + a.dev_frac) * n else "holdout")
    dests = {"train": a.out_train, "dev": a.out_dev, "holdout": a.out_holdout}
    counts = Counter()
    for name, rows in (("pairs.tsv", host), ("pairs_blind.tsv", blind)):
        for split, d in dests.items():
            if not d:
                continue
            os.makedirs(d, exist_ok=True)
            sel = [r for r in rows if part[r[4]] == split]
            if name == "pairs_blind.tsv" and split != "dev":
                continue  # 遮域名一轮只作 dev 护栏
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write("url_a\turl_b\tpref\tgroup\tqid\n")
                for r in sel:
                    f.write("\t".join(r) + "\n")
            counts[f"{split}/{name}"] = len(sel)
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    print(f"query 数：train {sum(1 for q in part.values() if q == 'train')}, dev {sum(1 for q in part.values() if q == 'dev')}, holdout {sum(1 for q in part.values() if q == 'holdout')}")


def cmd_grade(a):
    client = Client(a.base, a.key, a.model, a.timeout, a.json_mode, a.max_tokens)
    pages = [p for p in read_jsonl(a.pages) if a.rank_min <= int(p.get("rank", 0)) <= a.rank_max and p.get("url")]
    rnd = random.Random(a.seed)
    rnd.shuffle(pages)
    if a.limit:
        pages = pages[: a.limit]
    done = {}
    if os.path.exists(a.out):
        for j in read_jsonl(a.out):
            done[j["url"]] = j
    tasks = [p for p in pages if p["url"] not in done]
    print(f"待打档位 {len(tasks)} 页（已完成 {len(done)}）", flush=True)
    lock = threading.Lock()

    def work(p):
        rec = {"url": p["url"], "qid": p.get("qid"), "model": a.model, "prompt_version": "grade-v1"}
        try:
            text, usage = client.chat(build_grade_messages(p.get("query", ""), p))
            rec["raw"], rec["usage"] = text, usage
            rec["parsed"] = parse_json(text)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
        return rec

    with open(a.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, p) for p in tasks]), 1):
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[rec["url"]] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    if a.grades_tsv:
        n = 0
        with open(a.grades_tsv, "w", encoding="utf-8") as f:
            f.write("url\tgrade\tgroup\n")
            for rec in done.values():
                g = (rec.get("parsed") or {}).get("authority_grade")
                try:
                    g = int(g)
                except (TypeError, ValueError):
                    continue
                f.write(f"{rec['url']}\t{max(0, min(4, g))}\tall\n")
                n += 1
        print(f"→ {a.grades_tsv}（{n} 行）")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="从 pages.jsonl 抽头尾对")
    s.add_argument("--pages", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-query", type=int, default=1)
    s.add_argument("--head-min", type=int, default=1)
    s.add_argument("--head-max", type=int, default=10)
    s.add_argument("--tail-min", type=int, default=11)
    s.add_argument("--tail-max", type=int, default=20)
    s.add_argument("--require-text", action="store_true", help="没有正文的页面不参与抽样")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("prompt", help="打印某一对的 prompt")
    s.add_argument("--todo", required=True)
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--blind", action="store_true")
    s.set_defaults(fn=cmd_prompt)

    for name, fn in (("judge", cmd_judge), ("grade", cmd_grade)):
        s = sub.add_parser(name, help="调用判官" if name == "judge" else "单页绝对档位")
        if name == "judge":
            s.add_argument("--todo", required=True)
            s.add_argument("--no-blind", action="store_true", help="不跑遮域名一轮")
        else:
            s.add_argument("--pages", required=True)
            s.add_argument("--rank-min", type=int, default=21)
            s.add_argument("--rank-max", type=int, default=50)
            s.add_argument("--grades-tsv", default=None, help="同时导出 grades.tsv")
            s.add_argument("--seed", type=int, default=7)
        s.add_argument("--out", required=True, help="judgments.jsonl，追加写、可续跑")
        s.add_argument("--base", default=os.environ.get("OPENAI_BASE_URL"))
        s.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", "none"))
        s.add_argument("--model", default=os.environ.get("JUDGE_MODEL"))
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--limit", type=int, default=0, help="只处理前 N 对/页，用于小批量试跑")
        s.add_argument("--timeout", type=float, default=120)
        s.add_argument("--max-tokens", type=int, default=600)
        s.add_argument("--json-mode", action="store_true", help="请求 response_format=json_object（服务端支持时）")
        s.set_defaults(fn=fn)

    s = sub.add_parser("build", help="一致性过滤并生成 pairs.tsv")
    s.add_argument("--judgments", required=True)
    s.add_argument("--todo", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("split", help="按 qid 切 train/dev/holdout")
    s.add_argument("--labels", required=True, help="build 的输出目录")
    s.add_argument("--out-train", required=True)
    s.add_argument("--out-dev", required=True)
    s.add_argument("--out-holdout", default=None)
    s.add_argument("--train-frac", type=float, default=0.6)
    s.add_argument("--dev-frac", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=cmd_split)

    a = p.parse_args(argv)
    if a.cmd in ("judge", "grade") and (not a.base or not a.model):
        p.error("需要 --base 与 --model（或环境变量 OPENAI_BASE_URL、JUDGE_MODEL）")
    a.fn(a)


if __name__ == "__main__":
    main()
