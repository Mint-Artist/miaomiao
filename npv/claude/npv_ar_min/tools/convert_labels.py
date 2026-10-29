#!/usr/bin/env python3
"""把 labeled_train.jsonl / labeled_test.jsonl（gemini 判官产物，18 字段）转成 harness 用的 pairs.tsv，并输出统计。

  python tools/convert_labels.py --train output/labeled_train.jsonl --test output/labeled_test.jsonl \\
      --out-train labels/train --out-test ../npv_ar_private/min_test --aux data

产出：
  <out-train>/pairs.tsv, <out-test>/pairs.tsv      五列 url_a url_b pref group qid；url_a 固定为 pos_url
  <aux>/urls.txt                                   所有出现过的 URL（去重），用于向上游要打分字段（input.tsv）
  <aux>/online_npv.tsv                             url \\t 线上 npv（1~1000），与打分出的原始分配对后可拟合 level_map
  <aux>/label_stats.json                           标签①②③分布、npv 对 LLM 的一致率、按分差分桶
"""
import argparse
import json
import os
from collections import Counter, OrderedDict


def read(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def convert(rows, tie_band):
    pairs, urls, online = [], OrderedDict(), {}
    st = {"llm_verdict": Counter(), "npv_verdict": Counter(), "npv_correct": Counter(), "seq_pairs": Counter(),
          "by_diff": OrderedDict((k, [0, 0.0]) for k in ("<=band", "band-100", "100-200", "200-400", ">400"))}
    strict_num = strict_den = 0.0
    for r in rows:
        st["llm_verdict"][r.get("llm_verdict")] += 1
        st["npv_verdict"][r.get("npv_verdict")] += 1
        st["npv_correct"][str(r.get("npv_correct"))] += 1
        st["seq_pairs"][f"{r.get('seq1_verdict')}|{r.get('seq2_verdict')}"] += 1
        for u in (r.get("pos_url"), r.get("neg_url")):
            if u:
                urls[u] = None
        for k in ("pos", "neg"):
            if r.get(f"{k}_npv") is not None and r.get(f"{k}_url"):
                online[r[f"{k}_url"]] = r[f"{k}_npv"]
        v = r.get("llm_verdict")
        if v not in ("pos", "neg", "tie"):
            continue
        qid = r["id"].rsplit("_p", 1)[0] if "_p" in r.get("id", "") else r.get("id", "")
        pref = {"pos": "a", "neg": "b", "tie": "tie"}[v]
        pairs.append((r["pos_url"], r["neg_url"], pref, "all", qid))
        # 严格一致率口径：npv 方向对=1，分差 ≤ band=0.5，否则 0；只算 LLM 有明确方向的对
        if pref != "tie" and r.get("pos_npv") is not None and r.get("neg_npv") is not None:
            d = float(r["pos_npv"]) - float(r["neg_npv"])
            if abs(d) <= tie_band:
                val, bucket = 0.5, "<=band"
            else:
                val = 1.0 if (d > 0) == (pref == "a") else 0.0
                ad = abs(d)
                bucket = "band-100" if ad <= 100 else "100-200" if ad <= 200 else "200-400" if ad <= 400 else ">400"
            strict_num += val
            strict_den += 1
            st["by_diff"][bucket][0] += 1
            st["by_diff"][bucket][1] += val
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in st.items()}
    stats["seq_pairs"] = dict(Counter(st["seq_pairs"]).most_common(12))
    stats["pairs_kept"] = len(pairs)
    stats["pairs_decided"] = sum(1 for p in pairs if p[2] != "tie")
    stats["queries"] = len({p[4] for p in pairs})
    tc = st["npv_correct"]
    stats["npv_correct_rate_官方口径"] = round(tc["true"] / max(1, tc["true"] + tc["false"]), 4)
    stats["strict_acc_同档算半对"] = round(strict_num / strict_den, 4) if strict_den else None
    stats["by_diff"] = {k: {"n": n, "acc": round(s / n, 4) if n else None} for k, (n, s) in st["by_diff"].items()}
    return pairs, list(urls), online, stats


def write_pairs(path, pairs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("url_a\turl_b\tpref\tgroup\tqid\n")
        for p in pairs:
            f.write("\t".join(p) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", default=None)
    ap.add_argument("--aux", required=True, help="urls.txt、online_npv.tsv、label_stats.json 的输出目录")
    ap.add_argument("--tie-band", type=float, default=50.0, help="线上 1~1000 尺度下的同档区间")
    a = ap.parse_args(argv)
    os.makedirs(a.aux, exist_ok=True)
    all_urls, all_online, all_stats = [], {}, {}
    for name, src, dst in (("train", a.train, a.out_train), ("test", a.test, a.out_test)):
        if not src:
            continue
        pairs, urls, online, stats = convert(read(src), a.tie_band)
        write_pairs(os.path.join(dst, "pairs.tsv"), pairs)
        all_urls += urls
        all_online.update(online)
        all_stats[name] = stats
        print(f"{name}: 保留 {len(pairs)} 对（有方向 {stats['pairs_decided']}，query {stats['queries']}）→ {dst}/pairs.tsv")
    seen = set()
    with open(os.path.join(a.aux, "urls.txt"), "w", encoding="utf-8") as f:
        for u in all_urls:
            if u not in seen:
                seen.add(u)
                f.write(u + "\n")
    with open(os.path.join(a.aux, "online_npv.tsv"), "w", encoding="utf-8") as f:
        for u, v in all_online.items():
            f.write(f"{u}\t{v}\n")
    with open(os.path.join(a.aux, "label_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"URL {len(seen)} 个 → {a.aux}/urls.txt；线上 npv {len(all_online)} 条 → online_npv.tsv")
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
