"""离线评估：偏序标签 + 绝对档位标签，对任意版本的打分输出算指标。

标签格式（TSV，可带以 url 开头的表头行）：
  pairs.tsv : url_a  url_b  pref(a|b|tie)  [group]
  grades.tsv: url    grade(0-4)            [group]
打分输出：run_lab.py / run_npv.py 的 npv_ori.tsv（url, score, fea_json）。
"""
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .stats import mean, quantile, sign_test_pvalue, spearman

DEFAULT_GAP_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, float("inf"))


def read_scores(path: str, with_features: bool = False) -> Dict[str, Tuple[float, Optional[dict]]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            try:
                score = float(cols[1])
            except ValueError:
                continue
            fea = None
            if with_features:
                try:
                    fea = json.loads(cols[-1])
                except (ValueError, IndexError):
                    fea = None
            out[cols[0]] = (score, fea)
    return out


def read_pairs(path: str) -> List[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or cols[0] == "url_a":
                continue
            pairs.append({"url_a": cols[0], "url_b": cols[1], "pref": cols[2].strip().lower(),
                          "group": cols[3] if len(cols) > 3 else "all"})
    return pairs


def read_grades(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2 or cols[0] == "url":
                continue
            rows.append({"url": cols[0], "grade": float(cols[1]),
                         "group": cols[2] if len(cols) > 2 else "all"})
    return rows


def _gap_bucket(gap: float, edges: Sequence[float]) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= gap < hi:
            return f"[{lo:g},{hi:g})"
    return f"[{edges[-2]:g},inf)"


def pairwise_metrics(scores: Dict[str, Tuple[float, Optional[dict]]], pairs: List[dict],
                     gap_edges: Sequence[float] = DEFAULT_GAP_EDGES) -> Dict:
    n_missing = n_label_tie = n_score_tie = 0
    correct_flags: List[Tuple[bool, float, str]] = []  # (对错, 分差, group)
    tie_gaps: List[float] = []
    per_pair: List[Optional[bool]] = []  # 与 pairs 等长，None 表示未计入
    for p in pairs:
        a, b = scores.get(p["url_a"]), scores.get(p["url_b"])
        if a is None or b is None:
            n_missing += 1
            per_pair.append(None)
            continue
        gap = abs(a[0] - b[0])
        if p["pref"] not in ("a", "b"):
            n_label_tie += 1
            tie_gaps.append(gap)
            per_pair.append(None)
            continue
        if a[0] == b[0]:
            n_score_tie += 1
            per_pair.append(None)
            continue
        ok = (a[0] > b[0]) == (p["pref"] == "a")
        correct_flags.append((ok, gap, p["group"]))
        per_pair.append(ok)

    def acc(flags):
        return sum(1 for f in flags if f) / len(flags) if flags else float("nan")

    by_bucket: Dict[str, List[bool]] = defaultdict(list)
    by_group: Dict[str, List[bool]] = defaultdict(list)
    for ok, gap, g in correct_flags:
        by_bucket[_gap_bucket(gap, gap_edges)].append(ok)
        by_group[g].append(ok)
    n_used = len(correct_flags)
    a = acc([ok for ok, _, _ in correct_flags])
    se = math.sqrt(a * (1 - a) / n_used) if n_used and not math.isnan(a) else float("nan")
    bucket_keys = [_gap_bucket(lo, gap_edges) for lo in gap_edges[:-1]]
    return {
        "n_pairs": len(pairs), "n_used": n_used, "n_missing_url": n_missing,
        "n_label_tie": n_label_tie, "n_score_tie": n_score_tie,
        "accuracy": a, "accuracy_ci95": (a - 1.96 * se, a + 1.96 * se) if n_used else (float("nan"),) * 2,
        "accuracy_by_gap": {k: {"n": len(by_bucket[k]), "accuracy": acc(by_bucket[k])} for k in bucket_keys if by_bucket[k]},
        "hard_accuracy": acc(by_bucket[bucket_keys[0]]) if by_bucket[bucket_keys[0]] else float("nan"),
        "accuracy_by_group": {g: {"n": len(v), "accuracy": acc(v)} for g, v in sorted(by_group.items())},
        "label_tie_mean_gap": mean(tie_gaps) if tie_gaps else float("nan"),
        "nontie_mean_gap": mean([gap for _, gap, _ in correct_flags]) if correct_flags else float("nan"),
        "_per_pair": per_pair,
    }


def grade_metrics(scores: Dict[str, Tuple[float, Optional[dict]]], grades: List[dict],
                  bottom_frac: float = 0.2, top_frac: float = 0.2, good_grade: float = 3,
                  bad_grade: float = 1, n_bins: int = 10) -> Dict:
    rows = [(scores[g["url"]][0], g["grade"]) for g in grades if g["url"] in scores]
    if len(rows) < 2:
        return {"n_used": len(rows), "n_missing_url": len(grades) - len(rows)}
    xs, ys = [r[0] for r in rows], [r[1] for r in rows]
    rows_sorted = sorted(rows)
    n = len(rows_sorted)
    bottom = rows_sorted[: max(1, int(n * bottom_frac))]
    top = rows_sorted[n - max(1, int(n * top_frac)):]
    bins = []
    for i in range(n_bins):
        chunk = rows_sorted[i * n // n_bins:(i + 1) * n // n_bins]
        if chunk:
            bins.append({"score_lo": chunk[0][0], "score_hi": chunk[-1][0], "n": len(chunk),
                         "mean_grade": mean([g for _, g in chunk])})
    violations = sum(1 for a, b in zip(bins, bins[1:]) if b["mean_grade"] < a["mean_grade"])
    return {
        "n_used": n, "n_missing_url": len(grades) - n,
        "spearman": spearman(xs, ys),
        "buried_good_rate": sum(1 for _, g in bottom if g >= good_grade) / len(bottom),  # 误压率
        "top_bad_rate": sum(1 for _, g in top if g <= bad_grade) / len(top),            # 误抬率
        "calibration_bins": bins,
        "calibration_violations": violations,
        "good_share_overall": sum(1 for g in ys if g >= good_grade) / n,
    }


def constraint_checks(scores: Dict[str, Tuple[float, Optional[dict]]],
                      score_min: float = 0.0, score_max: float = 110.0) -> Dict:
    vals = [s for s, _ in scores.values()]
    nan = sum(1 for v in vals if math.isnan(v) or math.isinf(v))
    out_of_range = sum(1 for v in vals if not math.isnan(v) and not (score_min <= v <= score_max))
    return {
        "n_scored": len(vals), "nan_or_inf": nan, "out_of_range": out_of_range,
        "score_p50": quantile(vals, 0.5) if vals else float("nan"),
        "passed": nan == 0 and out_of_range == 0 and len(vals) > 0,
    }


def compare_pairwise(scores_a: Dict, scores_b: Dict, pairs: List[dict]) -> Dict:
    """同一批对上的配对比较：A 对 B 错、A 错 B 对的数量与符号检验 p 值。"""
    ma, mb = pairwise_metrics(scores_a, pairs), pairwise_metrics(scores_b, pairs)
    a_only = b_only = both = neither = 0
    for ra, rb in zip(ma["_per_pair"], mb["_per_pair"]):
        if ra is None or rb is None:
            continue
        if ra and not rb:
            a_only += 1
        elif rb and not ra:
            b_only += 1
        elif ra and rb:
            both += 1
        else:
            neither += 1
    return {"accuracy_a": ma["accuracy"], "accuracy_b": mb["accuracy"],
            "a_right_b_wrong": a_only, "b_right_a_wrong": b_only, "both_right": both, "both_wrong": neither,
            "sign_test_p": sign_test_pvalue(a_only, b_only)}


def evaluate(scores_path: str, pairs_path: Optional[str] = None, grades_path: Optional[str] = None,
             gap_edges: Sequence[float] = DEFAULT_GAP_EDGES) -> Dict:
    scores = read_scores(scores_path)
    out = {"scores_path": scores_path, "constraints": constraint_checks(scores)}
    if pairs_path:
        m = pairwise_metrics(scores, read_pairs(pairs_path), gap_edges)
        m.pop("_per_pair", None)
        out["pairwise"] = m
    if grades_path:
        out["grades"] = grade_metrics(scores, read_grades(grades_path))
    return out


def format_report(r: Dict) -> str:
    L = ["# 评估结果", "", f"- 打分文件：{r['scores_path']}"]
    c = r["constraints"]
    L.append(f"- 硬约束：{'通过' if c['passed'] else '**未通过**'}（记录 {c['n_scored']}，NaN/Inf {c['nan_or_inf']}，越界 {c['out_of_range']}）")
    if "pairwise" in r:
        p = r["pairwise"]
        lo, hi = p["accuracy_ci95"]
        L += ["", "## 偏序一致率", "",
              f"- 总体 **{p['accuracy']:.1%}**（95% 区间 {lo:.1%} 到 {hi:.1%}），有效对 {p['n_used']} / {p['n_pairs']}",
              f"- 剔除：url 未打分 {p['n_missing_url']}，标签为相当 {p['n_label_tie']}，模型分数相同 {p['n_score_tie']}",
              f"- 小分差桶（最难的一档）一致率 **{p['hard_accuracy']:.1%}**",
              f"- 标签为相当的对平均分差 {p['label_tie_mean_gap']:.2f}，非相当的对平均分差 {p['nontie_mean_gap']:.2f}",
              "", "| 分差桶 | n | 一致率 |", "| --- | --- | --- |"]
        for k, v in p["accuracy_by_gap"].items():
            L.append(f"| {k} | {v['n']} | {v['accuracy']:.1%} |")
        if len(p["accuracy_by_group"]) > 1:
            L += ["", "| 分组 | n | 一致率 |", "| --- | --- | --- |"]
            for g, v in p["accuracy_by_group"].items():
                L.append(f"| {g} | {v['n']} | {v['accuracy']:.1%} |")
    if "grades" in r and r["grades"].get("n_used", 0) >= 2:
        g = r["grades"]
        L += ["", "## 绝对档位", "",
              f"- 有效 {g['n_used']}，Spearman 相关 **{g['spearman']:.3f}**",
              f"- 误压率（分数最低 20% 中被判为好的比例）**{g['buried_good_rate']:.1%}**，全体中好页占比 {g['good_share_overall']:.1%}",
              f"- 误抬率（分数最高 20% 中被判为差的比例）**{g['top_bad_rate']:.1%}**",
              f"- 校准曲线单调性违反次数 {g['calibration_violations']}",
              "", "| 分数区间 | n | 平均档位 |", "| --- | --- | --- |"]
        for b in g["calibration_bins"]:
            L.append(f"| {b['score_lo']:.1f} 到 {b['score_hi']:.1f} | {b['n']} | {b['mean_grade']:.2f} |")
    return "\n".join(L)
