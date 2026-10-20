"""冻结的指标：严格偏序一致率、护栏、配对比较（按 qid 聚类自助法）。

严格一致率 = (判对数 + 0.5 × 分数相同数) / (非"相当"标签的对数，含打分缺失的对)
打分缺失的对计为错，分数相同的对计半对，两者都不再被排除在分母之外。
"""
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from lab.evaluate import constraint_checks, grade_metrics  # noqa: F401  (复用冻结基线实验层的实现)
from lab.stats import sign_test_pvalue

HARD_GAP = 2.0


def strict_pairwise(scores: Dict[str, Tuple[float, Optional[dict]]], pairs: List[dict]) -> dict:
    """返回总体指标与逐对结果。逐对结果 outcome ∈ {1.0, 0.5, 0.0}，missing 的对 outcome=0。"""
    per_pair = []
    n_missing = n_tie = n_label_tie = 0
    by_group = defaultdict(list)
    hard = []
    for p in pairs:
        if p["pref"] not in ("a", "b"):
            n_label_tie += 1
            continue
        a, b = scores.get(p["url_a"]), scores.get(p["url_b"])
        if a is None or b is None:
            n_missing += 1
            outcome, gap = 0.0, None
        elif a[0] == b[0]:
            n_tie += 1
            outcome, gap = 0.5, 0.0
        else:
            outcome = 1.0 if (a[0] > b[0]) == (p["pref"] == "a") else 0.0
            gap = abs(a[0] - b[0])
        per_pair.append({"qid": p["qid"], "group": p["group"], "outcome": outcome, "gap": gap,
                         "url_a": p["url_a"], "url_b": p["url_b"]})
        by_group[p["group"]].append(outcome)
        if gap is not None and gap < HARD_GAP:
            hard.append(outcome)
    n = len(per_pair)
    acc = sum(r["outcome"] for r in per_pair) / n if n else float("nan")
    return {
        "n_eligible": n, "n_label_tie": n_label_tie, "n_missing": n_missing, "n_score_tie": n_tie,
        "accuracy": acc,
        "hard_accuracy": (sum(hard) / len(hard)) if hard else float("nan"),
        "n_hard": len(hard),
        "accuracy_by_group": {g: {"n": len(v), "accuracy": sum(v) / len(v)} for g, v in sorted(by_group.items())},
        "per_pair": per_pair,
    }


def subset_accuracy(per_pair: List[dict], keep) -> Tuple[float, int]:
    sel = [r["outcome"] for r in per_pair if keep(r)]
    return (sum(sel) / len(sel) if sel else float("nan")), len(sel)


def paired_compare(per_a: List[dict], per_b: List[dict], n_boot: int = 1000, seed: int = 7) -> dict:
    """A 相对 B 的逐对差：均值、按 qid 聚类自助法标准误、符号检验 p 值。两份 per_pair 必须同序同长。"""
    assert len(per_a) == len(per_b)
    diffs = [ra["outcome"] - rb["outcome"] for ra, rb in zip(per_a, per_b)]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "delta": float("nan"), "se_cluster": float("nan"), "wins": 0, "losses": 0, "p_sign": 1.0}
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    clusters = defaultdict(list)
    for r, d in zip(per_a, diffs):
        clusters[r["qid"]].append(d)
    keys = list(clusters)
    rnd = random.Random(seed)
    means = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _k in range(len(keys)):
            c = clusters[keys[rnd.randrange(len(keys))]]
            tot += sum(c)
            cnt += len(c)
        means.append(tot / cnt if cnt else 0.0)
    mu = sum(means) / len(means)
    se = math.sqrt(sum((m - mu) ** 2 for m in means) / (len(means) - 1)) if len(means) > 1 else float("nan")
    return {"n": n, "delta": sum(diffs) / n, "se_cluster": se, "n_clusters": len(keys),
            "wins": wins, "losses": losses, "p_sign": sign_test_pvalue(wins, losses)}


def complexity(features_path: str, fea_weight: Dict[str, float], n_rules: int) -> dict:
    loc = 0
    with open(features_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                loc += 1
    nonzero = sum(1 for v in fea_weight.values() if abs(v) > 1e-9)
    return {"loc": loc, "nonzero_weights": nonzero, "pc_rules": n_rules, "total": loc + 5 * nonzero + 2 * n_rules}
