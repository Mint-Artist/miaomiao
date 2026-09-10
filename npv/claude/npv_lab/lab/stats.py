"""纯标准库的统计小工具。"""
import math
from typing import Dict, List, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def variance(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def quantile(xs: Sequence[float], q: float) -> float:
    """线性插值分位数，xs 无需预排序。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def quantiles(xs: Sequence[float], qs=(0.0, 0.1, 0.5, 0.9, 1.0)) -> Dict[str, float]:
    return {f"p{int(q * 100)}": quantile(xs, q) for q in qs}


def ranks_avg_ties(xs: Sequence[float]) -> List[float]:
    """平均秩（并列取平均），用于 Spearman。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ranks_avg_ties(xs), ranks_avg_ties(ys))


def sign_test_pvalue(n_pos: int, n_neg: int) -> float:
    """双侧符号检验（配对比较：A 对 B 错 vs A 错 B 对），精确二项分布。"""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def between_within_variance(groups: Dict[str, List[float]]) -> Tuple[float, float, float]:
    """返回 (总方差, 组间方差, 组内方差)，组间/总 即"站点级解释的方差占比"。"""
    allv = [v for vs in groups.values() for v in vs]
    if not allv:
        return float("nan"), float("nan"), float("nan")
    gm = mean(allv)
    n = len(allv)
    between = sum(len(vs) * (mean(vs) - gm) ** 2 for vs in groups.values()) / n
    within = sum(sum((v - mean(vs)) ** 2 for v in vs) for vs in groups.values()) / n
    return variance(allv), between, within
