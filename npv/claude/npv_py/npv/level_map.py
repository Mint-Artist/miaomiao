"""分数 -> 等级 的阈值表：让样本数据也能算出与线上可比的 npv。

背景：线上 npv 是按名次映射的（generateUrlInterval），同一页面的等级取决于和谁一起排。
样本上直接排名得到的等级与线上不可比。解决办法是用线上均匀抽样的 (npv_ori[, npv]) 拟合一张
"每个等级的最低分数"阈值表，再用阈值表给新页面定级。

两种拟合方法：
- quantile：只需样本的 npv_ori。按理论桶质量（与 generate_url_interval 相同的截断正态）在样本分数的
  经验分布上取分位点作为各等级阈值。要求样本是对参与归一化的页面均匀随机抽的。
- observed：还需要样本的 npv。每个等级取观测到的最低分数作阈值；样本里没出现的等级用 quantile 结果补，
  最后强制单调。有 npv 列时两种方法都会在样本上做回代校验。

阈值表文件：TSV，`level \\t min_score`，等级 1..buckets，min_score 随等级非递减。
"""
import bisect
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .normalization import normal_cdf


class LevelMap:
    def __init__(self, thresholds: Sequence[float], meta: Optional[Dict[str, str]] = None):
        self.thresholds = [float(t) for t in thresholds]  # thresholds[L-1] = 等级 L 的最低分
        self.meta = dict(meta or {})
        for a, b in zip(self.thresholds, self.thresholds[1:]):
            if b < a:
                raise ValueError("阈值表必须随等级非递减")

    @property
    def levels(self) -> int:
        return len(self.thresholds)

    def lookup(self, score: float) -> int:
        """最大的等级 L 使 min_score[L] <= score；低于等级 1 的阈值也记为 1。"""
        if score != score:  # NaN
            return 1
        return max(1, bisect.bisect_right(self.thresholds, score))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for k, v in self.meta.items():
                f.write(f"# {k}: {v}\n")
            f.write("level\tmin_score\n")
            for i, t in enumerate(self.thresholds, 1):
                f.write(f"{i}\t{t!r}\n")

    @classmethod
    def load(cls, path: str) -> "LevelMap":
        meta, rows = {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#"):
                    k, _, v = line[1:].partition(":")
                    meta[k.strip()] = v.strip()
                    continue
                a, b = line.split("\t")[:2]
                if a == "level":
                    continue
                rows[int(a)] = float(b)
        if not rows or sorted(rows) != list(range(1, max(rows) + 1)):
            raise ValueError(f"阈值表等级不连续或为空: {path}")
        return cls([rows[i] for i in range(1, max(rows) + 1)], meta)


def bucket_masses(z_start: float = -2.0, z_end: float = 3.0, buckets: int = 1000) -> List[float]:
    """从顶部（z_end）往下每个桶的概率质量，index 0 对应最高等级，与 generate_url_interval 同序。"""
    total = normal_cdf(z_end) - normal_cdf(z_start)
    step = (z_end - z_start) / buckets
    return [(normal_cdf(z_start + i * step) - normal_cdf(z_start + (i - 1) * step)) / total
            for i in range(buckets, 0, -1)]


def fit_quantile(scores: Iterable[float], z_start: float = -2.0, z_end: float = 3.0,
                 buckets: int = 1000) -> LevelMap:
    s = sorted((x for x in scores if x == x), reverse=True)
    n = len(s)
    if n < buckets:
        raise ValueError(f"样本只有 {n} 条，少于等级数 {buckets}，无法拟合")
    thresholds = [0.0] * buckets
    cum = 0.0
    for idx, m in enumerate(bucket_masses(z_start, z_end, buckets)):
        cum += m
        k = min(n, max(1, math.ceil(cum * n)))  # 前 cum 比例的最后一名（1-based）
        thresholds[buckets - idx - 1] = s[k - 1]
    for i in range(buckets - 2, -1, -1):  # 保证非递减（重复分数可能造成相等，不会倒挂，这里只是兜底）
        thresholds[i] = min(thresholds[i], thresholds[i + 1])
    return LevelMap(thresholds, {"method": "quantile", "n_sample": str(n)})


def fit_observed(scores: Sequence[float], levels: Sequence[int], z_start: float = -2.0,
                 z_end: float = 3.0, buckets: int = 1000) -> LevelMap:
    base = fit_quantile(scores, z_start, z_end, buckets).thresholds
    observed: Dict[int, float] = {}
    for sc, lv in zip(scores, levels):
        if sc != sc or not 1 <= lv <= buckets:
            continue
        observed[lv] = min(observed.get(lv, float("inf")), sc)
    thresholds = list(base)
    for lv, mn in observed.items():
        thresholds[lv - 1] = mn
    for i in range(1, buckets):  # 强制非递减：低等级的阈值不能高于高等级
        if thresholds[i] < thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1]
    return LevelMap(thresholds, {"method": "observed", "n_sample": str(len(scores)),
                                 "levels_observed": str(len(observed))})


def validate(level_map: LevelMap, scores: Sequence[float], levels: Sequence[int]) -> Dict[str, float]:
    """把阈值表回代到带 npv 的样本上，看与真实等级的差距。"""
    diffs = [abs(level_map.lookup(s) - lv) for s, lv in zip(scores, levels) if s == s]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "exact_match": sum(1 for d in diffs if d == 0) / n,
        "within_5": sum(1 for d in diffs if d <= 5) / n,
        "within_20": sum(1 for d in diffs if d <= 20) / n,
        "mean_abs_diff": sum(diffs) / n,
        "max_abs_diff": max(diffs),
    }


def assign_levels_with_map(items: Iterable, score_of, level_map: LevelMap) -> List[Tuple[object, int]]:
    return [(it, level_map.lookup(score_of(it))) for it in items]
