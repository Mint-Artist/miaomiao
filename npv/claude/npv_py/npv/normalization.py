"""排名归一化，对应 Java 的 generateUrlInterval 与 normalizationScore。"""
import bisect
import math
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def normal_cdf(x: float) -> float:
    """标准正态分布累积函数，替代 commons-math 的 NormalDistribution(0,1).cumulativeProbability。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def generate_url_interval(start: float, end: float, url_num: int, buckets: int = 1000) -> List[int]:
    """把 [start, end] 上的标准正态分布切成 buckets 段，按概率质量分配 URL 数，返回累计边界。

    返回列表长度为 buckets + 1，intervals[0] = 0。intervals[i] 之前的名次属于第 i-1 个桶。
    第 0 个桶对应 z 最靠近 end 的那一段（即分数最高的 URL），与 Java 的从 i=buckets 倒序累加一致。
    """
    total = normal_cdf(end) - normal_cdf(start)
    intervals = [0]
    step = (end - start) / buckets
    for i in range(buckets, 0, -1):
        diff = normal_cdf(start + i * step) - normal_cdf(start + (i - 1) * step)
        interval = math.ceil(diff / total * url_num)
        intervals.append(interval + intervals[buckets - i])
    return intervals


def level_for_rank(rank: int, intervals: Sequence[int]) -> int:
    """名次 -> 等级（Java: 1000 - idx；落在所有区间之外时为 1001 - len）。"""
    buckets = len(intervals) - 1
    # intervals 严格递增，所以二分与 Java 的线性扫描结果一致
    idx = bisect.bisect_right(intervals, rank) - 1
    if 0 <= idx < buckets:
        return buckets - idx
    return buckets + 1 - len(intervals)


def assign_levels(items: Iterable[T], score_of: Callable[[T], float],
                  tie_key: Callable[[T], object] = None,
                  z_start: float = -2.0, z_end: float = 3.0,
                  buckets: int = 1000) -> List[Tuple[T, int, int]]:
    """按分数降序排名并映射为等级，返回 [(item, rank, level)]。

    与 Java 的差异：Java 对并列分数的顺序不确定（取决于分区物理顺序），
    这里用 tie_key（默认为 item 本身的字符串形式）做二级排序，保证结果可复现。
    """
    items = list(items)
    if tie_key is None:
        tie_key = str
    items.sort(key=lambda it: (-score_of(it), tie_key(it)))
    intervals = generate_url_interval(z_start, z_end, len(items), buckets)
    return [(it, rank, level_for_rank(rank, intervals)) for rank, it in enumerate(items)]
