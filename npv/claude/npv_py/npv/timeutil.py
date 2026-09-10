"""时间衰减，对应 Java 立即修复版的 decayFactor / parseTimestampSeconds / daysBetween。"""
import math
from datetime import datetime, tzinfo
from typing import Optional, Union


def parse_timestamp_seconds(time: Union[str, int, None]) -> int:
    """解析秒级时间戳。13 位按毫秒截为 10 位；只接受 10 位纯数字且 > 0，其余返回 -1。"""
    if time is None:
        return -1
    t = str(time).strip()
    if len(t) == 13:
        t = t[:10]
    if len(t) != 10 or not t.isdigit():
        return -1
    v = int(t)
    return v if v > 0 else -1


def days_between(ts_seconds: int, now_seconds: int, tz: Optional[tzinfo] = None) -> int:
    """按日历日计算天数（默认本地时区，与 Java 默认时区行为一致），负值钳制为 0。"""
    start = datetime.fromtimestamp(ts_seconds, tz).date()
    end = datetime.fromtimestamp(now_seconds, tz).date()
    return max(0, (end - start).days)


def decay_factor(time: Union[str, int, None], now_seconds: int, scale_days: float,
                 tz: Optional[tzinfo] = None) -> float:
    """2 / (e^(d/scale) + 1)。缺失/非法时间戳返回 1.0（中性）。"""
    ts = parse_timestamp_seconds(time)
    if ts < 0:
        return 1.0
    days = days_between(ts, now_seconds, tz)
    return 2.0 / (math.exp(days / scale_days) + 1.0)
