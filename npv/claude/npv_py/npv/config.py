"""打分模型的全部可调参数。

把 Java 里散落的魔法数字集中到一个 dataclass，实验时只需构造一个不同的 ScoreConfig，
不必改打分代码。默认值与 Java 立即修复版完全一致。
"""
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class PcRule:
    """页面类别规则：命中 pc 的某一位，或 URL 含某些子串，则乘以 factor。

    规则按列表顺序匹配，取第一个命中的（对应 Java 的 if-else 链，顺序即优先级）。
    """
    name: str
    bit: int
    factor: float
    url_substrings: Tuple[str, ...] = ()
    lowercase_url: bool = False  # Java 中 "news" 是在 url.toLowerCase() 上匹配，"/u/" 不是

    def matches(self, pc: int, url: str) -> bool:
        if (pc >> self.bit) & 1 == 1:
            return True
        if self.url_substrings:
            target = url.lower() if self.lowercase_url else url
            return any(s in target for s in self.url_substrings)
        return False


DEFAULT_PC_RULES: Tuple[PcRule, ...] = (
    PcRule("bit10", 10, 0.7),
    PcRule("bit19", 19, 0.8),
    PcRule("bit13", 13, 0.6),
    PcRule("bit18", 18, 0.9),
    PcRule("bit16", 16, 0.8),
    PcRule("bit33", 33, 0.7),
    PcRule("bit24", 24, 0.6),
    PcRule("bit38", 38, 0.6),
    PcRule("bit22", 22, 0.7),
    PcRule("bit11_or_news", 11, 0.9, ("news",), lowercase_url=True),
    PcRule("bit26_or_user", 26, 0.6, ("/u/", "/user/")),
    PcRule("bit29", 29, 0.9),
    PcRule("bit32", 32, 0.8),
)


@dataclass(frozen=True)
class ScoreConfig:
    # 基础分权重（Java: FEA_WEIGHT）
    fea_weight: Dict[str, float] = field(
        default_factory=lambda: {"spr_sr": 60, "dr": 12, "ow": 4})  # 2026-09-11 起去掉 pr

    # dr 归一化分母（Java: MAX_DR）
    max_dr: float = 3.0

    # spr_sr 合成（Java: getSrSprScore）
    spr_th1: float = 0.36
    spr_th2: float = 0.2
    spr_th1_out: float = 0.9      # spr > th1 时映射到 [th1_out, 1]
    spr_th2_out: float = 0.2      # th2 < spr <= th1 时映射到 [th2_out, th1_out]
    spr_zero_fill: float = 0.1    # spr 恰为 0 时置为该值
    spr_coef: float = 0.25        # srNew = (sr + spr_coef * spr) / sr_spr_div
    sr_spr_div: float = 1.1
    sr_missing_score: float = 0.25  # sr < 0 时 spr_sr 的取值

    # 时间衰减常数（天）
    pct_decay_days: float = 2000.0
    pt_decay_days: float = 3000.0

    # 正文长度：(阈值, 系数)，按顺序取第一个 pure_text_len < 阈值 的
    text_len_rules: Tuple[Tuple[int, float], ...] = ((300, 0.8), (500, 0.9))

    # 页面类别
    pc_floor_bit: int = 20        # 命中该位则 score = max(score, pc_floor)
    pc_floor: float = 60.0
    pc_rules: Tuple[PcRule, ...] = DEFAULT_PC_RULES

    # adc 加分
    adc_bonus: Dict[int, float] = field(default_factory=lambda: {2: 10.0, 3: 15.0})
    adc_whitelist_level: int = 3  # url 命中 adc 白名单时赋予的等级
    adc_skip_adjust_from: int = 2  # adc >= 该值时跳过页面类别与 pt 衰减

    # 排名归一化
    norm_z_start: float = -2.0
    norm_z_end: float = 3.0
    norm_buckets: int = 1000


def region_site_list(region_flag: str) -> Sequence[str]:
    """Java main 中按区域硬编码的官网白名单后缀。"""
    if region_flag == "zh":
        # TODO 待业务方确认：".edu.com" 疑为 ".edu.cn" 笔误，与 Java 保持一致暂不改
        return (".gov.cn", ".bendibao.com", ".edu.com", ".baike.com", "baike.baidu.com")
    return (".gov", ".edu")
