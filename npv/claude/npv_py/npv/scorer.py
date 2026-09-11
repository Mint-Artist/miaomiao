"""打分核心，对应 Java 立即修复版 PageValueScore。

2026-09-11：离线端不再生产 pr，已删除全部 pr 逻辑（ScoreInput.pr、pre_pr、特征 pr 与权重）。

与 Java 的差异（有意为之）：
- score() 返回 (分数, 特征 dict)，不再通过实例字段 featureList 传递结果；
- 全部参数来自 ScoreConfig，可注入替换；
- Python float 是 64 位，Java 特征是 32 位 float，个别结果会有 1e-6 量级差异。
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from .config import ScoreConfig
from .site import parse_site as default_parse_site
from .tables import Tables
from .timeutil import decay_factor

TimeLike = Union[str, int, None]


@dataclass
class ScoreInput:
    """一条待打分记录，字段与 Java score() 的入参一一对应。"""
    url: str
    adc: int            # adc.level
    pc: int             # 页面类别位图（Java: pcLong）
    pct: TimeLike       # 秒级时间戳，缺失可为 0/None
    pt: TimeLike
    pure_text_len: int
    sr: int
    spr: float


@dataclass
class ScoreResult:
    score: float
    features: Dict[str, float]


class PageValueScore:
    def __init__(self, tables: Tables, now_seconds: int,
                 config: Optional[ScoreConfig] = None,
                 parse_site: Callable[[str], str] = default_parse_site):
        self.t = tables
        self.now = now_seconds
        self.cfg = config or ScoreConfig()
        self.parse_site = parse_site

    # ---------- 特征预处理（Java: preProcess*） ----------

    def pre_sr(self, sr: int) -> float:
        return sr / 100.0

    def pre_spr(self, spr: float) -> float:
        return spr / self.t.spr_max

    def pre_adc(self, url: str, adc: int) -> float:
        if adc > 0:
            return float(adc)
        for w in self.t.adc_whitelist:
            if w in url:  # 与 Java 一致：整个 URL 上做子串匹配（报告 P1-3）
                return float(self.cfg.adc_whitelist_level)
        return float(adc)

    def pre_dr(self, site: str) -> float:
        v = self.t.dr_site.get(site)
        if v is not None:
            return float(v) / self.cfg.max_dr
        for suffix, val in self.t.dr_suffix.items():
            if site.endswith(suffix):
                return float(val) / self.cfg.max_dr
        return 1.0 / self.cfg.max_dr

    def pre_ow(self, site: str, site_list: Sequence[str]) -> float:
        low = site.lower()
        for s in site_list:
            if low.endswith(s):
                return 1.0
        for b in self.t.ow_blacklist:
            if site.endswith(b):
                return 0.0
        if float(self.t.ow.get(site, "0")) > 0:
            return 1.0
        return 0.0

    def sr_spr_score(self, sr: float, spr: float) -> float:
        """Java: getSrSprScore。sr、spr 均为已归一化值。"""
        c = self.cfg
        if sr < 0:
            return c.sr_missing_score
        spr = max(spr, 0.0)
        if spr > c.spr_th1:
            spr = (spr - c.spr_th1) * ((1.0 - c.spr_th1_out) / (1.0 - c.spr_th1)) + c.spr_th1_out
        elif spr > c.spr_th2:
            spr = (spr - c.spr_th2) * ((c.spr_th1_out - c.spr_th2_out) / (c.spr_th1 - c.spr_th2)) \
                + c.spr_th2_out
        if spr == 0:
            spr = c.spr_zero_fill
        return (sr + c.spr_coef * spr) / c.sr_spr_div

    def gen_features(self, x: ScoreInput, site_list: Sequence[str]) -> Dict[str, float]:
        site = self.parse_site(x.url)
        f: Dict[str, float] = {}
        f["sr"] = self.pre_sr(x.sr)
        f["spr"] = self.pre_spr(x.spr)
        f["spr_sr"] = self.sr_spr_score(f["sr"], f["spr"])
        f["dr"] = self.pre_dr(site)
        f["ow"] = self.pre_ow(site, site_list)
        f["adc"] = self.pre_adc(x.url, x.adc)
        return f

    # ---------- 分数调整（Java: adjustScore*） ----------

    def basic_score(self, f: Dict[str, float]) -> float:
        return sum(f[k] * w for k, w in self.cfg.fea_weight.items())

    def adjust_pc(self, url: str, pc: int, score: float) -> Tuple[float, str]:
        """返回 (分数, 命中的规则名)，规则名便于统计各规则命中率。"""
        c = self.cfg
        if (pc >> c.pc_floor_bit) & 1 == 1:
            return max(score, c.pc_floor), f"floor_bit{c.pc_floor_bit}"
        for rule in c.pc_rules:
            if rule.matches(pc, url):
                return score * rule.factor, rule.name
        return score, ""

    def adjust_text_len(self, pure_text_len: int, score: float) -> float:
        for threshold, factor in self.cfg.text_len_rules:
            if pure_text_len < threshold:
                return score * factor
        return score

    def adjust_adc(self, score: float, adc_new: float) -> float:
        bonus = self.cfg.adc_bonus.get(int(adc_new)) if adc_new == int(adc_new) else None
        return score + bonus if bonus else score

    # ---------- 主入口（Java: score） ----------

    def score(self, x: ScoreInput, site_list: Sequence[str]) -> ScoreResult:
        c = self.cfg
        f = self.gen_features(x, site_list)
        adc_new = f["adc"]
        s = self.basic_score(f)
        s *= decay_factor(x.pct, self.now, c.pct_decay_days)
        if adc_new < c.adc_skip_adjust_from:
            s, _rule = self.adjust_pc(x.url, x.pc, s)
            s *= decay_factor(x.pt, self.now, c.pt_decay_days)
        s = self.adjust_text_len(x.pure_text_len, s)
        s = self.adjust_adc(s, adc_new)
        return ScoreResult(score=s, features=f)
