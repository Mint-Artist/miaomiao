"""把一条记录的分数分解成每一步的贡献，与 PageValueScore.score 的流水线一一对应。"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from npv import PageValueScore, ScoreInput
from npv.timeutil import decay_factor


@dataclass
class Explanation:
    url: str
    features: Dict[str, float]
    contributions: Dict[str, float]      # 基础分中各特征的加权贡献
    basic: float
    steps: List[Tuple[str, float, float]] = field(default_factory=list)  # (步骤名, 系数或增量, 该步之后的分数)
    pc_rule: str = ""
    skipped_page_adjust: bool = False    # adc >= 阈值时跳过页面类别与 pt 衰减
    final: float = 0.0


def explain(scorer: PageValueScore, x: ScoreInput, site_list: Sequence[str]) -> Explanation:
    c = scorer.cfg
    f = scorer.gen_features(x, site_list)
    contrib = {k: f.get(k, 0.0) * w for k, w in c.fea_weight.items()}
    s = sum(contrib.values())
    e = Explanation(url=x.url, features=f, contributions=contrib, basic=s)

    fac = decay_factor(x.pct, scorer.now, c.pct_decay_days)
    s *= fac
    e.steps.append(("pct_decay", fac, s))

    adc_new = f["adc"]
    if adc_new < c.adc_skip_adjust_from:
        before = s
        s, rule = scorer.adjust_pc(x.url, x.pc, s)
        e.pc_rule = rule
        e.steps.append(("pc_" + (rule or "none"), (s / before) if before else 1.0, s))
        fac = decay_factor(x.pt, scorer.now, c.pt_decay_days)
        s *= fac
        e.steps.append(("pt_decay", fac, s))
    else:
        e.skipped_page_adjust = True
        e.steps.append(("page_adjust_skipped", 1.0, s))

    before = s
    s = scorer.adjust_text_len(x.pure_text_len, s)
    e.steps.append(("text_len", (s / before) if before else 1.0, s))

    before = s
    s = scorer.adjust_adc(s, adc_new)
    e.steps.append(("adc_bonus", s - before, s))
    e.final = s
    return e


def format_explanation(e: Explanation) -> str:
    lines = [f"url: {e.url}", "features:"]
    for k, v in e.features.items():
        w = e.contributions.get(k)
        lines.append(f"  {k:<14} {v:>10.4f}" + (f"   × w -> {w:>8.3f}" if w is not None else ""))
    lines.append(f"basic score: {e.basic:.4f}")
    for name, val, after in e.steps:
        kind = "+" if name == "adc_bonus" else "×"
        lines.append(f"  {name:<22} {kind} {val:>8.4f}   -> {after:.4f}")
    if e.skipped_page_adjust:
        lines.append("  (adc >= 阈值，页面类别与 pt 衰减被跳过)")
    lines.append(f"final: {e.final:.4f}")
    return "\n".join(lines)
