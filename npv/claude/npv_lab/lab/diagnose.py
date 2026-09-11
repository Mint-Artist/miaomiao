"""对一批真实数据做结构诊断：方差来源、规则命中、缺失率、站内分差。不改模型，只回答"该先改哪里"。"""
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from npv import PageValueScore, ScoreInput
from npv.timeutil import parse_timestamp_seconds

from .explain import explain
from .stats import between_within_variance, mean, quantile, quantiles, variance


def diagnose(scorer: PageValueScore, rows: Iterable[Tuple[ScoreInput, str]],
             site_list: Sequence[str], min_site_rows: int = 5) -> Dict:
    feats: Dict[str, List[float]] = defaultdict(list)
    contribs: Dict[str, List[float]] = defaultdict(list)
    basics: List[float] = []
    finals: List[float] = []
    by_site_final: Dict[str, List[float]] = defaultdict(list)
    by_site_basic: Dict[str, List[float]] = defaultdict(list)
    step_factors: Dict[str, List[float]] = defaultdict(list)
    rule_hits: Counter = Counter()
    missing: Counter = Counter()
    n = 0
    for x, _flag in rows:
        e = explain(scorer, x, site_list)
        n += 1
        site = scorer.parse_site(x.url)
        for k, v in e.features.items():
            feats[k].append(v)
        for k, v in e.contributions.items():
            contribs[k].append(v)
        basics.append(e.basic)
        finals.append(e.final)
        by_site_final[site].append(e.final)
        by_site_basic[site].append(e.basic)
        for name, val, _after in e.steps:
            key = "pc" if name.startswith("pc_") else name
            if key in ("pct_decay", "pt_decay", "text_len", "pc", "adc_bonus"):
                step_factors[key].append(val)
        rule_hits["skipped(adc>=%d)" % scorer.cfg.adc_skip_adjust_from if e.skipped_page_adjust
                  else (e.pc_rule or "none")] += 1
        if x.sr < 0:
            missing["sr<0"] += 1
        if x.spr <= 0:
            missing["spr<=0"] += 1
        if parse_timestamp_seconds(x.pct) < 0:
            missing["pct_missing"] += 1
        if parse_timestamp_seconds(x.pt) < 0:
            missing["pt_missing"] += 1
        if x.adc > 0:
            missing["adc>0"] += 1
        if e.features.get("dr") == 1.0 / scorer.cfg.max_dr and site not in scorer.t.dr_site:
            missing["dr_default"] += 1

    var_basic = variance(basics)
    contrib_share = {k: (variance(v) / var_basic if var_basic > 0 else float("nan"))
                     for k, v in contribs.items()}
    tot_f, between_f, within_f = between_within_variance(by_site_final)
    tot_b, between_b, within_b = between_within_variance(by_site_basic)

    big_sites = {s: v for s, v in by_site_final.items() if len(v) >= min_site_rows}
    within_spread = [quantile(v, 0.9) - quantile(v, 0.1) for v in big_sites.values()]
    global_spread = quantile(finals, 0.9) - quantile(finals, 0.1) if finals else float("nan")

    return {
        "rows": n,
        "sites": len(by_site_final),
        "sites_with_min_rows": len(big_sites),
        "final_quantiles": quantiles(finals),
        "basic_quantiles": quantiles(basics),
        "feature_quantiles": {k: quantiles(v) for k, v in feats.items()},
        "contribution_mean": {k: mean(v) for k, v in contribs.items()},
        "contribution_variance_share": contrib_share,
        "site_level_variance_share": {
            "final": between_f / tot_f if tot_f > 0 else float("nan"),
            "basic": between_b / tot_b if tot_b > 0 else float("nan"),
        },
        "within_site_spread_p90_p10": {
            "median_over_sites": quantile(within_spread, 0.5) if within_spread else float("nan"),
            "global": global_spread,
        },
        "step_factor_mean": {k: mean(v) for k, v in step_factors.items()},
        "step_factor_active_rate": {k: sum(1 for f in v if (f != 1.0 if k != "adc_bonus" else f != 0.0)) / len(v)
                                    for k, v in step_factors.items() if v},
        "pc_rule_hits": dict(rule_hits.most_common()),
        "missing_rates": {k: v / n for k, v in missing.items()} if n else {},
        "weights": dict(scorer.cfg.fea_weight),
    }


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def format_report(d: Dict) -> str:
    L = ["# NPV 结构诊断", ""]
    L.append(f"- 记录数 {d['rows']}，站点数 {d['sites']}，样本数 ≥ 阈值的站点 {d['sites_with_min_rows']}")
    L.append("")
    L.append("## 1. 分数来自哪里")
    L.append("")
    s = d["site_level_variance_share"]
    L.append(f"- 站点级解释的方差占比：最终分 **{s['final']:.1%}**，基础分 {s['basic']:.1%}"
             "（越接近 100% 说明同站页面几乎不区分）")
    w = d["within_site_spread_p90_p10"]
    L.append(f"- 站内分差（p90-p10 的站点中位数）{w['median_over_sites']:.2f}，全局分差 {w['global']:.2f}")
    L.append("")
    L.append("| 特征 | 权重 | 平均贡献 | 方差占比 |")
    L.append("| --- | --- | --- | --- |")
    for k in d["weights"]:
        L.append(f"| {k} | {d['weights'][k]} | {d['contribution_mean'][k]:.2f} | {d['contribution_variance_share'][k]:.1%} |")
    L.append("")
    L.append("## 2. 调整项")
    L.append("")
    L.append("| 步骤 | 平均系数/增量 | 生效比例 |")
    L.append("| --- | --- | --- |")
    for k, v in d["step_factor_mean"].items():
        L.append(f"| {k} | {v:.4f} | {d['step_factor_active_rate'].get(k, 0):.1%} |")
    L.append("")
    L.append("页面类别规则命中：")
    L.append("")
    for k, v in d["pc_rule_hits"].items():
        L.append(f"- {k}: {v} ({v / d['rows']:.1%})")
    L.append("")
    L.append("## 3. 缺失与兜底")
    L.append("")
    for k, v in sorted(d["missing_rates"].items()):
        L.append(f"- {k}: {v:.1%}")
    L.append("")
    L.append("## 4. 分布")
    L.append("")
    L.append("| 项 | p0 | p10 | p50 | p90 | p100 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for name, q in [("final", d["final_quantiles"]), ("basic", d["basic_quantiles"])] + \
            list(d["feature_quantiles"].items()):
        L.append(f"| {name} | " + " | ".join(_fmt(q[p]) for p in ("p0", "p10", "p50", "p90", "p100")) + " |")
    return "\n".join(L)
