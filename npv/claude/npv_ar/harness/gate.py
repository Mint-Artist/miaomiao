"""门禁：两道关。第一道看 train 交叉验证，第二道看 dev 的 Ladder 门槛与护栏。纯函数，便于测试。"""
import math


def stage1(cur_cv: float, best_cv: float, cur_cx: int, best_cx: int, cfg: dict):
    """返回 (是否通过, 原因)。通过的条件：CV 提升超过 cv_gain_min；或 CV 基本持平且更简单。"""
    if cur_cv is None or best_cv is None or math.isnan(cur_cv):
        return True, "无 CV 数据，直接进入 dev 检验"
    gain = cur_cv - best_cv
    if gain >= cfg["cv_gain_min"]:
        return True, f"CV +{gain:.4f}"
    if gain >= -cfg["cv_tol"] and cur_cx < best_cx:
        return True, f"CV 持平({gain:+.4f})且更简单({best_cx}->{cur_cx})"
    return False, f"CV {gain:+.4f} 未过门槛 {cfg['cv_gain_min']}"


def threshold(se_cluster: float, cfg: dict) -> float:
    if se_cluster is None or math.isnan(se_cluster):
        return cfg["eps_floor"]
    return max(cfg["eps_floor"], se_cluster)


def coarse(delta: float, eps: float) -> str:
    if delta < -eps:
        return "明显变差"
    if delta < eps * 0.5:
        return "持平"
    return "略好但未过门槛"


def guardrails(cur: dict, best: dict, cfg: dict):
    """返回未通过的护栏名列表。cur/best 含 hard_accuracy、buried_good_rate、new_site_accuracy、blind_accuracy（可缺）。"""
    tol = cfg["guard_tol"]
    fails = []

    def worse_low(k):  # 越高越好的指标
        a, b = cur.get(k), best.get(k)
        return a is not None and b is not None and not math.isnan(a) and not math.isnan(b) and a < b - tol

    def worse_high(k):  # 越低越好的指标
        a, b = cur.get(k), best.get(k)
        return a is not None and b is not None and not math.isnan(a) and not math.isnan(b) and a > b + tol

    for k in ("hard_accuracy", "new_site_accuracy", "blind_accuracy"):
        if worse_low(k):
            fails.append(k)
    for k in ("buried_good_rate", "top_bad_rate"):
        if worse_high(k):
            fails.append(k)
    return fails


def stage2(delta: float, se_cluster: float, cur: dict, best: dict, cur_cx: int, best_cx: int, cfg: dict):
    """返回 (status, reason, eps)。status ∈ keep / discard。"""
    eps = threshold(se_cluster, cfg)
    fails = guardrails(cur, best, cfg)
    if fails:
        return "discard", "护栏未过: " + ",".join(fails), eps
    if delta >= eps:
        return "keep", f"dev +{delta:.4f} ≥ 门槛 {eps:.4f}", eps
    if delta >= -eps * 0.5 and cur_cx < best_cx:
        return "keep", f"dev 持平({delta:+.4f})且更简单({best_cx}->{cur_cx})", eps
    return "discard", f"dev {coarse(delta, eps)}", eps
