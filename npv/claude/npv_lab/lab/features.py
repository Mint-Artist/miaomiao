"""实验特征生成。

自改进循环中，agent 只允许修改这个文件和 configs/*.json；打分流水线、评估、标签都不许动。

约定：
- 返回 dict，键是特征名，值是 float；
- 前 7 个键必须与基线一致（sr, spr, spr_sr, dr, ow, pr, adc），流水线的 adc 逻辑依赖 "adc"；
- 在 config.fea_weight 中没有出现的特征，权重视为 0，只记录不参与打分，因此可以放心添加；
- 新特征加进 fea_weight 之后才会影响分数。
"""
import math
from urllib.parse import urlparse

# 基线 pc 规则涉及的位，作为独立 0/1 特征暴露出来（报告里"页面类别改成独立特征"的第一步）
PC_BITS = (10, 11, 13, 16, 18, 19, 20, 22, 24, 26, 29, 32, 33, 38)


def baseline_features(scorer, x, site, site_list):
    """与 npv_py 基线完全一致的 7 个特征。不要改这段，改下面的 extra_features。"""
    f = {}
    f["sr"] = scorer.pre_sr(x.sr)
    f["spr"] = scorer.pre_spr(x.spr)
    f["spr_sr"] = scorer.sr_spr_score(f["sr"], f["spr"])
    f["dr"] = scorer.pre_dr(site)
    f["ow"] = scorer.pre_ow(site, site_list)
    f["pr"] = scorer.pre_pr(x.pr)
    f["adc"] = scorer.pre_adc(x.url, x.adc)
    return f


def extra_features(scorer, x, site):
    """实验特征。默认权重 0，不改变基线分数。"""
    u = x.url if "://" in x.url else "http://" + x.url
    try:
        path = urlparse(u).path or "/"
    except ValueError:
        path = "/"
    segments = [p for p in path.split("/") if p]
    f = {
        # URL 形态
        "url_depth": min(len(segments), 8) / 8.0,
        "is_root": 1.0 if not segments else 0.0,
        # 正文长度的连续版本（基线只有 300/500 两档阈值），归一到约 [0, 1]
        "text_len_log": math.log1p(max(x.pure_text_len, 0)) / math.log1p(100000),
        # 时间戳是否存在（衰减项本身在流水线里，这里只暴露"有没有"）
        "has_pct": 1.0 if str(x.pct).strip() not in ("", "0", "None") else 0.0,
        "has_pt": 1.0 if str(x.pt).strip() not in ("", "0", "None") else 0.0,
    }
    # 页面类别位一位一个特征
    for bit in PC_BITS:
        f[f"pc_bit{bit}"] = float((x.pc >> bit) & 1)
    return f


def gen_features(scorer, x, site_list):
    site = scorer.parse_site(x.url)
    f = baseline_features(scorer, x, site, site_list)
    f.update(extra_features(scorer, x, site))
    return f
