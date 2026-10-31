"""拟合用的特征定义（25 列左右），fit_formula.py 与 sandbox/features.py 共用同一份，保证拟合与评估一致。

全部由 input.tsv 的 8 个字段与六张表算出。连续特征都缩放到大致 [0, 1]，方便比较权重大小。
"""
import math
import re
from urllib.parse import urlparse

PC_BITS = (10, 11, 13, 16, 18, 19, 20, 22, 24, 26, 29, 32, 33, 38)
UGC_SUBDOMAINS = {"bbs", "blog", "tieba", "zhidao", "wenku", "jingyan", "club", "forum", "space", "i", "my", "home", "user"}
LIST_PAT = re.compile(r"/(list|tag|tags|category|cat|forum|search|topic|index_\d+)(/|\.|$)|[?&](page|sort|order|p)=", re.I)
USER_PAT = re.compile(r"/(u|user|users|space|people|member|author|home|profile)/", re.I)
TEXT_BINS = (300, 500, 1500, 5000)

EXPECTED_SIGN = {  # 符号检查：拟合出相反符号的项要怀疑是在拟合噪声
    "spr_sr": +1, "sr": +1, "spr": +1, "dr": +1, "ow": +1, "adc": +1, "adc_ge2": +1, "adc_eq3": +1,
    "is_user_page": -1, "is_list_page": -1, "is_ugc_subdomain": -1, "pc_bit26": -1, "pc_bit10": -1, "pc_bit13": -1,
}


def _age_years(ts, now):
    try:
        t = int(float(ts))
    except (TypeError, ValueError):
        return None
    if t <= 0:
        return None
    if t > 10 ** 11:  # 毫秒
        t //= 1000
    return max(0.0, (now - t) / 86400.0 / 365.0)


def extra_features(scorer, x, site):
    u = x.url if "://" in x.url else "http://" + x.url
    try:
        pr = urlparse(u)
        path, query, host = pr.path or "/", pr.query or "", (pr.hostname or "").lower()
    except ValueError:
        path, query, host = "/", "", ""
    segments = [p for p in path.split("/") if p]
    sub = host.split(".")[0] if host.count(".") >= 2 else ""
    f = {
        # 页面级原始字段
        "text_len_log": math.log1p(max(x.pure_text_len, 0)) / 12.0,
        "pct_missing": 1.0 if _age_years(x.pct, scorer.now) is None else 0.0,
        "pt_missing": 1.0 if _age_years(x.pt, scorer.now) is None else 0.0,
        "pct_age": min((_age_years(x.pct, scorer.now) or 0.0), 10.0) / 10.0,
        "pt_age": min((_age_years(x.pt, scorer.now) or 0.0), 10.0) / 10.0,
        # adc 拆成两个 0/1
        "adc_ge2": 1.0 if x.adc >= 2 else 0.0,
        "adc_eq3": 1.0 if x.adc >= 3 else 0.0,
        # URL 形态
        "url_depth": min(len(segments), 8) / 8.0,
        "is_root": 1.0 if not segments else 0.0,
        "is_user_page": 1.0 if USER_PAT.search(path) else 0.0,
        "is_list_page": 1.0 if LIST_PAT.search(path + ("?" + query if query else "")) else 0.0,
        "has_query": 1.0 if query else 0.0,
        "is_ugc_subdomain": 1.0 if sub in UGC_SUBDOMAINS else 0.0,
        "leaf_is_number": 1.0 if segments and re.fullmatch(r"\d+(\.\w+)?", segments[-1]) else 0.0,
    }
    for i, edge in enumerate(TEXT_BINS):  # 正文长度分档：>= 边界 的累计指示，便于线性模型表达台阶
        f[f"text_ge_{edge}"] = 1.0 if x.pure_text_len >= edge else 0.0
    for bit in PC_BITS:
        f[f"pc_bit{bit}"] = float((x.pc >> bit) & 1)
    return f


def gen_features(scorer, x, site_list):
    """与 npv_lab/npv_ar 的 features.py 同签名：前 6 个为基线特征，其余为上面的扩展特征。"""
    site = scorer.parse_site(x.url)
    f = {}
    f["sr"] = scorer.pre_sr(x.sr)
    f["spr"] = scorer.pre_spr(x.spr)
    f["spr_sr"] = scorer.sr_spr_score(f["sr"], f["spr"])
    f["dr"] = scorer.pre_dr(site)
    f["ow"] = scorer.pre_ow(site, site_list)
    f["adc"] = scorer.pre_adc(x.url, x.adc)
    f.update(extra_features(scorer, x, site))
    return f
