#!/usr/bin/env python3
"""原始 PageValueScore / PageValueScoreMain（OCR 还原版）的逐行复刻，**保留全部已知缺陷**。

目的：拿真实表和真实输入在本地跑，得到与原 Spark 作业尽可能一致的分数，用来定位复现差异。
不要在这里修 bug、不要加特征；修复版与实验版在 ../npv_py 与 ../npv_lab。

保留的缺陷（与解读报告编号对应）：
  P0-1 preProcessPr 区间判断恒为假，pr 特征只有 0.01 / 1.0 两个值
  P0-2 spr 先在 preProcessSpr 除以 sprMax，再在 getSrSprScore 里除第二次
  P0-3 isScroll=1 时 sortByKey 用 Tuple2 作键，原作业会抛 ClassCastException（本文件默认同样报错，--assume-sort-works 可跳过）
  P0-4 pct/pt 长度不足 10 时按固定日期 2024-05-22 计算衰减
  P0-5 时间戳在未来时天数为负，衰减系数大于 1
  P0-6 任一行解析失败整个作业失败（默认同样直接抛错，--skip-bad-rows 可改为跳过）
  P0-7 sprMax 为 0 时产生 NaN / Infinity
  另：pt 只有 length>5 才衰减；每条记录各自取"当前时间"；名单文件里的空行原样参与匹配；
      后缀表按 Java HashMap 迭代顺序取第一个命中；特征 JSON 按 Java HashMap 顺序输出。

额外模拟的 Java 语义：float 为 32 位（每步运算后舍入）、String.split 丢弃尾部空列、
Long/Integer/Float.parseXxx 的判定规则、Double/Float.toString 与 fastjson 的数字格式、按日历日计算天数。

用法：
  python npv_exact.py --input in.tsv --spr spr.tsv --dr-site dr_site.tsv --dr-suffix dr_suffix.tsv \
      --ow ow.tsv --pr-split pr_split.tsv --ow-blacklist ow_black.txt --adc-whitelist adc_white.txt \
      --output out --region zh --scroll 0 [--now 1757030400] [--tz Asia/Shanghai] [--skip-bad-rows]
输出 out/npv_ori/part-00000（url, npv_ori, npv_fea）与 out/npv/part-00000（url, npv_ori, npv, npv_fea）。
"""
import argparse
import bisect
import json
import math
import os
import re
import struct
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

# ----------------------------------------------------------------------------
# Java 语义模拟
# ----------------------------------------------------------------------------


class JavaJobFailure(Exception):
    """对应原 Spark 作业因未捕获异常而失败的情形。message 里写明 Java 会抛的异常类型。"""


def f32(x):
    """把 double 舍入到 Java float（32 位）。"""
    return struct.unpack("f", struct.pack("f", x))[0]


def F(literal):
    """Java float 字面量，如 F("0.25") 对应 0.25f。"""
    return f32(float(literal))


def java_div(a, b):
    """IEEE 除法：除以 0 得 ±Infinity 或 NaN，不像 Python 那样抛 ZeroDivisionError。"""
    if b == 0:
        if a == 0 or a != a:
            return float("nan")
        return math.copysign(float("inf"), a) * math.copysign(1.0, b)
    return a / b


_LONG_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|NaN|Infinity)[fFdD]?$")


def java_parse_long(s):
    if s is None or not _LONG_RE.match(s):
        raise JavaJobFailure(f"NumberFormatException: Long.parseLong({s!r})")
    v = int(s)
    if not -(1 << 63) <= v < (1 << 63):
        raise JavaJobFailure(f"NumberFormatException: Long.parseLong({s!r}) 溢出")
    return v


def java_parse_int(s):
    if s is None:
        raise JavaJobFailure("NumberFormatException: Integer.parseInt(null)")
    if not _LONG_RE.match(s):
        raise JavaJobFailure(f"NumberFormatException: Integer.parseInt({s!r})")
    v = int(s)
    if not -(1 << 31) <= v < (1 << 31):
        raise JavaJobFailure(f"NumberFormatException: Integer.parseInt({s!r}) 溢出")
    return v


def _java_parse_decimal(s, what):
    if s is None:
        raise JavaJobFailure(f"NullPointerException: {what}(null)")
    t = s.strip()
    if not _FLOAT_RE.match(t):
        raise JavaJobFailure(f"NumberFormatException: {what}({s!r})")
    t = t.rstrip("fFdD")
    return float(t)


def java_parse_float(s):
    return f32(_java_parse_decimal(s, "Float.parseFloat"))


def java_parse_double(s):
    return _java_parse_decimal(s, "Double.parseDouble")


def java_split_tab(s):
    """Java String.split("\\t")：丢弃尾部空串；无分隔符时返回 [s]。"""
    if "\t" not in s:
        return [s]
    parts = s.split("\t")
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def java_string_hash(s):
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


def _hashmap_spread(h):
    h &= 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFFFFFF


def _table_size_for(cap):
    n = 1
    while n < cap:
        n <<= 1
    return n


def java_hashmap_order(keys_in_insertion_order, capacity=None):
    """Java HashMap 的迭代顺序：按桶下标升序，桶内按插入顺序（不模拟树化）。

    capacity 缺省按 new HashMap() 后 putAll(n 个) 的容量计算；featureList 那种逐个 put 的 7 个键用 16。
    桶内顺序在原作业里取决于 Scala HashMap 的顺序，这里用文件顺序近似。
    """
    keys = list(keys_in_insertion_order)
    if capacity is None:
        capacity = _table_size_for(int(len(keys) / 0.75 + 1.0)) if keys else 16
    return sorted(keys, key=lambda k: (_hashmap_spread(java_string_hash(k)) & (capacity - 1), keys.index(k)))


def _shortest_roundtrip(v, is_float32):
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 0:
        return "-0.0" if math.copysign(1, v) < 0 else "0.0"
    if is_float32:
        for p in range(1, 10):
            s = f"{v:.{p}g}"
            if f32(float(s)) == v:
                break
    else:
        s = repr(v)
    d = float(s)
    a = abs(d)
    if 1e-3 <= a < 1e7:
        if "e" in s or "E" in s:
            s = f"{d:.17g}" if not is_float32 else f"{d:.9g}"
            s = s.rstrip("0").rstrip(".") if "." in s else s
        if "." not in s:
            s += ".0"
        return s
    # Java 科学计数：d.dddE±n
    mant, exp = f"{d:.{16 if not is_float32 else 8}e}".split("e")
    if is_float32:
        for p in range(0, 9):
            m2 = f"{d:.{p}e}".split("e")[0]
            if f32(float(m2 + "e" + exp)) == v:
                mant = m2
                break
    else:
        for p in range(0, 17):
            m2 = f"{d:.{p}e}".split("e")[0]
            if float(m2 + "e" + exp) == d:
                mant = m2
                break
    if "." not in mant:
        mant += ".0"
    return f"{mant}E{int(exp)}"


def java_double_to_string(d):
    return _shortest_roundtrip(d, False)


def java_float_to_string(v):
    return _shortest_roundtrip(v, True)


def fastjson_number(v, is_float32):
    """fastjson 序列化 Float/Double：Xxx.toString() 后去掉结尾的 ".0"。"""
    s = java_float_to_string(v) if is_float32 else java_double_to_string(v)
    return s[:-2] if s.endswith(".0") else s


def fastjson_get_string(obj, key):
    v = obj.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return json.dumps(v)  # BigDecimal/Integer.toString 的近似
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _cast_number_string(s, what):
    t = s.replace(",", "")
    t = re.sub(r"\.0*$", "", t)
    if not _LONG_RE.match(t):
        raise JavaJobFailure(f"JSONException: {what} 无法把 {s!r} 转成数字")
    return int(t)


def fastjson_get_long(obj, key):
    """getLong：缺失或非法在原代码里被 try/catch 接住后取 0，这里返回 None 让调用方决定。"""
    v = obj.get(key)
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return _cast_number_string(v, "castToLong")
        except JavaJobFailure:
            return None
    return None


def fastjson_get_integer(obj, key):
    v = obj.get(key)
    if v is None or v == "":
        raise JavaJobFailure(f"NullPointerException: getInteger({key!r}) 为 null 后自动拆箱")
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        return _cast_number_string(v, "castToInt")
    raise JavaJobFailure(f"JSONException: getInteger({key!r}) 类型不支持")


def fastjson_get_double(obj, key):
    v = obj.get(key)
    if v is None or v == "":
        raise JavaJobFailure(f"NullPointerException: getDouble({key!r}) 为 null 后自动拆箱")
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return java_parse_double(v.replace(",", ""))
    raise JavaJobFailure(f"JSONException: getDouble({key!r}) 类型不支持")


def date_string_from_ms(ms):
    """SimpleDateFormat("yyyy-MM-dd").format(ms)，默认时区。超出 Python datetime 范围时用算术近似。"""
    try:
        return datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d")
    except (OverflowError, ValueError, OSError):
        days = ms // 86400000
        return f"__days__{days}"


def days_between_date_strings(a, b):
    """Joda Days.daysBetween(parse(a), parse(b))，默认时区按日历日。"""
    if a is None or b is None:
        raise JavaJobFailure("IllegalArgumentException: 日期字符串为 null")
    if a.startswith("__days__") or b.startswith("__days__"):
        da = int(a[8:]) if a.startswith("__days__") else _ordinal_days(a)
        db = int(b[8:]) if b.startswith("__days__") else _ordinal_days(b)
        return db - da
    return (datetime.strptime(b, "%Y-%m-%d").date() - datetime.strptime(a, "%Y-%m-%d").date()).days


def _ordinal_days(s):
    return (datetime.strptime(s, "%Y-%m-%d").date() - datetime(1970, 1, 1).date()).days


def parse_site(url):
    """原项目 ParseSiteUtil.parseSite 未知，这里取 host 小写去端口。复现不一致时首先核对此处。"""
    u = url.strip()
    if "://" not in u:
        u = "http://" + u
    try:
        host = urlparse(u).hostname
    except ValueError:
        host = None
    return (host or "").lower()


# ----------------------------------------------------------------------------
# 表加载（对应构造函数）
# ----------------------------------------------------------------------------


def _text_file_lines(path):
    """jsc.textFile(path).collect()：逐行，去掉行尾 \\r\\n，**保留空行**。"""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def load_map_tsv(path):
    """loadMapAndBcPathString：line.split("\\t")[0] / [1]，列数不足则 ArrayIndexOutOfBounds。"""
    items = []
    for line in _text_file_lines(path):
        cols = java_split_tab(line)
        if len(cols) < 2:
            raise JavaJobFailure(f"ArrayIndexOutOfBoundsException: {path} 中的行 {line!r} 不足两列")
        items.append((cols[0], cols[1]))
    m = {}
    for k, v in items:
        m[k] = v  # 重复键后者覆盖
    return m


class Tables:
    def __init__(self, spr_path, dr_site_path, dr_suffix_path, ow_path, pr_split_path,
                 ow_blacklist_path, adc_whitelist_path):
        self.dr_site = load_map_tsv(dr_site_path)
        self.dr_suffix = load_map_tsv(dr_suffix_path)
        self.dr_suffix_order = java_hashmap_order(list(self.dr_suffix))
        self.ow = load_map_tsv(ow_path)
        spr_list = [java_parse_float(java_split_tab(l)[1]) if len(java_split_tab(l)) > 1
                    else _oob(spr_path, l) for l in _text_file_lines(spr_path)]
        if not spr_list:
            raise JavaJobFailure("NoSuchElementException: Collections.max 于空列表")
        self.spr_max = max(spr_list)  # 可能为 0 或负数，原代码不检查
        self.pr_split = [java_parse_float(java_split_tab(l)[1]) if len(java_split_tab(l)) > 1
                         else _oob(pr_split_path, l) for l in _text_file_lines(pr_split_path)]
        self.ow_blacklist = _text_file_lines(ow_blacklist_path)   # 含空行
        self.adc_whitelist = _text_file_lines(adc_whitelist_path)  # 含空行：url.contains("") 恒为 true


def _oob(path, line):
    raise JavaJobFailure(f"ArrayIndexOutOfBoundsException: {path} 中的行 {line!r} 不足两列")


# ----------------------------------------------------------------------------
# 打分（对应 PageValueScore）
# ----------------------------------------------------------------------------

FEA_WEIGHT = (("spr_sr", 60), ("pr", 4), ("dr", 12), ("ow", 4))  # ImmutableMap 插入顺序
MAX_DR = 3
ONE_HUNDRED = 1000
FEATURE_JSON_ORDER = java_hashmap_order(["sr", "spr", "spr_sr", "dr", "ow", "pr", "adc"], capacity=16)


class PageValueScoreExact:
    def __init__(self, tables, now_seconds=None):
        self.t = tables
        self.now_seconds = now_seconds  # None 表示每条记录各取当前时间（原行为）
        self.feature_list = {}
        self.suffix_multi_match = 0  # 诊断：站点命中多个后缀的次数（此时结果依赖 HashMap 顺序）

    # --- preProcess* ---
    def pre_sr(self, sr):
        return f32(f32(float(sr)) / 100)

    def pre_spr(self, spr):
        return f32(java_div(f32(spr), self.t.spr_max))

    def pre_adc(self, url, adc):
        if adc > 0:
            return float(adc)
        for w in self.t.adc_whitelist:
            if w in url:
                return 3.0
        return float(adc)

    def pre_pr(self, pr):
        if not self.t.pr_split:
            raise JavaJobFailure("IndexOutOfBoundsException: prSplit 为空，prSplit.get(0) 失败（pr_split 文件至少要有一行）")
        if pr < self.t.pr_split[0]:
            return F("0.01")
        for idx in range(len(self.t.pr_split) - 1):
            if pr > self.t.pr_split[idx] and pr <= self.t.pr_split[idx]:  # 原样保留：恒为假
                return f32(idx / F("100"))
        return 1.0

    def pre_dr(self, url):
        site = parse_site(url)
        v = self.t.dr_site.get(site)
        if v is not None:
            return f32(java_parse_float(v) / MAX_DR)
        hits = [k for k in self.t.dr_suffix_order if site.endswith(k)]
        if len(hits) > 1:
            self.suffix_multi_match += 1
        if hits:
            return f32(java_parse_float(self.t.dr_suffix[hits[0]]) / MAX_DR)
        return f32(F("1.0") / MAX_DR)

    def pre_ow(self, url, site_list):
        site = parse_site(url)
        for s in site_list:
            if site.lower().endswith(s):
                return 1.0
        for b in self.t.ow_blacklist:
            if site.endswith(b):
                return 0.0
        if java_parse_float(self.t.ow.get(site, "0")) > 0:
            return 1.0
        return 0.0

    def get_sr_spr_score(self, sr, spr, spr_max):
        if sr >= 0:
            if spr < 0:
                spr = 0.0
            spr = f32(java_div(spr, spr_max))  # 第二次归一化（P0-2）
            th1 = F("0.36")
            th2 = F("0.2")
            if spr > th1:
                spr = f32(f32(f32(spr - th1) * f32(F("0.1") / f32(F("1.0") - th1))) + F("0.9"))
            elif spr > th2:
                spr = f32(f32(f32(spr - th2) * f32(F("0.7") / f32(th1 - th2))) + F("0.2"))
            if spr == 0:
                spr = F("0.1")
            return f32(f32(sr + f32(F("0.25") * spr)) / F("1.1"))
        return F("0.25")

    def gen_feature(self, url, sr, spr, pr, adc, site_list):
        fl = self.feature_list
        fl["sr"] = self.pre_sr(sr)
        fl["spr"] = self.pre_spr(spr)
        fl["spr_sr"] = self.get_sr_spr_score(fl["sr"], fl["spr"], self.t.spr_max)
        fl["dr"] = self.pre_dr(url)
        fl["ow"] = self.pre_ow(url, site_list)
        fl["pr"] = self.pre_pr(pr)
        fl["adc"] = self.pre_adc(url, adc)

    # --- 时间衰减（原样，含 2024-05-22 默认值与负天数） ---
    @staticmethod
    def time_stamp_2_date(t):
        if len(t) < 10:
            return "2024-05-22"
        ms = java_parse_long(t + "000")
        try:
            return date_string_from_ms(ms)
        except Exception:  # noqa: BLE001  原代码 catch 后返回 null
            return None

    @classmethod
    def time_diff(cls, t, now_str):
        return days_between_date_strings(cls.time_stamp_2_date(t), cls.time_stamp_2_date(now_str))

    @classmethod
    def down_factor(cls, t, now_str, scale):
        try:
            diff_days = cls.time_diff(t, now_str) * 1.0
        except Exception:  # noqa: BLE001
            diff_days = 1.0
        return 2 / (math.exp(diff_days / scale) + 1)

    def _now_str(self):
        now = self.now_seconds if self.now_seconds is not None else int(time.time())
        return str(now)

    def adjust_score_pct(self, pct, score):
        if len(pct) == 13:
            pct = pct[:10]
        return score * self.down_factor(pct, self._now_str(), 2000.0)

    def adjust_score_pt(self, pt, score):
        if len(pt) == 13:
            pt = pt[:10]
        return score * self.down_factor(pt, self._now_str(), 3000.0)

    # --- 其余调整（原样） ---
    @staticmethod
    def adjust_score_pc(url, pc, score):
        if (pc >> 20) & 1 == 1:
            return max(score, 60)
        if (pc >> 10) & 1 == 1:
            score *= 0.7
        elif (pc >> 19) & 1 == 1:
            score *= 0.8
        elif (pc >> 13) & 1 == 1:
            score *= 0.6
        elif (pc >> 18) & 1 == 1:
            score *= 0.9
        elif (pc >> 16) & 1 == 1:
            score *= 0.8
        elif (pc >> 33) & 1 == 1:
            score *= 0.7
        elif (pc >> 24) & 1 == 1:
            score *= 0.6
        elif (pc >> 38) & 1 == 1:
            score *= 0.6
        elif (pc >> 22) & 1 == 1:
            score *= 0.7
        elif ((pc >> 11) & 1 == 1) | ("news" in url.lower()):
            score *= 0.9
        elif ((pc >> 26) & 1 == 1) | ("/u/" in url) | ("/user/" in url):
            score *= 0.6
        elif (pc >> 29) & 1 == 1:
            score *= 0.9
        elif (pc >> 32) & 1 == 1:
            score *= 0.8
        return score

    @staticmethod
    def adjust_score_text_len(pure_text_len, score):
        if pure_text_len < 300:
            score = score * 0.8
        elif pure_text_len < 500:
            score = score * 0.9
        return score

    @staticmethod
    def adjust_score_adc(score, adc_new):
        if adc_new == 2:
            score += 10
        elif adc_new == 3:
            score += 15
        return score

    def get_basic_score(self, score):
        for fea, w in FEA_WEIGHT:
            score += f32(self.feature_list[fea] * w)  # Float * Integer 在 float 里算，再加到 double
        return score

    def score(self, url, pr, adc, pc, pct, pt, pure_text_len, sr, spr, site_list):
        self.feature_list = {}
        score = 0.0
        self.gen_feature(url, sr, spr, pr, adc, site_list)
        adc_new = self.feature_list["adc"]
        score = self.get_basic_score(score)
        score = self.adjust_score_pct(pct, score)
        if adc_new < 2:
            score = self.adjust_score_pc(url, pc, score)
            if len(pt) > 5:
                score = self.adjust_score_pt(pt, score)
        score = self.adjust_score_text_len(pure_text_len, score)
        score = self.adjust_score_adc(score, adc_new)
        return score

    def feature_json(self):
        return "{" + ",".join(f'"{k}":{fastjson_number(self.feature_list[k], True)}'
                              for k in FEATURE_JSON_ORDER) + "}"


# ----------------------------------------------------------------------------
# 归一化（对应 generateUrlInterval / normalizationScore）
# ----------------------------------------------------------------------------


def normal_cdf(x):
    """commons-math NormalDistribution(0,1).cumulativeProbability：0.5 * erfc(-x / sqrt2)。"""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def generate_url_interval(start, end, url_num):
    total = normal_cdf(end) - normal_cdf(start)
    intervals = [0]
    for i in range(ONE_HUNDRED, 0, -1):
        diff = normal_cdf(start + i * (end - start) / ONE_HUNDRED) \
            - normal_cdf(start + (i - 1) * (end - start) / ONE_HUNDRED)
        interval = int(math.ceil((diff / total) * url_num))
        intervals.append(interval + intervals[ONE_HUNDRED - i])
    return intervals


def normalization_score(rows, assume_sort_works):
    """rows: [(url, score, fea)]。返回 [(url, score, level, fea)]。"""
    if len(rows) >= 2 and not assume_sort_works:
        raise JavaJobFailure("ClassCastException: scala.Tuple2 cannot be cast to java.lang.Comparable "
                             "（原作业 isScroll=1 时 sortByKey 用 Tuple2 作键会在此失败；"
                             "若确认线上能跑通，加 --assume-sort-works 并请核对原环境的实现）")
    intervals = generate_url_interval(-2, 3, len(rows))
    # 假设 Tuple2 可比较：先按分数、再按 "url\tfea" 字符串，降序
    ordered = sorted(rows, key=lambda r: (r[1], r[0] + "\t" + r[2]), reverse=True)
    out = []
    for rank, (url, score, fea) in enumerate(ordered):
        idx = bisect.bisect_right(intervals, rank) - 1
        level = 1000 - idx if 0 <= idx < len(intervals) - 1 else 1001 - len(intervals)
        out.append((url, score, level, fea))
    return out


# ----------------------------------------------------------------------------
# 主流程（对应 PageValueScoreMain）
# ----------------------------------------------------------------------------


def parse_input_line(line):
    """getScoreRDD 里的解析，异常语义与 Java 一致（抛 JavaJobFailure）。"""
    s_line = java_split_tab(line)
    if len(s_line) < 6:
        raise JavaJobFailure(f"ArrayIndexOutOfBoundsException: 行只有 {len(s_line)} 列，需要 sLine[5]")
    url = s_line[0]
    try:
        obj = json.loads(s_line[5])
    except ValueError as e:
        raise JavaJobFailure(f"JSONException: 第 5 列不是合法 JSON（{e}）") from e
    if not isinstance(obj, dict):
        raise JavaJobFailure("JSONException: 第 5 列不是 JSON 对象")
    pr = fastjson_get_string(obj, "pr")
    adc = fastjson_get_string(obj, "adc")
    pc_long = fastjson_get_long(obj, "pcLong")
    pc_long = 0 if pc_long is None else pc_long
    pct = fastjson_get_long(obj, "pct")
    pct = 0 if pct is None else pct
    pt = fastjson_get_long(obj, "pt")
    pt = 0 if pt is None else pt
    pure_text_len = fastjson_get_integer(obj, "pureTextLen")
    sr = fastjson_get_integer(obj, "sr")
    spr = fastjson_get_double(obj, "spr")
    level = "0"
    if adc is None:
        raise JavaJobFailure("NullPointerException: adc 为 null 时调用 isEmpty()")
    if adc != "":
        try:
            adc_obj = json.loads(adc)
        except ValueError as e:
            raise JavaJobFailure(f"JSONException: adc 不是合法 JSON（{e}）") from e
        if not isinstance(adc_obj, dict):
            raise JavaJobFailure("JSONException: adc 不是 JSON 对象")
        level = fastjson_get_string(adc_obj, "level")
    if pr is None:
        raise JavaJobFailure("NullPointerException: pr 为 null 时调用 isEmpty()")
    if pr == "":
        pr = "0"
    return dict(url=url, flag=s_line[2], pr=java_parse_double(pr), level=java_parse_int(level),
                pc=pc_long, pct=str(pct), pt=str(pt), pure_text_len=pure_text_len, sr=sr, spr=spr)


def region_site_list(region_flag):
    if region_flag == "zh":
        return [".gov.cn", ".bendibao.com", ".edu.com", ".baike.com", "baike.baidu.com"]
    return [".gov", ".edu"]


def run(args):
    if args.tz:
        os.environ["TZ"] = args.tz
        time.tzset()
    tables = Tables(args.spr, args.dr_site, args.dr_suffix, args.ow, args.pr_split,
                    args.ow_blacklist, args.adc_whitelist)
    scorer = PageValueScoreExact(tables, args.now)
    site_list = region_site_list(args.region)
    hqw = {"1", "2", "5"}
    os.makedirs(os.path.join(args.output, "npv_ori"), exist_ok=True)
    stats = {"rows": 0, "bad_rows": 0, "npv_ori_rows": 0, "npv_rows": 0}
    hqw_rows = []
    with open(args.input, encoding="utf-8") as fin, \
            open(os.path.join(args.output, "npv_ori", "part-00000"), "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.rstrip("\r\n")
            try:
                r = parse_input_line(line)
                score = scorer.score(r["url"], r["pr"], r["level"], r["pc"], r["pct"], r["pt"],
                                     r["pure_text_len"], r["sr"], r["spr"], site_list)
            except JavaJobFailure as e:
                if args.skip_bad_rows:
                    stats["bad_rows"] += 1
                    continue
                raise JavaJobFailure(f"第 {lineno} 行导致作业失败：{e}\n行内容：{line[:300]}") from None
            stats["rows"] += 1
            fea = scorer.feature_json()
            if args.scroll == 1 and r["flag"] in hqw:
                hqw_rows.append((r["url"], score, fea))
            else:
                fout.write(f"{r['url']}\t{java_double_to_string(score)}\t{fea}\n")
                stats["npv_ori_rows"] += 1
    if args.scroll == 1:
        os.makedirs(os.path.join(args.output, "npv"), exist_ok=True)
        with open(os.path.join(args.output, "npv", "part-00000"), "w", encoding="utf-8") as fout:
            for url, score, level, fea in normalization_score(hqw_rows, args.assume_sort_works):
                fout.write(f"{url}\t{java_double_to_string(score)}\t{level}\t{fea}\n")
                stats["npv_rows"] += 1
    stats["spr_max"] = tables.spr_max
    stats["suffix_multi_match_rows"] = scorer.suffix_multi_match
    stats["empty_lines_in_adc_whitelist"] = sum(1 for l in tables.adc_whitelist if l == "")
    stats["empty_lines_in_ow_blacklist"] = sum(1 for l in tables.ow_blacklist if l == "")
    return stats


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--spr", required=True)
    p.add_argument("--dr-site", required=True)
    p.add_argument("--dr-suffix", required=True)
    p.add_argument("--ow", required=True)
    p.add_argument("--pr-split", required=True)
    p.add_argument("--ow-blacklist", required=True)
    p.add_argument("--adc-whitelist", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--region", default="")
    p.add_argument("--scroll", type=int, default=0)
    p.add_argument("--now", type=int, default=None,
                   help="固定'当前时间'（秒）。原作业每条记录各取一次当前时间，不传则同样如此")
    p.add_argument("--tz", default=None, help="模拟集群默认时区，如 Asia/Shanghai；不传用本机时区")
    p.add_argument("--skip-bad-rows", action="store_true", help="坏行跳过并计数（原作业会整体失败）")
    p.add_argument("--assume-sort-works", action="store_true",
                   help="isScroll=1 时假设 sortByKey 能执行（原作业会 ClassCastException）")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        stats = run(args)
    except JavaJobFailure as e:
        print(f"[作业失败，与原 Spark 作业行为一致] {e}", file=sys.stderr)
        sys.exit(1)
    for k, v in stats.items():
        print(f"{k}\t{v}", file=sys.stderr)


if __name__ == "__main__":
    main()
