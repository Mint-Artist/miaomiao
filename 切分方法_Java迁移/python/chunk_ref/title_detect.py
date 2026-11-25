"""类标题行（编号小标题）识别：标题吸附与容器级切分共用。"""
import re
from .config import *
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

_STYLES = [
    ("数字", re.compile(r"^\d{1,2}(?:[、)）]|[.．](?!\d))")),
    ("括号数字", re.compile(r"^[（(]\d{1,2}[)）]")),
    ("中文数字", re.compile(r"^[一二三四五六七八九十]{1,3}[、.．]")),
    ("括号中文数字", re.compile(r"^[（(][一二三四五六七八九十]{1,3}[)）]")),
    ("第X", re.compile(r"^第[一二三四五六七八九十百零\d]{1,3}[章节部分条步篇讲课]")),
    ("圈号", re.compile(r"^[①-⑳]")),
    ("步骤", re.compile(r"^(?:step|步骤)\s*\d{1,2}", re.I)),
]


_NOT_TITLE_END = set("。！？!?；;，,、…")


def num_style(text):
    t = text.strip()
    return next((name for name, p in _STYLES if p.match(t)), None)


def title_like(text):
    """类标题行返回 "编号标题" / "引导句"，否则 None。
    以冒号结尾的行：带编号（如"1、准备工作："）仍算编号标题；不带编号的引导句只在 ADH_LEAD_IN=True 时算。"""
    t = text.strip()
    if not t or len(t) > ADH_MAX_LEN:
        return None
    if t[-1] in "：:":
        if num_style(t):
            return "编号标题"
        return "引导句" if ADH_LEAD_IN else None
    if t[-1] in _NOT_TITLE_END:
        return None
    return "编号标题" if num_style(t) else None


def _tl_info(text):
    """(类型, 编号样式) 或 None；打包时逐段记录。"""
    k = title_like(text)
    return (k, num_style(text) if k == "编号标题" else None) if k else None
