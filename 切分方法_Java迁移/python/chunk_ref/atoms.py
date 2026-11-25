"""结构单元（原结构切分 / 吸附版用）与 html 表格行提取。"""
from dataclasses import dataclass
from typing import List
from bs4 import BeautifulSoup, NavigableString, Tag
from .text_utils import clean_inline
# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。

@dataclass
class Atom:
    """结构单元：heading（level 1–7，7 = 加粗短行这类伪标题）/ para / li / row（表格行，header = 该表的表头行文本）/ code。"""
    kind: str
    text: str
    level: int = 0
    table_id: int = -1
    header: str = ""


def table_rows_from_soup(table) -> (List[str], str):
    """表格 → 每行一个 "单元格 | 单元格" 字符串；有 <th> 表头行或 <thead> 时返回表头文本。合并单元格按出现位置展开为文本。"""
    rows, header = [], ""
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # 嵌套表格的行交给内层表格处理
        cells = [clean_inline(c.get_text("\n")).replace("\n", " ") for c in tr.find_all(["td", "th"], recursive=False)]
        if not any(cells):
            continue
        text = " | ".join(cells)
        is_head = tr.find_parent("thead") is not None or all(c.name == "th" for c in tr.find_all(["td", "th"], recursive=False))
        if is_head and not rows and not header:
            header = text
        else:
            rows.append(text)
    return rows, header


try:
    import lxml  # noqa: F401
    BS_PARSER = "lxml"
except ImportError:
    BS_PARSER = "html.parser"


H_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "head", "title", "meta", "link"}


INLINE_TAGS = {"a", "abbr", "b", "big", "cite", "code", "em", "font", "i", "img", "label", "mark", "q", "s", "small",
               "span", "strike", "strong", "sub", "sup", "time", "u", "br", "del", "ins"}
