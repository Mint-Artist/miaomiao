"""从 original/ 里按名字抽取函数 / 常量源码（逐字），重新生成 chunk_ref 包。原版规则改动后运行：python build_ref.py"""
import ast, os, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(BASE, "original")
OUT = os.path.join(BASE, "chunk_ref")

def load(fn):
    src = open(os.path.join(ORIG, fn), encoding="utf-8").read()
    return src, ast.parse(src)

def grab(fn, names):
    src, tree = load(fn)
    found = {}
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        elif isinstance(node, ast.Try):  # BS_PARSER 的 try/except
            nm = "BS_PARSER" if "BS_PARSER" in ast.get_source_segment(src, node) else None
        if nm in names and nm not in found:
            seg = ast.get_source_segment(src, node)
            deco = "".join("@" + ast.get_source_segment(src, d) + "\n" for d in getattr(node, "decorator_list", []))
            found[nm] = deco + seg
    missing = [n for n in names if n not in found]
    assert not missing, (fn, missing)
    return "\n\n\n".join(found[n] for n in names)

HDR = '# 本文件由 build_ref.py 从 python/original/ 逐字抽取生成，不要手改；规则变动请改原版后重新生成。\n'
mods = {
 "config.py": ('"""全部参数（与 notebook 原版默认值一致）。"""\n',
               [("recall_pipeline_v2.py", ["SOFT", "HARD", "MIN_LEN", "TABLE_SOFT"]),
                ("chunk_adhere.py", ["ADH_MAX_LEN", "ADH_MAX_LINES", "ADH_LEAD_IN"])],
               ""),
 "text_utils.py": ('"""文本工具：空白清洗、分句、超长段落切分、伪标题判断、Markdown 行内清洗。"""\nimport re\nfrom typing import List\n',
               [("recall_pipeline_v2.py", ["WS", "SENT_END", "END_PUNCT", "clean_inline", "split_sentences", "split_long",
                                           "MD_HEADING", "MD_LIST", "MD_PIPE_SEP", "MD_IMAGE", "MD_LINK", "MD_EMPH",
                                           "MD_BOLD_LINE", "HTML_TAG", "md_inline", "is_pseudo_heading"])], ""),
 "atoms.py": ('"""结构单元（原结构切分 / 吸附版用）与 html 表格行提取。"""\nfrom dataclasses import dataclass\nfrom typing import List\n'
              'from bs4 import BeautifulSoup, NavigableString, Tag\nfrom .text_utils import clean_inline\n',
               [("recall_pipeline_v2.py", ["Atom", "table_rows_from_soup", "BS_PARSER", "H_TAGS", "SKIP_TAGS", "INLINE_TAGS"])], ""),
 "parse_md.py": ('"""Markdown → 结构单元。"""\nfrom typing import List\nfrom bs4 import BeautifulSoup\n'
                 'from .text_utils import *\nfrom .atoms import Atom, table_rows_from_soup, BS_PARSER\n',
               [("recall_pipeline_v2.py", ["parse_markdown"])], ""),
 "parse_html.py": ('"""html → 结构单元（原结构切分 / 吸附版用）。"""\nfrom typing import List\nfrom bs4 import BeautifulSoup, NavigableString, Tag\n'
                   'from .text_utils import *\nfrom .atoms import *\n',
               [("recall_pipeline_v2.py", ["parse_html"])], ""),
 "title_detect.py": ('"""类标题行（编号小标题）识别：标题吸附与容器级切分共用。"""\nimport re\nfrom .config import *\n',
               [("chunk_adhere.py", ["_STYLES", "_NOT_TITLE_END", "num_style", "title_like", "_tl_info"])], ""),
 "section_chunker.py": ('"""小节打包 + 相邻小节合并：md/html_struct_secmerge_t512 及其标题吸附版（_adh）。"""\nfrom collections import Counter\nfrom typing import List\n'
                        'from .config import *\nfrom .text_utils import split_long\nfrom .atoms import Atom\nfrom .title_detect import _tl_info\n',
               [("recall_pipeline_v2.py", ["chunk_struct_sections_t", "merge_sections", "chunk_secmerge_t"]),
                ("chunk_adhere.py", ["chunk_struct_sections_adh", "chunk_secmerge_adh"])], ""),
 "container_chunker.py": ('"""容器级结构切分：html_container（html 解析为容器单元 + 装块 + 合并）。"""\nfrom collections import Counter\nfrom dataclasses import dataclass, field\nfrom typing import List\n'
                          'from bs4 import BeautifulSoup, NavigableString, Tag\n'
                          'from .config import *\nfrom .text_utils import clean_inline, split_long, is_pseudo_heading\n'
                          'from .atoms import table_rows_from_soup, BS_PARSER, H_TAGS, SKIP_TAGS, INLINE_TAGS\nfrom .title_detect import _tl_info\n',
               [("chunk_container.py", ["CT_HARD", "CT_SPLIT", "CT_TINY", "CT_HEAD_MAX", "CT_LEAD", "CT_LEAD_MAX", "CT_LEAD_KINDS",
                                        "NOTE_CLASSES", "META_CLASSES", "CONTAINER_KINDS", "BLOCK_TAGS", "Unit", "block_text",
                                        "is_layout_table", "parse_html_blocks", "chunk_container_recs", "chunk_container"])], ""),
}
os.makedirs(OUT, exist_ok=True)
for fn, (doc, parts, tail) in mods.items():
    body = "\n\n\n".join(grab(src, names) for src, names in parts)
    open(os.path.join(OUT, fn), "w", encoding="utf-8").write(doc + HDR + "\n" + body + "\n" + tail)
    print("写入", fn)
