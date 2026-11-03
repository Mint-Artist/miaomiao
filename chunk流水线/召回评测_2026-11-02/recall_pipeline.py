# %% [markdown]
# # 切分 → 向量推理 → 召回评测
#
# 数据：云端若干分片 part-00000 …，每行一个 JSON：
#   {"url": ..., "content": EC 正文（段落以 <br> 分隔）, "markdown": Markdown 正文（表格以 HTML 保存）, "html": 正文 HTML}
#
# 流程（每段可单独运行；"定义"段只定义函数，"运行"段才真正干活）：
#   1. 配置
#   2–7. 定义：读数据、线上基线切分、通用工具、Markdown / HTML 解析、各切分方法
#   8. 运行：抽一条样例，打印各方法的切分结果
#   9. 运行：全量切分（多进程，按分片写出，每个文档一个 string list）
#   10–11. 定义 + 运行：加载你自己的向量模型，推理全部 chunk 向量（fp16，按分片存 .npy，可断点续跑，带百分比进度）
#   12. 定义：精确向量检索（GPU 分块暴力检索，不依赖 faiss）
#   13–14. 定义 + 运行：读 query、检索、计算召回指标、方法间配对显著性；导出候选池供大模型 / 精排 / 人工打分
#
# 产出目录（OUT_ROOT/方法名/）：
#   chunks/part-xxxxx.jsonl    每行 {"url": ..., "chunks": [str, ...]}，与输入分片逐行对应（没有 chunk 的文档也占一行）
#   chunks/part-xxxxx.urls.txt 每行一个 url（与上面逐行对应，检索时快速取 url）
#   chunks/part-xxxxx.lineidx.npy  每个文档那一行在 jsonl 里的字节偏移（检索命中后按需读取 chunk 文本）
#   emb/part-xxxxx.npy         [该分片 chunk 数, 维度] float16，顺序 = 按文档顺序展开的 chunk 顺序
#   emb/part-xxxxx.offsets.npy [文档数 + 1] int64，第 d 个文档的 chunk 在 [offsets[d], offsets[d+1]) 行
#   emb/part-xxxxx.lens.npy    每个 chunk 的字符数（算"按字数预算"的召回指标用）
#   emb/meta.json              模型路径、前缀、池化方式、维度等
#
# 分批入库（先推理随机网页库，之后再补入 query 相关网页）：
#   索引就是 emb/ 目录下所有已完成（有 .done）的分片，不需要单独的"合并"步骤。补入时把 DATA_GLOB 指向新数据、
#   PART_PREFIX 设为 "inj-"、DEDUP_URLS 设为 True，重跑第 9、11 段即可：新分片以 inj-part-xxxxx 的名字写到同一目录，
#   已在库里的 url 与新数据内部的重复 url 会被跳过；第 12 段检索时自动包含新旧全部分片。

# %% 1. 配置（按需修改）
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 先加载模型再 fork 多进程时，避免 tokenizers 的死锁警告

DATA_GLOB = "/path/to/dataset/part-*"          # 云端分片路径，支持 .gz
PART_PREFIX = ""                                # 输出分片名前缀。随机网页库用 ""；之后补入 query 相关网页时改成 "inj-"
                                                # （两批输入都叫 part-00000 …，不加前缀会被当成已完成而直接跳过）
DEDUP_URLS = False                              # 补入时设为 True：跳过库里已有的 url，以及这一批内部重复的 url
OUT_ROOT = "/path/to/output/chunk_eval"         # 所有产出的根目录
N_PROC = 32                                     # 切分用的进程数
TITLE_FIELD = None                              # 如果数据里有页面标题字段（如 "title"），填上：带上下文的切分会把它作为前缀

# 切分参数（字符数，与线上一致：软目标 256，硬上限 512，过滤 < 20 字的碎片）
SOFT = 256
HARD = 512
MIN_LEN = 20
CTX_MAX = 100                                   # 上下文前缀（标题路径）的最大字数

# 要比较的切分方法（定义见第 7 段 METHODS）
RUN_METHODS = ["content_br", "md_struct", "html_struct"]
# 暂不使用（定义仍保留在 METHODS 里，需要时加回上面即可）："md_fixed", "md_recursive", "md_struct_ctx", "html_struct_ctx"

# 向量模型
MODEL_PATH = "/path/to/your/embedding/model"    # 你自己的模型目录
LOADER = "transformers"                         # "transformers"（AutoModel + 池化）或 "sentence_transformers"
POOLING = "cls"                                 # 只对 LOADER="transformers" 生效："cls" / "mean" / "last"（decoder 类向量模型一般用 last）
                                                # LOADER="sentence_transformers" 时忽略，池化方式由模型目录里的 1_Pooling/config.json 决定
QUERY_PREFIX = ""                               # query 前缀，例如 "为这个句子生成表示以用于检索相关文章："
DOC_PREFIX = ""                                 # chunk 前缀
MAX_LENGTH = 512                                # 模型截断长度（token）
BATCH_SIZE = 256                                # 显存不够会自动减半重试
DEVICE = "cuda:0"
TRUST_REMOTE_CODE = True

# 多卡：同一份代码开多个 notebook / 进程，各自设 SHARD_ID = 0..NUM_SHARDS-1，按分片轮流分配
NUM_SHARDS = 1
SHARD_ID = 0

# 检索评测
QUERY_FILE = "/path/to/queries.jsonl"           # 每行 {"qid": ..., "query": ..., "relevant_urls": [...]}，格式见第 13 段
TOPK = 100
BASELINE_METHOD = "content_br"                  # 配对比较的基线

# %% 2. 定义：读数据
import glob
import gzip
import json
from urllib.parse import urlsplit


def list_parts(pattern=DATA_GLOB):
    return sorted(p for p in glob.glob(pattern) if not p.endswith((".crc", "_SUCCESS")) and os.path.isfile(p))


def part_name(path):
    base = os.path.basename(path)
    for ext in (".gz", ".jsonl", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return PART_PREFIX + base


def norm_url(u: str) -> str:
    """url 归一化（去协议、www、末尾斜杠），去重和评测都用它。"""
    try:
        s = urlsplit(u.strip())
        host = (s.netloc or "").lower()
        host = host[4:] if host.startswith("www.") else host
        return host + (s.path.rstrip("/") or "/") + ("?" + s.query if s.query else "")
    except Exception:
        return u.strip().lower()


def iter_records(path):
    """逐行读取一个分片；坏行跳过但计数（返回值里不出现）。"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                yield d


# %% 3. 定义：线上基线切分（按 content 的 <br> 切，原样保留，勿改）
from typing import List

SINGLE_CHUNK_MAX_LENGTH = 256


def split(ctt: str, single_chunk_max_length: int = SINGLE_CHUNK_MAX_LENGTH) -> List[str]:
    if not ctt or not ctt.strip():
        return []
    ctt_small = ctt.replace("<br>", "")
    if len(ctt_small) <= single_chunk_max_length:
        return _filter_valid([ctt_small])
    # 超长：按 <br> 切段，累加打包
    ctt_split = ctt.split("<br>")
    res: List[str] = []
    seg_list: List[str] = []
    sb: List[str] = []
    i = 0
    while i < len(ctt_split):
        seg = ctt_split[i]
        if seg == "":
            i += 1
            continue
        # V13：裸拼接，段间不补任何分隔符
        sb.append(seg)
        seg_list.append(seg)
        # 累加超过单 chunk 上限：结算当前 chunk
        if len("".join(sb)) > single_chunk_max_length:
            sb1_parts = [seg_list[0]]
            if len(seg_list) > 1:
                for j in range(1, len(seg_list) - 1):
                    # V13：结算时同样裸拼接
                    sb1_parts.append(seg_list[j])
                i -= 1  # 最后一段留到下一轮重新累加
            res.append("".join(sb1_parts))
            sb = []
            seg_list = []
        i += 1
    # 收尾：缓冲区剩余内容作为最后一个 chunk
    if seg_list:
        res.append("".join(sb))
    return _filter_valid(res)


def _filter_valid(res: List[str]) -> List[str]:
    """过滤掉空白和长度 < 20 的无意义短片段。"""
    return [s for s in res if s.strip() and len(s) >= 20]


# %% 4. 定义：通用工具（分句、超长段落切分、结构单元）
import re
from dataclasses import dataclass

WS = re.compile(r"[ \t\r\f\v 　]+")
SENT_END = re.compile(r"[。！？!?；;…]+[”’\"）)】》」』]*|\n+")
END_PUNCT = re.compile(r"[。！？!?；;，,、…]$")


def clean_inline(s: str) -> str:
    """合并行内空白，保留换行。"""
    lines = [WS.sub(" ", x).strip() for x in s.split("\n")]
    return "\n".join(x for x in lines if x)


def split_sentences(text: str, max_len: int) -> List[str]:
    """按句末标点与换行分句；单句超过 max_len 时先在逗号处切，再硬切。"""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        if m.end() > start:
            out.append(text[start:m.end()])
            start = m.end()
    if start < len(text):
        out.append(text[start:])
    res = []
    for s in out:
        while len(s) > max_len:
            cut = max(s.rfind("，", 0, max_len), s.rfind(",", 0, max_len), s.rfind("、", 0, max_len))
            cut = cut + 1 if cut > 0 else max_len
            res.append(s[:cut])
            s = s[cut:]
        if s.strip():
            res.append(s)
    return res


def split_long(text: str, soft: int) -> List[str]:
    """超过硬上限的段落：按句贪心打包到软目标以内。"""
    pieces, cur = [], ""
    for s in split_sentences(text, soft):
        if cur and len(cur) + len(s) > soft:
            pieces.append(cur)
            cur = ""
        cur += s
    if cur.strip():
        pieces.append(cur)
    return [p.strip() for p in pieces if p.strip()]


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


# %% 5. 定义：Markdown 解析 → 结构单元
from bs4 import BeautifulSoup

try:
    import lxml  # noqa: F401
    BS_PARSER = "lxml"
except ImportError:
    BS_PARSER = "html.parser"

MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
MD_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(.*)$")
MD_PIPE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_EMPH = re.compile(r"(\*\*|__)(.+?)\1")
MD_BOLD_LINE = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*$")
HTML_TAG = re.compile(r"<[^>]+>")


def md_inline(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = MD_IMAGE.sub(r"\1", s)
    s = MD_LINK.sub(r"\1", s)
    s = MD_EMPH.sub(r"\2", s)
    s = HTML_TAG.sub("", s)
    return clean_inline(s)


def is_pseudo_heading(t: str) -> bool:
    return 0 < len(t) <= 40 and not END_PUNCT.search(t)


def parse_markdown(md: str) -> List[Atom]:
    atoms: List[Atom] = []
    if not md:
        return atoms
    lines = md.replace("\r\n", "\n").split("\n")
    i, tid = 0, 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # HTML 表格（可能跨多行）
        if s.lower().startswith("<table"):
            buf = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
            i += 1
            soup = BeautifulSoup("\n".join(buf), BS_PARSER)
            for table in soup.find_all("table"):
                rows, header = table_rows_from_soup(table)
                for r in rows:
                    atoms.append(Atom("row", r, table_id=tid, header=header))
                tid += 1
            continue
        # 代码块
        if s.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            if "".join(buf).strip():
                atoms.append(Atom("code", "\n".join(buf)))
            continue
        # 管道表格
        if s.startswith("|") and i + 1 < len(lines) and MD_PIPE_SEP.match(lines[i + 1]):
            header = " | ".join(c.strip() for c in s.strip("|").split("|"))
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = " | ".join(md_inline(c) for c in lines[i].strip().strip("|").split("|"))
                atoms.append(Atom("row", row, table_id=tid, header=md_inline(header)))
                i += 1
            tid += 1
            continue
        m = MD_HEADING.match(s)
        if m:
            t = md_inline(m.group(2))
            if t:
                atoms.append(Atom("heading", t, level=len(m.group(1))))
            i += 1
            continue
        m = MD_BOLD_LINE.match(s)
        if m and is_pseudo_heading(md_inline(m.group(2))):
            atoms.append(Atom("heading", md_inline(m.group(2)), level=7))
            i += 1
            continue
        m = MD_LIST.match(line)
        if m:
            t = md_inline(m.group(1))
            if t:
                atoms.append(Atom("li", t))
            i += 1
            continue
        t = md_inline(s)
        if t:
            atoms.append(Atom("para", t))
        i += 1
    return atoms


# %% 6. 定义：HTML 解析 → 结构单元
from bs4 import NavigableString, Tag

H_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "head", "title", "meta", "link"}
INLINE_TAGS = {"a", "abbr", "b", "big", "cite", "code", "em", "font", "i", "img", "label", "mark", "q", "s", "small",
               "span", "strike", "strong", "sub", "sup", "time", "u", "br", "del", "ins"}


def parse_html(html: str) -> List[Atom]:
    atoms: List[Atom] = []
    if not html:
        return atoms
    soup = BeautifulSoup(html, BS_PARSER)
    for br in soup.find_all("br"):
        br.replace_with("\n")
    tid = [0]
    buf: List[str] = []

    def flush():
        t = clean_inline("".join(buf))
        buf.clear()
        if t:
            atoms.append(Atom("para", t))

    def own_text(el, exclude=("ul", "ol", "table")):
        parts = []
        for c in el.children:
            if isinstance(c, NavigableString):
                parts.append(str(c))
            elif isinstance(c, Tag) and c.name not in exclude and c.name not in SKIP_TAGS:
                parts.append(c.get_text())
        return clean_inline("".join(parts))

    def handle(c, depth):
        name = c.name
        if depth > 200:  # 极深嵌套直接取文字
            buf.append(c.get_text())
        elif name in H_TAGS:
            flush()
            t = clean_inline(c.get_text())
            if t:
                atoms.append(Atom("heading", t, level=H_TAGS[name]))
        elif name == "p":
            flush()
            t = clean_inline(c.get_text())
            kids = [k for k in c.children if not (isinstance(k, NavigableString) and not k.strip())]
            if t and len(kids) == 1 and isinstance(kids[0], Tag) and kids[0].name in ("strong", "b") and is_pseudo_heading(t):
                atoms.append(Atom("heading", t, level=7))  # 整段加粗的短行视为小标题
            elif t:
                atoms.append(Atom("para", t))
        elif name in ("ul", "ol"):
            flush()
            for li in c.find_all("li", recursive=False):
                t = own_text(li)
                if t:
                    atoms.append(Atom("li", t))
                for sub in li.find_all(["ul", "ol"], recursive=False):
                    handle(sub, depth + 1)
        elif name == "table":  # 嵌套表格的文字已包含在外层单元格里，不再单独处理
            flush()
            rows, header = table_rows_from_soup(c)
            for r in rows:
                atoms.append(Atom("row", r, table_id=tid[0], header=header))
            tid[0] += 1
        elif name == "pre":
            flush()
            t = c.get_text()
            if t.strip():
                atoms.append(Atom("code", t.strip("\n")))
        elif name in INLINE_TAGS:
            buf.append(c.get_text())
        else:  # div / section / blockquote 等容器：作为段落边界
            flush()
            walk(c, depth + 1)
            flush()

    def walk(node, depth=0):
        for c in node.children:
            if isinstance(c, NavigableString):
                buf.append(str(c))
            elif isinstance(c, Tag) and c.name not in SKIP_TAGS:
                handle(c, depth)

    walk(soup)
    flush()
    return atoms


# %% 7. 定义：切分方法
from functools import cached_property


def atoms_to_text(atoms: List[Atom]) -> str:
    """结构单元拼成纯文本（定长 / 递归切分用）：表格每行一行，表头只出现一次。"""
    out, last_table = [], None
    for a in atoms:
        if a.kind == "row" and a.table_id != last_table and a.header:
            out.append(a.header)
        last_table = a.table_id if a.kind == "row" else None
        out.append(a.text)
    return "\n".join(out)


def chunk_fixed(text: str, size: int = SOFT) -> List[str]:
    """定长切分（最朴素的对照）：去掉换行后每 size 字一块。"""
    t = text.replace("\n", "")
    return [t[i:i + size] for i in range(0, len(t), size)]


RECURSIVE_SEPS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")


def chunk_recursive(text: str, size: int = SOFT, seps=RECURSIVE_SEPS) -> List[str]:
    """递归字符切分（LangChain RecursiveCharacterTextSplitter 的做法，分隔符留在前一段末尾，无重叠）。"""
    sep = next((s for s in seps if s == "" or s in text), "")
    rest = seps[seps.index(sep) + 1:] if sep else ()
    if sep:
        raw = text.split(sep)
        splits = [p + sep for p in raw[:-1]] + [raw[-1]]
    else:
        splits = [text[i:i + size] for i in range(0, len(text), size)]
    out, good = [], []

    def merge():
        cur = ""
        for g in good:
            if cur and len(cur) + len(g) > size:
                out.append(cur)
                cur = ""
            cur += g
        if cur:
            out.append(cur)
        good.clear()

    for s in splits:
        if not s:
            continue
        if len(s) <= size:
            good.append(s)
        else:
            merge()
            out.extend(chunk_recursive(s, size, rest) if rest else [s[i:i + size] for i in range(0, len(s), size)])
    merge()
    return [c.strip() for c in out if c.strip()]


def chunk_struct(atoms: List[Atom], soft: int = SOFT, hard: int = HARD, with_context: bool = False,
                 title: str = "", ctx_max: int = CTX_MAX) -> List[str]:
    """结构切分（对应之前实验里效果最好的"结构类型保护"）：
    - 标题处断开，标题与它后面的第一段内容放在同一块（不会出现标题悬在块尾）；
    - 同一节内，同类单元（正文段落 / 列表项 / 同一张表的行）贪心打包到软目标；
    - 不超过硬上限的段落不拆；超过硬上限的段落按句切到软目标以内；
    - 表格按行打包，每块都带上表头行；
    - with_context=True 时，在块前加上标题路径（以及页面标题，若有），即之前实验里唯一稳定提升召回的做法。"""
    chunks, cur, cur_family = [], [], None
    cur_path: List[str] = []
    path: List[tuple] = []  # [(level, text)]
    pending: List[str] = []

    def flush():
        nonlocal cur, cur_family
        if cur:
            body = "\n".join(cur)
            if len(body.replace("\n", "")) >= MIN_LEN:
                if with_context:
                    ctx = [t for t in ([title] if title else []) + cur_path if t and t not in body[: len(t) + 2]]
                    prefix = ""
                    for t in reversed(ctx):  # 从最近的标题往上取，超长时丢最远的
                        cand = t + (" > " + prefix if prefix else "")
                        if len(cand) > ctx_max:
                            break
                        prefix = cand
                    chunks.append(prefix + "\n" + body if prefix else body)
                else:
                    chunks.append(body)
        cur, cur_family = [], None

    for a in atoms:
        if a.kind == "heading":
            flush()
            while path and path[-1][0] >= a.level:
                path.pop()
            path.append((a.level, a.text))
            pending.append(a.text)
            continue
        family = ("row", a.table_id) if a.kind == "row" else ("li",) if a.kind == "li" else ("text",)
        pieces = [a.text] if len(a.text) <= hard else split_long(a.text, soft)
        for piece in pieces:
            size = sum(len(x) for x in cur) + len(cur)
            if cur and (family != cur_family or size + len(piece) > soft):
                flush()
            add = piece
            if not cur:
                if a.kind == "row" and a.header:  # 表格的每一块都带表头
                    add = a.header + "\n" + add
                if pending:  # 标题与紧随的第一段放在同一块
                    add = "\n".join(pending) + "\n" + add
                cur_path = [t for _, t in path if t not in pending]
                cur_family = family
                pending = []
            cur.append(add)
    if pending:  # 页面以标题结尾
        cur.append("\n".join(pending))
    flush()
    return chunks


class Doc:
    """一条记录；解析结果按需计算并缓存，多个方法共用。"""

    def __init__(self, rec: dict):
        self.rec = rec
        self.title = (rec.get(TITLE_FIELD) or "") if TITLE_FIELD else ""

    @cached_property
    def md_atoms(self):
        return parse_markdown(self.rec.get("markdown") or "")

    @cached_property
    def html_atoms(self):
        return parse_html(self.rec.get("html") or "")

    @cached_property
    def md_text(self):
        return atoms_to_text(self.md_atoms)


def _min_len(chunks: List[str]) -> List[str]:
    return [c for c in chunks if c.strip() and len(c.replace("\n", "")) >= MIN_LEN]


METHODS = {
    # 线上基线：content 按 <br> 打包到 256
    "content_br": lambda d: split(d.rec.get("content") or ""),
    # Markdown：定长 / 递归 / 结构 / 结构 + 标题路径
    "md_fixed": lambda d: _min_len(chunk_fixed(d.md_text, SOFT)),
    "md_recursive": lambda d: _min_len(chunk_recursive(d.md_text, SOFT)),
    "md_struct": lambda d: chunk_struct(d.md_atoms, SOFT, HARD),
    "md_struct_ctx": lambda d: chunk_struct(d.md_atoms, SOFT, HARD, with_context=True, title=d.title),
    # HTML：结构 / 结构 + 标题路径（HTML 里的 h 标签、加粗段落比 Markdown 更完整）
    "html_struct": lambda d: chunk_struct(d.html_atoms, SOFT, HARD),
    "html_struct_ctx": lambda d: chunk_struct(d.html_atoms, SOFT, HARD, with_context=True, title=d.title),
}
# 想加别的块长对照（之前实验显示块长本身就会显著影响召回指标），例如：
# METHODS["md_struct_400"] = lambda d: chunk_struct(d.md_atoms, 400, HARD)


def chunk_record(rec: dict, methods=None) -> dict:
    d = Doc(rec)
    out = {}
    for m in methods or RUN_METHODS:
        try:
            out[m] = METHODS[m](d)
        except Exception:  # 单条出错不影响整批，记为空列表
            out[m] = []
    return out


# %% 8. 运行：抽一条样例看各方法的切分结果
_parts = list_parts()
print(f"{len(_parts)} 个分片，例如 {_parts[:3]}")
if _parts:
    _rec = next(iter_records(_parts[0]))
    print(_rec["url"])
    for _m, _cs in chunk_record(_rec).items():
        print(f"\n== {_m}: {len(_cs)} 块")
        for _i, _c in enumerate(_cs):
            print(f"[{_i}] ({len(_c)}字) {_c[:120]!r}")

# %% 9. 运行：全量切分（多进程；每个方法每个分片一个输出文件；已完成的分片自动跳过）
import numpy as np
from tqdm.auto import tqdm


def _empty_stats():
    return {"docs": 0, "chunks": 0, "chars": 0, "empty_docs": 0, "dup_skipped": 0}


DROP = {}  # {分片名: {要跳过的记录序号}}，由 prepare_dedup 填写；fork 出的子进程直接继承


def prepare_dedup(parts, method=None):
    """补入数据前调用：找出库里已有的 url，以及这一批内部重复的 url（只保留第一次出现）。
    只把要跳过的记录序号留给子进程，几百万 url 的集合用完即释放。"""
    global DROP
    method = method or RUN_METHODS[0]  # 各方法的文档集合相同，看一个方法的 urls.txt 就够
    have = set()
    for f in glob.glob(os.path.join(OUT_ROOT, method, "chunks", "*.urls.txt")):
        name = os.path.basename(f)[: -len(".urls.txt")]
        if (PART_PREFIX and name.startswith(PART_PREFIX)) or not os.path.exists(f[: -len(".urls.txt")] + ".done"):
            continue  # 本批已完成的分片不算"库里已有"，断点续跑时按本批内部去重处理
        with open(f, encoding="utf-8") as fu:
            have.update(norm_url(u) for u in fu.read().split("\n") if u)
    first, drop, n_old, n_dup, n_all = set(), {}, 0, 0, 0
    for path in tqdm(parts, desc="扫描 url 去重"):
        name = part_name(path)
        for i, rec in enumerate(iter_records(path)):
            n_all += 1
            u = norm_url(rec.get("url") or "")
            if not rec.get("url"):
                continue
            if u in have:
                n_old += 1
            elif u in first:
                n_dup += 1
            else:
                first.add(u)
                continue
            drop.setdefault(name, set()).add(i)
    DROP = drop
    print(f"库里已有 {len(have)} 个 url；本批 {n_all} 条，其中已在库里 {n_old} 条、本批内部重复 {n_dup} 条，"
          f"实际入库 {n_all - n_old - n_dup} 条")


def chunk_part(path: str, methods=None, out_root: str = OUT_ROOT) -> dict:
    methods = methods or RUN_METHODS
    name = part_name(path)
    paths = {m: os.path.join(out_root, m, "chunks", name) for m in methods}
    todo = [m for m in methods if not os.path.exists(paths[m] + ".done")]
    stats = {m: _empty_stats() for m in todo}
    if not todo:
        return {"part": name, "skipped": True, "stats": {}}
    files = {}
    for m in todo:
        os.makedirs(os.path.dirname(paths[m]), exist_ok=True)
        files[m] = (open(paths[m] + ".jsonl", "wb"), open(paths[m] + ".urls.txt", "w", encoding="utf-8"), [])
    drop = DROP.get(name, ())
    for i, rec in enumerate(iter_records(path)):
        if i in drop:
            for m in todo:
                stats[m]["dup_skipped"] += 1
            continue
        res = chunk_record(rec, todo)
        url = (rec.get("url") or "").replace("\n", " ")
        for m in todo:
            f, fu, idx = files[m]
            idx.append(f.tell())
            f.write((json.dumps({"url": url, "chunks": res[m]}, ensure_ascii=False) + "\n").encode("utf-8"))
            fu.write(url + "\n")
            s = stats[m]
            s["docs"] += 1
            s["chunks"] += len(res[m])
            s["chars"] += sum(len(c) for c in res[m])
            s["empty_docs"] += not res[m]
    for m in todo:
        f, fu, idx = files[m]
        f.close()
        fu.close()
        np.save(paths[m] + ".lineidx.npy", np.asarray(idx, dtype=np.int64))
        with open(paths[m] + ".done", "w") as fd:
            json.dump(stats[m], fd)
    return {"part": name, "skipped": False, "stats": stats}


def run_chunking(parts=None, n_proc: int = N_PROC):
    parts = parts if parts is not None else list_parts()
    if DEDUP_URLS:
        prepare_dedup(parts)
    total = {m: _empty_stats() for m in RUN_METHODS}
    import multiprocessing as mp
    with mp.get_context("fork").Pool(n_proc) as pool:  # 用 fork：子进程直接继承 notebook 里定义的函数
        for r in tqdm(pool.imap_unordered(chunk_part, parts), total=len(parts), desc="切分分片"):
            for m, s in r["stats"].items():
                for k, v in s.items():
                    total[m][k] += v
    for m in RUN_METHODS:  # 汇总所有分片（含之前已完成的）
        agg = _empty_stats()
        for done in glob.glob(os.path.join(OUT_ROOT, m, "chunks", "*.done")):
            for k, v in json.load(open(done)).items():
                agg[k] += v
        agg["chunks_per_doc"] = agg["chunks"] / max(agg["docs"], 1)
        agg["avg_chunk_chars"] = agg["chars"] / max(agg["chunks"], 1)
        json.dump(agg, open(os.path.join(OUT_ROOT, m, "chunk_stats.json"), "w"), ensure_ascii=False, indent=1)
        print(f"{m:18s} 文档 {agg['docs']:>9}  块 {agg['chunks']:>10}  块/文档 {agg['chunks_per_doc']:5.2f}  "
              f"平均块长 {agg['avg_chunk_chars']:6.1f}  无块文档 {agg['empty_docs']}")


run_chunking()

# %% 10. 定义：加载向量模型
import torch
import torch.nn.functional as F


class Encoder:
    def __init__(self, path=MODEL_PATH, device=DEVICE, loader=LOADER, pooling=POOLING, max_length=MAX_LENGTH,
                 dtype=torch.float16, normalize=True, trust_remote_code=TRUST_REMOTE_CODE):
        self.device, self.pooling, self.max_length, self.normalize, self.loader = device, pooling, max_length, normalize, loader
        if loader == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path, device=device, trust_remote_code=trust_remote_code)
            self.model.max_seq_length = max_length
            if str(device).startswith("cuda"):
                self.model.half()
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
            dt = dtype if str(device).startswith("cuda") else torch.float32
            self.model = AutoModel.from_pretrained(path, torch_dtype=dt, trust_remote_code=trust_remote_code).to(device).eval()
            self.dim = None

    @torch.inference_mode()
    def _batch(self, texts: List[str]) -> np.ndarray:
        if self.loader == "sentence_transformers":
            v = self.model.encode(texts, batch_size=len(texts), convert_to_tensor=True, normalize_embeddings=self.normalize,
                                  show_progress_bar=False)
            return v.float().cpu().numpy().astype(np.float16)
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"]
        if self.pooling == "cls":
            v = h[:, 0]
        elif self.pooling == "mean":
            m = mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "last":
            if self.tok.padding_side == "left":
                v = h[:, -1]
            else:
                v = h[torch.arange(h.size(0), device=h.device), mask.sum(1) - 1]
        else:
            raise ValueError(self.pooling)
        v = v.float()
        if self.normalize:
            v = F.normalize(v, dim=-1)
        return v.cpu().numpy().astype(np.float16)

    def encode(self, texts: List[str], prefix: str = "", batch_size: int = BATCH_SIZE, pbar=None) -> np.ndarray:
        """按长度从长到短排序后分批（减少 padding，且最长的批先跑，显存不够会立刻暴露），结果按原顺序返回。"""
        n = len(texts)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out, i, bs = None, 0, batch_size
        while i < n:
            idx = order[i:i + bs]
            try:
                v = self._batch([prefix + texts[j] for j in idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"显存不足，batch 降到 {bs}")
                continue
            if out is None:
                out = np.empty((n, v.shape[1]), dtype=np.float16)
                self.dim = v.shape[1]
            out[idx] = v
            i += len(idx)
            if pbar is not None:
                pbar.update(len(idx))
        return out if out is not None else np.zeros((0, self.dim or 0), dtype=np.float16)


encoder = Encoder()

# %% 11. 运行：推理 chunk 向量（按分片写 fp16 .npy；已完成的分片自动跳过；多卡时用 SHARD_ID / NUM_SHARDS 分工）


def read_chunks(chunk_jsonl: str):
    texts, offsets, urls = [], [0], []
    with open(chunk_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.extend(d["chunks"])
            offsets.append(len(texts))
            urls.append(d["url"])
    return texts, np.asarray(offsets, dtype=np.int64), urls


def embed_method(method: str, enc: Encoder, shard_id: int = SHARD_ID, num_shards: int = NUM_SHARDS, block: int = 200_000):
    cdir = os.path.join(OUT_ROOT, method, "chunks")
    edir = os.path.join(OUT_ROOT, method, "emb")
    os.makedirs(edir, exist_ok=True)
    names = sorted(os.path.basename(p)[: -len(".jsonl")] for p in glob.glob(os.path.join(cdir, "*.jsonl")))
    names = [n for k, n in enumerate(names) if k % num_shards == shard_id]
    todo = [n for n in names if not os.path.exists(os.path.join(edir, n + ".done"))]
    meta = {"model_path": MODEL_PATH, "loader": LOADER, "pooling": POOLING, "max_length": MAX_LENGTH,
            "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX, "dtype": "float16", "normalized": True}
    mpath = os.path.join(edir, "meta.json")
    if os.path.exists(mpath):  # 分批入库时，新旧向量必须来自同一模型、同一 chunk 前缀，否则不能放进同一个索引
        old = json.load(open(mpath))
        diff = {k: (old.get(k), meta[k]) for k in ("model_path", "loader", "pooling", "max_length", "doc_prefix")
                if old.get(k) != meta[k]}
        if diff:
            raise ValueError(f"[{method}] 与已有向量的配置不一致（旧值, 新值）：{diff}；确需更换模型请换一个 OUT_ROOT 重跑")
    total = sum(json.load(open(os.path.join(cdir, n + ".done")))["chunks"] for n in todo)
    print(f"[{method}] 分片 {len(names)} 个，待推理 {len(todo)} 个，共 {total} 个 chunk")
    with tqdm(total=total, desc=f"推理 {method}", unit="chunk", smoothing=0.05) as pbar:
        for n in todo:
            texts, offsets, _ = read_chunks(os.path.join(cdir, n + ".jsonl"))
            path = os.path.join(edir, n + ".npy")
            mm = None
            for s in range(0, len(texts), block):  # 分块推理、逐块写盘，内存占用与分片大小无关
                v = enc.encode(texts[s:s + block], prefix=DOC_PREFIX, pbar=pbar)
                if mm is None:
                    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), v.shape[1]))
                mm[s:s + len(v)] = v
            if mm is None:  # 空分片
                np.save(path, np.zeros((0, enc.dim or 1), dtype=np.float16))
            else:
                mm.flush()
                del mm
            np.save(os.path.join(edir, n + ".offsets.npy"), offsets)
            np.save(os.path.join(edir, n + ".lens.npy"), np.asarray([len(t) for t in texts], dtype=np.int32))
            open(os.path.join(edir, n + ".done"), "w").close()
    meta["dim"] = enc.dim
    json.dump(meta, open(mpath, "w"), ensure_ascii=False, indent=1)


for _m in RUN_METHODS:
    embed_method(_m, encoder)

# %% 12. 定义：精确向量检索（GPU 分块暴力检索 + 流式 top-k；评测切分方法时用精确检索，避免近似索引的误差混进比较）


class VectorIndex:
    def __init__(self, method: str, select=None):
        """索引 = emb/ 下所有已完成的分片（随机库 part-* 与补入的 inj-part-* 一起）。
        select：可选，按分片名筛选，例如 select=lambda n: not n.startswith("inj-") 只检索随机库。"""
        self.method = method
        self.edir = os.path.join(OUT_ROOT, method, "emb")
        self.cdir = os.path.join(OUT_ROOT, method, "chunks")
        self.names = sorted(os.path.basename(p)[: -len(".done")] for p in glob.glob(os.path.join(self.edir, "*.done")))
        self.names = [n for n in self.names if select is None or select(n)]
        self.mats = [np.load(os.path.join(self.edir, n + ".npy"), mmap_mode="r") for n in self.names]
        self.offsets = [np.load(os.path.join(self.edir, n + ".offsets.npy")) for n in self.names]
        self.lens = [np.load(os.path.join(self.edir, n + ".lens.npy")) for n in self.names]
        self.base = np.concatenate([[0], np.cumsum([len(m) for m in self.mats])]).astype(np.int64)
        self._urls, self._lineidx, self._files = {}, {}, {}
        print(f"[{method}] {len(self.names)} 个分片，{self.base[-1]} 个向量")

    def search(self, q: np.ndarray, topk: int = TOPK, device: str = DEVICE, block_rows: int = 500_000,
               q_batch: int = 1024, desc=None):
        """每次把 block_rows 行向量搬上 GPU，query 按 q_batch 一批打分并合并 top-k；
        打分矩阵最大 q_batch × block_rows（默认约 2GB），每块向量只从磁盘读一次。"""
        dt = torch.float16 if str(device).startswith("cuda") else torch.float32
        qt = torch.from_numpy(np.asarray(q)).to(device=device, dtype=dt)
        best_s = torch.full((len(q), topk), -float("inf"), device=device)
        best_i = torch.full((len(q), topk), -1, dtype=torch.long, device=device)
        jobs = [(p, s) for p, m in enumerate(self.mats) for s in range(0, len(m), block_rows)]
        for p, s in tqdm(jobs, desc=desc or f"检索 {self.method}", leave=False):
            xb = torch.from_numpy(np.ascontiguousarray(self.mats[p][s:s + block_rows])).to(device=device, dtype=dt)
            gbase = int(self.base[p]) + s
            for a in range(0, len(q), q_batch):
                sc = (qt[a:a + q_batch] @ xb.T).float()
                v, i = sc.topk(min(topk, sc.shape[1]), dim=1)
                allv = torch.cat([best_s[a:a + q_batch], v], 1)
                alli = torch.cat([best_i[a:a + q_batch], i + gbase], 1)
                top, pos = allv.topk(topk, dim=1)
                best_s[a:a + q_batch], best_i[a:a + q_batch] = top, alli.gather(1, pos)
            del xb
        return best_s.cpu().numpy(), best_i.cpu().numpy()

    def locate(self, gid: int):
        """全局行号 → (分片下标, 文档行号, 文档内第几个 chunk)。"""
        p = int(np.searchsorted(self.base, gid, side="right") - 1)
        r = gid - int(self.base[p])
        d = int(np.searchsorted(self.offsets[p], r, side="right") - 1)
        return p, d, r - int(self.offsets[p][d])

    def url(self, gid: int) -> str:
        p, d, _ = self.locate(gid)
        if p not in self._urls:
            with open(os.path.join(self.cdir, self.names[p] + ".urls.txt"), encoding="utf-8") as f:
                self._urls[p] = f.read().split("\n")
        return self._urls[p][d]

    def chunk_len(self, gid: int) -> int:
        p = int(np.searchsorted(self.base, gid, side="right") - 1)
        return int(self.lens[p][gid - int(self.base[p])])

    def text(self, gid: int) -> str:
        p, d, c = self.locate(gid)
        if p not in self._lineidx:
            self._lineidx[p] = np.load(os.path.join(self.cdir, self.names[p] + ".lineidx.npy"))
        if p not in self._files:
            self._files[p] = open(os.path.join(self.cdir, self.names[p] + ".jsonl"), "rb")
        f = self._files[p]
        f.seek(int(self._lineidx[p][d]))
        return json.loads(f.readline())["chunks"][c]


# %% 13. 定义：读 query、检索、计算指标、配对比较、导出候选池
# query 文件每行一个 JSON：
#   {"qid": "q1", "query": "……", "relevant_urls": ["http://...", ...]}                 # 只知道相关网页（点击 / 标注）
#   {"qid": "q2", "query": "……", "relevance": {"http://a": 2, "http://b": 1}}          # 有分级标注（如 GSB 打分映射）
# 两种都没有时只能导出候选池，再用大模型 / 精排模型 / 人工打分（见 export_pool 与 judged_metrics）。

KS = (1, 5, 10, 20, 50, 100)
CHAR_BUDGETS = (1000, 3000, 10000)  # 按"送进精排的字数"截断的召回，抵消不同切分方法块长不同的影响



def load_queries(path: str = QUERY_FILE):
    qs = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            rel = d.get("relevance") or {u: 1 for u in d.get("relevant_urls", [])}
            d["rel"] = {norm_url(u): g for u, g in rel.items() if g > 0}
            qs.append(d)
    return qs


def keep_answerable(qs, method=None):
    """只保留库里确实有的相关网页（补入时抓取失败、被过滤的网页没法被任何方法召回），没有任何相关网页在库里的 query 去掉。
    各方法的文档集合相同，看一个方法的 urls.txt 就够；某方法把网页切成 0 块属于该方法自身的失败，不在这里剔除。"""
    method = method or RUN_METHODS[0]
    have = set()
    for f in glob.glob(os.path.join(OUT_ROOT, method, "chunks", "*.urls.txt")):
        with open(f, encoding="utf-8") as fu:
            have.update(norm_url(u) for u in fu.read().split("\n") if u)
    n_rel = sum(len(q["rel"]) for q in qs)
    out = []
    for q in qs:
        rel = {u: g for u, g in q["rel"].items() if u in have}
        if rel:
            out.append({**q, "rel": rel})
    n_in = sum(len(q["rel"]) for q in out)
    print(f"相关网页 {n_rel} 个，其中在库里 {n_in} 个（{100 * n_in / max(n_rel, 1):.1f}%）；"
          f"query {len(qs)} 条 → 可评测 {len(out)} 条")
    return out


def retrieve(method: str, qvec: np.ndarray, topk: int = TOPK, select=None):
    idx = VectorIndex(method, select)
    scores, gids = idx.search(qvec, topk)
    return idx, scores, gids


def url_metrics(idx: VectorIndex, gids_row: np.ndarray, rel: dict) -> dict:
    """单个 query 的指标：网页级相关（与切分方法无关，所以不同切分方法可直接比较）。"""
    urls = [norm_url(idx.url(int(g))) for g in gids_row if g >= 0]
    lens = [idx.chunk_len(int(g)) for g in gids_row if g >= 0]
    r = {}
    first = next((i for i, u in enumerate(urls) if u in rel), None)
    r["MRR"] = 0.0 if first is None else 1.0 / (first + 1)
    for k in KS:
        got = set(urls[:k]) & set(rel)
        r[f"hit@{k}"] = float(bool(got))
        r[f"recall@{k}"] = len(got) / len(rel)
    cum = np.cumsum(lens) if lens else np.zeros(0)
    for b in CHAR_BUDGETS:
        n = max(1, int(np.searchsorted(cum, b, side="right"))) if len(cum) else 0
        r[f"hit@{b}字"] = float(bool(set(urls[:n]) & set(rel)))
    # 网页级 nDCG@10（去重后的网页排序，分级相关）
    seen, ranked = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ranked.append(u)
    dcg = sum((2 ** rel.get(u, 0) - 1) / np.log2(i + 2) for i, u in enumerate(ranked[:10]))
    ideal = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
    r["nDCG@10"] = dcg / ideal if ideal > 0 else 0.0
    return r


def paired_bootstrap(a: List[float], b: List[float], n: int = 2000, seed: int = 0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def evaluate(methods=None, qs=None, enc=None, topk: int = TOPK, baseline: str = BASELINE_METHOD, select=None):
    import pandas as pd
    methods = methods or RUN_METHODS
    qs = qs if qs is not None else load_queries()
    labeled = [q for q in qs if q["rel"]]
    print(f"query {len(qs)} 条，其中有相关网页标注的 {len(labeled)} 条")
    qvec = (enc or encoder).encode([q["query"] for q in qs], prefix=QUERY_PREFIX)
    per_q, hits = {}, {}
    for m in methods:
        idx, scores, gids = retrieve(m, qvec, topk, select)
        hits[m] = (idx, scores, gids)
        per_q[m] = [url_metrics(idx, gids[i], q["rel"]) for i, q in enumerate(qs) if q["rel"]]
    if not labeled:
        return None, hits
    keys = list(per_q[methods[0]][0].keys())
    summary = pd.DataFrame({m: {k: np.mean([r[k] for r in per_q[m]]) for k in keys} for m in methods}).T
    if baseline in per_q:
        for m in methods:
            for k in ("MRR", "hit@10", "recall@100", f"hit@{CHAR_BUDGETS[1]}字"):
                mean, lo, hi = paired_bootstrap([r[k] for r in per_q[m]], [r[k] for r in per_q[baseline]])
                summary.loc[m, f"Δ{k}"] = f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return summary, hits


def export_pool(hits: dict, qs, path: str, depth: int = 10):
    """每个 query 合并各方法 top-depth 的 chunk（按文本去重），写出待打分的候选池。
    每行 {"qid", "query", "chunk_id", "text", "url", "methods": {方法: 名次}}。"""
    import hashlib
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs):
            pool = {}
            for m, (idx, scores, gids) in hits.items():
                for rank, g in enumerate(gids[i][:depth]):
                    if g < 0:
                        continue
                    t = idx.text(int(g))
                    cid = hashlib.md5(t.encode("utf-8")).hexdigest()[:16]
                    e = pool.setdefault(cid, {"qid": q["qid"], "query": q["query"], "chunk_id": cid, "text": t,
                                              "url": idx.url(int(g)), "methods": {}})
                    e["methods"][m] = rank + 1
            for e in pool.values():
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


def judged_metrics(pool_path: str, judgments: dict, methods, depth: int = 10, good: int = 1):
    """候选池打分后的指标。judgments: {(qid, chunk_id): 分数}，分数可以是 0/1/2（不相关 / 部分相关 / 完全相关），
    来源可以是大模型、精排模型打分映射或人工。返回每个方法的 P@depth、nDCG@depth、good-hit@depth（top-depth 里至少一个分数 ≥ good）。"""
    import pandas as pd
    by_q = {}
    for line in open(pool_path, encoding="utf-8"):
        e = json.loads(line)
        by_q.setdefault(e["qid"], []).append(e)
    rows = {m: [] for m in methods}
    for qid, es in by_q.items():
        grades = {e["chunk_id"]: judgments.get((qid, e["chunk_id"]), 0) for e in es}
        ideal = sorted(grades.values(), reverse=True)[:depth]
        idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
        for m in methods:
            ranked = sorted((e["methods"][m], e["chunk_id"]) for e in es if m in e["methods"])
            gs = [grades[c] for _, c in ranked][:depth]
            dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gs))
            rows[m].append({f"P@{depth}": np.mean([g >= good for g in gs]) if gs else 0.0,
                            f"nDCG@{depth}": dcg / idcg if idcg > 0 else 0.0,
                            f"good-hit@{depth}": float(any(g >= good for g in gs))})
    return pd.DataFrame({m: pd.DataFrame(r).mean() for m, r in rows.items()}).T


# %% 14. 运行：评测
queries = keep_answerable(load_queries())
summary, hits = evaluate(RUN_METHODS, queries)
if summary is not None:
    _cols = ["MRR", "hit@1", "hit@10", "hit@100", "recall@100", f"hit@{CHAR_BUDGETS[1]}字", "nDCG@10"]
    _cols += [c for c in summary.columns if c.startswith("Δ")]
    print(summary[_cols].round(4).to_string())
# 导出候选池，交给大模型 / 精排 / 人工打分后，用 judged_metrics 计算：
export_pool(hits, queries, os.path.join(OUT_ROOT, "pool_top10.jsonl"), depth=10)
