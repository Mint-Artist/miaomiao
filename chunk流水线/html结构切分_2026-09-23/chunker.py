#!/usr/bin/env python3
"""简版 HTML → chunk 切分器（结构约束下的顺序装箱）。

输入是"简版HTML提取"项目产出的受控标签 HTML（h1–h6/p/ul/ol/li/table/
div.note/div.metadata/a/img/strong/em 等约二十个标签）。

管线（对应 chunk_research_2026-09-20 调研报告 §7）：
  解析 → 原子单元（段落/列表项/提示框/表格行）+ 标题层级
       → 上下文绑定（标题路径、表头、可选 metadata）
       → 预算内同结构作用域贪心装箱（章节硬边界）
       → 超长单元按类型降级（句→分句→子句→硬断，表格按行分组）
       → JSONL：text / contextual_text / 父子关系 / 锚点 / 拆分原因

设计决策（2026-09-23 与需求方确认）：
- 预算硬上限优先于元素完整性，冲突按类型降级，不做 LangChain 式整表保护；
- 上下文注入进 contextual_text 且计入预算；text 字段保持纯原文；
- 短章节不跨主题合并（merge_short_sections 实验开关，默认关）；
- 预算口径 256 Unicode 字符，不含图片 URL；预留 token 计数接口；
- LLM 上下文是可选增强层（llm_context 开关），第一版只留接口。

用法：
  python3 chunker.py --html 简版.html --doc-id gk100 --url https://... \
      --out chunks.jsonl --report report.json
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

CHUNKER_VERSION = '0.1.0'


# ---------------------------------------------------------------- 配置

@dataclass
class ChunkConfig:
    budget: int = 256                # Unicode 字符，含注入上下文
    hard_budget: bool = True         # 硬上限；False 时软上限（预留）
    merge_short_sections: bool = False   # 短章节跨标题合并（实验开关）
    pseudo_heading_boundary: bool = True  # 整段加粗短段落视为软边界（其前断块）
    context_title_path: bool = True      # 祖先标题路径注入
    context_table_header: bool = True    # 表格块重复表头
    context_metadata: bool = False       # 适用条件等 metadata 注入（默认关）
    llm_context: bool = False            # LLM 上下文增强（预留接口，v1 未接入）
    counter: str = 'char'                # char | token（token 预留）
    max_context_ratio: float = 0.4       # 上下文最长占预算比例，超出逐级丢深层标题

    def count(self, text):
        return len(text)               # token 计数器预留：此处替换

    def config_hash(self):
        blob = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- 原子单元

@dataclass
class Atom:
    atom_id: str
    kind: str                          # p / li / note / row / figure / quote / pre
    text: str                          # 纯原文正文（无上下文、无枚举前缀）
    scope: tuple                       # 结构作用域 = 当前标题路径
    data_src: list = field(default_factory=list)
    images: list = field(default_factory=list)
    table_header: str = ''             # 仅表格行：表头行文本
    parent_atom: str = ''              # 拆分时指向来源原子
    split_level: str = ''              # sentence/semicolon/clause/hard_wrap/list_item/table_cell
    continued: bool = False
    pseudo_heading: bool = False       # 整段加粗的短段落（伪标题），装箱时其前断块


# 句切分的标点层级：句号/问号/叹号 → 分号 → 逗号/顿号；都不行才硬断
SPLIT_LEVELS = [
    ('sentence', r'(?<=[。！？!?])'),
    ('semicolon', r'(?<=[；;])'),
    ('clause', r'(?<=[，,、])'),
]
CLOSERS = '》」』”’）】)]}'            # 闭合符号应跟在前一段末尾


def merge_closers(segs):
    """把以闭合符号开头的段并回前一段（“。”。” 不应把引号留给下一句）。"""
    out = []
    for seg in segs:
        if out and seg and seg[0] in CLOSERS:
            i = 0
            while i < len(seg) and seg[i] in CLOSERS:
                i += 1
            out[-1] += seg[:i]
            seg = seg[i:]
            if not seg:
                continue
        out.append(seg)
    return out


def greedy_pack(pieces, avail, sep=''):
    """按序装箱：累加到再加就超 avail 时闭合。返回 list[list]。"""
    groups, cur = [], []
    cur_len = 0
    for piece in pieces:
        add = len(piece) + (len(sep) if cur else 0)
        if cur and cur_len + add > avail:
            groups.append(cur)
            cur, cur_len = [], 0
            add = len(piece)
        cur.append(piece)
        cur_len += add
    if cur:
        groups.append(cur)
    return groups


def split_text(text, avail):
    """长文本降级链：整句 → 分句 → 子句 → 硬断。
    返回 list[(segment, level)]，level 为实际生效的拆分级别。"""
    if len(text) <= avail:
        return [(text, '')]
    for name, pattern in SPLIT_LEVELS:
        segs = merge_closers([s for s in re.split(pattern, text) if s])
        if len(segs) > 1:
            out = []
            for group in greedy_pack(segs, avail):
                joined = ''.join(group)
                if len(joined) <= avail:
                    out.append((joined, name))
                else:
                    out.extend(split_text(joined, avail))
            return out
    return [(text[i:i + avail], 'hard_wrap')
            for i in range(0, len(text), avail)]


# ---------------------------------------------------------------- 解析

def text_of(node):
    return ''.join(node.stripped_strings)


def src_of(node):
    return [node['data-src']] if node.has_attr('data-src') else []


def imgs_of(node):
    return [img['src'] for img in node.find_all('img') if img.get('src')]


def parse_atoms(soup, doc_title):
    """把简版 HTML 顶层元素流解析成原子序列，同时维护标题路径。"""
    atoms = []
    headings = []                      # [(level, text)] 当前生效的标题栈
    seq = 0

    def next_id(node):
        nonlocal seq
        seq += 1
        return node['data-src'] if node.has_attr('data-src') else f'a{seq}'

    def scope():
        return tuple(t for _, t in headings)

    def push_heading(level, text):
        while headings and headings[-1][0] >= level:
            headings.pop()
        headings.append((level, text))

    for el in soup.children:
        if not isinstance(el, Tag):
            continue
        if el.name == 'h1':
            continue                   # 文档标题不进入正文流
        if el.name == 'div' and 'metadata' in (el.get('class') or []):
            continue                   # metadata 只作上下文来源
        if re.fullmatch(r'h[2-6]', el.name or ''):
            push_heading(int(el.name[1]), text_of(el))
            continue
        if el.name == 'p':
            atom = Atom(next_id(el), 'p', text_of(el), scope(),
                        src_of(el), imgs_of(el))
            children = [c for c in el.children
                        if not (isinstance(c, NavigableString) and not str(c).strip())]
            if (len(children) == 1 and isinstance(children[0], Tag)
                    and children[0].name == 'strong'
                    and 0 < len(atom.text) < 40
                    and not atom.text.endswith(('。', '，', '；', '：'))):
                atom.pseudo_heading = True
            atoms.append(atom)
        elif el.name in ('blockquote', 'pre'):
            atoms.append(Atom(next_id(el), el.name if el.name != 'blockquote' else 'quote',
                              text_of(el), scope(), src_of(el), imgs_of(el)))
        elif el.name == 'figure':
            atoms.append(Atom(next_id(el), 'figure', text_of(el), scope(),
                              src_of(el), imgs_of(el)))
        elif el.name in ('ul', 'ol'):
            for li in el.find_all('li', recursive=False):
                atoms.append(Atom(next_id(li), 'li', text_of(li), scope(),
                                  src_of(li), imgs_of(li)))
        elif el.name == 'div' and 'note' in (el.get('class') or []):
            atoms.append(Atom(next_id(el), 'note', text_of(el), scope(),
                              src_of(el), imgs_of(el)))
        elif el.name == 'table':
            header_cells = [text_of(th) for th in el.select('thead th')]
            header = ' | '.join(header_cells)
            for tr in el.select('tbody tr'):
                cells = [text_of(c) for c in tr.find_all(['td', 'th'], recursive=False)]
                atoms.append(Atom(next_id(tr), 'row', ' | '.join(cells), scope(),
                                  src_of(tr), imgs_of(tr), table_header=header))
        else:                          # 简版 HTML 不应出现，兜底为段落
            if text_of(el):
                atoms.append(Atom(next_id(el), 'p', text_of(el), scope(),
                                  src_of(el), imgs_of(el)))
    return atoms


# ---------------------------------------------------------------- 上下文

def build_prefix(atom, doc_title, doc_metadata, cfg, stats):
    parts, sources = [], []
    if cfg.context_title_path:
        path = [doc_title] + [t for t in atom.scope if t != doc_title]
        # 上下文超长时逐级丢最深的标题，至少留文档标题
        while len(path) > 1 and len('\n'.join(path)) > cfg.budget * cfg.max_context_ratio:
            path.pop()
            stats['context_trimmed'] += 1
        parts.extend(path)
        sources.append('title_path')
    if cfg.context_table_header and atom.kind == 'row' and atom.table_header:
        parts.append('表头：' + atom.table_header)
        sources.append('table_header')
    if cfg.context_metadata and doc_metadata:
        parts.append(doc_metadata)
        sources.append('metadata')
    return '\n'.join(parts), sources


# ---------------------------------------------------------------- 超长降级

def degrade(atom, avail):
    """把超预算原子拆成不超 avail 的子原子序列。avail 为正文可用预算。"""
    if len(atom.text) <= avail:
        return [atom]
    pieces = []
    if atom.kind == 'row':
        # 单行超预算：拆最长单元格，其余单元格随首段保留、续段留空位
        cells = atom.text.split(' | ')
        longest = max(range(len(cells)), key=lambda i: len(cells[i]))
        cell_avail = avail - (len(atom.text) - len(cells[longest]))
        for seg, level in split_text(cells[longest], cell_avail):
            new_cells = list(cells)
            new_cells[longest] = seg
            pieces.append((' | '.join(new_cells), level or 'table_cell'))
    else:
        pieces = split_text(atom.text, avail)
        pieces = [(seg, level or 'sentence') for seg, level in pieces]
    out = []
    for i, (text, level) in enumerate(pieces):
        sub = Atom(atom.atom_id, atom.kind, text, atom.scope,
                   atom.data_src, atom.images, atom.table_header,
                   parent_atom=atom.atom_id,
                   split_level=level if i or len(pieces) > 1 else '',
                   continued=i > 0)
        sub._prefix = atom._prefix      # 继承来源原子的上下文
        sub._sources = atom._sources
        out.append(sub)
    return out


# ---------------------------------------------------------------- 装箱

def chunk_document(html, doc_id, url, cfg):
    soup = BeautifulSoup(html, 'html.parser')
    h1 = soup.find('h1')
    doc_title = text_of(h1) if h1 else doc_id
    meta = soup.select_one('div.metadata')
    doc_metadata = '\n'.join(text_of(p) for p in meta.find_all('p')) if meta else ''

    atoms = parse_atoms(soup, doc_title)
    stats = Counter()

    # 上下文绑定（上下文长度与作用域内 atom 无关，先算一次）
    prefixes, sources_map = {}, {}
    for atom in atoms:
        key = (atom.scope, atom.kind == 'row', atom.table_header)
        if key not in prefixes:
            prefixes[key], sources_map[key] = build_prefix(
                atom, doc_title, doc_metadata, cfg, stats)
        atom_prefix, atom_sources = prefixes[key], sources_map[key]
        atom._prefix = atom_prefix
        atom._sources = atom_sources

    # 超长降级：正文可用 = 预算 − 上下文 − 换行
    expanded = []
    for atom in atoms:
        avail = cfg.budget - (len(atom._prefix) + 1 if atom._prefix else 0)
        if avail < 10:
            stats['context_overflow'] += 1
            avail = cfg.budget
        before = len(expanded)
        expanded.extend(degrade(atom, avail))
        if len(expanded) - before > 1:
            stats[f'split:{expanded[before].split_level}'] += 1

    # 同结构作用域内贪心装箱
    chunks = []                        # 每块：list[Atom]
    cur = []

    def chunk_context(group):
        prefix = group[0]._prefix
        # 混合块：首原子是段落但块内含表格行时，补表头上下文
        if cfg.context_table_header and any(a.kind == 'row' for a in group):
            line = '表头：' + next(a.table_header for a in group
                                   if a.kind == 'row' and a.table_header)
            if line not in prefix:
                prefix = (prefix + '\n' + line) if prefix else line
        return prefix

    def chunk_sources(group):
        sources = list(group[0]._sources)
        if any(a.kind == 'row' for a in group) and 'table_header' not in sources:
            sources.append('table_header')
        return sources

    def body_len(group):
        return sum(len(a.text) for a in group) + max(0, len(group) - 1)

    def total_len(group):
        prefix = chunk_context(group)
        return body_len(group) + (len(prefix) + 1 if prefix else 0)

    def close():
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = []

    for atom in expanded:
        boundary = (cur and not cfg.merge_short_sections
                    and atom.scope != cur[0].scope)
        if boundary or (cur and cfg.pseudo_heading_boundary and atom.pseudo_heading):
            close()
        if cur and total_len(cur + [atom]) > cfg.budget:
            close()
        cur.append(atom)
    close()

    # 组装输出
    records = []
    atom_chunk = {}                    # atom_id → 该原子的首个 chunk 序号
    for i, group in enumerate(chunks):
        prefix = chunk_context(group)
        body = '\n'.join(a.text for a in group)
        contextual = (prefix + '\n' + body) if prefix else body
        kinds = {a.kind for a in group}
        first_atom = group[0].atom_id
        atom_chunk.setdefault(first_atom, i)
        split_levels = [a.split_level for a in group if a.split_level]
        records.append({
            'chunk_id': f'{doc_id}-C{i + 1:02d}',
            'doc_id': doc_id,
            'url': url,
            'text': body,
            'contextual_text': contextual,
            'chars': len(body),
            'contextual_chars': len(contextual),
            'context_sources': chunk_sources(group),
            'heading_path': list(group[0].scope),
            'parent_id': None,         # 续块在下一轮回填
            'continued': group[0].continued,
            'prev': None, 'next': None,
            'data_src': [s for a in group for s in a.data_src],
            'kind': kinds.pop() if len(kinds) == 1 else 'mixed',
            'split_reason': split_levels[0] if split_levels else None,
            'cross_section': len({a.scope for a in group}) > 1,
            'images': sorted({src for a in group for src in a.images}),
            'atom_ids': [a.atom_id for a in group],
            'chunker_version': CHUNKER_VERSION,
            'config_hash': cfg.config_hash(),
        })
    for i, rec in enumerate(records):
        rec['prev'] = records[i - 1]['chunk_id'] if i else None
        rec['next'] = records[i + 1]['chunk_id'] if i + 1 < len(records) else None
        first_atom = rec['atom_ids'][0]
        if rec['continued']:
            rec['parent_id'] = records[atom_chunk[first_atom]]['chunk_id']

    # 不变量统计（注意：Counter.update 对非 int 值会做加法，这里逐个赋值）
    atom_seq = [a.atom_id for a in atoms]
    chunk_seq = [aid for r in records for aid in
                 dict.fromkeys(r['atom_ids'])]  # 去重保持顺序
    stats['atoms'] = len(atoms)
    stats['expanded_atoms'] = len(expanded)
    stats['chunks'] = len(records)
    stats['coverage_ok'] = sorted(atom_seq) == sorted(set(chunk_seq))
    stats['order_ok'] = ([a for a in atom_seq if a in set(chunk_seq)]
                         == list(dict.fromkeys(chunk_seq)))
    stats['over_budget'] = sum(1 for r in records
                               if r['contextual_chars'] > cfg.budget)
    stats['cross_section_chunks'] = sum(1 for r in records if r['cross_section'])
    stats['continued_chunks'] = sum(1 for r in records if r['continued'])
    stats['avg_chars'] = round(sum(r['chars'] for r in records)
                               / max(1, len(records)), 1)
    stats['kind_counts'] = dict(Counter(r['kind'] for r in records))
    return records, stats


def build_report(records, stats, doc_id, url, cfg):
    return {
        'chunker_version': CHUNKER_VERSION,
        'doc_id': doc_id,
        'url': url,
        'config': asdict(cfg),
        'config_hash': cfg.config_hash(),
        'stats': dict(sorted(stats.items())),
        'chunks': [{'chunk_id': r['chunk_id'], 'kind': r['kind'],
                    'chars': r['chars'], 'contextual_chars': r['contextual_chars'],
                    'continued': r['continued'], 'split_reason': r['split_reason'],
                    'cross_section': r['cross_section']} for r in records],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--html', type=Path, required=True, help='简版 HTML 文件')
    ap.add_argument('--doc-id', required=True)
    ap.add_argument('--url', default='')
    ap.add_argument('--out', type=Path, help='JSONL 输出路径')
    ap.add_argument('--report', type=Path)
    ap.add_argument('--budget', type=int, default=256)
    ap.add_argument('--merge-short-sections', action='store_true')
    ap.add_argument('--context-metadata', action='store_true')
    ap.add_argument('--no-title-path', action='store_true')
    args = ap.parse_args()

    cfg = ChunkConfig(budget=args.budget,
                      merge_short_sections=args.merge_short_sections,
                      context_metadata=args.context_metadata,
                      context_title_path=not args.no_title_path)
    html = args.html.read_text(encoding='utf-8')
    records, stats = chunk_document(html, args.doc_id, args.url, cfg)
    lines = '\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(lines, encoding='utf-8')
    else:
        print(lines, end='')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = build_report(records, stats, args.doc_id, args.url, cfg)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8')
    import sys
    print(f"[{args.doc_id}] chunks={stats['chunks']} over_budget={stats['over_budget']} "
          f"coverage={stats['coverage_ok']} order={stats['order_ok']}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
