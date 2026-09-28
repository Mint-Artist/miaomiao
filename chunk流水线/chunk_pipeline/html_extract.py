#!/usr/bin/env python3
"""原始 HTML → 简版 HTML 提取器（当前维护版本）。

源自冻结的 2026-09-23/v0.1.0；v0.1.1 跳过被祖先 decompose 销毁的节点。

设计目标（对应 EC网页正文标签保留需求_简版.md 的接口约定）：
- 输出仍是单个 HTML 字符串，但保留标题层级、段落、列表、表格行列、
  图文位置、链接等语义结构，不把结构统一压成 <br>。
- 提取是固定、确定性的建库前置步骤；chunk 切分策略在下游另行迭代。
- 信息宁多勿少：rowspan/colspan/href/alt/start 等属性保留，
  供下游策略自行取舍；此阶段丢弃的信息下游无法找回。

用法：
  python3 -m chunk_pipeline.html_extract --html samples/xxx.html --url https://... \
      --out output/xxx.simplified.html --report output/xxx.report.json
可选：
  --profile NAME             强制使用某个站点配置（默认按 URL 自动匹配）
  --anchors                  给块级元素加 data-src 序号锚点（评估回溯/父子扩展用）
  --promote-pseudo-headings  把"整段加粗的短段落"提升为标题并标 data-origin="inferred"
                             （默认关闭：规范要求不新增原文不存在的小标题）

依赖：beautifulsoup4。报告文件记录 EXTRACTOR_VERSION 与命中的站点配置，
便于提取器修 bug 后按版本定位需要重跑的页面范围。
"""
import argparse
import json
import re
from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import urljoin

from bs4 import (BeautifulSoup, Comment, Doctype, NavigableString,
                 ProcessingInstruction, Tag)
from soupsieve import compile as compile_selector, SelectorSyntaxError

EXTRACTOR_VERSION = '0.1.2'


def validate_content_options(content_selector=None, content_fallback='strict'):
    if content_fallback not in ('strict', 'body'):
        raise ValueError('content_fallback must be strict or body')
    if content_selector is not None:
        if not content_selector.strip():
            raise ValueError('content_selector must not be empty')
        try:
            compile_selector(content_selector)
        except SelectorSyntaxError as exc:
            raise ValueError('Invalid content selector: ' + str(exc)) from exc

# ---------------------------------------------------------------- 规则表

# 保留的语义标签（白名单）
KEEP_TAGS = {
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'br', 'ul', 'ol', 'li',
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'colgroup', 'col',
    'figure', 'figcaption', 'blockquote', 'pre', 'code',
    'a', 'img', 'strong', 'em',
}

# 同义改名
RENAME_TAGS = {'b': 'strong', 'i': 'em'}

# 直接丢弃（内容一并删除）
DROP_TAGS = {
    'script', 'style', 'noscript', 'template', 'iframe', 'svg', 'canvas',
    'form', 'input', 'button', 'select', 'textarea', 'object', 'embed',
    'video', 'audio', 'link', 'meta', 'base',
    'nav', 'header', 'footer', 'aside',
}

# 行内容器：去标签但保留文字与子孙
UNWRAP_INLINE = {
    'span', 'font', 'u', 's', 'strike', 'del', 'ins', 'mark', 'small', 'big',
    'abbr', 'time', 'cite', 'q', 'sub', 'sup', 'label',
}

# 块级容器但默认无语义：拆壳，子孙上浮
UNWRAP_BLOCKS = {'section', 'article', 'main', 'center', 'hgroup', 'address'}

# div 只有 class 命中语义关键词时才保留外壳，class 归一化为该关键词
SEMANTIC_DIV_KEYWORDS = ['note', 'metadata', 'warning', 'tip', 'info', 'alert',
                         'caution', 'notice']
DIV_CLASS_ALIAS = {'tip': 'note', 'notice': 'note'}

# 噪声：id/class 词元命中即整块删除（广告、推广、推荐、分享等）
NOISE_TOKENS = {
    'ad', 'ads', 'advert', 'banner', 'promo', 'recommend', 'related',
    'share', 'comment', 'breadcrumb', 'sidebar', 'ymzy', 'tg',
}

# 每个标签允许保留的属性
ATTR_WHITELIST = {
    'a': {'href'},
    'img': {'src', 'alt'},
    'ol': {'start', 'reversed'},
    'li': {'value'},
    'td': {'rowspan', 'colspan', 'scope', 'headers'},
    'th': {'rowspan', 'colspan', 'scope', 'headers'},
    'col': {'span'},
}
EXTRA_ATTRS = {'data-src', 'data-origin'}  # 提取器自身添加的锚点/来源标记

VOID_TAGS = {'br', 'img', 'col', 'hr'}
INLINE_TAGS = {'a', 'strong', 'em', 'code', 'br', 'img'}  # 序列化时不换行
# td/th 内容按行内序列化；li/div/blockquote/figure 允许混合内容
ONE_LINE_CONTAINERS = {'td', 'th', 'tr', 'caption', 'figcaption'}

ANCHOR_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'tr',
               'table', 'figure', 'blockquote', 'pre', 'div']

# ---------------------------------------------------------------- 站点配置
# content：正文容器选择器，按序取第一个有足够文字的命中；
# metadata：每项 {selector, label?}，每个命中块变成一个 metadata 段落；
# strip：在清理前从正文容器里整块移除的选择器（如已移入 metadata 的信息栏）；
# noise：站点专属的噪声选择器（在通用词元规则之外追加）。
PROFILES = [
    {
        'name': 'huawei_support',
        'url_contains': ['consumer.huawei.com'],
        'title': ['#knowledgeTitle', 'h1'],
        'content': ["div.jd-content div[id^='body']", "div[id^='body']"],
        'metadata': [{'selector': '.s-tips'}],
        'strip': [],
        'noise': [],
    },
    {
        'name': 'gk100',
        'url_contains': ['gk100.com'],
        'title': ['.article-title', 'h1'],
        'content': ['.article article', '.article'],
        'metadata': [{'selector': '.article > p.text-12 > span', 'label': '作者'},
                     {'selector': '.article > p.text-12 > time', 'label': '发布时间'}],
        'strip': [],
        'noise': ['.showModuleYMZY', '[id$="YmzyTg"]'],
    },
    {
        'name': 'lvtu_zs',
        'url_contains': ['64365.com'],
        'title': ['h1', 'title'],
        'content': ['div.w810.fl > div:not([class])'],
        'metadata': [{'selector': '.mt15 div.s-cb.lh24 span'}],
        'strip': ['.mt15', '#btadv', '.statement-bar', '.detail-conts > div.mt20'],
        'noise': [],
    },
    {
        'name': 'generic',
        'url_contains': [],
        'title': ['h1', 'title'],
        'content': ['article', 'main', '[role="main"]', '#content', '.article',
                    '.content', '.post-body', '.entry-content'],
        'metadata': [],
        'strip': [],
        'noise': [],
    },
]


def pick_profile(url, forced=None):
    if forced:
        for p in PROFILES:
            if p['name'] == forced:
                return p
        raise ValueError(f'未知 profile: {forced}，可选：{[p["name"] for p in PROFILES]}')
    for p in PROFILES:
        if any(s in (url or '') for s in p['url_contains']):
            return p
    return PROFILES[-1]


def first_with_text(soup, selectors, min_chars=1):
    for sel in selectors:
        for node in soup.select(sel):
            if len(''.join(node.stripped_strings)) >= min_chars:
                return node, sel
    return None, None


# ---------------------------------------------------------------- 清理

def class_id_tokens(node):
    text = ' '.join(node.get('class', [])) + ' ' + (node.get('id') or '')
    return {t for t in re.split(r'[^a-z0-9]+', text.lower()) if t}


def semantic_div_class(node):
    for cls in node.get('class', []):
        low = cls.lower()
        for kw in SEMANTIC_DIV_KEYWORDS:
            if kw in low:
                return DIV_CLASS_ALIAS.get(kw, kw)
    return None


def normalize_whitespace(fragment):
    """压缩文本节点空白并去掉首尾空白；pre 内保持原样。
    中文排版里换行缩进本无语义；英文/数字内部的单个空格保留。"""
    for node in list(fragment.find_all(string=True)):
        if isinstance(node, (Comment, Doctype, ProcessingInstruction)):
            node.extract()
            continue
        if node.find_parent('pre'):
            continue
        text = re.sub(r'\s+', ' ', str(node)).strip()
        node.replace_with(NavigableString(text))


def clean_tree(root, noise_selectors, stats):
    noise_ids = set()
    for sel in noise_selectors:
        for n in root.select(sel):
            noise_ids.add(id(n))
    for tag in list(root.find_all(True)):
        # decompose() also destroys descendants retained in this traversal snapshot.
        if tag.attrs is None:
            continue
        if tag.name in DROP_TAGS or id(tag) in noise_ids \
                or class_id_tokens(tag) & NOISE_TOKENS:
            stats[f'dropped:{tag.name}'] += 1
            tag.decompose()
    # 拆壳需多轮（嵌套 div/span）；unknown 标签有内容就拆壳，空则删除
    changed = True
    while changed:
        changed = False
        for tag in list(root.find_all(True)):
            if tag.attrs is None:
                continue
            if tag.name in RENAME_TAGS:
                tag.name = RENAME_TAGS[tag.name]
                changed = True
            elif tag.name == 'div':
                cls = semantic_div_class(tag)
                if cls:
                    tag.attrs = {'class': [cls]}
                    stats[f'kept_div:{cls}'] += 1
                else:
                    tag.unwrap()
                    changed = True
            elif tag.name in UNWRAP_INLINE or tag.name in UNWRAP_BLOCKS:
                tag.unwrap()
                changed = True
            elif tag.name not in KEEP_TAGS:
                stats[f'unknown:{tag.name}'] += 1
                if ''.join(tag.stripped_strings).strip() or tag.find('img'):
                    tag.unwrap()
                else:
                    tag.decompose()
                changed = True
    # 属性白名单（保留提取器自加的 data-* 标记；div 的 class 已归一化为语义关键词）
    for tag in root.find_all(True):
        allowed = ATTR_WHITELIST.get(tag.name, set()) | EXTRA_ATTRS
        if tag.name == 'div':
            allowed = allowed | {'class'}
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in allowed}
    return stats


def normalize_table(doc, table):
    """重建 table 子结构：caption/colgroup + thead（全 th 行）+ tbody + tfoot。
    原始 HTML 常见 thead/tbody 顺序错乱、多 tbody、或没有 thead，这里统一。"""
    caption = table.find('caption', recursive=False)
    colgroup = table.find('colgroup', recursive=False)
    header_rows, body_rows, foot_rows = [], [], []
    for cont in [c for c in table.children
                 if isinstance(c, Tag) and c.name in ('thead', 'tbody', 'tfoot')]:
        rows = cont.find_all('tr', recursive=False)
        if cont.name == 'thead':
            header_rows.extend(rows)
        elif cont.name == 'tfoot':
            foot_rows.extend(rows)
        else:
            body_rows.extend(rows)
    body_rows.extend(table.find_all('tr', recursive=False))
    real_body = []
    for row in body_rows:
        cells = row.find_all(['td', 'th'], recursive=False)
        if cells and all(c.name == 'th' for c in cells):
            header_rows.append(row)
        else:
            real_body.append(row)
    for child in list(table.children):
        child.extract()
    if caption:
        table.append(caption)
    if colgroup:
        table.append(colgroup)
    if header_rows:
        thead = doc.new_tag('thead')
        table.append(thead)
        for r in header_rows:
            thead.append(r)
    tbody = doc.new_tag('tbody')
    table.append(tbody)
    for r in real_body:
        tbody.append(r)
    if foot_rows:
        tfoot = doc.new_tag('tfoot')
        table.append(tfoot)
        for r in foot_rows:
            tfoot.append(r)


def blockify_root(doc, root):
    """根级裸露的行内内容（文字、a、img、strong…）按连续段收进 <p>。
    li/td/th/div.note 等容器内部允许行内内容直接存在，不在此处理。"""
    groups, buffer = [], []

    def flush():
        if any(not (isinstance(n, NavigableString) and not str(n).strip())
               for n in buffer):
            groups.append(list(buffer))
        buffer.clear()

    for child in list(root.children):
        if isinstance(child, NavigableString) or child.name in INLINE_TAGS:
            buffer.append(child)
        else:
            flush()
            groups.append(child)
    flush()
    for item in groups:
        if isinstance(item, list):
            p = doc.new_tag('p')
            for n in item:
                p.append(n.extract())
            root.append(p)
        else:
            root.append(item.extract())


def prune_empty(root):
    """删除无文字且无 img/table 后代的空块；td/th/tr 保留（承载结构）。"""
    pruned = 0
    changed = True
    while changed:
        changed = False
        for tag in list(root.find_all(['p', 'li', 'div', 'figure', 'figcaption',
                                       'blockquote', 'pre',
                                       'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            if tag.attrs is None:
                continue
            if ''.join(tag.stripped_strings).strip() or tag.find(['img', 'table']):
                continue
            tag.decompose()
            pruned += 1
            changed = True
    return pruned


def add_anchors(root):
    for i, tag in enumerate(root.find_all(ANCHOR_TAGS), 1):
        tag['data-src'] = f'e{i}'
    return i if root.find_all(ANCHOR_TAGS) else 0


def promote_pseudo_headings(doc, root):
    """整段加粗且短（<40 字、不以句号结尾）的 p 提升为标题，
    级别取前文最近标题 +1；标 data-origin="inferred" 供下游区分原生与推断。"""
    promoted = 0
    last_level = 1
    for tag in root.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        if tag.name != 'p':
            last_level = int(tag.name[1])
            continue
        children = [c for c in tag.children
                    if not (isinstance(c, NavigableString) and not str(c).strip())]
        text = tag.get_text(strip=True)
        if (len(children) == 1 and isinstance(children[0], Tag)
                and children[0].name == 'strong' and text
                and len(text) < 40 and not text.endswith(('。', '，', '；', '：'))):
            level = min(last_level + 1, 6)
            new_h = doc.new_tag(f'h{level}')
            new_h['data-origin'] = 'inferred'
            new_h.append(NavigableString(text))
            tag.replace_with(new_h)
            promoted += 1
    return promoted


# ---------------------------------------------------------------- 序列化

def attrs_text(tag):
    parts = []
    for k, v in tag.attrs.items():
        if isinstance(v, list):
            v = ' '.join(v)
        if v is None or v == '':
            parts.append(f' {k}')
        else:
            parts.append(f' {k}="{escape(str(v), quote=True)}"')
    return ''.join(parts)


def serialize_inline(node):
    """序列化行内内容（文本、a/strong/em/img/br/code 及其内部）。"""
    if isinstance(node, NavigableString):
        return escape(str(node))
    if node.name in VOID_TAGS:
        return f'<{node.name}{attrs_text(node)}>'
    inner = ''.join(serialize_inline(c) for c in node.children)
    return f'<{node.name}{attrs_text(node)}>{inner}</{node.name}>'


def serialize_block(node, indent=0):
    """块级元素序列化：块换行缩进，行内内容保持连续不拆。"""
    pad = '  ' * indent
    if node.name in VOID_TAGS:
        return f'{pad}<{node.name}{attrs_text(node)}>\n'
    if node.name in ONE_LINE_CONTAINERS:
        if node.name == 'tr':
            cells = ''.join(serialize_block(c, 0).strip() + ' '
                            for c in node.children if isinstance(c, Tag)).rstrip()
            return f'{pad}<tr{attrs_text(node)}>{cells}</tr>\n'
        inner = ''.join(serialize_inline(c) for c in node.children)
        return f'{pad}<{node.name}{attrs_text(node)}>{inner}</{node.name}>\n'
    # 一般块：把子节点分成"行内连续段"和"块级子元素"
    pieces, buffer = [], []

    def flush():
        if buffer:
            pieces.append(('inline', ''.join(serialize_inline(n) for n in buffer)))
            buffer.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            if str(child).strip():
                buffer.append(child)
        elif child.name in INLINE_TAGS:
            buffer.append(child)
        else:
            flush()
            pieces.append(('block', child))
    flush()
    if not pieces:
        return ''
    if len(pieces) == 1 and pieces[0][0] == 'inline':
        return f'{pad}<{node.name}{attrs_text(node)}>{pieces[0][1]}</{node.name}>\n'
    out = f'{pad}<{node.name}{attrs_text(node)}>\n'
    for kind, payload in pieces:
        if kind == 'inline':
            out += f'{pad}  {payload}\n'
        else:
            out += serialize_block(payload, indent + 1)
    out += f'{pad}</{node.name}>\n'
    return out


def serialize(root_children):
    return ''.join(serialize_block(c) for c in root_children).rstrip() + '\n'


# ---------------------------------------------------------------- 主流程

def extract(html, url=None, profile_name=None, anchors=False,
            promote_pseudo=False, content_selector=None, content_fallback='strict'):
    validate_content_options(content_selector, content_fallback)
    profile = pick_profile(url, profile_name)
    soup = BeautifulSoup(html, 'html.parser')
    warnings = []

    title_node, title_sel = first_with_text(soup, profile['title'])
    title = ''
    if title_node:
        title = re.sub(r'\s+', ' ', title_node.get_text()).strip()
        if title_node.name == 'title':  # <title> 常带站点后缀，如 "… _ 华为官网"
            title = re.split(r'\s*[_|—-]\s*[^_|—-]*$', title)[0].strip()
    else:
        warnings.append('未找到标题')

    if content_selector is not None:
        root, content_sel = first_with_text(soup, [content_selector], min_chars=1)
        extraction_mode = 'explicit_selector'
    else:
        root, content_sel = first_with_text(soup, profile['content'], min_chars=20)
        extraction_mode = 'profile_selector'
    if root is None and content_fallback == 'body':
        root = soup.body if soup.body is not None else soup
        content_sel = 'body' if root is not soup else '[document]'
        extraction_mode = 'body_fallback'
        warnings.append('正文选择器未命中，回退清理整个 body/HTML 片段；可能包含导航等非正文，请抽样检查')
    if root is None:
        selectors = [content_selector] if content_selector is not None else profile['content']
        raise ValueError(f'未找到正文容器（profile={profile["name"]}, selectors={selectors}）；'
                         '可指定 --content-selector，或显式启用 --content-fallback body 后抽样检查')

    doc = BeautifulSoup('', 'html.parser')
    out_root = doc.new_tag('div')  # 临时容器，序列化时取其子节点
    doc.append(out_root)
    h1 = doc.new_tag('h1')
    h1.append(NavigableString(title))
    out_root.append(h1)

    meta_ps = []
    for rule in profile['metadata']:
        for node in soup.select(rule['selector']):
            text = ''.join(node.stripped_strings)
            if not text:
                continue
            if rule.get('label') and not text.startswith(rule['label']):
                text = f"{rule['label']}：{text}"
            meta_ps.append(text)
    if meta_ps:
        div = doc.new_tag('div')
        div['class'] = ['metadata']
        for text in meta_ps:
            p = doc.new_tag('p')
            p.append(NavigableString(text))
            div.append(p)
        out_root.append(div)

    # 正文子树复制到独立 fragment 再清理，避免改动原 soup
    fragment = BeautifulSoup(str(root), 'html.parser')
    # Fragment fallback must not turn document head/title into body content.
    if extraction_mode == 'body_fallback':
        for head in list(fragment.find_all('head')):
            if head.attrs is not None:
                head.decompose()
    stats = Counter()
    for sel in profile.get('strip', []):
        for node in fragment.select(sel):
            if node.attrs is None:
                continue
            node.decompose()
            stats[f'stripped:{sel}'] += 1
    normalize_whitespace(fragment)
    clean_tree(fragment, profile['noise'], stats)
    for table in fragment.find_all('table'):
        normalize_table(doc, table)
    blockify_root(doc, fragment)
    pruned = prune_empty(fragment)
    if pruned:
        stats['pruned_empty'] = pruned
    if extraction_mode == 'body_fallback' and not fragment.get_text(strip=True) and not fragment.find('img'):
        raise ValueError('正文回退后没有可用文字或图片；请检查 pg 是否为空、仅脚本或需要浏览器渲染')

    for child in list(fragment.children):
        if isinstance(child, NavigableString):
            continue
        out_root.append(child.extract())

    # 标题去重：正文第一块若与 h1 同文则删除
    # out_root 的直接子节点：h1, [div.metadata], blocks...
    blocks = [c for c in out_root.children if isinstance(c, Tag)]
    for candidate in blocks[1:]:
        if candidate.name == 'div' and 'metadata' in (candidate.get('class') or []):
            continue
        if candidate.name in ('h1', 'h2', 'h3') and \
                candidate.get_text(strip=True) == title:
            candidate.decompose()
            stats['dedup_title_heading'] = 1
        break

    # 相对链接/图片地址绝对化（有页面 URL 时）
    if url:
        for tag, attr in ([(t, 'href') for t in out_root.find_all('a')]
                          + [(t, 'src') for t in out_root.find_all('img')]):
            v = tag.get(attr)
            if v and not re.match(r'^[a-z]+://', v):
                tag[attr] = urljoin(url, v)

    n_anchors = add_anchors(out_root) if anchors else 0
    n_promoted = promote_pseudo_headings(doc, out_root) if promote_pseudo else 0

    tag_counts = Counter(t.name for t in out_root.find_all(True))
    text_chars = len(''.join(out_root.stripped_strings))
    for img in out_root.find_all('img'):
        if not img.get('alt'):
            warnings.append(f'img 缺 alt: {img.get("src", "")[:80]}')
    for a in out_root.find_all('a'):
        if not a.get('href'):
            warnings.append(f'a 缺 href: {a.get_text(strip=True)[:40]}')
    tables = []
    for t in out_root.find_all('table'):
        rows = t.find_all('tr')
        widths = {len(r.find_all(['td', 'th'], recursive=False)) for r in rows}
        tables.append({'rows': len(rows),
                       'header_rows': len(t.select('thead tr')),
                       'col_widths': sorted(widths)})
        if len(widths) > 1:
            warnings.append(f'table 各行单元格数不一致: {sorted(widths)}'
                            '（若有 rowspan/colspan 属正常）')
    if anchors:
        stats['anchors'] = n_anchors
    if promote_pseudo:
        stats['promoted_pseudo_headings'] = n_promoted

    report = {
        'extractor_version': EXTRACTOR_VERSION,
        'url': url,
        'profile': profile['name'],
        'content_selector': content_sel,
        'extraction_mode': extraction_mode,
        'content_fallback_used': extraction_mode == 'body_fallback',
        'title_selector': title_sel,
        'title': title,
        'text_chars': text_chars,
        'tag_counts': dict(sorted(tag_counts.items())),
        'tables': tables,
        'stats': dict(sorted(stats.items())),
        'warnings': warnings,
    }
    return serialize([c for c in out_root.children if isinstance(c, Tag)]), report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--html', type=Path, required=True, help='原始 HTML 文件')
    ap.add_argument('--url', help='页面 URL（用于匹配站点配置和绝对化链接）')
    ap.add_argument('--profile', help='强制站点配置名')
    ap.add_argument('--content-selector', help='显式正文 CSS 选择器')
    ap.add_argument('--content-fallback', choices=['strict', 'body'], default='strict')
    ap.add_argument('--anchors', action='store_true', help='块级元素加 data-src 锚点')
    ap.add_argument('--promote-pseudo-headings', action='store_true',
                    help='加粗短段落提升为标题（标 data-origin=inferred）')
    ap.add_argument('--out', type=Path, help='简版 HTML 输出路径')
    ap.add_argument('--report', type=Path, help='JSON 报告输出路径')
    args = ap.parse_args()

    html = args.html.read_text(encoding='utf-8')
    simplified, report = extract(html, url=args.url, profile_name=args.profile,
                                 anchors=args.anchors,
                                 promote_pseudo=args.promote_pseudo_headings,
                                 content_selector=args.content_selector, content_fallback=args.content_fallback)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(simplified, encoding='utf-8')
    else:
        print(simplified)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8')
    print(f"[{report['profile']}] {report['text_chars']} chars, "
          f"tags={report['tag_counts']}, warnings={len(report['warnings'])}",
          file=__import__('sys').stderr)


if __name__ == '__main__':
    main()
