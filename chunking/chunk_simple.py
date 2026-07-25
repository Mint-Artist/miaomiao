# -*- coding: utf-8 -*-
"""标题吸附 + 贪心装箱(<br> 分隔) —— 精简版

用法::

    from chunk_simple import chunk
    chunks = chunk(正文, max_size=256)   # -> list[str]
"""

import re

# (层级, 正则),层级越小越高。按语料实际情况增补。
_PATS = [(lv, re.compile(p)) for lv, p in [
    (1, r'^第[一二三四五六七八九十百零\d]+[章节篇]'),      # 第一章
    (1, r'^[一二三四五六七八九十]+[、.．]\s*\S'),          # 一、
    (2, r'^[（(][一二三四五六七八九十]+[)）]\s*\S'),       # (一)
    (2, r'^\d{1,2}[、.．]\s*\S'),                          # 1、
    (3, r'^[（(]\d{1,2}[)）]\s*\S'),                       # (1)
    (3, r'^\d{1,2}[）)]\s*\S'),                            # 1)
]]
_END = re.compile(r'[。！？；!?;]\s*$')


def _level(seg, max_head=40):
    """是标题返回层级,否则 None。三重防误判缺一不可:
    长度上限(防"1、物理类在705分…"标题正文同行)、
    句末标点(防"一、二等奖各奖励一名。")、编号前缀。
    """
    s = seg.strip()
    if not s or len(s) > max_head or _END.search(s):
        return None
    for lv, pat in _PATS:
        if pat.match(s):
            return lv
    return None


def chunk(ctt, max_size=256, sep=''):
    """输入含 <br> 的正文,返回标题吸附后的 chunk 字符串列表。

    不变量:任何一段正文所在的 chunk,一定带着它的标题路径 —— 这是消除
    "标题跟错数字"的根基。小节内部若需再切,标题会复制到每一片。
    """
    segs = [s for s in ctt.split('<br>') if s.strip()]
    if not segs:
        return []

    # 1) 标题栈 -> 划分小节 [(标题路径, [正文段…]), …]
    sections, stack, path, bodies = [], [], [], []
    for s in segs:
        lv = _level(s)
        if lv is None:
            bodies.append(s)
            continue
        if bodies:
            sections.append((path, bodies))
        while stack and stack[-1][0] >= lv:      # 同级或更深的标题出栈
            stack.pop()
        stack.append((lv, s))
        path, bodies = [t for _, t in stack], []
    if bodies or path:
        sections.append((path, bodies))

    # 2) 以"整个小节"为单位贪心装箱
    out, buf, blen, seen = [], [], 0, set()

    def flush():
        nonlocal buf, blen, seen
        if buf:
            out.append(sep.join(buf))
            buf, blen, seen = [], 0, set()

    for path, bodies in sections:
        def make(skip):
            """同一 chunk 内已出现过的标题不再重复(省 token,提高精度)。"""
            keep = [h for h in path if h not in skip]
            return sep.join(keep + bodies), keep

        whole, keep = make(seen)
        if len(whole) <= max_size:
            # 整节装得下:多个完整小节可以合并 —— 每段正文的标题都还在,不会错配
            if blen + len(whole) > max_size:
                flush()
                whole, keep = make(seen)      # flush 后 seen 已清空,重算
            buf.append(whole)
            blen += len(whole)
            seen.update(keep)
        else:
            # 整节太大:内部按段再切,每一片都重复标题路径
            flush()
            head = sep.join(path)
            piece, plen = list(path), len(head)
            for b in bodies:
                if plen + len(b) > max_size and len(piece) > len(path):
                    out.append(sep.join(piece))
                    piece, plen = list(path), len(head)
                piece.append(b)
                plen += len(b)
            out.append(sep.join(piece))
    flush()
    return out


if __name__ == '__main__':
    demo = (
        '2026安徽高考600分：历史类对应位次是3121，物理类对应位次是29273。<br>'
        '一、安徽2026高考一分一段表<br>'
        '2026年成绩将于6月25日公布，下面参考安徽2025年数据：<br>'
        '1、物理类<br>'
        '在705分及以上的有18人，600分对应29273名、500分对应136697名。<br>'
        '2、历史类<br>'
        '分数在671分及以上的考生有10人，600分对应3121名、500分对应39038名。<br>'
        '二、安徽高考位次排名对应大学<br>'
        '篇幅有限，本文仅展示2025年前1万名对应的大学名单（物理组）。'
    )
    for i, c in enumerate(chunk(demo, 256), 1):
        print(f'[{i}] len={len(c)}\n    {c}\n')
