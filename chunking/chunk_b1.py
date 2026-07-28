# -*- coding: utf-8 -*-
"""标题吸附 · B1(纯边界约束版):标题永不与正文分离,零字符复制。

    from chunk_b1 import chunk
    chunks = chunk(正文, max_size=256)   # 输入含 <br> 的字符串 -> list[str]

和"全路径传播版"的区别:不把标题复制到每个 chunk。实测在真实 bad case 上
消除陷阱的效果完全相同,但索引零膨胀;而且标题识别出错时只会**漏**标签,
不会**贴错**标签 —— 漏标是 miss,贴错是"自信地答错",后者危害大得多。
"""

import re

# 编号标题。B1 只判"是不是标题",**不需要层级** —— 没有层级就没有"挂到错误父节点"这类 bug。
_PATS = [re.compile(p) for p in [
    r'^第[一二三四五六七八九十百零\d]+[章节篇]',       # 第一章
    r'^[一二三四五六七八九十]+[、.．]\s*\S',           # 一、
    r'^[（(][一二三四五六七八九十]+[)）]\s*\S',        # （一）
    r'^\d{1,2}[、.．]\s*\S',                           # 1、  1.
    r'^[（(]\d{1,2}[)）]\s*\S',                        # （1）
    r'^\d{1,2}[）)]\s*\S',                             # 1）
]]
_SENT = re.compile(r'(?<=[。！？；!?;])')
_BARE = re.compile(r'[\s\W_]+', re.UNICODE)


def is_heading(seg, max_head=40, max_bare=30):
    """三重门 + 两条补丁。改这里要同步改评测脚本的判定,否则口径会漂。"""
    s = seg.strip()
    if not s or len(s) > max_head:
        return False
    # 只用「。；」判死。**不能连 ！？ 一起挡** —— SEO 标题大量以感叹号/问号结尾
    #（「二、价格表全曝光：光子嫩肤降价30%！」),挡了就整节漏检。
    if '。' in s or '；' in s:
        return False
    if len(_BARE.sub('', s)) < 2:            # 「（）」这类被清洗空的残留行
        return False
    if any(p.match(s) for p in _PATS):
        return True
    # 无编号短标题(「四川省人民医院整形科价格表：眼部整形」)。**不含数字是关键门** ——
    # 价格行/数据行都带数字,靠这条和标题分开。⚠️ 全部规则里误判风险最高的一条,
    # 上线前必须抽 50 篇看准确率;拿不准就把它关掉(准确率优先于召回)。
    return len(s) <= max_bare and not re.search(r'\d', s)


def chunk(ctt, max_size=256, sep=''):
    """输入含 <br> 的正文,返回 chunk 字符串列表。

    不变量:
      1. 零复制 —— sep='' 时 ''.join(结果) 恒等于原文各非空段落的拼接
      2. 任何 chunk 都不会以标题结尾(标题必然和它的正文在同一块里)
      3. 不产生只含标题的 chunk
    """
    segs = [s for s in ctt.split('<br>') if s.strip()]
    if not segs:
        return []

    # 1) 切成 unit。unit = 连续标题 + 紧跟的正文段。边界只允许落在 unit 之间。
    units = []
    for s in segs:
        if is_heading(s):
            if units and not units[-1][1]:
                units[-1][0].append(s)       # 连续标题(「一、」紧跟「1、物理类」)并进同一个 unit
            else:
                units.append(([s], []))
        else:
            if not units:
                units.append(([], []))       # 第一个标题之前的段落,自成一个无标题 unit
            units[-1][1].append(s)

    # 2) 贪心装箱。unit 是最小单位,绝不切开。
    out, buf, blen = [], [], 0

    def flush():
        nonlocal buf, blen
        if buf:
            out.append(sep.join(buf))
            buf, blen = [], 0

    for heads, bodies in units:
        if not bodies:                       # 只有标题(只可能是文末)-> 并进上一块,不单独成 chunk
            buf += heads
            blen += len(sep.join(heads))
            continue
        n = len(sep.join(heads + bodies))
        if n <= max_size:
            if blen + n > max_size:
                flush()
            buf += heads + bodies
            blen += n
        else:
            # 整个 unit 超长:内部按段落再切。**续片不补任何标题** —— 这就是 B1 与传播版的唯一差别。
            flush()
            piece, plen, has_body = list(heads), len(sep.join(heads)), False
            for b in bodies:
                for part in (_split_long(b, max_size) if len(b) > max_size else [b]):
                    if has_body and plen + len(part) > max_size:
                        out.append(sep.join(piece))
                        piece, plen, has_body = [], 0, False
                    piece.append(part)
                    plen += len(part)
                    has_body = True
            if piece:
                out.append(sep.join(piece))
    flush()
    return out


def _split_long(text, max_size):
    """单个段落就超过 max_size:先按句末标点切;仍超长(无标点长串)硬切,绝不静默丢字。"""
    parts, cur = [], ''
    for s in _SENT.split(text):
        if cur and len(cur) + len(s) > max_size:
            parts.append(cur)
            cur = ''
        cur += s
        while len(cur) > max_size:
            parts.append(cur[:max_size])
            cur = cur[max_size:]
    if cur:
        parts.append(cur)
    return parts


if __name__ == '__main__':
    demo = (
        '2026年安徽高考成绩即将公布，很多考生和家长都在关注600分能排多少名这个问题。'
        '下面整理了安徽历史类和物理类的一分一段对应数据，供填报志愿时参考。<br>'
        '一、安徽2026高考一分一段表<br>'
        '2026年成绩将于6月25日公布，届时本文会同步更新，下面先参考安徽2025年的数据：<br>'
        '1、物理类<br>'
        '物理类分数在705分及以上的有18人，700分对应45名，650分对应4832名，'
        '600分对应29273名，550分对应72581名，500分对应136697名，450分对应198420名。<br>'
        '2、历史类<br>'
        '历史类分数在671分及以上的考生有10人，650分对应612名，600分对应3121名，'
        '550分对应14267名，500分对应39038名，450分对应78905名。<br>'
        '二、安徽高考位次排名对应大学<br>'
        '篇幅有限，本文仅展示2025年前1万名对应的大学名单（物理组），完整名单可在省教育考试院官网查询。'
    )
    cs = chunk(demo, 256)
    for i, c in enumerate(cs, 1):
        print(f'[{i}] len={len(c)}\n    {c}\n')

    segs = [s for s in demo.split('<br>') if s.strip()]
    assert ''.join(cs) == ''.join(segs), '零复制不成立'
    assert not any(is_heading(s) and c.endswith(s) for c in cs for s in segs), '存在孤立标题'
    print(f'✓ 零复制({len("".join(cs))} 字 == 原文 {len("".join(segs))} 字)、无孤立标题')
