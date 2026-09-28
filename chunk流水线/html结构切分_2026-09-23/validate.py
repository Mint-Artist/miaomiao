#!/usr/bin/env python3
"""chunk 切分结果的验收检查。

两类检查：
- 通用不变量（每个文档）：预算硬上限、文本覆盖不重不漏、顺序保持、
  prev/next 链、continued↔parent_id 一致、schema 字段齐全；
- 样本结构断言：华为页功能/须知/流程三块，高考页 171 数据行与表头重复，
  律图页伪标题不粘前块。

用法：python3 validate.py（先跑 run_examples.py）
退出码非零表示有检查未通过。
"""
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).parent
OUT = HERE / 'output'
BUDGET = 256

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f'  ({detail})' if detail else ''))


def load_chunks(name):
    return [json.loads(l) for l in
            (OUT / f'{name}.chunks.jsonl').read_text(encoding='utf-8').splitlines()]


def norm(s):
    """覆盖比对口径：去空白和表格行分隔符（分隔符是切分阶段加的，源文没有）。"""
    return re.sub(r'[\s|]+', '', s)


def body_text_without_headings(name):
    soup = BeautifulSoup((OUT / f'{name}.simplified.html').read_text(encoding='utf-8'),
                         'html.parser')
    for t in soup.find_all(re.compile('^h[1-6]$')):
        t.decompose()
    for t in soup.select('thead'):   # 表头行属于上下文，不进 chunk 正文
        t.decompose()
    meta = soup.select_one('div.metadata')
    if meta:
        meta.decompose()
    return norm(soup.get_text())


SCHEMA = {'chunk_id', 'doc_id', 'url', 'text', 'contextual_text', 'chars',
          'contextual_chars', 'context_sources', 'heading_path', 'parent_id',
          'continued', 'prev', 'next', 'data_src', 'kind', 'split_reason',
          'cross_section', 'images', 'atom_ids', 'chunker_version', 'config_hash'}


def invariants(name):
    chunks = load_chunks(name)
    label = name.split('_')[0]
    over = [c['chunk_id'] for c in chunks if c['contextual_chars'] > BUDGET]
    check(f'{label}: 全部块含上下文不超 {BUDGET} 字', not over,
          f'超限 {over}' if over else '')
    concat = norm(''.join(c['text'] for c in chunks))
    expect = body_text_without_headings(name)
    check(f'{label}: 文本覆盖不重不漏', concat == expect,
          f'chunks {len(concat)} 字, 正文 {len(expect)} 字')
    chain_ok = all(
        (c['prev'] == (chunks[i - 1]['chunk_id'] if i else None))
        and (c['next'] == (chunks[i + 1]['chunk_id'] if i + 1 < len(chunks) else None))
        for i, c in enumerate(chunks))
    check(f'{label}: prev/next 链一致', chain_ok)
    ids = [c['chunk_id'] for c in chunks]
    parent_ok = all(
        (c['continued'] and c['parent_id'] in ids)
        or (not c['continued'] and c['parent_id'] is None) for c in chunks)
    check(f'{label}: continued↔parent_id 一致', parent_ok)
    missing = [c['chunk_id'] for c in chunks if set(c) < SCHEMA]
    check(f'{label}: schema 字段齐全', not missing)
    check(f'{label}: 无空块', all(c['text'].strip() for c in chunks))
    return chunks


def validate_huawei():
    chunks = invariants('huawei_zh-cn16079039')
    label = '华为页'
    check(f'{label}: 功能/须知/流程 3 块', len(chunks) == 3
          and [c['kind'] for c in chunks] == ['p', 'note', 'li'],
          f'实际 {[(c["kind"], c["chars"]) for c in chunks]}')
    note = next((c for c in chunks if c['kind'] == 'note'), None)
    check(f'{label}: 须知 4 项在同一块', note is not None
          and all(s in note['text'] for s in
                  ['蓝牙已开启', '最新版本', '跨品牌交友', '发送短信']))
    steps = chunks[-1]
    check(f'{label}: 三个步骤在同一块且型号分支未移位',
          all(s in steps['text'] for s in
              ['在手表主界面', '华为儿童手表 5 系列', '添加后，家长在']))
    check(f'{label}: 每块 contextual_text 以文档标题开头',
          all(c['contextual_text'].startswith('华为儿童手表跨品牌添加联系人')
              for c in chunks))
    check(f'{label}: metadata 默认不注入正文',
          all('适用产品' not in c['text'] for c in chunks))
    check(f'{label}: data-src 锚点存在', all(c['data_src'] for c in chunks))


def validate_gk100():
    chunks = invariants('gk100_read_27177662')
    label = '高考页'
    rows = sum(len([ln for ln in c['text'].split('\n') if ' | ' in ln])
               for c in chunks)
    check(f'{label}: 171 条数据行全覆盖', rows == 171, f'实际 {rows}')
    row_chunks = [c for c in chunks if ' | ' in c['text']]
    check(f'{label}: 含表格行的块都带表头上下文',
          all('表头：学校名 | 专业组 | 2025分数 | 2025位次' in c['contextual_text']
              for c in row_chunks))
    check(f'{label}: 表格块标题路径含所属章节',
          all('二、安徽高考位次排名对应大学' in c['heading_path'] for c in row_chunks))
    bad = [c['chunk_id'] for c in chunks
           if '清华大学' in c['text'] and '一分一段表将于' in c['text']]
    check(f'{label}: 章节一/二内容不混装', not bad)
    phy = next(c for c in chunks if '1、物理类' in c['text'])
    check(f'{label}: 伪标题与其内容同块（粘后不粘前）',
          '705分及以上' in phy['text'])
    no_orphan = all(not re.match(r'^[12]、', c['text'].split('\n')[-1])
                    for c in chunks)
    check(f'{label}: 块尾不出现孤立伪标题', no_orphan)
    check(f'{label}: 首末行位置正确',
          '清华大学 | 004组 | 688 | 85' in chunks[4]['text']
          and '苏州大学 | 005组 | 631 | 9977' in chunks[-1]['text'])


def validate_lvtu():
    chunks = invariants('lvtu_8655766')
    label = '律图页'
    for mark in ['一、公司裁员', '二、公司裁员', '三、公司裁员']:
        owner = next((c for c in chunks if mark in c['text']), None)
        check(f'{label}: 伪标题 {mark[:2]} 位于块首行',
              owner is not None and owner['text'].split('\n')[0].startswith(mark[:2]))
    no_orphan = all(not re.match(r'^[一二三]、', c['text'].split('\n')[-1])
                    for c in chunks)
    check(f'{label}: 块尾不出现孤立伪标题', no_orphan)
    check(f'{label}: 法条锚文本留在正文', any('《劳动法》' in c['text'] for c in chunks))


validate_huawei()
validate_gk100()
validate_lvtu()
failed = [r for r in results if not r[1]]
print(f'\n{len(results) - len(failed)}/{len(results)} 项通过')
sys.exit(1 if failed else 0)
