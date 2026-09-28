#!/usr/bin/env python3
"""简版 HTML 提取结果的验收检查。

对照两类标准：
- 结构验收：华为页应解析出 1 个标题、2 条元信息、2 个介绍段落、
  1 个 note 提示框（4 项）、1 个三步有序列表；高考页表格应为
  1 表头行（4 th）+ 171 数据行（每行 4 td），单元格内 <br> 保留。
- 文字覆盖：输出与人工核对过的参照稿（华为页面精简正文示例.html、
  downloads/gk100_read_27177662/正文精简.html）去空白后逐字一致。

用法：python3 validate.py   （先跑 run_examples.py 生成 output/）
退出码非零表示有检查未通过。
"""
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).parent
OUT = HERE / 'output'

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f'  ({detail})' if detail else ''))


def norm_text(soup):
    return re.sub(r'\s+', '', soup.get_text())


def load(path):
    return BeautifulSoup(Path(path).read_text(encoding='utf-8'), 'html.parser')


def hygiene(soup, label):
    bad_tags = soup.find_all(['script', 'style', 'noscript', 'nav', 'header',
                              'footer', 'aside', 'iframe', 'form', 'span'])
    check(f'{label}: 无 script/style/nav/span 等残留', not bad_tags,
          f'残留 {[t.name for t in bad_tags][:5]}' if bad_tags else '')
    bad_attrs = []
    for tag in soup.find_all(True):
        for attr in tag.attrs:
            ok = attr in ('href', 'src', 'alt', 'rowspan', 'colspan', 'scope',
                          'headers', 'start', 'reversed', 'value', 'span',
                          'class', 'data-src', 'data-origin')
            if not ok or attr.startswith('on') or attr in ('style', 'id'):
                bad_attrs.append(f'{tag.name}[{attr}]')
    check(f'{label}: 属性白名单生效（无 style/id/on* 等）', not bad_attrs,
          ', '.join(bad_attrs[:5]) if bad_attrs else '')
    classes = {c for t in soup.find_all(True) for c in (t.get('class') or [])}
    check(f'{label}: class 只剩语义分组', classes <= {'metadata', 'note'},
          f'实际 {sorted(classes)}')


def validate_huawei():
    soup = load(OUT / 'huawei_zh-cn16079039.simplified.html')
    label = '华为页'
    h1 = soup.find_all('h1')
    check(f'{label}: 单一 h1 标题', len(h1) == 1
          and h1[0].get_text(strip=True) == '华为儿童手表跨品牌添加联系人')
    meta = soup.select_one('div.metadata')
    meta_ps = meta.find_all('p') if meta else []
    check(f'{label}: metadata 两条（适用产品/适用版本）',
          len(meta_ps) == 2 and meta_ps[0].get_text().startswith('适用产品：')
          and meta_ps[1].get_text().startswith('适用版本：'))
    top = [c for c in soup.find_all(recursive=False) or soup.children]
    body = soup  # 输出无外层包裹，直接在根找
    intro = [p for p in body.find_all('p', recursive=True)
             if p.find_parent('div') is None and p.find_parent('li') is None]
    check(f'{label}: 2 个介绍段落', len(intro) == 2,
          f'实际 {len(intro)}')
    note = soup.select_one('div.note')
    note_li = note.select('ul > li') if note else []
    check(f'{label}: note 提示框含 4 项 ul', note is not None and len(note_li) == 4)
    check(f'{label}: note 内提示图标 img 保留', note is not None
          and note.find('img') is not None)
    ol = soup.find_all('ol')
    steps = ol[0].find_all('li', recursive=False) if ol else []
    check(f'{label}: 1 个三步 ol', len(ol) == 1 and len(steps) == 3)
    if len(steps) == 3:
        check(f'{label}: 步骤内型号分支 p 未移位',
              len(steps[0].find_all('p')) == 2 and len(steps[1].find_all('p')) == 1
              and '华为儿童手表 5 系列' in steps[0].find_all('p')[0].get_text())
    check(f'{label}: 4 张图片在位', len(soup.find_all('img')) == 4)
    links = soup.find_all('a')
    check(f'{label}: 正文链接保留 href', len(links) == 1
          and 'consumer.huawei.com' in (links[0].get('href') or ''))
    hygiene(soup, label)
    ref = load(HERE.parent.parent / '调研与资料' / '华为页面精简正文示例.html')
    check(f'{label}: 文字与人工参照稿逐字一致',
          norm_text(soup) == norm_text(ref),
          f'输出 {len(norm_text(soup))} 字，参照 {len(norm_text(ref))} 字')


def validate_gk100():
    soup = load(OUT / 'gk100_read_27177662.simplified.html')
    label = '高考页'
    h1 = soup.find_all('h1')
    check(f'{label}: 单一 h1 标题', len(h1) == 1
          and '一分一段表' in h1[0].get_text())
    meta_ps = soup.select('div.metadata p')
    check(f'{label}: metadata 两条（作者/发布时间）',
          len(meta_ps) == 2 and meta_ps[0].get_text().startswith('作者：')
          and meta_ps[1].get_text().startswith('发布时间：'))
    tables = soup.find_all('table')
    check(f'{label}: 1 张表格', len(tables) == 1)
    if tables:
        t = tables[0]
        head = t.select('thead tr')
        body = t.select('tbody tr')
        check(f'{label}: 表头 1 行 4 个 th',
              len(head) == 1 and len(head[0].find_all('th')) == 4)
        check(f'{label}: 171 数据行，每行 4 个 td',
              len(body) == 171
              and all(len(r.find_all('td', recursive=False)) == 4 for r in body))
        first = [c.get_text(strip=True) for c in body[0].find_all('td')] if body else []
        last = [c.get_text(strip=True) for c in body[-1].find_all('td')] if body else []
        check(f'{label}: 首行=清华大学/004组/688/85',
              first == ['清华大学', '004组', '688', '85'], f'实际 {first}')
        check(f'{label}: 末行=苏州大学/005组/631/9977',
              last == ['苏州大学', '005组', '631', '9977'], f'实际 {last}')
        check(f'{label}: 单元格内 <br> 保留',
              any(c.find('br') for c in t.find_all(['th', 'td'])))
        nested_p = t.find_all('p')
        check(f'{label}: 表格内无 p 标签混入', not nested_p)
    pseudo = [p.get_text(strip=True) for p in soup.find_all('p')
              if p.find('strong') and p.get_text(strip=True) in ('1、物理类', '2、历史类')]
    check(f'{label}: 伪标题按原样保留为 p>strong（默认不新增标题）',
          len(pseudo) == 2, f'实际 {pseudo}')
    check(f'{label}: 原生章节标题级别保留（原文 h3）',
          [h.get_text(strip=True)[:2] for h in soup.find_all('h3')] == ['一、', '二、'])
    hygiene(soup, label)
    ref = load(HERE.parent.parent / '调研与资料' / 'downloads/gk100_read_27177662/正文精简.html')
    check(f'{label}: 文字与人工参照稿逐字一致',
          norm_text(soup) == norm_text(ref),
          f'输出 {len(norm_text(soup))} 字，参照 {len(norm_text(ref))} 字')


def validate_lvtu():
    """律图页：无人工参照稿，做结构与噪声断言。"""
    soup = load(OUT / 'lvtu_8655766.simplified.html')
    label = '律图页'
    h1 = soup.find_all('h1')
    check(f'{label}: 单一 h1 标题', len(h1) == 1
          and h1[0].get_text(strip=True) == '公司裁员提成部分能要回吗')
    meta_ps = soup.select('div.metadata p')
    check(f'{label}: metadata 含来源/时间/阅读量',
          len(meta_ps) == 3 and meta_ps[0].get_text().startswith('来源：'))
    heads = [p.get_text(strip=True)[:2] for p in soup.find_all('p')
             if p.find('strong') and p.get_text(strip=True)[:1] in ('一', '二', '三')]
    check(f'{label}: 三个加粗伪标题按原样保留', heads == ['一、', '二、', '三、'],
          f'实际 {heads}')
    text = soup.get_text()
    boilerplate = ['免责声明', '网站地图', '更多#劳动纠纷', '获取专业解读']
    hits = [w for w in boilerplate if w in text]
    check(f'{label}: 无免责声明/推荐/导流等样板噪声', not hits,
          f'残留 {hits}' if hits else '')
    check(f'{label}: 法条内链 href 保留', len(soup.find_all('a')) >= 20,
          f"实际 {len(soup.find_all('a'))}")
    img = soup.find('img')
    check(f'{label}: 配图保留且有 alt', img is not None and bool(img.get('alt')))
    hygiene(soup, label)


validate_huawei()
validate_gk100()
validate_lvtu()
failed = [r for r in results if not r[1]]
print(f'\n{len(results) - len(failed)}/{len(results)} 项通过')
sys.exit(1 if failed else 0)
