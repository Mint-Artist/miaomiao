"""HtmlSimplify.scala 的输出与现有链路（Trafilatura 定位 + 自研简化，209 页）逐页对比。

指标：
- 正文一致性：两边纯文本的 8 字片段（去空白、NFKC）互相覆盖的比例。
  参照召回 = 参照正文的片段有多少出现在新输出里；参照精确 = 新输出的片段有多少出现在参照里。
  两者都不是"真值"，差异页要人工看，见 --diff。
- 证据保留：430 道合成问题的证据句（取自参照正文）在新输出中能否找到。
- 标题：944 个参照正文标题（LLM 标注、过滤后，人工抽查约 83% 为真）在新输出中是否成为 h1–h6；
  新输出中推断出的标题，按 LLM 标注判断对错（LLM 标为标题 = 对，LLM 标为非标题 = 错，参照里找不到 = 未知）。
- 表格行、列表项数量。

用法：python3 compare.py out_v1.jsonl [--diff N]
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / '开源对照_2026-09-28'
SEM = ROOT / '语义切分_2026-09-29'
AUD = ROOT / '简版HTML结构审计_2026-10-16'
sys.path.insert(0, str(ROOT))
from chunk_pipeline.normalize import normalize_html  # noqa: E402

WS = re.compile(r'\s+')
CANON = str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'", '—': '-', '–': '-', '…': '...'})
K = 8


def canon(s):
    return unicodedata.normalize('NFKC', WS.sub('', s or '')).translate(CANON)


def grams(s):
    s = canon(s)
    return {s[i:i + K] for i in range(max(1, len(s) - K + 1))} if s else set()


def cover(a, b):
    """a 的片段有多少在 b 中。"""
    return len(a & b) / len(a) if a else 1.0


def html_counts(html):
    soup = BeautifulSoup(html or '', 'html.parser')
    heads = [(h.name, h.get('data-origin'), canon(h.get_text())) for h in soup.find_all(re.compile('^h[1-6]$'))]
    return heads, len(soup.find_all('tr')), len(soup.find_all('li'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--diff', type=int, default=0, help='打印差异最大的 N 页的缺失 / 多出行')
    args = ap.parse_args()
    new = {}
    for line in open(HERE / args.out, encoding='utf-8'):
        d = json.loads(line)
        new[d['id']] = d
    ref_ids = [json.loads(l)['id'] for l in (SRC / 'results' / 'summary.jsonl').open(encoding='utf-8')]
    labels = json.loads((SEM / 'results' / 'heading_labels.json').read_text())
    gold_rows = json.loads((AUD / 'results' / 'heading_stages.json').read_text())
    questions = [json.loads(l) for l in (SEM / 'results' / 'questions.jsonl').open(encoding='utf-8')]

    per = {}
    ref_lines, ref_heading_text, ref_nonheading_text = {}, {}, {}
    for pid in ref_ids:
        url = new[pid]['url']
        S = (SRC / 'results' / 'pages' / pid / 'simplified.html').read_text(encoding='utf-8')
        doc = normalize_html(S, pid, url)
        T = doc.text
        lines = [T[a.start:a.end].strip() for a in doc.atoms]
        ref_lines[pid] = lines
        lab = set(labels.get(pid) or [])
        ref_heading_text[pid] = {canon(lines[i]) for i in lab}
        ref_nonheading_text[pid] = {canon(t) for i, t in enumerate(lines) if i not in lab}
        rt, nt = new[pid]['text'], ''
        nt = new[pid]['text']
        ga, gb = grams(T), grams(nt)
        heads, trs, lis = html_counts(new[pid]['html'])
        per[pid] = {'recall': cover(ga, gb), 'precision': cover(gb, ga), 'ref_chars': len(canon(T)),
                    'new_chars': len(canon(nt)), 'status': new[pid]['status'], 'locate': new[pid]['locate'],
                    'ref_rows': sum(a.kind in ('row', 'table_header') for a in doc.atoms), 'new_rows': trs,
                    'ref_li': sum(a.kind == 'li' for a in doc.atoms), 'new_li': lis, 'heads': heads}

    n = len(per)
    rec = sorted(p['recall'] for p in per.values())
    prec = sorted(p['precision'] for p in per.values())
    tot_ref = sum(p['ref_chars'] for p in per.values())
    tot_new = sum(p['new_chars'] for p in per.values())
    w_rec = sum(p['recall'] * p['ref_chars'] for p in per.values()) / tot_ref
    w_prec = sum(p['precision'] * p['new_chars'] for p in per.values()) / max(tot_new, 1)
    print(f'== 正文一致性（{n} 页，参照 = Trafilatura + 自研简化）')
    print(f'  参照召回：按字加权 {w_rec:.3f}，逐页中位数 {rec[n // 2]:.3f}，<0.8 的页 {sum(r < 0.8 for r in rec)}，<0.5 的页 {sum(r < 0.5 for r in rec)}')
    print(f'  参照精确：按字加权 {w_prec:.3f}，逐页中位数 {prec[n // 2]:.3f}，<0.8 的页 {sum(r < 0.8 for r in prec)}，<0.5 的页 {sum(r < 0.5 for r in prec)}')
    print(f'  总字数：参照 {tot_ref}，新 {tot_new}（{tot_new / tot_ref:.2f} 倍）')
    print('  新输出状态：', Counter(p['status'] for p in per.values()), ' 定位方式：', Counter(p['locate'] for p in per.values()))
    others = [d for k, d in new.items() if k not in per]
    print(f'  参照链路没有产出的 {len(others)} 页：', Counter(d['status'] for d in others),
          ' 新输出字数中位数', sorted(d['textChars'] for d in others)[len(others) // 2] if others else '-')

    found = 0
    for q in questions:
        if canon(q['evidence']) in canon(new[q['id']]['text']):
            found += 1
    print(f'\n== 证据保留：{found}/{len(questions)} = {found / len(questions):.3f}')

    hit = 0
    for r in gold_rows:
        heads = {t for _, _, t in per[r['id']]['heads']}
        hit += canon(r['text']) in heads
    print(f'\n== 标题召回（参照正文标题 {len(gold_rows)} 个）：新 {hit / len(gold_rows):.3f}；现有链路 {sum(r["S_kind"] in ("heading", "pseudo_heading") for r in gold_rows) / len(gold_rows):.3f}')
    inf = Counter()
    for pid, p in per.items():
        for name, origin, t in p['heads']:
            if origin != 'inferred':
                continue
            if t in ref_heading_text[pid]:
                inf['对'] += 1
            elif t in ref_nonheading_text[pid]:
                inf['错'] += 1
            else:
                inf['未知'] += 1
    judged = inf['对'] + inf['错']
    print(f'  推断标题 {sum(inf.values())} 个：{dict(inf)}；可判定部分准确率 {inf["对"] / max(judged, 1):.3f}')

    rr = sum(p['ref_rows'] for p in per.values()); nr = sum(p['new_rows'] for p in per.values())
    rl = sum(p['ref_li'] for p in per.values()); nl = sum(p['new_li'] for p in per.values())
    print(f'\n== 结构数量：表格行 参照 {rr} / 新 {nr}；列表项 参照 {rl} / 新 {nl}')
    print(f'  表格行差异最大的页：', sorted(((p['new_rows'] - p['ref_rows'], pid) for pid, p in per.items()), key=lambda x: -abs(x[0]))[:6])

    if args.diff:
        worst = sorted(per.items(), key=lambda kv: min(kv[1]['recall'], kv[1]['precision']))[:args.diff]
        for pid, p in worst:
            print(f"\n---- {pid} {new[pid]['url']}  召回 {p['recall']:.2f} 精确 {p['precision']:.2f}  参照 {p['ref_chars']} 字 / 新 {p['new_chars']} 字  {p['locate']}")
            gb = grams(new[pid]['text'])
            miss = [l for l in ref_lines[pid] if len(canon(l)) >= 10 and cover(grams(l), gb) < 0.5]
            ga = grams('\n'.join(ref_lines[pid]))
            extra = [l for l in new[pid]['text'].split('\n') if len(canon(l)) >= 10 and cover(grams(l), ga) < 0.5]
            print('  参照有、新缺：', [l[:50] for l in miss[:6]], f'…共 {len(miss)} 行')
            print('  新有、参照无：', [l[:50] for l in extra[:6]], f'…共 {len(extra)} 行')
    json.dump(per, open(HERE / (Path(args.out).stem + '_compare.json'), 'w'), ensure_ascii=False, indent=0, default=str)


if __name__ == '__main__':
    main()
