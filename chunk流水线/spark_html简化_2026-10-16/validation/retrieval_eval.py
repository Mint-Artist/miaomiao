"""端到端检索对比：HtmlSimplify.scala 的简版 HTML vs 现有链路（Trafilatura + 自研简化），走同一套切分与检索。

- 语料：现有链路有产出的 209 页（两边相同的页面集合）。
- 切分：chunk_pipeline 的 structure_protect 与 structure_protect_context（软 256 / 硬 512）。
- 问题：../../语义切分_2026-09-29/results/questions.jsonl 的 430 题；证据在新正文中按去空白匹配定位，找不到记为未命中。
- 检索：硅基流动 Pro/BAAI/bge-m3（510 token 截断，结果缓存），全库余弦排序。
- 指标口径与 ../../语义切分_2026-09-29/eval_retrieval.py 相同；字数预算按块正文计。

用法（用语义切分目录的 venv）：../../语义切分_2026-09-29/.venv/bin/python retrieval_eval.py out_v11.jsonl
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEM = HERE.parent.parent / '语义切分_2026-09-29'
sys.path.insert(0, str(SEM))
from common import load_pages  # noqa: E402  （设置 sys.path，并载入现有链路的 209 页）
import methods as M  # noqa: E402
from chunk_pipeline.normalize import normalize_html  # noqa: E402
from eval_retrieval import coverage, single_cover, CHAR_BUDGETS  # noqa: E402

WS = re.compile(r'\s+')
CONFIGS = ['structure_protect', 'structure_protect_context']


def locate_evidence(T, ev):
    """去空白后匹配，返回 T 中的 [start, end)；找不到返回 None。"""
    idx = [i for i, c in enumerate(T) if not c.isspace()]
    flat = ''.join(T[i] for i in idx)
    e = WS.sub('', ev)
    pos = flat.find(e)
    if pos < 0 or not e:
        return None
    return idx[pos], idx[pos + len(e) - 1] + 1


def chunks_for(doc, cfg):
    page = {'doc': doc, 'T': doc.text}
    return M.run_own(page, cfg)


def evaluate(qs, spans_by_q, corpus, emb):
    """corpus: [(page_id, (s, e), index_text)]；返回逐题指标。"""
    ids = np.array([c[0] for c in corpus])
    spans = [c[1] for c in corpus]
    lens = np.array([e - s for s, e in spans])
    cv = emb.encode([c[2] for c in corpus])
    qv = emb.encode([q['question'] for q in qs])
    sims = qv @ cv.T
    rows = []
    for qi, q in enumerate(qs):
        span = spans_by_q[qi]
        if span is None:
            rows.append({'cov@5': 0.0, 'cov@1024字': 0.0, 'MRR': 0.0, 'loc@1': 0.0, 'found': 0.0})
            continue
        order = np.argsort(-sims[qi])
        same = ids[order] == q['id']
        r = {'found': 1.0}
        top = order[:5][same[:5]]
        r['cov@5'] = coverage(span, [spans[i] for i in top])
        cum = np.cumsum(lens[order])
        n = max(1, int(np.searchsorted(cum, 1024, side='right')))
        r['cov@1024字'] = coverage(span, [spans[i] for i in order[:n][same[:n]]])
        rr = 0.0
        for rank, i in enumerate(order[:100]):
            if ids[i] == q['id'] and single_cover(span, *spans[i]) >= 0.5:
                rr = 1 / (rank + 1)
                break
        r['MRR'] = rr
        own = np.where(ids == q['id'])[0]
        r['loc@1'] = float(single_cover(span, *spans[own[np.argmax(sims[qi][own])]]) >= 0.5) if len(own) else 0.0
        rows.append(r)
    return rows


def main():
    out_file = sys.argv[1]
    from api import ApiEmbedder
    emb = ApiEmbedder()
    new = {}
    for line in (HERE / out_file).open(encoding='utf-8'):
        d = json.loads(line)
        new[d['id']] = d
    pages = load_pages()
    qs = [json.loads(l) for l in (SEM / 'results' / 'questions.jsonl').open(encoding='utf-8')]
    keys = ['found', 'cov@5', 'cov@1024字', 'MRR', 'loc@1']
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(qs), (2000, len(qs)))
    report = {}
    for cfg in CONFIGS:
        res = {}
        for side in ('现有链路', '新实现'):
            corpus, docs = [], {}
            for p in pages:
                if side == '现有链路':
                    doc = p['doc']
                else:
                    html = new[p['id']]['html']
                    try:
                        doc = normalize_html(html, p['id'], p['url']) if html.strip() else None
                    except Exception as exc:  # noqa: BLE001
                        print('normalize failed', p['id'], exc)
                        doc = None
                docs[p['id']] = doc
                if doc is None or not doc.text.strip():
                    continue
                for c in chunks_for(doc, cfg):
                    corpus.append((p['id'], c['span'], c['index']))
            spans_by_q = []
            for q in qs:
                doc = docs.get(q['id'])
                spans_by_q.append(None if doc is None else
                                  (tuple(q['span']) if side == '现有链路' else locate_evidence(doc.text, q['evidence'])))
            rows = evaluate(qs, spans_by_q, corpus, emb)
            res[side] = rows
            print(f'{cfg:28s} {side}  块数 {len(corpus)}  ' + '  '.join(f'{k} {np.mean([r[k] for r in rows]):.3f}' for k in keys), flush=True)
        delta = {}
        for k in keys[1:]:
            d = np.array([a[k] - b[k] for a, b in zip(res['新实现'], res['现有链路'])])
            bs = d[idx].mean(1)
            delta[k] = [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        print(f'{cfg:28s} 差值（新 − 现有，×100，95% 区间）：' +
              '  '.join(f'{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]' for k, v in delta.items()))
        both = [i for i, r in enumerate(res['新实现']) if r['found']]
        bidx = rng.integers(0, len(both), (2000, len(both)))
        d2 = {}
        for k in keys[1:]:
            d = np.array([res['新实现'][i][k] - res['现有链路'][i][k] for i in both]); bs = d[bidx].mean(1)
            d2[k] = [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        print(f'{cfg:28s} 仅两边都能找到证据的 {len(both)} 题：' +
              '  '.join(f'{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]' for k, v in d2.items()))
        report[cfg] = {side: {k: float(np.mean([r[k] for r in rows])) for k in keys} for side, rows in res.items()}
        report[cfg]['per_question'] = {side: rows for side, rows in res.items()}
        report[cfg]['delta'] = delta
        report[cfg]['delta_found_only'] = d2
    (HERE / (Path(out_file).stem + '_retrieval.json')).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
