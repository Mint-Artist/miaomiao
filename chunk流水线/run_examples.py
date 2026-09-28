#!/usr/bin/env python3
"""Run all local policies on identical frozen HTML; never mutate legacy outputs."""
import argparse
import json
from pathlib import Path

from chunk_pipeline import ChunkConfig, normalize_html
from chunk_pipeline.pipeline import chunk_document, retrieval_views
from chunk_pipeline.io import dumps, write_run
from chunk_pipeline.batch import summary_chunks

HERE = Path(__file__).resolve().parent
CASES = [
    ('huawei_zh-cn16079039', 'https://consumer.huawei.com/cn/support/content/zh-cn16079039/'),
    ('gk100_read_27177662', 'https://m.gk100.com/read_27177662.htm'),
    ('lvtu_8655766', 'https://www.64365.com/zs/8655766.aspx'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--absolute-chars', type=int, required=True, help='Explicit demo H; does not freeze production H')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        ap.error('Choose a new output directory')
    configs = []
    for p in sorted((HERE / 'configs').glob('*.json')):
        obj = json.loads(p.read_text())
        if obj.get('budget_policy') == 'protect':
            obj['absolute_chars'] = args.absolute_chars
        configs.append((p.stem, ChunkConfig.from_dict(obj)))
    rows, prepared, summaries = [], [], []
    for name, url in CASES:
        html = (HERE / 'html结构切分_2026-09-23/output' / (name + '.simplified.html')).read_text()
        doc = normalize_html(html, name, url)
        expected = (HERE / '规格/vNext/样例' / (name + '.normalized.txt')).read_text()
        if doc.text != expected:
            raise ValueError('Canonical fixture differs for ' + name)
        summary = {'url': url}
        for config_name, cfg in configs:
            records, report = chunk_document(doc, cfg)
            views = retrieval_views(doc, records)
            prepared.append((config_name, name, doc, records, report, views, html))
            summary[config_name] = summary_chunks(records, views, 'index_text')
            rows.append({'sample': name, 'strategy': config_name, 'config_hash': cfg.config_hash,
                         'text_sha256': doc.text_sha256, **report['stats']})
        summaries.append(summary)
    # Compute and validate everything before publishing any output.
    for config_name, name, doc, records, report, views, html in prepared:
        write_run(args.out_dir / config_name / name, doc, records, report, views, simplified_html=html)
    (args.out_dir / 'summary.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in summaries), encoding='utf-8')
    (args.out_dir / 'comparison.json').write_text(dumps(rows), encoding='utf-8')
    md = ['# 本地策略比较', '', 'H=%d 是本次实验参数，不是生产默认。所有方案共享同一份规范化正文。' % args.absolute_chars,
          '此处块数/长度仅描述切分，不代表检索效果。', '',
          '| 样本 | 策略 | 块数 | 超目标 | 被拆原子 | 最大含上下文字符数 | 严格还原 |',
          '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        md.append('| {sample} | {strategy} | {chunks} | {over_target} | {split_atoms} | {maxlen} | 通过 |'.format(
            maxlen=row['lengths']['max'], **row))
    (args.out_dir / 'comparison.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
