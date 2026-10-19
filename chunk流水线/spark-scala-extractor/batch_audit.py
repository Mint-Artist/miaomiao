#!/usr/bin/env python3
"""汇总 Spark 提取批次的工程指标，并展开每页简版 HTML 供人工抽检。

用法：python3 batch_audit.py --run-dir output/batch_235_20261019 \
  --input /tmp/spark_extract_batch_235.jsonl

注意：成功率只表示没有解析异常，不等于正文提取质量；质量仍需按 host 抽样审查。
"""
import argparse
import collections
import json
import statistics
from pathlib import Path
from urllib.parse import urlparse


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    raw = {r['doc_id']: r for r in read_jsonl(args.input)}
    rows = []
    for path in sorted((run / 'spark-json').glob('part-*.json')):
        for row in read_jsonl(path):
            report = json.loads(row['report_json'])
            source = raw[row['doc_id']]
            simplified = row.get('simplified_html')
            rows.append({
                'doc_id': row['doc_id'], 'url': source['url'],
                'host': urlparse(source['url']).netloc,
                'status': row['status'], 'confidence': round(row['confidence'], 3),
                'profile': row['profile'], 'content_root': row['content_root'],
                'error': row.get('error'), 'raw_chars': len(source['pg']),
                'simplified_chars': len(simplified) if simplified else 0,
                'text_chars': report.get('text_chars', 0),
                'warnings': report.get('warnings', []),
                'ratio': round(len(simplified) / max(1, len(source['pg'])), 4) if simplified else 0,
            })
    rows.sort(key=lambda r: r['doc_id'])
    (run / 'pages').mkdir(exist_ok=True)
    for path in sorted((run / 'spark-json').glob('part-*.json')):
        for row in read_jsonl(path):
            if row.get('simplified_html'):
                (run / 'pages' / f"{row['doc_id']}.html").write_text(
                    row['simplified_html'], encoding='utf-8')
    with (run / 'index.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    total = len(rows)
    statuses = collections.Counter(r['status'] for r in rows)
    profiles = collections.Counter(r['profile'] for r in rows)
    errors = collections.Counter(r['error'] for r in rows if r['error'])
    ratios = [r['ratio'] for r in rows if r['simplified_chars']]
    text_chars = [r['text_chars'] for r in rows if r['simplified_chars']]
    no_title = sum(any('未找到标题' in w for w in r['warnings']) for r in rows)
    suspicious = [r for r in rows if r['simplified_chars'] and
                  (r['text_chars'] < 80 or r['ratio'] > 0.8 or r['content_root'] == 'body')]
    suspicious.sort(key=lambda r: (r['text_chars'], -r['ratio']))
    failed = [r for r in rows if r['status'] == 'failed']
    degraded = [r for r in rows if r['status'] == 'degraded']
    hosts = {}
    for host, group in __import__('itertools').groupby(sorted(rows, key=lambda r: r['host']), key=lambda r: r['host']):
        group = list(group)
        hosts[host] = {'pages': len(group), 'ok': sum(g['status'] == 'ok' for g in group),
                       'degraded': sum(g['status'] == 'degraded' for g in group),
                       'failed': sum(g['status'] == 'failed' for g in group)}

    summary = {
        'total': total, 'status_counts': dict(statuses), 'profile_counts': dict(profiles),
        'host_count': len(hosts), 'missing_title_pages': no_title,
        'median_ratio': statistics.median(ratios) if ratios else None,
        'median_text_chars': statistics.median(text_chars) if text_chars else None,
        'suspicious_candidates': len(suspicious), 'hosts': hosts,
        'errors': {str(k): v for k, v in errors.items()},
    }
    (run / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def table(items, fields, limit=20):
        lines = []
        for row in items[:limit]:
            values = []
            for field in fields:
                value = row.get(field)
                values.append(str(value).replace('|', '\\|') if value is not None else '')
            lines.append('| ' + ' | '.join(values) + ' |')
        return lines

    lines = [
        '# 批量提取审计报告', '',
        f'- 输入：{total} 个网页，{len(hosts)} 个 host',
        f"- 状态：{dict(statuses)}",
        f"- 提取模式：{dict(profiles)}",
        f"- 未找到标题：{no_title} 页",
        f"- 简版/原始字符比中位数：{summary['median_ratio']}",
        f"- 简版正文字符数中位数：{summary['median_text_chars']}",
        f'- 建议优先抽检（正文过短、压缩比异常或 body 兜底）：{len(suspicious)} 页',
        '',
        '> 状态成功只表示未发生解析异常，不等于正文提取质量。质量结论必须结合人工抽检。',
        '', '## 失败页', '',
        '| doc_id | host | error | url |', '|---|---|---|---|']
    lines += table(failed, ['doc_id', 'host', 'error', 'url'], 50)
    lines += ['', '## 降级页（body 兜底）', '',
              '| doc_id | host | text_chars | url |', '|---|---|---:|---|']
    lines += table(degraded, ['doc_id', 'host', 'text_chars', 'url'], 50)
    lines += ['', '## 优先抽检候选', '',
              '| doc_id | host | text_chars | ratio | content_root | url |',
              '|---|---|---:|---:|---|---|']
    lines += table(suspicious, ['doc_id', 'host', 'text_chars', 'ratio', 'content_root', 'url'], 50)
    lines += ['', '## 文件', '',
              '- `pages/`：每页简版 HTML（按 doc_id 命名）',
              '- `index.jsonl`：每页状态、置信度、字符数、content_root、告警与 URL',
              '- `audit.json`：机器可读汇总', '']
    (run / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'total': total, 'statuses': statuses, 'profiles': profiles,
                      'hosts': len(hosts), 'suspicious': len(suspicious),
                      'report': str(run / 'REPORT.md')}, ensure_ascii=False, default=dict))


if __name__ == '__main__':
    main()
