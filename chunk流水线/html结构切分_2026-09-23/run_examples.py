#!/usr/bin/env python3
"""端到端：原始 HTML →（提取器，带锚点）→ 简版 HTML →（切分器）→ chunks.jsonl。

提取器代码从 ../简版HTML提取_2026-09-23 导入，保证两阶段用的是同一份实现。
输出：output/<doc>.simplified.html、output/<doc>.chunks.jsonl、output/<doc>.report.json

用法：python3 run_examples.py [--budget 256] [--merge-short-sections]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXTRACTOR_DIR = HERE.parent / '简版HTML提取_2026-09-23'
sys.path.insert(0, str(EXTRACTOR_DIR))

from simplify_html import extract  # noqa: E402

from chunker import ChunkConfig, build_report, chunk_document  # noqa: E402

CASES = [
    ('huawei_zh-cn16079039', 'https://consumer.huawei.com/cn/support/content/zh-cn16079039/'),
    ('gk100_read_27177662', 'https://m.gk100.com/read_27177662.htm'),
    ('lvtu_8655766', 'https://www.64365.com/zs/8655766.aspx'),
]


def main():
    budget = 256
    merge = '--merge-short-sections' in sys.argv
    if '--budget' in sys.argv:
        budget = int(sys.argv[sys.argv.index('--budget') + 1])
    cfg = ChunkConfig(budget=budget, merge_short_sections=merge)
    out_dir = HERE / 'output'
    out_dir.mkdir(exist_ok=True)
    for name, url in CASES:
        raw = (EXTRACTOR_DIR / 'samples' / f'{name}.raw.html').read_text(encoding='utf-8')
        simplified, extract_report = extract(raw, url=url, anchors=True)
        (out_dir / f'{name}.simplified.html').write_text(simplified, encoding='utf-8')
        records, stats = chunk_document(simplified, name, url, cfg)
        (out_dir / f'{name}.chunks.jsonl').write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n',
            encoding='utf-8')
        report = build_report(records, stats, name, url, cfg)
        (out_dir / f'{name}.report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"[{name}] chunks={stats['chunks']} over_budget={stats['over_budget']} "
              f"coverage={stats['coverage_ok']} order={stats['order_ok']} "
              f"avg={stats['avg_chars']}")


if __name__ == '__main__':
    main()
