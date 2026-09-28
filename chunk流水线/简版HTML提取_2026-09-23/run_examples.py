#!/usr/bin/env python3
"""对 samples/ 下两个样例页面跑提取，结果写入 output/。

用法：python3 run_examples.py [--anchors] [--promote-pseudo-headings]
附加参数会透传给 simplify_html.extract。
"""
import sys
from pathlib import Path

from simplify_html import extract

HERE = Path(__file__).parent

CASES = [
    ('huawei_zh-cn16079039', 'https://consumer.huawei.com/cn/support/content/zh-cn16079039/'),
    ('gk100_read_27177662', 'https://m.gk100.com/read_27177662.htm'),
    ('lvtu_8655766', 'https://www.64365.com/zs/8655766.aspx'),
]


def main():
    import json
    anchors = '--anchors' in sys.argv
    promote = '--promote-pseudo-headings' in sys.argv
    out_dir = HERE / 'output'
    out_dir.mkdir(exist_ok=True)
    for name, url in CASES:
        html = (HERE / 'samples' / f'{name}.raw.html').read_text(encoding='utf-8')
        simplified, report = extract(html, url=url, anchors=anchors,
                                     promote_pseudo=promote)
        (out_dir / f'{name}.simplified.html').write_text(simplified, encoding='utf-8')
        (out_dir / f'{name}.report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"[{report['profile']}] {name}: {report['text_chars']} chars, "
              f"warnings={len(report['warnings'])}")


if __name__ == '__main__':
    main()
