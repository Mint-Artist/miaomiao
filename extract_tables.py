"""从网页正文中根据同事提取的表格内容，定位并切分出表格片段。"""
import json
import re
import sys
from difflib import SequenceMatcher


def normalize(text: str) -> tuple[str, list[int]]:
    """去掉 <br>、\n、\r、\t、空格等，返回归一化文本和原始位置映射。"""
    chars, mapping = [], []
    for i, c in enumerate(text):
        if c not in (' ', '\n', '\r', '\t'):
            chars.append(c)
            mapping.append(i)
    clean = ''.join(chars)
    # 再去掉 <br> <br/> <br /> 标签
    result_chars, result_map = [], []
    j = 0
    while j < len(clean):
        # 匹配 <br>, <br/>, <br />
        m = re.match(r'<br\s*/?\s*>', clean[j:], re.IGNORECASE)
        if m:
            j += m.end()
        else:
            result_chars.append(clean[j].lower())
            result_map.append(mapping[j])
            j += 1
    return ''.join(result_chars), result_map


def get_table_body(content: str) -> str:
    """去掉表头（第一个<br>之前的部分），返回表格正文。"""
    # 按第一个 <br> 分割
    m = re.search(r'<br\s*/?\s*>', content, re.IGNORECASE)
    if m:
        return content[m.end():]
    return content


def extract_tables(ctt: str, tabel_info_str: str):
    """
    返回:
        segments: list[str]  - 切分后的文本片段
        types:    list[int]  - 0=普通文本, 1=表格
        similarities: list[float] - 每个表格的相似度
    """
    # 解析 tabel_info
    try:
        tabel_info = json.loads(tabel_info_str) if isinstance(tabel_info_str, str) else tabel_info_str
    except (json.JSONDecodeError, TypeError):
        tabel_info = []

    # 提取有效的表格正文
    table_bodies = []
    for item in tabel_info:
        content = item.get('content', '') if isinstance(item, dict) else ''
        if not content or not content.strip():
            continue
        body = get_table_body(content)
        if body.strip():
            table_bodies.append(body)

    if not table_bodies:
        return [ctt], [0], []

    # 归一化原文
    norm_ctt, ctt_map = normalize(ctt)

    # 在归一化原文中定位每个表格，记录原文中的 (start, end)
    spans = []
    for body in table_bodies:
        norm_body, _ = normalize(body)
        idx = norm_ctt.find(norm_body)
        if idx != -1:
            orig_start = ctt_map[idx]
            orig_end = ctt_map[idx + len(norm_body) - 1] + 1
            spans.append((orig_start, orig_end))
        else:
            # 精确匹配失败，尝试滑动窗口找最大相似子串
            best_ratio, best_start, best_len = 0, 0, len(norm_body)
            step = max(1, len(norm_body) // 4)
            for s in range(0, len(norm_ctt) - len(norm_body) + 1, step):
                candidate = norm_ctt[s:s + len(norm_body)]
                r = SequenceMatcher(None, norm_body, candidate).ratio()
                if r > best_ratio:
                    best_ratio, best_start = r, s
            # 精细搜索
            fine_start = max(0, best_start - step)
            fine_end = min(len(norm_ctt) - len(norm_body) + 1, best_start + step + 1)
            for s in range(fine_start, fine_end):
                candidate = norm_ctt[s:s + len(norm_body)]
                r = SequenceMatcher(None, norm_body, candidate).ratio()
                if r > best_ratio:
                    best_ratio, best_start = r, s
            if best_ratio > 0.5:
                orig_start = ctt_map[best_start]
                end_idx = min(best_start + len(norm_body) - 1, len(ctt_map) - 1)
                orig_end = ctt_map[end_idx] + 1
                spans.append((orig_start, orig_end))
            else:
                spans.append(None)

    # 按位置排序有效span，切分原文
    valid = [(s, i) for i, s in enumerate(spans) if s is not None]
    valid.sort(key=lambda x: x[0][0])

    segments, types, similarities = [], [], []
    pos = 0
    for (start, end), body_idx in valid:
        if start > pos:
            segments.append(ctt[pos:start])
            types.append(0)
        extracted = ctt[start:end]
        segments.append(extracted)
        types.append(1)
        pos = end
        # 计算相似度
        norm_ext, _ = normalize(extracted)
        norm_body, _ = normalize(table_bodies[body_idx])
        sim = SequenceMatcher(None, norm_body, norm_ext).ratio()
        similarities.append(sim)

    if pos < len(ctt):
        segments.append(ctt[pos:])
        types.append(0)

    # 对于未匹配的表格，补相似度0
    for i, s in enumerate(spans):
        if s is None:
            similarities.append(0.0)

    return segments, types, similarities


def process_file(input_path: str, output_path: str = None):
    """处理jsonl文件。"""
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ctt = record.get('ctt', '')
            tabel_info = record.get('tabel_info', '[]')
            segments, types, similarities = extract_tables(ctt, tabel_info)
            results.append({
                'segments': segments,
                'types': types,
                'similarities': similarities,
            })
            # 打印摘要
            n_tables = sum(1 for t in types if t == 1)
            print(f"片段数: {len(segments)}, 表格数: {n_tables}, "
                  f"相似度: {[f'{s:.4f}' for s in similarities]}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\n结果已写入: {output_path}")

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python extract_tables.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    process_file(input_path, output_path)
