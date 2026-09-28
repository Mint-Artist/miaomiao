"""Retrieval serialization. It never mutates canonical document text."""
import re


def overlapping_atoms(doc, start, end):
    return [a for a in doc.atoms if a.start < end and a.end > start]


def build_context(doc, start, end, cfg):
    result = {'prefix': '', 'sources': [], 'trimmed': []}
    if cfg.context_mode == 'none':
        return result
    atoms = overlapping_atoms(doc, start, end)
    if not atoms:
        return result
    headings = {'heading', 'pseudo_heading'}
    scope = next((a.scope for a in atoms if a.kind not in headings), atoms[-1].scope)
    if cfg.title_policy == 'document_nearest' and len(scope) > 1:
        selected = [scope[0], scope[-1]]
    else:
        selected = list(scope)
    for nid in scope:
        if nid not in selected:
            result['trimmed'].append({'node_id': nid, 'reason': 'title_policy'})
    titles = []
    for nid in selected:
        n = doc.nodes[nid]
        if start <= n['start'] and n['end'] <= end:
            continue  # Deduplicate by source location, not by matching strings.
        titles.append((doc.text[n['start']:n['end']], n))
    while titles and len('\n'.join(t[0] for t in titles)) > cfg.max_title_chars:
        _, n = titles.pop()
        result['trimmed'].append({'node_id': n['node_id'], 'reason': 'title_budget'})
    parts = []

    def source(kind, node):
        result['sources'].append({'kind': kind, 'node_id': node['node_id'],
                                  'start': node['start'], 'end': node['end']})

    table_ids = list(dict.fromkeys(a.table_id for a in atoms if a.table_id))
    for tid in table_ids:
        table = doc.tables[tid]
        rows = [r for r in table['rows'] if not r['is_header']
                and r['start'] < end and r['end'] > start]
        if not rows:
            continue
        for nid in table['header_ids']:
            n = doc.nodes[nid]
            if start <= n['start'] and n['end'] <= end:
                continue
            row = next(r for r in table['rows'] if r['node_id'] == nid)
            values = [re.sub(r'\s+', ' ', doc.text[c['start']:c['end']]).strip()
                      for c in row['cells']]
            parts.append('表头：' + ' | '.join(values))
            source('table_header', n)
        inherited = set()
        all_cells = [c for r in table['rows'] for c in r['cells']]
        for row in rows:
            if start > row['start'] or end < row['end']:
                cells = [c for c in row['cells'] if c['start'] < end and c['end'] > start]
                columns = ','.join(str(c['column'] + 1) for c in cells)
                parts.append('表格续片：行%d；列%s' % (row['row'] + 1, columns or '空'))
                source('table_fragment', row)
            for cell in all_cells:
                if (cell['row'] < row['row'] < cell['row'] + cell['rowspan']
                        and cell['node_id'] not in inherited
                        and not (start <= cell['start'] and cell['end'] <= end)):
                    value = doc.text[cell['start']:cell['end']]
                    parts.append('跨行继承：列%d=%s' % (cell['column'] + 1, value))
                    source('rowspan', cell)
                    inherited.add(cell['node_id'])
    # Mandatory row/header context has priority over optional title copies.
    # Always reserve room for at least one source character and the separator.
    while titles and len('\n'.join([v for v, _ in titles] + parts)) + 2 > cfg.target_chars:
        _, n = titles.pop()
        result['trimmed'].append({'node_id': n['node_id'], 'reason': 'required_context_budget'})
    title_sources = [{'kind': 'title_path', 'node_id': n['node_id'],
                      'start': n['start'], 'end': n['end']} for _, n in titles]
    result['sources'] = title_sources + result['sources']
    result['prefix'] = '\n'.join([v for v, _ in titles] + parts)
    return result


def contextualize(doc, start, end, cfg):
    context = build_context(doc, start, end, cfg)
    body = doc.text[start:end]
    value = context['prefix'] + '\n' + body if context['prefix'] else body
    return value, context
