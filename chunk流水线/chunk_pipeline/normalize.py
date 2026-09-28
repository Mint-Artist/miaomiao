"""One canonical projection of simplified HTML, independent of chunk policy.

Renderer events retain DOM boundaries even for zero-width images/empty cells.
All coordinates refer to normalized text, not raw HTML byte positions.
"""
import re
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .model import Atom, Document, PipelineError, digest

BLOCKS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol',
          'li', 'blockquote', 'figure', 'caption', 'section', 'article',
          'main', 'html', 'body'}
NOTES = {'note', 'warning', 'tip', 'info', 'important', 'caution'}


def _has_text(events):
    return any(e[0] == 'char' for e in events)


def _clean_run(events):
    # Fold spaces across inline tags, without dropping DOM boundary events.
    out = []
    last_char = None
    for e in events:
        if e[0] == 'char':
            if e[1] == ' ' and last_char == ' ':
                continue
            last_char = e[1]
        out.append(e)
    indices = [i for i, e in enumerate(out) if e[0] == 'char']
    remove = set()
    for j, i in enumerate(indices):
        char = out[i][1]
        before = out[indices[j - 1]][1] if j else None
        after = out[indices[j + 1]][1] if j + 1 < len(indices) else None
        if char == ' ' and (before in (None, '\n') or after in (None, '\n')):
            remove.add(i)
    kept = [i for i in indices if i not in remove]
    for sequence in (kept, reversed(kept)):
        for i in sequence:
            if out[i][1] not in ' \n':
                break
            remove.add(i)
    return [e for i, e in enumerate(out) if i not in remove]


def normalize_html(html, doc_id, url=''):
    if not isinstance(html, str) or not doc_id:
        raise PipelineError('html must be text and doc_id must be nonempty')
    soup = BeautifulSoup(html, 'html.parser')
    if soup.find(['script', 'style', 'iframe', 'noscript']):
        raise PipelineError('Expected simplified HTML; active/noise tags must be removed by the extractor')
    # Nested tables require a dedicated retrieval policy. Fail rather than
    # quietly representing an inner row as an outer row.
    if any(t.find_parent('table') for t in soup.find_all('table')):
        raise PipelineError('Nested tables are not supported in v0.2; retain input for review')
    tags = list(soup.find_all(True))
    ids = {id(t): 'n%d' % (i + 1) for i, t in enumerate(tags)}
    nodes = {}
    for tag in tags:
        node_id = ids[id(tag)]
        parent = ids.get(id(tag.parent))
        nodes[node_id] = {'node_id': node_id, 'tag': tag.name, 'parent_id': parent,
                          'data_src': tag.get('data-src'), 'attrs': dict(tag.attrs),
                          'start': 0, 'end': 0}

    def char_events(text, owner):
        return [('char', c, owner) for c in text]

    def join_parts(parts, owner, separator='\n'):
        out, seen = [], False
        for part in parts:
            if _has_text(part):
                if seen:
                    out += char_events(separator, owner)
                seen = True
            out += part
        return out

    def render(node, owner=None, verbatim=False):
        if isinstance(node, Comment):
            return []
        if isinstance(node, NavigableString):
            value = str(node)
            if verbatim:
                value = value.replace('\r\n', '\n').replace('\r', '\n')
            else:
                value = re.sub(r'[ \t\r\n\f]+', ' ', value)
            return char_events(value, owner)
        if not isinstance(node, Tag):
            return []
        nid = ids[id(node)]
        head, tail = [('open', '', nid)], [('close', '', nid)]
        if node.name == 'img':
            return head + tail
        if node.name == 'br':
            return head + char_events('\n', nid) + tail
        if node.name == 'pre' or verbatim:
            return head + [e for c in node.children for e in render(c, nid, True)] + tail
        if node.name == 'tr':
            events = []
            for i, cell in enumerate(node.find_all(['td', 'th'], recursive=False)):
                if i:
                    events += char_events('\t', nid)
                events += render(cell, nid)
            return head + events + tail
        if node.name in ('table', 'thead', 'tbody', 'tfoot'):
            parts = [render(c, nid) for c in node.children if isinstance(c, Tag)]
            return head + join_parts(parts, nid) + tail
        if node.name not in BLOCKS and node.name not in ('td', 'th'):
            return head + [e for c in node.children for e in render(c, nid)] + tail
        parts, run = [], []
        for child in node.children:
            if isinstance(child, Tag) and (child.name in BLOCKS or child.name in ('table', 'pre')):
                parts.append(_clean_run(run))
                run = []
                parts.append(render(child, nid))
            else:
                run.extend(render(child, nid))
        parts.append(_clean_run(run))
        return head + join_parts(parts, nid) + tail

    events = []
    for child in soup.children:
        if isinstance(child, Comment):
            continue
        rendered = render(child)
        if not isinstance(child, Tag):
            rendered = _clean_run(rendered)
        events += rendered
        if _has_text(rendered):
            events += [('char', '\n', ids.get(id(child)))]
    text = []
    for kind, char, nid in events:
        if kind == 'char':
            text.append(char)
        elif kind == 'open':
            nodes[nid]['start'] = len(text)
        elif kind == 'close':
            nodes[nid]['end'] = len(text)
    text = ''.join(text)
    resources = []
    for tag in tags:
        if tag.name in ('img', 'a'):
            node = nodes[ids[id(tag)]]
            resources.append({'node_id': node['node_id'], 'kind': tag.name,
                              'start': node['start'], 'end': node['end'],
                              'src': tag.get('src'), 'href': tag.get('href'),
                              'alt': tag.get('alt')})

    warnings, tables = [], {}
    row_table = {}
    for table in soup.find_all('table'):
        tid = ids[id(table)]
        occupied, rows = {}, []
        for r, row in enumerate(table.find_all('tr')):
            rid = ids[id(row)]
            row_table[rid] = tid
            cells, col = [], 0
            for cell in row.find_all(['th', 'td'], recursive=False):
                while (r, col) in occupied:
                    col += 1
                try:
                    rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
                except (ValueError, TypeError) as exc:
                    raise PipelineError('Invalid table span in ' + tid) from exc
                if not 1 <= rs <= 1000 or not 1 <= cs <= 1000:
                    raise PipelineError('Unsupported table span in ' + tid)
                cid = ids[id(cell)]
                item = {'node_id': cid, 'row': r, 'column': col, 'rowspan': rs,
                        'colspan': cs, 'start': nodes[cid]['start'], 'end': nodes[cid]['end'],
                        'is_header': cell.name == 'th', 'scope': cell.get('scope'),
                        'headers': cell.get('headers')}
                for rr in range(r, r + rs):
                    for cc in range(col, col + cs):
                        if (rr, cc) in occupied:
                            raise PipelineError('Overlapping table spans in ' + tid)
                        occupied[rr, cc] = cid
                cells.append(item)
                col += cs
            is_header = row.find_parent('thead') is not None
            if not is_header and not rows and cells and all(c['is_header'] for c in cells):
                is_header = True
            rows.append({'node_id': rid, 'row': r, 'start': nodes[rid]['start'],
                         'end': nodes[rid]['end'], 'is_header': is_header,
                         'section': row.parent.name, 'cells': cells})
        tables[tid] = {'node_id': tid, 'rows': rows,
                       'header_ids': [r['node_id'] for r in rows if r['is_header']]}
        if rows and not tables[tid]['header_ids']:
            warnings.append({'code': 'table_without_header', 'node_id': tid})

    candidates = []

    def collect(tag, metadata_id=None):
        if not isinstance(tag, Tag):
            return
        nid = ids[id(tag)]
        name, classes = tag.name, set(tag.get('class', []))
        if 'metadata' in classes:
            metadata_id = nid
        node = nodes[nid]
        value = text[node['start']:node['end']]
        kind = None
        if re.fullmatch(r'h[1-6]', name):
            kind = 'heading'
        elif name == 'div' and classes & NOTES:
            kind = 'note'
        elif name in ('p', 'li', 'pre', 'blockquote', 'figure', 'caption'):
            kind = {'blockquote': 'quote'}.get(name, name)
            if name == 'p' and not metadata_id:
                children = [c for c in tag.children if isinstance(c, Tag) or str(c).strip()]
                if (len(children) == 1 and isinstance(children[0], Tag)
                        and children[0].name == 'strong' and 0 < len(value) < 40
                        and not value.endswith(('。', '，', '；', '：'))):
                    kind = 'pseudo_heading'
        elif name == 'tr':
            tid = row_table[nid]
            kind = 'table_header' if nid in tables[tid]['header_ids'] else 'row'
        if kind and value:
            if metadata_id:
                kind = 'metadata'
            candidates.append(Atom(nid, kind, node['start'], node['end'],
                                   table_id=row_table.get(nid), metadata_id=metadata_id))
            return
        for child in tag.children:
            collect(child, metadata_id)

    for child in soup.children:
        collect(child)
    candidates.sort(key=lambda a: a.start)
    atoms, cursor = [], 0
    for atom in candidates:
        if atom.start < cursor:
            raise PipelineError('Overlapping source atoms')
        gap = text[cursor:atom.start]
        if gap.strip():
            atoms.append(Atom('gap%d' % cursor, 'p', cursor, atom.start))
        elif atoms:
            atoms[-1].end = atom.start
        else:
            atom.start = 0
        atoms.append(atom)
        cursor = atom.end
    if cursor < len(text):
        if text[cursor:].strip() or not atoms:
            atoms.append(Atom('gap%d' % cursor, 'p', cursor, len(text)))
        else:
            atoms[-1].end = len(text)

    # Attach whitespace-only rows/empty structures to adjacent visible atoms;
    # their DOM coordinates still remain in the side table.
    compact = []
    pending_start = None
    for atom in atoms:
        if not text[atom.start:atom.end].strip():
            if compact:
                compact[-1].end = atom.end
            else:
                pending_start = atom.start if pending_start is None else pending_start
            continue
        if pending_start is not None:
            atom.start, pending_start = pending_start, None
        compact.append(atom)
    if text and not compact:
        raise PipelineError('Whitespace-only rendered document cannot form a nonempty text chunk')
    atoms = compact
    stack, pseudo = [], None
    for atom in atoms:
        if atom.kind == 'heading':
            level = int(nodes[atom.atom_id]['tag'][1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, atom.atom_id))
            pseudo = None
        elif atom.kind == 'pseudo_heading':
            pseudo = atom.atom_id
        atom.scope = [nid for _, nid in stack] + ([pseudo] if pseudo else [])
    for resource in resources:
        if resource['kind'] == 'img' and not resource['alt']:
            warnings.append({'code': 'image_without_alt', 'node_id': resource['node_id']})
    return Document(doc_id, url, text, digest(html), nodes, atoms, tables, resources, warnings)
