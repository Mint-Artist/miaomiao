"""Boundary strategies share normalization, exact slicing, and serialization."""
import re
from .context import contextualize, overlapping_atoms
from .model import Piece, PipelineError

HEADINGS = {'heading', 'pseudo_heading'}
CLOSERS = '》」』”’）】)]}'


def size(doc, start, end, cfg):
    return len(contextualize(doc, start, end, cfg)[0])


def choose_cut(doc, start, stop, bound, atoms, use_structure=True):
    if bound >= stop:
        return stop, None
    text = doc.text
    table = use_structure and any(a.table_id for a in atoms)
    ends = []
    if table:
        for tid in {a.table_id for a in atoms if a.table_id}:
            for row in doc.tables[tid]['rows']:
                for cell in row['cells']:
                    end = cell['end']
                    while end < stop and end < bound and text[end] in '\t\n':
                        end += 1
                    if start < end <= bound:
                        ends.append(end)
    elif use_structure:
        for node in doc.nodes.values():
            if node['tag'] in ('p', 'li', 'pre', 'blockquote'):
                end = node['end']
                while end < stop and end < bound and text[end] == '\n':
                    end += 1
                if start < end <= bound:
                    ends.append(end)
    reason = 'table_cell' if table else 'child_block'
    cut = max(ends) if ends else None
    if cut is None:
        for label, pattern in [('line', r'\n+'), ('sentence', r'[。！？!?]'),
                               ('semicolon', r'[；;]'), ('clause', r'[，,、]')]:
            positions = []
            for match in re.finditer(pattern, text[start:bound]):
                end = start + match.end()
                while end < stop and text[end] in CLOSERS:
                    end += 1
                if end <= bound and text[start:end].strip():
                    positions.append(end)
            if positions:
                cut, reason = positions[-1], label
                break
    if cut is None:
        cut, reason = bound, 'hard_wrap'
    # Do not leave a separator-only trailing chunk at exact length boundaries.
    if not text[cut:stop].strip():
        while cut > start and not text[cut:stop].strip():
            cut -= 1
        reason = 'hard_wrap'
    if cut <= start or not text[start:cut].strip():
        raise PipelineError('Cannot split without a whitespace-only chunk at offset %d' % start)
    return cut, reason


def split_range(doc, start, end, atoms, cfg, limit, use_structure=True):
    pieces = []
    while start < end:
        if size(doc, start, end, cfg) <= limit:
            pieces.append(Piece(start, end, overlapping_atoms(doc, start, end)))
            break
        bound = min(end, start + limit)
        while bound > start:
            excess = size(doc, start, bound, cfg) - limit
            if excess <= 0:
                break
            bound -= max(1, excess)
        if bound <= start:
            raise PipelineError('Required context cannot fit the budget at offset %d; no text was dropped' % start)
        cut, reason = choose_cut(doc, start, end, bound, atoms, use_structure)
        # Partial rows introduce coordinate context; recheck the actual cut.
        while size(doc, start, cut, cfg) > limit:
            bound = cut - max(1, size(doc, start, cut, cfg) - limit)
            if bound <= start:
                raise PipelineError('Table/context budget leaves no room at offset %d' % start)
            cut, reason = choose_cut(doc, start, end, bound, atoms, use_structure)
        pieces.append(Piece(start, cut, overlapping_atoms(doc, start, cut), reason))
        start = cut
    if len(pieces) > 1 and pieces[-1].split_reason is None:
        pieces[-1].split_reason = pieces[-2].split_reason
    return pieces


def _family(piece, cfg):
    atom = next((a for a in piece.atoms if a.kind not in HEADINGS), piece.atoms[-1])
    if atom.table_id:
        return ('table', atom.table_id)
    if atom.metadata_id and cfg.metadata_mode == 'separate':
        return ('metadata', atom.metadata_id)
    if atom.kind in ('note', 'pre', 'quote'):
        return (atom.kind, atom.atom_id)
    return ('body',)


def _scope(piece):
    return next((a.scope for a in piece.atoms if a.kind not in HEADINGS), piece.atoms[-1].scope)


def structural(doc, cfg):
    bundles = []
    i = 0
    while i < len(doc.atoms):
        atom = doc.atoms[i]
        bundle = [atom]
        i += 1
        if atom.kind in HEADINGS:
            while i < len(doc.atoms) and doc.atoms[i].kind in HEADINGS:
                bundle.append(doc.atoms[i])
                i += 1
            if i < len(doc.atoms):
                nxt = doc.atoms[i]
                if not nxt.table_id and not (nxt.metadata_id and cfg.metadata_mode == 'separate'):
                    bundle.append(nxt)
                    i += 1
        elif atom.kind == 'table_header':
            while (i < len(doc.atoms) and doc.atoms[i].kind == 'table_header'
                   and doc.atoms[i].table_id == atom.table_id):
                bundle.append(doc.atoms[i])
                i += 1
            if (i < len(doc.atoms) and doc.atoms[i].kind == 'row'
                    and doc.atoms[i].table_id == atom.table_id):
                bundle.append(doc.atoms[i])
                i += 1
        bundles.append(bundle)

    expanded = []
    for bundle in bundles:
        limit = cfg.target_chars if any(a.table_id for a in bundle) else cfg.limit
        start, end = bundle[0].start, bundle[-1].end
        if size(doc, start, end, cfg) <= limit:
            reason = 'heading_binding' if len(bundle) > 1 and bundle[0].kind in HEADINGS else 'protected_unit'
            expanded.append(Piece(start, end, bundle, allowance=reason))
        else:
            # A binding must not force splitting a paragraph which alone fits H.
            for atom in bundle:
                parts = split_range(doc, atom.start, atom.end, [atom], cfg,
                                    cfg.target_chars if atom.table_id else cfg.limit)
                for p in parts:
                    p.allowance = 'protected_unit' if not atom.table_id else None
                expanded.extend(parts)
    chunks, current = [], None
    for piece in expanded:
        starts_heading = (piece.atoms[0].kind in HEADINGS
                          and piece.start == piece.atoms[0].start)
        boundary = current is not None and (
            _family(current, cfg) != _family(piece, cfg)
            or (not cfg.merge_short_sections and
                (starts_heading or _scope(current) != _scope(piece))))
        can_merge = (current is not None and not boundary
                     and size(doc, current.start, current.end, cfg) <= cfg.target_chars
                     and size(doc, current.start, piece.end, cfg) <= cfg.target_chars)
        if can_merge:
            current = Piece(current.start, piece.end,
                            overlapping_atoms(doc, current.start, piece.end),
                            current.split_reason or piece.split_reason)
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def recursive(doc, cfg):
    if not doc.text:
        return []
    return split_range(doc, 0, len(doc.text), doc.atoms, cfg, cfg.target_chars, use_structure=False)


STRATEGIES = {'structure': structural, 'recursive': recursive}
