from dataclasses import asdict
from collections import Counter
import math

from . import __version__
from .context import contextualize
from .model import PipelineError, digest
from .strategies import STRATEGIES, HEADINGS

SCHEMA_VERSION = '2.0'


def validate_records(doc, records, cfg):
    cursor = 0
    ids = [r['chunk_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise PipelineError('Duplicate chunk IDs')
    for i, rec in enumerate(records):
        if (rec['start'] != cursor or rec['end'] <= cursor
                or rec['text'] != doc.text[rec['start']:rec['end']]
                or not rec['text'].strip()):
            raise PipelineError('Non-contiguous or incorrect source slice')
        if rec['chars'] != len(rec['text']) or rec['contextual_chars'] != len(rec['contextual_text']):
            raise PipelineError('Incorrect length accounting')
        if rec['contextual_text'] != contextualize(doc, rec['start'], rec['end'], cfg)[0]:
            raise PipelineError('Retrieval serialization differs from declared context policy')
        if rec['contextual_chars'] > cfg.limit:
            raise PipelineError('Absolute budget exceeded')
        if rec['over_target'] and (cfg.budget_policy != 'protect' or not rec['over_target_reason']):
            raise PipelineError('Unaccounted target overflow')
        if any(a.table_id for a in doc.atoms if a.atom_id in rec['atom_ids']) and rec['over_target']:
            raise PipelineError('Table exceeds target budget')
        if rec['prev'] != (ids[i - 1] if i else None) or rec['next'] != (ids[i + 1] if i + 1 < len(ids) else None):
            raise PipelineError('Broken chunk chain')
        if rec['continued'] and rec['parent_id'] not in ids[:i]:
            raise PipelineError('Continuation must point to an earlier chunk')
        if not rec['continued'] and rec['parent_id'] is not None:
            raise PipelineError('Non-continuation has a parent chunk')
        cursor = rec['end']
    if cursor != len(doc.text) or ''.join(r['text'] for r in records) != doc.text:
        raise PipelineError('Full reconstruction failed')


def chunk_document(doc, cfg, token_counter=None, tokenizer_id=None):
    if cfg.max_embedding_tokens is not None and (token_counter is None or not tokenizer_id):
        raise PipelineError('max_embedding_tokens requires a real token_counter and tokenizer_id')
    pieces = STRATEGIES[cfg.boundary_strategy](doc, cfg)
    records, first_chunks = [], {}
    for i, piece in enumerate(pieces):
        text = doc.text[piece.start:piece.end]
        contextual, context = contextualize(doc, piece.start, piece.end, cfg)
        aid = piece.atoms[0].atom_id
        continued = piece.start > piece.atoms[0].start
        parent = first_chunks.get(aid) if continued else None
        cid = '%s-C%04d' % (doc.doc_id, i + 1)
        for atom in piece.atoms:
            first_chunks.setdefault(atom.atom_id, cid)
        scope_ids = next((a.scope for a in piece.atoms if a.kind not in HEADINGS), piece.atoms[-1].scope)
        kinds = set(a.kind for a in piece.atoms)
        sources = [n['data_src'] for n in doc.nodes.values() if n['data_src']
                   and n['start'] < piece.end and n['end'] > piece.start]
        images = [r['src'] for r in doc.resources if r['kind'] == 'img' and r['src']
                  and (piece.start <= r['start'] < piece.end
                       or (i == len(pieces) - 1 and r['start'] == piece.end))]
        over = len(contextual) > cfg.target_chars
        rec = {'schema_version': SCHEMA_VERSION, 'chunk_id': cid, 'doc_id': doc.doc_id,
               'url': doc.url, 'start': piece.start, 'end': piece.end, 'text': text,
               'contextual_text': contextual, 'chars': len(text), 'contextual_chars': len(contextual),
               'context_sources': context['sources'], 'context_trimmed': context['trimmed'],
               'heading_path': [doc.text[doc.nodes[n]['start']:doc.nodes[n]['end']] for n in scope_ids],
               'parent_id': parent, 'continued': continued,
               'prev': records[-1]['chunk_id'] if records else None, 'next': None,
               'data_src': list(dict.fromkeys(sources)), 'kind': next(iter(kinds)) if len(kinds) == 1 else 'mixed',
               'split_reason': piece.split_reason, 'cross_section': len({tuple(a.scope) for a in piece.atoms if a.kind not in HEADINGS}) > 1,
               'images': list(dict.fromkeys(images)), 'atom_ids': [a.atom_id for a in piece.atoms],
               'chunker_version': __version__, 'normalizer_version': doc.normalizer_version,
               'config_hash': cfg.config_hash, 'over_target': over,
               'over_target_reason': piece.allowance if over else None, 'absolute_limit': cfg.limit,
               'source_spans': [{'start': piece.start, 'end': piece.end}],
               'orphan_headings': []}
        if cfg.max_embedding_tokens is not None:
            count = token_counter(contextual)
            if type(count) is not int or count < 0:
                raise PipelineError('token_counter must return a nonnegative integer')
            if count > cfg.max_embedding_tokens:
                raise PipelineError('Embedding token limit exceeded for ' + cid)
            rec['embedding_tokens'] = count
        if records:
            records[-1]['next'] = cid
        records.append(rec)
    # Account for any heading ending a record without its following content.
    for rec, piece in zip(records, pieces):
        for atom in piece.atoms:
            if atom.kind not in HEADINGS or atom.end > piece.end:
                continue
            index = doc.atoms.index(atom)
            following = next((a for a in doc.atoms[index + 1:] if a.kind not in HEADINGS), None)
            if following is None or following.start >= piece.end:
                reason = 'end_of_document' if following is None else (
                    'metadata_separation' if following.metadata_id and cfg.metadata_mode == 'separate' else 'budget_or_structure_boundary')
                rec['orphan_headings'].append({'atom_id': atom.atom_id, 'reason': reason})
    validate_records(doc, records, cfg)
    lengths = sorted(r['contextual_chars'] for r in records)

    def percentile(p):
        return lengths[max(0, math.ceil(len(lengths) * p) - 1)] if lengths else 0

    split_atoms = {a.atom_id for a in doc.atoms if sum(a.atom_id in r['atom_ids'] for r in records) > 1}
    unit_lengths = {}
    for kind in sorted({a.kind for a in doc.atoms}):
        values = sorted(a.end - a.start for a in doc.atoms if a.kind == kind)
        unit_lengths[kind] = {'count': len(values), 'p95': values[math.ceil(len(values) * .95) - 1],
                              'max': values[-1], 'over_target_body_only': sum(n > cfg.target_chars for n in values)}
    report = {'schema_version': SCHEMA_VERSION, 'chunker_version': __version__,
              'normalizer_version': doc.normalizer_version, 'doc_id': doc.doc_id,
              'url': doc.url, 'html_sha256': doc.html_sha256, 'text_sha256': doc.text_sha256,
              'config': asdict(cfg), 'config_hash': cfg.config_hash,
              'status': 'local_experiment_not_production',
              'token_limit_verified': cfg.max_embedding_tokens is not None, 'tokenizer_id': tokenizer_id,
              'stats': {'chunks': len(records), 'atoms': len(doc.atoms), 'normalized_chars': len(doc.text),
                        'reconstruction_ok': True, 'over_target': sum(r['over_target'] for r in records),
                        'over_target_rate': sum(r['over_target'] for r in records) / max(1, len(records)),
                        'over_absolute': 0, 'split_atoms': len(split_atoms),
                        'split_atoms_by_kind': dict(Counter(a.kind for a in doc.atoms if a.atom_id in split_atoms)),
                        'atom_lengths': unit_lengths,
                        'split_reason_counts': dict(Counter(r['split_reason'] for r in records if r['split_reason'])),
                        'over_target_reasons': dict(Counter(r['over_target_reason'] for r in records if r['over_target'])),
                        'orphan_headings': sum(len(r['orphan_headings']) for r in records),
                        'context_trimmed': sum(len(r['context_trimmed']) for r in records),
                        'cross_section_chunks': sum(r['cross_section'] for r in records),
                        'continued_chunks': sum(r['continued'] for r in records),
                        'context_chars': sum(r['contextual_chars'] - r['chars'] for r in records),
                        'lengths': {'p50': percentile(.5), 'p95': percentile(.95), 'p99': percentile(.99),
                                    'max': max(lengths) if lengths else 0},
                        'tables': len(doc.tables),
                        'table_data_rows': sum(not row['is_header'] and row['section'] != 'tfoot'
                                               for t in doc.tables.values() for row in t['rows']),
                        'resources': len(doc.resources)},
              'warnings': doc.warnings}
    return records, report


def llm_cache_key(doc, record):
    return digest({'document_sha256': doc.text_sha256,
                   'start': record['start'], 'end': record['end'],
                   'contextual_text_sha256': digest(record['contextual_text'])})


def retrieval_views(doc, records, llm_contexts=None, absolute_chars=None,
                    token_counter=None, max_tokens=None, tokenizer_id=None):
    """One-to-one index adapter; external LLM context is a fixed-boundary ablation.

    llm_contexts maps llm_cache_key(doc, record) to context + provenance.
    No external API is called. Missing/stale entries fail explicitly.
    """
    if max_tokens is not None and (token_counter is None or not tokenizer_id):
        raise PipelineError('Retrieval token limit requires a real tokenizer')
    if absolute_chars is not None and (type(absolute_chars) is not int or absolute_chars < 1):
        raise PipelineError('View absolute_chars must be a positive integer')
    if llm_contexts is not None and not isinstance(llm_contexts, dict):
        raise PipelineError('Cached LLM contexts must be an object keyed by request hash')
    views = []
    for rec in records:
        value = rec['contextual_text']
        provenance = None
        if llm_contexts is not None:
            key = llm_cache_key(doc, rec)
            entry = llm_contexts.get(key)
            if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) and entry[k].strip()
                                                     for k in ('text', 'model', 'prompt_version')):
                raise PipelineError('Missing valid cached LLM context for request key ' + key)
            if entry.get('document_sha256') != doc.text_sha256:
                raise PipelineError('LLM context document hash mismatch')
            value = entry['text'] + '\n' + value
            provenance = {'cache_key': key, 'source_text_sha256': digest(rec['text']), 'model': entry['model'],
                          'prompt_version': entry['prompt_version'], 'context_sha256': digest(entry['text'])}
        limit = absolute_chars if absolute_chars is not None else rec['absolute_limit']
        if len(value) > limit:
            raise PipelineError('Enhanced retrieval view exceeds absolute_chars: ' + rec['chunk_id'])
        count = token_counter(value) if max_tokens is not None else None
        if count is not None and (type(count) is not int or count < 0 or count > max_tokens):
            raise PipelineError('Enhanced view exceeds or has invalid token count')
        views.append({'view_id': rec['chunk_id'] + '-V-' + digest(value)[:12],
                      'chunk_id': rec['chunk_id'], 'doc_id': doc.doc_id, 'url': doc.url,
                      'index_text': value, 'source_spans': rec['source_spans'],
                      'index_text_sha256': digest(value), 'llm_context': provenance,
                      'document_sha256': doc.text_sha256, 'config_hash': rec['config_hash'],
                      'chars': len(value), 'absolute_limit': limit, 'embedding_tokens': count,
                      'tokenizer_id': tokenizer_id, 'role': 'retrieval_only_do_not_concatenate'})
    return views
