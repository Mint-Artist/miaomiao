"""Write a complete run atomically; never overwrite an existing experiment."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import platform
import bs4

from .model import PipelineError, digest
from .pipeline import llm_cache_key


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def write_run(path, doc, records, report, views, simplified_html=None):
    if simplified_html is not None and digest(simplified_html) != doc.html_sha256:
        raise PipelineError('Simplified HTML does not match normalized document')
    path = Path(path)
    if path.exists():
        raise PipelineError('Output already exists; use a new experiment directory: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.chunk-run-', dir=str(path.parent)))
    try:
        payloads = {'normalized.txt': doc.text,
                    'document.json': dumps(doc.to_dict()),
                    'chunks.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records),
                    'retrieval.jsonl': ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in views),
                    'report.json': dumps(report),
                    'llm_requests.jsonl': ''.join(json.dumps({
                        'cache_key': llm_cache_key(doc, r), 'document_sha256': doc.text_sha256,
                        'chunk_id': r['chunk_id'], 'start': r['start'], 'end': r['end'],
                        'text': r['text'], 'contextual_text': r['contextual_text'],
                        'required_response_fields': ['text', 'model', 'prompt_version', 'document_sha256']},
                        ensure_ascii=False) + '\n' for r in records)}
        if simplified_html is not None:
            payloads['simplified.html'] = simplified_html
        for name, content in payloads.items():
            (stage / name).write_bytes(content.encode('utf-8'))
        (stage / 'manifest.json').write_text(dumps({
            'status': 'local_experiment_not_production', 'doc_id': doc.doc_id,
            'config_hash': report['config_hash'],
            'runtime': {'python': platform.python_version(), 'beautifulsoup4': bs4.__version__},
            'implementation_sha256': {p.name: digest(p.read_text(encoding='utf-8'))
                                      for p in sorted(Path(__file__).parent.glob('*.py'))},
            'view_config_hash': digest([{'context': v['llm_context'], 'limit': v['absolute_limit'],
                                        'tokenizer': v['tokenizer_id'], 'text_hash': v['index_text_sha256']}
                                       for v in views]),
            'files_sha256': {name: digest(content) for name, content in payloads.items()}}), encoding='utf-8')
        if path.exists():
            raise PipelineError('Output directory appeared during generation')
        os.rename(str(stage), str(path))
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
