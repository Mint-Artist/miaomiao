from dataclasses import asdict, dataclass, field
from typing import Optional
import hashlib
import json


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class PipelineError(ValueError):
    """An explicit failure; never return a partially covered document."""


@dataclass(frozen=True)
class ChunkConfig:
    target_chars: int = 256
    budget_policy: str = 'hard'
    absolute_chars: Optional[int] = None
    boundary_strategy: str = 'structure'
    context_mode: str = 'none'
    title_policy: str = 'ancestors'
    max_title_chars: int = 100
    metadata_mode: str = 'inline'
    merge_short_sections: bool = False
    max_embedding_tokens: Optional[int] = None

    def __post_init__(self):
        for name in ('target_chars', 'max_title_chars'):
            n = getattr(self, name)
            if type(n) is not int or n < (8 if name == 'target_chars' else 0):
                raise PipelineError('Invalid ' + name)
        for name in ('absolute_chars', 'max_embedding_tokens'):
            n = getattr(self, name)
            if n is not None and (type(n) is not int or n < 1):
                raise PipelineError('Invalid ' + name)
        choices = {'budget_policy': {'hard', 'protect'},
                   'boundary_strategy': {'structure', 'recursive'},
                   'context_mode': {'none', 'structure'},
                   'title_policy': {'ancestors', 'document_nearest'},
                   'metadata_mode': {'inline', 'separate'}}
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise PipelineError('Unsupported ' + name)
        if type(self.merge_short_sections) is not bool:
            raise PipelineError('merge_short_sections must be boolean')
        if self.absolute_chars is not None and self.absolute_chars < self.target_chars:
            raise PipelineError('absolute_chars must be >= target_chars')
        if self.budget_policy == 'protect' and self.absolute_chars is None:
            raise PipelineError('protect requires an explicit experimental absolute_chars; no production default exists')
        if self.boundary_strategy == 'recursive' and (
                self.budget_policy != 'hard' or self.context_mode != 'none'
                or self.metadata_mode != 'inline' or self.merge_short_sections):
            raise PipelineError('recursive baseline supports hard budget, no context, inline metadata only')

    @property
    def limit(self):
        return self.target_chars if self.budget_policy == 'hard' else self.absolute_chars

    @property
    def config_hash(self):
        return digest(asdict(self))[:16]

    @classmethod
    def from_dict(cls, obj):
        try:
            return cls(**obj)
        except TypeError as exc:
            raise PipelineError('Unknown or invalid config field: ' + str(exc)) from exc


@dataclass
class Atom:
    atom_id: str
    kind: str
    start: int
    end: int
    scope: list = field(default_factory=list)
    table_id: Optional[str] = None
    metadata_id: Optional[str] = None


@dataclass
class Document:
    doc_id: str
    url: str
    text: str
    html_sha256: str
    nodes: dict
    atoms: list
    tables: dict
    resources: list
    warnings: list
    normalizer_version: str = '1.0.0'

    @property
    def text_sha256(self):
        return digest(self.text)

    def to_dict(self):
        value = asdict(self)
        value['text_sha256'] = self.text_sha256
        value['offset_unit'] = 'unicode_code_point'
        return value


@dataclass
class Piece:
    start: int
    end: int
    atoms: list
    split_reason: Optional[str] = None
    allowance: Optional[str] = None
