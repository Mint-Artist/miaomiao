from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuleFlag:
    code: str
    severity: str
    detail: str = ""
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Block:
    block_id: str
    type: str
    text: str
    raw_text: str
    heading_path: List[str]
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_structured(self) -> bool:
        return self.type in {"list", "table", "code"}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    doc_id: str
    url: str
    title: str
    blocks: List[Block]
    source_row: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedBlock:
    block: Block
    flags: List[RuleFlag] = field(default_factory=list)


@dataclass
class Fragment:
    fragment_id: str
    doc_id: str
    url: str
    doc_title: str
    heading_path: List[str]
    content: str
    block_ids: List[str]
    block_types: List[str]
    token_count: int
    start_line: int
    end_line: int
    source_row: int
    context_before: str = ""
    context_after: str = ""
    flags: List[RuleFlag] = field(default_factory=list)
    decision: str = "accepted"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "doc_id": self.doc_id,
            "url": self.url,
            "doc_title": self.doc_title,
            "heading_path": self.heading_path,
            "content": self.content,
            "block_ids": self.block_ids,
            "block_types": self.block_types,
            "token_count": self.token_count,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_row": self.source_row,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "flags": [flag.to_dict() for flag in self.flags],
            "decision": self.decision,
            "metadata": self.metadata,
        }
