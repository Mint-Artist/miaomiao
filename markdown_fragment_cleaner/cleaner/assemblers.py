from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from .chunker import SemanticChunker
from .models import (
    Document,
    Fragment,
    PreparedBlock,
    RuleFlag,
    SectionSpan,
    WholeDocumentRecord,
)
from .tokenization import TokenCounter


class FragmentAssembler:
    """Compatibility wrapper around the existing semantic chunker."""

    def __init__(self, chunker: SemanticChunker) -> None:
        self.chunker = chunker

    def assemble(
        self,
        document: Document,
        prepared: Sequence[Optional[PreparedBlock]],
    ) -> List[Fragment]:
        return self.chunker.chunk_document(document, prepared)


class WholeDocumentAssembler:
    """Build at most one cleaned record while retaining internal block structure."""

    def __init__(self, token_counter: TokenCounter) -> None:
        self.counter = token_counter

    def assemble(
        self,
        document: Document,
        prepared: Sequence[Optional[PreparedBlock]],
        removed_blocks: Sequence[Dict[str, Any]],
    ) -> Optional[WholeDocumentRecord]:
        kept = [item for item in prepared if item is not None and item.block.text.strip()]
        if not kept:
            return None

        parts: List[str] = []
        sections: List[SectionSpan] = []
        offset = 0
        for item in kept:
            text = item.block.text.strip()
            if parts:
                offset += 2
            start = offset
            parts.append(text)
            offset += len(text)
            end = offset

            path = list(item.block.heading_path)
            if sections and sections[-1].heading_path == path:
                sections[-1].block_ids.append(item.block.block_id)
                sections[-1].end_char = end
            else:
                sections.append(
                    SectionSpan(
                        heading_path=path,
                        block_ids=[item.block.block_id],
                        start_char=start,
                        end_char=end,
                    )
                )

        content = "\n\n".join(parts)
        block_ids = [item.block.block_id for item in kept]
        raw_id = "%s\x1f%d\x1f%s\x1f%s" % (
            document.doc_id,
            document.source_row,
            ",".join(block_ids),
            content,
        )
        digest = hashlib.blake2b(raw_id.encode("utf-8"), digest_size=12).hexdigest()

        flags: List[RuleFlag] = []
        seen_flags = set()
        repairs: List[Dict[str, Any]] = []
        for item in kept:
            for flag in item.flags:
                key = (flag.code, flag.severity, flag.detail)
                if key not in seen_flags:
                    flags.append(flag)
                    seen_flags.add(key)
            block_repairs = item.block.metadata.get("repairs", [])
            if isinstance(block_repairs, list):
                repairs.extend(repair for repair in block_repairs if isinstance(repair, dict))

        metadata: Dict[str, Any] = {"assembler_version": "whole-document-v1"}
        if repairs:
            metadata["repairs"] = repairs
        return WholeDocumentRecord(
            record_id="docrec-" + digest,
            doc_id=document.doc_id,
            url=document.url,
            doc_title=document.title,
            content=content,
            block_ids=block_ids,
            block_types=[item.block.type for item in kept],
            token_count=self.counter.count(content),
            start_line=min(item.block.start_line for item in kept),
            end_line=max(item.block.end_line for item in kept),
            source_row=document.source_row,
            sections=sections,
            removed_blocks=list(removed_blocks),
            flags=flags,
            metadata=metadata,
        )
