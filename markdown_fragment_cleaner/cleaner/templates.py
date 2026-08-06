from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from .config import TemplateConfig
from .models import Block, Document


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/.年](?:0?[1-9]|1[0-2])(?:[-/.月](?:0?[1-9]|[12]\d|3[01])日?)?\b"
)
_LONG_NUMBER_RE = re.compile(r"\b\d{2,}\b")
_HEX_ID_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def canonical_host(url: str) -> str:
    if not url:
        return "__unknown__"
    parsed = urlparse(url if "://" in url else "https://" + url)
    host = (parsed.hostname or "__unknown__").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def normalize_for_template(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = _URL_RE.sub("<url>", value)
    value = _EMAIL_RE.sub("<email>", value)
    value = _DATE_RE.sub("<date>", value)
    value = _HEX_ID_RE.sub("<id>", value)
    value = _LONG_NUMBER_RE.sub("<num>", value)
    return _SPACE_RE.sub(" ", value).strip()


def fingerprint(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()


@dataclass
class TemplateEntry:
    host: str
    fingerprint: str
    normalized_text: str
    document_count: int
    host_document_count: int
    edge_count: int
    reason: str
    example: str

    @property
    def document_fraction(self) -> float:
        return self.document_count / self.host_document_count if self.host_document_count else 0.0

    @property
    def edge_ratio(self) -> float:
        return self.edge_count / self.document_count if self.document_count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["document_fraction"] = self.document_fraction
        value["edge_ratio"] = self.edge_ratio
        return value


class TemplateIndex:
    """Learn exact normalized, host-specific boilerplate with bounded memory."""

    def __init__(self, config: Optional[TemplateConfig] = None) -> None:
        self.config = config or TemplateConfig()
        self.config.validate()
        self.entries: List[TemplateEntry] = []
        self._index: Dict[Tuple[str, str], TemplateEntry] = {}

    def fit(self, documents: Iterable[Document]) -> "TemplateIndex":
        if not self.config.enabled:
            self.entries = []
            self._index = {}
            return self

        with tempfile.TemporaryDirectory(prefix="markdown-template-", dir=self.config.work_dir) as directory:
            database = str(Path(directory) / "template_counts.sqlite3")
            connection = sqlite3.connect(database)
            try:
                _initialize_database(connection)
                cursor = connection.cursor()
                processed = 0
                for document in documents:
                    host = canonical_host(document.url)
                    # Without a host, corpus-wide repetition is unsafe: common
                    # phrases could be mistaken for a site template.
                    if host == "__unknown__":
                        continue
                    blocks = [block for block in document.blocks if block.text.strip()]
                    signature = fingerprint("\x1e".join(normalize_for_template(block.text) for block in blocks))
                    cursor.execute(
                        "INSERT OR IGNORE INTO seen_documents(host, signature) VALUES (?, ?)",
                        (host, signature),
                    )
                    if cursor.rowcount == 0:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO host_stats(host, document_count) VALUES (?, 1)
                        ON CONFLICT(host) DO UPDATE SET document_count = document_count + 1
                        """,
                        (host,),
                    )

                    last_index = len(blocks) - 1
                    edge_width = min(self.config.edge_blocks, max(1, len(blocks) // 4))
                    per_document: Dict[str, Tuple[str, str, int]] = {}
                    for index, block in enumerate(blocks):
                        normalized = normalize_for_template(block.text)
                        if len(normalized) < self.config.min_block_chars:
                            continue
                        block_fingerprint = fingerprint(normalized)
                        is_edge = int(index < edge_width or index > last_index - edge_width)
                        previous = per_document.get(block_fingerprint)
                        if previous is None or (is_edge and not previous[2]):
                            per_document[block_fingerprint] = (normalized, block.text[:500], is_edge)

                    cursor.executemany(
                        """
                        INSERT INTO block_stats(
                            host, fingerprint, normalized_text, example, document_count, edge_count
                        ) VALUES (?, ?, ?, ?, 1, ?)
                        ON CONFLICT(host, fingerprint) DO UPDATE SET
                            document_count = document_count + 1,
                            edge_count = edge_count + excluded.edge_count
                        """,
                        [
                            (host, key, normalized, example, is_edge)
                            for key, (normalized, example, is_edge) in per_document.items()
                        ],
                    )
                    processed += 1
                    if processed % 1000 == 0:
                        connection.commit()
                connection.commit()
                self.entries = self._select_entries(connection)
            finally:
                connection.close()

        self.entries.sort(key=lambda entry: (entry.host, -entry.document_count, entry.fingerprint))
        self._index = {(entry.host, entry.fingerprint): entry for entry in self.entries}
        return self

    def match(self, url: str, block: Block) -> Optional[TemplateEntry]:
        host = canonical_host(url)
        if host == "__unknown__":
            return None
        normalized = normalize_for_template(block.text)
        if len(normalized) < self.config.min_block_chars:
            return None
        return self._index.get((host, fingerprint(normalized)))

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "method": "host_normalized_exact_document_frequency",
            "config": asdict(self.config),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _select_entries(self, connection: sqlite3.Connection) -> List[TemplateEntry]:
        rows = connection.execute(
            """
            SELECT b.host, b.fingerprint, b.normalized_text, b.example,
                   b.document_count, b.edge_count, h.document_count
            FROM block_stats AS b
            JOIN host_stats AS h ON h.host = b.host
            WHERE h.document_count >= ? AND b.document_count >= ?
            """,
            (self.config.min_host_documents, self.config.min_template_documents),
        )
        entries: List[TemplateEntry] = []
        for host, key, normalized, example, doc_count, edge_count, host_count in rows:
            document_fraction = doc_count / host_count
            edge_ratio = edge_count / doc_count
            reason = ""
            if document_fraction >= self.config.min_document_fraction and edge_ratio >= self.config.min_edge_ratio:
                reason = "repeated_at_page_edge"
            elif (
                document_fraction >= self.config.ubiquitous_document_fraction
                and len(normalized) <= self.config.max_ubiquitous_chars
                and edge_ratio >= self.config.min_ubiquitous_edge_ratio
            ):
                reason = "ubiquitous_short_block"
            if reason:
                entries.append(
                    TemplateEntry(
                        host=host,
                        fingerprint=key,
                        normalized_text=normalized,
                        document_count=doc_count,
                        host_document_count=host_count,
                        edge_count=edge_count,
                        reason=reason,
                        example=example,
                    )
                )
        return entries


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE seen_documents (
            host TEXT NOT NULL,
            signature TEXT NOT NULL,
            PRIMARY KEY(host, signature)
        ) WITHOUT ROWID;
        CREATE TABLE host_stats (
            host TEXT PRIMARY KEY,
            document_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE block_stats (
            host TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            example TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            PRIMARY KEY(host, fingerprint)
        ) WITHOUT ROWID;
        """
    )
