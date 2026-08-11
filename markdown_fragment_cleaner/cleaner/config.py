from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class InputConfig:
    """Dotted paths are supported, for example ``payload.markdown``."""

    markdown_key: str = "content"
    id_key: Optional[str] = "doc_id"
    url_key: Optional[str] = "url"
    title_key: Optional[str] = "title"


@dataclass
class ContentPolicy:
    keep_paragraphs: bool = True
    keep_lists: bool = True
    keep_blockquotes: bool = True
    keep_tables: bool = True
    keep_code_blocks: bool = False
    keep_html_visible_text: bool = True
    keep_image_alt_text: bool = False
    preserve_inline_code_markers: bool = True


@dataclass
class ChunkConfig:
    target_min_tokens: int = 300
    target_max_tokens: int = 768
    hard_max_tokens: int = 1536
    attach_intro_to_structured_block: bool = True
    context_chars: int = 240

    def validate(self) -> None:
        if self.target_min_tokens <= 0:
            raise ValueError("target_min_tokens must be positive")
        if self.target_max_tokens < self.target_min_tokens:
            raise ValueError("target_max_tokens must be >= target_min_tokens")
        if self.hard_max_tokens < self.target_max_tokens:
            raise ValueError("hard_max_tokens must be >= target_max_tokens")
        if self.context_chars < 0:
            raise ValueError("context_chars must not be negative")


@dataclass
class RuleConfig:
    hard_min_tokens: int = 300
    structured_hard_min_tokens: int = 300
    soft_min_tokens: int = 300
    hard_max_tokens: int = 1536
    max_replacement_chars: int = 2
    max_replacement_ratio: float = 0.005
    max_control_ratio: float = 0.002
    max_url_ratio: float = 0.35
    max_duplicate_line_ratio: float = 0.60
    max_repeated_ngram_ratio: float = 0.65
    repeated_character_run: int = 12
    max_markdown_artifact_ratio: float = 0.30
    context_dependent_prefixes: Tuple[str, ...] = (
        "因此", "所以", "但是", "然而", "不过", "此外", "同时", "上述", "前述",
        "由此", "综上", "其中", "对此", "这些", "这种", "该方法", "该系统", "该模型",
    )


@dataclass
class NormalizationConfig:
    """Conservative text repairs applied after AST parsing and before chunking."""

    enabled: bool = True
    softbreak_policy: str = "smart"
    min_structured_softbreak_lines: int = 2
    preserve_complete_sentence_lines: bool = True
    strip_boundary_artifacts: bool = True
    normalize_prose_spacing: bool = True
    deduplicate_adjacent_prose_blocks: bool = True
    deduplicate_repeated_sentence_sequences: bool = True
    min_duplicate_sentence_chars: int = 15
    min_duplicate_sequence_chars: int = 30
    max_duplicate_sequence_sentences: int = 20

    def validate(self) -> None:
        if self.softbreak_policy not in {"smart", "preserve", "unwrap"}:
            raise ValueError("softbreak_policy must be smart, preserve, or unwrap")
        if self.min_structured_softbreak_lines < 2:
            raise ValueError("min_structured_softbreak_lines must be >= 2")
        if self.min_duplicate_sentence_chars <= 0:
            raise ValueError("min_duplicate_sentence_chars must be positive")
        if self.min_duplicate_sequence_chars <= 0:
            raise ValueError("min_duplicate_sequence_chars must be positive")
        if self.max_duplicate_sequence_sentences <= 0:
            raise ValueError("max_duplicate_sequence_sentences must be positive")


@dataclass
class AssemblyConfig:
    """Choose fragment, whole-document, or paired output assembly."""

    output_mode: str = "fragment"
    document_min_tokens: int = 300
    document_max_tokens: int = 6000
    document_over_max_policy: str = "review"

    def validate(self) -> None:
        if self.output_mode not in {"fragment", "document", "both"}:
            raise ValueError("output_mode must be fragment, document, or both")
        if self.document_min_tokens <= 0:
            raise ValueError("document_min_tokens must be positive")
        if self.document_max_tokens < self.document_min_tokens:
            raise ValueError("document_max_tokens must be >= document_min_tokens")
        if self.document_over_max_policy not in {"review", "reject", "allow"}:
            raise ValueError("document_over_max_policy must be review, reject, or allow")


@dataclass
class TemplateConfig:
    enabled: bool = True
    min_host_documents: int = 20
    min_template_documents: int = 5
    min_document_fraction: float = 0.05
    min_edge_ratio: float = 0.70
    ubiquitous_document_fraction: float = 0.50
    min_ubiquitous_edge_ratio: float = 0.20
    max_ubiquitous_chars: int = 500
    edge_blocks: int = 3
    min_block_chars: int = 4
    work_dir: Optional[str] = None

    def validate(self) -> None:
        if self.min_host_documents <= 0 or self.min_template_documents <= 0:
            raise ValueError("template document thresholds must be positive")
        for name in (
            "min_document_fraction",
            "min_edge_ratio",
            "ubiquitous_document_fraction",
            "min_ubiquitous_edge_ratio",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError("%s must be in [0, 1]" % name)


@dataclass
class OutputConfig:
    accepted_path: str = "output/accepted.jsonl"
    review_path: str = "output/review.jsonl"
    rejected_path: str = "output/rejected.jsonl"
    templates_path: str = "output/templates.json"
    statistics_path: str = "output/statistics.json"
    preview_path: str = "output/preview.md"
    preview_fragments: int = 100
    document_accepted_path: str = "output/documents/accepted.jsonl"
    document_review_path: str = "output/documents/review.jsonl"
    document_rejected_path: str = "output/documents/rejected.jsonl"
    document_preview_path: str = "output/documents/preview.md"
    preview_documents: int = 100


@dataclass
class CleanerConfig:
    input_path: str
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    content: ContentPolicy = field(default_factory=ContentPolicy)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    tokenizer_name_or_path: Optional[str] = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)

    def validate(self) -> None:
        if not self.input_path:
            raise ValueError("input_path is required")
        if not self.input.markdown_key:
            raise ValueError("markdown_key is required")
        self.normalization.validate()
        self.assembly.validate()
        self.chunk.validate()
        self.templates.validate()
        if self.rules.hard_min_tokens <= 0:
            raise ValueError("hard_min_tokens must be positive")
        if self.rules.soft_min_tokens < self.rules.hard_min_tokens:
            raise ValueError("soft_min_tokens must be >= hard_min_tokens")
        if self.rules.hard_max_tokens < self.rules.soft_min_tokens:
            raise ValueError("hard_max_tokens must be >= soft_min_tokens")
