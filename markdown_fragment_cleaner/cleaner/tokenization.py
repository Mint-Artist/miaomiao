from __future__ import annotations

import re
from typing import Protocol


_APPROX_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9]+(?:['._-][A-Za-z0-9]+)*|[^\s]",
    re.UNICODE,
)


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        ...


class ApproxTokenCounter:
    """Cheap deterministic counter for mixed Chinese and English text."""

    def count(self, text: str) -> int:
        return len(_APPROX_TOKEN_RE.findall(text))


class HuggingFaceTokenCounter:
    def __init__(self, name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install transformers to use a Hugging Face tokenizer") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=False)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
