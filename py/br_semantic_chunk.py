import re
from pathlib import Path

import numpy as np


DEFAULT_SEPARATOR_PATTERN = (
    r"<\s*br\s*/?\s*>"
    r"|<\s*/?\s*p[^>]*>"
    r"|<\s*/?\s*li[^>]*>"
    r"|<\s*/?\s*h[1-6][^>]*>"
    r"|\r\n|\n|\r"
)
FALLBACK_SENTENCE_PATTERN = r"(?<=[。！？!?；;])"


def split_text_units(
    text,
    separator_pattern=DEFAULT_SEPARATOR_PATTERN,
    fallback_sentence_split=True,
):
    """Split text into candidate units before semantic boundary scoring."""
    blocks = [
        block.strip()
        for block in re.split(separator_pattern, text, flags=re.I)
        if block.strip()
    ]
    if len(blocks) <= 1 and fallback_sentence_split:
        blocks = [
            block.strip()
            for block in re.split(FALLBACK_SENTENCE_PATTERN, text)
            if block.strip()
        ]
    return blocks


def split_by_br_semantic(
    text,
    model=None,
    model_name="BAAI/bge-small-zh-v1.5",
    buffer=1,
    threshold_percentile=95,
    min_chars=300,
    max_chars=900,
    batch_size=32,
    separator_pattern=DEFAULT_SEPARATOR_PATTERN,
    fallback_sentence_split=True,
):
    """Split separated text by semantic boundary scores."""
    blocks = split_text_units(
        text,
        separator_pattern=separator_pattern,
        fallback_sentence_split=fallback_sentence_split,
    )
    if len(blocks) <= 1:
        return blocks, blocks, np.array([]), None

    windows = []
    for i in range(len(blocks)):
        left = max(0, i - buffer)
        right = min(len(blocks), i + buffer + 1)
        windows.append("\n".join(blocks[left:right]))

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

    emb = model.encode(
        windows,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    distances = 1 - np.sum(emb[:-1] * emb[1:], axis=1)
    threshold = np.percentile(distances, threshold_percentile)

    chunks = []
    current = []
    for i, block in enumerate(blocks):
        current.append(block)
        current_text = "\n".join(current)

        hit_boundary = i < len(distances) and distances[i] >= threshold
        hit_max_len = len(current_text) >= max_chars
        long_enough = len(current_text) >= min_chars

        if (hit_boundary and long_enough) or hit_max_len:
            chunks.append(current_text)
            current = []

    if current:
        chunks.append("\n".join(current))

    return chunks, blocks, distances, threshold


def split_txt_by_br_semantic(txt_path, encoding="utf-8", **kwargs):
    """Read a txt file and split its separated text."""
    text = Path(txt_path).read_text(encoding=encoding)
    return split_by_br_semantic(text, **kwargs)


def print_boundaries(blocks, distances, threshold, preview_chars=40):
    """Print adjacent block distances and mark selected cut points."""
    for i, distance in enumerate(distances):
        mark = "CUT" if threshold is not None and distance >= threshold else ""
        left = blocks[i][:preview_chars].replace("\n", " ")
        right = blocks[i + 1][:preview_chars].replace("\n", " ")
        print(i, round(float(distance), 4), mark, left, "=>", right)
