import re

import numpy as np


def split_by_br_semantic(
    text,
    model=None,
    model_name="BAAI/bge-small-zh-v1.5",
    buffer=1,
    threshold_percentile=95,
    min_chars=300,
    max_chars=900,
    batch_size=32,
):
    """Split <br>-separated text by semantic boundary scores."""
    blocks = [
        block.strip()
        for block in re.split(r"<br\s*/?>", text, flags=re.I)
        if block.strip()
    ]
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


def print_boundaries(blocks, distances, threshold, preview_chars=40):
    """Print adjacent block distances and mark selected cut points."""
    for i, distance in enumerate(distances):
        mark = "CUT" if threshold is not None and distance >= threshold else ""
        left = blocks[i][:preview_chars].replace("\n", " ")
        right = blocks[i + 1][:preview_chars].replace("\n", " ")
        print(i, round(float(distance), 4), mark, left, "=>", right)
