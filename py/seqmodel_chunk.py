import re


DEFAULT_SEQMODEL_ID = "damo/nlp_bert_document-segmentation_chinese-base"
BR_RE = re.compile(r"<br\s*/?>", re.I)


def load_seqmodel(model_id=DEFAULT_SEQMODEL_ID):
    """Load ModelScope SeqModel document segmentation pipeline once."""
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    return pipeline(task=Tasks.document_segmentation, model=model_id)


def split_by_seqmodel(
    text,
    segmenter=None,
    model_id=DEFAULT_SEQMODEL_ID,
    remove_br=True,
    min_chars=0,
):
    """Split text into paragraphs with ModelScope SeqModel.

    In Jupyter, prefer:
        segmenter = load_seqmodel()
        chunks, raw = split_by_seqmodel(text, segmenter=segmenter)
    """
    if segmenter is None:
        segmenter = load_seqmodel(model_id)

    document = BR_RE.sub("", text) if remove_br else text
    result = segmenter(documents=document)
    raw_text = _extract_text(result)
    chunks = _split_output(raw_text)

    if min_chars > 0:
        chunks = _merge_short_chunks(chunks, min_chars=min_chars)

    return chunks, raw_text


def _extract_text(result):
    try:
        from modelscope.outputs import OutputKeys

        if isinstance(result, dict) and OutputKeys.TEXT in result:
            return result[OutputKeys.TEXT]
    except Exception:
        pass

    if isinstance(result, dict):
        return result.get("text") or result.get("output") or str(result)
    return result


def _split_output(raw_text):
    if isinstance(raw_text, (list, tuple)):
        return [str(item).strip() for item in raw_text if str(item).strip()]

    text = str(raw_text).strip()
    if not text:
        return []

    parts = re.split(r"(?:<br\s*/?>|\n{1,})", text, flags=re.I)
    return [part.strip() for part in parts if part.strip()]


def _merge_short_chunks(chunks, min_chars):
    merged = []
    current = []
    for chunk in chunks:
        current.append(chunk)
        current_text = "\n".join(current)
        if len(current_text) >= min_chars:
            merged.append(current_text)
            current = []
    if current:
        if merged:
            merged[-1] = merged[-1] + "\n" + "\n".join(current)
        else:
            merged.append("\n".join(current))
    return merged


def print_chunks(chunks, preview_chars=80):
    for i, chunk in enumerate(chunks):
        preview = chunk[:preview_chars].replace("\n", " ")
        print(i, len(chunk), preview)
