"""
直接运行的 Markdown JSONL 清洗入口。

使用方式：
1. 只修改下面“用户配置区”的变量。
2. 安装依赖：python3 -m pip install -r requirements.txt
3. 在本目录运行：python3 run_cleaning.py

本脚本没有 CLI 参数，也不会调用任何大模型。
"""

import json
from pathlib import Path

from cleaner import CleanerConfig, clean_jsonl
from cleaner.config import (
    ChunkConfig,
    ContentPolicy,
    InputConfig,
    NormalizationConfig,
    OutputConfig,
    RuleConfig,
    TemplateConfig,
)


# ============================= 用户配置区 =============================

PROJECT_DIR = Path(__file__).resolve().parent

# 输入 JSONL。可以写绝对路径，也可以像下面这样写相对本文件的路径。
INPUT_PATH = PROJECT_DIR / "examples" / "input.jsonl"

# 支持嵌套键，例如 "data.markdown"、"payload.content"。
MARKDOWN_KEY = "content"
ID_KEY = "doc_id"       # 没有该字段时自动使用 JSONL 行号
URL_KEY = "url"         # 没有 URL 时仍可运行，但会跳过站点模板学习
TITLE_KEY = "title"     # 没有标题字段时使用 Markdown 中的第一个 H1/标题

OUTPUT_DIR = PROJECT_DIR / "output"

# 片段长度使用近似中英混合 token 计数。若指定真实 tokenizer 路径，需安装 transformers。
TOKENIZER_NAME_OR_PATH = None  # 例如 "/models/Qwen3-0.6B"
TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 768
HARD_MAX_TOKENS = 1536
HARD_MIN_TOKENS = 300
STRUCTURED_HARD_MIN_TOKENS = 300

# 保守文本修复：仅删除边界处明确的 HTML 注释残片，并修复相邻的完全重复正文。
ENABLE_TEXT_NORMALIZATION = True
STRIP_BOUNDARY_ARTIFACTS = True
NORMALIZE_PROSE_SPACING = True
DEDUPLICATE_ADJACENT_PROSE_BLOCKS = True
DEDUPLICATE_REPEATED_SENTENCE_SEQUENCES = True
MIN_DUPLICATE_SENTENCE_CHARS = 15
MIN_DUPLICATE_SEQUENCE_CHARS = 30
MAX_DUPLICATE_SEQUENCE_SENTENCES = 20

# 正文类型策略。
KEEP_LISTS = True
KEEP_BLOCKQUOTES = True
KEEP_TABLES = True
KEEP_CODE_BLOCKS = False
KEEP_HTML_VISIBLE_TEXT = True
KEEP_IMAGE_ALT_TEXT = False

# 同站点重复模板检测。数据量很小或 URL 不可靠时可设为 False。
ENABLE_SITE_TEMPLATE_DETECTION = True
TEMPLATE_MIN_HOST_DOCUMENTS = 20
TEMPLATE_MIN_DOCUMENTS = 5

# 最多把多少个 accepted/review 片段写入可读 Markdown 预览。
PREVIEW_FRAGMENTS = 100


# =========================== 配置装配与运行 ===========================

CONFIG = CleanerConfig(
    input_path=str(INPUT_PATH),
    input=InputConfig(
        markdown_key=MARKDOWN_KEY,
        id_key=ID_KEY,
        url_key=URL_KEY,
        title_key=TITLE_KEY,
    ),
    output=OutputConfig(
        accepted_path=str(OUTPUT_DIR / "accepted.jsonl"),
        review_path=str(OUTPUT_DIR / "review.jsonl"),
        rejected_path=str(OUTPUT_DIR / "rejected.jsonl"),
        templates_path=str(OUTPUT_DIR / "templates.json"),
        statistics_path=str(OUTPUT_DIR / "statistics.json"),
        preview_path=str(OUTPUT_DIR / "preview.md"),
        preview_fragments=PREVIEW_FRAGMENTS,
    ),
    content=ContentPolicy(
        keep_lists=KEEP_LISTS,
        keep_blockquotes=KEEP_BLOCKQUOTES,
        keep_tables=KEEP_TABLES,
        keep_code_blocks=KEEP_CODE_BLOCKS,
        keep_html_visible_text=KEEP_HTML_VISIBLE_TEXT,
        keep_image_alt_text=KEEP_IMAGE_ALT_TEXT,
    ),
    normalization=NormalizationConfig(
        enabled=ENABLE_TEXT_NORMALIZATION,
        strip_boundary_artifacts=STRIP_BOUNDARY_ARTIFACTS,
        normalize_prose_spacing=NORMALIZE_PROSE_SPACING,
        deduplicate_adjacent_prose_blocks=DEDUPLICATE_ADJACENT_PROSE_BLOCKS,
        deduplicate_repeated_sentence_sequences=DEDUPLICATE_REPEATED_SENTENCE_SEQUENCES,
        min_duplicate_sentence_chars=MIN_DUPLICATE_SENTENCE_CHARS,
        min_duplicate_sequence_chars=MIN_DUPLICATE_SEQUENCE_CHARS,
        max_duplicate_sequence_sentences=MAX_DUPLICATE_SEQUENCE_SENTENCES,
    ),
    chunk=ChunkConfig(
        target_min_tokens=TARGET_MIN_TOKENS,
        target_max_tokens=TARGET_MAX_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    ),
    rules=RuleConfig(
        hard_min_tokens=HARD_MIN_TOKENS,
        structured_hard_min_tokens=STRUCTURED_HARD_MIN_TOKENS,
        soft_min_tokens=TARGET_MIN_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    ),
    templates=TemplateConfig(
        enabled=ENABLE_SITE_TEMPLATE_DETECTION,
        min_host_documents=TEMPLATE_MIN_HOST_DOCUMENTS,
        min_template_documents=TEMPLATE_MIN_DOCUMENTS,
    ),
    tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
)


def main() -> None:
    summary = clean_jsonl(CONFIG)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    print("\n完成。人工预览：%s" % CONFIG.output.preview_path)


if __name__ == "__main__":
    main()
