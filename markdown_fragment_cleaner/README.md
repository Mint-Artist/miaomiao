# Markdown Cleaner

该目录是一套独立的、无大模型依赖的 Markdown 正文清洗器。它直接读取 JSONL 中某个字段的原始 Markdown，不要求上游先生成 `blocks`，也不需要使用 CLI 参数。清洗后既可以沿用原来的语义切片，也可以输出不切片的全文，或者同时生成两套结果。

处理流程：

```text
JSONL 原始 Markdown
→ CommonMark/GFM AST 解析
→ 标题树与正文 block
→ 保守的边界噪声清理与相邻重复修复
→ 同站点模板检测
→ 确定性 block 规则
→ fragment：同标题路径语义切片
  document：按原顺序组装一条清洗后全文
→ 各自的长度与完整性规则
→ accepted / review / rejected
```

## 快速运行

安装唯一必需依赖：

```bash
cd "/Users/awh/Documents/ChatGPT/New project/markdown_fragment_cleaner"
python3 -m pip install -r requirements.txt
```

打开 [run_cleaning.py](run_cleaning.py)，修改文件顶部“用户配置区”：

```python
INPUT_PATH = Path("/path/to/input.jsonl")
MARKDOWN_KEY = "data.markdown"
ID_KEY = "doc_id"
URL_KEY = "url"
TITLE_KEY = "title"
OUTPUT_DIR = Path("/path/to/output")

# fragment：只输出旧切片；document：只输出全文；both：两套都输出
OUTPUT_MODE = "both"
DOCUMENT_MIN_TOKENS = 300
DOCUMENT_MAX_TOKENS = 6000
```

然后直接运行：

```bash
python3 run_cleaning.py
```

`MARKDOWN_KEY` 等键支持点分隔的嵌套路径，例如 `payload.content`。ID、URL 或标题缺失不会中断：ID 会退化为 JSONL 行号，标题会退化为 Markdown 的首个 H1/标题；缺少 URL 时只跳过站点模板检测。

## 输出模式与文件

- `OUTPUT_MODE = "fragment"`：只运行原有切片组装器，输出格式和路径保持兼容。
- `OUTPUT_MODE = "document"`：每篇输入文档最多生成一条清洗后全文，不做切片、不截断。
- `OUTPUT_MODE = "both"`：共享同一次解析、规范化、去重和 block 过滤，同时输出切片与全文。

片段结果仍写在输出根目录：`accepted.jsonl`、`review.jsonl`、`rejected.jsonl` 和 `preview.md`。

全文结果单独写在 `documents/` 下：

- `documents/accepted.jsonl`：位于全文 token 范围内且没有软风险。
- `documents/review.jsonl`：默认包含超过 6000 token 但仍保留完整内容的全文，以及其他软风险全文。
- `documents/rejected.jsonl`：不足 300 token、清洗后为空或命中全文硬规则的数据。
- `documents/preview.md`：前若干条 accepted/review 全文的可读预览。

`templates.json` 和 `statistics.json` 是两种模式共享的模板审计与运行统计。

片段输出继续采用“一条片段一行 JSONL”。主要字段包括：

```json
{
  "fragment_id": "frag-...",
  "doc_id": "doc-1",
  "url": "https://example.com/a",
  "doc_title": "文档标题",
  "heading_path": ["文档标题", "子章节"],
  "content": "清洗后的正文……",
  "block_ids": ["doc-1-b000001"],
  "block_types": ["paragraph"],
  "token_count": 423,
  "start_line": 3,
  "end_line": 11,
  "source_row": 1,
  "context_before": "前一个有效 block 的末尾……",
  "context_after": "后一个有效 block 的开头……",
  "flags": [],
  "decision": "accepted",
  "metadata": {
    "chunker_version": "markdown-ast-section-v1"
  }
}
```

全文输出采用“一篇输入文档最多一行 JSONL”。主要字段包括：

```json
{
  "record_id": "docrec-...",
  "record_type": "whole_document",
  "doc_id": "doc-1",
  "url": "https://example.com/a",
  "doc_title": "文档标题",
  "content": "第一段清洗后的正文。\n\n姓名：张三\n电话：13800000000",
  "block_ids": ["doc-1-b000001", "doc-1-b000002"],
  "block_types": ["paragraph", "paragraph"],
  "token_count": 1850,
  "start_line": 3,
  "end_line": 96,
  "source_row": 1,
  "sections": [
    {
      "heading_path": ["文档标题", "基本信息"],
      "block_ids": ["doc-1-b000001", "doc-1-b000002"],
      "start_char": 0,
      "end_char": 42
    }
  ],
  "removed_blocks": [
    {"block_id": "doc-1-b000003", "reason": "hard_rule", "flags": ["known_boilerplate"]}
  ],
  "flags": [],
  "decision": "accepted",
  "metadata": {"assembler_version": "whole-document-v1"}
}
```

`sections` 保存标题路径和它在 `content` 中的字符区间；标题本身不混入正文。`removed_blocks` 是全文组装前被模板或硬规则删除的 block 审计信息。

## AST 与正文策略

解析基于 `markdown-it-py` 的 CommonMark/GFM AST，而不是正则模拟 Markdown：

- 支持 ATX 和 Setext 标题；标题用于 `heading_path` 和切片边界，不进入 `content`。
- YAML/TOML front matter 作为元数据区跳过，同时保留正确的原始行号。
- 相邻普通段落在同一标题路径内合并。
- 列表及嵌套列表保持完整；“如下：”一类引导段会和列表/表格绑定。
- GFM 表格重新渲染为无链接目标的干净 Markdown 表格。
- 引用保留 `>` 结构。
- 代码块默认排除，inline code 保留反引号标记。
- Markdown 链接只保留可见 anchor，删除目标 URL。
- 图片默认全部删除；可以通过 `KEEP_IMAGE_ALT_TEXT` 保留 alt。
- HTML block 删除标签、脚本和样式，只保留可见正文。
- 超长普通段落只在句末拆分，不使用 overlap window。
- 任意 block 被删除后都会形成切片边界，不会把噪声两侧强行拼接。

## 保守文本修复

AST 解析之后、模板检测和语义切片之前，会执行独立的文本规范化层：

- 只在正文边界删除明确的 HTML 注释残片，例如 `-->`、`<!--`、空注释以及边界 BOM/零宽字符；正文内部出现的 `-->` 保留。
- Markdown 软换行默认采用 `smart` 策略：字段行（如“姓名：”“电话：”）、连续步骤、引导式结构和逐行完整句保留 `\n`；普通网页视觉折行会展开，两侧都是 ASCII 时补一个空格，否则直接拼接。
- 可把 `SOFTBREAK_POLICY` 改为 `preserve` 以保留全部软换行，或改为 `unwrap` 以展开全部软换行。AST 中的段落、列表、表格等 block 边界不受这个变量影响。
- 普通正文中的连续多空格压缩为一个，汉字之间及汉字与中文标点之间的空格删除；原有中英文单空格保留。
- 相邻且完全相同的普通正文 block 只保留第一份。
- 对普通段落、HTML 可见正文和引用按句切分，识别 `A A`、`A B A B` 一类相邻完全重复序列，只保留第一份。
- 单句自动去重要求至少 15 个非空白字符，多句重复组要求每份至少 30 个非空白字符，避免删除短促强调。
- 非相邻重复、只有语义相似但文字不同的句子不会自动删除；列表、表格和代码不会执行正文空格改写或句子去重。

所有实际修复都记录在结果 `metadata.repairs` 中，汇总数量会写入 `statistics.json`。修复发生在两种组装器之前，因此切片与全文使用完全相同的清洗结果。

## 模板检测

模板检测会扫描 JSONL 两遍。第一遍按 host 统计规范化 block 的 document frequency，第二遍执行删除和切片。规范化只用于匹配，不会写回正文。

以下内容才会被识别为模板：

- 在足够多的同站点去重文档中重复出现；
- 主要出现在页面前部或尾部；或者是覆盖该站点大量页面的短 block。

完全重复的文档只计数一次，避免抓取重复把正文误学成模板。没有可靠 URL 的文档不会参与模板学习，避免把跨站点常见正文误删。

模板阈值在 [run_cleaning.py](run_cleaning.py) 中修改。新数据第一次运行后，应优先检查 `templates.json` 中频率最高的条目。

## 长度阈值

默认近似计数把单个汉字、英文词和标点视为 token 单元。切片模式默认值：

- 目标下限：300 token。
- 目标上限：768 token。
- 普通正文硬下限：300 token，不足时直接进入 `rejected`。
- 列表、表格、引用硬下限：300 token，不再使用短结构化内容例外。
- 硬上限：1536 token。

全文模式使用独立阈值：

- `DOCUMENT_MIN_TOKENS = 300`：不足时进入全文 `rejected`。
- `DOCUMENT_MAX_TOKENS = 6000`：超过时默认完整进入全文 `review`，不切片、不截断。
- `DOCUMENT_OVER_MAX_POLICY = "review"`：也可以明确改为 `reject` 或 `allow`。

修改全文范围不会影响旧切片阈值；修改切片阈值也不会影响全文分流。

如果后续训练固定使用 Qwen tokenizer，可设置：

```python
TOKENIZER_NAME_OR_PATH = "/models/Qwen3-0.6B"
```

并安装 `transformers`。阈值必须结合真实语料抽检校准，不应把当前默认值直接当作最终业务标准。

## 测试

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

测试覆盖：标题与 front matter、链接/图片、嵌套列表、表格、引用、代码排除、智能软换行、边界残片清理、保守相邻去重、站点模板、300 token 硬下限、嵌套 JSON key、无效 JSON 审计、三种输出模式、全文长度分流与不截断保证。

## 明确边界

该实现只完成大模型标注之前的确定性工作。它不会判断事实正确性、论证深度或细微语病，也不会把 `review` 强行变成高质量数据。正式全量运行前，最有价值的校准材料是：

- `templates.json` 高频条目 50～100 个；
- 每种 hard rule 的拒绝样本 30～50 个；
- accepted 与 review 各随机抽样 200～500 个；
- 按站点、语言、长度、block 类型分别统计保留率。
