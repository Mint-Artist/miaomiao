# Markdown Fragment Cleaner

该目录是一套独立的、无大模型依赖的 Markdown 正文片段清洗器。它直接读取 JSONL 中某个字段的原始 Markdown，不要求上游先生成 `blocks`，也不需要使用 CLI 参数。

处理流程：

```text
JSONL 原始 Markdown
→ CommonMark/GFM AST 解析
→ 标题树与正文 block
→ 保守的边界噪声清理与相邻重复修复
→ 同站点模板检测
→ 确定性 block 规则
→ 同标题路径语义切片
→ candidate 完整性规则
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
```

然后直接运行：

```bash
python3 run_cleaning.py
```

`MARKDOWN_KEY` 等键支持点分隔的嵌套路径，例如 `payload.content`。ID、URL 或标题缺失不会中断：ID 会退化为 JSONL 行号，标题会退化为 Markdown 的首个 H1/标题；缺少 URL 时只跳过站点模板检测。

## 输出

- `accepted.jsonl`：通过全部硬规则和软风险检查，可直接进入后续抽检的数据。
- `review.jsonl`：没有确定性缺陷，但存在承接式开头、不闭环等软风险的数据。
- `rejected.jsonl`：确定性噪声、禁用类型、无效输入及完整拒绝原因。
- `templates.json`：按 host 学到的高频页眉、页脚和站点模板，建议第一次运行后人工检查。
- `statistics.json`：输入规模、保留 token、各规则命中次数以及本次完整配置。
- `preview.md`：前若干个 accepted/review 片段的肉眼检查版本。

输出采用“一条片段一行 JSONL”。主要字段包括：

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
- Markdown 软换行仅在两侧都是 ASCII 字符时补一个空格；中文与英文、数字或中文之间因折行产生的边界直接拼接。
- 普通正文中的连续多空格压缩为一个，汉字之间及汉字与中文标点之间的空格删除；原有中英文单空格保留。
- 相邻且完全相同的普通正文 block 只保留第一份。
- 对普通段落、HTML 可见正文和引用按句切分，识别 `A A`、`A B A B` 一类相邻完全重复序列，只保留第一份。
- 单句自动去重要求至少 15 个非空白字符，多句重复组要求每份至少 30 个非空白字符，避免删除短促强调。
- 非相邻重复、只有语义相似但文字不同的句子不会自动删除；列表、表格和代码不会执行正文空格改写或句子去重。

所有实际修复都记录在片段 `metadata.repairs` 中，汇总数量会写入 `statistics.json`。修复发生在切片之前，因此删除重复内容后，切片器仍可继续合并后续同章节正文。

## 模板检测

模板检测会扫描 JSONL 两遍。第一遍按 host 统计规范化 block 的 document frequency，第二遍执行删除和切片。规范化只用于匹配，不会写回正文。

以下内容才会被识别为模板：

- 在足够多的同站点去重文档中重复出现；
- 主要出现在页面前部或尾部；或者是覆盖该站点大量页面的短 block。

完全重复的文档只计数一次，避免抓取重复把正文误学成模板。没有可靠 URL 的文档不会参与模板学习，避免把跨站点常见正文误删。

模板阈值在 [run_cleaning.py](run_cleaning.py) 中修改。新数据第一次运行后，应优先检查 `templates.json` 中频率最高的条目。

## 长度阈值

默认近似计数把单个汉字、英文词和标点视为 token 单元，主要用于便宜稳定的切片。默认值：

- 目标下限：300 token。
- 目标上限：768 token。
- 普通正文硬下限：300 token，不足时直接进入 `rejected`。
- 列表、表格、引用硬下限：300 token，不再使用短结构化内容例外。
- 硬上限：1536 token。

如果后续训练固定使用 Qwen tokenizer，可设置：

```python
TOKENIZER_NAME_OR_PATH = "/models/Qwen3-0.6B"
```

并安装 `transformers`。阈值必须结合真实语料抽检校准，不应把当前默认值直接当作最终业务标准。

## 测试

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

测试覆盖：标题与 front matter、链接/图片、嵌套列表、表格、引用、代码排除、边界残片清理、保守相邻去重、站点模板、300 token 硬下限、嵌套 JSON key、无效 JSON 审计、accepted/review/rejected 分流。

## 明确边界

该实现只完成大模型标注之前的确定性工作。它不会判断事实正确性、论证深度或细微语病，也不会把 `review` 强行变成高质量数据。正式全量运行前，最有价值的校准材料是：

- `templates.json` 高频条目 50～100 个；
- 每种 hard rule 的拒绝样本 30～50 个；
- accepted 与 review 各随机抽样 200～500 个；
- 按站点、语言、长度、block 类型分别统计保留率。
