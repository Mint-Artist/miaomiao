# Chunk 流水线

本地实验实现 v0.2.0：固定简版 HTML 输入，共享规范化和 DOM 位置映射，比较不同切分策略，并分别输出完整正文块和检索视图。

本版本尚未接入生产 B0、真实 embedding 或搜索链路；尚未选定生产预算/胜出配置。512 仅出现在下面显式指定的本地实验中。

## 快速运行

### 服务器 JSONL 批处理

将整个 `chunk流水线/` 目录复制到服务器（包括 `configs/`、`chunk_pipeline/`，
旧提取器目录作为测试基线保留），使用 Python 3.8+：

```bash
cd /path/to/chunk流水线
python3 -m pip install -r requirements.txt
python3 run_jsonl.py --input /data/pages.jsonl --out-dir /data/chunk_run_001 --absolute-chars 512
```

输入为 UTF-8 JSONL，每行一个对象；`pg` 是原始 HTML 字符串，`url` 是网页 URL。
HTML 内的换行由 JSON 编码为 `\n`，不能将一个 JSON 对象跨多条物理行存储。

```json
{"url":"https://example.com/a","pg":"<html><h1>标题</h1><article><p>这里是网页正文……</p></article></html>"}
```

运行全部五套配置，每页只提取、规范化一次。512 仍是显式实验参数。
输入逐行处理，内存不随网页总数增长；本入口是串行文件批处理，尚未做服务器吞吐基准。

输出布局：

```text
chunk_run_001/
  summary.jsonl                 # 每个成功网页一行：url + 每种策略的字符串列表
  pages.jsonl                   # 输入行号、URL 与网页目录的对应关系
  errors.jsonl                  # 失败行号、URL、阶段和错误原因
  run.json                      # 完成状态、成功/失败数、配置和汇总口径
  pages/
    000000001_<url哈希>/
      simplified.html          # 实际送入规范化/切分的简版 HTML，可直接查看结构
      extraction.json          # 提取器命中的站点、正文选择器和警告
      recursive_hard/           # 保留原有详细产物
      structure_hard/
      structure_protect/
      structure_hard_context/
      structure_protect_context/
```

`summary.jsonl` 示例（文本仅示意）：

```json
{"url":"https://example.com/a","recursive_hard":["块1","块2"],"structure_hard":["块1","块2"],"structure_protect":["完整段落"],"structure_hard_context":["所属标题\n块1","所属标题\n块2"],"structure_protect_context":["所属标题\n完整段落"]}
```

默认列表取自 `retrieval.jsonl` 的 `index_text`，包括结构上下文，适合后续检索输入。
如需用于严格正文还原的列表，增加 `--summary-field text`，列表改用 `chunks.jsonl` 的
`text`。这两种口径记录在 `run.json`；默认检索列表不能用于拼接还原正文。

可用 `--html-field HTML字段名`、`--url-field URL字段名` 改字段名，
`--config-dir 配置目录` 选择实验配置集；默认读取项目自带五套配置。
`--progress-every 100` 控制进度日志（0 关闭）。URL 重复时保留每条输入，不去重、不覆盖。

空行跳过；坏 JSON、缺字段、提取失败或任一策略失败会写入 `errors.jsonl`，该网页不进入
汇总，其他网页继续处理。已完成提取的失败网页保留简版 HTML 和 `error.json`。
退出码：0 全部成功；1 处理结束但存在失败网页；2 配置或 I/O 等运行错误。
文件写入失败会终止批次，`run.json` 标记 `interrupted`；已有输出目录禁止覆盖。
断点续跑尚未实现，重跑请使用新目录。

原始网页使用现有提取器：已知站点使用对应规则，其他站点使用通用正文选择器；
找不到正文容器会明确报错。`--profile generic` 可显式覆盖站点选择。
本次没有扩大提取规则的站点覆盖范围，需要结合服务器上的 `errors.jsonl` 检查实际覆盖率。

批处理使用当前维护的 `chunk_pipeline/html_extract.py`（v0.1.2）。此版本保留了 v0.1.1 对
清理父节点后继续访问已销毁子节点导致的 `AttributeError: 'NoneType' object has no attribute 'get'`
的修复，并将单页 `AttributeError` 隔离到错误日志。历史提取器 v0.1.0 保留作冻结基线，不用于批处理。
更新后重跑请指定新的输出目录，例如 `chunk_run_002`，保留中断批次供核对。

遇到大量“未找到正文容器”时，先区分完整网页和已提取的正文片段。默认
`--content-fallback strict` 保持原有严格选择行为；完整网页的正文定位还需针对实际模板确认。
如已确认正文位于某个元素，可指定 `--content-selector '#gov-text'`，覆盖站点正文选择器。
显式选择器允许短正文（至少 1 个文字字符），未命中时仍按回退选项处理。

为排查容器不匹配，可显式加 `--content-fallback body`：原选择器未命中时才清理整个
`body`；没有 `body` 标签时清理 HTML 片段并去掉 `head`。保留现有噪声过滤和结构标签，
但不保证去除所有导航、侧栏等非正文。仅脚本或空页面不会因回退被算作成功。
这不是新的正文识别模型；完整 HTML 上使用后须抽样检查简版 HTML，不能用成功率证明提取质量。

`extraction.json` 的 `extraction_mode` 区分 `profile_selector`、`explicit_selector` 和
`body_fallback`，同时记录实际 `content_selector`、`content_fallback_used` 和警告。
`run.json` 的 `fallback_extracted` 统计成功完成回退提取的页数（包含随后切分失败的页），
与整条流水线的 `succeeded` 是不同口径。所有策略仍共享同一次提取结果。

### 本地样例与单页

环境：Python 3.8+、beautifulsoup4 4.12.3。安装依赖：

```bash
cd /Users/awh/work/chunk2026/chunk流水线
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

三个已有样本、五套配置使用相同输入。输出目录必须是新的，防止覆盖实验结果：

```bash
python3 run_examples.py --absolute-chars 512 --out-dir output/my_experiment
```

运行后可在指定输出目录查看 `comparison.md`、`summary.jsonl` 和各网页的 `simplified.html`。

单个简版 HTML：

```bash
python3 -m chunk_pipeline \
  --html html结构切分_2026-09-23/output/huawei_zh-cn16079039.simplified.html \
  --doc-id huawei \
  --url https://consumer.huawei.com/cn/support/content/zh-cn16079039/ \
  --config configs/structure_protect_context.json \
  --absolute-chars 512 \
  --out-dir output/huawei_trial
```

`protect` 不提供默认绝对上限；未传入时明确报错。`hard` 始终以 `target_chars` 为最终字符上限，传入更大的 `absolute_chars` 不会放宽它。原始网页需先走现有提取器，CLI 不代替抓取和正文提取。

## 项目结构

| 路径 | 职责 |
|---|---|
| `chunk_pipeline/normalize.py` | 简版 HTML → 文本 T、DOM 范围、结构原子、表格坐标、资源位置 |
| `chunk_pipeline/html_extract.py` | 当前维护的原始 HTML → 简版 HTML 提取器；跳过清理中已销毁的后代节点 |
| `chunk_pipeline/strategies.py` | 结构装箱、类型保护、递归基线；按原文范围切片 |
| `chunk_pipeline/context.py` | 标题、表头、跨行继承和表格片段坐标的检索上下文 |
| `chunk_pipeline/pipeline.py` | JSONL schema、严格验收、统计和独立检索视图/LLM 缓存适配 |
| `chunk_pipeline/io.py` | 完整目录落盘、哈希与运行环境记录，禁止覆盖旧实验 |
| `configs/` | 每个实验一份配置，共享同一套实现 |
| `tests/` | 三个完整文本参照、边界案例、CLI 和固定种子随机验收 |
| `baselines/legacy_manifest.json` | 旧原型 31 个文件的冻结哈希、配置和依赖版本 |
| `简版HTML提取_2026-09-23/` | 既有提取器，保留原实现 |
| `html结构切分_2026-09-23/` | 冻结的历史结构原型，不能当作生产 B0 |

单项目内维护公共代码；历史目录只作为冻结参照，不为每个新实验复制源码。生产入口待胜出配置及模型限制确定后冻结，当前不提供一个假定胜出的 `production.py`。

## 五个可运行对照

| 配置 | 边界 | 预算 | 上下文 |
|---|---|---|---|
| `recursive_hard` | 规范化文本的换行→句界→分句→硬断，不查询 DOM 边界 | 256 硬上限 | 无 |
| `structure_hard` | 标题/段落/列表/提示框/表格 | 256 硬上限 | 无 |
| `structure_protect` | 相同结构层 | 256 目标、显式 H | 无 |
| `structure_hard_context` | 相同结构层 | 256 硬上限 | 标题路径/表头等 |
| `structure_protect_context` | 相同结构层 | 256 目标、显式 H | 标题路径/表头等 |

结构策略优先把标题链与后续首单元绑定。保护的是一个段落、一个 li 或整个提示框，不能把多个普通段落任意合并后超目标。表格始终按目标预算装箱，超长行只切原始范围，不重复相邻列进正文。

默认不跨章节合并，正文与不同表格分开；note/pre/quote 也独立于普通段落装箱。`merge_short_sections=true` 可合并目标内的短章节；`metadata_mode=separate` 隔离 metadata 区域，但不把它移动或删除。

上下文按源位置避免重复注入。`max_title_chars` 限制标题副本，`title_policy` 可选 `ancestors` 或 `document_nearest`。必要表头/跨行信息优先，可选标题放不下时整级丢弃并记账；必要上下文也放不下则显式失败，不静默省略列对应关系。

## 输出及正文契约

每个运行目录包含：

| 文件 | 语义 |
|---|---|
| `normalized.txt` | 从简版 HTML 派生的固定正文 T，不是新的上游交付要求 |
| `simplified.html` | 单页 CLI / 样例运行保存的实际切分输入；JSONL 批处理在网页目录共享保存一份 |
| `document.json` | T、结构原子、全部 DOM 元素的 `[start,end)`、表格/资源侧表 |
| `chunks.jsonl` | 完整、有序、不重叠的正文交付集合；`text` 是 T 的精确切片 |
| `retrieval.jsonl` | 一对一检索视图，字段 `index_text`；禁止用它拼回正文 |
| `llm_requests.jsonl` | 既有 LLM 上下文能力可消费的片段请求及缓存键 |
| `report.json` | 配置、版本、输入/正文哈希、覆盖/超限/拆分/上下文/长度统计 |
| `manifest.json` | 产物哈希、实现源码哈希、Python/依赖版本和检索视图归因哈希 |

```python
assert ''.join(chunk['text'] for chunk in chunks) == document.text
assert all(chunk['text'] == document.text[chunk['start']:chunk['end']]
           for chunk in chunks)
```

分隔符已经包含在 `text` 中，不再 `strip()` 或额外拼换行。所有坐标均为 Unicode 码点下标，不是 HTML 字节位置。标题、metadata、表头都参与覆盖；图像 alt/URL 不进入正文，但在资源侧表保留。

`parent_id` 仅表示续块对应的首个 chunk；DOM/章节父节点使用节点侧表及其 `parent_id`，它们是不同命名空间。未来的重叠窗口、父子返回或生成命题应扩展检索视图，不能加入正文交付列表。

## 接入已有 LLM 上下文

本版不调用外部模型。第一遍先生成正文块和 `llm_requests.jsonl`；用既有模型处理请求，整篇文本可从 `normalized.txt` 读取，将响应保存成以下 JSON 对象：

```json
{
  "请求中的cache_key": {
    "text": "由既有模型生成的片段背景",
    "model": "实际模型及版本",
    "prompt_version": "实际提示词版本",
    "document_sha256": "请求中的document_sha256"
  }
}
```

再次使用相同 HTML/配置运行 CLI，增加 `--llm-contexts contexts.json` 和新的 `--out-dir`。这会保持正文边界，生成增强后的检索视图。缓存键包含正文版本、片段范围和原上下文表示，避免同文档重复段落或不同切法串用背景；缺失、过期、字段不齐或超限会拒绝整次输出。

默认增强视图沿用块的绝对字符上限。表示消融需要更大输入预算时，可显式传 `--view-absolute-chars N`，报告会记录增强后的长度和超目标数；这不是相同输入预算的对照，也不会改变正文分块。模型和 prompt 的版本变更需要选择另一份缓存文件；结果的 provenance/hash 会记录实际使用内容。

## Embedding 接口与尚未接入的能力

库接口支持真正的 token 检查：

```python
records, report = chunk_document(
    document, ChunkConfig(max_embedding_tokens=model_limit),
    token_counter=actual_tokenizer_count, tokenizer_id=model_tokenizer_version,
)
```

token 计数须覆盖实际模型会消费的特殊 token 等输入；CLI 当前未加载模型 tokenizer。报告明确 `token_limit_verified=false`，不会拿字符数冒充 token。LLM 增强后的检索视图也可通过 `retrieval_views(..., token_counter=..., max_tokens=..., tokenizer_id=...)` 独立复检。

当前未实现句向量语义切分、Late Chunking、学习式层级、父子返回、多格式表格检索序列化、真实生成调用或搜索评测。未实现的策略/配置会报错，不提供无效开关。

## 已知限制

- 只有三个真实网页样本；跨行跨列、长行、长无标点文本等通过合成案例验证，仍需真实网页验收。
- 嵌套表格及 `rowspan=0` 等未支持结构明确拒绝处理，保留输入用于复核。极宽表头无法带必要上下文满足预算时同样报错。
- 纯图片页可以产生空正文块列表，图片仍在资源侧表；不做 OCR。全空白渲染文档若不能形成有效正文块则显式报错。
- 标题识别包含短整段加粗启发式；解析没有增加模型推断，其准确率尚未在新站点评估。
- 当前位置映射在内存中逐字符保留渲染事件；尚未进行大文档内存及亿级吞吐基准，不应据本地耗时外推生产容量。
- 长行续片有表头、行列坐标和来源映射，但不自动推断业务主键。需要更强自包含行描述时，通过独立检索表示实验处理。
- H 未选定，未冻结生产配置；`baseline_id=legacy-structure-0.1.0` 不等于生产 B0。

## 验证

```bash
python3 -m unittest discover -s tests -v
python3 简版HTML提取_2026-09-23/validate.py
python3 html结构切分_2026-09-23/validate.py
```

新版 58 个测试方法（包括嵌套噪声清理、严格/显式选择器/整页回退、JSONL 原始网页批处理、失败隔离、重复 URL、服务器路径入口，
以及 15 个样本×配置组合和 80 次固定种子随机预算/还原检查）；旧版验收分别 37/37、36/36。
测试已在 Python 3.8.20 和 3.9.6 上通过。测试中的字符计数替身仅验证接口守卫，不是实际 embedding token 验证。

规格依据：[正文契约](规格/vNext/正文契约与切分规则.md)、[前沿调研与建议](../调研与资料/chunk方法审视_2026-09-24/前沿调研与方案建议.md)。
