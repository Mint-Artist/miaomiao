# 截断与完整性评测：LLM 判断与结果整理指南（给 code agent）

> 输入数据由 `judge_export.py` 在服务器 notebook 里导出（已完成）。本文档指导 code agent 完成：**调用 LLM 打标 → 校验 → 汇总指标 → 写报告**。
> 不需要 GPU、不需要重新切分或检索；只读两份 jsonl.gz，调用 LLM API，写结果。

---

## 0. 背景与目标

我们在比较几种网页切分方法（`content_br` = 线上基线，`html_struct_secmerge_t512` = 原结构切分，`html_container` = 容器级结构切分，可能还有 `md_struct_secmerge_t512`）。网页级召回指标（MRR、hit@k）测不出"块是否完整"，所以用 LLM 直接判断两件事：

1. **query 视角**：每种方法召回的前 10 块，**能不能回答 query、答案有没有在块边界被切断、答案是否落在向量模型看不到的截断部分**；
2. **页面视角**：每种方法把同一个网页切成的块，**切点有没有切坏东西、每块能不能独立读懂、有没有混装**。

"向量模型截断"：向量模型最多看 512 token，块里超出的部分在检索时不可见。导出数据里已经算好每块的 `visible_chars`（不需要 LLM）。

---

## 1. 输入文件

默认位置（以 notebook 里 `OUT_ROOT` 为准）：

| 文件 | 内容 |
|---|---|
| `OUT_ROOT/judge/judge_query_top10.jsonl.gz` | query 视角：300 条 query × 各方法前 10 块 |
| `OUT_ROOT/judge/judge_pages.jsonl.gz` | 页面视角：约 100 个网页 × 各方法完整切分 + 切点 |
| `OUT_ROOT/judge/judge_pages.viewer.jsonl.gz` | 同一批网页，给 `chunk_viewer.html` 人工对照用（不参与打标） |

都是 gzip 压缩的 jsonl，第 1 行是 `{"type": "meta", ...}`。

### 1.1 query 文件（每行一条 query）

```json
{"type": "query", "qid": "...", "query": "...", "gold_urls": ["归一化 url", ...],
 "methods": {
   "html_container": [
     {"rank": 1, "url": "...", "is_gold": 0, "pos": 3, "of": 8,
      "chunk_id": "16 位哈希", "text": "块全文", "chars": 431,
      "tokens": 548, "visible_chars": 539, "truncated": true}, ...],
   "content_br": [...], ...}}
```

| 字段 | 含义 |
|---|---|
| `rank` | 该方法检索结果里的名次（1–10） |
| `is_gold` | 该块所在网页是否是标准答案网页（> 0 表示是；这是**网页级**标注，块本身不一定含答案） |
| `pos` / `of` | 该块是所在网页的第几块 / 共几块 |
| `chunk_id` | 块文本的哈希；不同方法切出完全相同的块时 id 相同 |
| `chars` | 块字数（**不计换行**） |
| `tokens` | 块正文的 token 数（与向量推理同一个 tokenizer、同一个前缀） |
| `visible_chars` | **`text` 原始字符串**里模型能看到的前多少个字符（**计换行**）；`text[:visible_chars]` 即可见部分 |
| `truncated` | 是否被向量模型截断（`visible_chars < len(text)`） |

### 1.2 页面文件（每行一个网页）

```json
{"type": "page", "url": "...", "source": "...",
 "methods": {
   "html_container": {
     "chunks": [{"i": 1, "kind": "list", "chunk_id": "...", "text": "...", "chars": 224,
                 "tokens": 230, "visible_chars": 229, "truncated": false}, ...],
     "boundaries": [{"i": 1, "tail": "第 1 块末尾 150 字", "head": "第 2 块开头 150 字"}, ...],
     "stats": {...}},
   "content_br": {...}, ...}}
```

`boundaries[j]` 是第 `i` 块和第 `i+1` 块之间的切点。`kind` 只有 `html_container` 有（`para` / `list` / `table` / `note` / `metadata` / `code` / …）。

---

## 2. 运行环境与 API 约定

- **API**：SiliconFlow，OpenAI 兼容接口：`base_url = "https://api.siliconflow.cn/v1"`，`POST /chat/completions`。
- **密钥**：只从环境变量 `SILICONFLOW_API_KEY` 读取。**不得**打印、写进代码、日志、结果文件或提交到仓库。
- **模型**：用账号里可用的较强中文模型（DeepSeek / Qwen 系列的较新版本），写成配置项 `JUDGE_MODEL`，**整个评测只用一个模型**，并写进结果 meta。
- **参数**：`temperature = 0`；要求 JSON 输出（支持 `response_format={"type": "json_object"}` 就用，否则靠 prompt 约束 + 解析）。
- **并发与限流**：并发 4–8；遇到 429 / 5xx 指数退避重试（1s、2s、4s… 最多 6 次）；单条请求超时 60s。
- **缓存与断点续跑**：每条判断结果**追加写入** jsonl（一行一条，含 `key`），启动时读已有结果、跳过已完成的 key。中途停掉重跑不能重复调用。
- **失败处理**：JSON 解析失败最多重试 2 次；仍失败记为 `{"key": ..., "error": "..."}`，**不得静默丢弃**，在报告里写失败条数。
- **不要提交数据**：输入和结果都是第三方网页内容，只放在服务器的 `OUT_ROOT/judge/` 下，不要推送到公开仓库。

规模估算：query 视角约 300 × 4 方法 × 10 块 = 12000 块，按 (qid, chunk_id) 去重后通常 8000–10000 次调用；页面视角约 100 页 × 3 方法 ×（约 12 块 + 11 个切点）≈ 7000 次。每次输入约 300–1000 token。先用 20 条跑通再全量。

---

## 3. 任务一：query 视角（块能否回答 query）

### 3.1 构造任务

- 遍历每条 query、每个方法的前 10 块。
- **去重**：判断对象是 (query, 块文本)，key = `md5(qid + "|" + chunk_id)`。不同方法召回了同一块，只判一次，结果复用。
- **盲评**：prompt 里**不出现方法名、名次、是否标准答案网页**。
- **截断标记**：若 `truncated` 为真，在 `text[visible_chars:]` 前插入一行标记：
  `【⚠ 以下内容超出向量模型长度，检索时模型看不到】`
  未截断的块不插标记。

### 3.2 Prompt 模板（query × 块）

**system：**
```
你是搜索质量评估员。你会看到一个用户查询和一段从网页切出来的文本块，请判断这段文本块对回答该查询的价值，并检查它的完整性。只输出 JSON，不要输出其他内容。
```

**user：**
````
【查询】
{query}

【文本块】
```
{chunk_text_with_marker}
```

请按以下标准判断：

1. relevance（只看文本块本身，不要用你自己的知识补全）：
   - 3 = 能完整回答查询：包含回答所需的关键信息，读者不需要再看别处
   - 2 = 能部分回答：包含部分关键信息，但不完整
   - 1 = 相关但不能回答：主题相关，但没有回答查询所需的信息
   - 0 = 不相关

2. answer_cut（只在 relevance ≥ 1 时判断，否则填 "无"）：回答查询所需的内容是否在文本块的开头或结尾处被切断——例如句子没写完、表格只有一半或缺表头、列表/步骤只有一部分、标题和它的正文分开、开头明显接着前文。
   - "开头"：开头处缺了必要内容
   - "结尾"：结尾处没写完
   - "两端"：两处都有
   - "无"：没有被切断

3. hidden（文本块中出现"【⚠ 以下内容超出向量模型长度，检索时模型看不到】"标记时才判断）：回答查询的关键信息位于标记前还是标记后。
   - "未截断"：没有出现该标记
   - "无答案"：有标记，但 relevance ≤ 1
   - "全部在标记前"
   - "部分在标记后"
   - "全部在标记后"

4. reason：用不超过 40 字说明理由。

只输出如下 JSON：
{"relevance": 0, "answer_cut": "无", "hidden": "未截断", "reason": ""}
````

### 3.3 结果文件

`OUT_ROOT/judge/results/judgments_query.jsonl`，每行：
```json
{"key": "...", "qid": "...", "chunk_id": "...", "relevance": 3, "answer_cut": "无",
 "hidden": "未截断", "reason": "...", "model": "...", "prompt_version": "q-v1"}
```

**校验**：`relevance ∈ {0,1,2,3}`；`answer_cut ∈ {无, 开头, 结尾, 两端}`；`hidden` 取值合法；未截断的块 `hidden` 必须是"未截断"，截断块不能是"未截断"（不一致的按规则纠正并计数，写进报告）。

### 3.4 汇总指标（每个方法，在同一批 query 上）

把判断结果按 key 回填到每个方法的每一块，再按 query 计算：

| 指标 | 定义 |
|---|---|
| **完整可答率@k**（k = 1, 3, 10） | 前 k 块中至少有一块 `relevance = 3` 的 query 占比 |
| **可答率@k** | 前 k 块中至少有一块 `relevance ≥ 2` 的 query 占比 |
| **有用块数@10** | 前 10 块中 `relevance ≥ 2` 的块数，按 query 平均 |
| **MRR_答** | 第一个 `relevance ≥ 2` 的块的名次倒数，按 query 平均（没有记 0） |
| **答案切断率** | 所有 `relevance ≥ 2` 的块中，`answer_cut ≠ 无` 的比例；另分列 开头 / 结尾 / 两端 |
| **截断块占比** | 前 10 块中 `truncated` 的比例（导出时已有，不需要 LLM） |
| **答案落在截断部分** | `relevance ≥ 2` 且被截断的块中，`hidden ∈ {部分在标记后, 全部在标记后}` 的比例；以及它占全部 `relevance ≥ 2` 块的比例 |
| **截断影响 query 数** | 至少有一个"`relevance ≥ 2` 且答案部分/全部在标记后"的块的 query 数 |

**显著性**：对"完整可答率@10、可答率@10、MRR_答、有用块数@10"做相对 `content_br` 的**配对 bootstrap**（按 query 重采样 2000 次，种子 0），给出差值和 95% 置信区间（单位：百分点；有用块数用原单位）。

**分组**（可选，样本够时做）：按 query 的标准答案网页是否含表格 / 列表（看该 query 召回块里 `html_container` 的 `kind`，或不分组）；按块是否 `is_gold` 分开统计 relevance 分布。

---

## 4. 任务二：页面视角（切分本身的完整性）

两类判断，都**盲评**（不出现方法名），对同一网页不同方法的块**打乱顺序**后再发请求。

### 4.1 切点判断

每个方法、每个 `boundaries` 元素一条任务，key = `md5(url + "|" + method + "|b" + i)`。

**system：**
```
你是文本切分质量评估员。你会看到一篇网页正文被切开处前后的两段文字，请判断这个切点是否切坏了内容。只输出 JSON。
```

**user：**
````
【切点前（上一块的末尾）】
```
{tail}
```

【切点后（下一块的开头）】
```
{head}
```

判断这个切点：
- severity：
  - 0 = 切点落在两个完整、相对独立的内容之间（例如两个段落、两个不同话题之间），下一块可以自然地开头
  - 1 = 轻微：没有切断句子，但把本该放在一起的内容分开了——同一个列表/步骤序列/表格被分到两块、标题或引导句和它的正文被分开、问题和答案被分开
  - 2 = 严重：切断了一句话、一个列表项、一个步骤、一个表格行或单元格
- broken：被切坏的对象，取值 "无" / "句子" / "列表项" / "步骤" / "表格" / "标题与正文" / "段落" / "问答" / "其他"
- reason：不超过 40 字

注意：文字两端可能因为截取窗口而不完整（上一块末尾的开头、下一块开头的结尾），只判断切点处。
如果下一块开头重复了表头（例如"学校 | 分数"后接表格行），这是续块的表头，不算切坏。

只输出 JSON：
{"severity": 0, "broken": "无", "reason": ""}
````

### 4.2 块完整性判断

每个方法、每个块一条任务，key = `md5(url + "|" + chunk_id)`（同一网页不同方法切出相同的块只判一次）。块被截断时同样插入截断标记（只作提示，不影响这里的判断标准）。

**system：**
```
你是文本切分质量评估员。你会看到从一篇网页切出来的一个文本块，请判断它作为一个独立的检索单元是否完整、可读。只输出 JSON。
```

**user：**
````
【文本块】
```
{chunk_text_with_marker}
```

请判断：
- standalone：
  - 3 = 能独立读懂：主题明确，内容完整
  - 2 = 基本能读懂，但缺少一些上下文（例如不知道"它"指什么、缺了标题）
  - 1 = 读不懂或明显残缺
- start_broken：开头是否从半句话、半个列表、半个表格或明显承接前文的位置开始（true/false）
- end_broken：结尾是否没写完（true/false）
- mixed：是否混入了与主体无关的另一部分内容，例如把"注意事项"和"操作步骤"、正文和评论/导航/免责声明拼在一起（true/false）
- reason：不超过 40 字

只输出 JSON：
{"standalone": 3, "start_broken": false, "end_broken": false, "mixed": false, "reason": ""}
````

### 4.3 结果文件

- `OUT_ROOT/judge/results/judgments_boundary.jsonl`：`{"key", "url", "method", "i", "severity", "broken", "reason", "model", "prompt_version": "b-v1"}`
- `OUT_ROOT/judge/results/judgments_chunk.jsonl`：`{"key", "url", "chunk_id", "standalone", "start_broken", "end_broken", "mixed", "reason", "model", "prompt_version": "c-v1"}`

### 4.4 汇总指标（每个方法）

| 指标 | 定义 |
|---|---|
| 块/页、平均字数 | 来自导出数据 |
| **坏切点率** | `severity ≥ 1` 的切点 / 全部切点 |
| **严重坏切点率** | `severity = 2` 的切点 / 全部切点 |
| 每页坏切点数 | 坏切点数按网页平均（块越少切点越少，这个指标和"坏切点率"要一起看） |
| 坏切点类型分布 | `broken` 各取值的占比 |
| **独立可读均分** | `standalone` 平均值；以及 `standalone = 3` 的比例 |
| 开头残缺率 / 结尾残缺率 | `start_broken` / `end_broken` 为真的块占比 |
| **混装率** | `mixed` 为真的块占比 |
| 截断块占比 / 模型看不到的字占比 | 来自导出数据（`truncated`、`visible_chars`） |

**按页对比**：对每个网页，比较 `html_container` 与 `content_br`、与 `html_struct_secmerge_t512` 的"严重坏切点数"和"独立可读均分"，统计 赢 / 平 / 输 的网页数（配对符号检验或 bootstrap 给出置信区间）。

---

## 5. 质量控制

1. **先小规模跑通**：每类任务先跑 20 条，人工看输出是否符合定义，再全量。
2. **一致性抽查**：每类任务随机抽 50 条用同一模型再判一次，报告一致率（relevance 完全一致率、severity 完全一致率）。一致率 < 80% 时在报告里标注"判断不稳定"。
3. **人工抽查清单**：每类任务随机抽 30 条，导出成 `results/spot_check_*.md`（输入 + LLM 输出），供人工复核。
4. **记录元信息**：模型名、prompt 版本、调用次数、失败次数、开始结束时间，写入 `results/run_meta.json`。
5. **不要改 prompt 中途重跑一部分**：prompt 一旦改动，`prompt_version` 加 1，并对全部任务重跑。

---

## 6. 最终产出

在 `OUT_ROOT/judge/results/` 下：

| 文件 | 内容 |
|---|---|
| `judgments_query.jsonl` / `judgments_boundary.jsonl` / `judgments_chunk.jsonl` | 原始判断 |
| `summary_query.csv` / `summary_page.csv` | 各方法的指标表 |
| `report.md` | 报告（结构见下） |
| `examples.md` | 典型案例 |
| `spot_check_*.md`、`run_meta.json` | 质检材料 |

### report.md 结构

1. **结论（3–5 句）**：各方法在"可答率、答案切断、截断影响、坏切点、混装"上的排序和差距，`html_container` 相对 `content_br` 和原结构切分是好是坏、差多少、是否显著。
2. **query 视角表**：行 = 方法，列 = 完整可答率@1/@3/@10、可答率@10、有用块数@10、MRR_答、答案切断率、截断块占比、答案落在截断部分；相对 `content_br` 的差值和 95% 置信区间。
3. **截断专项**：`html_container` 被截断的块有多少、其中多少块的答案部分/全部在截断后、影响多少条 query；与其他方法对比。
4. **页面视角表**：行 = 方法，列 = 块/页、坏切点率、严重坏切点率、每页坏切点数、独立可读均分、开头/结尾残缺率、混装率、截断块占比；按页赢/平/输。
5. **坏切点类型分布**（每个方法一行）。
6. **质检**：失败条数、一致率、人工抽查结论。

### examples.md

各挑 5 个，附 url / qid、各方法的相关块原文（截断标记保留）和 LLM 理由：
- `html_container` 明显更好的 query（例如其他方法答案被切断、它完整）；
- `html_container` 明显更差的 query（重点看是否因为截断：答案在标记后）；
- `html_container` 坏切点明显少的网页 / 明显多的网页。

---

## 7. 实现提示

- 读 gzip jsonl：`gzip.open(path, "rt", encoding="utf-8")`，逐行 `json.loads`，第 1 行是 meta。
- 插入截断标记：`t[:v] + "\n【⚠ 以下内容超出向量模型长度，检索时模型看不到】\n" + t[v:]`（`v = visible_chars`，只在 `truncated` 为真时插入）。
- JSON 解析：先直接 `json.loads`；失败时提取第一个 `{...}` 再解析；字段缺失或取值非法视为解析失败。
- 汇总时注意：一条 query 的某方法可能不足 10 块（结果列表里本来就少），按实际块数计算，不要补 0 块。
- bootstrap：对 query 级的差值向量 `d = 指标(方法) - 指标(content_br)` 重采样均值，取 2.5% / 97.5% 分位数。
