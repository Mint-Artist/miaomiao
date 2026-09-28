# HTML 结构切分（简版 HTML → chunk JSONL）

2026-09-23。两段式架构的**后半段**：以
[简版HTML提取_2026-09-23](../简版HTML提取_2026-09-23/) 产出的受控标签 HTML 为输入，
按 HTML 结构在 256 字符预算内装箱切分。切分策略是架构中的动态层，
本项目的每个决策都收敛到 `ChunkConfig`，可随实验调整。

## 目录

| 文件 | 说明 |
|---|---|
| `chunker.py` | 切分器（库 + CLI），含 `CHUNKER_VERSION` |
| `run_examples.py` | 端到端：原始 HTML → 提取（带锚点）→ 切分，跑三个样本 |
| `validate.py` | 验收：通用不变量 + 样本结构断言，36 项 |
| `output/` | 简版 HTML、chunks.jsonl、报告 |

## 用法

```bash
python3 run_examples.py                 # 三个样本端到端
python3 validate.py                     # 验收，36 项全过退出码为 0

# 单个文档
python3 chunker.py --html output/gk100_read_27177662.simplified.html \
    --doc-id gk100 --url "https://m.gk100.com/read_27177662.htm" \
    --out chunks.jsonl --report report.json
# 实验开关：--budget N / --merge-short-sections / --context-metadata / --no-title-path
```

## 管线

```
简版 HTML → 原子流（p / li / note / 表格行 / figure…）+ 标题路径栈
  → 上下文绑定（标题路径、表头、可选 metadata；超长逐级丢深层标题）
  → 超长降级：li 按子结构 → 段落按 句号→分号→逗号/顿号 → 硬断（continued+parent_id）；
    表格行按连续行分组，单行超预算拆最长单元格
  → 同结构作用域贪心装箱：章节硬边界；伪标题（整段加粗短段落）前断块
  → JSONL 输出 + 不变量统计报告
```

## 关键决策（2026-09-23 与需求方确认）

| 决策 | 取值 | 理由 |
|---|---|---|
| 预算与完整性冲突 | 预算硬上限，按类型降级 | 不做 LangChain 式整表保护（可冲破预算） |
| 上下文注入位置 | `contextual_text`，计入预算 | Anthropic/百炼均注入可检索文本；`text` 保持纯原文 |
| 上下文内容 | 文档标题 + 祖先标题路径 + 表头（表格块） | 只带祖先链不带兄弟章节；占预算比超 40% 逐级丢深层 |
| metadata 注入 | 默认关（开关） | 华为"适用产品"90+ 字，注入吃掉三分之一预算 |
| 短章节合并 | 默认关（开关） | 不为凑满 256 跨主题 |
| 伪标题 | 装箱时其前断块（开关） | 律图/高考的加粗短段落是事实章节头，不能粘在前块尾 |
| LLM 上下文 | 预留 `llm_context` 接口，v1 不接入 | 收益需实验验证（P01 结论：不稳定），接已有能力时按 chunk 哈希缓存 |
| 预算口径 | 256 Unicode 字符，含上下文，不含图片 URL | 与现有基线一致；`counter` 字段预留 token 计数 |

## 输出 schema（JSONL 每行一个 chunk）

`chunk_id / doc_id / url / text / contextual_text / chars / contextual_chars /
context_sources / heading_path / parent_id / continued / prev / next /
data_src / kind / split_reason / cross_section / images / atom_ids /
chunker_version / config_hash`

- `text` 纯原文（不加枚举前缀、不加表头）；`contextual_text` = 上下文 + text；
- 续块 `continued=true` 且 `parent_id` 指向该原子的首个 chunk；
- `split_reason` 记录降级生效级别（sentence/semicolon/clause/hard_wrap/table_cell）；
- `data_src` 来自提取器 `--anchors` 的元素序号，可回溯原文位置。

## 样本结果（validate.py 36 项全过）

| 文档 | 块数 | 说明 |
|---|---|---|
| 华为页 | 3 | 功能介绍 / 须知 note / 操作流程 ol，与人工预期分组一致 |
| 高考页 | 33 | 27 个纯表格块覆盖 163 行 + 边界混合块 8 行 = 171 行全覆盖；每块带表头与章节路径 |
| 律图页 | 6 | 三个伪标题章节各自起块；末块为编辑导流段（提取阶段有意保留） |

报告（`output/*.report.json`）含不变量统计：覆盖、顺序、超限数、
跨章节块数、续块数、降级计数、填充率——实验对比时逐项比较。

## 已知限制

- 章节边界处的段落与表格行可同块（同作用域按预算装箱），如"篇幅有限…"与
  前 7 行同块；如需纯表格块可加边界规则，留作实验项。
- rowspan/colspan 表格的拆行分组只有代码路径，缺真实样本验收。
- 预算处硬断（hard_wrap）目前三个样本未触发，需长无标点文本样本验证。
- 短孤儿块（如律图 C03，33 字）是尊重边界的正常产物，是否合并进实验
  （merge_short_sections）由 A/B 决定。
- token 计数接口、LLM 上下文、软硬预算切换均为预留，未在样本上验证。

## 下一步

按调研 §9 的实验矩阵推进：A=压平 br 基线（现有 256 算法）、B=本项目默认配置、
C=B+上下文（本项目的 contextual_text）、C′=C+LLM 上下文、E=父子/邻块返回。
比较时同时报固定 top-k 和固定 token 预算，config_hash 随结果记录。
