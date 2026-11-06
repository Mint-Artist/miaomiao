# 论文资料导航：文档切分与 RAG

本目录集中整理项目中已经完成原文、全文中文翻译和论文解读的论文。每个论文目录名称包含论文简称、会议/期刊和年份；目录内统一分为“原文”“全文翻译”“解读”三部分。

| 论文 | 会议/年份 | 内容概述 | 核心亮点 |
|---|---|---|---|
| [HtmlRAG](01_HtmlRAG_WWW_2025/) | WWW 2025 | 研究在 RAG 中直接保留和压缩 HTML 结构，比较 HTML、纯文本与 Markdown 的知识表示，并通过结构块树和两阶段剪枝组织检索结果。 | 说明标题、表格、代码、链接等结构信息在纯文本化时会丢失；提出清理、结构块树、嵌入粗剪和生成式细剪的漏斗式流程。适合参考“在线证据组织”，不能直接证明离线 chunk 边界最优。 |
| [AutoChunker](02_AutoChunker_ACL_2025/) | ACL Industry Track 2025 | 使用 LLM 按句子/行的结构合并文本，并用噪声、完整性、上下文转换和任务相关性评估切分质量。 | 把“索引粒度”和“返回粒度”分离：小块负责召回，命中后返回子树上下文；同时显示 LLM 切分成本明显高于递归切分，且切分器和评估器同源会产生偏置。 |
| [HiChunk](03_HiChunk_ACL_2026/) | ACL Long Papers 2026 | 面向长文档 RAG 的层级切分与查询时 Auto-Merge，构建 HiCBench 证据密集评测集。 | 证明证据稀疏问题难以区分切分方法；层级结构配合查询时合并能提升证据召回和事实覆盖；评测应按固定 token 预算，而不是只比较 top-k 块数。 |
| [Is Semantic Chunking Worth the Computational Cost?](04_SemanticChunkingCost_Findings-NAACL_2025/) | Findings of NAACL 2025 | 比较固定大小、语义断点和语义聚类切分在检索与问答任务上的效果，并讨论计算成本。 | 语义切分不必然带来稳定收益，简单基线仍应保留；论文支持先按章节等结构约束再切分，但未证明网页 HTML 结构方案一定更优。 |
| [Beyond Chunk-Then-Embed](05_BeyondChunkThenEmbed_SIGIR_2026/) | SIGIR 2026 | 统一评测结构化、语义、LLM 引导切分，以及嵌入前切分和 late/contextual chunking 在文档内与语料库内检索中的差异。 | 最优策略依赖任务轴：语料库内检索中简单结构化方法更稳，文档内证据定位才更可能受益于高级语义切分；late chunking 对两类任务的收益方向相反。 |

## 阅读建议

- 先看“解读/论文解读报告.md”，了解结论、证据边界和工程映射。
- 再看“全文翻译/全文中文翻译.md”，核对方法、表格和限定条件。
- 需要引用或复现实验时，使用“原文”中的 PDF。

## 项目中的其他相关资料

项目根目录中的 `chunk_research_2026-09-20/` 仍保留更广泛的论文原文、网页资料、厂商实践和实验方案；本目录只集中整理已完成三类交付物的 5 个论文主题（其中 `SemanticChunkingCost` 的原文来自 Findings of NAACL 2025）。

> 说明：`chunk_research_2026-09-20/papers/` 中还有仅下载、尚未完成全文翻译和解读的参考论文；它们暂不复制到本目录，避免把不完整材料伪装成三件套。来源清单保存在 `00_整理索引_sources_manifest.json`。

## 许可与署名

五篇论文原文均为开放许可，可在注明出处的前提下转载和改编：HtmlRAG、AutoChunker、HiChunk、SemanticChunkingCost 为 CC BY 4.0，Beyond Chunk-Then-Embed 为 CC BY-SA 4.0。各目录"原文"中保存原始 PDF，出处与校验值见 `00_整理索引_sources_manifest.json`。"全文翻译"和"解读"是基于原文的改编作品，版权归原作者，按原文许可共享（Beyond Chunk-Then-Embed 的翻译与解读按 CC BY-SA 4.0 共享）。
