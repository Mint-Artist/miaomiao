# 业界是否根据 HTML 结构切分 chunk：证据、实现差异与落地建议

调研日期：2026-09-20。面向离线网页、搜索与 RAG 共用的数据链路；现有基线为 256 字符加贪心装箱，上游目前仅保留 `<br>`。

## 1. 结论

**会。利用 HTML／文档结构切分，已经是公开云产品和开源框架中的成熟工程路线。更准确的说法是：先解析结构，再结合长度预算生成 chunk。** Google Cloud 有明确的 HTML layout-aware chunking 配置，Azure 有结构解析与分段流水线；LangChain、Unstructured、Docling 可以在公开实现中核对具体逻辑。[Google 文档][D01]、[Azure 文档][D02]、[LangChain 源码][C01]、[Unstructured 文档][D04]、[Docling 文档][D06]。

但公开资料不支持以下推断：所有公司采用同一种算法；任意 HTML 标签都应成为切分点；结构切分在所有数据上优于其他方法；云知识库的配置就是同公司通用 Web 搜索引擎的内部索引策略。

**对你们的建议是保留并使用结构，继续保留长度控制与贪心合并，将 embedding 语义边界作为后续可对照的组件。** 这是结合证据与现有链路作出的工程判断，不是已经在你们查询集上验证的效果结论。

## 2. 先把四件事分开

| 环节 | 解决的问题 | HTML 结构的作用 |
|---|---|---|
| 正文抽取 | 哪些内容进入索引 | 区分正文、导航、推荐、页脚；不能仅靠换行 |
| 边界生成 | 在哪里断开 | 标题、段落、步骤、表格、列表提供候选边界 |
| chunk 表示 | 单个片段如何独立理解 | 补充所属标题、表头、适用条件、来源 |
| 检索与返回 | 找到什么、返回多少 | 小块检索，必要时返回父章节或相邻块 |

例如，给 chunk 添加标题属于“表示”；检索到子块后展开父块属于“返回”；两者都不能单凭名字归为一种新的切分边界算法。HTML 可以先转换成结构化对象或 Markdown，最终索引纯文本；中间没有丢掉关系，仍然是结构感知方案。

## 3. 可以核实的厂商与框架

### 3.1 Google Cloud：直接面向 HTML 的布局感知切分

官方明确将 HTML 列为 layout parser 支持格式，识别标题、段落、表格、列表等元素。`layoutBasedChunkingConfig` 控制切分，`includeAncestorHeadings` 可带入祖先标题；还可按标签、class、id 排除 HTML 噪声。[官方文档][D01]

这是“HTML 结构参与入库切分”的直接产品证据。它也说明正文去噪、结构识别、chunk 大小和上下文继承是不同配置问题。不要据此推断 Google Search 的整个网页索引实现。

### 3.2 Microsoft Azure：结构解析后，在章节内部限制长度

公开流水线是 Document Layout 识别结构，生成包含标题和内容的 Markdown 表示，再通过 Text Split 在 Markdown section 内分段，最后生成向量并保留父文档／标题字段；支持格式包含 HTML。[官方文档][D02]

当前文档还推荐新流水线采用 Content Understanding；其 semantic chunking 为预览功能，描述为遵循段落和标题边界。该新教程以 PDF 为例，本文不把它当成已确认的 HTML 专用实现，也不把“semantic”自动解释为句向量距离算法。[新版教程][D03]

### 3.3 LangChain：HTML 标题切分与元素保护是不同能力

`HTMLHeaderTextSplitter` 使用配置的标题层级，给输出附带标题元数据；它本身不能等同于完整的预算控制流水线。`HTMLSemanticPreservingSplitter` 则按标题分组，允许指定 `elements_to_preserve`，并用递归字符切分处理长内容。这里的 semantic preserving 不依赖逐句 embedding。[官方源码][C01]

需要特别检查两点：保护元素要显式配置；保护整张表可能使最终 chunk 超过 `max_chunk_size`。另外，“不拆元素”也不等于完整保留原 HTML 的行列编码，表格输出格式仍需核验。因此不能把默认参数直接作为你们的 256 字硬上限实现。

### 3.4 Unstructured：先得到元素，再组合成 chunk

`partition_html` 将网页解析成文档元素，chunking 再组合这些元素；`basic` 顺序装箱，`by_title` 增加章节边界。它不是先把全部内容抹成纯文本再找换行。[HTML 实践][D14]、[分段文档][D04]

`by_title` 的短章节合并参数可能重新跨过标题边界；若要求严格分章，需检查 `combine_text_under_n_chars`。本次固定的源码还提供表头重复、表格隔离以及跳过表格切分等选项，不能把不同选项下的长度语义混为一谈。[标题切分源码][C03]

有一处资料差异：partitioning 总表将 HTML 的 Table Support 标成 No，而当前 HTML parser 源码已有 `TableBlock`，输出 `Table` 和 `text_as_html`。这证明“只查介绍页”会误判，也不代表复杂跨行跨列表格已被完整支持；应锁定实际部署版本验收。[文档][D05]、[HTML parser 源码][C04]

### 3.5 Docling：文档层次、token 预算、上下文序列化结合

Docling 支持 HTML 输入，先转成统一文档表示。`HybridChunker` 在层次切分上进行 token 长度调整：拆分过长块，合并可合并的相邻块；`contextualize` 用于输出带上下文的表示。[格式支持][D07]、[切分文档][D06]

当前固定源码中，单独表格有逐行分段、重复表头的路径，普通文本有另一条细分路径；标题／图注会影响可用预算。还存在上下文本身超限时的退化处理，因此要同时检查最终长度与上下文是否仍在。[HybridChunker 源码][C06]

这是很适合参考的分层设计，但本次没有安装运行 Docling，也没有把其源码行为视为你们数据上的效果证明。

### 3.6 LlamaIndex：支持 HTML，不代表默认就是完整标题树

当前 `HTMLNodeParser` 按配置标签抽取文本，在遍历中组合连续同名标签，输出标签元数据。默认标签包含标题、段落、列表项和部分行内标签，但没有 `table/tr/th/td`；这一 parser 中也没有统一的 token 上限配置。[官方源码][C02]

因此，它是 HTML 感知解析的证据，却不能直接称为完善的“标题路径＋表格行切分器”。要处理你们的高考网页，仍需要表格专门逻辑和预算控制。也不要把这个 parser 的行为扩大为 LlamaIndex 全部产品能力。

### 3.7 AWS Bedrock：固定、父子、语义策略并存

Bedrock 同时提供固定大小、父子层次、embedding 语义等策略。其文档还明确指出，对解析结果或 HTML 转换内容，切分可能尊重页面／section 等逻辑边界，即使长度允许也不跨界合并。[官方文档][D08]

语义策略提供句子缓冲、距离分位数阈值等控制，并有额外模型费用。父子策略控制父／子大小与返回扩展，不能简单等同于 HTML 的 h1/h2 层次。这体现了多种方法并存，而非纯语义切分统一替代结构规则。

### 3.8 阿里云百炼：标题路径已经进入切片表示

百炼公开支持按标题切分，超长时仍受最大分段长度约束；切片可带最近一级标题 `title` 和完整层级路径 `hier_title`。[官方文档][D09]

这证明标题结构和切片表示在国内产品中也有明确实践。但该说明主要是通用文档能力，本文不据此认定其任意 HTML 输入都保留完整 DOM，或推断夸克等线上搜索产品的内部实现。

### 3.9 OpenAI、Anthropic、豆包等：公开能力与内部实现的边界

Anthropic 的 Contextual Retrieval 给现有 chunk 添加文档背景，改善其检索表示；它提供的是上下文补全思路，不是网页 DOM 边界算法的直接证据。[官方文章][D13]

这次检索没有获得足以复现 OpenAI、豆包通用网页搜索离线切分的公开细节，不能替它们断言采用某种 HTML 或 embedding 边界策略。火山知识库相关页面在本次网页工具中也未能读取，不用它支撑新的 HTML 专项结论。

Google Search 的公开说明确认有 passage ranking，但其表述是借助 passage 理解网页相关性，并未给出完整的离线 DOM chunk 算法。[Google Search 文档][D10]。对其他公司的“搜索”与“知识库”，同样应分别讨论。

## 4. 实现比较：不是一句“结构切分”就能覆盖

下表是上述资料的工程归纳；“可借鉴”不代表默认参数即可上线。

| 方案 | 主要边界依据 | 长度控制 | 表格／上下文重点 | 对你们的用途 |
|---|---|---|---|---|
| Google layout-aware | 解析后的元素及层次 | 配置 token 预算 | 可配置祖先标题、HTML 去噪 | 证明上游结构接口有产品先例 |
| Azure Layout + Split | section，再细分文本 | section 内约束 | 标题与父文档字段 | 参考流水线解耦 |
| LangChain HTML | 配置标题、保护元素 | 递归字符切分；可超限 | 整表保护与硬预算有冲突 | 快速对照，不直接承诺 256 字 |
| Unstructured | 文档元素、Title | 合并与超长拆分 | 短章合并、表格选项需显式设定 | 接近“元素＋装箱”的基线 |
| Docling Hybrid | 统一文档结构 | token 细分与合并 | 上下文预算、表格分段路径 | 参考组件划分和边界验收 |
| LlamaIndex HTMLNodeParser | 标签遍历 | 需要后续补足 | 默认表格处理不充分 | 基础解析对照 |
| Bedrock | 多策略可选 | token 预算 | 父子返回和语义边界相互独立 | 对照算法维度 |
| 百炼 | 标题等文档边界 | 最大长度 | 短标题及标题路径 | 参考 chunk 上下文字段 |

其中的差异至少包括：是否跨标题合并、标题放正文还是 metadata、表格是否整块保护、超长元素如何退化、预算按字符还是 token、上下文是否计入预算。算法对比必须把这些配置记录下来。

## 5. 论文与前沿：结构、语义、上下文正在组合

| 研究 | 可确认的方向 | 与你们问题的关系和限制 |
|---|---|---|
| [HtmlRAG，WWW 2025][D11] | 清理 HTML，构建块树，按相关性剪枝 | 直接讨论标题和表格结构的信息价值；重点是检索后知识表示与压缩，不能直接证明固定 256 字离线索引最优 |
| [AutoChunker，ACL 2025 Industry][D15] | 结构感知、语言模型、树表示、噪声处理结合 | 论文报告用于在线产品支持系统；可以支持结构与模型结合的路线，但不是任意 DOM 标签的机械切分 |
| [HiChunk，ACL 2026][D16] | 层次文档组织、Auto-Merge 检索、证据密集评测 | 值得把“如何切”和“如何合并返回”一起研究；不能用它证明只保留 h 标签就足够 |
| [Is Semantic Chunking Worth the Computational Cost?，NAACL 2025 Findings][D12] | 比较固定与语义切分，发现收益并不稳定 | 不支持默认用 embedding 边界替代基线；也不是 HTML 结构切分的直接对照实验 |
| [Late Chunking][D17] | 先在长上下文中编码，再按块池化 | 改的是 embedding 上下文，结构边界仍可使用；不是从丢失的 HTML 中恢复结构 |
| [Beyond Chunk-Then-Embed，2026][D18] | 区分切分方法和 embedding 范式，并对比两类检索任务 | 效果依赖任务；文中 structure-based 包含固定、句子、段落等，不应读成“已证明 HTML DOM 最优” |
| [pplx-embed 技术报告，2026][D19] | 标准与上下文化 embedding，使用 late chunking 思路 | 是搜索厂商研究上下文表示的公开例子，不能反推其线上 HTML 边界实现 |

本节只概括本次核对的论文页面和已有资料，不声称复现了这些论文，也不拼接不同数据集的收益百分比做排行榜。

**可以观察到的方向是：结构边界、模型辅助判断、上下文表示、层次返回共同优化。没有充分证据说明行业正在统一放弃结构、改用纯 embedding 语义切分。** 这是对公开样本的归纳，不是行业市场份额统计。

## 6. 为什么只有 `<br>` 会妨碍你们

问题不在于输入是一个字符串。字符串完全可以表达简化 HTML；问题在于不同关系被压成同一种分隔符。

```html
<h2>2025 年物理组大学位次</h2>
<table>
  <tr><th>学校</th><th>分数</th><th>位次</th></tr>
  <tr><td>甲大学</td><td>680</td><td>285</td></tr>
</table>
```

如果变成 `2025 年物理组大学位次<br>学校<br>分数<br>位次<br>甲大学<br>680<br>285`，下游无法可靠区分标题、列名和数据行。相同纯文本还可能来自列表或普通段落，恢复并不唯一。语义模型可以猜测，无法保证无损重建。

但“只有 br”不等于完全不能切：仍可用标点、长度、编号模式、文本语义等产生基线。应准确描述为**缺少可靠的显式结构信号，尤其损害表格关系、标题作用域和步骤嵌套的利用**。

对于 EC，建议最小交付仍是一个字符串，保留以下必要语义：

| 信息 | 建议标签／属性 | 用途 |
|---|---|---|
| 标题 | `h1`～`h6` | 划定章节，建立标题路径 |
| 段落、局部换行 | `p`、`br` | 区分段落边界和单元格内换行 |
| 列表与步骤 | `ul`、`ol`、`li`；需要时保留 `start/value` | 保留顺序、嵌套和步骤附属说明 |
| 表格 | `table/caption/thead/tbody/tfoot/tr/th/td`；`rowspan/colspan`，已有 `scope/headers` 尽量保留 | 重建行列及表头对应关系 |
| 图及说明 | `figure/figcaption/img`；`src/alt` | 绑定图片和文字；OCR 是另一环节 |
| 链接 | `a[href]` | 保留锚文本与跳转对象 |
| 提示／适用范围 | 简化 `section` 或 `div`，约定少量语义 class，如 `note/metadata` | 保留局部条件和提示容器 |
| 代码、引用等实际出现的类型 | `pre/code/blockquote` | 避免被普通段落规则破坏 |

这是本项目的接口建议，不是这些厂商共用的标准。没有必要保留全部样式 div/class；去掉容器时应保留子内容、顺序和边界。`strong/b` 可以作为“疑似标题”的线索，不能一律升为标题。规范化标题需区分原生与推断来源，低置信度时保留原结构。

## 7. 你们的推荐第一版：结构约束下的顺序装箱

以下是建议实现，不是对任一厂商源码的照搬。

```text
简化 HTML 字符串
  → 解析并校验正文、阅读顺序
  → 建立标题层次和块类型
  → 构造 paragraph / list_item / table_row / figure 等候选单元
  → 绑定所属标题、表头及明确的局部适用条件
  → 计算加入上下文后的预算
  → 在同一允许的结构范围内顺序贪心合并
  → 对超长单元按类型降级细分
  → 产出 chunk、父块关系、源位置和拆分原因
```

第一版可先将同级章节作为硬边界，章内段落作为优先边界；不要为了填满 256 字跨到下一个主题。短章节是否合并应单独开实验开关。`div` 本身不自动构成硬边界，因为它也可能只是布局容器。

长列表优先按完整 `li` 分组；步骤编号、步骤内图片和条件说明要关联。长表格优先按连续完整行分组，并附表头和局部范围。长段落再按句子拆；单句或单行仍超限时，采用明确的续块策略并保留父 ID，不能静默截断，也不能一面要求硬上限一面无限保护整表。

**256 的定义先不改变，但必须写清计数口径。** 之前演示脚本用的是 Unicode 字符数，包含加入正文的标题、表头、空格、换行、标点，不包含单独存储的 metadata／图片 URL。它不是 256 个汉字，也不是 256 tokens。若线上口径不同，先对齐再比较；此外还需检查模型实际输入的 token 上限。

对只放 metadata 的字段，要明确检索器是否使用：保存了标题不等于向量已看见标题。可分别保存原文 `text`、用于检索的 `contextual_text`、父章节引用和源位置。上游仍然只传字符串，内部输出可以是结构化对象。

### 用此前两个页面理解

华为说明页：把适用产品／版本、功能限制、使用前条件和操作步骤识别为不同类型。适用范围可以作为共享元数据；不能把整份产品清单无条件重复到每个 256 字块。操作步骤中的型号分支属于步骤内部说明，不能因遇到一个 `p` 就自动与步骤拆散。查询命中具体操作时，可按需要返回所属步骤和关联条件。

高考网页：物理类／历史类子标题在原始 HTML 中是短的加粗段落，需要规范化或有依据地推断；表格内的 `br` 是单元格内部换行，不能当新数据行。大学表应继承本地“2025 年、物理组”的范围，不应因网页总标题含 2026，就给每个表格 chunk 写成 2026 数据。

已生成的 [256 字切分结果](/Users/awh/work/chunk2026/downloads/gk100_read_27177662/chunks/切分结果_256.md) 是上述方向的页面示范：35 个块，其中 30 个表格块覆盖 171 条数据行。**这验证了该示范的结构覆盖和长度约束，没有验证检索效果更好。** 表头／条件占用预算、图片无 OCR、对页面结构的规则依赖，仍需进入评估。

## 8. 结构和语义该如何选择

| 页面条件 | 建议先做 | 再考虑 |
|---|---|---|
| 标题、步骤、表格清晰 | 结构约束＋长度预算 | 上下文继承、父块返回 |
| 很长的无标题正文 | 段落／句子基线 | embedding 边界或模型主题识别 |
| 大量伪标题、布局 div | 改进正文抽取和结构规范化 | 在低置信度块上模型辅助 |
| 表格很多 | 行列解析、表头与条件绑定 | 表格检索／结构化查询，不能仅靠句相似度 |
| 很短但高度依赖上下文的片段 | 标题／条件补充 | Contextual Retrieval、late chunking |

我的优先级判断：**先修复上游结构保真，再实现可解释的结构基线，然后研究语义补充。** 原因是结构保留可以直接解决当前可见的关系丢失，而完整的模型语义方案仍依赖输入质量，也需要额外成本与效果验证。

## 9. 最小但公平的实验

建议不要把“改解析、改大小、补标题、改返回数量”同时做完只报一个收益。保持同一批原始 HTML、正文内容、查询、检索模型和重排模型，逐步比较：

| 实验 | 输入／方法 | 希望隔离的问题 |
|---|---|---|
| A | 将同一正文压为 br，跑现有 256 字算法 | 复现可比基线 |
| B | 保留结构，结构约束装箱；先不增加上下文文本 | 只改变边界是否有价值 |
| C | B 加标题、表头、局部条件，并计入同一预算 | 上下文补充的净效果 |
| D | 与 C 使用同样上下文，在允许范围内用 embedding 找边界 | 语义决策是否进一步增益 |
| E | 固定 C 或 D 的索引，增加有预算限制的父块返回 | 单独评估检索后扩展 |

表格在 B 中至少保留当前行内部的列对应关系；重复表头属于 C 的上下文处理。这样的定义虽不覆盖所有组合，但能避免把表示改进误报成纯边界收益。

先检查工程不变量：正文覆盖、顺序、非预期重复、跨章节比例、表格行列关系、上下文丢失和最终输入超限率。再用真实 query 做网页级相关性评估，以及 RAG 的证据覆盖、答案支持性评估。比较时同时报告固定 top-k 和固定返回 token 预算；不同 chunk 大小下相同 k 不代表相同成本。

没有证据标注时，可从不同方法的候选结果池中取样，由人工标到原文位置；模型辅助标注需抽检，不应仅让模型判“这个 chunk 看起来完整”。按模板、站点、结构类型分层，保留未参与调参的测试集，同时记录索引体积、chunk 数和延迟。256／512 等预算扫描应在边界策略对比之后独立进行。

## 10. 证据与版本说明

本报告依据官方产品文档、官方开源代码和作者／会议论文页面。没有调用收费云 API，没有运行统一跨框架基准，也没有测得你们链路的收益。

代码按获取时的 Git commit 固定下载，区别于文档中的发布版本；源码新功能未必存在于已经安装的 PyPI 版本。网页文档也存在差异：例如 Google 页面顶部的默认解析器叙述与 REST 省略配置时的说明并不完全一致。因此，本报告使用明确配置支持的行为，不依赖模糊的“默认”。

本目录保存原始资料快照、下载程序和 SHA-256 清单。部分资料描述预览能力或开发中的实现，实际实验应记录所用依赖版本和参数；表格、嵌套列表、长标题、空标题和超长单元要用自己的页面逐类验收。

[D01]: https://docs.cloud.google.com/gemini/enterprise/docs/parse-chunk-documents
[D02]: https://learn.microsoft.com/en-us/azure/search/search-how-to-semantic-chunking
[D03]: https://learn.microsoft.com/en-us/azure/search/search-how-to-semantic-chunking-content-understanding
[D04]: https://docs.unstructured.io/open-source/core-functionality/chunking
[D05]: https://docs.unstructured.io/open-source/core-functionality/partitioning
[D06]: https://docling-project.github.io/docling/concepts/chunking/
[D07]: https://docling-project.github.io/docling/
[D08]: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html
[D09]: https://help.aliyun.com/zh/model-studio/rag-knowledge-base
[D10]: https://developers.google.com/search/docs/appearance/ranking-systems-guide
[D11]: https://arxiv.org/abs/2411.02959
[D12]: https://aclanthology.org/2025.findings-naacl.114/
[D13]: https://www.anthropic.com/engineering/contextual-retrieval
[D14]: https://unstructured.io/blog/easy-web-scraping-and-chunking-by-document-elements-for-llms
[D15]: https://aclanthology.org/2025.acl-industry.69/
[D16]: https://aclanthology.org/2026.acl-long.1372/
[D17]: https://arxiv.org/abs/2409.04701
[D18]: https://arxiv.org/abs/2602.16974
[D19]: https://arxiv.org/abs/2602.11151
[C01]: https://github.com/langchain-ai/langchain/blob/68754c23c956e16cdebec00dcfe076257b4278ed/libs/text-splitters/langchain_text_splitters/html.py
[C02]: https://github.com/run-llama/llama_index/blob/f475afd8a9bbda84f252567e045d89d07b5701b3/llama-index-core/llama_index/core/node_parser/file/html.py
[C03]: https://github.com/Unstructured-IO/unstructured/blob/3376cc96a49521669f3b64028799db16ca76136f/unstructured/chunking/title.py
[C04]: https://github.com/Unstructured-IO/unstructured/blob/3376cc96a49521669f3b64028799db16ca76136f/unstructured/partition/html/parser.py
[C05]: https://github.com/docling-project/docling/blob/890dd42d017497c955a56a1d2cfc3f0af5bc2aa9/docling/backend/html_backend.py
[C06]: https://github.com/docling-project/docling-core/blob/4582a183c55b95ccfb1aea9e3febfe19cc0bd595/docling_core/transforms/chunker/hybrid_chunker.py
