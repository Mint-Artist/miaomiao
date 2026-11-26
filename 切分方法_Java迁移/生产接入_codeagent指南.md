# 生产接入指南（给内网 code agent）

> 任务：把 `java/` 里的 5 种结构切分方法接入现有 Java 生产切分服务，作为新的可选切分方法，与现有 `content_br` 并存。
> 核心原则：**这是"接入"，不是"重写"**。算法已与 Python 离线评测逐字对齐（见 `Python与Java差异对照.md`），任何对算法类的"优化 / 简化 / 改写"都会破坏一致性，离线评测结论随之失效。

---

## 0. 先读什么

1. `README.md`：包结构、API、构建方式。
2. `切分方法说明.md`：每种方法的规则和参数。只需理解，不要据此重写。
3. `Python与Java差异对照.md`：哪些写法是刻意的（`PyStr`、`Soup`、补 doctype 等），**不能改回 Java 惯用写法**。
4. 生产代码库：找出以下几点，写进第 1 步的调研结论。
   - 现有切分入口：接口或抽象类、`content_br` 的实现类、调用方。
   - 切分方法如何选择：配置项、枚举、开关。
   - 输入：markdown、html 分别从哪个字段来，进入切分前是否做过清洗、截断、转码。
   - 输出：chunk 的数据结构（文本、序号、类型、偏移、标题前缀等），以及切分后还有哪些处理（上下文 LLM chunk、长度截断、向量化）。
   - 运行环境：JDK 版本、Maven 私服是否有 jsoup 1.21.2、项目里是否已有其他版本的 jsoup。

---

## 1. 调研（只读，不改代码）

输出一份调研结论，内容包括：

| 项 | 要回答的问题 |
|---|---|
| 接入点 | 新方法挂在哪个接口下；方法名如何在配置里表达 |
| 输入一致性 | 生产拿到的 markdown/html 与离线评测用的是否**同一份字符串**（同一字段、同一份原始数据、有无额外预处理） |
| 输出映射 | `Chunk.getText()` / `getKind()` 映射到生产 chunk 的哪些字段；生产需要但本包不提供的字段（如偏移）如何处理 |
| 依赖冲突 | `mvn dependency:tree` 中 jsoup 的版本；有冲突时的方案 |
| 后处理 | 切分后的处理对新旧方法是否一致 |

**输入一致性是最大风险**：如果生产的 html 比离线评测的多一步清洗，或者 markdown 来自另一个转换器，即使 Java 和 Python 逐字一致，线上效果也不等于离线评测。发现不一致时，先停下来报告，不要自行调整算法去适配。

---

## 2. 在内网把包单独跑通（不碰生产代码）

```bash
cd 切分方法_Java迁移/java
mvn test        # GoldenParityTest 必须全部通过（已知差异 1 页单独计数，不算失败）
mvn package
```

- 必须通过才能进入下一步。不通过时，把 GoldenRunner 的输出（`--show 10`）发回来排查，**不要改算法类去凑测试**。
- 私服缺少 jsoup 1.21.2 时，先申请引入。不要换成别的版本凑合（见第 6 节）。

---

## 3. 接入方式（二选一，推荐 A）

**A. 作为依赖引入（推荐）**
- 把 `structure-chunker-1.0.0.jar` 发布到内网私服，生产项目加依赖。
- 算法代码与生产代码物理隔离，以后规则更新只需替换 jar 再跑一遍对照。

**B. 源码拷入生产仓库**
- 把 `com.chunk.structure` 下的 15 个类整体拷入，可以改包名，但**只允许改 package 声明和 import**。
- 同时拷入 `GoldenParityTest`、`GoldenRunner` 和 `testdata/golden_cases.jsonl.gz`（放到 test resources，改测试里的默认路径）。
- 在目录里放一个 README，注明"本目录代码与 Python 离线评测逐字对齐，禁止修改，更新流程见迁移包 README 第 5 节"。

两种方式都**不要**做以下事情：

- 把算法类拆散、合并，或改造进生产已有的工具类。
- 把 `PyStr.len / isSpace / strip / split` 换成 `String.length() / trim() / strip() / split()`。
- 改正则，例如去掉 `UNIX_LINES`，或把 `PyStr.S`、`PyStr.D` 换回 `\s`、`\d`。
- 去掉 `Soup` 里"补 doctype 再删除"的解析步骤，或换用别的 html 解析器。
- 修改 `ChunkConfig` 的默认值。
- 使用 Java 9 及以上的 API（`String.strip`、`List.of`、`var` 等）。目标是 Java 8。

---

## 4. 写适配层（生产侧真正要写的代码）

在生产代码里新增一个适配类，实现生产现有的切分接口，内部调用 `StructureChunker`：

```java
public class StructureChunkAdapter implements <生产切分接口> {
    // 线程安全，全局共享一个实例即可（Spark 中可放在静态字段或广播）
    private static final StructureChunker CHUNKER = new StructureChunker();
    private final ChunkMethod method;

    public StructureChunkAdapter(String methodName) {
        this.method = ChunkMethod.of(methodName);   // 例如 "md_struct_secmerge_t512"
    }

    @Override
    public List<<生产Chunk>> split(<生产文档> doc) {
        String md = method.input() == ChunkMethod.Input.MARKDOWN ? doc.<markdown字段>() : null;
        String html = method.input() == ChunkMethod.Input.HTML ? doc.<html字段>() : null;
        List<Chunk> chunks;
        try {
            chunks = CHUNKER.chunkWithKind(method, md, html);
        } catch (RuntimeException e) {
            // 记录 url + 异常，按第 4.2 条的兜底策略处理
        }
        // 转成生产 chunk：text 原样使用，不做 trim / 规范化；kind 可选写入
    }
}
```

### 4.1 方法注册

- 在生产的方法枚举或配置里新增 5 个方法名，**名字与 Python 保持一致**：`md_struct_secmerge_t512`、`html_struct_secmerge_t512`、`md_struct_secmerge_t512_adh`、`html_struct_secmerge_t512_adh`、`html_container`。这样线上日志、评测结果、文档之间可以直接对应。
- `content_br` 保持不变，仍然是默认方法。

### 4.2 需要找我（需求方）确认的决策点

agent 不要自行决定以下几点，应列出选项交给需求方：

1. **输入缺失或切分结果为空时的兜底**：html/markdown 字段为空、解析异常、或结果为空列表时，是返回空，还是回退到 `content_br`？离线评测的口径以评测代码为准。
2. **生产需要、本包不提供的字段**（如在原文中的偏移、所属标题路径）：是不填，还是另外计算？另外计算时不能改动 chunk 文本本身。
3. **超长页面**：是否需要在切分前限制输入长度。离线评测没有限制，加了限制就与评测口径不一致。

### 4.3 切分后的处理

- chunk 文本**原样**交给下游，不要额外做 trim、合并空白、去换行、拼标题前缀，除非 `content_br` 也做了完全相同的处理。
- 下游的上下文 LLM chunk、向量化截断等环节，新旧方法走同一条路径。

---

## 5. 测试与验收

| 层级 | 内容 | 通过标准 |
|---|---|---|
| 库对照 | `GoldenParityTest` 在生产构建中运行 | 全部一致（已知差异除外） |
| 适配层单测 | 每个方法各一个小样例；md 方法只读 markdown、html 方法只读 html；null/空输入；异常兜底 | 全部通过 |
| **端到端真实数据对照** | 服务器上用离线评测的 Python 环境运行 `python/make_golden.py --input <真实分片> --limit 5000`，再**经过生产适配层**（而不只是直接调用库）跑出结果，与 golden 逐页比较 | md 两种方法 100%；html 三种方法的不一致全部属于差异文档第 3 节的类型 |
| 统计对比 | 同一批页面上，每种方法的块数分布、平均块长、空结果率、超 1024 字的块数 | 与离线评测的统计接近；`html_container` 不应出现超过 1024 字的块 |
| 性能 | 单页耗时 P50 / P99、最大页耗时 | 参考值：本机 0.36–0.51 ms/页 |

"端到端经过适配层"这一项最重要，它能发现字段取错、输入被预处理、文本被下游改写等接入问题，单测发现不了这些问题。

---

## 6. 上线

1. 新方法默认关闭，通过配置开启。
2. 灰度或旁路：同一批文档同时用 `content_br` 和新方法切分，只写入影子索引，对比统计和召回效果。
3. 监控：异常数、兜底次数、空结果率、块数和块长分布。
4. 先上线已有离线评测结果的方法（`md_struct_secmerge_t512`、`html_struct_secmerge_t512`、`html_container`）。`_adh` 两种方法没有离线评测结果，补完评测再开启。

---

## 7. 以后改规则或升级依赖

- **改规则**：先改 Python 原版并通过离线评测，再按 `README.md` 第 5 节同步到 Java，重新生成 golden 并对照通过，最后替换生产里的 jar 或源码。**不允许只改 Java。**
- **升级 jsoup**：升级后必须重新跑第 5 节的库对照和真实数据对照。生产被迫使用其他 jsoup 版本时（例如依赖冲突），优先用 maven-shade 对本包的 jsoup 做 relocate；无法 relocate 时，用目标版本重新跑对照，并把结果报告给需求方。

---

## 8. 交付清单（agent 完成后提交）

- [ ] 调研结论（第 1 节的表格，标出输入一致性结论）
- [ ] 内网 `mvn test` 输出截图或日志
- [ ] 适配层代码、方法注册、配置开关的 diff
- [ ] 第 4.2 节决策点的提问及需求方的答复
- [ ] 适配层单测
- [ ] 端到端真实数据对照结果（5 种方法的一致率 + 不一致样例分类）
- [ ] 统计对比和性能数据
- [ ] 确认算法类未改动（方式 A：jar 版本；方式 B：除 package/import 外与迁移包 diff 为空）

---

## 附：可直接发给 agent 的任务提示词

```
你要把 <迁移包路径>/切分方法_Java迁移 中的结构切分库接入我们的生产切分服务，新增 5 种切分方法：
md_struct_secmerge_t512、html_struct_secmerge_t512、md_struct_secmerge_t512_adh、html_struct_secmerge_t512_adh、html_container。

严格按 <迁移包路径>/切分方法_Java迁移/生产接入_codeagent指南.md 执行：
1. 先读 README.md、Python与Java差异对照.md 和指南，再只读调研生产代码，产出指南第 1 节的调研结论，交给我确认后再动手。
2. 算法类（com.chunk.structure 下的所有类）一律不改，只写适配层、方法注册和配置开关。
3. 指南第 4.2 节的决策点列出选项问我，不要自行决定。
4. 每完成一步，按指南第 5 节给出验证结果。不一致时报告样例，不要修改算法去凑结果。
5. 最后按指南第 8 节的清单交付。
```
