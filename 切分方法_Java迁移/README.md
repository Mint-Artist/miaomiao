# 结构切分方法 Java 迁移包

把离线评测用的 5 种网页结构切分方法从 Python（notebook）迁移到 Java 8，结果与 Python 原版**逐字一致**（残缺 html 上有少量已知差异，见差异文档）。

| 方法 | 输入 | 离线评测 |
|---|---|---|
| `md_struct_secmerge_t512` | markdown | 已评测，hit@3000字 最好 |
| `html_struct_secmerge_t512` | html | 已评测 |
| `md_struct_secmerge_t512_adh` | markdown | **未评测**（标题吸附版，上线前建议补一轮评测） |
| `html_struct_secmerge_t512_adh` | html | **未评测** |
| `html_container` | html | 已评测 |

## 1. 目录

```
切分方法_Java迁移/
├── README.md                     本文件
├── 切分方法说明.md                 实现规格：通用约定、解析规则、5 种方法的算法、参数、边界情况、示例
├── Python与Java差异对照.md          语言 / 库差异及处理方式；无法消除的已知差异；对照结果
├── 生产接入_codeagent指南.md        给内网 code agent 的接入步骤、禁改清单、验收标准、任务提示词
├── python/
│   ├── original/                  notebook 原版（recall_pipeline_v2.py、chunk_adhere.py、chunk_container.py），只作溯源
│   ├── chunk_ref/                 独立参考实现（由 build_ref.py 从原版逐字抽取，不依赖 notebook）
│   ├── build_ref.py               重新生成 chunk_ref
│   ├── check_ref_vs_original.py   验证 chunk_ref 与原版逐字一致
│   ├── fuzz_pages.py              随机测试页生成器（固定种子）
│   ├── make_golden.py             生成标准答案（golden），供 Java 对照
│   ├── sample_real.py             服务器上抽样真实网页（原始数据 + 切分结果），输出同 golden 格式（命令行版）
│   ├── sample_real_jupyter.py     同上，Jupyter 版：第 1 段改配置，按段运行
│   └── sample_offline_jupyter.py  Jupyter 版：直接取离线评测落盘的切分结果做标准答案（不依赖 chunk_ref）
├── java/                          Maven 工程（Java 8，唯一运行时依赖 jsoup 1.21.2）
│   ├── pom.xml
│   └── src/main/java/com/chunk/structure/   实现
│       src/test/java/com/chunk/structure/   GoldenParityTest（JUnit）、GoldenRunner（命令行对照）
└── testdata/
    ├── cases/                     构造用例 42 个（每条规则、每个边界情况；含 EC 需求文档里的华为样例）
    ├── golden_cases.jsonl.gz      构造用例 + 随机 html 300 页 + 随机 markdown 300 页的标准答案
    └── known_diffs.txt            已知差异页面清单
```

## 2. Java 用法

```xml
<dependency>
  <groupId>com.chunk</groupId>
  <artifactId>structure-chunker</artifactId>
  <version>1.0.0</version>
</dependency>
```

```java
StructureChunker chunker = new StructureChunker();          // 默认参数 = Python 原版
List<String> a = chunker.chunk(ChunkMethod.MD_STRUCT_SECMERGE_T512, markdown, null);
List<String> b = chunker.chunk(ChunkMethod.HTML_CONTAINER, null, html);
List<Chunk>  c = chunker.chunkWithKind(ChunkMethod.HTML_CONTAINER, null, html);   // 带块类型：para/list/table/note/...
ChunkMethod m = ChunkMethod.of("html_struct_secmerge_t512");                       // 按 Python 方法名查找
```

- md_* 方法只用 markdown 参数，html_* 方法只用 html 参数；对应输入为 null / 空串时返回空列表。
- **线程安全**：实例不保存切分状态，可以在多线程 / Spark executor 中共用一个实例。
- 调参：`new StructureChunker(cfg)`，`cfg` 是 `ChunkConfig`（字段与 Python 参数一一对应，见规格文档第 1 节）。构造后不要再改 `cfg`。**改了参数就和离线评测不一致了**，除非同时改 Python 并重新评测。
- 性能（本机单线程，页面平均 8.6K 字符）：每页 0.36–0.51 ms。

主要类：

| 类 | 作用 | 对应 Python |
|---|---|---|
| `StructureChunker` | 对外入口 | `chunk_ref.chunk` |
| `ChunkMethod` / `Chunk` / `ChunkConfig` | 方法枚举 / 结果 / 参数 | 方法名 / — / 各参数 |
| `MarkdownParser` | markdown → 结构单元 | `parse_markdown` |
| `HtmlAtomParser` | html → 扁平结构单元；表格行提取 | `parse_html`、`table_rows_from_soup` |
| `HtmlBlockParser` | html → 容器单元 | `parse_html_blocks`、`block_text`、`is_layout_table` |
| `SectionChunker` | 小节打包（含吸附开关）+ 小节合并 | `chunk_struct_sections_adh`、`merge_sections` |
| `ContainerChunker` | 容器装块 + 块合并 | `chunk_container_recs` |
| `TitleDetector` | 类标题行识别 | `num_style`、`title_like`、`_tl_info` |
| `TextUtils` / `PyStr` / `Soup` | 文本工具 / Python 字符串语义 / BeautifulSoup 行为复刻 | `clean_inline` 等 |

## 3. 构建与测试

```bash
cd java
mvn test          # 编译 + 与 testdata/golden_cases.jsonl.gz 逐字对照（GoldenParityTest）
mvn package       # 打包 target/structure-chunker-1.0.0.jar
```

不用 Maven 也可以直接对照（只需 jsoup 和 gson 两个 jar）：

```bash
javac -encoding UTF-8 --release 8 -cp jsoup-1.21.2.jar:gson-2.10.1.jar -d out \
  src/main/java/com/chunk/structure/*.java src/test/java/com/chunk/structure/GoldenRunner.java
java -cp out:jsoup-1.21.2.jar:gson-2.10.1.jar com.chunk.structure.GoldenRunner \
  ../testdata/golden_cases.jsonl.gz --known ../testdata/known_diffs.txt --show 5
```

输出每个方法的一致率和不一致样例（第一个不同的块）；全部一致时退出码为 0。

本地已验证：JDK 25 以 `--release 8` 编译通过；GoldenRunner 对 `golden_cases` 全部一致。JUnit 测试（`mvn test`）本地没有跑（本机没有 Maven / JUnit），请在你们的构建环境里跑一次。

## 4. 在服务器上用真实网页对照（上线前必做）

用与**离线评测相同的 Python 环境**（同一 lxml 版本）：

```bash
cd python
pip install beautifulsoup4 lxml          # 若环境里还没有
# 输入：jsonl / jsonl.gz，每行至少有 markdown、html 字段（url 可选），例如离线评测用的原始分片
python make_golden.py --input /path/to/part-00000.jsonl.gz --limit 5000 --out golden_real.jsonl.gz
# 或者：从三批数据里随机抽约 1 万页，同时带上离线评测落盘的切分结果并核对（推荐，用法见 sample_real.py 开头）
python sample_real.py --source "=/path/to/random/part-*" --source "inj-=/path/to/inj/part-*" \
  --source "ggl-=/path/to/google_pages_parts/part-*" --out-root /path/to/output/chunk_eval --n 10000 --out golden_real.jsonl.gz

cd ../java
mvn test -Dgolden.path=../python/golden_real.jsonl.gz -Dgolden.known=none
# 或用 GoldenRunner 看完整的不一致样例：
java -cp target/classes:target/test-classes:<jsoup.jar>:<gson.jar> com.chunk.structure.GoldenRunner \
  ../python/golden_real.jsonl.gz --show 20
```

判断标准：
- markdown 两种方法应 100% 一致。
- html 三种方法的不一致应全部属于《Python与Java差异对照.md》第 3 节的已知类型（残缺 html 的修复差异）。如有其他类型，把 GoldenRunner 的样例输出发回来排查。
- 把一致率填进差异文档第 5 节。

## 5. 改规则时的流程

1. 改 notebook 原版（`召回评测_2026-11-02/` 下的 Python 文件），并在离线评测里验证。
2. 把改后的文件复制到 `python/original/`，运行 `python build_ref.py` 重新生成参考实现。
3. `python check_ref_vs_original.py`：确认参考实现与原版一致。
4. 同步修改 Java 和 `切分方法说明.md`。
5. `python make_golden.py --cases ../testdata/cases --fuzz-html 300 --fuzz-md 300 --out ../testdata/golden_cases.jsonl.gz`，再跑 Java 对照，直到全部一致。
