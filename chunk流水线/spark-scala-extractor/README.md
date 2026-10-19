# Spark Scala HTML 简版提取器

输入完整 HTML 字符串，输出保留语义结构的简版 HTML 字符串。
**服务器工件面向 Scala 2.11 + Spark 2.4.x（JVM 1.8）**；本地开发测试仍可用
Scala CLI 2.12 + Spark 3.5 跑同一份源码。无 Python 运行时依赖。

## 架构

```text
完整 HTML
  → URL 命中 profiles.json？ → 是：用站点选择器定位（重点站点保障/线上修复通道）
                            → 否：通用正文定位（文本密度、链接密度、包装块下钻）
  → 标签/属性白名单、噪声剔除、表格重建、空白整理
  → 自定义序列化
  → simplified_html + status/confidence/profile/content_root/report_json/error
```

- `profiles.json` 是**覆盖层**，不是千万级网页的主算法；新增/修复站点只改 JSON，不用改 jar。
- 未配置站点走 `ContentLocator.scala` 的通用定位；`status=degraded` 表示只能退回 body，需抽检。
- 每行独立处理；单页失败只写 `error`，不会让 Spark 作业失败。

## 文件

| 文件 | 行数 | 作用 |
|---|---:|---|
| `SimplifyHtml.scala` | 619 | 归一化、表格重建、序列化、报告 |
| `ContentLocator.scala` | 170 | 通用正文定位与置信度 |
| `SimplifyHtmlJob.scala` | 98 | Spark 表输入/输出与行级异常隔离 |
| `Profiles.scala` + `profiles.json` | 51 + 配置 | 站点覆盖层 |
| `SimplifyFile.scala` | 40 | 单文件调试入口 |
| `SimplifyHtml.test.scala` | 94 | 黄金测试、通用路径与空内容回归断言 |
| `pom.xml` | — | Scala 2.11.12 + Spark 2.4.8 服务器构建 |
| `batch_audit.py` | — | 批次指标汇总、失败/降级/抽检清单 |
| `samples/`、`expected/` | — | 三样本原始 HTML 与 Python 0.1.2 基准输出 |
| `output/generic/` | — | 三样本强制通用路径的实际输出与报告 |
| `output/batch_235_20261019/` | — | 235 个真实网页的批处理产物与复核报告 |

## 已验证结果

### 配置路径：与 Python 版逐字节一致

华为支持页、高考一分一段页、律图法律页均通过：

```bash
scala-cli test .
```

测试还覆盖：通用路径核心内容、表格 thead/tbody 重建、合并单元格、格内 `<br>`、
锚点与伪标题开关、空页/纯脚本页明确失败，以及正文被噪声规则清空后的
空容器修剪回归（防止死循环）。

### Spark 本地模式端到端冒烟

使用 JSONL（`doc_id/url/pg`）跑 `local[2]`：3 个配置样本经完整 Spark 作业后
仍与基准**逐字节一致**；一个通用简单页成功；一个空 HTML 行单列 `failed`。
合计 4 行 `ok`、1 行 `failed`。

### 235 个真实网页批量验证

输入 `开源对照_2026-09-28/data/raw/` 的 235 个 HTML（127 个 host），
Spark `local[4]` 全量运行：**208 行 ok、7 行 degraded、20 行 failed**。
运行中还修复了两个真实问题：空内容页死循环（0.2.1）、body 类名含 `ad`
导致正常网易正文被误删（0.2.2）。最终 20 个失败已分类为空文件、JS 外壳、
文章删除/404、反爬验证和无静态正文；逐页复核未发现“静态正文正常但仍失败”的条目。
抽样另发现 Q&A 投票/分页 UI、新闻站 APP 导流等模板噪声；`status=ok`
不等于正文准确率。完整清单与复核见
`output/batch_235_20261019/REPORT.md` 和 `REVIEW.md`。

### 通用路径与配置路径的真实差距

对三个样本强制 `--generic`，人工比对结果：

| 样本 | 通用路径结果 |
|---|---|
| 华为页 | 去空白后正文与配置路径一致 |
| 高考页 | 正文和表格保留；缺“作者/发布时间”元信息 |
| 律图页 | 正文保留；会残留“免责声明/网站地图”样板文字 |

这正是 profiles 覆盖层的价值。**通用路径的提取质量不能按配置路径的质量承诺。**

## 本地运行与测试

依赖：`scala-cli`（自动下载 JDK 17 与依赖）。

```bash
# 全部测试
scala-cli test .

# 单文件：配置路径
scala-cli run . --main-class extractor.SimplifyFile -- \
  samples/huawei_zh-cn16079039.raw.html /tmp/huawei.simplified.html \
  --url 'https://consumer.huawei.com/cn/support/content/zh-cn16079039/' \
  --report /tmp/huawei.report.json

# 强制通用路径
scala-cli run . --main-class extractor.SimplifyFile -- \
  samples/lvtu_8655766.raw.html /tmp/lvtu.generic.html \
  --url 'https://www.64365.com/zs/8655766.aspx' --generic
```

## 打包

### 服务器工件：Scala 2.11 + Spark 2.4.x

```bash
mvn -DskipTests package
cp target/spark-scala-extractor-scala211.jar spark-scala-extractor.jar
```

`pom.xml` 使用 Scala 2.11.12、Spark 2.4.8（provided）、JVM 1.8 目标；
Shade 会把 Jsoup 与 org.json 重定位到 `extractor.internal.*`，避免集群依赖冲突。
当前已验证：

- `spark-scala-extractor.jar` 主类字节码 major version = 52（JVM 1.8）；
- 用该 jar 对三个样本分别运行，输出与 Python 基准**逐字节一致**。

### 本地开发：Scala CLI 2.12

Scala CLI 1.18 已不再支持 Scala 2.11；`project.scala` 仅用于本地开发和测试，
不参与服务器工件构建：

```bash
scala-cli test .
```

## 集群提交

输入 JSONL 每行一个对象；`pg` 为完整 HTML 字符串，`url` 用于站点匹配和链接绝对化：

```json
{"doc_id":"page-1","url":"https://example.com/a","pg":"<html><h1>标题</h1><main><p>正文……</p></main></html>"}
```

服务器需为 **Spark 2.4.x / Scala 2.11 运行时**；不要提交到 Spark 3.x 集群：

```bash
spark-submit \
  --class extractor.SimplifyHtmlJob \
  spark-scala-extractor.jar \
  --input hdfs:///raw/pages.jsonl \
  --output hdfs:///simplified_pages/ \
  --input-format json \
  --output-format parquet \
  --html-col pg \
  --url-col url \
  --profiles /local/path/on/driver/profiles.json \
  --mode overwrite
```

`--profiles` 只需在 **driver 本地**可读；读取后配置会广播到 executor。
Parquet/ORC 等表输入改 `--input-format` 即可。默认丢弃原 HTML 列，
需要保留时加 `--keep-original`。

输出列：原有列（默认去掉 `pg`）+

| 列 | 含义 |
|---|---|
| `simplified_html` | 简版 HTML 字符串；失败时为 null |
| `report_json` | 提取器版本、标签统计、表格统计、告警等完整报告 |
| `status` | `ok` / `degraded` / `failed` |
| `confidence` | 通用定位置信度（配置路径固定为 1.0；仅作抽检排序，不是校准概率） |
| `profile` | 命中的站点配置或 `generic` |
| `content_root` | 正文容器选择器或通用定位的 DOM 路径 |
| `error` | 单行失败原因；成功时为 null |

其他参数：`--partitions N` 重分区，`--max-input-mb N` 超大输入护栏（默认 20），
`--generic` 强制通用定位，`--anchors` 添加 `data-src` 元素锚点。

## 已知限制

- 通用正文识别目前只在三个样本和一个简单合成页上验证；千万级上线前必须按
  host/模板/页面类型分层抽检，先评估成功率、空输出率、降级率和正文保真度。
- `confidence` 是规则分数，用于把低置信页面排到抽检前列，不是训练或校准过的概率。
- 无浏览器渲染；需要 JS 执行的页面会明确失败或降级，不应当作提取成功。
- rowspan/colspan 属性会保留，但复杂跨行跨列表格的语义还原尚无真实样本验收。
- 图片只保留 `src/alt`，不做 OCR；缺 alt 会进入 `report_json.warnings`。
- 当前 `max-input-mb` 按 JVM 字符数近似控制，不是严格的 UTF-8 字节数。
