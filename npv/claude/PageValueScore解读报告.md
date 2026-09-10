# PageValueScore 网页权威度打分程序解读报告

- 分析对象：`PageValueScore.java`（打分模型）、`PageValueScoreMain.java`（Spark 作业入口）
- 代码来源：两份扫描 PDF 经 OCR 还原（见项目根目录 `还原说明.md`）。本报告中涉及运行时行为的关键判断，已回到 PDF 原图逐行核对，未发现属于 OCR 误差的情况。
- 分析日期：2026-09-05
- 分析方式：静态阅读与推演。本机无 Java 运行时与依赖，未做编译或运行验证；结论中凡标注"需核实"的条目，建议在原项目环境中确认。

---

## 1. 程序概览

这是一个跑在 Spark 上的离线批处理作业，为每个 URL 计算一个"网页价值/权威度"分数（npv，推测为 Net Page Value 之类的缩写），输出两类结果：

| 输出目录 | 内容 | 列 |
| --- | --- | --- |
| `{output}/npv_ori` | 原始分数 | `url, npv_ori, npv_fea` |
| `{output}/npv`（仅 `isScroll=1`） | 经排名归一化后的 1~1000 等级分 | `url, npv_ori, npv, npv_fea` |

`npv_fea` 是各特征预处理后的 JSON，便于离线分析。

### 1.1 入参（12 个位置参数，无校验）

| 序号 | 参数 | 说明 |
| --- | --- | --- |
| 0 | input | 输入 TSV。第 0 列 url，第 2 列一个页面类别标记（取值 1/2/5 被视为"高质量网页"），第 5 列是 JSON，含 `pr, adc, pcLong, pct, pt, pureTextLen, sr, spr` |
| 1 | sprPath | 站点 spr 值表（仅用于求全局最大值 `sprMax`） |
| 2 | drSitePath | 站点级 dr 表（site → dr） |
| 3 | drSuffixPath | 域名后缀级 dr 表（suffix → dr） |
| 4 | owPath | 官网（official website，推测）表（site → 值） |
| 5 | prSplitPath | pr 分位数切分点 |
| 6 | owBlackPath | 官网黑名单（域名后缀） |
| 7 | adcWhitePath | 权威站点白名单（子串匹配） |
| 8 | output | 输出根目录 |
| 9 | partitionToSave | 输出分区数（同时也决定了计算并行度，见 4.2） |
| 10 | regionFlag | 区域，`zh` 时使用中文站点白名单 |
| 11 | isScroll | 是否执行"高质量网页"拆分与归一化 |

### 1.2 主流程（PageValueScoreMain）

```
textFile(input) → coalesce(partitionToSave) → map: 解析 JSON, 调用 PageValueScore.score()
   → midRdd(url, npv_ori, npv_fea_json, 类别标记)
   ├─ isScroll=0：全部写入 npv_ori
   └─ isScroll=1：
        ├─ 类别 ∉ {1,2,5} → 写入 npv_ori
        └─ 类别 ∈ {1,2,5} → persist(DISK_ONLY) → 按分数降序排名
                             → 正态分布分桶映射到 1..1000 → 写入 npv
```

---

## 2. 打分模型详解（PageValueScore）

### 2.1 特征预处理

| 特征 | 输入 | 预处理 | 取值范围 | 权重 |
| --- | --- | --- | --- | --- |
| sr | 站点等级 sr（整数） | `sr / 100` | 通常 [0, 1]，缺失时为负 | 不直接加权 |
| spr | 站点 spr（浮点） | `spr / sprMax` | [0, 1] | 不直接加权 |
| spr_sr | 由 sr、spr 合成 | 见 2.2 | [0.25 或 0, ~1.136] | **60** |
| dr | 由 url 解析出 site | 先查站点表，再按后缀表 `endsWith` 匹配，都没有取 1；结果 `/ MAX_DR(3)` | [1/3, 1] | **12** |
| ow | site | 区域白名单后缀 → 1；黑名单后缀 → 0；官网表中 > 0 → 1；否则 0 | {0, 1} | **4** |
| pr | 页面 pr | 按 prSplit 分位数切分（当前实现有 bug，见 3.1） | 实际只有 {0.01, 1.0} | **4** |
| adc | adc.level 与白名单 | level > 0 直接用；否则 url 含白名单子串 → 3；否则 0 | {0,1,2,3,…} | 不加权，用于控制流与加分 |

四个加权特征权重之和为 80，其中 spr_sr 占 75%。**站点级信号（sr/spr/dr/ow）几乎决定了全部基础分，页面级信号（pr）只占 5%**，这是一个站点权威度模型而非页面模型。

### 2.2 spr_sr 合成（getSrSprScore）

```
if sr >= 0:
    spr = max(spr, 0) / sprMax           # 注意：这里第二次除以 sprMax
    spr ∈ (0.36, 1]  → 线性映射到 (0.9, 1.0]
    spr ∈ (0.2, 0.36] → 线性映射到 (0.2, 0.9]
    spr ∈ [0, 0.2]   → 不变；恰好为 0 时置 0.1
    spr_sr = (sr + 0.25 * spr) / 1.1
else:
    spr_sr = 0.25
```

设计意图是：sr 为主，spr 为辅，并对 spr 做分段放大（0.2~0.36 这段被拉伸了 4.4 倍）。两个可疑点：

1. **spr 被除以了两次 sprMax**。`preProcessSpr` 已做过 `spr / sprMax`，传入后又 `spr / sprMax`。若 sprMax > 1，则 spr 实际变成 `spr / sprMax²`，几乎永远达不到 0.2 的阈值，分段映射形同虚设，spr 对分数的贡献被压缩到 `60 × 0.25 × (1/sprMax) / 1.1` 以下。若 sprMax 恰好为 1 则无影响。需要用真实数据确认 sprMax 的量级。
2. 分母 1.1 与分子最大值 1.25 不匹配，spr_sr 最大可达 1.136，基础分上限因此变成 88.18 而不是 80。若意图是归一到 1，分母应为 1.25。

### 2.3 分数流水线（score 方法）

```
score = 60·spr_sr + 4·pr + 12·dr + 4·ow                 # 基础分，上限 ≈ 88.2
score *= f_pct(距今天数)                                  # 时间衰减一：对所有页面
if adc < 2:                                              # 非权威站点才做下列调整
    score = 按 pc 位标记降权 / 保底                        # 页面类别
    if len(pt) > 5: score *= f_pt(距今天数)                # 时间衰减二
if pureTextLen < 300: score *= 0.8
elif pureTextLen < 500: score *= 0.9
if adc == 2: score += 10
elif adc == 3: score += 15
```

最终分数理论范围约 [0, 103]，没有归一到 [0, 100]。

### 2.4 时间衰减

两处衰减都是 `2 / (e^{d/T} + 1)` 形式的 sigmoid，d 为天数，pct 用 T=2000，pt 用 T=3000：

| 距今天数 | f_pct (T=2000) | f_pt (T=3000) |
| --- | --- | --- |
| 0 | 1.000 | 1.000 |
| 365 | 0.909 | 0.939 |
| 730 | 0.820 | 0.880 |
| 1095 | 0.733 | 0.820 |
| 2000 | 0.538 | 0.660 |
| 3650 | 0.278 | 0.457 |

衰减平缓，三年前的页面仍保留 73%（非权威站点两项叠加约 60%）。

### 2.5 页面类别位标记（adjustScorePc）

`pc` 是一个 long 型位图，按 if-else 顺序取第一个命中的规则，**顺序即优先级**：

| 优先级 | 条件 | 动作 |
| --- | --- | --- |
| 0 | bit 20 | `score = max(score, 60)`（保底） |
| 1 | bit 10 | ×0.7 |
| 2 | bit 19 | ×0.8 |
| 3 | bit 13 | ×0.6 |
| 4 | bit 18 | ×0.9 |
| 5 | bit 16 | ×0.8 |
| 6 | bit 33 | ×0.7 |
| 7 | bit 24 | ×0.6 |
| 8 | bit 38 | ×0.6 |
| 9 | bit 22 | ×0.7 |
| 10 | bit 11 或 url 含 "news" | ×0.9 |
| 11 | bit 26 或 url 含 "/u/"、"/user/" | ×0.6 |
| 12 | bit 29 | ×0.9 |
| 13 | bit 32 | ×0.8 |

代码里没有任何位的语义说明，规则含义只能靠上游文档。bit 20 的"保底 60"发生在正文长度衰减之前，所以并不是真正的保底（60 × 0.8 = 48）。

### 2.6 排名归一化（generateUrlInterval + normalizationScore）

只对"高质量网页"（类别 1/2/5）执行：

1. 按 `npv_ori` 降序排序，`zipWithIndex` 取名次。
2. 把标准正态分布在 z ∈ [-2, 3] 上切成 1000 个等宽区间，每个区间按其概率质量分配 URL 数量（向上取整），得到 1001 个累计边界。
3. 名次落在第 idx 个区间 → `npv = 1000 - idx`。

效果：npv 等级近似正态分布，z=0 对应 npv≈400，是众数；[400, 1000] 这一段（z ∈ [0,3]）承载约 50% 的 URL，但越靠近 1000 越稀疏，头部区分度高；[1, 400] 段（z ∈ [-2,0]）被压缩，尾部区分度低。这是一个有意为之的不对称设计。

注意 `ONE_HUNDRED` 常量的值是 1000，名字与值不符。

---

## 3. 现状评估：问题清单

按严重程度分为三级。位置均指向还原后的 Java 文件。

### 3.1 正确性缺陷（会产生错误结果或运行失败）

**P0-1 preProcessPr 的区间判断永远为假** — `PageValueScore.java:107`

```java
if (pr > this.prSplit.get(idx) & pr <= this.prSplit.get(idx)) {   // 两边都是 idx
```

`x > a && x <= a` 不可能成立，循环体从不执行，所有 `pr >= prSplit[0]` 的页面一律返回 1.0。pr 特征退化为二值（0.01 / 1.0），prSplit 文件白加载。修正后还应注意：`idx / 100.f` 在 idx=0 时返回 0.0，反而低于"低于最小切分点"的 0.01，边界值也不自洽。

**P0-2 spr 被二次归一化** — `PageValueScore.java:86` 与 `:151`

见 2.2。两处只应保留一处。这是权重最大的特征（60 分），影响最大。

**P0-3 `sortByKey(false)` 使用 `Tuple2` 作为键，需核实是否能运行** — `PageValueScoreMain.java:140`

Java API 的 `JavaPairRDD.sortByKey(boolean)` 内部使用 Guava 的自然序比较器，要求键实现 `Comparable`；`scala.Tuple2` 并未实现该接口，按此实现会在采样分区边界时抛出 `ClassCastException`。已回看 PDF 原图确认该行无比较器参数。如果 `isScroll=1` 分支在生产环境确实运行过，请确认是否另有处理；否则应改为对 `JavaRDD` 直接 `sortBy(row -> row.getDouble(1), false, n)`。

**P0-4 硬编码的默认日期是一颗定时炸弹** — `PageValueScore.java:317`

pct/pt 缺失（`String.valueOf(0L)` 长度为 1）时，`timeStamp2Date` 返回固定的 `"2024-05-22"`，衰减按"距今天数"计算。于是缺失时间的页面会被施加一个随日历推移而不断加重的惩罚：

| 运行日期 | 缺失 pct 的页面得到的衰减系数 |
| --- | --- |
| 2024-05-22 | 1.000 |
| 2025-05-22 | 0.909 |
| 2026-09-05（今天） | 0.794 |
| 2028-05-22 | 0.538 |

同一份输入，不同日期跑出的分数不同，且与"缺失应当中性处理"的直觉相悖。pt 有 `length() > 5` 的保护，pct 没有，两者行为也不一致。

**P0-5 未来时间戳会让分数放大** — `PageValueScore.java:227, 303`

`daysBetween` 为负时 `2/(e^{-x}+1) > 1`，脏数据或时钟偏差会导致加分而非中性。应对天数做 `max(0, d)` 钳制。

**P0-6 单条记录异常会导致整个作业失败** — `PageValueScoreMain.java:94-133`

map 内的以下情况都会抛异常且未捕获：第 5 列 JSON 缺 `pureTextLen`/`sr`/`spr`（`getInteger`/`getDouble` 返回 null 后自动拆箱 NPE）、`pr` 或 `adc` 字段为 null（`isEmpty()` NPE）、`level` 非数字、行列数不足 6、JSON 非法。Spark 会重试 4 次后终止整个作业。后面已经有 `.filter(Objects::nonNull)`，说明原意是"坏行返回 null 过滤掉"，但 map 里从未返回 null。

**P0-7 `sprMax == 0` 时产生 NaN** — `PageValueScore.java:86`

除零得到 Infinity/NaN，NaN 会一路传播到输出并使排序结果不可预期。

### 3.2 逻辑疑点（可能不符合业务意图，需业务方确认）

**P1-1 中文区白名单 `.edu.com`** — `PageValueScoreMain.java:45`。中国教育网域名是 `.edu.cn`，`.edu.com` 疑为笔误。

**P1-2 后缀匹配没有域名边界** — `PageValueScore.java:120, 130, 135`。`endsWith("baike.baidu.com")` 会命中 `xbaike.baidu.com`；若后缀表键不带前导点（如 `gov.cn`），`notgov.cn` 也会命中。

**P1-3 adc 白名单用 `url.contains`** — `PageValueScore.java:95`。在整个 URL 上做子串匹配，`evil.com/?ref=gov.cn` 之类的 URL 会被判为权威站点（adc=3，跳过页面类别降权并加 15 分）。应改为在 site 上做边界匹配。`site` 变量在此方法中计算了但没有使用，说明原本就打算按站点匹配。

**P1-4 "news" 子串规则过宽** — `PageValueScore.java:252`。`newsletter`、`hknews`、站点名含 news 的均会被降权 0.9。

**P1-5 保底与衰减的顺序** — 见 2.5，bit 20 的保底会被后续正文长度系数打穿。

**P1-6 分数量纲不统一** — 基础分上限 88.2、adc 加分 15、保底 60，三者不在同一尺度上；最终范围 [0, 103]。下游若假设 0~100 会出问题。

**P1-7 pr 分位切分的映射假设** — `idx / 100.f` 隐含 prSplit 恰好有 100 个切分点；文件行数不同时映射范围会变。

**P1-8 排序结果的稳定性** — `zipWithIndex` 对相同分数的 URL 赋予不同名次，并列顺序取决于分区物理顺序，两次运行同一输入 npv 可能不同。

**P1-9 小样本下归一化失真** — 1000 个桶各自向上取整，累计最多多分配 1000 个名额。若高质量网页总数只有数千，头部桶几乎每桶 1 个 URL，正态形状被破坏；数亿规模时可忽略。

**P1-10 时间基准在 executor 上逐条取 `System.currentTimeMillis()`** — 跨零点运行的作业前后两批数据的"今天"不同；也无法离线复现。

### 3.3 工程质量与性能

**P2-1 midRdd 被计算两次** — `PageValueScoreMain.java:56-77`。`isScroll=1` 时，写 `npv_ori` 触发一次完整打分，随后 `hqwRdd.persist` + `count()` 再触发一次，打分逻辑（含 JSON 解析）重复执行；`dataCount` 累加器也因此翻倍。应在 `midRdd` 上 persist。

**P2-2 `coalesce(partitionToSave)` 放在 map 之前** — `PageValueScoreMain.java:94`。coalesce 是窄依赖，会把打分阶段的并行度压到 `min(输入分区数, partitionToSave)`。若 partitionToSave 是为了控制输出文件数而设得较小，整个计算阶段就被拖慢。应在写出前再 coalesce，或使用 `repartition`。

**P2-3 排序键携带了完整载荷** — `PageValueScoreMain.java:138-141`。键是 `(score, url + "\t" + feaJson)`，shuffle 时整个 JSON 跟着键走；并用 `split("\t")` 重复拆四次。改用 `sortBy` + 保留 Row 即可。

**P2-4 每条 URL 线性扫描后缀表与 1000 个区间** — `PageValueScore.java:119`、`PageValueScoreMain.java:144`。后缀表若有数千项，每 URL 数千次 `endsWith`；区间查找应改为二分（`Collections.binarySearch`）。后缀匹配可改为逐级剥离域名标签后做 map 查找，复杂度从 O(表大小) 降到 O(标签数)。

**P2-5 `parseSite(url)` 每条被调用 3 次**（dr、ow、adc）。

**P2-6 对象状态作为返回值** — `PageValueScore.featureList` 是实例字段，`score()` 先清空再写入，Main 调用完 `score()` 后再读取它拼 JSON。当前 Spark 会为每个 task 反序列化独立副本，所以没有真正的并发问题，但这是隐式耦合：不可重入、不可并行测试、`HashMap` 导致 JSON 键序不稳定（用于 diff 或下游解析都不友好）。

**P2-7 黑白名单与 prSplit 未广播** — 作为实例字段随闭包序列化，每个 task 都要传一份，列表大时开销明显；与 dr/ow 表用 Broadcast 的做法不一致。

**P2-8 基于异常的控制流** — `PageValueScoreMain.java:101-117`，用 try/catch NPE 处理字段缺失，缺失比例高时开销可观；`timeStamp2Date` 里 `format → parse → format` 的往返也是无意义的。

**P2-9 死代码与瑕疵** — `srBC`/`sprBC` 字段永不赋值；`preProcessAdc` 内 `site` 未用、`{;` 多余分号；`Float.parseFloat(String.valueOf(sr))` 应为 `(float) sr`；`getBasicScore(score)` 的入参恒为 0；`ONE_HUNDRED = 1000`；`MAX_DR` 是 public 非 final 的 `Integer`；`FEA_WEIGHT` public；信息性日志用 `logger.error`；`dataCount` 累加器从未读出；`loadMapAndBcPathString` 用原始类型 `Map` 且每行 `split` 两次、少于两列直接越界。

**P2-10 配置全部硬编码** — 权重、阈值（0.36/0.2/0.25/1.1）、衰减常数（2000/3000）、正文长度阈值、位标记与系数、区域白名单、类别标记 {1,2,5} 全在代码里，调参必须改代码重新打包。

**P2-11 没有测试** — 打分逻辑本质上是给定几张表的纯函数，非常适合单元测试，但 Spark 上下文被直接注入构造函数，测试时无法脱离集群。

**P2-12 参数与依赖** — 12 个位置参数无校验、无帮助信息；引入 HBase 的 `Pair` 仅为了返回两个对象。

---

## 4. 改进建议

### 4.1 立即修复（不改变模型设计，只纠正明显错误）

1. `preProcessPr`：改为 `pr > prSplit[idx] && pr <= prSplit[idx+1]`，返回值按 `(idx + 1) / prSplit.size()` 之类与切分点数量无关的方式计算，并让"低于最小值"的返回值小于第一档。
2. `getSrSprScore`：去掉方法内的 `spr / sprMax`（或去掉 `preProcessSpr` 里的），用真实 sprMax 回归比对分数分布变化。
3. `normalizationScore`：改为 `rdd.sortBy(r -> r.getDouble(1), false, numPartitions).zipWithIndex()`，避免 `Tuple2` 作键。
4. 时间处理：缺失 pct/pt 时衰减系数取 1（或一个固定常数），删除 `"2024-05-22"`；天数钳制 `max(0, d)`；把"当前时间"作为作业参数传入（默认取 driver 启动时刻），保证同一次作业内基准一致、可复现。
5. `getScoreRDD` 的 map 体整体 try/catch，失败行返回 null 并计数（用一个 `badRows` 累加器），让已有的 `filter(Objects::nonNull)` 真正起作用；作业结束时把 `dataCount`/`badRows` 打到日志。
6. `sprMax <= 0` 时抛出明确异常终止作业，而不是静默产出 NaN。
7. 与业务方确认 `.edu.com`。

### 4.2 短期重构（一到两个迭代内）

1. **数据流**：`midRdd.persist(DISK_ONLY)` 一次；`coalesce` 移到 `write()` 之前；两处 filter 共用同一份缓存。
2. **匹配逻辑**：写一个 `matchSuffix(site, map)`，从 site 逐级剥离标签（`a.b.c.com → b.c.com → c.com → com`）做 map 查找，同时解决边界问题与 O(n) 扫描；黑白名单、dr 后缀表、区域白名单统一走这一个函数；adc 白名单改为按 site 匹配。
3. **纯函数化**：`score()` 返回一个 `ScoreResult {double score; Map<String, Float> features}`（`LinkedHashMap` 保证键序），去掉 `featureList` 字段；`parseSite` 只算一次并透传。
4. **可测试性**：把"从 Spark 加载表"与"打分"拆成两个类。打分类的构造函数只接收几张 `Map`/`List`，Spark 加载放到工厂方法里。这样 `preProcessPr`、`getSrSprScore`、`adjustScorePc`、`generateUrlInterval` 都能写 JUnit 表格化测试，也方便用一批标注 URL 做回归。
5. **位标记**：为 pc 的每一位定义常量或枚举并写明含义，规则表用有序列表表达，让优先级显式化。
6. **配置外置**：权重、阈值、衰减常数、正文长度阈值、区域白名单、类别标记放到一个配置文件（properties/JSON），作业参数改为 `--key value` 形式并校验。
7. **区间查找**：`urlIntervals` 用二分查找；对小样本（比如 n < 10 万）改用按比例的浮点边界而不是逐桶向上取整。
8. **清理**：删除死字段、修复瑕疵、日志级别改为 info、常量改名、`Pair` 换成自定义小类或 Spark 自带的 `Tuple2`。

### 4.3 中期演进（模型层面）

1. **量纲统一**：把基础分归一到 [0, 1] 或 [0, 100]（修正 1.1 → 1.25），adc 加分、保底改为同一尺度上的相对值，明确最终分数范围并在输出中钳制。
2. **调整顺序**：把"保底"放到所有乘法调整之后，或明确说明保底只对页面类别有效。
3. **页面级信号偏弱**：当前页面级只有 pr（4 分）与正文长度、时间。代码注释里列出的四项待办（资源类站点降权、索引页/列表页识别、新闻页识别、论坛页降权）都是页面级信号，可以纳入同一个"页面类型 → 系数"的规则表，而不是继续在 if-else 链上叠加。
4. **权重数据驱动**：有了 `npv_fea` 输出，就可以用一批人工标注的权威度样本拟合权重（哪怕是简单的逻辑回归），替代目前的手工 60/12/4/4。
5. **并列名次**：归一化前对相同分数按 url 做次级排序，保证结果确定性。
6. **可观测性**：每次作业输出特征分布、各调整规则的命中率、缺失时间戳比例、NaN/坏行数量，作为模型健康度指标；这些数据也是后续调参的基础。

---

## 5. 需在原环境核实的事项

| 事项 | 影响 |
| --- | --- |
| `sprMax` 的实际量级 | 决定 P0-2 的实际影响大小 |
| `isScroll=1` 分支是否在生产成功运行过 | 决定 P0-3 是否已被其他方式规避 |
| `parseSite` 是否已做小写化 | 决定 P1-2 中大小写不一致是否真实存在 |
| 输入第 2 列（类别 1/2/5）与 pc 各位的语义 | 编写常量与文档所需 |
| 上游 sr/spr 缺失时的取值约定（是否为 -1） | `sr >= 0` 分支与 `spr < 0` 钳制的正确性 |
| prSplit 文件的行数与排序 | P0-1 修复后的映射方式 |

---

## 附录 A：参考实现片段

pr 分位映射修正：

```java
float preProcessPr(double pr) {
    int n = prSplit.size();
    if (pr <= prSplit.get(0)) return 0.0f;
    for (int idx = 0; idx < n - 1; idx++) {
        if (pr > prSplit.get(idx) && pr <= prSplit.get(idx + 1)) {
            return (idx + 1) / (float) n;
        }
    }
    return 1.0f;
}
```

域名后缀匹配（同时解决边界与性能）：

```java
static Optional<String> matchSuffix(String site, Map<String, String> table) {
    String s = site.toLowerCase();
    while (true) {
        String v = table.get(s);
        if (v != null) return Optional.of(v);
        int dot = s.indexOf('.');
        if (dot < 0) return Optional.empty();
        s = s.substring(dot + 1);
    }
}
```

排序归一化：

```java
JavaPairRDD<Row, Long> ranked = rdd
        .sortBy(r -> r.getDouble(1), false, rdd.getNumPartitions())
        .zipWithIndex();
// 区间查找用 Collections.binarySearch(urlIntervals, rank) 取桶号
```

时间衰减（缺失中性、负值钳制、基准注入）：

```java
static double decay(long tsSeconds, long nowSeconds, double scaleDays) {
    if (tsSeconds <= 0) return 1.0;                 // 缺失视为中性
    long days = Math.max(0, (nowSeconds - tsSeconds) / 86400);
    return 2.0 / (Math.exp(days / scaleDays) + 1.0);
}
```
