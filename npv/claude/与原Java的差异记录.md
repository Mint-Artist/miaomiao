# 与原 Java 的差异记录

用途：当 Python 版本的分数与线上 Spark 作业对不上时，按本文逐项排查。日期：2026-09-10。

> **2026-09-11 更新：** 离线端不再生产 pr 特征，fixed Java 与三个 Python 版本已一并删除全部 pr 逻辑（`prSplit` 表与 `--pr-split` 参数、`preProcessPr`、特征 `pr` 及其 4 分权重、Main 里对 JSON `pr` 字段的读取）。fixed Java 的位置参数因此前移一位（原 args[6] 到 args[12] 变为 args[5] 到 args[11]）。若线上 Java 尚未删除而输入 JSON 已没有 `pr` 键，线上会在 `pr.isEmpty()` 处空指针整体失败。下文关于 P0-1 与第二部分第 9 项之前对 pr 的描述仅作历史记录。

四个版本的关系：

```
原 Java（PageValueScore.java / PageValueScoreMain.java，OCR 还原）
 ├─ claude/npv_exact/      Python 逐行复刻，保留全部缺陷       ← 与线上对齐用这个
 ├─ claude/fixed/          Java，七处立即修复
 │    └─ claude/npv_py/    Python 复刻 fixed 版，冻结基线
 │         └─ claude/npv_lab/  实验层，只替换特征生成
```

结论先行：**要验证"分数是否一致"，请用 `npv_exact`，不要用 `npv_py`。** npv_py 相对原 Java 有第二部分列出的二十来处差异，其中七处是有意的修复，分数必然不同；合成样例上 4419 条只有 191 条分数相同，平均差 3.1 分，最大差 62 分。

---

## 第一部分：fixed Java 相对原 Java 的七处改动

详见 `立即修复变更说明.md`，这里只列索引。

| 编号 | 位置（原 Java） | 改动 | 对分数的影响 |
| --- | --- | --- | --- |
| P0-1 | `preProcessPr` :107 | 区间上界改为 `prSplit[idx+1]`，返回 `(idx+1)/n`，低于最小值返回 0 | 原来 pr 只有 0.01 / 1.0 两个值，修复后落在 (0,1]。多数页面 pr 贡献下降，最多 4 分 |
| P0-2 | `getSrSprScore` :151 | 去掉第二次 `spr / sprMax` | sprMax > 1 时 spr_sr 普遍上升，权重 60，影响最大 |
| P0-3 | `normalizationScore` :140 | `sortByKey` 改 `sortBy(score)` | 原作业 isScroll=1 会崩，无分数可比 |
| P0-4 | `timeStamp2Date` :317 | 删除默认日期 2024-05-22，缺失时间系数取 1 | 原来缺失 pct 的页按 2024-05-22 到今天衰减（今天约 0.79） |
| P0-5 | `getPtDownFactor` / `getPctDownFactor` | 天数钳制 ≥ 0 | 原来未来时间戳系数 > 1 |
| P0-6 | `getScoreRDD` | 坏行跳过并计数 | 原作业整体失败 |
| P0-7 | `getSprMax` | sprMax ≤ 0 报错退出 | 原来产出 NaN |
| 附带 | `score` :180 | 删除 `pt.length() > 5` 判断 | 由缺失中性处理覆盖，效果相同 |
| 附带 | 构造函数 / Main | "当前时间"改为作业启动时统一注入 | 原来每条记录各取一次 |

---

## 第二部分：npv_py 相对原 Java 的全部差异

前七项继承自 fixed 版（见第一部分），下面是 Python 侧另外引入的。每项给出"复现不一致时如何判断是它"。

| 序号 | 项目 | 原 Java | npv_py | 影响范围 | 如何判断 |
| --- | --- | --- | --- | --- | --- |
| 1 | 站点解析 | 原项目 `ParseSiteUtil.parseSite`，实现未知 | `npv/site.py`：host 小写、去端口、不去 www | dr、ow、黑白名单四处查表 | 整批站点级特征对不上，尤其 dr 全是默认值 1/3 或 ow 全 0 |
| 2 | 名单文件空行 | `textFile().collect()` 保留空行；空白名单条目使 `url.contains("")` 恒真，所有页 adc=3；空黑名单条目使所有页 ow=0 | 跳过空行 | adc、ow | 线上所有页 adc 都是 3 或 ow 都是 0，而 Python 不是 |
| 3 | 后缀表匹配顺序 | 按 Java `HashMap` 迭代顺序取第一个命中 | 按文件顺序 | 只影响同时命中多个后缀的站点（如表里同时有 `edu.cn` 和 `cn`） | 差异集中在特定后缀的站点；npv_exact 会报告"命中多个后缀的行数" |
| 4 | 浮点精度 | 特征为 32 位 float，每步舍入 | 64 位 | 所有行，量级约 1e-6 | 差异绝对值 < 1e-4 即可忽略 |
| 5 | 并列分数排序 | `zipWithIndex` 顺序取决于分区，不确定 | 按 url 二级排序 | 只影响 npv 等级列，且只在分数并列时 | npv_ori 一致而 npv 等级在并列处不同 |
| 6 | 特征 JSON | fastjson 按 `HashMap` 顺序输出（`adc,pr,spr,spr_sr,ow,dr,sr`），整数值不带 `.0` | 插入顺序（`sr,spr,...`），Python 数字格式 | 只影响 npv_fea 列文本，不影响分数 | 忽略 |
| 7 | 坏行 | 任一行异常整个作业失败 | 跳过并计数 | 行数 | Python 输出行数比输入少，线上作业根本没跑完 |
| 8 | 排序实现 | `sortByKey` 以 Tuple2 为键，会抛 ClassCastException | `sortBy(score)` | isScroll=1 | 线上 isScroll=1 能否跑通需核实 |
| 9 | 时间戳位数 | 13 位截 10 位；其余长度 ≥ 10 一律拼 `"000"` 当毫秒解析，11、12、14+ 位会得到离谱日期；< 10 位取默认日期 | 只接受 10 位和 13 位，其余中性 | 时间戳格式异常的行 | 差异只出现在 pct/pt 位数不是 10 或 13 的行 |
| 10 | 非数字时间戳 | 异常被 catch，按 1 天衰减（系数 ≈ 0.9997） | 中性（系数 1） | 极少 | 可忽略 |
| 11 | JSON 解析 | fastjson 宽松：单引号、无引号键、数字含逗号、`"85.0"` 转 int | 标准 JSON | 只在输入不是标准 JSON 时 | Python 把行判为坏行而线上正常 |
| 12 | 时区 | 集群默认时区 | 运行机器的本地时区 | 日历日边界附近的时间戳 | 差异只在少数行且恰好对应一天的衰减量（约 0.03%） |
| 13 | "当前时间" | 每条记录各取一次 `System.currentTimeMillis()` | 作业统一一个值（`--now`） | 跨零点的作业 | 差异恰好等于一天衰减量 |
| 14 | 分数输出格式 | `Double.toString`，< 1e-3 或 ≥ 1e7 用科学计数 | Python `repr` | 只影响文本 | 忽略 |
| 15 | 输出布局 | `npv_ori/part-*` 多文件 | `npv_ori.tsv` 单文件 | 无 | 用 `compare_with_java.py` 对齐 |
| 16 | 正态分布 CDF | commons-math `Erf` | Python `math.erf` | 只可能在极少数分桶边界差 1 个名次 | 只影响 npv 等级列 |
| 17 | 行切分 | `String.split` 丢弃尾部空列 | 保留 | 第 5 列之后的空列无影响；第 5 列本身为空时两边都判坏行 | 忽略 |
| 18 | 区间查找 | 线性扫描 | 二分 | 无（区间单调时等价） | 忽略 |
| 19 | 表值解析时机 | dr、ow 表的值在查到时才 `Float.parseFloat`，非数字值只在命中时才崩 | 同样在查到时才转换 | 无 | 忽略 |
| 20 | `Float.parseFloat` 接受范围 | 接受 `"3f"`、`"3d"`、前后空白 | `float()` 接受前后空白但不接受 `f/d` 后缀 | 表值带后缀时 | Python 报 ValueError 而线上正常 |

---

## 第三部分：npv_exact 相对原 Java 的残余差异

npv_exact 已经模拟了第二部分 2、3（桶顺序）、4、6、7（默认失败）、8（默认失败）、9、10、13、14、17、20 各项。仍然无法保证一致的只剩下面几条。

| 项目 | 说明 | 处理 |
| --- | --- | --- |
| 站点解析 | `parseSite` 实现未知 | 拿到原实现后改 `npv_exact.py` 里的 `parse_site`，这是**首要核对项** |
| 后缀表桶内顺序 | 桶顺序按 Java HashMap 模拟，桶内顺序原本取决于 Scala HashMap，这里用文件顺序近似 | 看输出统计 `suffix_multi_match_rows`，为 0 则无影响；不为 0 时请检查后缀表是否有互相包含的键 |
| 当前时间 | 原作业逐条取，不可复现 | 用 `--now` 传作业运行当天的时间戳；跨零点作业允许一天误差 |
| 时区 | 需知道集群默认时区 | `--tz Asia/Shanghai` |
| float 双重舍入 | Python 先算 double 再舍入到 float32，与 Java 直接 float32 运算在极罕见情形差 1 ulp | 忽略 |
| erf 实现 | 见第二部分 16 | 只影响 npv 等级列 |
| `Float.toString` | JDK 19 之前某些值输出非最短位数 | 只影响 npv_fea 文本 |
| fastjson 宽松解析 | 见第二部分 11 | 输入若是标准 JSON 无影响 |
| `sortByKey` | 原作业会崩，`--assume-sort-works` 是假设它能跑 | 若线上 isScroll=1 确实能跑，说明原环境与还原代码不同，需核对 |
| pr 为数字时的字符串形式 | fastjson 对 BigDecimal 的 `toString` 可能给科学计数（如 `1E-7`） | 解析回 double 后相同，无影响 |

---

## 第四部分：排查顺序

1. **先用 npv_exact 对齐线上输出。** 同一份输入和同一套表，`--tz` 传集群时区，`--now` 传作业当天时间戳，`--skip-bad-rows` 跳过坏行。用 `npv_py/compare_with_java.py --tol 1e-4` 比较 `npv_ori`。
2. **全部不一致**：先看输出统计里的 `spr_max` 是否与线上日志一致；再核对 `parse_site`；再核对表文件的键格式（后缀是否带前导点、站点是否带 www）。
3. **只有部分站点不一致**：看 `suffix_multi_match_rows` 和两个 `empty_lines_in_*` 统计；看不一致的站点是否集中在某个后缀。
4. **只有部分行不一致、且差异等于一天衰减量（约 0.03% 到 0.05%）**：时区或跨零点问题。
5. **只有 pct/pt 格式异常的行不一致**：第二部分第 9 项，看这些行的时间戳位数。
6. **差异 < 1e-4**：浮点，忽略。
7. **只有 npv_fea 列不同**：格式，忽略。
8. **npv_exact 一致但 npv_py 不一致**：这是预期的，对照第一、二部分逐项解释。P0-1 与 P0-2 是最大的两项：合成样例上 pr 特征平均相差 0.50、spr_sr 平均相差 0.13。
9. **npv_py 内部**：`npv_lab` 默认配置下与 `npv_py` 逐条一致（有回归测试），若不一致是实验层的 bug。
