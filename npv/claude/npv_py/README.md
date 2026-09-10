# npv_py：NPV 网页权威度打分的 Python 复刻

目的：在没有 Spark 集群和 Java 环境的情况下，用同一套逻辑在本地对样本数据快速跑分、做特征与参数实验、搭建自动评估循环。逻辑对应 `claude/fixed/` 下的 Java 立即修复版（含七处 P0 修复），不是原始有 bug 的版本。

只依赖 Python 3.8+ 标准库。

## 目录

| 文件 | 对应 Java | 说明 |
| --- | --- | --- |
| `npv/scorer.py` | `PageValueScore` 的 preProcess* / getSrSprScore / adjustScore* / score | 打分核心，纯函数式，返回 `(score, features)` |
| `npv/config.py` | 散落在 Java 各处的魔法数字 | `ScoreConfig`：权重、阈值、衰减常数、正文长度规则、pc 位规则表、adc 加分。实验时只换配置不改代码 |
| `npv/tables.py` | 构造函数里的表加载 | dr 站点表、dr 后缀表、官网表、黑白名单、pr 切分点、sprMax（含 > 0 校验） |
| `npv/timeutil.py` | decayFactor / parseTimestampSeconds / daysBetween | 时间衰减 |
| `npv/normalization.py` | generateUrlInterval / normalizationScore | 正态分桶排名归一化，`math.erf` 替代 commons-math |
| `npv/site.py` | 原项目的 `ParseSiteUtil.parseSite`（实现未知） | 默认取 host 小写去端口，**需与原实现对齐** |
| `npv/io.py` | `getScoreRDD` 里的行解析 | 输入行 -> `ScoreInput`，坏行抛 `BadRow` |
| `run_npv.py` | `PageValueScoreMain` | 命令行入口，输出格式与 Java 作业一致 |
| `make_sample.py` | 无 | 生成一套合成样例数据（输入 + 七张表） |
| `compare_with_java.py` | 无 | 在原环境把 Java 输出与 Python 输出按 url 对齐比较 |
| `tests/test_npv.py` | 无 | 单元测试 + 端到端测试 |

## 快速开始

```bash
cd claude/npv_py
python make_sample.py --out sample --rows 5000
python run_npv.py --input sample/input.tsv --spr sample/spr.tsv --dr-site sample/dr_site.tsv \
    --dr-suffix sample/dr_suffix.tsv --ow sample/ow.tsv --pr-split sample/pr_split.tsv \
    --ow-blacklist sample/ow_blacklist.txt --adc-whitelist sample/adc_whitelist.txt \
    --output out --region zh --scroll 1 --now 1757030400
python -m unittest discover -s tests -v
```

`--now` 是时间衰减基准（秒级时间戳），固定它可以保证两次运行结果完全一致；不传则取当前时间。

输出：

- `out/npv_ori.tsv`：`url \t npv_ori \t npv_fea`，`--scroll 1` 时只含非高质量网页（第 2 列不在 {1,2,5}）
- `out/npv.tsv`：`url \t npv_ori \t npv \t npv_fea`，高质量网页的排名归一化等级 1~1000

## 在代码里做实验

```python
from npv import PageValueScore, ScoreInput, ScoreConfig, load_tables
from npv.config import PcRule, region_site_list

tables = load_tables("sample/spr.tsv", "sample/dr_site.tsv", "sample/dr_suffix.tsv", "sample/ow.tsv",
                     "sample/pr_split.tsv", "sample/ow_blacklist.txt", "sample/adc_whitelist.txt")

# 基线
base = PageValueScore(tables, now_seconds=1757030400)

# 实验：调权重、加一条页面类别规则
cfg = ScoreConfig(
    fea_weight={"spr_sr": 50, "pr": 10, "dr": 12, "ow": 8},
    pc_rules=(PcRule("forum", 40, 0.7),) + ScoreConfig().pc_rules,
)
exp = PageValueScore(tables, now_seconds=1757030400, config=cfg)

x = ScoreInput(url="https://www.zhihu.com/question/1", pr=2.3, adc=0, pc=0,
               pct=1740000000, pt=0, pure_text_len=1200, sr=88, spr=7.0)
print(base.score(x, region_site_list("zh")))
print(exp.score(x, region_site_list("zh")))
```

`score()` 返回 `ScoreResult(score, features)`，`features` 是有序 dict（sr, spr, spr_sr, dr, ow, pr, adc），可直接落成表做分析。`adjust_pc()` 额外返回命中的规则名，便于统计各规则命中率。

要加新特征：在 `scorer.py` 的 `gen_features` 里增加一项，在 `ScoreConfig.fea_weight` 里给权重即可；`basic_score` 会自动按权重表求和。

## 与 Java 的对应关系与已知差异

忠实复刻的部分：特征预处理、spr_sr 分段映射、基础分加权、pct/pt 衰减公式与常数、pc 位规则及其优先级、正文长度系数、adc 加分与跳过逻辑、正态分桶的边界计算、输入行解析与坏行处理、区域白名单（含未改动的 `.edu.com`）。

有意的差异：

| 项 | Java | Python | 原因 |
| --- | --- | --- | --- |
| 结果传递 | 写实例字段 `featureList` | 返回 `ScoreResult` | 纯函数便于测试与并行 |
| 参数 | 硬编码 | `ScoreConfig` | 实验需要 |
| 并列分数的排名 | 不确定（取决于分区顺序） | 按 url 二级排序 | 可复现 |
| 特征 JSON 键序 | HashMap 无序 | 固定插入顺序 | 可 diff |
| 浮点精度 | 特征为 32 位 float | 64 位 | 差异约 1e-6，比较时用 `--tol 1e-4` |
| 分桶查找 | 线性扫描 | 二分 | 区间严格递增，结果相同 |
| parseSite | 原项目工具类 | 本地默认实现 | 原实现未知，**首要核对项** |

无法在本地验证的部分：`parseSite` 的行为、真实表文件的格式细节（例如后缀表的键是否带前导点）、时区（两边都用系统默认时区按日历日计算）。拿到原环境后用 `compare_with_java.py` 做一次对齐：

```bash
python compare_with_java.py --java /hdfs-local-copy/output/npv_ori --py out/npv_ori.tsv --tol 1e-4
```

## 下一步：接评估循环

这份代码已经把"打分"做成了纯函数加配置对象，接 autoresearch 式循环时还需要：

1. 带标签的样本（人工标注权威度或下游搜索评测集）与一个冻结的 `eval.py`，输出单一指标和硬约束检查（无 NaN、范围、单调性、分布漂移）；
2. 训练/验证/隐藏 holdout 三份数据；
3. 一个只允许修改 `npv/scorer.py` 与 `ScoreConfig` 的 agent 规则文件，每轮记录指标与改动摘要。

## 注意：本目录是冻结基线

不要在这里做实验或加特征。诊断、解释、评估、实验代码都在 `../npv_lab/`，它通过 `LabScorer` 继承这里的打分器；`npv_lab/tests` 里有一条回归测试保证默认配置下两边分数逐条一致。
