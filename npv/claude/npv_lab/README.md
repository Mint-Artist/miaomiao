# npv_lab：在冻结基线之上做诊断、解释、评估与实验

`../npv_py/` 是策略的忠实复刻，**冻结不动**，作为一切实验的对照。本目录的代码都通过路径引用它，本身不修改它。

只依赖 Python 3.8+ 标准库。

## 目录

| 文件 | 作用 |
| --- | --- |
| `lab/features.py` | **实验特征**。自改进循环中 agent 只允许改这个文件与 `configs/*.json`。前 7 个特征与基线一致；后面的实验特征默认权重 0 |
| `lab/scorer.py` | `LabScorer`：继承基线打分器，只替换特征生成；`fea_weight` 里没有的特征自动权重 0 |
| `lab/config_io.py` | `ScoreConfig` 与 JSON 互转 |
| `lab/explain.py` | 把一条记录的分数分解成每一步：各特征加权贡献、衰减系数、页面类别规则、正文长度、adc 加分 |
| `lab/diagnose.py` | 结构诊断：站点级方差占比、各特征方差占比、规则命中率、缺失率、站内分差 |
| `lab/evaluate.py` | 离线评估：偏序一致率（总体、按分差桶、按分组）、Spearman、误压率、误抬率、校准曲线、硬约束、配对符号检验 |
| `lab/fit.py` | 用偏序标签拟合基础分权重（pairwise 逻辑回归，纯 Python） |
| `lab/stats.py` | 分位数、方差分解、Spearman、符号检验 |
| `run_lab.py` | 用实验特征与配置打分，输出与 `run_npv.py` 同格式 |
| `explain_url.py` | 解释某几条 url 的分数 |
| `diagnose_data.py` | 输出诊断报告（Markdown / JSON） |
| `eval_scores.py` | 评估一份打分输出，可与对照做配对比较 |
| `fit_weights.py` | 拟合权重并生成新配置 |
| `experiment.py` | 一轮实验：打分、评估、与基线配对比较、追加到 `results.tsv` |
| `make_sample_labels.py` | 为合成样例生成合成标签，只用于跑通流程 |
| `configs/baseline.json` | 基线参数（等于 `ScoreConfig()` 默认值） |
| `labels/` | 标签文件 |
| `runs/` | 每次打分的输出与 `eval.json` |
| `tests/test_lab.py` | 18 个测试，含"默认配置下 LabScorer 与基线逐条一致"的回归检查 |

## 标签格式

```
labels/pairs.tsv    url_a  url_b  pref(a|b|tie)  [group]
labels/grades.tsv   url    grade(0-4)             [group]
```

偏序对用于主指标；绝对档位用于误压率、误抬率与校准曲线。`group` 可填 head/tail、垂类等，评估会分组报告。真实标签来自大模型或人工标注，`make_sample_labels.py` 生成的只是合成的。

## 快速开始（合成数据）

```bash
cd claude/npv_lab
S=../npv_py/sample
T=(--spr $S/spr.tsv --dr-site $S/dr_site.tsv --dr-suffix $S/dr_suffix.tsv --ow $S/ow.tsv \
   --pr-split $S/pr_split.tsv --ow-blacklist $S/ow_blacklist.txt --adc-whitelist $S/adc_whitelist.txt \
   --region zh --now 1757030400)

python run_lab.py --input $S/input.tsv $T --output runs/baseline --config configs/baseline.json
python make_sample_labels.py --input $S/input.tsv --out labels
python eval_scores.py --scores runs/baseline/npv_ori.tsv --pairs labels/pairs.tsv --grades labels/grades.tsv
python diagnose_data.py --input $S/input.tsv $T --out runs/diagnose_baseline.md
python explain_url.py --input $S/input.tsv $T --grep "tieba.baidu.com/u/" --limit 1
python fit_weights.py --scores runs/baseline/npv_ori.tsv --pairs labels/pairs.tsv \
    --features spr_sr,pr,dr,ow,pc_bit26,url_depth --base-config configs/baseline.json --out configs/fitted_demo.json
python experiment.py --name fitted_demo --config configs/fitted_demo.json --note "偏序拟合" \
    --input $S/input.tsv $T --pairs labels/pairs.tsv --grades labels/grades.tsv --baseline runs/baseline/npv_ori.tsv
python -m unittest discover -s tests -v
```

zsh 下数组变量 `$T` 会自动展开为多个参数；bash 下请写成 `"${T[@]}"`。

## 一轮实验的流程

1. 改 `lab/features.py`（加特征）或复制一份 `configs/*.json` 改参数。
2. `experiment.py --name xxx --config configs/xxx.json --baseline runs/baseline/npv_ori.tsv ...`
3. 看 `results.tsv` 的这一行：`pairwise_acc`、`hard_acc`（小分差桶）、`buried_good_rate`（误压率）、`vs_baseline_p`（与基线配对符号检验的 p 值）。
4. 显著变好且误压率不升则保留，否则回退。

`hard_acc` 与 `buried_good_rate` 是两个"防作弊"指标：前者防止只在容易的对上刷分，后者防止靠一刀切压低小站刷分。

## 读诊断报告的方法

`diagnose_data.py` 输出里最重要的三个数：

- **站点级解释的方差占比**：越接近 100%，同站页面越不区分，页面级特征越值得做。
- **各特征方差占比**：某特征占比接近 0，说明它的权重再怎么调也测不出差别。
- **规则命中率**：命中数只有个位数的规则，改它的系数没有统计意义。

合成样例上的结果（仅示意格式，不代表真实分布）：站点级占最终分方差 62%、基础分方差 86%；spr_sr 一项占基础分方差 82%，pr、dr、ow 合计不到 3%。

## `fit_weights.py` 的局限

它只拟合基础分那一层线性权重，衰减、页面类别、正文长度这些乘法调整原样保留，所以结果是近似最优。把乘法项也纳入拟合需要先把流水线改成单一线性形式（解读报告 4.3）。用它的正确姿势是：先在 `features.py` 里造特征、跑一遍 `run_lab.py` 让特征值落到输出里，再用它给新特征定权重，而不是手工猜。
