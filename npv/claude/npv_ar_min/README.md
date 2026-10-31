# npv_ar_min：薄版本 autoresearch（Karpathy 原版玩法）

三样东西：`program.md`（规矩）、`eval.py`（冻结评估）、`sandbox/`（agent 改的特征与参数）。git 提交、回退、写 `results.tsv` 都由 agent 自己做，和 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 一样。防护只靠 opencode 的权限配置和 `program.md` 的自觉；要完整的门禁、防作弊和噪声控制用 `../npv_ar/`。两者评估口径相同，结果可直接比。

依赖：同级的 `../npv_py`、`../npv_lab`、`../npv_ar`（复用它的 `harness/common.py`、`metrics.py`、`score.py`）；Python 3.8+；git；opencode（内网叫 `codeagent`）。

## 目录

| 路径 | 谁改 | 作用 |
| --- | --- | --- |
| `program.md` | 人 | 给 agent 的指令。本版本允许 agent 直接改 `config.json` 里的数值 |
| `eval.py` | 冻结 | 打分 + 严格一致率，只打印一行。`--test` 读仓库外测试集，只给人用 |
| `sandbox/features.py`、`config.json`、`lists/` | agent | 对应 Karpathy 的 `train.py` |
| `data/` | 人 | `input.tsv`、六张表、`meta.json`（region、now、tie_band）、可选 `level_map.tsv` |
| `labels/train/pairs.tsv` | 人 | agent 的验证集（你的 80%） |
| `../npv_ar_private/min_test/pairs.tsv` | 人 | 你的 20%，仓库外，agent 读不到 |
| `results.tsv`、`journal.md` | agent | 实验记录与笔记 |
| `opencode.json` | 人 | 权限：只能改 `sandbox/`、`results.tsv`、`journal.md`；bash 只放行 `python eval.py` 和少数 git 命令 |
| `run_once.sh`、`loop.sh` | 人 | 跑一个会话 / 反复跑 |
| `tools/fit_formula.py`、`tools/features_v2.py` | 人 | 用偏序标签拟合加法公式：特征表导出、L1 稀疏、项数-一致率曲线、Java 伪代码 |
| `tools/convert_labels.py` | 人 | 把 gemini 判官的 `labeled_*.jsonl` 转成 `pairs.tsv`，顺便出标签统计、URL 清单、线上 npv 表 |
| `tools/prepare_synthetic.py` | 人 | 合成数据，只为跑通 |

## 最短路径：先用拟合替掉手定权重（`tools/fit_formula.py`）

```bash
python3 tools/fit_formula.py --data data --train labels/train/pairs.tsv --test ../npv_ar_private/min_test/pairs.tsv \
    --out work/fit --features-out work/features.tsv --drop-decay
```

- 用 `tools/features_v2.py` 给每页算约 30 列（6 个基线特征、pc 14 位各自独立、正文长度对数与分档、距今年数、URL 形态），`--features-out` 导出的表可直接喂 xgboost。
- 偏序逻辑回归拟合加法公式 `score = max(0, Σ w·f)`，一组 L1 强度给出"项数 对 一致率"曲线，自动选离最高值 0.3 个点以内最稀疏的那条；打印权重、符号检查、Java 伪代码。
- `--drop-decay` 把 pct/pt 的时效项排除在权威分之外（LLM 不看新旧；时效在线上单独乘回去）。
- 复现到 `eval.py`：`sandbox/features.py` 换成 `tools/features_v2.py`，`config.json` 里 `fea_weight` 用 `work/fit/fea_weight.json`、`fit` 置空、加 `"linear_only": true`。两边的数字应完全一致（合成数据上已核对）。
- `linear_only=true` 时 `eval.py` 输出 `form=linear`：分数就是加法公式本身，不再乘衰减、页面类别、正文长度系数。

## 评估口径

`acc` = (判对 + 0.5 × 同档) / LLM 有明确方向的对数，打分缺失的对算错。同档的定义在 `data/meta.json`：

```json
"tie_band": {"unit": "level", "value": 50}
```

`unit=level` 时先用 `data/level_map.tsv` 把原始分映射到线上 1~1000 等级再比，和你造数据集时"分差 ≤ 50 算同档"的口径一致，但同档只记半对，不记全对，否则 agent 把分数压扁就能刷分。没有 level_map 时用 `{"unit": "raw", "value": 0}`。

`hard` 是分差 ≤ 2×band 的难对上的一致率；`ties`、`missing` 上升要警惕。

## 跑起来

在公司内网从零开始的完整步骤见 [内网操作步骤.md](内网操作步骤.md)。下面是速查。

```bash
cd claude/npv_ar_min
# 1. 数据：真实数据用 convert_labels；跑通流程用合成数据
python3 tools/convert_labels.py --train output/labeled_train.jsonl --test output/labeled_test.jsonl \
    --out-train labels/train --out-test ../npv_ar_private/min_test --aux data
#    然后把 data/urls.txt 里所有 URL 的打分字段（input.tsv）和六张表放进 data/，写 data/meta.json
#    有了 input.tsv 后可拟合 level_map：先用 ../npv_py/run_npv.py 打分得到 npv_ori，再和 data/online_npv.tsv 配对，
#    用 ../npv_py/fit_level_map.py 生成 data/level_map.tsv
python3 tools/prepare_synthetic.py        # 或者：合成数据

# 2. 基线
git init -b main && git add -A && git commit -m init
python3 eval.py
python3 eval.py --test                    # 人看一眼测试集

# 3. 跑 agent
OPENCODE_BIN=codeagent OPENCODE_MODEL=provider/model ./run_once.sh     # 一个会话，约 10 次实验
./loop.sh 20                                                          # 反复跑，STOP 文件可停
```

每晚结束后人做两件事：`python3 eval.py --test` 看 agent 保留的版本在 20% 上有没有真的变好；翻 `results.tsv` 和 `git log`，看它有没有写死域名、有没有把分数压扁。

## 和 npv_ar 的区别

| | npv_ar_min | npv_ar |
| --- | --- | --- |
| 保留/回退 | agent 自己判断、自己 git | 冻结的 run.py 判定并执行 |
| 数值参数 | agent 可直接改 | agent 不能改，优化器拟合 |
| 验证集 | agent 看到逐条结果，可反复查 | dev 只给汇总，不保留时只给粗档，有查询预算 |
| 噪声门槛 | program.md 里写死 0.003 | 按聚类标准误自动算 |
| 防作弊 | opencode 权限 + 自觉 | 守卫扩展 + lint + 基线特征检查 + 泄漏检查 |
| 适合 | 先看 agent 自己转起来是什么样 | 无人值守过夜、结果要可信 |
