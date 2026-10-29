# npv autoresearch（薄版本）：研究指令

你在做一个自动研究循环：反复修改网页权威度打分的特征与参数，用冻结的评估脚本检验，变好就保留，变差就回退。人可能在睡觉，不要停下来问问题。

## 目标

提高 `python eval.py` 打印的 `acc`（严格偏序一致率，越高越好）。`hard` 不能明显下降。

## 文件

- `sandbox/config.json`：**本阶段唯一可以改的文件。** `score` 里是全部打分参数：
  - 基础分权重 `fea_weight`（spr_sr、dr、ow）；
  - spr 的拉伸与合成参数 `spr_th1`、`spr_th2`、`spr_th1_out`、`spr_th2_out`、`spr_zero_fill`、`spr_coef`、`sr_spr_div`、`sr_missing_score`；
  - 衰减天数 `pct_decay_days`、`pt_decay_days`；
  - 正文长度阈值与乘数 `text_len_rules`；
  - 每条页面类别规则的乘数 `pc_rules[*].factor`、保底分 `pc_floor`；
  - adc 加分 `adc_bonus`、免降权阈值 `adc_skip_adjust_from`、白名单等级 `adc_whitelist_level`、`max_dr`。
  只改数值，不要增删键。`fit` 列表保持为空。
- `sandbox/features.py`：本阶段不改，也没有权限改。
- `eval.py`、`data/`、`labels/`：只读。
- `results.tsv`：实验记录，由你追加。
- `journal.md`：你的研究笔记。

## 启动

1. 读 `results.tsv`（如果有）、`journal.md`、`sandbox/config.json`；可以读 `sandbox/features.py` 了解每个特征是什么，但不要改它。
2. 第一次运行先跑基线：不改任何东西，`python eval.py`，把结果作为第一行写进 `results.tsv`，状态 `baseline`。
3. `git log --oneline -3` 确认当前提交。

## 每一轮

1. 想一个假设，在 `journal.md` 追加一行："#序号 假设：……；预期：……"。
2. 修改 `sandbox/config.json`。
3. `git add sandbox && git commit -m "描述"`。
4. `python eval.py`，读那一行输出。
5. 判断：
   - `acc` 比当前最好值高出至少 0.003，且 `hard` 没有下降超过 0.01 → **保留**：记 `keep`，更新"当前最好值"。
   - 持平或变差 → **回退**：`git reset --hard HEAD~1`，记 `discard`。
   - 报错或崩溃 → 小错修了重跑，同一想法最多修两次，否则 `git reset --hard HEAD~1`，记 `crash`。
6. 往 `results.tsv` 追加一行（制表符分隔）：`commit  acc  hard  ties  complexity  status  描述`。
7. 在 `journal.md` 补一句结论，回到第 1 步。

## 没想法时

- 看 `results.tsv` 里接近门槛的尝试，组合它们。
- 三个基础分权重的相对比例：spr_sr 占了绝大部分，试试把 dr、ow 的份额提高。
- 乘法链：衰减天数放长（评估不看新旧）、用户页和列表页的乘数降低、`adc_skip_adjust_from` 调高（让权威站的列表页也受页面级降权）。
- spr 的拐点与系数：`spr_th1`、`spr_th2`、`spr_coef`。
- 把某个乘数改回 1.0，看它到底有没有用。

## 不要做

- 不要为了同档数增加而把分数压扁：`ties` 大幅上升而 `acc` 上涨，不算改进，回退。
- 不要让某些页面打分失败来提高 `acc`：`missing` 上升就回退。
- 不要修改 `eval.py`、`data/`、`labels/`、`sandbox/features.py`。
