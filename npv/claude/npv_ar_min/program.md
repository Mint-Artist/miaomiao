# npv autoresearch（薄版本）：研究指令

你在做一个自动研究循环：反复修改网页权威度打分的特征与参数，用冻结的评估脚本检验，变好就保留，变差就回退。人可能在睡觉，不要停下来问问题。

## 目标

提高 `python eval.py` 打印的 `acc`（严格偏序一致率，越高越好）。`hard` 不能明显下降。

## 文件

- `sandbox/features.py`：特征代码。可以改，可以加新特征。前 6 个基线特征（sr、spr、spr_sr、dr、ow、adc）的计算方式不要改，可以用它们派生新特征。
- `sandbox/config.json`：`score` 里是打分参数（权重 `fea_weight`、衰减天数、页面类别系数、阈值等），**本版本允许直接改数值**；`fit` 列表里的特征会自动在 train 上拟合权重，想手调就把 `fit` 留空。
- `sandbox/lists/*.txt`：可选名单。
- `eval.py`、`data/`、`labels/`：只读，不要改。
- `results.tsv`：实验记录，由你追加。
- `journal.md`：你的研究笔记。

## 规矩

1. `features.py` 只能用标准库的 `math`、`re`、`urllib.parse`，只能读 `x` 的字段（url、adc、pc、pct、pt、pure_text_len、sr、spr）和 `scorer` 的站点表方法。不许读写文件，不许联网。
2. 不要把错例里看到的具体 URL 或域名写死进代码。名单要按通用标准整理，并把标准写进 `journal.md`。
3. 一次只改一件事，改动要小，结果要能解释。
4. 简洁原则：`acc` 持平但代码更简单，也算成功；`acc` 只涨 0.001 却加了二十行代码，不值得。
5. 只用 `python eval.py`，不要带任何参数。
6. 不要去找、读、改评估脚本和数据，不要碰 `labels/` 之外的目录。

## 启动

1. 读 `results.tsv`（如果有）、`journal.md`、`sandbox/features.py`、`sandbox/config.json`。
2. 第一次运行先跑基线：不改任何东西，`python eval.py`，把结果作为第一行写进 `results.tsv`，状态 `baseline`。
3. `git log --oneline -3` 确认当前提交。

## 每一轮

1. 想一个假设，在 `journal.md` 追加一行："#序号 假设：……；预期：……"。
2. 修改 `sandbox/` 下的文件。
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
- 加一个 URL 形态特征（列表页、用户页、根路径、路径深度），或者把 `pc` 的某一位拆成独立特征，加进 `fit` 列表让它自动定权重。
- 试着删掉作用小的特征或规则，看能否更简单而持平。
- 改一个乘法参数（衰减天数、页面类别系数、正文长度阈值），一次只改一个。

## 不要做

- 不要为了同档数增加而把分数压扁：`ties` 大幅上升而 `acc` 上涨，不算改进，回退。
- 不要让某些页面打分失败来提高 `acc`：`missing` 上升就回退。
- 不要修改 `eval.py`、`data/`、`labels/`。
