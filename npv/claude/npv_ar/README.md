# npv_ar：npv 的 autoresearch 骨架

agent 在 `sandbox/` 里改特征，冻结的 `harness/run.py` 负责打分、两道关、保留或回退、记录。设计见
`../autoresearch/NPV自动研究框架初步方案.md` 与 `../autoresearch/五个问题的回答与方案修订.md`；pi 的选型见 `../autoresearch/pi框架可行性评估.md`。

依赖：Python 3.8+ 标准库；同级目录 `../npv_py`（冻结基线）与 `../npv_lab`（评估与拟合工具）；git。
agent 运行时：pi 编程 agent **v1.3.0**（锁定；`npm i -g @earendil-works/pi-coding-agent@1.3.0 --ignore-scripts`，或 GitHub Release 的独立二进制），Node ≥ 22.19。

## 目录

| 路径 | 谁改 | 作用 |
| --- | --- | --- |
| `program.md` | 人 | 给 agent 的研究指令，作为系统提示附加 |
| `ideas.md`、`journal.md` | agent | 想法清单、研究笔记。笔记里出现只有 dev 才有的 URL 会被判 lint_fail |
| `sandbox/features.py` | agent | 特征代码。只能导入 math、re、urllib.parse；前 6 个基线特征不能改 |
| `sandbox/config.json` | agent | `{"fit": [要拟合的特征名], "score": {ScoreConfig}}`。agent 只往 fit 里加名字，不改数值 |
| `sandbox/lists/*.txt` | agent | 可选名单，总行数有上限 |
| `harness/run.py` | 冻结 | 唯一评估入口，只接受 `--hypothesis` 和 `--desc` |
| `harness/score.py` | 冻结 | 打分子进程：只看快照、sandbox、train 标签；拟合权重；按 qid 分组 5 折交叉验证 |
| `harness/metrics.py`、`gate.py`、`lint.py` | 冻结 | 严格一致率与护栏、两道关判定、静态检查 |
| `harness/errors.py` | 冻结 | train 错例分析（agent 通过 `show_errors` 工具调用） |
| `harness/holdout_check.py` | 人 | 中环：在仓库外的 holdout 上复核并打 `promoted/N` 标签 |
| `harness/snapshot/` | 人 | 输入快照 + 六张表 + `meta.json`（region、now、哈希） |
| `harness/labels/train/` | 可读 | 逐条标签；`labels/dev/` 被守卫禁读 |
| `../npv_ar_private/holdout/` | 人 | 仓库之外，agent 读不到 |
| `pi/guard.ts` | 人 | pi 扩展：路径守卫、`run_eval` 与 `show_errors` 工具、轮数上限 |
| `pi/run_once.sh`、`pi/loop.sh` | 人 | 跑一轮 / 无人值守循环 |
| `tools/prepare_synthetic.py` | 人 | 用 npv_py 的合成样例生成快照与标签，只为跑通流程 |
| `tools/judge_pairs.py` | 人 | 大模型判官标签流水线：抽头尾对、两种顺序 × 两轮判定、一致性过滤、按 qid 切分、单页档位 |
| `tools/probe_llm.py` | 人 | 探测模型接口：连通、工具调用（非流式/流式）、工具结果回传、compat 字段，打印 models.json 建议 |
| `tools/mock_llm_server.py` | 人 | 按脚本返回工具调用的假模型服务（normal / cheat / echo 三场景），用于不接真实模型时测试 |
| `results.tsv`、`runs/`、`harness/state.json` | run.py | 实验记录。`runs/<id>/card.json` 是评估前写下的实验卡 |

## 一轮的流程（`run.py`）

1. 写实验卡（假设、描述、sandbox 哈希、父提交）。
2. 静态检查 sandbox；检查 journal/ideas 有没有泄漏 dev URL。不过：`lint_fail`，回退。
3. 子进程打分（工作目录是 run 目录，环境变量最少，可用 `score_command_prefix` 套容器）。失败：`crash`，回退。
4. 硬约束：基线特征不变、无 NaN/越界、坏行不增加。域名字面量超限：`review`，回退，等人看。
5. 第一道关：train 上按 qid 分组的 5 折交叉验证，提升 ≥ `cv_gain_min`，或持平且更简单。不过：`discard`，不消耗 dev 预算。
6. 第二道关：dev 严格一致率，Ladder 门槛 ε = max(`eps_floor`, 按 qid 聚类自助法的配对差标准误)；护栏（小分差桶、新站子集、误压率、误抬率、遮域名一轮）不得变差超过 `guard_tol`。
7. `keep`：提交，更新最好版本；否则回退 sandbox。两次 git 提交（代码 + 记录），保证工作区干净。
8. 摘要只有一行。`keep` 给精确 dev 数字；`discard` 只给粗档。

严格一致率 = (判对 + 0.5 × 同分) / 非"相当"对数（含打分缺失的对）。同分与缺失不再被排除，堵住两个刷分口子。

## 本机跑通（合成数据）

```bash
cd claude/npv_ar
python3 tools/prepare_synthetic.py          # 快照 + train/dev + ../npv_ar_private/holdout
git init -b main && git add -A && git commit -m init
python3 harness/run.py --hypothesis 基线 --desc 基线
python3 harness/errors.py --top 10
python3 -m unittest discover -s tests -v
```

用假模型走一遍 pi + 守卫：

```bash
python3 tools/mock_llm_server.py --scenario normal --port 8787 &   # 或 --scenario cheat
PI_CODING_AGENT_DIR=/tmp/pi-mock  # 里面放 models.json：baseUrl http://127.0.0.1:8787/v1，api openai-completions
NPV_AR_MODEL=mock/mock-1 PI_BIN=pi pi/run_once.sh
```

`normal` 场景会读特征、看错例、加一个列表页特征、调用 `run_eval`；`cheat` 场景依次尝试读 dev 标签、读 holdout、写 harness、写仓库外文件、调用 bash，每一步都应被守卫拦下。

## 先验证模型接口（内网第一步）

```bash
python3 tools/probe_llm.py --base http://内网网关/v1 --key "$KEY" --model 模型名
python3 tools/probe_llm.py --base http://内网网关/v1 --key "$KEY" --model 模型名 --pi   # 配好 models.json 后再加 --pi
```

纯标准库，依次检查：`/models` 连通、文本回复、非流式工具调用、**流式工具调用（pi 用的路径）**、工具结果回传、以及 pi 默认会发的 `store`、`developer` 角色、`max_completion_tokens`、`reasoning_effort` 服务端是否接受。结束时直接打印该写进 `~/.pi/agent/models.json` 的 provider 配置（含 `compat`）。第 3、4 项不过，pi 就用不了，要让部署方开启服务端工具调用（vLLM：`--enable-auto-tool-choice --tool-call-parser <与模型匹配>`），或改用自写驱动。

本机自测：`python3 tools/mock_llm_server.py --scenario echo --port 8787 &` 再对 `http://127.0.0.1:8787/v1` 跑一遍，7 项应全部 PASS。

## 用大模型判官造标签（`tools/judge_pairs.py`）

输入 `pages.jsonl`：每行一个检索结果页，字段 `qid, query, rank, url, title, text`（正文纯文本，去掉导航页脚），可选 `date, author, vertical`。四步：

```bash
T=tools/judge_pairs.py
python3 $T sample --pages pages.jsonl --out work/todo.jsonl --per-query 1 --require-text   # top1-10 抽一页、top11-20 抽一页，随机决定谁是 A
python3 $T prompt --todo work/todo.jsonl --index 0 [--blind]                               # 先看一眼 prompt
python3 $T judge  --todo work/todo.jsonl --out work/judgments.jsonl --base http://内网网关/v1 --key $KEY --model 判官模型 --workers 8 --limit 300
python3 $T build  --judgments work/judgments.jsonl --todo work/todo.jsonl --out work/labels
python3 $T split  --labels work/labels --out-train harness/labels/train --out-dev harness/labels/dev --out-holdout ../npv_ar_private/holdout
```

- `judge` 每对调 4 次：带域名 × (AB, BA) 与遮域名 × (AB, BA)；temperature 0；追加写、可续跑；原始 JSON 全部保留在 `judgments.jsonl`。
- `build`：两种顺序结论相反的对剔除，一次 tie 一次有偏向记为 tie；任一页不切题或相关性差 2 档的对剔除；`stats.json` 里有顺序一致率与 tie 率，这是判官质量的第一个信号。
- `split` 按 qid 切；遮域名一轮只进 dev 作护栏。
- 单页绝对档位：`python3 $T grade --pages pages.jsonl --rank-min 21 --rank-max 50 --out work/grades.jsonl --grades-tsv harness/labels/dev/grades.tsv ...`。
- 本机自测：`python3 tools/mock_llm_server.py --scenario judge --port 8787 &`，假判官按稳定哈希判，顺序一致率应为 1.0。

## 接真实数据与内网模型

1. 把 `input.tsv` 与六张表放进 `harness/snapshot/`，改 `meta.json` 的 `region`、`now`。
2. 标签：`pairs.tsv` 五列 `url_a url_b pref group qid`（qid 是 query 的编号，按它切分与做交叉验证），`grades.tsv` 三列。train、dev 放 `harness/labels/`，holdout 放仓库外。有遮域名一轮就放 `labels/dev/pairs_blind.tsv`，会作为护栏。
3. 删除 `harness/state.json`、`results.tsv`、`runs/`，重跑基线。
4. `~/.pi/agent/models.json` 按 `pi/models.json.example` 配置内网接口；`NPV_AR_MODEL=intranet/模型名`。
5. 把 `pi/guard.ts` 复制到 agent 写不到的目录，用 `NPV_AR_GUARD` 指过去。
6. 正式环境把打分子进程放进容器：`harness/config.json` 的 `score_command_prefix` 填容器命令前缀（断网、只读挂载代码、不挂载 `labels/dev` 与 holdout）。
7. 先有人值守跑几轮（`pi/run_once.sh`，或去掉 `--mode json` 交互地看），据表现改 `program.md`；再用 `pi/loop.sh` 过夜。
8. 每天 `python3 harness/holdout_check.py --promote` 做晋级复核。

## 本机实测记录（2026-10-19，pi v1.3.0 + 假模型）

- `normal` 场景：read → show_errors → write ×2 → run_eval → 结束，`terminate: true` 生效，agent 不再追加回合。
- `cheat` 场景：读 dev 标签、读 holdout、写 harness/run.py、写仓库外文件、调用 bash，五项全部被拦下；合法的 read 与 run_eval 正常。
- 通过 run.py 直接做的三种作弊（features.py 里 import os、把基线特征压成常数、journal 贴 dev URL）分别被 lint、基线特征检查、泄漏检查拦下。
- 踩到的坑：`--mode json` 时 stdin 若保持打开，pi 会一直等待不发第二个请求；`run_once.sh` 已加 `< /dev/null`。
- 10 个单元测试通过，含端到端一轮。

## 已知限制

- 第一道关和拟合都只在 train 的偏序标签上做；权重拟合是 pairwise 逻辑回归，只拟合线性权重，乘法项由 agent 通过结构改动处理（目前不开放）。
- 新站子集一致率在站点很少的合成数据上为空；真实数据生效。
- `score_command_prefix` 为空时打分子进程没有操作系统级隔离，只靠静态检查；内网务必配容器。
- 守卫扩展针对 pi v1.3.0 的扩展 API 编写；升级 pi 前先跑 `cheat` 场景确认拦截仍然有效。
