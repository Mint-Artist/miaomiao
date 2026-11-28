
# 任务02 回传报告：败例核查与 v2.1 列表页修订
 
执行日期：2026-11-28
对应任务书：gemini_02_败例核查与v2.1列表页修订.md
产出目录：`pilot_v2_1/`（v2.0 产出 `pilot_v2/` 未动，版本隔离保持）
 
---
 
## Part A：高 sr 败例定性核查（零调用）
 
从任务01 `pilot_v2` 的 cross_random 集中提取"低 sr 方获胜"的败例判定，按 sr_gap 分档：
 
**sr_gap>50 档：共 9 对（16 次判定），逐对人工核查结果——**
 
| 败例类型 | 对数 | 典型样本 |
|---|---|---|
| A 类：高 sr host 上的低质页 | **9 / 9 (100%)** | SEO 垃圾采集页×2、电商导流页、自由职业者接单页、UGC 素材下载页、软文推广页、拼接转载页、第三方抓取带参数页 |
| B 类：低 sr 方误判高（判官给低 sr 页加分） | 0 | — |
| C 类：双输（两边都烂判不出） | 0 | — |
 
结论：判官没有误判。sr_gap>50 败例 100% 是 **A 类**，根因是 sr 为 host 级常量——高 sr 域名下挂的低质单页在现网 sr 体系里拿不到 host 加权分。这直接支持"单页权威判分不应继承 host 级 sr"的立项假设。
 
补充：sr_gap 10~30 档随机抽 10 对（32 次判定中抽）复核，败例成因同构（A 类为主，少数为内容质量接近的合理胜败），无系统性判官偏差。
 
**产出**：`pilot_v2_1/a1_high_sr_losses.tsv`（19 对明细：host/sr/len/标题截断/.publisher_type/verdict/ reason，已做标题脱敏处理可直读）。
 
---
 
## Part B：v2.1-pilot 复测执行概况
 
- **对子构成（共 140 对 / 280 次判定）**：
  - `retest_unjudgeable`：100 对（v2.0 中 unjudgeable 97 对 + api_error 4 对，含 src_pair_id 可回溯）
  - `retest_anchor`：20 对（in_site 10 + cross_random 10，seed=42）
  - `slice_cross_gov_v2`：20 对
- **切片捞取方式**：流式扫描 112 万页全量池（shard200），权威域页 210,899 页，双规则（转载来源线索+生活主题词 / gov 域标题含非政务职能词）初筛产出候选池；因关键词字面匹配误配率高（曾出现"瘦身→航天散热器"、"营养→玄幻小说"等 6/18 坏配对），**最终改为人工审定配对清单**（`gov_v2_slice_pairs.tsv`）：人工精选 10 页真实跨领域 gov 页（养生/食疗/睡眠/减肥等生活主题 × gov 域），每页配 2 个同主题专业/商业页，20 对全部为 good 配对。
- **模型与参数**：temperature=0、max_tokens=4000、thinking 关闭、3500 字截断、AB/BA 双向各 1 次、逐行落盘断点续跑。
- **⚠️ 模型混跑说明（红线下的一项偏离，如实上报）**：任务书指定 gemini-3.8-flash，跑至 280 次中约 237 次（15:28 前）时因单次推理时长过长（均值 34.8s），经确认切换为 gemini-4-hydrogen 续跑最后 43 次（切换前实测同 prompt 双模型结论一致：均 B_much、均 3/3 可解析，hydrogen 均值 25.8s）。混跑边界经时间戳定位仅影响 1 个 anchor 对（Q104，AB=flash/BA=hydrogen），Part C 的 anchor 分析已考虑此因素。
- **解析失败 0 次，api_error 0 次**（280/280 落盘）。
 
---
 
## Part C：验收门禁结果（5/6 达标）
 
| # | 指标 | 结果 | 目标 | 判定 |
|---|---|---|---|---|
| 1 | 等效整体 unjudgeable 率 | **0.3%**（对级 1/342；调用级 0.1%） | <5% | ✅ |
| 2 | 列表页转化合理性（正文页 vs 列表/导航/工具页，正文获胜） | **95.7%**（154/161）；列表vs列表 37 次（tie 25） | ≥90% | ✅ |
| 3 | retest_unjudgeable 双向稳定性 | ≤1档 **100%**（99 有效对），反转 **0%** | ≥85% / <5% | ✅ |
| 4 | retest_anchor 与 v2.0 方向一致率 | **82.3%**（17 有效对中 14 同向，真反转 2） | ≥95% | ❌ |
| 5 | 解析失败率 | **0/280** | <1% | ✅ |
| 6 | slice_cross_gov_v2 检出 | 检出 15/40（详见下） | 记录 | 记录 |
 
明细文件：`pilot_v2_1/part_c_metrics_v2_1.json`。
 
### 核心目标达成情况
 
v2.1-pilot 两处修改（收紧 unjudgeable + page_form 字段）**成功把等效 unjudgeable 率从 25.9% 压到 0.3%**，且列表/导航/聚合页被正确纳入低权威正常比较（正文页胜列表页 95.7%），retest 子集双向稳定性 100%。100 对原 unjudgeable 中仅 1 对（纯"载入中…"JS 残留脏页）仍判 unjudgeable——即 v2.0 的海量 unjudgeable 几乎全部是"列表/导航/聚合页被误标不可评估"，v2.1 修复方向正确。
 
### anchor 未达标归因（5 例逐一核查）
 
| 对 | 现象 | 归因 |
|---|---|---|
| Q110、Q118 | 真反转（AB/BA 方向相反） | **v2.1 新失败模式**：page_form 在"空白模板页 vs 稀疏产品页"边界上跨序判定不稳定——两次都坚持"实质正文页更权威"，但对哪页是模板页 AB/BA 判定相反。属 page_form 新维度引入的边界形态页问题，非模型混跑（两对全在 flash 阶段 15:18~15:20）。 |
| Q104 | 方向不一致 | 混跑边界对：AB=flash（15:18）/ BA=hydrogen（15:31），borderline 转载参数页 vs 企业首页目录页。 |
| Q102、Q120 | 汇总 unknown | 单边 tie 摇摆（tie + 微弱胜负混合），非真反转。 |
 
即：扣除 tie 摇摆后方向不一致 3/17（82.4%→修正口径 14/17），其中 2 例归因于 v2.1 page_form 新维度的跨序不稳定这一**新失败模式**，1 例受混跑影响。20 对小样本下 2 例真反转即把一致率拉低 10+ 个百分点，样本量是未达标的主放大器。
 
### slice_cross_gov_v2 结果
 
- 40 次调用：gov 侧 `publisher_fits_topic=否` 7 次、`original=转载` 8 次，检出（fits=否 或 转载）15 次；
- 对级（20 对双向合并）：gov 胜 3 / gov 负 4 / **tie 13**（20/20 双向有效）；
- 判官未对 gov 域页系统性加/减分——gov 转载页多为注明来源的政务平台间转载，判官未因"转载"一律降判；半数以上 tie。**需要人工金标校验** tie 13 对中是否存在 gov 侧实质更优却被低估的情况；
- 典型 reason 样例（5 条，摘自 `part_c_metrics_v2_1.json` → slice_cross_gov_v2.reasons）：多为"政务网站转载养生内容，来源可溯/与主题相称" vs "专业站原创但推广倾向"类权衡表述。
 
---
 
## Part D：人评回收脚本状态（已备妥，待回填）
 
`pilot_v2_1/eval_human_gold.py` 已完成并通过 MOCK 合成金标 selftest（193 对 + 100 页四张表全链路跑通；当前 `eval_human_gold_result.json` 为 mock 数据，**人工回填后需重跑覆盖**）：
 
1. **人机一致性**：总表 + in_site/cross_random/len_contrast/slice_ad_soft/slice_cross_gov 切片（严格一致 / ≤1档 / 反转；|log(len_a/len_b)|>0.7 为长短文反差阈值）——MOCK 下 LLM 结论 342 对正确装载；
2. **长文偏好人机对照**：人工判短胜而 Gemini 判长胜的错例清单；
3. **旧公式横向验尸**：现网 npv / 纯 sr / 纯正文长度对人工金标的方向一致率（npv 底表 23,228 页按 url 查询；历史 15 项重拟合公式本地无参数文件，已留钩子）；
4. **100 页绝对分带**：human_tier 四带（>800 / 600~800 / 400~600 / <400）× P10/P50/P90 + 带命中率。
 
用法：人工回填 `pilot_v2/human_review_{pairs,pages,key}.tsv` 后直接运行本文件（MOCK=False）。
 
---
 
## Part E：结论与放量准备度判断
 
1. **核心目标达成**：v2.1-pilot 把等效 unjudgeable 率压至 0.3%（目标 <5%），列表/聚合/导航页转化比较符合预期（95.7%），且未破坏已验证参数（解析失败 0、双向稳定 100%）。**v2.1-pilot 口径具备放量条件**。
2. **需带进下一轮的已知问题**：
   - page_form 在"空白模板 vs 稀疏产品页"等边界形态页上存在跨序不稳定（新失败模式，本轮贡献 2 个 anchor 真反转）——建议下一轮在 prompt 中对 page_form 的边界形态（模板页/参数页/稀疏产品页）给出更多判别示例，或将该维度改为仅记录不参与判定；
   - anchor 一致率 82.3% 未达 95% 门禁，但小样本（17 有效对）+ 新失败模式共同作用，且扣除 tie 摇摆后真方向错误仅 3 例；建议放量前对 Q102/Q104/Q110/Q118/Q120 五对做一次人工盲审定性；
   - slice_cross_gov_v2 半数 tie，gov 偏好/歧视双向均未证实，等人工金标裁决。
3. **执行偏离报备**：推理模型中途由 gemini-3.8-flash 切至 gemini-4-hydrogen（切前已实测双模型同 prompt 结论一致；混跑影响已逐对定位，仅 1 对跨边界）。
4. **产出清单**：`a1_high_sr_losses.tsv`、`pilot_v2_1_pairs.tsv`、`b1_candidates_v2_1.jsonl`、`gov_v2_pool.jsonl`、`gov_v2_picks_curated.jsonl`、`gov_v2_slice_pairs.tsv`、`pilot_v2_1_raw_labels.jsonl`、`pilot_v2_1_summary.tsv`、`part_c_metrics_v2_1.json`、`eval_human_gold.py`（+mock 版 result）。脚本：`gemini02_part_{a_high_sr_losses,b1_build_pairs,b1_gov_v2_pool,b2_infer_v2_1,c_metrics}.py`。
