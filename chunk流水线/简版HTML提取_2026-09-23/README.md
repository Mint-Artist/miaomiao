# 简版 HTML 提取（原始网页 → 简版 HTML）

2026-09-23。实现"建库前固定提取、切分策略下游可调"两段式架构中的**前半段**：
把原始网页 HTML 转成保留语义结构的简版 HTML 字符串，接口约定见
[EC网页正文标签保留需求_简版.md](../../调研与资料/EC网页正文标签保留需求_简版.md)。
本目录是自己实现的实验版本，用于先跑通效果、再推动上游 EC 对齐。

## 目录

| 文件 | 说明 |
|---|---|
| `simplify_html.py` | 提取器（库 + CLI），含版本号 `EXTRACTOR_VERSION` |
| `run_examples.py` | 对三个样例页面一键跑提取，结果进 `output/` |
| `validate.py` | 验收脚本：结构检查 + 与人工参照稿逐字比对，37 项 |
| `samples/` | 原始 HTML 样本（华为支持页、高考一分一段页、律图知识页） |
| `output/` | 生成的简版 HTML 与 JSON 报告 |

## 用法

```bash
python3 run_examples.py     # 两个样例
python3 validate.py         # 验收，28 项全过退出码为 0

# 单个页面
python3 simplify_html.py --html samples/huawei_zh-cn16079039.raw.html \
    --url "https://consumer.huawei.com/cn/support/content/zh-cn16079039/" \
    --out output/x.html --report x.report.json

# 可选开关
--anchors                   # 块级元素加 data-src="e1..eN" 锚点（评估回溯、父子扩展用）
--promote-pseudo-headings   # 整段加粗短段落升为标题，标 data-origin="inferred"
--profile gk100             # 强制站点配置（默认按 URL 匹配，兜底 generic）
```

依赖：`pip install -r ../../调研与资料/webpage_tools/requirements.txt`（仅 beautifulsoup4）。

## 提取规则（固定、确定性）

1. **定位**：按站点配置取标题选择器、正文容器选择器、元信息选择器；
   generic 配置兜底（article/main/#content 等常见容器）。
2. **剔除**：站点 `strip` 选择器先整块移除已移入 metadata 的信息栏、
   免责声明等样板块；再删除 script/style/iframe/nav/footer 等标签；
   id/class 词元命中噪声表（ad/banner/promo/recommend/share/ymzy…）
   或站点噪声选择器的整块删除。
3. **标签归一**：白名单保留 h1–h6、p、br、ul/ol/li、table 全家、figure、
   blockquote、pre/code、a、img、strong/em；b/i 改名；span 等行内标签拆壳留字；
   div 仅当 class 命中语义关键词（note/warning/info…）时保留并归一化 class，
   否则拆壳；未知标签有内容拆壳、空则删。
4. **属性白名单**：a[href]、img[src/alt]、ol[start/reversed]、li[value]、
   td/th[rowspan/colspan/scope/headers]、div[class 限语义词]；其余全删
   （id/style/on* 一律删除）。有页面 URL 时相对 href/src 绝对化。
5. **表格重建**：统一为 caption/colgroup + thead（全 th 行）+ tbody + tfoot；
   修正原始 HTML 中 thead/tbody 错乱、多 tbody、缺 thead 的情况；
   单元格内 <br> 原样保留（是格内换行，不是新数据行）。
6. **收拢**：根级裸露行内内容包进 <p>；删除无文字且无图/表的空块
   （td/th/tr 不删，承载结构）；正文首块与 h1 同文时去重。
7. **组装**：h1（标题）+ div.metadata（元信息）+ 清理后正文，序列化为
   块级换行缩进、行内连续的 HTML 字符串。

## 与 EC 需求文档的对应及有意偏差

- 对应：标签/属性白名单、div 语义分组、br 只表段落内换行、实体合法编码。
- 偏差 1：**保留 strong/em**。高考页"1、物理类"是加粗伪标题，删掉加粗就丢了
  推断线索；规范只说 span 可去标签，未禁 strong。
- 偏差 2：**标题级别按原文保留**。高考页章节标题原文是 h3，不升为 h2；
  伪标题默认不提升（规范："不新增不存在的小标题"）。需要时开
  `--promote-pseudo-headings`，提升结果带 `data-origin="inferred"` 标记，
  供下游区分原生与推断。
- 偏差 3：可选 `data-src` 锚点是本实验版自加属性，对 EC 的正式交付字符串不含它。

## 样例验收结果（validate.py）

- 华为页：1 标题、2 元信息、2 介绍段、note 框 4 项、三步 ol、4 图 1 链接；
  与 `华为页面精简正文示例.html` 去空白后逐字一致（646 字）。
- 高考页：1 标题、2 元信息、表头 1×4 th + 171×4 td（首行清华 688/85，
  末行苏州大学 631/9977）、单元格内 br 保留；与
  `downloads/gk100_read_27177662/正文精简.html` 逐字一致（3822 字）。
- 律图页（无人工参照稿，做结构与噪声断言）：1 标题、3 元信息
  （来源/时间/阅读量）、三个加粗伪标题按原样保留、23 条法条内链带 href、
  免责声明/网站地图/相关推荐/导流入口等样板噪声已剔除。

## 已知限制

- 两个样本均无 rowspan/colspan 实例，跨行跨列表格只过了代码路径没有真实验收，
  推广前需补这类样本。
- 噪声词元是保守通用集，新站点可能误伤（如正文里出现 share 字样的 class）；
  加站点时先看 report.json 的 `dropped:*` 统计。
- 文本节点首尾空白一律去掉，对中英混排里依赖标签间空格的排版可能粘连；
  当前两个中文样本无此问题。
- 正文容器靠站点配置/启发式选择器，华为正文容器 id 含页面编号
  （`div[id^='body']` 前缀匹配），站点改版时需更新配置并重跑验收。
- 律图页正文末尾保留了编辑撰写的导流段（"点击网页底部的立即咨询按钮…"），
  它是正文写手加的而非模板块，规则剔除易误伤真实内容，留待下游按语义处理。
- 图片只保留 src/alt，无 OCR；缺 alt 会在报告 warnings 里列出。

## 下一步

简版 HTML 是固定产物；chunk 切分策略（256 字预算、结构边界、标题/表头
上下文、父子扩展）在其上另行迭代，对应调研报告的 A–E 对照实验
（`chunk_research_2026-09-20/html_structure_research/HTML结构切分_业界调研.md` 第 9 节）。
提取器修 bug 后按 `EXTRACTOR_VERSION` 定位需重跑的页面，避免全量重建。
