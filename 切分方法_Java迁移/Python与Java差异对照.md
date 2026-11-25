# Python 与 Java 差异对照

> 迁移目标：Java 版与 Python 原版**逐字一致**。本文档记录所有需要特别处理的语言 / 库差异、处理方式，以及无法消除的已知差异。

---

## 1. 字符串与正则（已全部对齐）

| 差异 | Python | Java 默认 | 处理（见 `PyStr.java`） |
|---|---|---|---|
| 字数 | `len()` 按码点计 | `String.length()` 按 UTF-16 计，emoji、生僻字（😀、𠀀）算 2 | 一律用 `PyStr.len`（码点数）；需要按位置切字符串的地方（分句）先转成码点数组 |
| 空白 | `str.isspace()` 认 29 个字符，含 NBSP `\xa0`、`\x1c–\x1f`、全角空格 | `String.strip()` / `Character.isWhitespace` 不认 NBSP，范围不同 | `PyStr.isSpace` 按 Python 的 29 个字符硬编码；`strip`、"是否为空白"都用它 |
| `\s` | Unicode 空白（同 isspace） | 只有 ASCII 空白 | 换成显式字符集 `PyStr.S` |
| `\d` | Unicode 十进制数字（含全角 `１`） | 只有 `0-9` | 换成 `\p{Nd}`（`PyStr.D`） |
| `$`、`.` | `$` 只在串尾或末尾 `\n` 前；`.` 不匹配 `\n` | 还认 `\r`、` ` 等行结束符 | 所有正则都加 `UNIX_LINES` |
| 忽略大小写 | `re.I` 对 Unicode 生效（`ſtep` 能匹配 `step`） | 只对 ASCII | `CASE_INSENSITIVE \| UNICODE_CASE` |
| `re.match` | 从开头匹配 | `matches()` 要求整串 | 用 `lookingAt()` |
| `str.split(sep)` | 保留末尾空串 | `String.split` 丢掉末尾空串、按正则切 | `PyStr.split`（字面分隔符，保留空串） |
| `str.strip(chars)` | 去两端属于 chars 的字符 | 无对应 | `PyStr.strip(s, chars)` |
| `re.sub(..., r"\1")` | 组引用 | `$1` | 替换串写 `$1` / `$2` |

## 2. BeautifulSoup（lxml）行为复刻（已全部对齐，见 `Soup.java`）

| 行为 | BeautifulSoup | jsoup 默认 | 处理 |
|---|---|---|---|
| 遍历子节点时的"字符串" | `isinstance(c, NavigableString)` 对**注释、doctype、`<?xml?>` 处理指令**也成立，原版会把它们的文字拼进正文 | 注释、doctype 是独立节点类型 | `Soup.navString`：注释取内容；doctype 按 `Doctype.for_name_and_ids` 拼（`<!DOCTYPE html>` → `html`）；`<?xml ...?>`（jsoup 解析成伪注释 `?xml ...?`）去掉开头的 `?` |
| `get_text()` | 只取普通文字：**不含注释**；也不含 `rt` / `rp`（拼音注音）、`script`、`style`、`template` 里的文字（它们是另一种字符串类型） | `text()` 会规范化空白，`wholeText()` 含所有文字 | `Soup.getText`：只收普通文字节点；按"最近的 rt / rp / script / style / template 祖先"判断字符串类型，和 BeautifulSoup 的类型过滤一致 |
| CDATA | lxml 直接丢掉 | 解析成伪注释 `[CDATA[...]]` | 这种注释跳过 |
| `get_text(sep)` | 所有字符串（含纯空白）用 sep 连接 | 无 | `Soup.getText(el, sep)` |
| `find_all` / `find` | 不含自己 | `getElementsByTag` 含自己 | 包装后排除自己 |
| `find_parent(name)` | 最近的同名祖先 | — | `Soup.findParent` |
| `<br>` | 原版替换为文字 `\n` | — | 同样替换为 `TextNode("\n")` |
| 没有 doctype 时的解析模式 | libxml2 没有 quirks 模式，`<p>` 后遇到 `<table>` 会先关闭 p | jsoup 进入 quirks 模式，把 table 放进 p 里 | 没有 doctype 时临时补 `<!DOCTYPE html>` 以标准模式解析，解析后删掉这个 doctype 节点（不进入正文） |
| 自动补 `<tbody>` | lxml 不补 | jsoup 补 | 不影响（取行时按"最近的 table 祖先"判断） |

## 3. 无法消除的已知差异（只出现在残缺 html 上）

根源：lxml 底层的 libxml2 按 **HTML4** 规则修复 html，不认识 `article`、`section`、`aside`、`header`、`footer`、`nav`、`main`、`figure` 等 **HTML5 标签**；jsoup 按 **HTML5 规范**修复。要在 Java 里复刻需要改 jsoup 的建树逻辑，没有公开接口，因此不复刻。

| 情况 | Python（lxml） | Java（jsoup） | 哪个更合理 |
|---|---|---|---|
| **没闭合的 `<p>` 后面紧跟 HTML5 块级标签**（`<p>文字<article>…`） | 不关闭 p，整个 article 被放进 p 里；p 按"一个段落"取文字，**整节的结构被压扁成一段** | 先关闭 p，article 正常解析 | Java |
| **页面开头的裸文本**（`文字<article>…`） | 自动包进 `<p>`，于是同上 | 不包，按普通文字处理 | Java |
| **页面以 `<script>` / `<style>` 开头，后面紧跟 HTML5 标签** | 把后面的标签放进 `<head>`，而 head 在跳过列表里，**整节内容被丢掉** | 正常放在 body 里 | Java |
| 开头裸文本里带注释（`文字<!-- 注释 --><ul>`） | 包进 `<p>` 后，原结构切分用 `get_text()` 取文字，注释被排除 | 注释文字作为字符串节点进入正文 | 差别很小 |

**影响范围**：
- 构造用例（42 个）：全部一致。
- 随机测试页（刻意加入约 3% 的残缺写法）：3000 个 html 页中 t512 / t512_adh 有 19 页、html_container 有 15 页不一致，**全部属于上表**；markdown 3000 页 100% 一致。
- EC 输出的是程序生成的精简 html，标签通常都闭合，预计真实数据上的差异远低于随机页。需要在服务器上用真实网页确认（README 第 4 节）。

已知差异的页面 id 记录在 `testdata/known_diffs.txt`，对照测试时单独计数，不算失败。

## 4. 环境依赖

- **Python 结果依赖 lxml / libxml2 版本**：残缺 html 的修复方式随 libxml2 版本变化（libxml2 2.14 起重写了 html 解析器）。生成标准答案（golden）时应使用与**离线评测相同**的 Python 环境；`make_golden.py` 会把 bs4、lxml 版本写进 golden 第一行。本目录的 golden 由 bs4 4.13.3 + lxml 5.3.1 生成。
- 没装 lxml 时 Python 会退回 `html.parser`，结果和 Java、和离线评测都会不同（`make_golden.py` 会警告）。
- Java 依赖 jsoup 1.21.2（与仓库里 `html-structure-spark` 一致）。升级 jsoup 后需要重新跑对照测试。

## 5. 对照结果汇总

| 数据 | md_t512 | md_t512_adh | html_t512 | html_t512_adh | html_container |
|---|---|---|---|---|---|
| 构造用例 42 个 + 随机页 300+300（`testdata/golden_cases.jsonl.gz`） | 100% | 100% | 100%（1 页已知差异） | 100%（1 页已知差异） | 100%（1 页已知差异） |
| 随机页 3000 + 3000 | 100% | 100% | 99.37% | 99.37% | 99.50% |
| 真实网页（服务器上跑） | 待测 | 待测 | 待测 | 待测 | 待测 |
