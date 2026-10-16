//> using scala 2.12.20
//> using jvm 17
//> using dep org.jsoup:jsoup:1.23.2
//> using dep org.apache.spark::spark-sql:3.5.9
//> using javaOpt -Xss8m
//> using javaOpt -XX:+IgnoreUnrecognizedVMOptions
//> using javaOpt --add-opens=java.base/java.lang=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.lang.invoke=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.lang.reflect=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.io=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.net=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.nio=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.util=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.nio.cs=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.security.action=ALL-UNNAMED
//> using javaOpt --add-opens=java.base/sun.util.calendar=ALL-UNNAMED

/*
 * HtmlSimplify.scala —— 原始网页 HTML → 只保留正文与核心结构的简版 HTML。
 *
 * 依赖：jsoup（解析 HTML）+ Spark（集群提供）。JSON 读写用 Spark 自带的 Jackson。
 * 上面的 //> using 指令只给 scala-cli 用；--add-opens 只在本地进程内跑 Spark 时需要，spark-submit 会自己加。
 *
 * 输出的简版 HTML：
 *   - 标签只有 h1–h6、p、br、ul/ol/li、table/caption/thead/tbody/tr/th/td、pre、blockquote、a[href]、img[src,alt]、strong、em、code；
 *   - 第一块是 <h1>页面标题</h1>，之后按原文顺序，每个块一行；
 *   - 原文没有用 h 标签、但从加粗 / 字号 / class / 中文编号推断出的小标题，输出为 <hN data-origin="inferred">，
 *     下游可以选择使用或当作普通段落。
 *
 * 处理步骤（simplify）：
 *   1. 解码：pg 多套一层 JSON 字符串时先解开；超长页面直接标记 too_large。
 *   2. 解析：jsoup（HTML5 规范容错解析）。
 *   3. 去掉不可能是正文的节点：脚本、样式、iframe、表单控件、nav/aside/footer、注释、行内声明为隐藏的短元素。
 *      注意不删除 <form>：很多 .aspx 站点把整页包在一个 form 里。
 *   4. 定位正文：站点规则 → 打分 → 退回 body。
 *      打分是"子树净得分"：每个文本块按像正文（有中文标点、链接少）给正分，像样板（导航、链接、广告卡片、
 *      无标点短行）给负分，表格单元格和代码块按字数给正分；节点得分 = 子树内全部文本块之和，取最高者。
 *      再向上 2 层合并得分够高、链接少的旁支（被拆成几块的正文、多个回答）。找不到任何正分时退回整个 body。
 *   5. 按白名单重建一棵新树（不在原树上删改）：div 等容器拆壳但作为段落边界；连续 <br><br> 或"<br>+全角缩进"切分段落；
 *      布局表格拆成段落，数据表格规整为 thead/tbody；链接密集的列表 / 区块和 class/id 命中噪声词的区块整块丢弃；
 *      懒加载图片取 data-src / data-original；链接、图片地址转成绝对地址。
 *   6. 去样板（界面按钮文字、版权 / 平台声明、"相关阅读"之后的推荐、首尾面包屑与"上一篇"等），
 *      推断小标题（规则见 promoteHeadings），标题放在第一块。
 *   7. 确定性序列化（同一输入永远得到同一输出），并输出统计字段，便于抽查。
 *
 * 用法：
 *   本地（不需要 Spark 集群，JSONL 进、JSONL 出）：
 *     scala-cli run HtmlSimplify.scala -- --local --input pages.jsonl --output out.jsonl
 *   集群：先打包（Spark 不打进包里），再提交：
 *     scala-cli --power package HtmlSimplify.scala --assembly --preamble=false --provided org.apache.spark::spark-sql -o html-simplify.jar
 *     spark-submit --class HtmlSimplify html-simplify.jar --input hdfs:///path/in --output hdfs:///path/out \
 *       [--input-format json|parquet|orc] [--url-col url] [--html-col pg] [--partitions 2000] \
 *       [--output-format parquet|json] [--mode overwrite] [--with-text] [--no-plain-headings] [--site-rules hdfs:///conf/site_rules.json]
 *   本地进程内跑 Spark（验证集群路径用）：再加 --master local[2]。
 *   调试某几页为什么抽成这样（本地模式）：--explain 页面id1,页面id2（JSONL 里要有 id 字段），会打印候选容器得分。
 *
 * 站点规则（可选，JSON）：{"example.com": {"content": ["div.article-body"], "remove": [".share-bar"]}}
 * 按域名后缀匹配；content 中所有选择器命中的元素（按文档顺序）作为正文，跳过打分。
 * 代码里内置了百度知道、搜狗问问两条（见 SiteRules.defaults），外部规则同名时覆盖。
 *
 * Scala 2.12 / 2.13 均可编译；集群若是 Scala 2.13 的 Spark，把第一行改为 2.13.x。
 */

import java.io.{File, PrintWriter}
import java.util.{IdentityHashMap => IdMap}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

import com.fasterxml.jackson.databind.ObjectMapper
import org.jsoup.Jsoup
import org.jsoup.nodes.{Comment, Document, Element, Node, TextNode}
import org.jsoup.select.{NodeTraversor, NodeVisitor}

object HtmlSimplify {

  val Version = "1.0.0"
  val MaxInputChars = 5000000

  final case class Input(url: String, pg: String)

  final case class Result(url: String, status: String, error: String, title: String, html: String, text: String,
                          textChars: Int, locate: String, headings: Int, inferredHeadings: Int, tables: Int,
                          lists: Int, images: Int, inputChars: Int, millis: Long, version: String)

  // ---------------------------------------------------------------- 词表与正则

  private val DropTags = Set("script", "style", "noscript", "template", "svg", "canvas", "iframe", "frame", "frameset",
    "object", "embed", "video", "audio", "source", "track", "map", "area", "input", "select", "textarea", "button",
    "option", "datalist", "link", "meta", "base", "nav", "aside", "footer", "dialog", "menu")
  private val InlineTags = Set("a", "abbr", "b", "bdi", "bdo", "big", "cite", "code", "data", "del", "dfn", "em", "font",
    "i", "ins", "kbd", "label", "mark", "nobr", "q", "s", "samp", "small", "span", "strike", "strong", "sub", "sup",
    "time", "tt", "u", "var", "wbr", "img", "br")
  private val HeadingTags = Set("h1", "h2", "h3", "h4", "h5", "h6")
  private val ParaTags = Set("p", "dt", "dd", "caption", "figcaption", "summary", "legend", "address")
  /** 直接含文字时，文字算在自己身上（而不是父节点）的容器标签 */
  private val ContainerTags = Set("div", "section", "article", "main", "body", "center", "form", "td", "font", "span")

  private val NoiseTokens = Set("ad", "ads", "adv", "advert", "advertisement", "banner", "promo", "promotion", "recommend",
    "recommended", "related", "relate", "share", "sharing", "social", "comment", "comments", "breadcrumb", "breadcrumbs",
    "crumb", "crumbs", "sidebar", "nav", "navbar", "menu", "footer", "login", "copyright", "toolbar", "qrcode", "ewm",
    "rank", "ranking", "hot", "hotnews", "pagination", "pager", "tags", "subscribe", "reward", "dashang")
  private val NegTokens = NoiseTokens ++ Set("header", "top", "bottom", "logo", "links", "friend", "about", "contact",
    "download", "app", "side", "list")
  private val PosTokens = Set("article", "content", "contents", "main", "post", "detail", "details", "entry", "text",
    "txt", "story", "zhengwen", "zw", "con", "cont", "trs", "editor", "neirong", "wenzhang", "body")
  private val TitleTokens = Set("title", "subtitle", "tit", "heading", "subheading", "biaoti", "bt", "h1", "h2", "h3",
    "h4", "h5", "h6")

  private val Ws = "[\\s\\u00a0\\u3000\\u2028\\u2029]+".r
  private val ZeroWidth = "[\\u200b-\\u200d\\u2060\\ufeff]".r
  private val CjkClass = "\\u3000-\\u303f\\u3400-\\u4dbf\\u4e00-\\u9fff\\uff00-\\uffef"
  private val CjkSpace = s"(?<=[$CjkClass]) (?=[$CjkClass])".r
  private val Punct = "，。；：！？、,;!?"
  private val HiddenStyle = "(?i)display\\s*:\\s*none|visibility\\s*:\\s*hidden".r
  private val BoldStyle = "(?i)font-weight\\s*:\\s*(bold|bolder|[6-9]00)".r
  private val FontSize = "(?i)font-size\\s*:\\s*([\\d.]+)\\s*(px|pt|em|rem|%)?".r
  private val FontSizeWord = "(?i)font-size\\s*:\\s*(large|x-large|xx-large|larger)".r
  private val EndPunct = "[。！？!?；;，,、…]$".r
  private val Num0 = "^(第[一二三四五六七八九十百零〇\\d]+[章节篇部分条]|[一二三四五六七八九十]+[、．.：:])".r
  private val Num1 = "^[（(][一二三四五六七八九十]+[)）]".r
  private val Num2 = "^\\d{1,2}[、．.：:](?!\\d)".r
  private val Num3 = "^([（(]\\d{1,2}[)）]|\\d{1,2}[)）])".r
  private val Boiler = ("^(上一篇|下一篇|上一页|下一页|【?责任编辑|【?编辑[:：]|免责声明|版权(所有|声明)|Copyright|©|扫一扫|扫码|分享到|" +
    "相关(阅读|推荐|文章|新闻)|热门(推荐|文章)|猜你喜欢|【?打印|【?关闭|返回顶部|举报|当前位置|您的位置|你的位置)").r
  private val UiLine = ("^(大字体|中字体|小字体|字号[:：].{0,10}|展开全部|展开全文|阅读全文|展开|收起|评论|点赞|收藏|分享|举报|提交|取消|关注|" +
    "已关注|×|上一页|下一页|返回|返回顶部|打印|关闭|顶|踩|赞|回复|复制链接|微信扫一扫|扫码分享)$").r
  private val Related = ("^[【\\[]?(相关(阅读|推荐|文章|新闻|链接|资讯|问题|内容)|热门(推荐|文章|阅读|问题)|推荐阅读|猜你喜欢|为你推荐|延伸阅读|" +
    "更多精彩|热点推荐|精彩推荐|往期回顾|往期推荐|大家都在看|看了又看|其他人还看了)").r
  private val Notice = "(©|著作权归|版权(所有|归|声明)|平台声明|免责声明|严正声明|【社区内容提示】)".r
  private val DateLike = "^\\s*(\\d{4}[-/.]\\d{1,2}[-/.]\\d{1,2}|\\d{1,2}:\\d{2}:\\d{2})".r // 以日期 / 时间开头
  private val NotHeading = "(来源|作者|编辑|发布|时间|日期|浏览|阅读|点击|字号|http|www\\.)".r

  // ---------------------------------------------------------------- 小工具

  private lazy val json = new ObjectMapper()
  /** 仅本地调试（--explain）时非空：打印候选容器得分。 */
  @volatile private var explainOut: java.io.PrintStream = null

  private def describe(e: Element): String = {
    val path = (e :: e.parents.asScala.toList.take(3)).reverse.map { x =>
      x.normalName + (if (x.id.nonEmpty) "#" + x.id else "") + (if (x.className.nonEmpty) "." + x.className.trim.replaceAll("\\s+", ".") else "")
    }
    path.mkString(" > ").take(160)
  }

  private def nonWs(s: String): Int = { var n = 0; var i = 0; while (i < s.length) { if (!isSpace(s.charAt(i))) n += 1; i += 1 }; n }
  private def isSpace(c: Char): Boolean = Character.isWhitespace(c) || c == ' ' || c == '　'
  private def isCjk(c: Char): Boolean = (c >= '　' && c <= '〿') || (c >= '㐀' && c <= '䶿') ||
    (c >= '一' && c <= '鿿') || (c >= '＀' && c <= '￯')
  private def norm(s: String): String = Ws.replaceAllIn(s, "")
  private def collapse(s: String): String = Ws.replaceAllIn(ZeroWidth.replaceAllIn(s, ""), " ")
  private def tokens(e: Element): Set[String] =
    (e.className + " " + e.id).replaceAll("([a-z0-9])([A-Z])", "$1 $2").toLowerCase.split("[^a-z0-9]+").filter(_.nonEmpty).toSet
  private def isInline(e: Element): Boolean = InlineTags(e.normalName) || e.normalName.contains(":")

  private def decodePg(pg: String): String = {
    var s = if (pg == null) "" else pg
    var k = 0
    while (k < 2 && s.length >= 2 && s.charAt(0) == '"') { // 多套了一层（或两层）JSON 字符串
      try s = json.readValue(s, classOf[String]) catch { case _: Exception => k = 2 }
      k += 1
    }
    if (s.nonEmpty && s.charAt(0) == '﻿') s.substring(1) else s
  }

  // ---------------------------------------------------------------- 站点规则

  final class SiteRules(val rules: Map[String, (Seq[String], Seq[String])]) extends Serializable {
    def lookup(url: String): Option[(Seq[String], Seq[String])] = {
      val host = try new java.net.URI(url).getHost catch { case _: Exception => null }
      if (host == null) None
      else rules.collectFirst { case (k, v) if host == k || host.endsWith("." + k) => v }
    }
  }

  object SiteRules {
    /** 内置规则：中文搜索里流量大、且多个回答分散在多层嵌套里、通用打分难以稳定处理的问答站。外部规则同名时覆盖。 */
    val defaults: Map[String, (Seq[String], Seq[String])] = Map(
      "zhidao.baidu.com" -> (Seq(".wgt-best", "#wgt-answers"), Nil),
      "wenwen.sogou.com" -> (Seq("#bestAnswers", ".replay-info-txt"), Nil))
    val empty = new SiteRules(defaults)
    def parse(text: String): SiteRules =
      if (text == null || text.trim.isEmpty) empty
      else {
        val root = json.readTree(text)
        val m = root.fields.asScala.map { e =>
          val v = e.getValue
          def list(k: String) = if (v.has(k)) v.get(k).elements.asScala.map(_.asText).toList else Nil
          e.getKey -> (list("content"), list("remove"))
        }.toMap
        new SiteRules(defaults ++ m)
      }
  }

  // ---------------------------------------------------------------- 统计（一次后序遍历，O(节点数)）

  private final class St {
    var text = 0; var link = 0; var punct = 0; var links = 0
    var inl = 0; var inlLink = 0; var inlPunct = 0 // 直接文字 + 行内子元素的文字
    var blockDesc = false
  }

  private def countText(s: String, st: St, inline: Boolean): Unit = {
    var n = 0; var p = 0; var i = 0
    while (i < s.length) { val c = s.charAt(i); if (!isSpace(c)) { n += 1; if (Punct.indexOf(c) >= 0) p += 1 }; i += 1 }
    st.text += n; st.punct += p
    if (inline) { st.inl += n; st.inlPunct += p }
  }

  private def computeStats(root: Element): IdMap[Element, St] = {
    val m = new IdMap[Element, St]()
    NodeTraversor.traverse(new NodeVisitor {
      def head(n: Node, depth: Int): Unit = ()
      override def tail(n: Node, depth: Int): Unit = n match {
        case e: Element =>
          val s = new St
          for (c <- e.childNodes.asScala) c match {
            case t: TextNode => countText(t.getWholeText, s, inline = true)
            case ce: Element =>
              val cs = m.get(ce)
              if (cs != null) {
                s.text += cs.text; s.link += cs.link; s.punct += cs.punct; s.links += cs.links
                if (isInline(ce) && !cs.blockDesc) { s.inl += cs.text; s.inlLink += cs.link; s.inlPunct += cs.punct }
                else s.blockDesc = true
              }
            case _ =>
          }
          if (e.normalName == "a") { s.link = s.text; s.links += 1; s.inlLink = s.inl }
          m.put(e, s)
        case _ =>
      }
    }, root)
    m
  }

  // ---------------------------------------------------------------- 步骤 3：去掉不可能是正文的节点

  private def removeJunk(doc: Document): Unit = {
    doc.select(DropTags.mkString(",")).remove()
    val comments = ArrayBuffer[Node]()
    NodeTraversor.traverse(new NodeVisitor {
      def head(n: Node, depth: Int): Unit = if (n.isInstanceOf[Comment]) comments += n
    }, doc)
    comments.foreach(c => if (c.parent != null) c.remove())
    for (e <- doc.select("[hidden], [aria-hidden=true], [style]").asScala.toList) {
      val hidden = e.hasAttr("hidden") || e.attr("aria-hidden") == "true" || HiddenStyle.findFirstIn(e.attr("style")).isDefined
      // 折叠起来的正文（"展开全文"）往往也是 display:none，长文本不删
      if (hidden && e.parent != null && nonWs(e.text) < 300) e.remove()
    }
  }

  // ---------------------------------------------------------------- 步骤 4：定位正文

  private def linkDensity(st: IdMap[Element, St], e: Element): Double = {
    val s = st.get(e); if (s == null || s.text == 0) 0.0 else s.link.toDouble / s.text
  }

  private def locate(doc: Document, st: IdMap[Element, St], rules: SiteRules, url: String): (Seq[Element], String) = {
    val body = doc.body
    rules.lookup(url).foreach { case (content, remove) =>
      remove.foreach(sel => doc.select(sel).remove())
      // 所有选择器命中的并集，按文档顺序；去掉被其他命中包含的元素
      val hits = if (content.isEmpty) Nil else doc.select(content.mkString(", ")).asScala.toList.filter(e => nonWs(e.text) > 0)
      val set = new IdMap[Element, java.lang.Boolean](); hits.foreach(set.put(_, true))
      val top = hits.filterNot(e => e.parents.asScala.exists(set.containsKey))
      if (top.nonEmpty) return (top, "site")
    }
    // 子树净得分：每个文本块按"像正文 / 像样板"给正负分，节点得分 = 子树内所有文本块之和，取最高者。
    // 内容分散在多层嵌套里时能汇总到共同祖先；导航、链接列表、广告卡片为负分，含它们越多的祖先越不容易被选中。
    val acc = new IdMap[Element, Array[Double]]()
    for (e <- body.getAllElements.asScala) {
      val s = st.get(e)
      if (s != null && !isInline(e) && s.inl > 0) {
        var insideLink = false; var inCell = false; var neg = tokens(e).exists(NegTokens)
        var a = e.parent; var k = 0
        while (a != null && k < 60) {
          val an = a.normalName
          if (an == "a") insideLink = true
          if (an == "td" || an == "th") inCell = true
          if (!neg && tokens(a).exists(NegTokens)) neg = true
          a = a.parent; k += 1
        }
        val chars = s.inl
        val lr = if (insideLink) 1.0 else s.inlLink.toDouble / chars
        val cell = inCell || e.normalName == "td" || e.normalName == "th" || e.normalName == "pre" // 代码块、表格单元格不要求有标点
        val w =
          if (cell) (if (lr < 0.5) chars / 60.0 * (if (neg) 0.2 else 1.0) else 0.0) // 数据表格里的链接单元格不扣分
          else if (lr < 0.5 && chars >= 10 && s.inlPunct >= 1)
            (math.min(chars, 500) / 40.0 + math.min(s.inlPunct, 10) * 0.3) * (1 - lr) * (if (neg) 0.2 else 1.0)
          else -(math.min(chars, 300) / 60.0 + (if (lr >= 0.5) 0.5 else 0.1))
        var x = e; k = 0
        while (x != null && k < 60) {
          val v = acc.get(x); if (v == null) acc.put(x, Array(w)) else v(0) += w
          x = x.parent; k += 1
        }
      }
    }
    var best: Element = null; var bestScore = Double.NegativeInfinity; var bestDepth = -1
    val finals = new IdMap[Element, java.lang.Double]()
    for ((e, a) <- acc.asScala if e.normalName != "html") {
      val f = a(0) + (if (tokens(e).exists(PosTokens)) 0.5 else 0.0)
      finals.put(e, f)
      val d = e.parents.size
      if (f > bestScore + 1e-9 || (math.abs(f - bestScore) <= 1e-9 && d > bestDepth)) { best = e; bestScore = f; bestDepth = d }
    }
    if (explainOut != null) {
      val top = finals.asScala.toSeq.sortBy(-_._2.doubleValue).take(8)
      top.foreach { case (e, f) =>
        explainOut.println(f"  ${f.doubleValue}%8.1f  ld ${linkDensity(st, e)}%.2f  ${describe(e)}  text ${st.get(e).text}")
      }
    }
    if (best == null || bestScore <= 0.5) return (Seq(body), "body") // 找不到任何"像正文"的块
    // 选中的只是单个段落（页面很短）时，父容器得分不低于它的一半就改选父容器，把旁边的短段落一起带上
    if ((ParaTags(best.normalName) || HeadingTags(best.normalName) || best.normalName == "pre") && best.parent != null) {
      val pf = finals.get(best.parent)
      if (pf != null && pf.doubleValue >= 0.5 * bestScore) { best = best.parent; bestScore = pf.doubleValue }
    }
    // 选中的是表格 / 列表内部节点时，取整个表格 / 列表
    var lift = best
    while (lift.parent != null && Set("tbody", "thead", "tfoot", "tr", "td", "th", "li")(lift.normalName)) lift = lift.parent
    if (Set("table", "ul", "ol")(lift.normalName)) best = lift
    if (best eq body) return (Seq(best), "score")
    // 正文容器及其向上 2 层祖先的旁支：得分够高、链接少的一并带上（多个回答、被拆成几块的正文）；
    // 第一层还带上紧挨在前面的 h1–h3。
    val threshold = math.max(0.6, bestScore * 0.15)
    def good(c: Element) = {
      val f = finals.get(c); val s = st.get(c); val ld = linkDensity(st, c)
      (f != null && f.doubleValue >= threshold && ld < 0.3) ||
        (s != null && c.normalName == "p" && s.inl >= 80 && ld < 0.25 && s.inlPunct >= 1)
    }
    var picked: Seq[Element] = Seq(best)
    var node = best; var level = 0
    while (level < 2 && node.parent != null && !(node.parent eq doc)) {
      val par = node.parent
      val sibs = par.children.asScala.toIndexedSeq
      val idx = sibs.indexWhere(_ eq node)
      val before = sibs.indices.filter(i => i < idx && (good(sibs(i)) ||
        (level == 0 && i >= idx - 2 && Set("h1", "h2", "h3")(sibs(i).normalName)))).map(sibs)
      val after = sibs.drop(idx + 1).filter(good)
      picked = before ++ picked ++ after
      if (par eq body) level = 2
      node = par; level += 1
    }
    (picked, "score")
  }

  // ---------------------------------------------------------------- 步骤 5：按白名单重建

  /** 行内上下文：是否加粗、字号（px，0 = 未知）、是否在含标题词的 class 下。 */
  private final case class Ctx(bold: Boolean, size: Double, cls: Boolean)
  private val NoCtx = Ctx(bold = false, size = 0.0, cls = false)
  private val FontAttrPx = Map("1" -> 10.0, "2" -> 13.0, "3" -> 16.0, "4" -> 18.0, "5" -> 24.0, "6" -> 32.0, "7" -> 48.0)

  /** 元素自身声明的字号（换算成 px）；相对单位按父级字号（未知时按 16px）换算。 */
  private def fontPx(e: Element, parent: Double): Option[Double] = {
    val base = if (parent > 0) parent else 16.0
    val style = e.attr("style")
    FontSize.findFirstMatchIn(style).flatMap { m =>
      val v = try m.group(1).toDouble catch { case _: Exception => 0.0 }
      if (v <= 0) None
      else Some(Option(m.group(2)).map(_.toLowerCase).getOrElse("px") match {
        case "px" => v; case "pt" => v * 4 / 3; case "%" => base * v / 100; case "rem" => 16 * v; case _ => base * v
      })
    }.orElse(FontSizeWord.findFirstMatchIn(style).map(_.group(1).toLowerCase match {
      case "large" => 18.0; case "x-large" => 24.0; case "xx-large" => 32.0; case _ => base * 1.2
    })).orElse(
      if (e.normalName == "big") Some(base * 1.2)
      else if (e.normalName == "font" && e.hasAttr("size")) {
        val a = e.attr("size").trim
        FontAttrPx.get(a).orElse(if (a.matches("\\+[1-4]")) Some(16.0 + 4 * a.drop(1).toInt) else None)
      } else None)
  }

  private def ctxOf(e: Element, c: Ctx): Ctx = {
    val n = e.normalName
    val bold = c.bold || n == "b" || n == "strong" || BoldStyle.findFirstIn(e.attr("style")).isDefined
    val size = fontPx(e, c.size).getOrElse(c.size)
    val cls = c.cls || tokens(e).exists(TitleTokens)
    if (bold == c.bold && size == c.size && cls == c.cls) c else Ctx(bold, size, cls)
  }

  private final class Block(val el: Element, val para: Boolean) {
    var chars = 0; var bold = 0; var cls = 0
    val sizes = scala.collection.mutable.HashMap[Int, Int]() // 字号（px，0 = 未知）→ 字数
  }

  private final class Builder(st: IdMap[Element, St], totalText: Int) {
    val blocks = ArrayBuffer[Block]()
    var tables = 0; var lists = 0; var images = 0
    private var cur: Block = null
    private var pendingBr = 0

    private def para(): Block = { if (cur == null) cur = new Block(new Element("p"), true); cur }

    def flush(): Unit = {
      if (cur != null) { tidy(cur.el); if (hasContent(cur.el)) blocks += cur }
      cur = null; pendingBr = 0
    }

    private def addBlock(b: Block): Unit = { flush(); tidy(b.el); if (hasContent(b.el)) blocks += b }

    private def count(b: Block, s: String, c: Ctx): Unit = if (b != null) {
      val n = nonWs(s)
      b.chars += n
      if (c.bold) b.bold += n
      val k = math.round(c.size).toInt; b.sizes(k) = b.sizes.getOrElse(k, 0) + n
      if (c.cls) b.cls += n
    }

    /** <br> 之后来了新内容：连续两个以上 <br>，或新内容以全角 / 两个以上空格缩进开头，视为新段落；否则保留为段内换行。 */
    private def resolveBr(nextRaw: String): Unit = if (pendingBr > 0) {
      val lead = nextRaw.replaceFirst("^[\\r\\n\\t ]*", "")
      val indent = lead.startsWith("　") || lead.startsWith("  ") || nextRaw.matches("(?s)^[\\r\\n]*[ \\t]{2,}\\S.*")
      if (cur != null && nonWs(cur.el.text) > 0) {
        if (pendingBr >= 2 || indent) flush() else cur.el.appendElement("br")
      }
      pendingBr = 0
    }

    private def isNoise(e: Element): Boolean = {
      if (isInline(e)) return false
      val s = st.get(e)
      val small = s == null || s.text < math.max(200, totalText / 2)
      val dataTable = e.normalName == "table" && !e.select("th").isEmpty
      small && (tokens(e).exists(NoiseTokens) ||
        (s != null && s.links >= 3 && s.link > 0.6 * s.text && e.normalName != "p" && !dataTable))
    }

    def walkRoot(e: Element): Unit = {
      if (e.normalName == "body" || ContainerTags(e.normalName) && !HeadingTags(e.normalName)) {
        e.childNodes.asScala.toList.foreach(walk(_, NoCtx, 1)); flush()
      } else { walk(e, NoCtx, 0); flush() }
    }

    def walk(n: Node, c: Ctx, depth: Int): Unit = n match {
      case t: TextNode =>
        val raw = t.getWholeText
        if (nonWs(raw) > 0) { resolveBr(raw); val s = collapse(raw); val b = para(); b.el.appendText(s); count(b, s, c) }
        else if (cur != null && raw.nonEmpty) cur.el.appendText(" ")
      case e: Element if isNoise(e) => ()
      case e: Element if depth > 300 => // 极深的嵌套：直接取文字
        val s = collapse(e.text); if (s.trim.nonEmpty) { val b = para(); b.el.appendText(" " + s); count(b, s, c) }
      case e: Element =>
        val name = e.normalName
        val cc = ctxOf(e, c)
        name match {
          case "br" => if (cur != null) pendingBr += 1
          case h if HeadingTags(h) =>
            flush(); val el = new Element(h)
            e.childNodes.asScala.toList.foreach(inline(_, el, NoCtx, null, depth + 1))
            addBlock(new Block(el, false))
          case "ul" | "ol" =>
            flush(); val l = buildList(e, depth)
            if (l != null) { lists += 1; addBlock(new Block(l, false)) }
          case "table" =>
            if (isLayoutTable(e)) container(e, cc, depth)
            else { flush(); val t = buildTable(e, depth); if (t != null) { tables += 1; addBlock(new Block(t, false)) } }
          case "pre" =>
            flush(); val el = new Element("pre"); el.appendText(e.wholeText); addBlock(new Block(el, false))
          case "blockquote" =>
            flush()
            val sub = new Builder(st, totalText)
            e.childNodes.asScala.toList.foreach(sub.walk(_, cc, depth + 1)); sub.flush()
            tables += sub.tables; lists += sub.lists; images += sub.images
            if (sub.blocks.nonEmpty) { val q = new Element("blockquote"); sub.blocks.foreach(b => q.appendChild(b.el)); addBlock(new Block(q, false)) }
          case "img" =>
            resolveBr(""); image(e, para().el)
          case p if ParaTags(p) =>
            flush(); e.childNodes.asScala.toList.foreach(walk(_, cc, depth + 1)); flush()
          case _ if isInline(e) && !Option(st.get(e)).exists(_.blockDesc) =>
            resolveBr(firstRawText(e)); val b = para(); inline(e, b.el, c, b, depth)
          case _ => container(e, cc, depth)
        }
      case _ =>
    }

    private def container(e: Element, c: Ctx, depth: Int): Unit = {
      flush(); e.childNodes.asScala.toList.foreach(walk(_, c, depth + 1)); flush()
    }

    /** 把源节点的行内内容写进 target；owner 用于累计字数与加粗 / 字号 / class 信号（可为 null）。 */
    def inline(n: Node, target: Element, c: Ctx, owner: Block, depth: Int): Unit = n match {
      case t: TextNode =>
        val s = collapse(t.getWholeText)
        if (s.nonEmpty) { target.appendText(s); count(owner, s, c) }
      case e: Element if depth > 300 => target.appendText(" " + collapse(e.text))
      case e: Element =>
        val cc = ctxOf(e, c)
        def kids(dst: Element): Unit = e.childNodes.asScala.toList.foreach(inline(_, dst, cc, owner, depth + 1))
        e.normalName match {
          case "br" => if (target.childNodeSize > 0) target.appendElement("br")
          case "img" => image(e, target)
          case "a" =>
            val href = e.absUrl("href")
            if (href.nonEmpty && !href.toLowerCase.startsWith("javascript:")) kids(target.appendElement("a").attr("href", href))
            else kids(target)
          case "b" | "strong" => kids(target.appendElement("strong"))
          case "i" | "em" => kids(target.appendElement("em"))
          case "code" => kids(target.appendElement("code"))
          case "ul" | "ol" if target.normalName == "li" =>
            val l = buildList(e, depth); if (l != null) { lists += 1; target.appendChild(l) }
          case other if isInline(e) => kids(target)
          case _ => // 行内上下文里出现的块元素（单元格 / 列表项里的段落等）：用 <br> 隔开
            if (isNoise(e)) ()
            else {
              if (target.childNodeSize > 0) target.appendElement("br")
              kids(target)
              if (target.childNodeSize > 0) target.appendElement("br")
            }
        }
      case _ =>
    }

    private def image(e: Element, target: Element): Unit = {
      val src = Seq("data-original", "data-src", "data-lazy-src", "data-actualsrc", "src").iterator
        .map(a => if (e.hasAttr(a)) e.absUrl(a) else "").find(s => s.nonEmpty && !s.startsWith("data:")).getOrElse("")
      if (src.nonEmpty) {
        val img = target.appendElement("img").attr("src", src)
        val alt = collapse(e.attr("alt")).trim
        if (alt.nonEmpty) img.attr("alt", alt)
        images += 1
      }
    }

    private def buildList(e: Element, depth: Int): Element = {
      val l = new Element(e.normalName)
      if (e.normalName == "ol" && e.attr("start").matches("\\d+")) l.attr("start", e.attr("start"))
      var stray: Element = null
      for (c <- e.childNodes.asScala.toList) c match {
        case li: Element if li.normalName == "li" =>
          stray = null; val o = l.appendElement("li")
          li.childNodes.asScala.toList.foreach(inline(_, o, NoCtx, null, depth + 1)); tidy(o)
          if (!hasContent(o)) o.remove()
        case other => // ul 下直接出现的文字 / 元素：并进一个列表项
          if (stray == null) stray = l.appendElement("li")
          inline(other, stray, NoCtx, null, depth + 1)
      }
      l.children.asScala.toList.foreach { li => tidy(li); if (!hasContent(li)) li.remove() }
      if (l.children.isEmpty) null else l
    }

    private def rowsOf(t: Element): Seq[(Element, Boolean)] = t.children.asScala.toList.flatMap { c =>
      c.normalName match {
        case "tr" => Seq(c -> false)
        case "thead" => c.children.asScala.filter(_.normalName == "tr").map(_ -> true)
        case "tbody" | "tfoot" => c.children.asScala.filter(_.normalName == "tr").map(_ -> false)
        case _ => Nil
      }
    }

    private def cellsOf(r: Element): Seq[Element] = r.children.asScala.filter(c => c.normalName == "td" || c.normalName == "th")

    /** 布局表格：嵌套表格、只有一列、或单元格里是大段块级内容且没有表头。 */
    private def isLayoutTable(t: Element): Boolean = {
      val rows = rowsOf(t)
      if (rows.isEmpty) return true
      val cells = rows.flatMap(r => cellsOf(r._1))
      if (cells.isEmpty) return true
      val maxCells = rows.map(r => cellsOf(r._1).size).max
      val hasTh = cells.exists(_.normalName == "th") || rows.exists(_._2)
      val nested = t.select("table").size > 1
      val blocky = cells.exists(c => Option(st.get(c)).exists(s => s.blockDesc && s.text > 100))
      val avgLen = cells.map(c => Option(st.get(c)).map(_.text).getOrElse(0)).sum.toDouble / cells.size
      nested || maxCells <= 1 || (!hasTh && (blocky || avgLen > 150))
    }

    private def buildTable(t: Element, depth: Int): Element = {
      val table = new Element("table")
      t.children.asScala.find(_.normalName == "caption").foreach { cap =>
        val s = collapse(cap.text).trim; if (s.nonEmpty) table.appendElement("caption").text(s)
      }
      val thead = new Element("thead"); val tbody = new Element("tbody")
      var headerPhase = true
      for ((r, inHead) <- rowsOf(t)) {
        val cs = cellsOf(r)
        if (cs.nonEmpty) {
          val tr = new Element("tr")
          for (c <- cs) {
            val o = tr.appendElement(c.normalName)
            for (a <- Seq("colspan", "rowspan")) if (c.attr(a).matches("[2-9]|\\d{2,3}")) o.attr(a, c.attr(a))
            c.childNodes.asScala.toList.foreach(inline(_, o, NoCtx, null, depth + 1)); tidy(o)
          }
          if (hasContent(tr)) {
            val isHead = headerPhase && (inHead || cs.forall(_.normalName == "th"))
            if (isHead) thead.appendChild(tr) else { headerPhase = false; tbody.appendChild(tr) }
          }
        }
      }
      if (thead.childNodeSize == 0 && tbody.childNodeSize == 0) return null
      if (thead.childNodeSize > 0) table.appendChild(thead)
      if (tbody.childNodeSize > 0) table.appendChild(tbody)
      table
    }
  }

  private def firstRawText(e: Element): String = {
    var out: String = null
    NodeTraversor.traverse(new NodeVisitor {
      def head(n: Node, depth: Int): Unit = n match {
        case t: TextNode if out == null && nonWs(t.getWholeText) > 0 => out = t.getWholeText
        case _ =>
      }
    }, e)
    if (out == null) "" else out
  }

  private def hasContent(e: Element): Boolean = nonWs(e.text) > 0 || !e.select("img").isEmpty

  /** 块内空白整理：去掉空的行内包装；中文之间的空格、行首 / 行尾空格、首尾和重复的 <br>。 */
  private def tidy(el: Element): Unit = {
    if (el.normalName == "pre") return
    var changed = true
    while (changed) {
      changed = false
      for (x <- el.select("strong, em, a, code").asScala.toList)
        if (x.parent != null && nonWs(x.text) == 0 && x.select("img").isEmpty) { x.unwrap(); changed = true }
    }
    val seq = ArrayBuffer[Node]()
    NodeTraversor.traverse(new NodeVisitor {
      def head(n: Node, depth: Int): Unit = n match {
        case t: TextNode => seq += t
        case e: Element if e.normalName == "br" || e.normalName == "img" || (e ne el) && !isInline(e) => seq += e
        case _ =>
      }
    }, el)
    var prev = '\n'
    for (n <- seq) n match {
      case t: TextNode =>
        var s = CjkSpace.replaceAllIn(t.getWholeText, "")
        if (s.startsWith(" ") && (prev == '\n' || prev == ' ' || (isCjk(prev) && s.length > 1 && isCjk(s.charAt(1))))) s = s.substring(1)
        if (s.startsWith(" ") && s.length == 1) s = if (prev == '\n') "" else s
        t.text(s)
        if (s.nonEmpty) prev = s.charAt(s.length - 1)
      case e: Element => prev = if (e.normalName == "img") 'x' else '\n'
    }
    var atEnd = true
    for (n <- seq.reverseIterator) n match {
      case t: TextNode =>
        if (atEnd) { val s = t.getWholeText.replaceAll("[ ]+$", ""); t.text(s); if (s.nonEmpty) atEnd = false }
      case e: Element => atEnd = e.normalName != "img"
    }
    // 去掉首尾与连续的 <br>
    def significant(nodes: Seq[Node]) = nodes.filter {
      case t: TextNode => t.getWholeText.nonEmpty
      case _ => true
    }
    var kids = significant(el.childNodes.asScala.toList)
    while (kids.headOption.exists(isBr)) { kids.head.remove(); kids = kids.tail }
    while (kids.lastOption.exists(isBr)) { kids.last.remove(); kids = kids.init }
    var last: Node = null
    for (k <- significant(el.childNodes.asScala.toList)) {
      if (last != null && isBr(last) && isBr(k)) k.remove() else last = k
    }
  }

  private def isBr(n: Node): Boolean = n match { case e: Element => e.normalName == "br"; case _ => false }

  // ---------------------------------------------------------------- 步骤 6：标题、推断小标题、首尾样板

  private def cleanTitle(t: String): String = {
    val s = collapse(t).trim
    val first = s.split("\\s*[_|｜]\\s*|\\s+[-—–]\\s+").headOption.getOrElse(s).trim
    if (nonWs(first) >= 4) first else s
  }

  private def pickTitle(doc: Document, st: IdMap[Element, St], metaTitle: String, ogTitle: String, roots: Seq[Element]): String = {
    def ok(t: String) = { val n = nonWs(t); n >= 4 && n <= 120 }
    def h1s(e: Element) = (if (e.normalName == "h1") Iterator(e) else e.select("h1").asScala.iterator)
      .filter(h => linkDensity(st, h) < 0.5).map(h => collapse(h.text).trim).filter(ok)
    val inside = roots.iterator.flatMap(h1s)
    val meta = cleanTitle(if (ogTitle.trim.nonEmpty) ogTitle else metaTitle)
    val mn = norm(meta); val raw = norm(metaTitle + ogTitle)
    val anywhere = h1s(doc.body).filter(h => mn.isEmpty || raw.contains(norm(h)) || norm(h).contains(mn))
    if (inside.hasNext) inside.next() else if (anywhere.hasNext) anywhere.next() else meta
  }

  /** 去样板：界面按钮文字、短的版权 / 平台声明；正文后半段出现"相关阅读"等分隔行时从那里截断；
   *  首尾的面包屑、上一篇 / 下一篇、责任编辑，以及末尾整段都是链接的块。 */
  private def cleanBlocks(blocks: ArrayBuffer[Block]): Unit = {
    def text(x: Block) = collapse(x.el.text).trim
    def short(x: Block, n: Int) = nonWs(x.el.text) <= n
    val keep = blocks.filterNot(x => x.para && (UiLine.findFirstIn(text(x)).isDefined || (short(x, 150) && Notice.findFirstIn(text(x)).isDefined)))
    blocks.clear(); blocks ++= keep
    val cut = blocks.indices.indexWhere(i => i >= math.max(3, blocks.length / 3) && short(blocks(i), 20) &&
      !blocks(i).el.normalName.matches("table|ul|ol") && Related.findFirstIn(text(blocks(i))).isDefined)
    if (cut > 0) blocks.remove(cut, blocks.length - cut)
    def boiler(x: Block) = x.para && short(x, 80) && Boiler.findFirstIn(text(x)).isDefined
    def allLink(x: Block) = x.para && { val t = nonWs(x.el.text); t > 0 && nonWs(x.el.select("a").text) >= t }
    while (blocks.nonEmpty && boiler(blocks.head)) blocks.remove(0)
    while (blocks.nonEmpty && (boiler(blocks.last) || allLink(blocks.last))) blocks.remove(blocks.length - 1)
  }

  private def numRank(t: String): Int =
    if (Num0.findFirstIn(t).isDefined) 0 else if (Num1.findFirstIn(t).isDefined) 1
    else if (Num2.findFirstIn(t).isDefined) 2 else if (Num3.findFirstIn(t).isDefined) 3 else -1

  /** 推断小标题。候选是短行：≤40 字、不以句末标点结尾、不像"来源 / 作者 / 时间"等元信息。满足其一即提升：
   *  - 信号：整段加粗、字号明显大于本页正文、或 class 含标题词（某种信号覆盖全页一半以上时停用该信号）；
   *    连续 1–3 行信号短行、后面跟着正文时，整组都提升；
   *  - 编号："一、""（一）""1.""（1）"等开头，下一块是正文、不是同级编号短行；
   *  - 无标记短行（plain=true 时）：≤20 字且不含逗号句号冒号等，夹在两段正文之间，后一段 ≥60 字。
   *  层级：编号按"一、→h2，（一）→h3，1.→h4，（1）→h5"；无编号的在有"一、"级标题的页面上为 h3，否则 h2。 */
  private def promoteHeadings(blocks: ArrayBuffer[Block], plain: Boolean): Int = {
    val paras = blocks.filter(_.para)
    val total = math.max(1, paras.map(_.chars).sum)
    // 正文主体字号：已知字号的字数占一半以上时取其众数，否则按 16px
    val hist = scala.collection.mutable.HashMap[Int, Int]()
    paras.foreach(_.sizes.foreach { case (k, v) => if (k > 0) hist(k) = hist.getOrElse(k, 0) + v })
    val base = if (hist.values.sum * 2 >= total) hist.maxBy(_._2)._1.toDouble else 16.0
    val bigMin = math.max(base + 2, base * 1.15)
    def bigChars(b: Block) = b.sizes.collect { case (k, v) if k >= bigMin => v }.sum
    val useBold = paras.map(_.bold).sum < total / 2
    val useBig = paras.map(bigChars).sum < total / 2
    val useCls = paras.map(_.cls).sum < total / 2
    def textOf(b: Block) = collapse(b.el.text).trim
    def shortLine(t: String) = {
      val n = nonWs(t)
      n >= 2 && n <= 40 && EndPunct.findFirstIn(t).isEmpty && NotHeading.findFirstIn(t).isEmpty &&
        DateLike.findFirstIn(t).isEmpty && t.count(_.isDigit) * 10 < n * 6 // 以日期开头、或数字为主的行不是标题
    }
    def cand(b: Block) = b.para && b.el.select("img").isEmpty && shortLine(textOf(b))
    def signal(b: Block) = cand(b) && b.chars > 0 &&
      ((useBold && b.bold >= 0.9 * b.chars) || (useBig && bigChars(b) >= 0.9 * b.chars) || (useCls && b.cls >= 0.9 * b.chars))
    def contentAfter(i: Int) = i < blocks.length && nonWs(blocks(i).el.text) >= 10 && !signal(blocks(i))
    val decided = scala.collection.mutable.LinkedHashMap[Int, Int]() // 块下标 → 编号层级（-1 无编号）
    var i = 0
    while (i < blocks.length) {
      if (signal(blocks(i))) {
        var j = i; while (j < blocks.length && signal(blocks(j))) j += 1
        if (j - i <= 3 && contentAfter(j)) (i until j).foreach(k => decided(k) = numRank(textOf(blocks(k))))
        i = j
      } else i += 1
    }
    for (i <- blocks.indices if !decided.contains(i) && cand(blocks(i)) && i + 1 < blocks.length) {
      val t = textOf(blocks(i)); val rank = numRank(t)
      val nx = blocks(i + 1); val nt = textOf(nx); val nn = nonWs(nt)
      if (rank >= 0 && nn >= 20 && !(nx.para && numRank(nt) == rank && shortLine(nt))) decided(i) = rank
      else if (plain && rank < 0 && nonWs(t) <= 20 && !t.exists("，,。：:；;！!？?".contains(_)) && nx.para && nn >= 60 &&
        nt.exists("，。；！？".contains(_)) && i > 0 && blocks(i - 1).para && nonWs(blocks(i - 1).el.text) >= 30) decided(i) = -1
    }
    val hasTop = decided.values.exists(_ == 0)
    for ((i, rank) <- decided) {
      val level = if (rank >= 0) math.min(6, 2 + rank) else if (hasTop) 3 else 2
      val h = new Element("h" + level).attr("data-origin", "inferred")
      blocks(i).el.childNodes.asScala.toList.foreach(h.appendChild)
      h.select("strong, em").asScala.toList.foreach(_.unwrap())
      tidy(h)
      blocks(i) = new Block(h, false)
    }
    decided.size
  }

  // ---------------------------------------------------------------- 步骤 7：序列化与纯文本

  private val BlockLike = Set("table", "thead", "tbody", "ul", "ol", "blockquote")

  private def write(e: Element, sb: StringBuilder): Unit = {
    if (BlockLike(e.normalName)) {
      sb.append('<').append(e.normalName)
      e.attributes.asScala.foreach(a => sb.append(' ').append(a.html))
      sb.append(">\n")
      e.children.asScala.foreach { c => write(c, sb); sb.append('\n') }
      sb.append("</").append(e.normalName).append('>')
    } else sb.append(e.outerHtml)
  }

  private def plainText(el: Element): String = {
    val sb = new StringBuilder
    NodeTraversor.traverse(new NodeVisitor {
      def head(n: Node, depth: Int): Unit = n match {
        case t: TextNode => sb.append(t.getWholeText)
        case e: Element if e.normalName == "br" => sb.append('\n')
        case e: Element if (e.normalName == "td" || e.normalName == "th") && sb.nonEmpty && sb.last != '\n' => sb.append('\t')
        case _ =>
      }
      override def tail(n: Node, depth: Int): Unit = n match {
        case e: Element if Set("li", "tr", "caption", "p")(e.normalName) && sb.nonEmpty && sb.last != '\n' => sb.append('\n')
        case _ =>
      }
    }, el)
    sb.toString.trim
  }

  // ---------------------------------------------------------------- 主函数

  def simplify(url: String, pg: String, rules: SiteRules = SiteRules.empty, withText: Boolean = true,
               plainHeadings: Boolean = true): Result = {
    val t0 = System.nanoTime
    def ms = (System.nanoTime - t0) / 1000000
    def fail(status: String, err: String, n: Int) =
      Result(url, status, err, "", "", "", 0, "", 0, 0, 0, 0, 0, n, ms, Version)
    try {
      val html = decodePg(pg)
      if (html.length > MaxInputChars) return fail("too_large", "", html.length)
      val doc = Jsoup.parse(html, if (url == null) "" else url)
      val metaTitle = doc.title
      val ogTitle = Option(doc.selectFirst("meta[property=og:title]")).map(_.attr("content")).getOrElse("")
      removeJunk(doc)
      if (doc.body == null) return fail("empty", "no body", html.length)
      val st = computeStats(doc.body)
      val (roots, how) = locate(doc, st, rules, url)
      if (explainOut != null) roots.foreach(r => explainOut.println(s"  选中[$how] ${describe(r)}"))
      val total = roots.map(r => Option(st.get(r)).map(_.text).getOrElse(0)).sum
      val b = new Builder(st, total)
      roots.foreach(b.walkRoot)
      val blocks = b.blocks
      val title = pickTitle(doc, st, metaTitle, ogTitle, roots)
      cleanBlocks(blocks)
      val inferred = promoteHeadings(blocks, plainHeadings)
      // 标题放第一块；前几块里已有同文的块则去重
      if (title.nonEmpty) {
        val tn = norm(title)
        val dup = blocks.indexWhere(x => norm(x.el.text) == tn)
        if (dup >= 0 && dup < 5) blocks.remove(dup)
        blocks.insert(0, new Block(new Element("h1").text(title), false))
      }
      val out = Document.createShell(if (url == null) "" else url)
      out.outputSettings.prettyPrint(false).charset("UTF-8")
      val sb = new StringBuilder
      for (x <- blocks) { out.body.appendChild(x.el); write(x.el, sb); sb.append('\n') }
      val text = blocks.map(x => plainText(x.el)).filter(_.nonEmpty).mkString("\n")
      val heads = blocks.count(x => HeadingTags(x.el.normalName))
      val textChars = nonWs(text) - nonWs(title)
      Result(url, if (textChars > 0) "ok" else "empty", "", title, sb.toString, if (withText) text else "",
        textChars, how, heads, inferred, b.tables, b.lists, b.images, html.length, ms, Version)
    } catch {
      case e: Throwable => fail("error", (e.getClass.getSimpleName + ": " + e.getMessage).take(300), Option(pg).map(_.length).getOrElse(0))
    }
  }

  // ---------------------------------------------------------------- 命令行

  private def parseArgs(args: Array[String]): Map[String, String] = {
    val m = scala.collection.mutable.Map[String, String]()
    var i = 0
    while (i < args.length) {
      val k = args(i).stripPrefix("--")
      if (i + 1 < args.length && !args(i + 1).startsWith("--")) { m(k) = args(i + 1); i += 2 } else { m(k) = "true"; i += 1 }
    }
    m.toMap
  }

  def main(args: Array[String]): Unit = {
    val o = parseArgs(args)
    require(o.contains("input") && o.contains("output"), "需要 --input 与 --output")
    if (o.contains("local")) runLocal(o) else runSpark(o)
  }

  private def runLocal(o: Map[String, String]): Unit = {
    val urlCol = o.getOrElse("url-col", "url"); val htmlCol = o.getOrElse("html-col", "pg")
    val rules = o.get("site-rules").map(p => SiteRules.parse(scala.io.Source.fromFile(p, "UTF-8").mkString)).getOrElse(SiteRules.empty)
    val src = scala.io.Source.fromFile(o("input"), "UTF-8")
    val out = new PrintWriter(new File(o("output")), "UTF-8")
    val status = scala.collection.mutable.Map[String, Int]().withDefaultValue(0)
    val locate = scala.collection.mutable.Map[String, Int]().withDefaultValue(0)
    var n = 0; var millis = 0L
    val t0 = System.nanoTime
    try {
      for (line <- src.getLines() if line.trim.nonEmpty) {
        val node = json.readTree(line)
        val id = node.path("id").asText("")
        val explain = o.get("explain").exists(ids => id.nonEmpty && ids.split(",").contains(id))
        if (explain) { println(s"== explain $id ${node.path(urlCol).asText("")}"); explainOut = System.out }
        val r = simplify(node.path(urlCol).asText(""), node.path(htmlCol).asText(""), rules, withText = true,
          plainHeadings = !o.contains("no-plain-headings"))
        val obj = json.createObjectNode()
        if (node.has("id")) obj.put("id", node.get("id").asText)
        obj.put("url", r.url).put("status", r.status).put("error", r.error).put("title", r.title).put("html", r.html)
          .put("text", r.text).put("textChars", r.textChars).put("locate", r.locate).put("headings", r.headings)
          .put("inferredHeadings", r.inferredHeadings).put("tables", r.tables).put("lists", r.lists).put("images", r.images)
          .put("inputChars", r.inputChars).put("millis", r.millis).put("version", r.version)
        out.println(json.writeValueAsString(obj))
        if (explain) { explainOut = null; println(r.html.take(o.getOrElse("explain-chars", "600").toInt)) }
        n += 1; millis += r.millis; status(r.status) += 1; locate(r.locate) += 1
      }
    } finally { out.close(); src.close() }
    val wall = (System.nanoTime - t0) / 1e9
    println(f"pages=$n  status=${status.toMap}  locate=${locate.toMap}  avg=${millis.toDouble / math.max(n, 1)}%.1f ms/page  wall=$wall%.1fs")
  }

  private def runSpark(o: Map[String, String]): Unit = {
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions.col
    val builder = SparkSession.builder.appName("html-simplify-" + Version)
    o.get("master").foreach(builder.master)
    val spark = builder.getOrCreate()
    import spark.implicits._
    val urlCol = o.getOrElse("url-col", "url"); val htmlCol = o.getOrElse("html-col", "pg")
    val df = o.getOrElse("input-format", "json") match {
      case "json" => spark.read.schema(s"`$urlCol` STRING, `$htmlCol` STRING").json(o("input"))
      case f => spark.read.format(f).load(o("input"))
    }
    val rulesText = o.get("site-rules").map(p => spark.read.textFile(p).collect().mkString("\n")).getOrElse("")
    val rulesB = spark.sparkContext.broadcast(rulesText)
    val withText = o.contains("with-text")
    val plain = !o.contains("no-plain-headings")
    var ds = df.select(col(urlCol).as("url"), col(htmlCol).as("pg")).as[Input]
    o.get("partitions").foreach(p => ds = ds.repartition(p.toInt))
    val out = ds.mapPartitions { it =>
      val rules = SiteRules.parse(rulesB.value) // 每个分区解析一次
      it.map(r => simplify(r.url, r.pg, rules, withText, plain))
    }
    out.write.mode(o.getOrElse("mode", "errorifexists")).format(o.getOrElse("output-format", "parquet")).save(o("output"))
    spark.stop()
  }
}
