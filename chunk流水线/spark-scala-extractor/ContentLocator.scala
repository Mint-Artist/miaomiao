package extractor

import java.util.IdentityHashMap

import org.jsoup.nodes.{Element, Node, TextNode}
import org.jsoup.select.{NodeTraversor, NodeVisitor}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

private[extractor] object TextUtil {
  private val Whitespace = "[\\s\\u00a0\\u3000]+".r

  def clean(s: String): String = Whitespace.replaceAllIn(s, " ").trim

  /** 对应 BeautifulSoup ''.join(stripped_strings)：逐文本节点去空白后无分隔拼接。 */
  def textOf(node: Node): String = {
    val out = new StringBuilder
    NodeTraversor.traverse(new NodeVisitor {
      override def head(n: Node, depth: Int): Unit = n match {
        case t: TextNode => out.append(clean(t.getWholeText))
        case _           =>
      }
      override def tail(n: Node, depth: Int): Unit = ()
    }, node)
    out.toString
  }

  def pathOf(e: Element, body: Element): String = {
    val chain = (e :: e.parents().asScala.toList.takeWhile(_ ne body)).reverse
    ("body" :: chain.map { el =>
      val id = el.id()
      val cls = el.classNames().asScala.headOption
      el.normalName + (if (id.nonEmpty) "#" + id else cls.map("." + _).getOrElse(""))
    }).mkString(" > ")
  }
}

/** 无站点配置时的通用正文定位：文本密度 + 链接密度 + 包装块逐级下钻。 */
private[extractor] object ContentLocator {
  final case class Located(
      root: Element,
      title: Option[Element],
      metadata: List[String],
      path: String,
      confidence: Double,
      degraded: Boolean)

  private val InlineTags = Set(
    "a", "abbr", "b", "bdi", "bdo", "big", "br", "cite", "code", "del", "dfn", "em",
    "font", "i", "img", "ins", "kbd", "label", "mark", "q", "s", "samp", "small",
    "span", "strike", "strong", "sub", "sup", "time", "tt", "u", "var", "wbr")
  private val ContentTags = Set(
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "dl", "li", "dt", "dd",
    "table", "caption", "thead", "tbody", "tfoot", "tr", "th", "td", "blockquote", "pre", "figure")
  private val Punct = "[，。；！？、,.;!?]".r
  private val KeyValue = "^([^：:，。,.!?！？\\d]{1,12})[：:]\\s*(\\S.*)$".r
  private val DescendRatio = 0.6

  private final class Stat {
    var text = 0
    var link = 0
    var punct = 0
    val sb = new StringBuilder
    def weight(cell: Boolean): Double =
      if (link * 2 > text) 0.0
      else if (!cell && text < 6 && punct == 0) 0.0
      else (text - link).toDouble
  }

  def locate(body: Element, docTitle: String): Located = {
    val stats = new IdentityHashMap[Element, Stat]()
    NodeTraversor.traverse(new NodeVisitor {
      override def head(n: Node, depth: Int): Unit = n match {
        case t: TextNode =>
          val txt = TextUtil.clean(t.getWholeText)
          if (txt.nonEmpty) {
            var block = t.parent().asInstanceOf[Element]
            var inLink = false
            while ((block ne body) && InlineTags(block.normalName)) {
              if (block.normalName == "a") inLink = true
              block = block.parent()
            }
            val stat = Option(stats.get(block)).getOrElse {
              val created = new Stat
              stats.put(block, created)
              created
            }
            stat.text += txt.length
            if (inLink) stat.link += txt.length
            stat.punct += Punct.findAllIn(txt).size
            stat.sb.append(txt).append(' ')
          }
        case _ =>
      }
      override def tail(n: Node, depth: Int): Unit = ()
    }, body)

    val score = new IdentityHashMap[Element, java.lang.Double]()
    def scoreOf(e: Element): Double = Option(score.get(e)).map(_.doubleValue).getOrElse(0.0)
    stats.asScala.foreach { case (element, stat) =>
      val weight = stat.weight(Set("td", "th")(element.normalName))
      if (weight > 0) {
        var ancestor = element
        while (ancestor != null) {
          score.put(ancestor, scoreOf(ancestor) + weight)
          ancestor = if (ancestor eq body) null else ancestor.parent()
        }
      }
    }

    var root = body
    var descending = true
    while (descending) {
      val total = scoreOf(root)
      val children = root.children().asScala.filterNot(c => ContentTags(c.normalName))
      if (total <= 0 || children.isEmpty) descending = false
      else {
        val best = children.maxBy(scoreOf)
        if (scoreOf(best) >= DescendRatio * total) root = best else descending = false
      }
    }

    val degraded = root eq body
    val concentration = if (scoreOf(body) > 0) scoreOf(root) / scoreOf(body) else 0.0
    val confidence = if (degraded) 0.35 else math.min(0.95, 0.60 + 0.40 * concentration)

    if (degraded) Located(body, None, Nil, "body", confidence, degraded = true)
    else {
      val order = new IdentityHashMap[Element, Integer]()
      body.getAllElements.asScala.zipWithIndex.foreach { case (e, i) => order.put(e, i) }
      def idx(e: Element): Int = order.get(e).intValue
      def isAncestor(a: Element, b: Element): Boolean = b.parents().asScala.exists(_ eq a)
      def matchesTitle(h: Element): Boolean = {
        val text = TextUtil.clean(h.text())
        text.length >= 4 && docTitle.contains(text)
      }

      val title =
        if (!root.getElementsByTag("h1").isEmpty || root.select("h2, h3").asScala.exists(matchesTitle)) None
        else {
          val before = body.select("h1, h2, h3").asScala.reverse
            .filter(h => idx(h) < idx(root) && !isAncestor(h, root) && TextUtil.clean(h.text()).nonEmpty)
          before.find(h => h.normalName == "h1" && matchesTitle(h))
            .orElse(before.find(matchesTitle))
            .orElse(before.find(_.normalName == "h1"))
        }

      val metadata = title match {
        case None => Nil
        case Some(heading) =>
          val lines = ArrayBuffer[String]()
          stats.asScala.toSeq
            .filter { case (element, _) =>
              idx(element) > idx(heading) && idx(element) < idx(root) &&
                !isAncestor(element, root) && !isAncestor(heading, element)
            }
            .sortBy { case (element, _) => idx(element) }
            .foreach { case (_, stat) =>
              TextUtil.clean(stat.sb.toString) match {
                case line @ KeyValue(_, _) if !lines.contains(line) => lines += line
                case _ =>
              }
            }
          lines.toList
      }
      Located(root, title, metadata, TextUtil.pathOf(root, body), confidence, degraded = false)
    }
  }
}
