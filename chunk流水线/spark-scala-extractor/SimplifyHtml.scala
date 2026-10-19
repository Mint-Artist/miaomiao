package extractor

import java.net.URL
import java.util.IdentityHashMap

import org.json.{JSONArray, JSONObject}
import org.jsoup.Jsoup
import org.jsoup.nodes.{Comment, Element, Node, TextNode}
import org.jsoup.select.{NodeTraversor, NodeVisitor}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

final case class ExtractionOptions(
    anchors: Boolean = false,
    promotePseudo: Boolean = false,
    forceGeneric: Boolean = false)

final case class TableReport(rows: Int, headerRows: Int, colWidths: List[Int]) {
  def toJson: JSONObject = new JSONObject()
    .put("rows", rows).put("header_rows", headerRows)
    .put("col_widths", new JSONArray(colWidths.asJava))
}

final case class ExtractionReport(
    extractorVersion: String,
    url: String,
    profile: String,
    extractionMode: String,
    contentSelector: String,
    titleSelector: String,
    title: String,
    status: String,
    confidence: Double,
    textChars: Int,
    tagCounts: Map[String, Int],
    tables: List[TableReport],
    stats: Map[String, Int],
    warnings: List[String],
    error: Option[String] = None) {

  def toJson: String = {
    val json = new JSONObject()
      .put("extractor_version", extractorVersion)
      .put("url", url)
      .put("profile", profile)
      .put("extraction_mode", extractionMode)
      .put("content_selector", contentSelector)
      .put("title_selector", titleSelector)
      .put("title", title)
      .put("status", status)
      .put("confidence", confidence)
      .put("text_chars", textChars)
      .put("tag_counts", new JSONObject(tagCounts.asJava))
      .put("tables", new JSONArray(tables.map(_.toJson).asJava))
      .put("stats", new JSONObject(stats.asJava))
      .put("warnings", new JSONArray(warnings.asJava))
    error.foreach(message => json.put("error", message))
    json.toString
  }
}

final case class ExtractionResult(html: String, report: ExtractionReport)

final case class SafeExtraction(
    simplifiedHtml: String,
    reportJson: String,
    status: String,
    confidence: Double,
    profile: String,
    contentRoot: String,
    error: String) {
  def isSuccess: Boolean = status != "failed"
}

/** 原始 HTML → 简版 HTML。配置路径与 Python 0.1.2 对齐，未配置站点走通用定位。 */
object SimplifyHtml {
  val ExtractorVersion = "0.2.2-scala"

  private val KeepTags = Set(
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "ul", "ol", "li",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col",
    "figure", "figcaption", "blockquote", "pre", "code", "a", "img", "strong", "em")
  private val RenameTags = Map("b" -> "strong", "i" -> "em")
  private val DropTags = Set(
    "script", "style", "noscript", "template", "iframe", "svg", "canvas", "form",
    "input", "button", "select", "textarea", "object", "embed", "video", "audio",
    "link", "meta", "base", "nav", "header", "footer", "aside")
  private val UnwrapInline = Set(
    "span", "font", "u", "s", "strike", "del", "ins", "mark", "small", "big",
    "abbr", "time", "cite", "q", "sub", "sup", "label")
  private val UnwrapBlocks = Set("section", "article", "main", "center", "hgroup", "address")
  private val SemanticDivKeywords = List("note", "metadata", "warning", "tip", "info", "alert", "caution", "notice")
  private val DivClassAlias = Map("tip" -> "note", "notice" -> "note")
  private val NoiseTokens = Set(
    "ad", "ads", "advert", "banner", "promo", "recommend", "related", "share",
    "comment", "breadcrumb", "sidebar", "ymzy", "tg")
  private val AttrWhitelist = Map(
    "a" -> Set("href"), "img" -> Set("src", "alt"), "ol" -> Set("start", "reversed"),
    "li" -> Set("value"), "td" -> Set("rowspan", "colspan", "scope", "headers"),
    "th" -> Set("rowspan", "colspan", "scope", "headers"), "col" -> Set("span"))
  private val ExtraAttrs = Set("data-src", "data-origin")
  private val VoidTags = Set("br", "img", "col", "hr")
  private val InlineTags = Set("a", "strong", "em", "code", "br", "img")
  private val OneLineContainers = Set("td", "th", "tr", "caption", "figcaption")
  private val AnchorTags = Set(
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "table", "figure", "blockquote", "pre", "div")
  private val Whitespace = "[\\s\\u00a0\\u3000]+".r
  private val TitleSuffix = "\\s*[_|—-]\\s*[^_|—-]*$".r
  private val AbsoluteScheme = "^[a-z]+://.*".r

  private def descendants(root: Element): List[Element] = {
    val out = ArrayBuffer[Element]()
    NodeTraversor.traverse(new NodeVisitor {
      override def head(n: Node, depth: Int): Unit = n match {
        case e: Element if e ne root => out += e
        case _                       =>
      }
      override def tail(n: Node, depth: Int): Unit = ()
    }, root)
    out.toList
  }

  private def identitySet[A]: java.util.Set[A] =
    java.util.Collections.newSetFromMap(new IdentityHashMap[A, java.lang.Boolean]())

  private def isGone(element: Element, removed: java.util.Set[Element]): Boolean = {
    var current = element
    var gone = false
    while (current != null && !gone) {
      if (removed.contains(current)) gone = true
      current = current.parent()
    }
    gone
  }

  private def firstWithText(doc: Element, selectors: List[String], minChars: Int): (Element, String) = {
    val found = selectors.view.flatMap(sel => doc.select(sel).asScala.map(_ -> sel))
      .find { case (element, _) => TextUtil.textOf(element).length >= minChars }
    found.orNull
  }

  private def classIdTokens(element: Element): Set[String] = {
    val text = element.className() + " " + element.id()
    text.toLowerCase.split("[^a-z0-9]+").filter(_.nonEmpty).toSet
  }

  private def semanticDivClass(element: Element): String = {
    element.classNames().asScala.view.flatMap { cls =>
      val lower = cls.toLowerCase
      SemanticDivKeywords.find(lower.contains).map(kw => DivClassAlias.getOrElse(kw, kw))
    }.headOption.orNull
  }

  private def normalizeWhitespace(fragment: Element): Unit = {
    val comments = ArrayBuffer[Node]()
    NodeTraversor.traverse(new NodeVisitor {
      override def head(n: Node, depth: Int): Unit = n match {
        case c: Comment  => comments += c
        case t: TextNode =>
          var parent = t.parent()
          var inPre = false
          while (parent != null && !inPre) {
            if (parent.normalName == "pre") inPre = true
            parent = parent.parent()
          }
          if (!inPre) {
            val normalized = Whitespace.replaceAllIn(t.getWholeText, " ").trim
            if (normalized.isEmpty) t.remove() else t.text(normalized)
          }
        case _ =>
      }
      override def tail(n: Node, depth: Int): Unit = ()
    }, fragment)
    comments.foreach(_.remove())
  }

  private def cleanTree(root: Element, noiseSelectors: List[String], stats: mutable.Map[String, Int]): Unit = {
    def bump(key: String): Unit = stats.update(key, stats.getOrElse(key, 0) + 1)
    val noiseIds = identitySet[Element]
    noiseSelectors.foreach(sel => root.select(sel).asScala.foreach(noiseIds.add))
    val removed = identitySet[Element]
    descendants(root).foreach { element =>
      if (!isGone(element, removed) &&
          (DropTags(element.normalName) || noiseIds.contains(element) ||
            (classIdTokens(element) & NoiseTokens).nonEmpty)) {
        bump(s"dropped:${element.normalName}")
        removed.add(element)
        element.remove()
      }
    }

    var changed = true
    while (changed) {
      changed = false
      descendants(root).foreach { element =>
        if (!isGone(element, removed)) {
          val name = element.normalName
          if (RenameTags.contains(name)) {
            element.tagName(RenameTags(name))
            changed = true
          } else if (name == "div") {
            val semantic = semanticDivClass(element)
            if (semantic != null) {
              element.attributes().asScala.toList.foreach(a => element.removeAttr(a.getKey))
              element.addClass(semantic)
              bump(s"kept_div:$semantic")
            } else {
              removed.add(element)
              element.unwrap()
              changed = true
            }
          } else if (UnwrapInline(name) || UnwrapBlocks(name)) {
            removed.add(element)
            element.unwrap()
            changed = true
          } else if (!KeepTags(name)) {
            bump(s"unknown:$name")
            removed.add(element)
            if (TextUtil.textOf(element).nonEmpty || !element.select("img").isEmpty) element.unwrap()
            else element.remove()
            changed = true
          }
        }
      }
    }

    descendants(root).foreach { element =>
      if (!isGone(element, removed)) {
        val allowed = AttrWhitelist.getOrElse(element.normalName, Set.empty[String]) ++ ExtraAttrs ++
          (if (element.normalName == "div") Set("class") else Set.empty[String])
        element.attributes().asScala.toList.foreach { attr =>
          if (!allowed(attr.getKey)) element.removeAttr(attr.getKey)
        }
      }
    }
  }

  private def directChildren(element: Element, tag: String): List[Element] =
    element.children().asScala.filter(_.normalName == tag).toList

  private def normalizeTable(table: Element): Unit = {
    val caption = table.children().asScala.find(_.normalName == "caption").orNull
    val colgroup = table.children().asScala.find(_.normalName == "colgroup").orNull
    val headerRows = ArrayBuffer[Element]()
    val bodyRows = ArrayBuffer[Element]()
    val footRows = ArrayBuffer[Element]()
    table.children().asScala.filter(e => Set("thead", "tbody", "tfoot")(e.normalName)).foreach { container =>
      val rows = directChildren(container, "tr")
      if (container.normalName == "thead") headerRows ++= rows
      else if (container.normalName == "tfoot") footRows ++= rows
      else bodyRows ++= rows
    }
    bodyRows ++= directChildren(table, "tr")
    val realBody = bodyRows.filter { row =>
      val cells = row.children().asScala.filter(c => Set("td", "th")(c.normalName))
      !(cells.nonEmpty && cells.forall(_.normalName == "th"))
    }
    val realBodyIds = identitySet[Element]
    realBody.foreach(realBodyIds.add)
    bodyRows.filterNot(realBodyIds.contains).foreach(headerRows += _)
    table.empty()
    if (caption != null) table.appendChild(caption)
    if (colgroup != null) table.appendChild(colgroup)
    if (headerRows.nonEmpty) {
      val thead = new Element("thead")
      table.appendChild(thead)
      headerRows.foreach(thead.appendChild)
    }
    val tbody = new Element("tbody")
    table.appendChild(tbody)
    realBody.foreach(tbody.appendChild)
    if (footRows.nonEmpty) {
      val tfoot = new Element("tfoot")
      table.appendChild(tfoot)
      footRows.foreach(tfoot.appendChild)
    }
  }

  private def blockifyRoot(root: Element): Unit = {
    val groups = ArrayBuffer[Node]()
    val buffer = ArrayBuffer[Node]()
    def flush(): Unit = {
      val meaningful = buffer.exists {
        case t: TextNode => t.getWholeText.trim.nonEmpty
        case _           => true
      }
      if (meaningful) {
        val p = new Element("p")
        buffer.foreach(p.appendChild)
        groups += p
      }
      buffer.clear()
    }
    root.childNodes().asScala.toList.foreach {
      case t: TextNode => buffer += t
      case e: Element if InlineTags(e.normalName) => buffer += e
      case other => flush(); groups += other
    }
    flush()
    root.empty()
    groups.foreach(root.appendChild)
  }

  private def pruneEmpty(root: Element): Int = {
    val candidates = Set("p", "li", "div", "figure", "figcaption", "blockquote", "pre",
      "h1", "h2", "h3", "h4", "h5", "h6")
    var pruned = 0
    var changed = true
    while (changed) {
      changed = false
      descendants(root).foreach { element =>
        if (candidates(element.normalName) && TextUtil.textOf(element).isEmpty &&
            element.select("img, table").isEmpty) {
          element.remove()
          pruned += 1
          changed = true
        }
      }
    }
    pruned
  }

  private def absolutize(outRoot: Element, url: String): Unit = {
    if (url != null && url.nonEmpty) {
      val links = descendants(outRoot).flatMap {
        case e if e.normalName == "a"  => Some(e -> "href")
        case e if e.normalName == "img" => Some(e -> "src")
        case _                         => None
      }
      links.foreach { case (element, attr) =>
        val value = element.attr(attr)
        if (value.nonEmpty && AbsoluteScheme.findFirstIn(value).isEmpty) {
          try element.attr(attr, new URL(new URL(url), value).toString)
          catch { case _: java.net.MalformedURLException => () }
        }
      }
    }
  }

  private def addAnchors(outRoot: Element): Int = {
    val targets = descendants(outRoot).filter(e => AnchorTags(e.normalName))
    targets.zipWithIndex.foreach { case (element, index) => element.attr("data-src", s"e${index + 1}") }
    targets.size
  }

  private def promotePseudoHeadings(outRoot: Element): Int = {
    var promoted = 0
    var lastLevel = 1
    val targets = descendants(outRoot).filter(e => e.normalName.matches("h[1-6]|p"))
    targets.foreach { element =>
      if (element.normalName != "p") lastLevel = element.normalName.drop(1).toInt
      else {
        val children = element.childNodes().asScala.filter {
          case t: TextNode => t.getWholeText.trim.nonEmpty
          case _           => true
        }
        val text = TextUtil.textOf(element)
        if (children.size == 1 && children.head.isInstanceOf[Element] &&
            children.head.asInstanceOf[Element].normalName == "strong" &&
            text.nonEmpty && text.length < 40 &&
            !Set("。", "，", "；", "：").exists(text.endsWith)) {
          val heading = new Element(s"h${math.min(lastLevel + 1, 6)}")
          heading.attr("data-origin", "inferred")
          heading.appendText(text)
          element.replaceWith(heading)
          promoted += 1
        }
      }
    }
    promoted
  }

  private def escapeText(text: String): String =
    text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

  private def escapeAttr(text: String): String =
    escapeText(text).replace("\"", "&quot;").replace("'", "&#x27;")

  private def attrsText(element: Element): String = {
    val out = new StringBuilder
    // Python 版会先 str(root) 再解析；BeautifulSoup 序列化时按属性名排序
    element.attributes().asScala.toList.sortBy(_.getKey).foreach { attr =>
      if (attr.getValue == null || attr.getValue.isEmpty) out.append(' ').append(attr.getKey)
      else out.append(' ').append(attr.getKey).append("=\"")
        .append(escapeAttr(attr.getValue)).append('"')
    }
    out.toString
  }

  private def serializeInline(node: Node): String = node match {
    case t: TextNode => escapeText(t.getWholeText)
    case e: Element if VoidTags(e.normalName) => s"<${e.normalName}${attrsText(e)}>"
    case e: Element =>
      val inner = e.childNodes().asScala.map(serializeInline).mkString
      s"<${e.normalName}${attrsText(e)}>$inner</${e.normalName}>"
    case _ => ""
  }

  private def serializeBlock(node: Node, indent: Int = 0): String = node match {
    case t: TextNode => escapeText(t.getWholeText)
    case e: Element =>
      val pad = "  " * indent
      val name = e.normalName
      if (VoidTags(name)) return s"$pad<$name${attrsText(e)}>\n"
      if (OneLineContainers(name)) {
        if (name == "tr") {
          val cells = e.children().asScala.map(c => serializeBlock(c).trim).mkString(" ")
          return s"$pad<tr${attrsText(e)}>$cells</tr>\n"
        }
        val inner = e.childNodes().asScala.map(serializeInline).mkString
        return s"$pad<$name${attrsText(e)}>$inner</$name>\n"
      }

      val pieces = ArrayBuffer[(String, String)]()
      val inlineBuffer = ArrayBuffer[Node]()
      def flush(): Unit = {
        if (inlineBuffer.nonEmpty) {
          pieces += ("inline" -> inlineBuffer.map(serializeInline).mkString)
          inlineBuffer.clear()
        }
      }
      e.childNodes().asScala.foreach {
        case t: TextNode if t.getWholeText.trim.nonEmpty => inlineBuffer += t
        case child: Element if InlineTags(child.normalName) => inlineBuffer += child
        case child: Element => flush(); pieces += ("block" -> serializeBlock(child, indent + 1))
        case _ =>
      }
      flush()
      val nonEmpty = pieces.filter { case (kind, value) => kind == "inline" && value.nonEmpty || kind == "block" && value.nonEmpty }
      if (nonEmpty.isEmpty) ""
      else if (nonEmpty.size == 1 && nonEmpty.head._1 == "inline")
        s"$pad<$name${attrsText(e)}>${nonEmpty.head._2}</$name>\n"
      else {
        val inner = nonEmpty.map { case (kind, value) =>
          if (kind == "inline") s"$pad  $value\n" else value
        }.mkString
        s"$pad<$name${attrsText(e)}>\n$inner$pad</$name>\n"
      }
    case _ => ""
  }

  private def serialize(outRoot: Element): String =
    outRoot.children().asScala.map(serializeBlock(_)).mkString.stripSuffix("\n") + "\n"

  def extract(
      html: String,
      url: String,
      profiles: List[Profile],
      options: ExtractionOptions = ExtractionOptions(),
      profileName: Option[String] = None): ExtractionResult = {
    val stats = mutable.Map[String, Int]()
    val warnings = ArrayBuffer[String]()
    val doc = Jsoup.parse(if (html == null) "" else html)
    val profile = profileName.map(Profiles.byName(profiles, _)).orElse(Profiles.byUrl(profiles, url))
    val useProfile = profile.nonEmpty && !options.forceGeneric

    var root: Element = null
    var titleNode: Element = null
    var titleSelector = ""
    var contentSelector = ""
    var extractionMode = ""
    var confidence = 1.0
    var status = "ok"
    var metadata = List[String]()

    if (useProfile) {
      val p = profile.get
      val titleResult = firstWithText(doc, p.title, 1)
      if (titleResult != null) {
        titleNode = titleResult._1
        titleSelector = titleResult._2
      }
      val contentResult = firstWithText(doc, p.content, 20)
      if (contentResult != null) {
        root = contentResult._1
        contentSelector = contentResult._2
        extractionMode = "profile_selector"
        metadata = p.metadata.flatMap { rule =>
          doc.select(rule.selector).asScala.map(TextUtil.textOf).filter(_.nonEmpty).map { text =>
            if (rule.label.exists(label => !text.startsWith(label))) rule.label.get + "：" + text else text
          }.toList
        }
      }
    }

    if (root == null) {
      if (useProfile) warnings += "站点选择器未命中，改用通用正文定位"
      val genericDoc = Jsoup.parse(if (html == null) "" else html)
      val body = genericDoc.body()
      val locationRemoved = identitySet[Element]
      (body :: descendants(body)).foreach { element =>
        // body 自身绝不能按噪声删除：float_ad_flag 这类容器类名会把整页正文误删
        if ((element ne body) && !isGone(element, locationRemoved) &&
            (DropTags(element.normalName) || (classIdTokens(element) & NoiseTokens).nonEmpty ||
              element.hasAttr("hidden") || element.attr("aria-hidden") == "true")) {
          locationRemoved.add(element)
          element.remove()
        }
      }
      val located = ContentLocator.locate(body, genericDoc.title())
      root = located.root
      contentSelector = located.path
      extractionMode = "generic_locator"
      confidence = located.confidence
      if (located.degraded) status = "degraded"
      metadata = located.metadata
      titleNode = located.title.orNull
      if (titleNode == null) {
        val titleResult = firstWithText(genericDoc, List("h1", "title"), 1)
        if (titleResult != null) titleNode = titleResult._1
      }
      titleSelector = "generic_locator"
    }

    if (root == null || TextUtil.textOf(root).isEmpty && root.select("img").isEmpty)
      throw new IllegalArgumentException("未找到可用正文（可能为空页、仅脚本或需要浏览器渲染）")

    var title = ""
    if (titleNode != null) {
      title = TextUtil.clean(titleNode.text())
      if (titleNode.normalName == "title") title = TitleSuffix.replaceFirstIn(title, "").trim
    } else warnings += "未找到标题"

    val outRoot = new Element("div")
    outRoot.appendChild(new Element("h1").appendText(title))
    if (metadata.nonEmpty) {
      val div = new Element("div").addClass("metadata")
      metadata.foreach(text => div.appendChild(new Element("p").appendText(text)))
      outRoot.appendChild(div)
    }

    val fragment = new Element("div")
    fragment.appendChild(root.clone())
    if (useProfile && extractionMode == "profile_selector") {
      profile.get.strip.foreach { selector =>
        fragment.select(selector).asScala.foreach { element =>
          element.remove()
          stats.update(s"stripped:$selector", stats.getOrElse(s"stripped:$selector", 0) + 1)
        }
      }
    }
    normalizeWhitespace(fragment)
    cleanTree(fragment, if (useProfile) profile.get.noise else Nil, stats)
    val tables = fragment.select("table").asScala.toList
    tables.foreach(normalizeTable)
    blockifyRoot(fragment)
    val pruned = pruneEmpty(fragment)
    if (pruned > 0) stats.update("pruned_empty", pruned)
    if (TextUtil.textOf(fragment).isEmpty && fragment.select("img, table").isEmpty)
      throw new IllegalArgumentException("正文清理后无可用内容")

    fragment.children().asScala.foreach(outRoot.appendChild)

    // 只检查 h1/metadata 后的第一个正文块；与文档标题同文则去重
    val blocks = outRoot.children().asScala.toList
    blocks.drop(1).dropWhile(c => c.normalName == "div" && c.hasClass("metadata")).headOption.foreach { candidate =>
      if (Set("h1", "h2", "h3")(candidate.normalName) && TextUtil.textOf(candidate) == title) {
        candidate.remove()
        stats.update("dedup_title_heading", 1)
      }
    }

    absolutize(outRoot, url)
    if (options.anchors) stats.update("anchors", addAnchors(outRoot))
    if (options.promotePseudo) stats.update("promoted_pseudo_headings", promotePseudoHeadings(outRoot))

    val all = descendants(outRoot)
    val tagCounts = all.groupBy(_.normalName).map { case (name, elements) => name -> elements.size }
    val tableReports = all.filter(_.normalName == "table").map { table =>
      val rows = table.select("tr").asScala.toList
      val widths = rows.map(row => row.children().asScala.count(c => Set("td", "th")(c.normalName))).distinct.sorted
      if (widths.size > 1)
        warnings += s"table 各行单元格数不一致: ${widths.mkString("[", ", ", "]")}（若有 rowspan/colspan 属正常）"
      TableReport(rows.size, table.select("thead tr").size, widths)
    }
    all.filter(_.normalName == "img").foreach { img =>
      if (!img.hasAttr("alt")) warnings += s"img 缺 alt: ${img.attr("src").take(80)}"
    }
    all.filter(_.normalName == "a").foreach { link =>
      if (!link.hasAttr("href")) warnings += s"a 缺 href: ${TextUtil.textOf(link).take(40)}"
    }

    val report = ExtractionReport(
      extractorVersion = ExtractorVersion,
      url = Option(url).getOrElse(""),
      profile = if (useProfile) profile.get.name else "generic",
      extractionMode = extractionMode,
      contentSelector = contentSelector,
      titleSelector = titleSelector,
      title = title,
      status = status,
      confidence = confidence,
      textChars = TextUtil.textOf(outRoot).length,
      tagCounts = tagCounts,
      tables = tableReports,
      stats = stats.toMap,
      warnings = warnings.toList)
    ExtractionResult(serialize(outRoot), report)
  }

  def extractSafe(
      html: String,
      url: String,
      profiles: List[Profile],
      options: ExtractionOptions = ExtractionOptions()): SafeExtraction = {
    try {
      val result = extract(html, url, profiles, options)
      SafeExtraction(result.html, result.report.toJson, result.report.status,
        result.report.confidence, result.report.profile, result.report.contentSelector, null)
    } catch {
      case error: Exception =>
        val message = Option(error.getMessage).getOrElse(error.getClass.getSimpleName).take(500)
        val report = ExtractionReport(ExtractorVersion, Option(url).getOrElse(""), "unknown",
          "failed", "", "", "", "failed", 0.0, 0, Map.empty, Nil, Map.empty, List(message), Some(message))
        SafeExtraction(null, report.toJson, "failed", 0.0, "unknown", "", message)
    }
  }
}
