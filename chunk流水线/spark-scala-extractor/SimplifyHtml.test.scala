package extractor

import java.nio.file.{Files, Paths}

import org.json.JSONObject
import org.jsoup.Jsoup

import scala.collection.JavaConverters._

class SimplifyHtmlTest extends munit.FunSuite {
  private val profiles = Profiles.load(Paths.get("profiles.json"))
  private val cases = List(
    ("huawei_zh-cn16079039", "https://consumer.huawei.com/cn/support/content/zh-cn16079039/"),
    ("gk100_read_27177662", "https://m.gk100.com/read_27177662.htm"),
    ("lvtu_8655766", "https://www.64365.com/zs/8655766.aspx"))

  private def read(path: String): String =
    new String(Files.readAllBytes(Paths.get(path)), java.nio.charset.StandardCharsets.UTF_8)

  cases.foreach { case (name, url) =>
    test(s"配置路径与 Python 基准逐字节一致: $name") {
      val raw = read(s"samples/$name.raw.html")
      val expected = read(s"expected/$name.simplified.html")
      val result = SimplifyHtml.extract(raw, url, profiles)
      assertEquals(result.html, expected)
      assertEquals(result.report.status, "ok")
      assertEquals(result.report.extractionMode, "profile_selector")
    }
  }

  test("通用路径：三个样本保留核心正文，状态与置信度可查") {
    cases.foreach { case (name, url) =>
      val raw = read(s"samples/$name.raw.html")
      val result = SimplifyHtml.extract(raw, url, profiles, ExtractionOptions(forceGeneric = true))
      assert(result.report.extractionMode == "generic_locator")
      assert(Set("ok", "degraded")(result.report.status))
      assert(result.report.confidence > 0 && result.report.confidence <= 1)
      val text = Jsoup.parse(result.html).text()
      name match {
        case "huawei_zh-cn16079039" =>
          assert(text.contains("华为儿童手表跨品牌添加联系人"))
          assert(text.contains("360 青少年手表 12X 需下载“跨品牌交友”应用"))
          assert(!text.contains("是否有帮助"))
        case "gk100_read_27177662" =>
          assert(text.contains("清华大学"))
          assert(text.contains("苏州大学"))
          assert(text.contains("学校名"))
        case "lvtu_8655766" =>
          assert(text.contains("公司裁员提成部分能要回吗"))
          assert(text.contains("中华人民共和国劳动法"))
          assert(!text.contains("更多#劳动纠纷"))
      }
    }
  }

  test("表格归一化：全 th 行进入 thead，数据行进入 tbody，br 与合并属性保留") {
    val html = "<h1>T</h1><table><tbody><tr><th>A<br>B</th><th>C</th></tr>" +
      "<tr><td colspan='2'>x</td></tr></tbody></table>"
    val result = SimplifyHtml.extract(html, "", profiles, ExtractionOptions(forceGeneric = true))
    val doc = Jsoup.parse(result.html)
    assertEquals(doc.select("thead th").size, 2)
    assertEquals(doc.select("tbody td").size, 1)
    assertEquals(doc.select("tbody td").first.attr("colspan"), "2")
    assert(doc.select("thead th").first.html().contains("<br>"))
  }

  test("锚点与伪标题开关") {
    val html = "<h1>T</h1><h3>章节</h3><p><strong>1、小节</strong></p><p>正文内容，足够长。</p>"
    val result = SimplifyHtml.extract(html, "", profiles,
      ExtractionOptions(forceGeneric = true, anchors = true, promotePseudo = true))
    val doc = Jsoup.parse(result.html)
    assert(doc.select("[data-src]").size() > 0)
    assertEquals(doc.select("h4[data-origin=inferred]").text(), "1、小节")
  }

  test("空输入和纯脚本输入明确失败，不产出伪成功") {
    Seq("", "<html><body><script>x()</script></body></html>").foreach { html =>
      val result = SimplifyHtml.extractSafe(html, "", profiles)
      assertEquals(result.status, "failed")
      assert(result.simplifiedHtml == null)
      assert(new JSONObject(result.reportJson).getString("status") == "failed")
    }
  }

  test("正文被噪声规则清空时明确失败，不在空容器修剪中死循环") {
    val html = "<html><body><h1 id='knowledgeTitle'>标题</h1>" +
      "<div class='jd-content'><div id='body0000002496719394'>" +
      "<p class='ad'>这段广告文字足够长用于触发正文选择器但在清理时被删除</p>" +
      "</div></div></body></html>"
    val result = SimplifyHtml.extractSafe(html, "https://consumer.huawei.com/cn/support/content/x/", profiles)
    assertEquals(result.status, "failed")
    assert(result.error.contains("正文清理后无可用内容"))
  }

  test("body 类名包含 ad 时不能误删整页正文") {
    val html = "<html><head><title>文章标题</title></head><body class='float_ad_flag'>" +
      "<div class='wrapper clearfix'><div class='post_main'>" +
      "<h1 class='post_title'>文章标题</h1>" +
      "<div id='content' class='post_content'><div class='post_body'>" +
      "<p>来源：测试。这里是一段足够长的正文，包含句号，应该被通用正文定位保留。</p>" +
      "</div></div>" +
      "<div class='post_recommends'><a href='/a'>推荐一</a><a href='/b'>推荐二</a></div>" +
      "</div></div></body></html>"
    val result = SimplifyHtml.extract(html, "https://example.com/a", profiles,
      ExtractionOptions(forceGeneric = true))
    assert(result.report.status == "ok")
    assert(result.html.contains("这里是一段足够长的正文"))
  }
}
