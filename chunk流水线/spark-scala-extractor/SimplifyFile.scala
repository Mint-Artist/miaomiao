package extractor

import java.nio.file.{Files, Path, Paths}

/** 单文件调试入口：scala-cli run . --main-class extractor.SimplifyFile -- raw.html out.html --url URL */
object SimplifyFile {
  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("用法: raw.html out.html [--url URL] [--profiles profiles.json] [--generic] [--anchors] [--promote-pseudo] [--report report.json]")
      sys.exit(2)
    }
    var url = ""
    var profilesPath = Paths.get("profiles.json")
    var reportPath: Option[Path] = None
    var forceGeneric = false
    var anchors = false
    var promote = false
    var i = 2
    while (i < args.length) {
      args(i) match {
        case "--url" => url = args(i + 1); i += 2
        case "--profiles" => profilesPath = Paths.get(args(i + 1)); i += 2
        case "--report" => reportPath = Some(Paths.get(args(i + 1))); i += 2
        case "--generic" => forceGeneric = true; i += 1
        case "--anchors" => anchors = true; i += 1
        case "--promote-pseudo" => promote = true; i += 1
        case other => System.err.println(s"未知参数: $other"); sys.exit(2)
      }
    }
    val profiles = Profiles.load(profilesPath)
    val html = new String(Files.readAllBytes(Paths.get(args(0))), java.nio.charset.StandardCharsets.UTF_8)
    val result = SimplifyHtml.extract(html, url, profiles,
      ExtractionOptions(anchors = anchors, promotePseudo = promote, forceGeneric = forceGeneric))
    Files.write(Paths.get(args(1)), result.html.getBytes(java.nio.charset.StandardCharsets.UTF_8))
    reportPath.foreach(path => Files.write(path, result.report.toJson.getBytes(java.nio.charset.StandardCharsets.UTF_8)))
    System.err.println(s"status=${result.report.status} confidence=${result.report.confidence} " +
      s"profile=${result.report.profile} mode=${result.report.extractionMode} " +
      s"chars=${result.report.textChars} warnings=${result.report.warnings.size}")
  }
}
