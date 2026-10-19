package extractor

import java.nio.file.Paths

import org.apache.spark.sql.api.java.UDF2
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, lit, udf}

/** Spark 批处理：输入含完整 HTML 字符串的表，输出 simplified_html 字符串及状态报告。 */
object SimplifyHtmlJob {
  private final case class Args(
      input: String = "", output: String = "", inputFormat: String = "json",
      outputFormat: String = "parquet", htmlCol: String = "html", urlCol: String = "url",
      profiles: String = "profiles.json", mode: String = "errorifexists",
      partitions: Int = 0, maxInputMb: Int = 20, keepOriginal: Boolean = false,
      forceGeneric: Boolean = false, anchors: Boolean = false)

  private def parseArgs(argv: Array[String]): Args = {
    var args = Args()
    var i = 0
    while (i < argv.length) {
      def next(): String = { val value = argv(i + 1); i += 1; value }
      argv(i) match {
        case "--input" => args = args.copy(input = next())
        case "--output" => args = args.copy(output = next())
        case "--input-format" => args = args.copy(inputFormat = next())
        case "--output-format" => args = args.copy(outputFormat = next())
        case "--html-col" => args = args.copy(htmlCol = next())
        case "--url-col" => args = args.copy(urlCol = next())
        case "--profiles" => args = args.copy(profiles = next())
        case "--mode" => args = args.copy(mode = next())
        case "--partitions" => args = args.copy(partitions = next().toInt)
        case "--max-input-mb" => args = args.copy(maxInputMb = next().toInt)
        case "--keep-original" => args = args.copy(keepOriginal = true)
        case "--generic" => args = args.copy(forceGeneric = true)
        case "--anchors" => args = args.copy(anchors = true)
        case other => throw new IllegalArgumentException(s"未知参数: $other")
      }
      i += 1
    }
    require(args.input.nonEmpty && args.output.nonEmpty, "--input 与 --output 必填")
    args
  }

  private val ResultSchema = StructType(Seq(
    StructField("simplified_html", StringType, nullable = true),
    StructField("report_json", StringType, nullable = false),
    StructField("status", StringType, nullable = false),
    StructField("confidence", DoubleType, nullable = false),
    StructField("profile", StringType, nullable = false),
    StructField("content_root", StringType, nullable = false),
    StructField("error", StringType, nullable = true)))

  def main(argv: Array[String]): Unit = {
    val args = parseArgs(argv)
    val spark = SparkSession.builder().appName("simplify-html").getOrCreate()
    try {
      val profiles = Profiles.load(Paths.get(args.profiles))
      val broadcast = spark.sparkContext.broadcast(profiles)
      val options = ExtractionOptions(anchors = args.anchors, forceGeneric = args.forceGeneric)
      val maxChars = args.maxInputMb.toLong * 1024L * 1024L

      val extractUdf = udf(new UDF2[String, String, Row] {
        override def call(html: String, url: String): Row = {
          val result =
            if (html != null && html.length > maxChars)
              SafeExtraction(null,
                s"""{"extractor_version":"${SimplifyHtml.ExtractorVersion}","status":"failed","error":"input exceeds ${args.maxInputMb} MB"}""",
                "failed", 0.0, "unknown", "", s"input exceeds ${args.maxInputMb} MB")
            else SimplifyHtml.extractSafe(html, url, broadcast.value, options)
          Row(result.simplifiedHtml, result.reportJson, result.status,
            result.confidence, result.profile, result.contentRoot, result.error)
        }
      }, ResultSchema)

      val input = spark.read.format(args.inputFormat).load(args.input)
      require(input.columns.contains(args.htmlCol), s"缺少 HTML 列: ${args.htmlCol}")
      val urlColumn = if (input.columns.contains(args.urlCol)) col(args.urlCol) else lit("")
      val extracted = input.withColumn("_extraction", extractUdf(col(args.htmlCol), urlColumn))
      val original = input.columns.filter(c => args.keepOriginal || c != args.htmlCol).map(col)
      val output = extracted.select(original ++ Seq(
        col("_extraction.simplified_html").as("simplified_html"),
        col("_extraction.report_json").as("report_json"),
        col("_extraction.status").as("status"),
        col("_extraction.confidence").as("confidence"),
        col("_extraction.profile").as("profile"),
        col("_extraction.content_root").as("content_root"),
        col("_extraction.error").as("error")): _*)
      val written = if (args.partitions > 0) output.repartition(args.partitions) else output
      written.write.format(args.outputFormat).mode(args.mode).save(args.output)

      val counts = written.groupBy("status").count().collect()
      println(s"输出: ${args.output}")
      counts.foreach(row => println(s"status=${row.getString(0)} count=${row.getLong(1)}"))
    } finally spark.stop()
  }
}
