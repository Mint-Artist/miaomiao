package extractor

import java.nio.file.{Files, Path}

import org.json.{JSONArray, JSONObject}

import scala.collection.JavaConverters._

final case class MetadataRule(selector: String, label: Option[String])
final case class Profile(
    name: String,
    urlContains: List[String],
    title: List[String],
    content: List[String],
    metadata: List[MetadataRule],
    strip: List[String],
    noise: List[String])

object Profiles {
  private def strings(array: JSONArray): List[String] =
    array.iterator().asScala.map(_.toString).toList

  private def parseProfile(value: JSONObject): Profile = {
    val metadata = value.getJSONArray("metadata").iterator().asScala.map { item =>
      val rule = item.asInstanceOf[JSONObject]
      MetadataRule(rule.getString("selector"),
        if (rule.has("label")) Some(rule.getString("label")) else None)
    }.toList
    Profile(
      name = value.getString("name"),
      urlContains = strings(value.getJSONArray("url_contains")),
      title = strings(value.getJSONArray("title")),
      content = strings(value.getJSONArray("content")),
      metadata = metadata,
      strip = strings(value.getJSONArray("strip")),
      noise = strings(value.getJSONArray("noise")))
  }

  def parse(json: String): List[Profile] =
    new JSONArray(json).iterator().asScala.map(v => parseProfile(v.asInstanceOf[JSONObject])).toList

  def load(path: Path): List[Profile] =
    parse(new String(Files.readAllBytes(path), java.nio.charset.StandardCharsets.UTF_8))

  def byUrl(profiles: List[Profile], url: String): Option[Profile] =
    profiles.find(p => p.urlContains.exists(token => url != null && url.contains(token)))

  def byName(profiles: List[Profile], name: String): Profile =
    profiles.find(_.name == name).getOrElse(
      throw new IllegalArgumentException(s"未知 profile: $name；可选：${profiles.map(_.name).mkString(", ")}"))
}
