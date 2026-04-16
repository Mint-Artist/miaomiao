import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Entities;
import org.jsoup.select.Elements;

import java.util.ArrayList;
import java.util.List;

/**
 * 从 HTML 文本中切出所有 <table>...</table> 片段, 保留原始 HTML 标签.
 *
 * 与 HtmlTableExtractor 的区别:
 *  - HtmlTableExtractor: 把每张表转成 Markdown 字符串 (加工)
 *  - HtmlTableSlicer:    只把每张表的 HTML 原样切出来 (不加工)
 *
 * 规则:
 *  - 返回 List<String>: 每个元素是一张表的完整 HTML (含 <table> 标签本身)
 *  - 顺序: 按文档出现先后
 *  - 嵌套表: 父表 + 嵌套表都独立成项 (同一段嵌套 HTML 会出现两次:
 *    一次在父表字符串内部, 一次作为独立条目)
 *  - 不做结构改写: 不展开 colspan/rowspan, 不转 Markdown, 不丢标签
 *  - 关闭了 Jsoup 的 pretty-print, 尽量贴近源文本形态
 */
public class HtmlTableSlicer {

    public static List<String> extract(String html) {
        Document doc = Jsoup.parse(html);
        doc.outputSettings()
           .prettyPrint(false)
           .escapeMode(Entities.EscapeMode.xhtml);

        Elements tables = doc.select("table");
        List<String> result = new ArrayList<>(tables.size());
        for (Element table : tables) {
            result.add(table.outerHtml());
        }
        return result;
    }

    public static void main(String[] args) {
        String html =
            "<p>正文前</p>" +
            "<table><tr><td>外层1</td><td>" +
            "<table><tr><td>内层</td></tr></table>" +
            "</td></tr></table>" +
            "<p>中间段落</p>" +
            "<table><tr><td>独立表 A</td><td>B</td></tr></table>";

        List<String> tables = extract(html);
        System.out.println("共抽取到 " + tables.size() + " 张表");
        for (int i = 0; i < tables.size(); i++) {
            System.out.println("--- Table #" + i + " ---");
            System.out.println(tables.get(i));
        }
    }
}
