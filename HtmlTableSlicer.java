import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Entities;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.Elements;

import java.util.ArrayList;
import java.util.List;

/**
 * 从 HTML 文本中按表格/非表格切块 (保留 HTML 标签形式).
 *
 * 主要方法:
 *  - extract(html): 返回所有 <table> 的原始 HTML (含嵌套), List<String>
 *  - segment(html): 线性切块 → Segmented {segments, flags}
 *      segments: 顶层表格的原始 HTML 与它们之间的非表格 HTML 交替
 *      flags:    0 = 非表格正文, 1 = 表格
 *
 * 规则 (segment):
 *  - 只把顶层 <table> 切出来; 嵌套表保留在父表字符串里, 不单独成段
 *  - 非表格段保留原 HTML 标签 (含 <p>/<div> 等)
 *  - 空白/纯换行的段会被丢弃
 */
public class HtmlTableSlicer {

    public static class Segmented {
        public final String[] segments;
        public final int[] flags;
        public Segmented(String[] segments, int[] flags) {
            this.segments = segments;
            this.flags = flags;
        }
    }

    public static List<String> extract(String html) {
        Document doc = Jsoup.parse(html);
        doc.outputSettings().prettyPrint(false).escapeMode(Entities.EscapeMode.xhtml);
        Elements tables = doc.select("table");
        List<String> result = new ArrayList<>(tables.size());
        for (Element table : tables) result.add(table.outerHtml());
        return result;
    }

    public static Segmented segment(String html) {
        Document doc = Jsoup.parse(html);
        doc.outputSettings().prettyPrint(false).escapeMode(Entities.EscapeMode.xhtml);

        // 找顶层表: 其祖先里没有另一个 <table>
        List<Element> topTables = new ArrayList<>();
        for (Element table : doc.select("table")) {
            if (table.parents().select("table").isEmpty()) topTables.add(table);
        }

        // 在替换前, 先把每张顶层表的字符串形态固定下来
        List<String> tableOutputs = new ArrayList<>(topTables.size());
        for (Element table : topTables) tableOutputs.add(table.outerHtml());

        // 用占位符替掉顶层表 (占位符里只有字母数字和下划线, 不会被 HTML escape)
        final String PREFIX = "__HTBL_PH_";
        final String SUFFIX = "__";
        for (int i = 0; i < topTables.size(); i++) {
            topTables.get(i).replaceWith(new TextNode(PREFIX + i + SUFFIX));
        }

        String body = doc.body() != null ? doc.body().html() : doc.html();
        return splitByPlaceholders(body, tableOutputs, PREFIX, SUFFIX);
    }

    /** 按占位符 PREFIX{i}SUFFIX 顺序切割, 拼出 Segmented */
    static Segmented splitByPlaceholders(String body, List<String> tableOutputs,
                                         String PREFIX, String SUFFIX) {
        List<String> segments = new ArrayList<>();
        List<Integer> flags = new ArrayList<>();
        String remaining = body;
        for (int i = 0; i < tableOutputs.size(); i++) {
            String marker = PREFIX + i + SUFFIX;
            int idx = remaining.indexOf(marker);
            if (idx < 0) continue;
            String before = remaining.substring(0, idx);
            if (!before.trim().isEmpty()) {
                segments.add(before);
                flags.add(0);
            }
            segments.add(tableOutputs.get(i));
            flags.add(1);
            remaining = remaining.substring(idx + marker.length());
        }
        if (!remaining.trim().isEmpty()) {
            segments.add(remaining);
            flags.add(0);
        }
        String[] segArr = segments.toArray(new String[0]);
        int[] flagArr = new int[flags.size()];
        for (int i = 0; i < flags.size(); i++) flagArr[i] = flags.get(i);
        return new Segmented(segArr, flagArr);
    }

    public static void main(String[] args) {
        String html =
            "<p>前言段落</p>" +
            "<table><tr><td>外层</td><td>" +
            "<table><tr><td>内层</td></tr></table>" +
            "</td></tr></table>" +
            "<div>表格之间的说明文字</div>" +
            "<table><tr><td>独立表 A</td><td>B</td></tr></table>" +
            "<p>结尾</p>";

        Segmented s = segment(html);
        System.out.println("共 " + s.segments.length + " 段");
        for (int i = 0; i < s.segments.length; i++) {
            System.out.println("[" + s.flags[i] + "] " + s.segments[i]);
        }
    }
}
