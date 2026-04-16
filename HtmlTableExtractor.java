import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeVisitor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 从 HTML 中抽取所有 table（含嵌套），每张表转成一个 Markdown 字符串。
 *
 * 规则:
 *  - 返回 List<String>: 每个元素是一张独立的 Markdown 表格
 *  - 嵌套 <table> 作为独立条目抓取, 父单元格文本不再重复其内容
 *  - colspan / rowspan 采用 "锚点 + 其余置空" 策略, 保证列数对齐且不重复
 *  - <br> 保留为 <br> (Markdown 表格单元格不支持真换行)
 *  - <thead> 多行头按列纵向合并 (用 <br> 拼接); 无 <thead> 则取首行为表头
 *  - 单元格中的 | 转义为 \|, 反斜杠转义为 \\
 */
public class HtmlTableExtractor {

    public static List<String> extract(String html) {
        Document doc = Jsoup.parse(html);
        Elements tables = doc.select("table");
        List<String> result = new ArrayList<>();
        for (Element table : tables) {
            String md = parseTableToMarkdown(table);
            if (md != null && !md.isEmpty()) result.add(md);
        }
        return result;
    }

    private static String parseTableToMarkdown(Element table) {
        List<Element> headerTrs = new ArrayList<>();
        List<Element> bodyTrs = new ArrayList<>();
        boolean hasThead = false;

        for (Element child : table.children()) {
            String tag = child.tagName();
            if ("tr".equals(tag)) {
                bodyTrs.add(child);
            } else if ("thead".equals(tag)) {
                hasThead = true;
                for (Element inner : child.children()) {
                    if ("tr".equals(inner.tagName())) headerTrs.add(inner);
                }
            } else if ("tbody".equals(tag) || "tfoot".equals(tag)) {
                for (Element inner : child.children()) {
                    if ("tr".equals(inner.tagName())) bodyTrs.add(inner);
                }
            }
        }

        if (!hasThead && !bodyTrs.isEmpty()) {
            headerTrs.add(bodyTrs.remove(0));
        }
        if (headerTrs.isEmpty() && bodyTrs.isEmpty()) return "";

        // 串联所有 tr 一次展开, 保证 rowspan 能跨 header / body
        List<Element> allTrs = new ArrayList<>();
        allTrs.addAll(headerTrs);
        allTrs.addAll(bodyTrs);
        List<List<String>> matrix = expandMatrix(allTrs);

        int headerCount = headerTrs.size();
        List<List<String>> headerRows = matrix.subList(0, headerCount);
        List<List<String>> bodyRows   = matrix.subList(headerCount, matrix.size());

        return toMarkdown(headerRows, bodyRows);
    }

    /** Y 策略: 锚点保留值, colspan/rowspan 覆盖的其他格填空串 */
    private static List<List<String>> expandMatrix(List<Element> trs) {
        List<List<String>> grid = new ArrayList<>();
        Map<Integer, Integer> rsRemain = new HashMap<>();
        for (Element tr : trs) {
            List<String> row = new ArrayList<>();
            List<Element> cells = directCells(tr);
            int ci = 0, col = 0;
            while (ci < cells.size() || rsRemain.getOrDefault(col, 0) > 0) {
                if (rsRemain.getOrDefault(col, 0) > 0) {
                    row.add("");
                    rsRemain.put(col, rsRemain.get(col) - 1);
                    col++;
                } else {
                    Element cell = cells.get(ci++);
                    int colspan = parseSpan(cell.attr("colspan"));
                    int rowspan = parseSpan(cell.attr("rowspan"));
                    String text = extractCellText(cell);
                    for (int c = 0; c < colspan; c++) {
                        row.add(c == 0 ? text : "");
                        if (rowspan > 1) rsRemain.put(col, rowspan - 1);
                        col++;
                    }
                }
            }
            grid.add(row);
        }
        return grid;
    }

    private static List<Element> directCells(Element tr) {
        List<Element> cells = new ArrayList<>();
        for (Element child : tr.children()) {
            String tag = child.tagName();
            if ("td".equals(tag) || "th".equals(tag)) cells.add(child);
        }
        return cells;
    }

    private static int parseSpan(String val) {
        if (val == null || val.isEmpty()) return 1;
        try {
            int n = Integer.parseInt(val.trim());
            return n < 1 ? 1 : n;
        } catch (NumberFormatException e) {
            return 1;
        }
    }

    private static String toMarkdown(List<List<String>> headerRows, List<List<String>> bodyRows) {
        int maxCols = 0;
        for (List<String> r : headerRows) maxCols = Math.max(maxCols, r.size());
        for (List<String> r : bodyRows)   maxCols = Math.max(maxCols, r.size());
        if (maxCols == 0) return "";

        // 多行 header 按列合并, 空串跳过, 重复去重
        List<String> mergedHeader = new ArrayList<>();
        for (int c = 0; c < maxCols; c++) {
            List<String> parts = new ArrayList<>();
            for (List<String> row : headerRows) {
                String v = c < row.size() ? row.get(c) : "";
                if (!v.isEmpty() && !parts.contains(v)) parts.add(v);
            }
            mergedHeader.add(escapeMd(String.join("<br>", parts)));
        }

        StringBuilder sb = new StringBuilder();
        sb.append("| ").append(String.join(" | ", mergedHeader)).append(" |\n");
        sb.append("|");
        for (int i = 0; i < maxCols; i++) sb.append(" --- |");
        sb.append("\n");
        for (List<String> r : bodyRows) {
            List<String> padded = new ArrayList<>();
            for (int c = 0; c < maxCols; c++) {
                padded.add(escapeMd(c < r.size() ? r.get(c) : ""));
            }
            sb.append("| ").append(String.join(" | ", padded)).append(" |\n");
        }
        int len = sb.length();
        if (len > 0 && sb.charAt(len - 1) == '\n') sb.deleteCharAt(len - 1);
        return sb.toString();
    }

    /** Markdown 表格单元格字符转义 */
    private static String escapeMd(String s) {
        if (s == null) return "";
        String out = s.replace("\\", "\\\\").replace("|", "\\|");
        out = out.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>");
        return out.trim();
    }

    private static String extractCellText(Element cell) {
        StringBuilder sb = new StringBuilder();
        cell.traverse(new NodeVisitor() {
            boolean skip = false;
            int skipDepth = -1;

            @Override public void head(Node node, int depth) {
                if (skip) return;
                if (node instanceof Element) {
                    Element el = (Element) node;
                    String tag = el.tagName();
                    if ("table".equals(tag) && el != cell) {
                        skip = true;
                        skipDepth = depth;
                        return;
                    }
                    if ("br".equals(tag)) sb.append('\n');
                } else if (node instanceof TextNode) {
                    sb.append(((TextNode) node).text());
                }
            }

            @Override public void tail(Node node, int depth) {
                if (skip) {
                    if (depth == skipDepth) { skip = false; skipDepth = -1; }
                    return;
                }
                if (node instanceof Element) {
                    String tag = ((Element) node).tagName();
                    if ("p".equals(tag) || "div".equals(tag) || "li".equals(tag)) {
                        sb.append('\n');
                    }
                }
            }
        });
        return normalizeWhitespace(sb.toString());
    }

    private static String normalizeWhitespace(String s) {
        String[] lines = s.split("\n", -1);
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].replaceAll("[ \\t\\u00A0]+", " ").trim();
            out.append(line);
            if (i < lines.length - 1) out.append('\n');
        }
        return out.toString().replaceAll("^\\n+", "").replaceAll("\\n+$", "");
    }

    public static void main(String[] args) {
        String html =
            "<table>" +
            "  <thead>" +
            "    <tr><th colspan=\"2\">姓名</th><th rowspan=\"2\">分数</th></tr>" +
            "    <tr><th>姓</th><th>名</th></tr>" +
            "  </thead>" +
            "  <tbody>" +
            "    <tr><td>张</td><td>三</td><td>90</td></tr>" +
            "    <tr><td>李</td><td>四</td><td>85<br>(补考)</td></tr>" +
            "  </tbody>" +
            "</table>";
        List<String> tables = extract(html);
        for (int i = 0; i < tables.size(); i++) {
            System.out.println("=== Table #" + i + " ===");
            System.out.println(tables.get(i));
            System.out.println();
        }
    }
}
