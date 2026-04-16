import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Entities;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeVisitor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 从 HTML 中抽取表格转成 Markdown. 合并单元格采用 Y 策略 (锚点 + 其余留空).
 *
 * 主要方法:
 *  - extract(html): 所有 <table> (含嵌套) 各转一份 Markdown, List<String>
 *  - segment(html): 线性切块 → Segmented {segments, flags}
 *      segments: 顶层表的 Markdown 与非表格 HTML 段交替出现
 *      flags:    0 = 非表格正文(保留 HTML), 1 = 表格 Markdown
 *
 * segment() 的行为:
 *  - 只切顶层 <table>; 嵌套表不单独成段, 其文本内容会被扁平化进父表对应单元格
 *  - 非表格段保留原 HTML 标签
 *  - 空白段丢弃
 */
public class HtmlTableExtractor {

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
        Elements tables = doc.select("table");
        List<String> result = new ArrayList<>();
        for (Element table : tables) {
            String md = parseTableToMarkdown(table, true);
            if (md != null && !md.isEmpty()) result.add(md);
        }
        return result;
    }

    public static Segmented segment(String html) {
        Document doc = Jsoup.parse(html);
        doc.outputSettings().prettyPrint(false).escapeMode(Entities.EscapeMode.xhtml);

        List<Element> topTables = new ArrayList<>();
        for (Element table : doc.select("table")) {
            if (table.parents().select("table").isEmpty()) topTables.add(table);
        }

        // 先把每张顶层表的 Markdown 形态固定下来 (此时嵌套表还在, 被扁平化进单元格)
        List<String> tableOutputs = new ArrayList<>(topTables.size());
        for (Element table : topTables) {
            String md = parseTableToMarkdown(table, false);
            tableOutputs.add(md == null ? "" : md);
        }

        final String PREFIX = "__HTBL_PH_";
        final String SUFFIX = "__";
        for (int i = 0; i < topTables.size(); i++) {
            topTables.get(i).replaceWith(new TextNode(PREFIX + i + SUFFIX));
        }

        String body = doc.body() != null ? doc.body().html() : doc.html();

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

    private static String parseTableToMarkdown(Element table, boolean skipNestedInCells) {
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

        List<Element> allTrs = new ArrayList<>();
        allTrs.addAll(headerTrs);
        allTrs.addAll(bodyTrs);
        List<List<String>> matrix = expandMatrix(allTrs, skipNestedInCells);

        int headerCount = headerTrs.size();
        List<List<String>> headerRows = matrix.subList(0, headerCount);
        List<List<String>> bodyRows   = matrix.subList(headerCount, matrix.size());

        return toMarkdown(headerRows, bodyRows);
    }

    /** Y 策略: 锚点保留值, colspan/rowspan 覆盖的其他格填空串 */
    private static List<List<String>> expandMatrix(List<Element> trs, boolean skipNestedInCells) {
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
                    String text = extractCellText(cell, skipNestedInCells);
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

    private static String escapeMd(String s) {
        if (s == null) return "";
        String out = s.replace("\\", "\\\\").replace("|", "\\|");
        out = out.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>");
        return out.trim();
    }

    /**
     * 提取单元格文本.
     * @param skipNestedTables true: 跳过嵌套表 (用于 extract, 嵌套表会独立成项)
     *                         false: 把嵌套表内容扁平化进来 (用于 segment, 嵌套表不独立)
     */
    private static String extractCellText(Element cell, boolean skipNestedTables) {
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
                        if (!skipNestedTables) {
                            sb.append(' ').append(el.text()).append(' ');
                        }
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
            "<p>前言段落</p>" +
            "<table><thead><tr><th>A</th><th>B</th></tr></thead>" +
            "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>" +
            "<div>中间说明</div>" +
            "<table><tr><td>X</td><td>Y</td></tr></table>" +
            "<p>结尾</p>";

        Segmented s = segment(html);
        System.out.println("共 " + s.segments.length + " 段");
        for (int i = 0; i < s.segments.length; i++) {
            System.out.println("--- [" + s.flags[i] + "] ---");
            System.out.println(s.segments[i]);
        }
    }
}
