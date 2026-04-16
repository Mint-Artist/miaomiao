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

public class HtmlTableExtractor {

    public static List<List<List<String>>> extract(String html) {
        Document doc = Jsoup.parse(html);
        Elements tables = doc.select("table");
        List<List<List<String>>> result = new ArrayList<>();
        for (Element table : tables) {
            result.add(parseTable(table));
        }
        return result;
    }

    private static List<List<String>> parseTable(Element table) {
        List<List<String>> grid = new ArrayList<>();
        Map<Integer, Integer> rsRemain = new HashMap<>();
        Map<Integer, String>  rsValue  = new HashMap<>();

        for (Element tr : directRows(table)) {
            List<String> row = new ArrayList<>();
            List<Element> cells = directCells(tr);
            int ci = 0, col = 0;

            while (ci < cells.size() || rsRemain.getOrDefault(col, 0) > 0) {
                if (rsRemain.getOrDefault(col, 0) > 0) {
                    row.add(rsValue.get(col));
                    rsRemain.put(col, rsRemain.get(col) - 1);
                    col++;
                } else {
                    Element cell = cells.get(ci++);
                    int colspan = parseSpan(cell.attr("colspan"));
                    int rowspan = parseSpan(cell.attr("rowspan"));
                    String text = extractCellText(cell);
                    for (int c = 0; c < colspan; c++) {
                        row.add(text);
                        if (rowspan > 1) {
                            rsRemain.put(col, rowspan - 1);
                            rsValue.put(col, text);
                        }
                        col++;
                    }
                }
            }
            grid.add(row);
        }
        return grid;
    }

    private static List<Element> directRows(Element table) {
        List<Element> rows = new ArrayList<>();
        for (Element child : table.children()) {
            String tag = child.tagName();
            if ("tr".equals(tag)) {
                rows.add(child);
            } else if ("thead".equals(tag) || "tbody".equals(tag) || "tfoot".equals(tag)) {
                for (Element inner : child.children()) {
                    if ("tr".equals(inner.tagName())) rows.add(inner);
                }
            }
        }
        return rows;
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
        return normalize(sb.toString());
    }

    private static String normalize(String s) {
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
        String html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>";
        List<List<List<String>>> tables = extract(html);
        for (int t = 0; t < tables.size(); t++) {
            System.out.println("=== Table #" + t + " (rows=" + tables.get(t).size() + ") ===");
            for (List<String> row : tables.get(t)) System.out.println(row);
        }
    }
}
