package com.chunk.structure;

import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static com.chunk.structure.HtmlAtomParser.H_TAGS;
import static com.chunk.structure.HtmlAtomParser.INLINE_TAGS;
import static com.chunk.structure.HtmlAtomParser.SKIP_TAGS;

/**
 * html → 容器级结构单元（html_container 用）。Python: block_text / is_layout_table / parse_html_blocks（chunk_container.py）。
 */
final class HtmlBlockParser {
    static final Set<String> BLOCK_TAGS = new HashSet<String>(Arrays.asList(
            "p", "div", "section", "article", "main", "aside", "header", "footer", "nav", "li", "ul", "ol",
            "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "figure", "figcaption", "table", "thead",
            "tbody", "tfoot", "tr", "caption", "dl", "dt", "dd", "details", "summary", "hr"));

    private final ChunkConfig cfg;

    HtmlBlockParser(ChunkConfig cfg) {
        this.cfg = cfg;
    }

    /** 元素文字：块级子元素之间换行，行内元素连在一起。Python: block_text */
    static String blockText(Element el) {
        List<StringBuilder> lines = new ArrayList<StringBuilder>();
        lines.add(new StringBuilder());
        rec(el, lines);
        List<String> xs = new ArrayList<String>();
        for (StringBuilder b : lines) xs.add(b.toString());
        return TextUtils.cleanInline(PyStr.join("\n", xs));
    }

    private static void rec(Node n, List<StringBuilder> lines) {
        for (Node c : n.childNodes()) {
            if (Soup.isNavString(c)) {
                String s = Soup.navString(c);
                if (s != null) lines.get(lines.size() - 1).append(s);
            } else if (c instanceof Element && !SKIP_TAGS.contains(Soup.name((Element) c))) {
                if (BLOCK_TAGS.contains(Soup.name((Element) c))) {
                    lines.add(new StringBuilder());
                    rec(c, lines);
                    lines.add(new StringBuilder());
                } else {
                    rec(c, lines);
                }
            }
        }
    }

    private static List<String> linesOf(Element el) {
        List<String> out = new ArrayList<String>();
        for (String x : PyStr.split(blockText(el), "\n")) {
            if (!x.isEmpty()) out.add(x);
        }
        return out;
    }

    /** 排版用的表格：没有表头，且每一行最多 1 个非空单元格。Python: is_layout_table */
    static boolean isLayoutTable(Element table, String header) {
        if (!header.isEmpty()) return false;
        for (Element tr : Soup.descendantsByName(table, "tr")) {
            if (Soup.findParent(tr, "table") != table) continue;
            int n = 0;
            for (Element td : Soup.childrenByName(tr, "td", "th")) {
                if (PyStr.hasText(TextUtils.cleanInline(Soup.getText(td)))) n++;
            }
            if (n > 1) return false;
        }
        return true;
    }

    List<Unit> parse(String html) {
        List<Unit> units = new ArrayList<Unit>();
        if (html == null || html.isEmpty()) return units;
        Document soup = Soup.parse(html);
        Soup.replaceBr(soup);
        new Walker(units).walk(soup, 0);
        return units;
    }

    private final class Walker {
        final List<Unit> units;
        final StringBuilder buf = new StringBuilder();

        Walker(List<Unit> units) {
            this.units = units;
        }

        void flush() {
            String t = TextUtils.cleanInline(buf.toString());
            buf.setLength(0);
            if (!t.isEmpty()) units.add(Unit.para(t));
        }

        void handle(Element c, int depth) {
            String name = Soup.name(c);
            Set<String> classes = Soup.classes(c);
            if (depth > 200) {
                buf.append(Soup.getText(c));
            } else if (H_TAGS.containsKey(name)) {
                flush();
                String t = blockText(c);
                if (!t.isEmpty() && PyStr.len(t) > cfg.ctHeadMax) {  // 标签没闭合好时"标题"会吞进大段正文，按段落处理
                    units.add(Unit.para(t));
                } else if (!t.isEmpty()) {
                    units.add(Unit.heading(t, H_TAGS.get(name)));
                }
            } else if (name.equals("p")) {
                flush();
                String t = blockText(c);
                if (!t.isEmpty() && HtmlAtomParser.onlyBoldChild(c) != null && TextUtils.isPseudoHeading(t, cfg.pseudoHeadingMax)) {
                    units.add(Unit.heading(t, 7));
                } else if (!t.isEmpty()) {
                    units.add(Unit.para(t));
                }
            } else if (name.equals("ul") || name.equals("ol")) {
                flush();
                List<Element> lis = Soup.childrenByName(c, "li");
                List<String> items = new ArrayList<String>();
                if (!lis.isEmpty()) {
                    for (Element li : lis) {
                        String x = blockText(li);
                        if (!x.isEmpty()) items.add(x);
                    }
                } else {
                    items = linesOf(c);
                }
                if (!items.isEmpty()) units.add(Unit.container("list", items));
            } else if (name.equals("table")) {
                flush();
                HtmlAtomParser.TableRows tr = HtmlAtomParser.tableRows(c);
                Element capEl = Soup.findFirst(c, "caption");
                String cap = capEl != null ? TextUtils.cleanInline(Soup.getText(capEl)).replace("\n", " ") : "";
                if (isLayoutTable(c, tr.header)) {  // 排版用的单列表格（论坛帖子、评论常见）：每行按段落处理
                    for (Element row : Soup.descendantsByName(c, "tr")) {
                        if (Soup.findParent(row, "table") == c) {
                            String t = blockText(row);
                            if (!t.isEmpty()) units.add(Unit.para(t));
                        }
                    }
                } else if (!tr.rows.isEmpty() || !tr.header.isEmpty()) {
                    units.add(new Unit("table", tr.rows, 0, tr.header, cap));
                }
            } else if (name.equals("pre")) {
                flush();
                String t = PyStr.strip(Soup.getText(c), "\n");
                if (PyStr.hasText(t)) units.add(Unit.container("code", PyStr.split(t, "\n")));
            } else if (name.equals("blockquote") || name.equals("figure")) {
                flush();
                List<String> items = linesOf(c);
                if (!items.isEmpty()) units.add(Unit.container(name.equals("blockquote") ? "quote" : "figure", items));
            } else if (name.equals("div") && (intersects(classes, cfg.noteClasses) || intersects(classes, cfg.metaClasses))) {
                flush();
                List<String> items = linesOf(c);
                if (!items.isEmpty()) units.add(Unit.container(intersects(classes, cfg.metaClasses) ? "metadata" : "note", items));
            } else if (INLINE_TAGS.contains(name)) {
                buf.append(Soup.getText(c));
            } else {  // 其他 div / section 等外壳：作为段落边界，向内展开
                flush();
                walk(c, depth + 1);
                flush();
            }
        }

        void walk(Node node, int depth) {
            for (Node c : new ArrayList<Node>(node.childNodes())) {
                if (Soup.isNavString(c)) {
                    String s = Soup.navString(c);
                    if (s != null) buf.append(s);
                } else if (c instanceof Element && !SKIP_TAGS.contains(Soup.name((Element) c))) {
                    handle((Element) c, depth);
                }
            }
        }
    }

    private static boolean intersects(Set<String> a, Set<String> b) {
        for (String x : a) {
            if (b.contains(x)) return true;
        }
        return false;
    }
}
