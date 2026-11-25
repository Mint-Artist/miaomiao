package com.chunk.structure;

import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * html → 结构单元（t512 / 吸附版用）。Python: parse_html、table_rows_from_soup（recall_pipeline_v2.py 第 4、6 段）。
 */
final class HtmlAtomParser {
    static final Map<String, Integer> H_TAGS = new HashMap<String, Integer>();
    static {
        for (int i = 1; i <= 6; i++) H_TAGS.put("h" + i, i);
    }
    static final Set<String> SKIP_TAGS = new HashSet<String>(Arrays.asList(
            "script", "style", "noscript", "template", "svg", "iframe", "head", "title", "meta", "link"));
    static final Set<String> INLINE_TAGS = new HashSet<String>(Arrays.asList(
            "a", "abbr", "b", "big", "cite", "code", "em", "font", "i", "img", "label", "mark", "q", "s", "small",
            "span", "strike", "strong", "sub", "sup", "time", "u", "br", "del", "ins"));

    private final ChunkConfig cfg;

    HtmlAtomParser(ChunkConfig cfg) {
        this.cfg = cfg;
    }

    /** 表格行与表头。Python: table_rows_from_soup 的返回值 */
    static final class TableRows {
        final List<String> rows;
        final String header;

        TableRows(List<String> rows, String header) {
            this.rows = rows;
            this.header = header;
        }
    }

    /** 表格 → 每行一个 "单元格 | 单元格"；第一行在 thead 里或全是 th 时作为表头。Python: table_rows_from_soup */
    static TableRows tableRows(Element table) {
        List<String> rows = new ArrayList<String>();
        String header = "";
        for (Element tr : Soup.descendantsByName(table, "tr")) {
            if (Soup.findParent(tr, "table") != table) continue;  // 嵌套表格的行交给内层表格处理
            List<Element> tds = Soup.childrenByName(tr, "td", "th");
            List<String> cells = new ArrayList<String>();
            boolean any = false;
            for (Element c : tds) {
                String x = TextUtils.cleanInline(Soup.getText(c, "\n")).replace("\n", " ");
                cells.add(x);
                if (!x.isEmpty()) any = true;
            }
            if (!any) continue;
            String text = PyStr.join(" | ", cells);
            boolean allTh = true;
            for (Element c : tds) {
                if (!Soup.name(c).equals("th")) {
                    allTh = false;
                    break;
                }
            }
            boolean isHead = Soup.findParent(tr, "thead") != null || allTh;
            if (isHead && rows.isEmpty() && header.isEmpty()) header = text;
            else rows.add(text);
        }
        return new TableRows(rows, header);
    }

    List<Atom> parse(String html) {
        List<Atom> atoms = new ArrayList<Atom>();
        if (html == null || html.isEmpty()) return atoms;
        Document soup = Soup.parse(html);
        Soup.replaceBr(soup);
        new Walker(atoms).walk(soup, 0);
        return atoms;
    }

    private final class Walker {
        final List<Atom> atoms;
        final StringBuilder buf = new StringBuilder();
        int tid = 0;

        Walker(List<Atom> atoms) {
            this.atoms = atoms;
        }

        void flush() {
            String t = TextUtils.cleanInline(buf.toString());
            buf.setLength(0);
            if (!t.isEmpty()) atoms.add(new Atom("para", t));
        }

        /** li 自己的文字（不含子列表 / 子表格）。Python: own_text */
        String ownText(Element el) {
            StringBuilder parts = new StringBuilder();
            for (Node c : el.childNodes()) {
                if (Soup.isNavString(c)) {
                    String s = Soup.navString(c);
                    if (s != null) parts.append(s);
                } else if (c instanceof Element) {
                    String n = Soup.name((Element) c);
                    if (!n.equals("ul") && !n.equals("ol") && !n.equals("table") && !SKIP_TAGS.contains(n)) {
                        parts.append(Soup.getText((Element) c));
                    }
                }
            }
            return TextUtils.cleanInline(parts.toString());
        }

        void handle(Element c, int depth) {
            String name = Soup.name(c);
            if (depth > 200) {  // 极深嵌套直接取文字
                buf.append(Soup.getText(c));
            } else if (H_TAGS.containsKey(name)) {
                flush();
                String t = TextUtils.cleanInline(Soup.getText(c));
                if (!t.isEmpty()) atoms.add(Atom.heading(t, H_TAGS.get(name)));
            } else if (name.equals("p")) {
                flush();
                String t = TextUtils.cleanInline(Soup.getText(c));
                Element only = onlyBoldChild(c);
                if (!t.isEmpty() && only != null && TextUtils.isPseudoHeading(t, cfg.pseudoHeadingMax)) {
                    atoms.add(Atom.heading(t, 7));  // 整段加粗的短行视为小标题
                } else if (!t.isEmpty()) {
                    atoms.add(new Atom("para", t));
                }
            } else if (name.equals("ul") || name.equals("ol")) {
                flush();
                for (Element li : Soup.childrenByName(c, "li")) {
                    String t = ownText(li);
                    if (!t.isEmpty()) atoms.add(new Atom("li", t));
                    for (Element sub : Soup.childrenByName(li, "ul", "ol")) handle(sub, depth + 1);
                }
            } else if (name.equals("table")) {  // 嵌套表格的文字已包含在外层单元格里，不再单独处理
                flush();
                TableRows tr = tableRows(c);
                for (String r : tr.rows) atoms.add(Atom.row(r, tid, tr.header));
                tid++;
            } else if (name.equals("pre")) {
                flush();
                String t = Soup.getText(c);
                if (PyStr.hasText(t)) atoms.add(new Atom("code", PyStr.strip(t, "\n")));
            } else if (INLINE_TAGS.contains(name)) {
                buf.append(Soup.getText(c));
            } else {  // div / section / blockquote 等容器：作为段落边界
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

    /**
     * p 去掉纯空白文字节点后只剩一个子节点、且它是 strong / b 时返回它，否则 null。
     * Python: kids = [k for k in c.children if not (isinstance(k, NavigableString) and not k.strip())]
     */
    static Element onlyBoldChild(Element p) {
        Node kid = null;
        int n = 0;
        for (Node k : p.childNodes()) {
            if (Soup.isNavString(k)) {
                String s = Soup.navString(k);
                if (s == null || !PyStr.hasText(s)) continue;
            } else if (!(k instanceof Element)) {
                continue;
            }
            n++;
            kid = k;
        }
        if (n != 1 || !(kid instanceof Element)) return null;
        String nm = Soup.name((Element) kid);
        return nm.equals("strong") || nm.equals("b") ? (Element) kid : null;
    }
}
