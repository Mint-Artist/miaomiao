package com.chunk.structure;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 在 jsoup 上复刻 Python 原版用到的 BeautifulSoup（lxml 解析器）行为。
 *
 * <p>BeautifulSoup 里有两种"取文字"的方式，原版两种都用到，结果不同：
 * <ul>
 *   <li>遍历子节点时 {@code isinstance(c, NavigableString)}：注释、doctype、处理指令、拼音注音（rt/rp）里的文字都算 → {@link #navString}</li>
 *   <li>{@code el.get_text()}：只取普通文字；注释、doctype 不算；rt / rp / script / style / template 里的文字也不算
 *       （BeautifulSoup 给它们单独的字符串类型）→ {@link #getText}</li>
 * </ul>
 * 已知无法完全复刻的地方（解析器修复残缺 html 的方式不同）见《Python与Java差异对照.md》。
 */
final class Soup {
    private Soup() {
    }

    /** 这些标签里的文字在 BeautifulSoup 里是特殊字符串类型（RubyTextString、Script 等），get_text() 时按类型过滤。 */
    private static final Set<String> SPECIAL_STRING_TAGS = new HashSet<String>(Arrays.asList("rt", "rp", "script", "style", "template"));

    /**
     * 解析 html。没有 doctype 时 jsoup 会进入 quirks 模式（例如 &lt;p&gt; 后面的 &lt;table&gt; 会被放进 p 里），
     * 而 lxml（libxml2）没有 quirks 模式、会先关闭 p。为与 lxml 一致：没有 doctype 时临时补一个 &lt;!DOCTYPE html&gt;
     * 以标准模式解析，解析完再删掉这个 doctype 节点（不让它进入正文）。
     */
    static Document parse(String html) {
        if (startsWithDoctype(html)) return Jsoup.parse(html);
        Document d = Jsoup.parse("<!DOCTYPE html>" + html);
        for (Node n : d.childNodes()) {
            if (n instanceof DocumentType) {
                n.remove();
                break;
            }
        }
        return d;
    }

    private static boolean startsWithDoctype(String html) {
        int i = 0;
        while (i < html.length() && (Character.isWhitespace(html.charAt(i)) || html.charAt(i) == '﻿')) i++;
        return html.regionMatches(true, i, "<!doctype", 0, 9);
    }

    /** soup.find_all("br") 全部替换成文字 "\n"。 */
    static void replaceBr(Element root) {
        for (Element br : root.getElementsByTag("br")) {
            br.replaceWith(new TextNode("\n"));
        }
    }

    static String name(Element e) {
        return e.normalName();
    }

    /** 是否是 BeautifulSoup 的 NavigableString（含注释、doctype 等）。 */
    static boolean isNavString(Node n) {
        return n instanceof TextNode || n instanceof DataNode || n instanceof Comment || n instanceof DocumentType;
    }

    /**
     * str(NavigableString)：节点在 BeautifulSoup 里的字符串值；返回 null 表示 lxml 不会产生这个节点（如 CDATA）。
     */
    static String navString(Node n) {
        if (n instanceof TextNode) return ((TextNode) n).getWholeText();
        if (n instanceof DataNode) return ((DataNode) n).getWholeData();
        if (n instanceof Comment) {
            String d = ((Comment) n).getData();
            if (d.startsWith("[CDATA[")) return null;   // lxml 直接丢掉 CDATA
            if (d.startsWith("?")) return d.substring(1); // <?xml ...?>：BeautifulSoup 是 ProcessingInstruction，值不含开头的 ?
            return d;
        }
        if (n instanceof DocumentType) {  // Doctype.for_name_and_ids
            DocumentType dt = (DocumentType) n;
            String v = dt.name();
            String pub = dt.publicId(), sys = dt.systemId();
            if (!pub.isEmpty()) {
                v += " PUBLIC \"" + pub + "\"";
                if (!sys.isEmpty()) v += " \"" + sys + "\"";
            } else if (!sys.isEmpty()) {
                v += " SYSTEM \"" + sys + "\"";
            }
            return v;
        }
        return null;
    }

    /** 文字节点在 BeautifulSoup 里的字符串类型：最近的特殊祖先标签名，没有则为 ""。 */
    private static String stringType(Node text) {
        Node pn = text.parentNode();
        for (Element p = pn instanceof Element ? (Element) pn : null; p != null; p = p.parent()) {
            if (SPECIAL_STRING_TAGS.contains(p.normalName())) return p.normalName();
        }
        return "";
    }

    private static void collectText(Element el, List<String> out) {
        String want = SPECIAL_STRING_TAGS.contains(el.normalName()) ? el.normalName() : "";
        collect(el, want, out);
    }

    private static void collect(Node n, String want, List<String> out) {
        for (Node c : n.childNodes()) {
            if (c instanceof TextNode) {
                if (stringType(c).equals(want)) out.add(((TextNode) c).getWholeText());
            } else if (c instanceof Element) {
                collect(c, want, out);
            }
        }
    }

    /** el.get_text() */
    static String getText(Element el) {
        List<String> xs = new ArrayList<String>();
        collectText(el, xs);
        return PyStr.join("", xs);
    }

    /** el.get_text(separator) */
    static String getText(Element el, String sep) {
        List<String> xs = new ArrayList<String>();
        collectText(el, xs);
        return PyStr.join(sep, xs);
    }

    /** el.find_parent(name)：最近的同名祖先（不含自己）。 */
    static Element findParent(Element el, String name) {
        for (Element p = el.parent(); p != null; p = p.parent()) {
            if (p.normalName().equals(name)) return p;
        }
        return null;
    }

    /** el.find_all(names, recursive=False)：直接子元素。 */
    static List<Element> childrenByName(Element el, String... names) {
        Set<String> want = new HashSet<String>(Arrays.asList(names));
        List<Element> out = new ArrayList<Element>();
        for (Element c : el.children()) {
            if (want.contains(c.normalName())) out.add(c);
        }
        return out;
    }

    /** el.find_all(name)：全部后代（不含自己），文档顺序。 */
    static List<Element> descendantsByName(Element el, String name) {
        List<Element> out = new ArrayList<Element>();
        for (Element e : el.getElementsByTag(name)) {
            if (e != el) out.add(e);
        }
        return out;
    }

    /** el.find(name)：第一个后代（不含自己）。 */
    static Element findFirst(Element el, String name) {
        for (Element e : el.getElementsByTag(name)) {
            if (e != el) return e;
        }
        return null;
    }

    /** set(el.get("class") or []) */
    static Set<String> classes(Element el) {
        return el.classNames();
    }
}
