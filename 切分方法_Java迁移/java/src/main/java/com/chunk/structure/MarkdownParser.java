package com.chunk.structure;

import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;

/**
 * Markdown → 结构单元。Python: parse_markdown（recall_pipeline_v2.py 第 5 段），逐行对应。
 * 识别顺序：HTML 表格（可跨行）→ 代码块 → 管道表格 → # 标题 → 整行加粗伪标题 → 列表项 → 普通段落。
 */
final class MarkdownParser {
    private final ChunkConfig cfg;

    MarkdownParser(ChunkConfig cfg) {
        this.cfg = cfg;
    }

    List<Atom> parse(String md) {
        List<Atom> atoms = new ArrayList<Atom>();
        if (md == null || md.isEmpty()) return atoms;
        List<String> lines = PyStr.split(md.replace("\r\n", "\n"), "\n");
        int i = 0, tid = 0;
        while (i < lines.size()) {
            String line = lines.get(i);
            String s = PyStr.strip(line);
            if (s.isEmpty()) {
                i++;
                continue;
            }
            // HTML 表格（可能跨多行）
            if (s.toLowerCase(Locale.ROOT).startsWith("<table")) {
                List<String> buf = new ArrayList<String>();
                buf.add(line);
                while (!lines.get(i).toLowerCase(Locale.ROOT).contains("</table>") && i + 1 < lines.size()) {
                    i++;
                    buf.add(lines.get(i));
                }
                i++;
                Document soup = Soup.parse(PyStr.join("\n", buf));
                for (Element table : Soup.descendantsByName(soup, "table")) {
                    HtmlAtomParser.TableRows tr = HtmlAtomParser.tableRows(table);
                    for (String r : tr.rows) atoms.add(Atom.row(r, tid, tr.header));
                    tid++;
                }
                continue;
            }
            // 代码块
            if (s.startsWith("```")) {
                List<String> buf = new ArrayList<String>();
                i++;
                while (i < lines.size() && !PyStr.strip(lines.get(i)).startsWith("```")) {
                    buf.add(lines.get(i));
                    i++;
                }
                i++;
                if (PyStr.hasText(PyStr.join("", buf))) atoms.add(new Atom("code", PyStr.join("\n", buf)));
                continue;
            }
            // 管道表格
            if (s.startsWith("|") && i + 1 < lines.size() && TextUtils.MD_PIPE_SEP.matcher(lines.get(i + 1)).lookingAt()) {
                List<String> hs = new ArrayList<String>();
                for (String c : PyStr.split(PyStr.strip(s, "|"), "|")) hs.add(PyStr.strip(c));
                String header = PyStr.join(" | ", hs);
                i += 2;
                while (i < lines.size() && PyStr.strip(lines.get(i)).startsWith("|")) {
                    List<String> cs = new ArrayList<String>();
                    for (String c : PyStr.split(PyStr.strip(PyStr.strip(lines.get(i)), "|"), "|")) cs.add(TextUtils.mdInline(c));
                    atoms.add(Atom.row(PyStr.join(" | ", cs), tid, TextUtils.mdInline(header)));
                    i++;
                }
                tid++;
                continue;
            }
            Matcher m = TextUtils.MD_HEADING.matcher(s);
            if (m.lookingAt()) {
                String t = TextUtils.mdInline(m.group(2));
                if (!t.isEmpty()) atoms.add(Atom.heading(t, m.group(1).length()));
                i++;
                continue;
            }
            m = TextUtils.MD_BOLD_LINE.matcher(s);
            if (m.lookingAt() && TextUtils.isPseudoHeading(TextUtils.mdInline(m.group(2)), cfg.pseudoHeadingMax)) {
                atoms.add(Atom.heading(TextUtils.mdInline(m.group(2)), 7));
                i++;
                continue;
            }
            m = TextUtils.MD_LIST.matcher(line);
            if (m.lookingAt()) {
                String t = TextUtils.mdInline(m.group(1));
                if (!t.isEmpty()) atoms.add(new Atom("li", t));
                i++;
                continue;
            }
            String t = TextUtils.mdInline(s);
            if (!t.isEmpty()) atoms.add(new Atom("para", t));
            i++;
        }
        return atoms;
    }
}
