package com.chunk.structure;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 小节打包 + 相邻小节合并：md/html_struct_secmerge_t512 及其标题吸附版（_adh）。
 * Python: chunk_struct_sections_adh（adhere=false 时与 chunk_struct_sections_t 逐字一致）、merge_sections、
 * chunk_secmerge_t / chunk_secmerge_adh。
 */
final class SectionChunker {
    private final ChunkConfig cfg;
    private final TitleDetector td;

    SectionChunker(ChunkConfig cfg, TitleDetector td) {
        this.cfg = cfg;
        this.td = td;
    }

    /** 打包结果：块文本、上级标题路径（标题在 atoms 里的序号）、是否以标题开头。 */
    static final class Rec {
        final String text;
        final List<Integer> parent;
        final boolean head;

        Rec(String text, List<Integer> parent, boolean head) {
            this.text = text;
            this.parent = parent;
            this.head = head;
        }
    }

    /** 完整流程：打包 → 合并 → 过滤 &lt; minLen。adhere=false 即 t512，true 即 t512_adh。 */
    List<String> chunk(List<Atom> atoms, boolean adhere) {
        List<String> out = new ArrayList<String>();
        for (String t : mergeSections(sections(atoms, adhere), cfg.soft)) {
            if (PyStr.lenNoNewline(t) >= cfg.minLen) out.add(t);
        }
        return out;
    }

    private static String family(Atom a) {
        if (a.kind.equals("row")) return "row:" + a.tableId;
        if (a.kind.equals("li")) return "li";
        return "text";
    }

    /** Python: chunk_struct_sections_adh(atoms, soft, hard, table_soft, adhere) */
    List<Rec> sections(List<Atom> atoms, boolean adhere) {
        final int soft = cfg.soft, hard = cfg.hard, tableSoft = cfg.tableSoft;
        List<Rec> recs = new ArrayList<Rec>();
        List<String> cur = new ArrayList<String>();
        List<TitleDetector.Tl> curTl = new ArrayList<TitleDetector.Tl>();
        String curFamily = null;
        List<Integer> curParent = Collections.emptyList();
        boolean curHead = false;
        List<int[]> path = new ArrayList<int[]>();            // [(level, 序号)]
        List<Object[]> pending = new ArrayList<Object[]>();   // [(序号, 标题文字)]

        for (int i = 0; i < atoms.size(); i++) {
            Atom a = atoms.get(i);
            if (a.kind.equals("heading")) {
                // flush()
                if (!cur.isEmpty()) recs.add(new Rec(PyStr.join("\n", cur), curParent, curHead));
                cur = new ArrayList<String>();
                curTl = new ArrayList<TitleDetector.Tl>();
                curFamily = null;
                while (!path.isEmpty() && path.get(path.size() - 1)[0] >= a.level) path.remove(path.size() - 1);
                path.add(new int[]{a.level, i});
                pending.add(new Object[]{i, a.text});
                continue;
            }
            String family = family(a);
            int lim = a.kind.equals("row") ? tableSoft : soft;
            boolean whole = PyStr.len(a.text) <= hard;
            List<String> pieces = whole ? Collections.singletonList(a.text) : TextUtils.splitLong(a.text, lim);
            for (String piece : pieces) {
                TitleDetector.Tl pieceTl = a.kind.equals("para") && whole ? td.tlInfo(piece) : null;
                int size = PyStr.sumLen(cur) + cur.size();
                List<String> carry = new ArrayList<String>();
                Object[] inherit = null;
                if (!cur.isEmpty() && (!family.equals(curFamily) || size + PyStr.len(piece) > lim)) {
                    int k = 0;
                    if (adhere) {
                        while (k < curTl.size() && curTl.get(curTl.size() - 1 - k) != null) k++;
                        int maxStyle = 0;
                        if (k > 0) {
                            Map<String, Integer> styles = new HashMap<String, Integer>();
                            for (TitleDetector.Tl t : curTl.subList(curTl.size() - k, curTl.size())) {
                                if (t.style != null) {
                                    Integer c = styles.get(t.style);
                                    styles.put(t.style, c == null ? 1 : c + 1);
                                }
                            }
                            for (int c : styles.values()) maxStyle = Math.max(maxStyle, c);
                        }
                        if (k > cfg.adhMaxLines) {
                            k = 0;  // 块尾连续类标题行超过上限
                        } else if (maxStyle >= 2) {
                            k = 0;  // 块尾同一编号样式出现多次（像列表）
                        } else if (k > 0 && pieceTl != null && TitleDetector.NUMBERED.equals(pieceTl.kind)
                                && TitleDetector.NUMBERED.equals(curTl.get(curTl.size() - 1).kind)) {
                            k = 0;  // 下一段也是编号标题
                        }
                        if (k > 0) {  // 挪过去后下一块（标题行 + 表头 + 这一段）不能超过硬上限
                            List<String> moved = PyStr.sumLen(cur.subList(0, cur.size() - k)) < cfg.minLen ? cur : cur.subList(cur.size() - k, cur.size());
                            int headLen = a.kind.equals("row") && !a.header.isEmpty() ? PyStr.len(a.header) + 1 : 0;
                            int movedLen = 0;
                            for (String x : moved) movedLen += PyStr.len(x) + 1;
                            if (movedLen + headLen + PyStr.len(piece) > hard) k = 0;
                        }
                    }
                    if (k > 0) {
                        if (PyStr.sumLen(cur.subList(0, cur.size() - k)) < cfg.minLen) {  // 剩下的太短：整块并入下一块
                            carry = new ArrayList<String>(cur);
                            inherit = new Object[]{curParent, curHead};
                            cur = new ArrayList<String>();
                            curTl = new ArrayList<TitleDetector.Tl>();
                            curFamily = null;
                        } else {
                            carry = new ArrayList<String>(cur.subList(cur.size() - k, cur.size()));
                            cur = new ArrayList<String>(cur.subList(0, cur.size() - k));
                            curTl = new ArrayList<TitleDetector.Tl>(curTl.subList(0, curTl.size() - k));
                            // flush()
                            if (!cur.isEmpty()) recs.add(new Rec(PyStr.join("\n", cur), curParent, curHead));
                            cur = new ArrayList<String>();
                            curTl = new ArrayList<TitleDetector.Tl>();
                            curFamily = null;
                        }
                    } else {
                        // flush()
                        recs.add(new Rec(PyStr.join("\n", cur), curParent, curHead));
                        cur = new ArrayList<String>();
                        curTl = new ArrayList<TitleDetector.Tl>();
                        curFamily = null;
                    }
                }
                String add = piece;
                if (cur.isEmpty()) {
                    if (a.kind.equals("row") && !a.header.isEmpty()) add = a.header + "\n" + add;  // 表格的每一块都带表头
                    curParent = parentPath(path, pending);
                    curHead = !pending.isEmpty();
                    if (!pending.isEmpty()) add = pendingText(pending) + "\n" + add;  // 标题与紧随的第一段放在同一块
                    if (!carry.isEmpty()) {  // 吸附过来的类标题行放在最前（表格块里在表头之前）
                        add = PyStr.join("\n", carry) + "\n" + add;
                        if (inherit != null) {
                            @SuppressWarnings("unchecked")
                            List<Integer> ip = (List<Integer>) inherit[0];
                            curParent = ip;
                            curHead = (Boolean) inherit[1];
                        }
                    }
                    curFamily = family;
                    pending = new ArrayList<Object[]>();
                }
                cur.add(add);
                curTl.add(pieceTl);
            }
        }
        if (!pending.isEmpty()) {  // 页面以标题结尾
            curParent = parentPath(path, pending);
            curHead = true;
            cur.add(pendingText(pending));
        }
        if (!cur.isEmpty()) recs.add(new Rec(PyStr.join("\n", cur), curParent, curHead));
        return recs;
    }

    /** tuple(j for _, j in path if j not in own) */
    static List<Integer> parentPath(List<int[]> path, List<Object[]> pending) {
        Set<Integer> own = new HashSet<Integer>();
        for (Object[] p : pending) own.add((Integer) p[0]);
        List<Integer> out = new ArrayList<Integer>();
        for (int[] p : path) {
            if (!own.contains(p[1])) out.add(p[1]);
        }
        return out;
    }

    static String pendingText(List<Object[]> pending) {
        List<String> xs = new ArrayList<String>();
        for (Object[] p : pending) xs.add((String) p[1]);
        return PyStr.join("\n", xs);
    }

    /** parent[:len(scope)] == scope */
    static boolean startsWith(List<Integer> parent, List<Integer> scope) {
        return scope.size() <= parent.size() && parent.subList(0, scope.size()).equals(scope);
    }

    /** parent[:-1] */
    static List<Integer> dropLast(List<Integer> parent) {
        return parent.isEmpty() ? parent : parent.subList(0, parent.size() - 1);
    }

    /** 相邻小块贪心合并：合并后不超过 target 字，且不跨越更高一级的标题。Python: merge_sections */
    static List<String> mergeSections(List<Rec> recs, int target) {
        List<StringBuilder> texts = new ArrayList<StringBuilder>();
        List<Integer> textLens = new ArrayList<Integer>();
        List<List<Integer>> scopes = new ArrayList<List<Integer>>();
        for (Rec r : recs) {
            if (!texts.isEmpty()) {
                int g = texts.size() - 1;
                int n = PyStr.len(r.text);
                if (startsWith(r.parent, scopes.get(g)) && textLens.get(g) + 1 + n <= target) {
                    texts.get(g).append('\n').append(r.text);
                    textLens.set(g, textLens.get(g) + 1 + n);
                    continue;
                }
            }
            texts.add(new StringBuilder(r.text));
            textLens.add(PyStr.len(r.text));
            scopes.add(r.head ? r.parent : dropLast(r.parent));
        }
        List<String> out = new ArrayList<String>();
        for (StringBuilder b : texts) out.add(b.toString());
        return out;
    }
}
