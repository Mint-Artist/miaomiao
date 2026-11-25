package com.chunk.structure;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 容器级结构切分 html_container 的装块与合并。Python: chunk_container_recs / chunk_container（chunk_container.py）。
 */
final class ContainerChunker {
    private final ChunkConfig cfg;
    private final TitleDetector td;

    ContainerChunker(ChunkConfig cfg, TitleDetector td) {
        this.cfg = cfg;
        this.td = td;
    }

    /** 合并后的块（未过滤 minLen）。Python: chunk_container_recs 返回的 {"text", "scope", "kind", ...} */
    static final class Group {
        final StringBuilder text;
        int textLen;
        final List<Integer> scope;
        String kind;
        boolean hasCont;
        final boolean mergeable;

        Group(String text, List<Integer> scope, String kind, boolean mergeable) {
            this.text = new StringBuilder(text);
            this.textLen = PyStr.len(text);
            this.scope = scope;
            this.kind = kind;
            this.hasCont = !kind.equals("para");
            this.mergeable = mergeable;
        }
    }

    /** 过滤 minLen 后的结果。Python: chunk_container + chunk_with_kind */
    List<Chunk> chunk(List<Unit> units) {
        List<Chunk> out = new ArrayList<Chunk>();
        for (Group g : groups(units)) {
            String t = g.text.toString();
            if (PyStr.lenNoNewline(t) >= cfg.minLen) out.add(new Chunk(t, g.kind));
        }
        return out;
    }

    /** Python: chunk_container_recs */
    List<Group> groups(List<Unit> units) {
        return new State(units).run();
    }

    /** 一次切分的可变状态（Python 函数里的局部变量和闭包）。 */
    private final class State {
        final List<Unit> units;
        final List<Object[]> recs = new ArrayList<Object[]>();      // [文本, 上级标题路径, 是否以标题开头, kind, 可合并]
        final List<int[]> path = new ArrayList<int[]>();             // 标题栈 [(level, 序号)]
        List<Object[]> pending = new ArrayList<Object[]>();          // 待吸附的标题 [(序号, 文本)]
        List<String> run = new ArrayList<String>();                  // 段落组当前块的各段
        List<TitleDetector.Tl> runTl = new ArrayList<TitleDetector.Tl>();
        List<Integer> runDesc = new ArrayList<Integer>();            // 每段若是完整段落则记其字数，否则 null
        List<Integer> rpParent = Collections.emptyList();
        boolean rpHead = false;

        State(List<Unit> units) {
            this.units = units;
        }

        /** 返回 [前缀标题文本列表, parent, head]，并清空 pending。Python: take_pending */
        Object[] takePending() {
            List<Integer> parent = SectionChunker.parentPath(path, pending);
            List<String> pre = new ArrayList<String>();
            for (Object[] p : pending) pre.add((String) p[1]);
            boolean head = !pending.isEmpty();
            pending = new ArrayList<Object[]>();
            return new Object[]{pre, parent, head};
        }

        void flushRun() {
            if (!run.isEmpty()) recs.add(new Object[]{PyStr.join("\n", run), rpParent, rpHead, "para", true});
            run = new ArrayList<String>();
            runTl = new ArrayList<TitleDetector.Tl>();
            runDesc = new ArrayList<Integer>();
        }

        /** 段落块末尾可挪走的类标题行数。Python: trailing_k */
        int trailingK() {
            int k = 0;
            while (k < runTl.size() && runTl.get(runTl.size() - 1 - k) != null) k++;
            if (k == 0) return 0;
            Map<String, Integer> styles = new HashMap<String, Integer>();
            for (TitleDetector.Tl t : runTl.subList(runTl.size() - k, runTl.size())) {
                if (t.style != null) {
                    Integer c = styles.get(t.style);
                    styles.put(t.style, c == null ? 1 : c + 1);
                }
            }
            int maxStyle = 0;
            for (int c : styles.values()) maxStyle = Math.max(maxStyle, c);
            if (k > cfg.adhMaxLines) return 0;      // 类标题行超过上限
            if (maxStyle >= 2) return 0;            // 同一编号样式出现多次（像列表）
            return k;
        }

        /** Python: tail_fits */
        boolean tailFits(int k, int extra) {
            boolean whole = PyStr.sumLen(run.subList(0, run.size() - k)) < cfg.minLen;
            List<String> moved = whole ? run : run.subList(run.size() - k, run.size());
            int n = 0;
            for (String x : moved) n += PyStr.len(x) + 1;
            return n + extra <= cfg.ctHard;
        }

        /** 从段落块末尾取走 k 段（剩下不足 minLen 时整块取走）；返回 [moved, inherit 或 null]。Python: take_tail */
        Object[] takeTail(int k, int extra) {
            if (!tailFits(k, extra)) return new Object[]{new ArrayList<String>(), null};
            boolean whole = PyStr.sumLen(run.subList(0, run.size() - k)) < cfg.minLen;
            List<String> moved = new ArrayList<String>(whole ? run : run.subList(run.size() - k, run.size()));
            Object[] inherit = whole ? new Object[]{rpParent, rpHead} : null;
            if (whole) {
                run = new ArrayList<String>();
                runTl = new ArrayList<TitleDetector.Tl>();
                runDesc = new ArrayList<Integer>();
            } else {
                run = new ArrayList<String>(run.subList(0, run.size() - k));
                runTl = new ArrayList<TitleDetector.Tl>(runTl.subList(0, runTl.size() - k));
                runDesc = new ArrayList<Integer>(runDesc.subList(0, runDesc.size() - k));
            }
            return new Object[]{moved, inherit};
        }

        /** Python: add_para_piece */
        @SuppressWarnings("unchecked")
        void addParaPiece(String piece, TitleDetector.Tl tl, Integer desc) {
            List<String> carry = new ArrayList<String>();
            Object[] inherit = null;
            if (!run.isEmpty() && PyStr.sumLen(run) + run.size() + PyStr.len(piece) > cfg.soft) {
                int k = trailingK();
                if (k > 0 && tl != null && TitleDetector.NUMBERED.equals(tl.kind)
                        && TitleDetector.NUMBERED.equals(runTl.get(runTl.size() - 1).kind)) {
                    k = 0;  // 下一段也是编号标题
                }
                if (k > 0) {
                    Object[] r = takeTail(k, PyStr.len(piece));
                    carry = (List<String>) r[0];
                    inherit = (Object[]) r[1];
                }
                flushRun();
            }
            if (run.isEmpty()) {
                Object[] tp = takePending();
                if (inherit != null) {
                    rpParent = (List<Integer>) inherit[0];
                    rpHead = (Boolean) inherit[1];
                } else {
                    rpParent = (List<Integer>) tp[1];
                    rpHead = (Boolean) tp[2];
                }
                List<String> parts = new ArrayList<String>((List<String>) tp[0]);
                parts.addAll(carry);
                parts.add(piece);
                run.add(PyStr.join("\n", parts));
            } else {
                run.add(piece);
            }
            runTl.add(tl);
            runDesc.add(desc);
        }

        /** Python: emit_container */
        @SuppressWarnings("unchecked")
        void emitContainer(Unit u) {
            List<String> lead = new ArrayList<String>();
            Object[] inherit = null;
            int k = trailingK();
            List<String> bodyParts = new ArrayList<String>();
            if (!u.caption.isEmpty()) bodyParts.add(u.caption);
            if (!u.header.isEmpty()) bodyParts.add(u.header);
            bodyParts.addAll(u.items);
            int bodyLen = PyStr.len(PyStr.join("\n", bodyParts));
            // 容器前的描述段：容器能整块保留时，把紧挨着的一个完整段落一起挪到容器开头；容器要拆分时不带
            if (cfg.ctLead && cfg.ctLeadKinds.contains(u.kind) && bodyLen <= cfg.ctHard) {
                int j = run.size() - k - 1;
                if (j >= 0 && runDesc.get(j) != null && runDesc.get(j) <= cfg.ctLeadMax && runTl.get(j) == null) {
                    if (tailFits(k + 1, bodyLen)) {
                        Object[] r = takeTail(k + 1, bodyLen);
                        lead = (List<String>) r[0];
                        inherit = (Object[]) r[1];
                    }
                }
            }
            if (lead.isEmpty() && k > 0) {
                int extra = bodyLen <= cfg.ctHard ? bodyLen
                        : PyStr.len(u.header) + PyStr.len(u.caption) + 2 + (u.items.isEmpty() ? 0 : PyStr.len(u.items.get(0)));
                Object[] r = takeTail(k, extra);
                lead = (List<String>) r[0];
                inherit = (Object[]) r[1];
            }
            flushRun();
            Object[] tp = takePending();
            List<Integer> parent = (List<Integer>) tp[1];
            boolean head = (Boolean) tp[2];
            if (inherit != null) {
                parent = (List<Integer>) inherit[0];
                head = (Boolean) inherit[1];
            }
            List<String> first = new ArrayList<String>((List<String>) tp[0]);
            first.addAll(lead);
            if (!u.caption.isEmpty()) first.add(u.caption);
            List<String> rep = u.kind.equals("table") && !u.header.isEmpty() ? Collections.singletonList(u.header) : Collections.<String>emptyList();
            List<String> all = new ArrayList<String>(first);
            all.addAll(rep);
            all.addAll(u.items);
            String whole = PyStr.join("\n", all);
            if (PyStr.len(whole) <= cfg.ctHard) {
                recs.add(new Object[]{whole, parent, head, u.kind, true});
                return;
            }
            // 超过上限：按子结构拆，每块装到 ctSplit；表格续块重复表头
            List<String> pieces = new ArrayList<String>();
            for (String it : u.items) {
                if (PyStr.len(it) <= cfg.ctSplit) pieces.add(it);
                else pieces.addAll(TextUtils.splitLong(it, cfg.ctSplit));
            }
            List<String> cur = new ArrayList<String>(first);
            cur.addAll(rep);
            boolean has = false;
            int nOut = 0;
            for (String p : pieces) {
                if (has && PyStr.sumLen(cur) + cur.size() + PyStr.len(p) > cfg.ctSplit) {
                    recs.add(new Object[]{PyStr.join("\n", cur), parent, head && nOut == 0, u.kind, false});  // 拆开的片段不参与合并
                    nOut++;
                    cur = new ArrayList<String>(rep);
                    has = false;
                }
                cur.add(p);
                has = true;
            }
            if (has || nOut == 0) recs.add(new Object[]{PyStr.join("\n", cur), parent, head && nOut == 0, u.kind, nOut == 0});
        }

        @SuppressWarnings("unchecked")
        List<Group> run() {
            for (int i = 0; i < units.size(); i++) {
                Unit u = units.get(i);
                if (u.kind.equals("heading")) {
                    flushRun();
                    while (!path.isEmpty() && path.get(path.size() - 1)[0] >= u.level) path.remove(path.size() - 1);
                    path.add(new int[]{u.level, i});
                    pending.add(new Object[]{i, u.items.get(0)});
                } else if (u.kind.equals("para")) {
                    String t = u.items.get(0);
                    if (PyStr.len(t) <= cfg.ctHard) {
                        addParaPiece(t, td.tlInfo(t), PyStr.len(t));
                    } else {
                        for (String p : TextUtils.splitLong(t, cfg.ctSplit)) addParaPiece(p, null, null);
                    }
                } else if (PyStr.sumLen(u.items) + PyStr.len(u.header) + PyStr.len(u.caption) < cfg.ctTiny) {
                    // 极小容器单独成块会被 minLen 过滤掉：按段落处理，可与相邻段落合并
                    List<String> xs = new ArrayList<String>();
                    if (!u.caption.isEmpty()) xs.add(u.caption);
                    if (!u.header.isEmpty()) xs.add(u.header);
                    xs.addAll(u.items);
                    addParaPiece(PyStr.join("\n", xs), null, null);
                } else {
                    emitContainer(u);
                }
            }
            flushRun();
            if (!pending.isEmpty()) {  // 页面以标题结尾
                Object[] tp = takePending();
                recs.add(new Object[]{PyStr.join("\n", (List<String>) tp[0]), tp[1], true, "para", true});
            }
            // 相邻块合并：同一小节范围内、合并后不超过 soft；一块里最多一个容器；拆开的容器片段不参与合并
            List<Group> out = new ArrayList<Group>();
            for (Object[] r : recs) {
                String text = (String) r[0];
                List<Integer> parent = (List<Integer>) r[1];
                boolean head = (Boolean) r[2];
                String kind = (String) r[3];
                boolean mergeable = (Boolean) r[4];
                if (!out.isEmpty()) {
                    Group g = out.get(out.size() - 1);
                    int n = PyStr.len(text);
                    if (mergeable && g.mergeable && !(g.hasCont && !kind.equals("para"))
                            && SectionChunker.startsWith(parent, g.scope) && g.textLen + 1 + n <= cfg.soft) {
                        g.text.append('\n').append(text);
                        g.textLen += 1 + n;
                        if (!kind.equals("para")) {
                            g.hasCont = true;
                            g.kind = kind;
                        }
                        continue;
                    }
                }
                out.add(new Group(text, head ? parent : SectionChunker.dropLast(parent), kind, mergeable));
            }
            return out;
        }
    }
}
