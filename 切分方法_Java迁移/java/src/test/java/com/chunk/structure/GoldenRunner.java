package com.chunk.structure;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * 与 Python 标准答案（make_golden.py 生成的 golden jsonl）逐字对照。不依赖 JUnit，命令行即可运行：
 *
 * <pre>
 * java -cp target/classes:target/test-classes:jsoup.jar:gson.jar com.chunk.structure.GoldenRunner golden.jsonl.gz [--show 5]
 * </pre>
 *
 * 输出每个方法的一致率，并打印前几个不一致样例（第一个不同的块）。全部一致时退出码 0，否则 1。
 */
public final class GoldenRunner {

    /** 一个方法的对照结果。 */
    public static final class Stat {
        public int pages;
        public int mismatches;
        public final List<String> examples = new ArrayList<String>();
    }

    /** 已知差异清单：每行一个页面 id（# 后为说明）；清单里的页面不一致时单独计数，不算失败。 */
    public static java.util.Set<String> loadKnown(String path) throws IOException {
        java.util.Set<String> ids = new java.util.HashSet<String>();
        if (path == null || !new java.io.File(path).exists()) return ids;
        for (String line : java.nio.file.Files.readAllLines(java.nio.file.Paths.get(path), StandardCharsets.UTF_8)) {
            String t = line.replaceAll("#.*$", "").trim();
            if (!t.isEmpty()) ids.add(t);
        }
        return ids;
    }

    public static Map<String, Stat> run(String path, int show, PrintStream log) throws IOException {
        return run(path, show, log, new java.util.HashSet<String>());
    }

    public static Map<String, Stat> run(String path, int show, PrintStream log, java.util.Set<String> known) throws IOException {
        StructureChunker chunker = new StructureChunker();
        int knownHit = 0;
        Map<String, Stat> stats = new LinkedHashMap<String, Stat>();
        Stat kinds = new Stat();
        try (BufferedReader r = open(path)) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                JsonObject o = JsonParser.parseString(line).getAsJsonObject();
                if (!"page".equals(str(o, "type"))) continue;
                String id = str(o, "id");
                String md = str(o, "markdown");
                String html = str(o, "html");
                JsonObject exp = o.getAsJsonObject("expected");
                for (Map.Entry<String, JsonElement> e : exp.entrySet()) {
                    ChunkMethod m = ChunkMethod.of(e.getKey());
                    List<String> want = list(e.getValue().getAsJsonArray());
                    List<String> got = chunker.chunk(m, md, html);
                    Stat st = stats.get(m.methodName());
                    if (st == null) stats.put(m.methodName(), st = new Stat());
                    st.pages++;
                    if (!want.equals(got)) {
                        if (known.contains(id)) {
                            knownHit++;
                            continue;
                        }
                        st.mismatches++;
                        if (st.examples.size() < show) st.examples.add(diff(id, want, got));
                    }
                }
                if (o.has("container_kinds") && !o.get("container_kinds").isJsonNull()) {
                    List<String> want = list(o.getAsJsonArray("container_kinds"));
                    List<String> got = new ArrayList<String>();
                    for (Chunk c : chunker.chunkWithKind(ChunkMethod.HTML_CONTAINER, md, html)) got.add(c.getKind());
                    kinds.pages++;
                    if (!want.equals(got) && !known.contains(id)) {
                        kinds.mismatches++;
                        if (kinds.examples.size() < show) kinds.examples.add(id + "\n    期望 " + want + "\n    实际 " + got);
                    }
                }
            }
        }
        stats.put("html_container（块类型）", kinds);
        for (Map.Entry<String, Stat> e : stats.entrySet()) {
            Stat st = e.getValue();
            log.printf("%-36s %6d 页  不一致 %4d  一致率 %.3f%%%n", e.getKey(), st.pages, st.mismatches,
                    st.pages == 0 ? 100.0 : 100.0 * (st.pages - st.mismatches) / st.pages);
            for (String ex : st.examples) log.println("  - " + ex);
        }
        if (!known.isEmpty()) log.printf("已知差异清单 %d 个页面，其中实际出现不一致 %d 次（不计入上面的不一致）%n", known.size(), knownHit);
        return stats;
    }

    private static String diff(String id, List<String> want, List<String> got) {
        int i = 0;
        while (i < want.size() && i < got.size() && want.get(i).equals(got.get(i))) i++;
        String w = i < want.size() ? want.get(i) : "（没有这一块）";
        String g = i < got.size() ? got.get(i) : "（没有这一块）";
        return id + "：块数 期望 " + want.size() + " / 实际 " + got.size() + "，第 " + (i + 1) + " 块起不同\n    期望 "
                + clip(w) + "\n    实际 " + clip(g);
    }

    private static String clip(String s) {
        String t = s.replace("\n", "⏎");
        return t.length() > 160 ? t.substring(0, 160) + "…（共 " + PyStr.len(s) + " 字）" : t;
    }

    private static List<String> list(JsonArray a) {
        List<String> out = new ArrayList<String>();
        for (JsonElement x : a) out.add(x.getAsString());
        return out;
    }

    private static String str(JsonObject o, String k) {
        return o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsString() : null;
    }

    private static BufferedReader open(String path) throws IOException {
        InputStream in = new FileInputStream(path);
        if (path.endsWith(".gz")) in = new GZIPInputStream(in, 1 << 16);
        return new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8), 1 << 16);
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("用法：GoldenRunner golden.jsonl[.gz] [--show N] [--known known_diffs.txt]");
            System.exit(2);
        }
        int show = 3;
        String knownPath = null;
        for (int i = 1; i < args.length - 1; i++) {
            if (args[i].equals("--show")) show = Integer.parseInt(args[i + 1]);
            if (args[i].equals("--known")) knownPath = args[i + 1];
        }
        PrintStream out = new PrintStream(System.out, true, "UTF-8");
        long t0 = System.currentTimeMillis();
        Map<String, Stat> stats = run(args[0], show, out, loadKnown(knownPath));
        boolean ok = true;
        for (Stat st : stats.values()) ok &= st.mismatches == 0;
        out.printf("耗时 %.1fs，%s%n", (System.currentTimeMillis() - t0) / 1000.0, ok ? "全部一致" : "存在不一致");
        System.exit(ok ? 0 : 1);
    }
}
