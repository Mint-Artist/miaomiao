package com.chunk.structure;

import java.util.ArrayList;
import java.util.List;

/**
 * 对外入口：5 种结构切分方法。
 *
 * <pre>
 * StructureChunker chunker = new StructureChunker();                 // 默认参数，与 Python 原版一致
 * List&lt;String&gt; a = chunker.chunk(ChunkMethod.MD_STRUCT_SECMERGE_T512, markdown, null);
 * List&lt;String&gt; b = chunker.chunk(ChunkMethod.HTML_CONTAINER, null, html);
 * List&lt;Chunk&gt;  c = chunker.chunkWithKind(ChunkMethod.HTML_CONTAINER, null, html);   // 带块类型
 * </pre>
 *
 * 线程安全：实例不保存任何切分状态，可以在多线程 / Spark executor 里共用一个实例（构造后不要再改 ChunkConfig）。
 */
public final class StructureChunker {
    private final MarkdownParser mdParser;
    private final HtmlAtomParser htmlParser;
    private final HtmlBlockParser blockParser;
    private final SectionChunker sectionChunker;
    private final ContainerChunker containerChunker;

    public StructureChunker() {
        this(new ChunkConfig());
    }

    public StructureChunker(ChunkConfig cfg) {
        TitleDetector td = new TitleDetector(cfg);
        this.mdParser = new MarkdownParser(cfg);
        this.htmlParser = new HtmlAtomParser(cfg);
        this.blockParser = new HtmlBlockParser(cfg);
        this.sectionChunker = new SectionChunker(cfg, td);
        this.containerChunker = new ContainerChunker(cfg, td);
    }

    /**
     * 切分。md_* 方法只用 markdown，html_* 方法只用 html；对应输入为 null 或空串时返回空列表。
     */
    public List<String> chunk(ChunkMethod method, String markdown, String html) {
        List<String> out = new ArrayList<String>();
        for (Chunk c : chunkWithKind(method, markdown, html)) out.add(c.getText());
        return out;
    }

    /** 同 {@link #chunk}，附带块类型（只有 HTML_CONTAINER 有类型，其他方法为 null）。 */
    public List<Chunk> chunkWithKind(ChunkMethod method, String markdown, String html) {
        String md = markdown == null ? "" : markdown;
        String h = html == null ? "" : html;
        switch (method) {
            case MD_STRUCT_SECMERGE_T512:
                return plain(sectionChunker.chunk(mdParser.parse(md), false));
            case MD_STRUCT_SECMERGE_T512_ADH:
                return plain(sectionChunker.chunk(mdParser.parse(md), true));
            case HTML_STRUCT_SECMERGE_T512:
                return plain(sectionChunker.chunk(htmlParser.parse(h), false));
            case HTML_STRUCT_SECMERGE_T512_ADH:
                return plain(sectionChunker.chunk(htmlParser.parse(h), true));
            case HTML_CONTAINER:
                return containerChunker.chunk(blockParser.parse(h));
            default:
                throw new IllegalArgumentException(String.valueOf(method));
        }
    }

    private static List<Chunk> plain(List<String> xs) {
        List<Chunk> out = new ArrayList<Chunk>(xs.size());
        for (String x : xs) out.add(new Chunk(x, null));
        return out;
    }
}
