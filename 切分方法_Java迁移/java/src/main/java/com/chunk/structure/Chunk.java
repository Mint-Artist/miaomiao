package com.chunk.structure;

/**
 * 一个切分结果块。
 * kind：容器级切分（html_container）给出块类型（para / list / table / note / metadata / quote / code / figure；
 * 段落与一个容器合并后的块取容器类型）；其他方法为 null。
 */
public final class Chunk {
    private final String text;
    private final String kind;

    public Chunk(String text, String kind) {
        this.text = text;
        this.kind = kind;
    }

    public String getText() {
        return text;
    }

    public String getKind() {
        return kind;
    }

    @Override
    public String toString() {
        return (kind != null ? "[" + kind + "] " : "") + text;
    }
}
