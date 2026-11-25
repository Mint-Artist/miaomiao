package com.chunk.structure;

/**
 * 结构单元（t512 / 吸附版用）。Python: Atom
 * kind：heading（level 1–7，7 = 整段加粗短行这类伪标题）/ para / li / row（表格行，header = 该表的表头行文本）/ code
 */
final class Atom {
    final String kind;
    final String text;
    final int level;
    final int tableId;
    final String header;

    Atom(String kind, String text) {
        this(kind, text, 0, -1, "");
    }

    Atom(String kind, String text, int level, int tableId, String header) {
        this.kind = kind;
        this.text = text;
        this.level = level;
        this.tableId = tableId;
        this.header = header;
    }

    static Atom heading(String text, int level) {
        return new Atom("heading", text, level, -1, "");
    }

    static Atom row(String text, int tableId, String header) {
        return new Atom("row", text, 0, tableId, header);
    }

    @Override
    public String toString() {
        return kind + (level > 0 ? "(h" + level + ")" : "") + (tableId >= 0 ? "[t" + tableId + "]" : "") + ": " + text;
    }
}
