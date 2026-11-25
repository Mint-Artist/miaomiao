package com.chunk.structure;

import java.util.Collections;
import java.util.List;

/**
 * 容器级切分的结构单元。Python: Unit（chunk_container.py）
 * kind：heading（level 1–7）/ para / 容器（list、table、note、metadata、quote、code、figure）
 * items：para / heading 只有 1 条；容器为子结构文本（列表项、表格行、提示框里的每行等）
 */
final class Unit {
    final String kind;
    final List<String> items;
    final int level;
    final String header;
    final String caption;

    Unit(String kind, List<String> items, int level, String header, String caption) {
        this.kind = kind;
        this.items = items;
        this.level = level;
        this.header = header;
        this.caption = caption;
    }

    static Unit para(String text) {
        return new Unit("para", Collections.singletonList(text), 0, "", "");
    }

    static Unit heading(String text, int level) {
        return new Unit("heading", Collections.singletonList(text), level, "", "");
    }

    static Unit container(String kind, List<String> items) {
        return new Unit(kind, items, 0, "", "");
    }

    @Override
    public String toString() {
        return kind + (level > 0 ? "(h" + level + ")" : "") + " " + items;
    }
}
