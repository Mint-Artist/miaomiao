package com.chunk.structure;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import static com.chunk.structure.PyStr.D;
import static com.chunk.structure.PyStr.S;

/**
 * 类标题行（没被识别成标题的编号小标题）识别：标题吸附与容器级切分共用。
 * Python: chunk_adhere.py 的 _STYLES / num_style / title_like / _tl_info
 */
final class TitleDetector {
    static final String NUMBERED = "编号标题";
    static final String LEAD_IN = "引导句";

    /** 编号样式：名称 + 正则（与 Python _STYLES 逐条对应，顺序有意义：取第一个匹配的）。 */
    private static final List<Object[]> STYLES = Arrays.asList(
            new Object[]{"数字", PyStr.re("^" + D + "{1,2}(?:[、)）]|[.．](?!" + D + "))")},
            new Object[]{"括号数字", PyStr.re("^[（(]" + D + "{1,2}[)）]")},
            new Object[]{"中文数字", PyStr.re("^[一二三四五六七八九十]{1,3}[、.．]")},
            new Object[]{"括号中文数字", PyStr.re("^[（(][一二三四五六七八九十]{1,3}[)）]")},
            new Object[]{"第X", PyStr.re("^第[一二三四五六七八九十百零" + D + "]{1,3}[章节部分条步篇讲课]")},
            new Object[]{"圈号", PyStr.re("^[①-⑳]")},
            new Object[]{"步骤", PyStr.reIgnoreCase("^(?:step|步骤)" + S + "*" + D + "{1,2}")}
    );
    /** Python: _NOT_TITLE_END = set("。！？!?；;，,、…") */
    private static final String NOT_TITLE_END = "。！？!?；;，,、…";

    /** 打包时逐段记录的类标题信息：类型 + 编号样式（引导句没有样式）。Python: _tl_info 的返回值 */
    static final class Tl {
        final String kind;
        final String style;

        Tl(String kind, String style) {
            this.kind = kind;
            this.style = style;
        }
    }

    private final int maxLen;
    private final boolean leadIn;

    TitleDetector(ChunkConfig cfg) {
        this.maxLen = cfg.adhMaxLen;
        this.leadIn = cfg.adhLeadIn;
    }

    /** 编号样式名，不是编号开头返回 null。Python: num_style */
    String numStyle(String text) {
        String t = PyStr.strip(text);
        for (Object[] st : STYLES) {
            if (((Pattern) st[1]).matcher(t).lookingAt()) return (String) st[0];
        }
        return null;
    }

    /** 类标题行返回 "编号标题" / "引导句"，否则 null。Python: title_like */
    String titleLike(String text) {
        String t = PyStr.strip(text);
        if (t.isEmpty() || PyStr.len(t) > maxLen) return null;
        int last = PyStr.lastCodePoint(t);
        if (last == '：' || last == ':') {
            if (numStyle(t) != null) return NUMBERED;
            return leadIn ? LEAD_IN : null;
        }
        if (NOT_TITLE_END.indexOf(last) >= 0) return null;
        return numStyle(t) != null ? NUMBERED : null;
    }

    /** Python: _tl_info */
    Tl tlInfo(String text) {
        String k = titleLike(text);
        if (k == null) return null;
        return new Tl(k, NUMBERED.equals(k) ? numStyle(text) : null);
    }
}
