package com.chunk.structure;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.chunk.structure.PyStr.D;
import static com.chunk.structure.PyStr.S;

/**
 * 文本工具，对应 Python recall_pipeline_v2.py 第 4、5 段：
 * clean_inline / split_sentences / split_long / is_pseudo_heading / md_inline 及相关正则。
 */
final class TextUtils {
    private TextUtils() {
    }

    // Python: WS = re.compile(r"[ \t\r\f\v\xa0　]+")
    static final Pattern WS = PyStr.re("[ \\t\\r\\f\\x0B\\u00A0\\u3000]+");
    // Python: SENT_END = re.compile(r"[。！？!?；;…]+[”’\"）)】》」』]*|\n+")
    static final Pattern SENT_END = PyStr.re("[。！？!?；;…]+[”’\"）)】》」』]*|\\n+");
    // Python: END_PUNCT = re.compile(r"[。！？!?；;，,、…]$")
    static final Pattern END_PUNCT = PyStr.re("[。！？!?；;，,、…]$");

    // Markdown 正则（Python 第 5 段）
    static final Pattern MD_HEADING = PyStr.re("^(#{1,6})" + S + "+(.*?)" + S + "*#*" + S + "*$");
    static final Pattern MD_LIST = PyStr.re("^" + S + "*(?:[-*+]|" + D + "{1,3}[.)])" + S + "+(.*)$");
    static final Pattern MD_PIPE_SEP = PyStr.re("^" + S + "*\\|?" + S + "*:?-{2,}:?" + S + "*(\\|" + S + "*:?-{2,}:?" + S + "*)*\\|?" + S + "*$");
    static final Pattern MD_IMAGE = PyStr.re("!\\[([^\\]]*)\\]\\([^)]*\\)");
    static final Pattern MD_LINK = PyStr.re("\\[([^\\]]*)\\]\\([^)]*\\)");
    static final Pattern MD_EMPH = PyStr.re("(\\*\\*|__)(.+?)\\1");
    static final Pattern MD_BOLD_LINE = PyStr.re("^" + S + "*(\\*\\*|__)(.+?)\\1" + S + "*$");
    static final Pattern HTML_TAG = PyStr.re("<[^>]+>");
    // Python: re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    static final Pattern MD_BR = PyStr.reIgnoreCase("<br" + S + "*/?>");

    /** 合并行内空白，保留换行；去掉空行。Python: clean_inline */
    static String cleanInline(String s) {
        StringBuilder b = new StringBuilder();
        for (String x : PyStr.split(s, "\n")) {
            String y = PyStr.strip(WS.matcher(x).replaceAll(" "));
            if (!y.isEmpty()) {
                if (b.length() > 0) b.append('\n');
                b.append(y);
            }
        }
        return b.toString();
    }

    /** 按句末标点与换行分句；单句超过 maxLen 时先在逗号处切，再硬切。Python: split_sentences */
    static List<String> splitSentences(String text, int maxLen) {
        List<String> out = new ArrayList<String>();
        int start = 0;
        Matcher m = SENT_END.matcher(text);
        while (m.find()) {
            if (m.end() > start) {
                out.add(text.substring(start, m.end()));
                start = m.end();
            }
        }
        if (start < text.length()) out.add(text.substring(start));
        List<String> res = new ArrayList<String>();
        for (String s0 : out) {
            int[] cps = s0.codePoints().toArray();
            int from = 0;
            while (cps.length - from > maxLen) {
                // Python: cut = max(s.rfind("，", 0, max_len), s.rfind(",", 0, max_len), s.rfind("、", 0, max_len))
                int cut = -1;
                for (int i = from + maxLen - 1; i >= from; i--) {
                    int c = cps[i];
                    if (c == '，' || c == ',' || c == '、') {
                        cut = i - from;
                        break;
                    }
                }
                cut = cut > 0 ? cut + 1 : maxLen;
                res.add(new String(cps, from, cut));
                from += cut;
            }
            String rest = new String(cps, from, cps.length - from);
            if (PyStr.hasText(rest)) res.add(rest);
        }
        return res;
    }

    /** 超长段落：按句贪心打包到 soft 以内。Python: split_long */
    static List<String> splitLong(String text, int soft) {
        List<String> pieces = new ArrayList<String>();
        StringBuilder cur = new StringBuilder();
        int curLen = 0;
        for (String s : splitSentences(text, soft)) {
            int n = PyStr.len(s);
            if (cur.length() > 0 && curLen + n > soft) {
                pieces.add(cur.toString());
                cur.setLength(0);
                curLen = 0;
            }
            cur.append(s);
            curLen += n;
        }
        if (PyStr.hasText(cur.toString())) pieces.add(cur.toString());
        List<String> res = new ArrayList<String>();
        for (String p : pieces) {
            String t = PyStr.strip(p);
            if (!t.isEmpty()) res.add(t);
        }
        return res;
    }

    /** 整段加粗的短行是否可视为伪标题：0 < 字数 ≤ max 且末尾不是标点。Python: is_pseudo_heading */
    static boolean isPseudoHeading(String t, int max) {
        int n = PyStr.len(t);
        return n > 0 && n <= max && !END_PUNCT.matcher(t).find();
    }

    /** Markdown 行内清洗：br → 换行，图片 / 链接留文字，去强调和行内 html 标签。Python: md_inline */
    static String mdInline(String s) {
        s = MD_BR.matcher(s).replaceAll("\n");
        s = MD_IMAGE.matcher(s).replaceAll("$1");
        s = MD_LINK.matcher(s).replaceAll("$1");
        s = MD_EMPH.matcher(s).replaceAll("$2");
        s = HTML_TAG.matcher(s).replaceAll("");
        return cleanInline(s);
    }
}
