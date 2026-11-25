package com.chunk.structure;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * 与 Python str 行为一致的字符串工具。迁移时最容易出错的地方都集中在这里：
 * <ul>
 *   <li>长度：Python len() 按码点计；Java length() 按 UTF-16 计（emoji、生僻字算 2）→ 一律用 {@link #len}</li>
 *   <li>空白：Python str.isspace() / strip() 认 29 个字符（含 NBSP、\x1c–\x1f、全角空格）；
 *       Java strip() / Character.isWhitespace 不一样 → 一律用 {@link #isSpace} / {@link #strip}</li>
 *   <li>正则：Python 的 \s、\d 对 str 是 Unicode 语义，$ 只认 \n → 用 {@link #S}、{@link #D}，并始终加 UNIX_LINES</li>
 * </ul>
 */
final class PyStr {
    private PyStr() {
    }

    /** Python 正则里 \s 对应的字符集（与 str.isspace() 完全相同的 29 个字符），不含方括号。 */
    static final String WS_CHARS = "\\t\\n\\x0B\\f\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";
    /** Python 的 \s */
    static final String S = "[" + WS_CHARS + "]";
    /** Python 的 \d（Unicode 十进制数字） */
    static final String D = "\\p{Nd}";

    /** 编译正则：始终 UNIX_LINES（$ 和 . 只把 \n 当行结束，与 Python 一致）。 */
    static Pattern re(String regex) {
        return Pattern.compile(regex, Pattern.UNIX_LINES);
    }

    static Pattern reIgnoreCase(String regex) {
        return Pattern.compile(regex, Pattern.UNIX_LINES | Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);
    }

    /** 与 Python str.isspace() 对单个字符的判断完全相同。 */
    static boolean isSpace(int cp) {
        if (cp >= 0x09 && cp <= 0x0D) return true;
        if (cp >= 0x1C && cp <= 0x20) return true;
        if (cp == 0x85 || cp == 0xA0 || cp == 0x1680) return true;
        if (cp >= 0x2000 && cp <= 0x200A) return true;
        return cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
    }

    /** Python len()：码点数。 */
    static int len(String s) {
        return s.codePointCount(0, s.length());
    }

    /** Python str.strip()（无参数）。 */
    static String strip(String s) {
        int b = 0, e = s.length();
        while (b < e) {
            int cp = s.codePointAt(b);
            if (!isSpace(cp)) break;
            b += Character.charCount(cp);
        }
        while (e > b) {
            int cp = s.codePointBefore(e);
            if (!isSpace(cp)) break;
            e -= Character.charCount(cp);
        }
        return s.substring(b, e);
    }

    /** Python str.strip(chars)：去掉两端属于 chars 的字符。 */
    static String strip(String s, String chars) {
        int b = 0, e = s.length();
        while (b < e) {
            int cp = s.codePointAt(b);
            if (chars.indexOf(cp) < 0) break;
            b += Character.charCount(cp);
        }
        while (e > b) {
            int cp = s.codePointBefore(e);
            if (chars.indexOf(cp) < 0) break;
            e -= Character.charCount(cp);
        }
        return s.substring(b, e);
    }

    /** Python 里 "if s.strip():" 的判断。 */
    static boolean hasText(String s) {
        return !strip(s).isEmpty();
    }

    /** Python str.split(sep)：保留空串（Java String.split 会丢掉末尾空串，这里不会）。 */
    static List<String> split(String s, String sep) {
        List<String> out = new ArrayList<String>();
        int start = 0;
        while (true) {
            int i = s.indexOf(sep, start);
            if (i < 0) {
                out.add(s.substring(start));
                return out;
            }
            out.add(s.substring(start, i));
            start = i + sep.length();
        }
    }

    static String join(String sep, List<String> xs) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < xs.size(); i++) {
            if (i > 0) b.append(sep);
            b.append(xs.get(i));
        }
        return b.toString();
    }

    /** 最后一个码点（s 非空）。 */
    static int lastCodePoint(String s) {
        return s.codePointBefore(s.length());
    }

    /** 一组字符串的码点总数（Python sum(len(x) for x in xs)）。 */
    static int sumLen(List<String> xs) {
        int n = 0;
        for (String x : xs) n += len(x);
        return n;
    }

    /** 去掉换行后的码点数（Python len(s.replace("\n", ""))）。 */
    static int lenNoNewline(String s) {
        int n = 0;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            if (cp != '\n') n++;
            i += Character.charCount(cp);
        }
        return n;
    }
}
