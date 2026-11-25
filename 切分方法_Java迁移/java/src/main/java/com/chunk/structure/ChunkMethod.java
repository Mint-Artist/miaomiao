package com.chunk.structure;

/**
 * 5 种切分方法。名字与 Python 原版 / 离线评测里的方法名完全相同。
 */
public enum ChunkMethod {
    /** Markdown 结构切分：小节打包 + 相邻小节合并，表格软上限 512。离线评测里 hit@3000字 最好。 */
    MD_STRUCT_SECMERGE_T512("md_struct_secmerge_t512", Input.MARKDOWN),
    /** 同上，输入为 html。 */
    HTML_STRUCT_SECMERGE_T512("html_struct_secmerge_t512", Input.HTML),
    /** md_struct_secmerge_t512 + 标题吸附（块尾的编号小标题挪到下一块）。未做过离线召回评测。 */
    MD_STRUCT_SECMERGE_T512_ADH("md_struct_secmerge_t512_adh", Input.MARKDOWN),
    /** html_struct_secmerge_t512 + 标题吸附。未做过离线召回评测。 */
    HTML_STRUCT_SECMERGE_T512_ADH("html_struct_secmerge_t512_adh", Input.HTML),
    /** 容器级结构切分：列表 / 表格 / 提示框等容器 ≤ 1024 字整块保留，更长的按子结构拆到 512。 */
    HTML_CONTAINER("html_container", Input.HTML);

    /** 方法需要的输入字段。 */
    public enum Input { MARKDOWN, HTML }

    private final String methodName;
    private final Input input;

    ChunkMethod(String methodName, Input input) {
        this.methodName = methodName;
        this.input = input;
    }

    /** 与 Python 相同的方法名，如 "md_struct_secmerge_t512"。 */
    public String methodName() {
        return methodName;
    }

    public Input input() {
        return input;
    }

    /** 按 Python 方法名查找。 */
    public static ChunkMethod of(String name) {
        for (ChunkMethod m : values()) {
            if (m.methodName.equals(name)) return m;
        }
        throw new IllegalArgumentException("未知切分方法: " + name);
    }
}
