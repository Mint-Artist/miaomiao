package com.chunk.structure;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * 全部切分参数，默认值与 Python 原版一致（recall_pipeline_v2.py / chunk_adhere.py / chunk_container.py）。
 * 构造 {@link StructureChunker} 之后不要再修改（StructureChunker 按只读使用，以保证线程安全）。
 * 字数一律按 Unicode 码点计（与 Python len() 相同）。
 */
public final class ChunkConfig {
    // ---- 通用（recall_pipeline_v2.py 第 1 段）----
    /** 软上限：段落装箱、相邻小节 / 段落块合并的上限。Python: SOFT */
    public int soft = 256;
    /** 硬上限：单个结构单元不超过它就不拆。Python: HARD */
    public int hard = 512;
    /** 最终过滤：不足这么多字（不计换行）的块丢弃。Python: MIN_LEN */
    public int minLen = 20;
    /** t512 方法里表格行打包的软上限。Python: TABLE_SOFT */
    public int tableSoft = 512;
    /** 伪标题（整段加粗短行）的最大字数。Python: is_pseudo_heading 里的 40 */
    public int pseudoHeadingMax = 40;

    // ---- 标题吸附（chunk_adhere.py）----
    /** 类标题行最长字数。Python: ADH_MAX_LEN */
    public int adhMaxLen = 40;
    /** 块尾最多挪几行。Python: ADH_MAX_LINES */
    public int adhMaxLines = 4;
    /** 不带编号、以冒号结尾的引导句是否参与吸附。Python: ADH_LEAD_IN */
    public boolean adhLeadIn = false;

    // ---- 容器级切分（chunk_container.py）----
    /** 容器整块保留的上限。Python: CT_HARD */
    public int ctHard = 1024;
    /** 超大容器拆开后每块的目标长度。Python: CT_SPLIT */
    public int ctSplit = 512;
    /** 不足这么多字的容器按段落处理。Python: CT_TINY（= MIN_LEN） */
    public int ctTiny = 20;
    /** 超过这么多字的 h1–h6 不当标题。Python: CT_HEAD_MAX */
    public int ctHeadMax = 100;
    /** 容器能整块保留时，把紧挨在前面的一个描述段落挪到容器开头。Python: CT_LEAD */
    public boolean ctLead = true;
    /** 描述段落最长多少字才带。Python: CT_LEAD_MAX */
    public int ctLeadMax = 256;
    /** 哪些容器带前文。Python: CT_LEAD_KINDS */
    public Set<String> ctLeadKinds = set("table", "list", "code");
    /** 视为提示框的 div class。Python: NOTE_CLASSES */
    public Set<String> noteClasses = set("note");
    /** 视为元信息的 div class。Python: META_CLASSES */
    public Set<String> metaClasses = set("metadata");

    private static Set<String> set(String... xs) {
        return Collections.unmodifiableSet(new HashSet<String>(Arrays.asList(xs)));
    }
}
