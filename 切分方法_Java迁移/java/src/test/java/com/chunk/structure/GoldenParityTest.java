package com.chunk.structure;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 与 Python 标准答案逐字对照（构造用例 + 随机 html / markdown 页）。
 * 标准答案路径默认 ../testdata/golden_cases.jsonl.gz，可用 -Dgolden.path=... 指定（例如服务器上生成的真实网页 golden）。
 */
class GoldenParityTest {

    @Test
    void matchesPythonReference() throws Exception {
        String path = System.getProperty("golden.path", "../testdata/golden_cases.jsonl.gz");
        assertTrue(new File(path).exists(), "找不到标准答案文件：" + path + "（先运行 python/make_golden.py）");
        String known = System.getProperty("golden.known", "../testdata/known_diffs.txt");
        Map<String, GoldenRunner.Stat> stats = GoldenRunner.run(path, 3, System.out, GoldenRunner.loadKnown(known));
        for (Map.Entry<String, GoldenRunner.Stat> e : stats.entrySet()) {
            assertEquals(0, e.getValue().mismatches, e.getKey() + " 与 Python 不一致的页数");
        }
    }

    @Test
    void basicUsage() {
        StructureChunker c = new StructureChunker();
        String html = "<h2>办理材料</h2><p>办理居住证需要准备以下材料。</p><ul><li>身份证原件及复印件两份</li><li>户口本原件及复印件</li></ul>";
        assertEquals(1, c.chunk(ChunkMethod.HTML_CONTAINER, null, html).size());
        assertEquals("list", c.chunkWithKind(ChunkMethod.HTML_CONTAINER, null, html).get(0).getKind());
        assertTrue(c.chunk(ChunkMethod.MD_STRUCT_SECMERGE_T512, null, html).isEmpty(), "md 方法只用 markdown 输入");
        assertTrue(c.chunk(ChunkMethod.HTML_CONTAINER, null, null).isEmpty());
    }
}
