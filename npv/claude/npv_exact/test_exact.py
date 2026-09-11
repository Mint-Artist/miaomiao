"""python -m unittest test_exact -v  —— 验证"缺陷被原样保留"以及 Java 语义模拟。"""
import math
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npv_exact as e  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "npv_py", "sample")
NOW = 1757030400


def tables(**kw):
    """不经文件直接构造 Tables。"""
    t = e.Tables.__new__(e.Tables)
    t.dr_site = kw.get("dr_site", {"www.gov.cn": "3"})
    t.dr_suffix = kw.get("dr_suffix", {"edu.cn": "3", ".com": "2"})
    t.dr_suffix_order = e.java_hashmap_order(list(t.dr_suffix))
    t.ow = kw.get("ow", {"www.zhihu.com": "1"})
    t.spr_max = kw.get("spr_max", e.f32(10.0))
    t.ow_blacklist = kw.get("ow_blacklist", ["spam.example.org"])
    t.adc_whitelist = kw.get("adc_whitelist", ["moe.edu.cn"])
    return t


class PreservedBugs(unittest.TestCase):
    def test_p0_2_spr_normalized_twice(self):
        s = e.PageValueScoreExact(tables(spr_max=e.f32(10.0)), NOW)
        s.gen_feature("https://x.com/", 100, 10.0, 0, [])
        self.assertEqual(s.feature_list["spr"], 1.0)                 # 第一次 /10
        # 第二次 /10 -> 0.1，不超过 0.2 阈值，不映射；srNew = (1 + 0.25*0.1)/1.1
        expect = e.f32(e.f32(1.0 + e.f32(e.F("0.25") * e.f32(0.1))) / e.F("1.1"))
        self.assertEqual(s.feature_list["spr_sr"], expect)

    def test_p0_4_default_date(self):
        s = e.PageValueScoreExact(tables(), NOW)
        days = (date(2026, 9, 5) - date(2024, 5, 22)).days  # 本机时区下 NOW 对应 2026-09-05
        now_day = e.date_string_from_ms(NOW * 1000)
        days = (date.fromisoformat(now_day) - date(2024, 5, 22)).days
        self.assertAlmostEqual(s.down_factor("0", str(NOW), 2000.0), 2 / (math.exp(days / 2000.0) + 1))
        self.assertAlmostEqual(s.adjust_score_pct("0", 100.0), 100.0 * 2 / (math.exp(days / 2000.0) + 1))

    def test_p0_5_future_amplifies(self):
        s = e.PageValueScoreExact(tables(), NOW)
        self.assertGreater(s.down_factor(str(NOW + 400 * 86400), str(NOW), 2000.0), 1.0)

    def test_p0_7_spr_max_zero_gives_nan(self):
        s = e.PageValueScoreExact(tables(spr_max=0.0), NOW)
        s.gen_feature("https://x.com/", 100, 0.0, 0, [])
        self.assertTrue(math.isnan(s.feature_list["spr"]) or math.isinf(s.feature_list["spr"]))

    def test_pt_gate_and_skip_for_adc(self):
        s = e.PageValueScoreExact(tables(), NOW)
        base = s.score("https://x.com/a", 0, 0, "0", "12345", 1000, 50, 5.0, [])
        old = s.score("https://x.com/a", 0, 0, "0", str(NOW - 3000 * 86400), 1000, 50, 5.0, [])
        self.assertLess(old, base)   # pt 长度 > 5 才衰减
        a2 = s.score("https://x.com/a", 2, 1 << 13, "0", str(NOW - 3000 * 86400), 1000, 50, 5.0, [])
        self.assertAlmostEqual(a2, s.adjust_score_pct("0", s.get_basic_score(0.0)) + 10)  # adc=2 跳过 pc 与 pt

    def test_empty_line_in_whitelist_marks_everything(self):
        s = e.PageValueScoreExact(tables(adc_whitelist=["moe.edu.cn", ""]), NOW)
        self.assertEqual(s.pre_adc("https://spam.example.org/", 0), 3.0)

    def test_non_numeric_timestamp_decays_one_day(self):
        s = e.PageValueScoreExact(tables(), NOW)
        self.assertAlmostEqual(s.down_factor("abcdefghij", str(NOW), 2000.0), 2 / (math.exp(1 / 2000.0) + 1))


class JavaSemantics(unittest.TestCase):
    def test_float32(self):
        s = e.PageValueScoreExact(tables(), NOW)
        self.assertEqual(s.pre_dr("https://www.unknown.org/"), e.f32(1 / 3))
        self.assertEqual(e.java_float_to_string(s.pre_dr("https://www.unknown.org/")), "0.33333334")
        self.assertEqual(s.pre_sr(86), e.f32(0.86))
        self.assertEqual(e.fastjson_number(s.pre_sr(86), True), "0.86")

    def test_feature_json_format(self):
        s = e.PageValueScoreExact(tables(), NOW)
        s.score("https://www.zhihu.com/q", 0, 0, "0", "0", 1000, 86, 5.0, [])
        j = s.feature_json()
        self.assertTrue(j.startswith('{"adc":0,"spr":'), j)
        self.assertIn('"sr":0.86}', j)

    def test_split_and_parsers(self):
        self.assertEqual(e.java_split_tab("a\t\tb\t\t"), ["a", "", "b"])
        self.assertEqual(e.java_split_tab(""), [""])
        self.assertEqual(e.java_parse_long("+12"), 12)
        for bad in ("1_0", " 12", "12.0", "abc"):
            with self.assertRaises(e.JavaJobFailure):
                e.java_parse_long(bad)
        self.assertEqual(e.java_parse_float(" 3f "), 3.0)
        self.assertEqual(e.fastjson_get_integer({"a": "85.0"}, "a"), 85)
        with self.assertRaises(e.JavaJobFailure):
            e.fastjson_get_integer({"a": "85.5"}, "a")

    def test_double_to_string(self):
        self.assertEqual(e.java_double_to_string(60.0), "60.0")
        self.assertEqual(e.java_double_to_string(0.00012345), "1.2345E-4")
        self.assertEqual(e.java_double_to_string(12345678.9), "1.23456789E7")

    def test_parse_line_failures(self):
        ok = 'https://a.com/x\tx\t5\tx\tx\t{"adc":"{\\"level\\":2}","pcLong":1024,"pureTextLen":800,"sr":70,"spr":3.2}'
        r = e.parse_input_line(ok)
        self.assertEqual((r["flag"], r["level"], r["pc"], r["pct"]), ("5", 2, 1024, "0"))
        for line, msg in (
            ("u\tx\t0", "ArrayIndexOutOfBounds"),
            ("u\tx\t0\tx\tx\tnot-json", "JSONException"),
            ('u\tx\t0\tx\tx\t{"adc":"","sr":1,"spr":1}', "pureTextLen"),
            ('u\tx\t0\tx\tx\t{"pureTextLen":1,"sr":1,"spr":1}', "adc 为 null"),
            ('u\tx\t0\tx\tx\t{"adc":"{}","pureTextLen":1,"sr":1,"spr":1}', "Integer.parseInt(null)"),
        ):
            with self.assertRaises(e.JavaJobFailure) as cm:
                e.parse_input_line(line)
            self.assertIn(msg, str(cm.exception))

    def test_normalization(self):
        rows = [("a", 1.0, "{}"), ("b", 2.0, "{}"), ("c", 1.0, "{}")]
        with self.assertRaises(e.JavaJobFailure):
            e.normalization_score(rows, False)
        out = e.normalization_score(rows, True)
        self.assertEqual([r[0] for r in out], ["b", "c", "a"])  # 并列按 "url\\tfea" 降序
        self.assertEqual([r[2] for r in out], [1000, 999, 998])
        iv = e.generate_url_interval(-2, 3, 1000000)
        self.assertEqual(len(iv), 1001)
        self.assertGreaterEqual(iv[-1], 1000000)


class EndToEnd(unittest.TestCase):
    def test_cli(self):
        args = ["--input", os.path.join(SAMPLE, "input.tsv")]
        for flag, f in (("--spr", "spr.tsv"), ("--dr-site", "dr_site.tsv"), ("--dr-suffix", "dr_suffix.tsv"),
                        ("--ow", "ow.tsv"), ("--ow-blacklist", "ow_blacklist.txt"),
                        ("--adc-whitelist", "adc_whitelist.txt")):
            args += [flag, os.path.join(SAMPLE, f)]
        args += ["--region", "zh", "--now", str(NOW)]
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, os.path.join(HERE, "npv_exact.py")] + args + ["--output", d],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)              # 样例含坏行：默认整体失败
            self.assertIn("作业失败", r.stderr)
            r = subprocess.run([sys.executable, os.path.join(HERE, "npv_exact.py")] + args +
                               ["--output", d, "--skip-bad-rows", "--scroll", "1", "--assume-sort-works"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(os.path.join(d, "npv", "part-00000")))
            line = open(os.path.join(d, "npv_ori", "part-00000")).readline().rstrip("\n").split("\t")
            self.assertEqual(len(line), 3)
            self.assertTrue(line[2].startswith('{"adc":'))


if __name__ == "__main__":
    unittest.main()
