"""单元测试：python -m unittest discover -s tests  或  pytest tests"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from npv import (PageValueScore, ScoreConfig, ScoreInput, Tables, assign_levels,  # noqa: E402
                 days_between, decay_factor, generate_url_interval, parse_site,
                 parse_timestamp_seconds)
from npv.config import PcRule  # noqa: E402
from npv.io import BadRow, parse_line  # noqa: E402
from npv.normalization import level_for_rank  # noqa: E402
from npv.tables import spr_max_from_file  # noqa: E402

NOW = 1757030400  # 2026-09-05 00:00:00 UTC
DAY = 86400


def make_tables(**kw) -> Tables:
    base = dict(
        dr_site={"www.gov.cn": "3"},
        dr_suffix={"edu.cn": "3", ".com": "2"},
        ow={"www.zhihu.com": "1"},
        spr_max=10.0,
        ow_blacklist=["spam.example.org"],
        adc_whitelist=["moe.edu.cn"],
    )
    base.update(kw)
    return Tables(**base)


def make_input(**kw) -> ScoreInput:
    base = dict(url="https://www.example.com/a", adc=0, pc=0, pct=0, pt=0,
                pure_text_len=1000, sr=50, spr=5.0)
    base.update(kw)
    return ScoreInput(**base)


class TimeTests(unittest.TestCase):
    def test_parse_timestamp(self):
        self.assertEqual(parse_timestamp_seconds("1757030400"), 1757030400)
        self.assertEqual(parse_timestamp_seconds("1757030400123"), 1757030400)  # 13 位毫秒
        self.assertEqual(parse_timestamp_seconds(1757030400), 1757030400)
        for bad in ("0", 0, None, "", "abc", "12345", "17570304001", "-1757030400"):
            self.assertEqual(parse_timestamp_seconds(bad), -1, bad)

    def test_days_between_clamps_negative(self):
        self.assertEqual(days_between(NOW - 3 * DAY, NOW, timezone.utc), 3)
        self.assertEqual(days_between(NOW + 3 * DAY, NOW, timezone.utc), 0)

    def test_decay_factor(self):
        self.assertEqual(decay_factor(0, NOW, 2000.0), 1.0)          # 缺失 -> 中性
        self.assertEqual(decay_factor("bad", NOW, 2000.0), 1.0)
        self.assertEqual(decay_factor(NOW + DAY, NOW, 2000.0), 1.0)  # 未来 -> 钳制
        f = decay_factor(NOW - 2000 * DAY, NOW, 2000.0, timezone.utc)
        self.assertAlmostEqual(f, 2 / (math.e + 1), places=9)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.s = PageValueScore(make_tables(), NOW)

    def test_sr_spr_score_piecewise(self):
        s = self.s
        self.assertEqual(s.sr_spr_score(-0.01, 0.5), 0.25)              # sr 缺失
        self.assertAlmostEqual(s.sr_spr_score(1.0, 1.0), 1.25 / 1.1)     # 上限 1.136
        self.assertAlmostEqual(s.sr_spr_score(0.5, 0.0), (0.5 + 0.25 * 0.1) / 1.1)  # spr=0 -> 0.1
        self.assertAlmostEqual(s.sr_spr_score(0.5, 0.36), (0.5 + 0.25 * 0.9) / 1.1)  # 阈值点映射到 0.9
        self.assertAlmostEqual(s.sr_spr_score(0.5, 0.2), (0.5 + 0.25 * 0.2) / 1.1)
        self.assertAlmostEqual(s.sr_spr_score(0.5, -1.0), (0.5 + 0.25 * 0.1) / 1.1)  # 负值钳 0 再补 0.1

    def test_spr_normalized_once(self):
        f = self.s.gen_features(make_input(spr=5.0, sr=100), ())
        self.assertAlmostEqual(f["spr"], 0.5)
        # 0.5 > 0.36 -> (0.5-0.36)*(0.1/0.64)+0.9
        expected_spr = (0.5 - 0.36) * (0.1 / 0.64) + 0.9
        self.assertAlmostEqual(f["spr_sr"], (1.0 + 0.25 * expected_spr) / 1.1)

    def test_dr_lookup_order(self):
        s = self.s
        self.assertAlmostEqual(s.pre_dr("www.gov.cn"), 1.0)          # 站点表
        self.assertAlmostEqual(s.pre_dr("www.pku.edu.cn"), 1.0)      # 后缀表
        self.assertAlmostEqual(s.pre_dr("www.example.com"), 2 / 3)
        self.assertAlmostEqual(s.pre_dr("www.unknown.org"), 1 / 3)   # 默认

    def test_ow_priority(self):
        s = self.s
        self.assertEqual(s.pre_ow("www.moe.gov.cn", (".gov.cn",)), 1.0)    # 区域白名单优先
        self.assertEqual(s.pre_ow("spam.example.org", ()), 0.0)            # 黑名单
        self.assertEqual(s.pre_ow("www.zhihu.com", ()), 1.0)               # 官网表
        self.assertEqual(s.pre_ow("www.example.com", ()), 0.0)

    def test_adc_whitelist(self):
        s = self.s
        self.assertEqual(s.pre_adc("https://www.moe.edu.cn/x", 0), 3.0)
        self.assertEqual(s.pre_adc("https://www.moe.edu.cn/x", 1), 1.0)   # 已有等级优先
        self.assertEqual(s.pre_adc("https://www.example.com/x", 0), 0.0)


class AdjustTests(unittest.TestCase):
    def setUp(self):
        self.s = PageValueScore(make_tables(), NOW)

    def test_pc_floor_and_priority(self):
        s = self.s
        self.assertEqual(s.adjust_pc("u", 1 << 20, 10.0), (60.0, "floor_bit20"))
        self.assertEqual(s.adjust_pc("u", 1 << 20, 80.0), (80.0, "floor_bit20"))
        self.assertEqual(s.adjust_pc("u", (1 << 10) | (1 << 13), 10.0), (7.0, "bit10"))  # 顺序优先
        self.assertEqual(s.adjust_pc("u", 1 << 13, 10.0), (6.0, "bit13"))
        self.assertEqual(s.adjust_pc("https://a.com/NEWS/1", 0, 10.0), (9.0, "bit11_or_news"))
        self.assertEqual(s.adjust_pc("https://a.com/u/1", 0, 10.0), (6.0, "bit26_or_user"))
        self.assertEqual(s.adjust_pc("https://a.com/U/1", 0, 10.0), (10.0, ""))  # /u/ 区分大小写
        self.assertEqual(s.adjust_pc("u", 0, 10.0), (10.0, ""))

    def test_text_len(self):
        s = self.s
        self.assertAlmostEqual(s.adjust_text_len(299, 10.0), 8.0)
        self.assertAlmostEqual(s.adjust_text_len(300, 10.0), 9.0)
        self.assertAlmostEqual(s.adjust_text_len(500, 10.0), 10.0)

    def test_adc_bonus(self):
        s = self.s
        self.assertEqual(s.adjust_adc(10.0, 2.0), 20.0)
        self.assertEqual(s.adjust_adc(10.0, 3.0), 25.0)
        self.assertEqual(s.adjust_adc(10.0, 1.0), 10.0)
        self.assertEqual(s.adjust_adc(10.0, 0.0), 10.0)


class ScoreTests(unittest.TestCase):
    def test_end_to_end_hand_computed(self):
        s = PageValueScore(make_tables(), NOW)
        # sr=100 -> 1.0; spr=10 -> 1.0 -> 映射 1.0; spr_sr = 1.25/1.1
        # dr: .com -> 2/3; ow: zhihu 官网 -> 1
        x = make_input(url="https://www.zhihu.com/question/1", sr=100, spr=10.0,
                       pct=NOW - 365 * DAY, pt=NOW - 365 * DAY, pure_text_len=1000, pc=1 << 10)
        r = s.score(x, ())
        basic = 60 * (1.25 / 1.1) + 12 * (2 / 3) + 4 * 1.0
        expected = basic * decay_factor(x.pct, NOW, 2000.0) * 0.7 * decay_factor(x.pt, NOW, 3000.0)
        self.assertAlmostEqual(r.score, expected, places=9)
        self.assertEqual(list(r.features), ["sr", "spr", "spr_sr", "dr", "ow", "adc"])

    def test_adc_high_skips_pc_and_pt(self):
        s = PageValueScore(make_tables(), NOW)
        x = make_input(adc=2, pc=1 << 13, pt=NOW - 3000 * DAY, pct=0)
        r = s.score(x, ())
        self.assertAlmostEqual(r.score, s.basic_score(r.features) + 10.0)

    def test_missing_time_is_neutral_and_deterministic(self):
        s1 = PageValueScore(make_tables(), NOW)
        s2 = PageValueScore(make_tables(), NOW + 400 * DAY)
        x = make_input(pct=0, pt=0)
        self.assertEqual(s1.score(x, ()).score, s2.score(x, ()).score)

    def test_config_injection(self):
        cfg = ScoreConfig(fea_weight={"spr_sr": 0, "dr": 0, "ow": 100},
                          pc_rules=(PcRule("only13", 13, 0.5),))
        s = PageValueScore(make_tables(), NOW, cfg)
        x = make_input(url="https://www.zhihu.com/", pc=1 << 10)  # bit10 不再是规则
        self.assertAlmostEqual(s.score(x, ()).score, 100.0)
        x = make_input(url="https://www.zhihu.com/", pc=1 << 13)
        self.assertAlmostEqual(s.score(x, ()).score, 50.0)


class NormalizationTests(unittest.TestCase):
    def test_intervals_shape(self):
        n = 1_000_000
        iv = generate_url_interval(-2, 3, n)
        self.assertEqual(len(iv), 1001)
        self.assertEqual(iv[0], 0)
        self.assertTrue(all(b > a for a, b in zip(iv, iv[1:])))  # 严格递增
        self.assertGreaterEqual(iv[-1], n)
        self.assertLess(iv[-1], n + 1000)                      # 向上取整最多多 1000
        # 最宽的桶应在 z=0 附近（idx≈600）；z=0 两侧密度几乎相同，向上取整后会并列，允许一点偏差
        widths = [b - a for a, b in zip(iv, iv[1:])]
        self.assertTrue(abs(widths.index(max(widths)) - 600) <= 15)
        self.assertLess(widths[0], widths[600])   # 顶部（z=3）桶远小于中部桶

    def test_level_for_rank(self):
        iv = [0, 1, 3, 6, 10]  # 4 桶
        self.assertEqual(level_for_rank(0, iv), 4)
        self.assertEqual(level_for_rank(1, iv), 3)
        self.assertEqual(level_for_rank(2, iv), 3)
        self.assertEqual(level_for_rank(9, iv), 1)
        self.assertEqual(level_for_rank(10, iv), 4 + 1 - 5)  # 落在区间外：Java 的 1001 - size

    def test_assign_levels_deterministic_ties(self):
        items = [("a", 1.0), ("b", 2.0), ("c", 1.0), ("d", 5.0)]
        out = assign_levels(items, score_of=lambda t: t[1], tie_key=lambda t: t[0])
        self.assertEqual([it[0] for it, _, _ in out], ["d", "b", "a", "c"])
        self.assertEqual([lv for _, _, lv in out], [1000, 999, 998, 997])
        self.assertEqual(out, assign_levels(list(reversed(items)), score_of=lambda t: t[1],
                                            tie_key=lambda t: t[0]))


class IoTests(unittest.TestCase):
    def test_parse_line(self):
        obj = {"adc": json.dumps({"level": 2}), "pcLong": 1024, "pct": "1757030400",
               "pureTextLen": 800, "sr": 70, "spr": 3.2}
        x, flag = parse_line("\t".join(["https://a.com/x", "x", "5", "x", "x", json.dumps(obj)]))
        self.assertEqual((x.url, flag, x.adc, x.pc, x.pct, x.pt), ("https://a.com/x", "5", 2, 1024, 1757030400, 0))
        obj["adc"] = {"level": 1}  # adc 直接是对象也接受
        x, _ = parse_line("\t".join(["u", "x", "0", "x", "x", json.dumps(obj)]))
        self.assertEqual(x.adc, 1)

    def test_bad_rows(self):
        for line in ("u\tx\t0\tx\tx\tnot-json", "u\tx\t0", "u\tx\t0\tx\tx\t" + json.dumps({"sr": 1})):
            with self.assertRaises(BadRow):
                parse_line(line)

    def test_spr_max_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "spr.tsv")
            with open(p, "w") as f:
                f.write("a\t0\nb\t-1\n")
            with self.assertRaises(ValueError):
                spr_max_from_file(p)
            with open(p, "w") as f:
                f.write("")
            with self.assertRaises(ValueError):
                spr_max_from_file(p)

    def test_parse_site(self):
        self.assertEqual(parse_site("https://WWW.Example.com:8080/a?b=1"), "www.example.com")
        self.assertEqual(parse_site("example.com/x"), "example.com")


class EndToEndTests(unittest.TestCase):
    def test_sample_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            sample, out = os.path.join(d, "sample"), os.path.join(d, "out")
            subprocess.run([sys.executable, os.path.join(ROOT, "make_sample.py"), "--out", sample,
                            "--rows", "2000"], check=True, capture_output=True)
            args = ["--input", f"{sample}/input.tsv", "--spr", f"{sample}/spr.tsv",
                    "--dr-site", f"{sample}/dr_site.tsv", "--dr-suffix", f"{sample}/dr_suffix.tsv",
                    "--ow", f"{sample}/ow.tsv",
                    "--ow-blacklist", f"{sample}/ow_blacklist.txt",
                    "--adc-whitelist", f"{sample}/adc_whitelist.txt",
                    "--output", out, "--region", "zh", "--scroll", "1", "--now", str(NOW)]
            r = subprocess.run([sys.executable, os.path.join(ROOT, "run_npv.py")] + args,
                               check=True, capture_output=True, text=True)
            stats = dict(l.split("\t") for l in r.stderr.strip().splitlines())
            self.assertGreater(int(stats["bad_rows"]), 0)
            ori = open(f"{out}/npv_ori.tsv").read().splitlines()
            npv = open(f"{out}/npv.tsv").read().splitlines()
            self.assertEqual(len(ori) + len(npv), int(stats["rows"]))
            levels = [int(l.split("\t")[2]) for l in npv]
            self.assertEqual(levels, sorted(levels, reverse=True))
            self.assertTrue(all(1 <= lv <= 1000 for lv in levels))
            for l in ori:
                url, score, fea = l.split("\t")
                self.assertFalse(math.isnan(float(score)))
                self.assertEqual(set(json.loads(fea)), {"sr", "spr", "spr_sr", "dr", "ow", "adc"})


if __name__ == "__main__":
    unittest.main()
