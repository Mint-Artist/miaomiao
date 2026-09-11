"""python -m unittest discover -s tests -v"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lab  # noqa: E402,F401
from lab import NPV_PY_DIR  # noqa: E402
from lab.cli_common import iter_rows  # noqa: E402
from lab.config_io import config_from_dict, config_to_dict, load_config  # noqa: E402
from lab.evaluate import (compare_pairwise, constraint_checks, grade_metrics,  # noqa: E402
                          pairwise_metrics)
from lab.explain import explain  # noqa: E402
from lab.diagnose import diagnose, format_report  # noqa: E402
from lab.fit import fit_pairwise_logistic, rescale_weights  # noqa: E402
from lab.scorer import LabScorer  # noqa: E402
from lab.stats import between_within_variance, quantile, sign_test_pvalue, spearman  # noqa: E402
from npv import PageValueScore, ScoreInput, load_tables  # noqa: E402
from npv.config import PcRule, ScoreConfig, region_site_list  # noqa: E402

SAMPLE = os.path.join(NPV_PY_DIR, "sample")
NOW = 1757030400
PY = sys.executable


def sample_tables():
    return load_tables(*(os.path.join(SAMPLE, f) for f in (
        "spr.tsv", "dr_site.tsv", "dr_suffix.tsv", "ow.tsv", "ow_blacklist.txt", "adc_whitelist.txt")))


def table_args():
    a = []
    for flag, f in (("--spr", "spr.tsv"), ("--dr-site", "dr_site.tsv"), ("--dr-suffix", "dr_suffix.tsv"),
                    ("--ow", "ow.tsv"), ("--ow-blacklist", "ow_blacklist.txt"),
                    ("--adc-whitelist", "adc_whitelist.txt")):
        a += [flag, os.path.join(SAMPLE, f)]
    return a + ["--region", "zh", "--now", str(NOW), "--input", os.path.join(SAMPLE, "input.tsv")]


class ParityTests(unittest.TestCase):
    """实验打分器在默认配置下必须与冻结基线逐条一致。"""

    def test_lab_scorer_matches_baseline(self):
        t = sample_tables()
        base, labs = PageValueScore(t, NOW), LabScorer(t, NOW)
        site_list = region_site_list("zh")
        n = 0
        for x, _ in iter_rows(os.path.join(SAMPLE, "input.tsv"), limit=500):
            self.assertEqual(base.score(x, site_list).score, labs.score(x, site_list).score, x.url)
            n += 1
        self.assertGreater(n, 400)

    def test_explain_matches_score(self):
        t = sample_tables()
        s = LabScorer(t, NOW)
        site_list = region_site_list("zh")
        for x, _ in iter_rows(os.path.join(SAMPLE, "input.tsv"), limit=300):
            e = explain(s, x, site_list)
            self.assertAlmostEqual(e.final, s.score(x, site_list).score, places=9)
            self.assertAlmostEqual(e.basic, sum(e.contributions.values()), places=9)

    def test_extra_features_have_zero_weight_by_default(self):
        s = LabScorer(sample_tables(), NOW)
        f = s.gen_features(ScoreInput("https://a.com/x/y", 0, 1 << 26, 0, 0, 500, 50, 1.0), ())
        self.assertIn("url_depth", f)
        self.assertEqual(f["pc_bit26"], 1.0)
        self.assertEqual(s.basic_score(f), sum(f[k] * w for k, w in ScoreConfig().fea_weight.items()))

    def test_check_weights_rejects_unknown_feature(self):
        s = LabScorer(sample_tables(), NOW, ScoreConfig(fea_weight={"spr_sr": 60, "typo": 1}))
        with self.assertRaises(KeyError):
            s.check_weights({"spr_sr": 1.0})


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        cfg = ScoreConfig(fea_weight={"spr_sr": 50, "url_depth": -3},
                          pc_rules=(PcRule("x", 40, 0.5, ("/x/",), True),), adc_bonus={2: 5.0})
        d = json.loads(json.dumps(config_to_dict(cfg)))
        self.assertEqual(config_from_dict(d), cfg)

    def test_baseline_json_is_default(self):
        self.assertEqual(load_config(os.path.join(ROOT, "configs", "baseline.json")), ScoreConfig())

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            config_from_dict({"nope": 1})


class StatsTests(unittest.TestCase):
    def test_quantile(self):
        self.assertEqual(quantile([3, 1, 2], 0.5), 2)
        self.assertEqual(quantile([1, 2, 3, 4], 0.5), 2.5)

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [1, 2, 2, 4]), 1.0)

    def test_sign_test(self):
        self.assertEqual(sign_test_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(sign_test_pvalue(5, 5), 1.0)
        self.assertLess(sign_test_pvalue(30, 5), 0.001)

    def test_between_within(self):
        tot, between, within = between_within_variance({"a": [1, 1, 1], "b": [5, 5, 5]})
        self.assertAlmostEqual(tot, 4.0)
        self.assertAlmostEqual(between, 4.0)
        self.assertAlmostEqual(within, 0.0)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.scores = {"a": (10.0, None), "b": (5.0, None), "c": (5.0, None), "d": (30.0, None)}
        self.pairs = [
            {"url_a": "a", "url_b": "b", "pref": "a", "group": "g1"},   # 对
            {"url_a": "b", "url_b": "d", "pref": "a", "group": "g1"},   # 错
            {"url_a": "b", "url_b": "c", "pref": "a", "group": "g2"},   # 分数相同，剔除
            {"url_a": "a", "url_b": "d", "pref": "tie", "group": "g2"}, # 标签相当，剔除
            {"url_a": "a", "url_b": "zz", "pref": "a", "group": "g2"},  # 缺 url
        ]

    def test_pairwise(self):
        m = pairwise_metrics(self.scores, self.pairs)
        self.assertEqual((m["n_used"], m["n_score_tie"], m["n_label_tie"], m["n_missing_url"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertEqual(m["accuracy_by_group"]["g1"]["n"], 2)
        self.assertAlmostEqual(m["label_tie_mean_gap"], 20.0)

    def test_compare(self):
        other = dict(self.scores, d=(1.0, None))  # 让第二对变对
        c = compare_pairwise(other, self.scores, self.pairs)
        self.assertEqual((c["a_right_b_wrong"], c["b_right_a_wrong"], c["both_right"]), (1, 0, 1))

    def test_grades(self):
        scores = {f"u{i}": (float(i), None) for i in range(100)}
        grades = [{"url": f"u{i}", "grade": 4 if i >= 80 else (0 if i < 20 else 2), "group": "all"} for i in range(100)]
        g = grade_metrics(scores, grades)
        self.assertGreater(g["spearman"], 0.8)  # 三档标签有大量并列，秩相关达不到 1
        self.assertEqual(g["buried_good_rate"], 0.0)
        self.assertEqual(g["top_bad_rate"], 0.0)
        self.assertEqual(g["calibration_violations"], 0)

    def test_constraints(self):
        self.assertTrue(constraint_checks(self.scores)["passed"])
        self.assertFalse(constraint_checks({"x": (float("nan"), None)})["passed"])
        self.assertFalse(constraint_checks({"x": (500.0, None)})["passed"])


class FitTests(unittest.TestCase):
    def test_recovers_sign(self):
        import random
        rnd = random.Random(1)
        diffs, targets = [], []
        for _ in range(400):
            d = [rnd.uniform(-1, 1), rnd.uniform(-1, 1)]
            targets.append(1.0 if 2 * d[0] - d[1] + rnd.gauss(0, 0.3) > 0 else 0.0)
            diffs.append(d)
        w, info = fit_pairwise_logistic(diffs, targets, ["f1", "f2"], epochs=300, lr=0.5)
        self.assertGreater(w["f1"], 0)
        self.assertLess(w["f2"], 0)
        self.assertGreater(info["train_accuracy"], 0.85)
        r = rescale_weights(w, 80)
        self.assertAlmostEqual(abs(r["f1"]) + abs(r["f2"]), 80)


class DiagnoseTests(unittest.TestCase):
    def test_diagnose_runs(self):
        s = LabScorer(sample_tables(), NOW)
        d = diagnose(s, iter_rows(os.path.join(SAMPLE, "input.tsv"), limit=800), region_site_list("zh"))
        self.assertEqual(d["rows"], 800)
        self.assertTrue(0 <= d["site_level_variance_share"]["final"] <= 1)
        self.assertAlmostEqual(sum(d["pc_rule_hits"].values()), 800)
        self.assertIn("spr_sr", format_report(d))


class CliTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            run = lambda script, *args: subprocess.run(  # noqa: E731
                [PY, os.path.join(ROOT, script)] + list(args), check=True, capture_output=True, text=True, env=env)
            run("run_lab.py", *table_args(), "--output", os.path.join(d, "base"),
                "--config", os.path.join(ROOT, "configs", "baseline.json"))
            run("make_sample_labels.py", "--input", os.path.join(SAMPLE, "input.tsv"), "--out", os.path.join(d, "labels"),
                "--pairs", "400", "--grades", "300")
            r = run("eval_scores.py", "--scores", os.path.join(d, "base", "npv_ori.tsv"),
                    "--pairs", os.path.join(d, "labels", "pairs.tsv"), "--grades", os.path.join(d, "labels", "grades.tsv"),
                    "--json", os.path.join(d, "eval.json"))
            self.assertIn("偏序一致率", r.stdout)
            j = json.load(open(os.path.join(d, "eval.json")))
            self.assertTrue(j["constraints"]["passed"])
            run("fit_weights.py", "--scores", os.path.join(d, "base", "npv_ori.tsv"),
                "--pairs", os.path.join(d, "labels", "pairs.tsv"), "--out", os.path.join(d, "fit.json"), "--epochs", "50")
            self.assertTrue(load_config(os.path.join(d, "fit.json")).fea_weight)
            r = run("experiment.py", "--name", "e1", "--config", os.path.join(d, "fit.json"), *table_args(),
                    "--pairs", os.path.join(d, "labels", "pairs.tsv"), "--baseline", os.path.join(d, "base", "npv_ori.tsv"),
                    "--runs-dir", os.path.join(d, "runs"), "--results", os.path.join(d, "results.tsv"))
            self.assertIn("vs_baseline_p=", r.stdout)
            self.assertEqual(len(open(os.path.join(d, "results.tsv")).read().splitlines()), 2)
            r = run("explain_url.py", *table_args(), "--grep", "gov.cn", "--limit", "1")
            self.assertIn("final:", r.stdout)
            r = run("diagnose_data.py", *table_args(), "--limit", "300")
            self.assertIn("站点级解释的方差占比", r.stdout)


if __name__ == "__main__":
    unittest.main()
