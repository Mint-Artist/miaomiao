"""npv_ar harness 的回归测试：严格指标、静态检查、门禁、端到端一轮。

  cd claude/npv_ar && python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))

import gate  # noqa: E402
from lint import lint_features  # noqa: E402
from metrics import paired_compare, strict_pairwise  # noqa: E402

GCFG = {"eps_floor": 0.002, "cv_gain_min": 0.002, "cv_tol": 0.002, "guard_tol": 0.01}


def pairs(n, pref="a"):
    return [{"url_a": f"a{i}", "url_b": f"b{i}", "pref": pref, "group": "g", "qid": f"q{i // 3}"} for i in range(n)]


class StrictMetricTest(unittest.TestCase):
    def test_ties_and_missing_cannot_inflate(self):
        ps = pairs(10)
        honest = {}
        for i in range(10):
            honest[f"a{i}"] = (10.0 if i < 6 else 1.0, None)
            honest[f"b{i}"] = (5.0, None)
        self.assertAlmostEqual(strict_pairwise(honest, ps)["accuracy"], 0.6)
        tied = dict(honest)
        for i in range(6, 10):
            tied[f"a{i}"] = (5.0, None)
        m = strict_pairwise(tied, ps)
        self.assertAlmostEqual(m["accuracy"], 0.8)  # 4 个同分各算半对，不是 1.0
        self.assertEqual(m["n_score_tie"], 4)
        missing = dict(honest)
        for i in range(6, 10):
            missing.pop(f"a{i}")
        m = strict_pairwise(missing, ps)
        self.assertAlmostEqual(m["accuracy"], 0.6)  # 缺失算错
        self.assertEqual(m["n_missing"], 4)

    def test_label_ties_excluded(self):
        ps = pairs(4) + pairs(2, pref="tie")
        s = {u: (1.0 + (u[0] == "a"), None) for p in ps for u in (p["url_a"], p["url_b"])}
        m = strict_pairwise(s, ps)
        self.assertEqual(m["n_eligible"], 4)
        self.assertEqual(m["n_label_tie"], 2)

    def test_paired_compare(self):
        ps = pairs(30)
        a = [{"qid": p["qid"], "outcome": 1.0} for p in ps]
        b = [{"qid": p["qid"], "outcome": 1.0 if i % 2 else 0.0} for i, p in enumerate(ps)]
        c = paired_compare(a, b, n_boot=200)
        self.assertAlmostEqual(c["delta"], 0.5)
        self.assertEqual(c["wins"], 15)
        self.assertEqual(c["losses"], 0)
        self.assertGreater(c["se_cluster"], 0)
        self.assertLess(c["p_sign"], 0.001)


class LintTest(unittest.TestCase):
    def _lint(self, src, limit=20):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(src)
        try:
            return lint_features(f.name, limit)
        finally:
            os.unlink(f.name)

    def test_blocks_io_and_imports(self):
        hard, _ = self._lint("import os\ndef gen_features(scorer, x, site_list):\n    return open('x').read()\n")
        self.assertTrue(any("os" in h for h in hard))
        self.assertTrue(any("open" in h for h in hard))

    def test_blocks_dunder(self):
        hard, _ = self._lint("def gen_features(scorer, x, site_list):\n    return x.__class__.__subclasses__()\n")
        self.assertTrue(hard)

    def test_allows_clean_code(self):
        hard, review = self._lint("import re\nfrom urllib.parse import urlparse\ndef gen_features(scorer, x, site_list):\n    return {'a': 1.0}\n")
        self.assertEqual(hard, [])
        self.assertEqual(review, [])

    def test_domain_flood_goes_to_review(self):
        doms = ", ".join(f"'site{i}.example.com'" for i in range(25))
        _, review = self._lint(f"def gen_features(scorer, x, site_list):\n    L = [{doms}]\n    return {{}}\n")
        self.assertTrue(review)


class GateTest(unittest.TestCase):
    def test_stage1(self):
        self.assertTrue(gate.stage1(0.80, 0.79, 10, 10, GCFG)[0])
        self.assertFalse(gate.stage1(0.79, 0.80, 10, 10, GCFG)[0])
        self.assertTrue(gate.stage1(0.799, 0.80, 9, 10, GCFG)[0])  # 持平且更简单
        self.assertFalse(gate.stage1(0.799, 0.80, 11, 10, GCFG)[0])

    def test_stage2_threshold_and_guardrails(self):
        best = {"hard_accuracy": 0.6, "buried_good_rate": 0.05}
        cur_ok = {"hard_accuracy": 0.6, "buried_good_rate": 0.05}
        st, _, eps = gate.stage2(0.01, 0.003, cur_ok, best, 10, 10, GCFG)
        self.assertEqual(st, "keep")
        self.assertAlmostEqual(eps, 0.003)
        st, reason, _ = gate.stage2(0.001, 0.003, cur_ok, best, 10, 10, GCFG)
        self.assertEqual(st, "discard")
        self.assertIn("持平", reason)
        cur_bad = {"hard_accuracy": 0.55, "buried_good_rate": 0.05}
        st, reason, _ = gate.stage2(0.05, 0.003, cur_bad, best, 10, 10, GCFG)
        self.assertEqual(st, "discard")
        self.assertIn("hard_accuracy", reason)
        st, _, _ = gate.stage2(-0.001, 0.003, cur_ok, best, 8, 10, GCFG)
        self.assertEqual(st, "keep")  # 持平且更简单


class EndToEndTest(unittest.TestCase):
    """把骨架复制到临时目录，走一遍：合成数据 -> 基线 -> 一次改动。需要 git。"""

    def test_baseline_then_change(self):
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, "claude", "npv_ar")
        os.makedirs(os.path.dirname(dst))
        for sib in ("npv_py", "npv_lab"):
            shutil.copytree(os.path.join(os.path.dirname(ROOT), sib), os.path.join(tmp, "claude", sib),
                            ignore=shutil.ignore_patterns("__pycache__", "runs"))
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", "runs", ".git", "results.tsv", "state.json", "snapshot", "labels", "logs"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "tools/prepare_synthetic.py", "--pairs", "900", "--grades", "300"], cwd=dst, check=True, capture_output=True, env=env)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=dst, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=dst, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=dst, check=True)
        r = subprocess.run([sys.executable, "harness/run.py", "--hypothesis", "基线", "--desc", "基线"], cwd=dst, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("status=keep", r.stdout)
        self.assertIn("stage=baseline", r.stdout)
        # 改动：加一个特征并纳入拟合
        fp = os.path.join(dst, "sandbox", "features.py")
        src = open(fp, encoding="utf-8").read().replace('"is_root": 1.0 if not segments else 0.0,',
                                                        '"is_root": 1.0 if not segments else 0.0,\n        "is_user_page": 1.0 if "/u/" in path or "/user/" in path else 0.0,')
        open(fp, "w", encoding="utf-8").write(src)
        cp = os.path.join(dst, "sandbox", "config.json")
        cfg = json.load(open(cp, encoding="utf-8"))
        cfg["fit"] = ["is_user_page"]
        json.dump(cfg, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        r = subprocess.run([sys.executable, "harness/run.py", "--hypothesis", "用户页降权", "--desc", "加 is_user_page"], cwd=dst, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout, r"status=(keep|discard)")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=dst, capture_output=True, text=True).stdout
        self.assertEqual(status.strip(), "", "每轮结束后工作区必须干净")
        rows = open(os.path.join(dst, "results.tsv"), encoding="utf-8").read().splitlines()
        self.assertEqual(len(rows), 3)  # 表头 + 2 行
        # 拒绝多余参数
        r = subprocess.run([sys.executable, "harness/run.py", "--hypothesis", "x", "--desc", "y", "--pairs", "z"], cwd=dst, capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
