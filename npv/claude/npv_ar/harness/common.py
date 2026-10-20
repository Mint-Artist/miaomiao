"""harness 公共代码：路径、冻结基线的导入、sandbox 的加载、标签读取。

npv_ar 依赖同级的 ../npv_py（冻结基线）与 ../npv_lab（评估与拟合工具），三者一起搬运。
"""
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HARNESS_DIR)
CLAUDE_DIR = os.path.dirname(ROOT)
NPV_PY_DIR = os.path.join(CLAUDE_DIR, "npv_py")
NPV_LAB_DIR = os.path.join(CLAUDE_DIR, "npv_lab")
for _p in (NPV_PY_DIR, NPV_LAB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from npv import PageValueScore, ScoreInput, load_tables  # noqa: E402
from npv.config import ScoreConfig, region_site_list  # noqa: E402
from npv.io import BadRow, features_json, format_row, parse_line  # noqa: E402
from lab.config_io import config_from_dict, config_to_dict  # noqa: E402

BASELINE_FEATURES = ("sr", "spr", "spr_sr", "dr", "ow", "adc")
TABLE_FILES = {
    "spr": "spr.tsv", "dr_site": "dr_site.tsv", "dr_suffix": "dr_suffix.tsv",
    "ow": "ow.tsv", "ow_blacklist": "ow_blacklist.txt", "adc_whitelist": "adc_whitelist.txt",
}


def load_harness_config(path: Optional[str] = None) -> dict:
    with open(path or os.path.join(HARNESS_DIR, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_sandbox_config(path: str) -> Tuple[ScoreConfig, List[str]]:
    """sandbox/config.json = {"fit": [特征名...], "score": {ScoreConfig 字段...}}"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or "score" not in d:
        raise ValueError("config.json 必须是 {\"fit\": [...], \"score\": {...}}")
    fit = d.get("fit", [])
    if not isinstance(fit, list) or not all(isinstance(s, str) for s in fit):
        raise ValueError("config.json 的 fit 必须是字符串列表")
    return config_from_dict(d["score"]), list(fit)


def save_sandbox_config(cfg: ScoreConfig, fit: Sequence[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fit": list(fit), "score": config_to_dict(cfg)}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_features_module(path: str):
    spec = importlib.util.spec_from_file_location("sandbox_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "gen_features"):
        raise AttributeError("features.py 必须定义 gen_features(scorer, x, site_list)")
    return mod


class SandboxScorer(PageValueScore):
    """用 sandbox/features.py 生成特征，其余流水线沿用冻结基线。"""

    def __init__(self, tables, now_seconds: int, config: ScoreConfig, feature_module):
        super().__init__(tables, now_seconds, config)
        self.fm = feature_module

    def gen_features(self, x: ScoreInput, site_list: Sequence[str]) -> Dict[str, float]:
        return self.fm.gen_features(self, x, site_list)

    def basic_score(self, f: Dict[str, float]) -> float:
        # 拟合出的权重可以为负，基础分下限钳为 0，与原流水线"基础分非负"的前提一致，避免最终分越界
        return max(0.0, sum(f.get(k, 0.0) * w for k, w in self.cfg.fea_weight.items()))

    def with_weights(self, w: Dict[str, float]) -> "SandboxScorer":
        return SandboxScorer(self.t, self.now, replace(self.cfg, fea_weight=dict(w)), self.fm)


def build_scorer(snapshot_dir: str, cfg: ScoreConfig, feature_module, region: str, now: int):
    paths = {k: os.path.join(snapshot_dir, v) for k, v in TABLE_FILES.items()}
    tables = load_tables(paths["spr"], paths["dr_site"], paths["dr_suffix"], paths["ow"],
                         paths["ow_blacklist"], paths["adc_whitelist"])
    return SandboxScorer(tables, now, cfg, feature_module), tuple(region_site_list(region))


def baseline_scorer(snapshot_dir: str, region: str, now: int):
    paths = {k: os.path.join(snapshot_dir, v) for k, v in TABLE_FILES.items()}
    tables = load_tables(paths["spr"], paths["dr_site"], paths["dr_suffix"], paths["ow"],
                         paths["ow_blacklist"], paths["adc_whitelist"])
    return PageValueScore(tables, now, ScoreConfig()), tuple(region_site_list(region))


def iter_input(path: str) -> Iterator[Tuple[ScoreInput, str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield parse_line(line)
            except BadRow:
                continue


def count_bad_rows(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                parse_line(line)
            except BadRow:
                n += 1
    return n


def read_pairs(path: str) -> List[dict]:
    """pairs.tsv: url_a url_b pref [group] [qid]。qid 是按 query 分组的键，缺失时用 url_a 代替。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or cols[0] == "url_a":
                continue
            out.append({
                "url_a": cols[0], "url_b": cols[1], "pref": cols[2].strip().lower(),
                "group": cols[3] if len(cols) > 3 and cols[3] else "all",
                "qid": cols[4] if len(cols) > 4 and cols[4] else cols[0],
            })
    return out


def read_grades(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2 or cols[0] == "url":
                continue
            out.append({"url": cols[0], "grade": float(cols[1]), "group": cols[2] if len(cols) > 2 else "all"})
    return out


def read_scores(path: str, with_features: bool = False) -> Dict[str, Tuple[float, Optional[dict]]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            try:
                s = float(cols[1])
            except ValueError:
                continue
            fea = None
            if with_features and len(cols) >= 3:
                try:
                    fea = json.loads(cols[-1])
                except ValueError:
                    fea = None
            out[cols[0]] = (s, fea)
    return out


def site_of(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0].lower()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
