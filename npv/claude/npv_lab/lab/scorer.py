"""实验打分器：继承基线 PageValueScore，只替换特征生成，其余流水线（衰减、页面类别、正文长度、adc）沿用。"""
from typing import Dict, Sequence

from npv import PageValueScore, ScoreInput  # noqa: F401  (npv_py 已由 lab/__init__ 加入路径)

from . import features as feature_module


class LabScorer(PageValueScore):
    def gen_features(self, x: ScoreInput, site_list: Sequence[str]) -> Dict[str, float]:
        return feature_module.gen_features(self, x, site_list)

    def basic_score(self, f: Dict[str, float]) -> float:
        return sum(f.get(k, 0.0) * w for k, w in self.cfg.fea_weight.items())

    def check_weights(self, f: Dict[str, float]) -> None:
        """确认 fea_weight 里的每个特征名都存在，避免打错名字被静默当成 0。"""
        missing = [k for k in self.cfg.fea_weight if k not in f]
        if missing:
            raise KeyError(f"fea_weight 中的特征不存在于 features.py 输出: {missing}")
