"""ScoreConfig <-> JSON。实验配置放在 configs/*.json，每轮实验只换配置文件。"""
import dataclasses
import json
from typing import Any, Dict

from npv.config import PcRule, ScoreConfig


def config_to_dict(cfg: ScoreConfig) -> Dict[str, Any]:
    d = dataclasses.asdict(cfg)
    d["pc_rules"] = [dataclasses.asdict(r) for r in cfg.pc_rules]
    d["text_len_rules"] = [list(t) for t in cfg.text_len_rules]
    d["adc_bonus"] = {str(k): v for k, v in cfg.adc_bonus.items()}
    return d


def config_from_dict(d: Dict[str, Any]) -> ScoreConfig:
    d = dict(d)
    if "pc_rules" in d:
        d["pc_rules"] = tuple(
            PcRule(name=r["name"], bit=int(r["bit"]), factor=float(r["factor"]),
                   url_substrings=tuple(r.get("url_substrings", ())),
                   lowercase_url=bool(r.get("lowercase_url", False)))
            for r in d["pc_rules"])
    if "text_len_rules" in d:
        d["text_len_rules"] = tuple((int(t), float(fa)) for t, fa in d["text_len_rules"])
    if "adc_bonus" in d:
        d["adc_bonus"] = {int(k): float(v) for k, v in d["adc_bonus"].items()}
    known = {f.name for f in dataclasses.fields(ScoreConfig)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"未知配置项: {sorted(unknown)}")
    return ScoreConfig(**d)


def load_config(path: str) -> ScoreConfig:
    with open(path, encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def save_config(cfg: ScoreConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, ensure_ascii=False, indent=2)
        f.write("\n")
