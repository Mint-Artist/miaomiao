"""外部表的加载，对应 Java 构造函数里的 loadMapAndBcPathString / getSprMax / textFile().collect()。"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Tables:
    dr_site: Dict[str, str]
    dr_suffix: Dict[str, str]
    ow: Dict[str, str]
    spr_max: float
    pr_split: List[float]
    ow_blacklist: List[str]
    adc_whitelist: List[str]


def _lines(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def load_map_tsv(path: str) -> Dict[str, str]:
    """前两列 -> dict。与 Java 一样，列数不足直接报错。"""
    result: Dict[str, str] = {}
    for line in _lines(path):
        cols = line.split("\t")
        result[cols[0]] = cols[1]
    return result


def load_list(path: str) -> List[str]:
    return list(_lines(path))


def load_second_col_floats(path: str) -> List[float]:
    return [float(line.split("\t")[1]) for line in _lines(path)]


def spr_max_from_file(path: str) -> float:
    values = load_second_col_floats(path)
    if not values:
        raise ValueError(f"spr file is empty: {path}")
    m = max(values)
    if not m > 0:  # 同时拦截 0、负数、NaN
        raise ValueError(f"sprMax must be > 0 but got {m} from {path}")
    return m


def load_tables(spr_path: str, dr_site_path: str, dr_suffix_path: str, ow_path: str,
                pr_split_path: str, ow_blacklist_path: str, adc_whitelist_path: str) -> Tables:
    return Tables(
        dr_site=load_map_tsv(dr_site_path),
        dr_suffix=load_map_tsv(dr_suffix_path),
        ow=load_map_tsv(ow_path),
        spr_max=spr_max_from_file(spr_path),
        pr_split=load_second_col_floats(pr_split_path),
        ow_blacklist=load_list(ow_blacklist_path),
        adc_whitelist=load_list(adc_whitelist_path),
    )
