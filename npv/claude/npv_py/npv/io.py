"""输入行解析与输出写出，对应 Java PageValueScoreMain.getScoreRDD 的解析部分。"""
import json
from typing import Optional, Tuple

from .scorer import ScoreInput


class BadRow(Exception):
    """单行解析失败（对应 Java 中被 catch 后返回 null 的情况）。"""


def _get_long(obj: dict, key: str) -> int:
    """Java: try { getLong(key) } catch { 0 }。"""
    v = obj.get(key)
    if v is None or v == "":
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def parse_line(line: str) -> Tuple[ScoreInput, str]:
    """解析一行输入 TSV，返回 (ScoreInput, 第 2 列的类别标记)。任何异常都转成 BadRow。"""
    try:
        cols = line.rstrip("\n").split("\t")
        url = cols[0]
        flag = cols[2]
        obj = json.loads(cols[5])
        if not isinstance(obj, dict):
            raise BadRow("col5 is not a JSON object")

        pr = obj.get("pr")
        pr = "0" if pr is None or pr == "" else str(pr)

        adc = obj.get("adc")
        level = "0"
        if adc is not None and adc != "":
            adc_obj = json.loads(adc) if isinstance(adc, str) else adc
            level = str(adc_obj.get("level", "0"))

        # Java: getInteger/getDouble 返回 null 后自动拆箱 NPE -> 坏行
        if obj.get("pureTextLen") is None or obj.get("sr") is None or obj.get("spr") is None:
            raise BadRow("missing pureTextLen/sr/spr")

        x = ScoreInput(
            url=url,
            pr=float(pr),
            adc=int(level),
            pc=_get_long(obj, "pcLong"),
            pct=_get_long(obj, "pct"),
            pt=_get_long(obj, "pt"),
            pure_text_len=int(obj["pureTextLen"]),
            sr=int(obj["sr"]),
            spr=float(obj["spr"]),
        )
        return x, flag
    except BadRow:
        raise
    except Exception as e:  # noqa: BLE001 - 与 Java 一致，任何异常都算坏行
        raise BadRow(str(e)) from e


def features_json(features: dict) -> str:
    return json.dumps(features, ensure_ascii=False, separators=(",", ":"))


def format_row(url: str, score: float, fea: str, level: Optional[int] = None) -> str:
    if level is None:
        return f"{url}\t{score!r}\t{fea}"
    return f"{url}\t{score!r}\t{level}\t{fea}"
