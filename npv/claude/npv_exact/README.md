# npv_exact：原 Java 的逐行复刻（保留全部缺陷）

> 2026-09-11：离线端不再生产 pr，本文件已删除 pr 逻辑（`--pr-split`、`preProcessPr`、pr 特征与 4 分权重）。除此之外仍与原 Java 一致。

单文件 `npv_exact.py`，无第三方依赖。它对应的是**原始** `PageValueScore.java` / `PageValueScoreMain.java`，不是修复版。用途只有一个：拿真实表和真实输入在本地跑出与原 Spark 作业尽量一致的分数，定位复现差异。

三个 Python 版本的关系：

| 目录 | 对应的 Java | 用途 |
| --- | --- | --- |
| `npv_exact/` | 原始 Java（含 bug） | 与线上输出对齐、排查差异 |
| `npv_py/` | `claude/fixed/`（七处修复） | 冻结基线 |
| `npv_lab/` | 无 | 实验、评估 |

"我到底改了什么"的完整清单见 `../与原Java的差异记录.md`。

## 运行

```bash
python npv_exact.py --input in.tsv --spr spr.tsv --dr-site dr_site.tsv --dr-suffix dr_suffix.tsv \
    --ow ow.tsv --ow-blacklist ow_black.txt --adc-whitelist adc_white.txt \
    --output out --region zh --scroll 0 --tz Asia/Shanghai
```

- `--tz`：集群的默认时区。时间衰减按日历日计算，时区错一格可能差一天。
- `--now`：固定"当前时间"（秒）。不传则和原作业一样每条记录各取一次当前时间；对齐线上输出时传作业运行当天的时间戳即可。
- `--skip-bad-rows`：原作业遇到任一坏行整体失败，本脚本默认同样退出并打印是第几行、Java 会抛什么异常；加此参数改为跳过并计数。
- `--assume-sort-works`：`--scroll 1` 时原作业会在 `sortByKey` 处抛 ClassCastException，本脚本默认同样报错；若线上确认能跑通，加此参数按"分数降序、并列按 url+特征字符串降序"排序。

输出与 Spark 一致：`out/npv_ori/part-00000`（url、npv_ori、npv_fea），`out/npv/part-00000`（url、npv_ori、npv、npv_fea）。可直接用 `../npv_py/compare_with_java.py --java <线上输出目录> --py out/npv_ori/part-00000` 对齐。

结束时打印几项诊断：`spr_max` 的实际值、站点命中多个后缀的行数（此时结果取决于 Java HashMap 顺序）、名单文件中的空行数（空行会让所有 url 命中）。

## 模拟到什么程度

- Java `float` 32 位：特征计算每一步运算后舍入到 float32，与 Java 一致到最后一位（极罕见的双重舍入除外）。
- `String.split("\t")` 丢弃尾部空列；`Long/Integer/Float.parseXxx` 的接受规则与溢出；fastjson `getString/getLong/getInteger/getDouble` 的取值与异常。
- `Double.toString`、`Float.toString` 与 fastjson 数字格式（去掉结尾 `.0`）。
- 特征 JSON 的键序按 Java `HashMap` 迭代顺序（`adc, spr, spr_sr, ow, dr, sr`）。
- 后缀表按 Java `HashMap` 桶顺序迭代取第一个命中；桶内顺序用文件顺序近似。
- 名单文件保留空行。

无法模拟、需在原环境核对的：`ParseSiteUtil.parseSite` 的实现（本文件取 host 小写去端口）；fastjson 对非标准 JSON 的宽松解析；commons-math 的 erf 与 Python 的 erfc 在最后一位的差异（只可能影响归一化分桶边界）。

## 测试

```bash
python -m unittest test_exact -v
```

测试内容是"缺陷确实被保留"：spr 被除两次、缺失时间按 2024-05-22 衰减、未来时间放大、sprMax 为 0 出 NaN、名单空行让所有 url 命中、坏行整体失败、scroll=1 排序失败。
