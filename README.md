# chunk2026 网页切分流水线

代码入口和详细说明见 [chunk流水线/README.md](chunk流水线/README.md)。

需要 Python 3.9+。在本分支检出目录执行：

```bash
cd chunk流水线
python3 -m pip install -r requirements.txt
python3 run_jsonl.py --input /data/pages.jsonl --out-dir /data/chunk_run_001 --absolute-chars 512
```

输入 JSONL 每行包含 `url` 和原始 HTML 字符串 `pg`。
输出每页的简版 HTML、五种策略的详细结果，以及包含 `url` 和各策略字符串列表的 `summary.jsonl`。
默认列表是含结构上下文的检索文本；用 `--summary-field text` 改为可拼接还原正文的文本。
512 是本次显式指定的实验上限。

```bash
python3 -m unittest discover -s tests -v
python3 run_examples.py --absolute-chars 512 --out-dir output/my_experiment
```

测试样例及冻结基线随代码保留；新实验输出由运行命令生成，未纳入版本控制。
