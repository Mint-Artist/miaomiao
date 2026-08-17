# sequence_BIO 服务器运行指南

本文档说明如何在 Linux 服务器上安装并运行 `sequence_BIO`，将原始网页文本与大模型的 deletion-only 清洗文本转换为 SELECT 风格的 BIO 训练数据。

## 1. 运行要求

- Python 3.9 或更高版本
- 能够读取 Hugging Face tokenizer 文件
- CPU 即可运行
- 不需要加载 Qwen3-0.6B 模型权重
- 不需要 GPU

程序只使用 `Qwen/Qwen3-0.6B-Base` 的 tokenizer 完成分词和 offset 映射。

## 2. 检查项目结构

进入包含 `sequence_BIO/` 的仓库根目录：

```bash
cd /path/to/miaomiao
```

目录结构应至少包含：

```text
miaomiao/
└── sequence_BIO/
    ├── __init__.py
    ├── __main__.py
    ├── alignment.py
    ├── labeling.py
    ├── cli.py
    ├── README.md
    ├── SERVER_RUN_GUIDE.md
    ├── examples/
    └── tests/
```

后续命令都应在仓库根目录执行，而不是进入 `sequence_BIO/` 后执行。

## 3. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "transformers>=4.51,<5"
```

如果服务器统一使用 Conda，也可以执行：

```bash
conda create -n sequence-bio python=3.11 -y
conda activate sequence-bio
python -m pip install "transformers>=4.51,<5"
```

## 4. 运行测试

正式处理数据前，先运行：

```bash
python -m unittest discover -s sequence_BIO/tests -v
```

预期结果：

```text
Ran 16 tests
OK
```

如果出现 `No module named sequence_BIO`，通常表示当前目录不是仓库根目录。请先执行：

```bash
pwd
ls sequence_BIO
```

## 5. 准备输入 JSONL

默认输入字段为：

- `id`：样本标识
- `source_text`：原始网页正文
- `refined_text`：大模型 deletion-only 清洗结果

一条输入示例：

```json
{"id":"doc-1","source_text":"首页导航\n这里是一段长度超过二十个字符的有效正文内容，用于进行BIO标签构造。\n版权所有","refined_text":"这里是一段长度超过二十个字符的有效正文内容，用于进行BIO标签构造。"}
```

必须注意：

1. JSONL 中每条样本只能占一个物理行。
2. 文本内部的换行在 JSON 文件中写成 `\n`。
3. `json.loads()` 读取后，`\n` 会成为真实换行符。
4. 输入中不再使用 `<br>`。
5. `refined_text` 应尽量遵守只删除、不改写、不重排的约束。

可以用下面的脚本检查输入格式：

```bash
python - <<'PY'
import json

path = "data/cleaned_pairs.jsonl"

with open(path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        assert isinstance(record["source_text"], str)
        assert isinstance(record["refined_text"], str)

print("JSONL format: OK")
PY
```

## 6. 使用项目样例运行

首次运行会下载 Qwen tokenizer，因此服务器需要能够访问 Hugging Face，或者已经准备好本地 tokenizer 目录。

```bash
python -m sequence_BIO \
  --input sequence_BIO/examples/pairs.jsonl \
  --output output/accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base
```

论文参数已经是程序默认值：

```text
L_min = 20 characters
Delta_max = 5 characters
max_length = 32768 tokens
```

显式写出全部参数时，命令为：

```bash
python -m sequence_BIO \
  --input sequence_BIO/examples/pairs.jsonl \
  --output output/accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base \
  --min-match-chars 20 \
  --max-adjust-chars 5 \
  --max-length 32768
```

## 7. 处理自己的数据

假设数据文件是：

```text
data/cleaned_pairs.jsonl
```

运行：

```bash
python -m sequence_BIO \
  --input data/cleaned_pairs.jsonl \
  --output output/accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base
```

程序会在终端输出统计，例如：

```text
processed=1000 accepted=850 aligned=800 adjusted=50 unaligned=140 truncated=10 errors=0
```

## 8. 输出文件

以上命令会自动生成四个文件：

```text
output/accepted.audit.jsonl
output/accepted.audit.sft.jsonl
output/accepted.audit.rejected.jsonl
output/accepted.audit.meta.json
```

### 8.1 `accepted.audit.jsonl`

保存完整审计信息，包括：

- 原始文本
- 教师清洗文本
- `aligned` 或 `adjusted` 状态
- 原文和清洗文本两侧的字符 span
- tokenizer 输出
- BIO 字符串序列
- BIO 整数标签
- token 边界调整信息

BIO 字符串序列位于：

```text
supervision.bio_tags
```

整数训练标签位于：

```text
supervision.labels
```

### 8.2 `accepted.audit.sft.jsonl`

这是后续训练 0.6B 模型应主要读取的文件。每条记录格式为：

```json
{
  "id": "doc-1",
  "input_ids": [],
  "attention_mask": [],
  "labels": []
}
```

必须满足：

```text
len(input_ids) == len(attention_mask) == len(labels)
```

标签映射为：

```text
O = 0
B = 1
I = 2
special token / padding = -100
```

### 8.3 `accepted.audit.rejected.jsonl`

保存未进入训练集的样本，包括：

- `unaligned`：无法按照论文规则可靠对齐
- `tokenization_truncated`：超过最大 token 长度
- `processing_error`：字段缺失、JSON 错误或其他异常

通过以下字段查看原因：

```text
rejection_reason
```

### 8.4 `accepted.audit.meta.json`

保存数据集配置与汇总统计，包括：

- tokenizer 名称
- `label2id` 和 `id2label`
- 对齐算法版本
- `L_min` 和 `Delta_max`
- 最大序列长度
- accepted、aligned、adjusted、unaligned、truncated 和 error 数量

## 9. 查看第一条 BIO 结果

```bash
python - <<'PY'
import json

path = "output/accepted.audit.jsonl"

with open(path, encoding="utf-8") as stream:
    record = json.loads(next(stream))

print("ID:", record.get("id"))
print("Alignment:", record["alignment"]["status"])
print("Matched spans:", record["alignment"]["matched_spans"])
print("Tokens:", record["tokenization"]["token_texts"])
print("BIO:", record["supervision"]["bio_tags"])
print("Labels:", record["supervision"]["labels"])
PY
```

检查时重点确认：

- 导航、广告、版权等噪声是否为 `O`
- 每个连续保留区域是否只有第一个 token 为 `B`
- 保留区域后续 token 是否为 `I`
- `input_ids`、`attention_mask` 和 `labels` 是否等长

## 10. 输入字段名称不同

如果数据格式为：

```json
{"uuid":"123","original_content":"原始文本","cleaned_content":"清洗文本"}
```

运行：

```bash
python -m sequence_BIO \
  --input data/cleaned_pairs.jsonl \
  --output output/accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base \
  --id-field uuid \
  --source-field original_content \
  --refined-field cleaned_content
```

## 11. 解析教师完整回复

如果输入字段保存的是：

```text
refinement_reason:
[doc]删除了导航和广告[/doc]

refined_text:
[doc]这里是清洗后的正文[/doc]
```

增加：

```text
--parse-teacher-response
```

完整命令：

```bash
python -m sequence_BIO \
  --input data/teacher_responses.jsonl \
  --output output/accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base \
  --parse-teacher-response
```

## 12. 推荐的首次全流程

不要直接从全量数据开始。先抽取 100 条：

```bash
head -n 100 data/cleaned_pairs.jsonl > data/cleaned_pairs.sample.jsonl
```

运行：

```bash
python -m sequence_BIO \
  --input data/cleaned_pairs.sample.jsonl \
  --output output/sample.accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base
```

查看统计：

```bash
cat output/sample.accepted.audit.meta.json
```

建议人工检查：

1. 至少 20 条 accepted 样本。
2. 至少 10 条 rejected 样本。
3. 所有 adjusted 样本的小幅修正是否合理。
4. `B/I/O` 标签是否与清洗结果一致。
5. 是否存在大量 `unaligned`，以及原因是否来自教师改写。

确认小样本结果合理后，再对全量数据运行相同命令。

## 13. 常见问题

### 13.1 无法访问 Hugging Face

可以提前将 tokenizer 下载到服务器本地，然后将 `--tokenizer` 改成本地目录：

```bash
python -m sequence_BIO \
  --input data/cleaned_pairs.jsonl \
  --output output/accepted.audit.jsonl \
  --tokenizer /data/models/Qwen3-0.6B-Base
```

本地目录必须包含完整 tokenizer 文件，并且能够被 `AutoTokenizer.from_pretrained()` 加载。

### 13.2 大量样本为 `unaligned`

重点检查教师清洗结果是否发生了：

- 改写或同义替换
- 标点标准化
- 空格或换行重排
- 内容顺序调整
- 新增总结或解释
- 保留片段短于 20 个字符

SELECT 数据构造要求清洗文本尽量由原文纯删除得到。

### 13.3 大量样本被截断

被截断样本不会进入 SFT 文件。可以先统计原始 token 长度，再决定：

- 保持论文的 32K 上限并丢弃超长样本
- 在进入本工具前按文档边界切分
- 针对实验调整 `--max-length`

不要在不了解数据分布的情况下静默截断并训练。

### 13.4 是否需要保存 transition 标签

不需要保存扁平化的 9 类 transition 标签。论文中的 transition 监督可以直接由 `labels` 动态构造：

```python
from sequence_BIO import derive_transition_supervision

from_labels, to_labels, transition_mask = derive_transition_supervision(labels)
```

真正训练 transition head 时，应对给定 `from_label` 后的三个候选 `to_label` 计算交叉熵。
