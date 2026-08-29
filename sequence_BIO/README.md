# sequence_BIO

按照 SELECT 论文的数据构造流程，将原始网页文本与教师大模型的 deletion-only 清洗文本对齐，验证样本，并生成 token 级 BIO 监督。

服务器部署和运行步骤见 [`SERVER_RUN_GUIDE.md`](SERVER_RUN_GUIDE.md)。

## 论文规则

- 使用附录 B Algorithm 1 的 longest-match-segments 算法。
- 最短精确匹配长度默认为 `L_min = 20` 个字符。
- `aligned`：所有匹配段无缺口地覆盖教师清洗文本。
- `adjusted`：仅存在匹配段之间的小幅改写，且原文 gap 与目标 gap 的长度差不超过 `Delta_max = 5`；使用原文 gap 修正并合并保留段。
- `unaligned`：不满足上述条件，丢弃。
- 每个保留 token span 的首 token 为 `B`，后续为 `I`，span 外为 `O`。

代码使用零基、左闭右开区间 `[start, end)`；论文公式使用的索引表示不同，但覆盖内容相同。

## 输入

JSONL 默认字段为 `source_text` 和 `refined_text`：

```json
{"id":"doc-1","source_text":"导航\n一段长度足够的原始正文……\n广告","refined_text":"一段长度足够的原始正文……"}
```

两个字段都必须已经使用真实换行符。JSONL 文件里换行显示为转义形式 `\n`，`json.loads` 后是 Python 字符串中的真实换行。代码不会执行 HTML 转换、`strip`、空白折叠或 Unicode 归一化。

若 `refined_text` 保存教师完整响应，增加 `--parse-teacher-response`，程序会提取：

```text
refined_text:
[doc]...[/doc]
```

## 安装

```bash
python -m pip install 'transformers>=4.51,<5'
```

BIO 构造只加载 tokenizer，不加载 0.6B 模型权重。BidirLM-0.6B-Base 与 Qwen3-0.6B-Base 使用同一 tokenizer，建议使用：

```text
Qwen/Qwen3-0.6B-Base
```

## 批量运行

从项目根目录执行：

```bash
python -m sequence_BIO \
  --input sequence_BIO/examples/pairs.jsonl \
  --output data/accepted.audit.jsonl \
  --tokenizer Qwen/Qwen3-0.6B-Base
```

若未显式指定其他路径，会生成：

```text
data/accepted.audit.jsonl          接受样本的完整审计数据
data/accepted.audit.sft.jsonl      最小 SFT 数据
data/accepted.audit.rejected.jsonl 拒绝、截断和错误样本
data/accepted.audit.meta.json      数据集配置及统计
```

也可分别传入：

```text
--sft-output
--rejected-output
--manifest-output
```

关键参数：

```text
--min-match-chars 20
--max-adjust-chars 5
--max-length 32768
```

超过 `max_length` 的样本会进入 rejected 文件，不会静默截断后进入训练。

## Accepted audit 数据

完整审计记录包含：

```json
{
  "id": "doc-1",
  "source_text": "...",
  "teacher_refined_text": "...",
  "adjusted_refined_text": "...",
  "dataset_status": "accepted",
  "rejection_reason": null,
  "alignment": {
    "status": "aligned",
    "algorithm": "select-longest-match-v1",
    "index_convention": "zero_based_half_open",
    "min_match_chars": 20,
    "max_adjust_chars": 5,
    "target_coverage": 1.0,
    "matched_spans": [
      {"source_span": [5, 36], "target_span": [0, 31], "length": 31}
    ],
    "retained_source_spans": [[5, 36]],
    "adjustments": []
  },
  "tokenization": {
    "tokenizer": "Qwen/Qwen3-0.6B-Base",
    "input_ids": [],
    "attention_mask": [],
    "token_texts": [],
    "offset_mapping": [],
    "token_spans": [],
    "boundary_adjustments": [],
    "truncated": false
  },
  "supervision": {
    "label2id": {"O": 0, "B": 1, "I": 2},
    "bio_tags": ["O", "B", "I"],
    "labels": [0, 1, 2],
    "ignore_index": -100
  }
}
```

真实完整样本见：

- `sequence_BIO/examples/real_bio_audit_example.json`
- `sequence_BIO/examples/real_bio_sft_example.json`

## 最小 SFT 数据

训练文件只包含：

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

special token 与 padding 的标签使用 `-100`，不参与 loss。

## Transition 监督

论文定义的是给定当前位置标签 `u` 后预测下一标签 `v` 的条件概率，不是一个全局 9 类分类任务。因此数据集不保存扁平化的 `transition_label_ids`。

训练时由 BIO 标签动态构造：

```python
from sequence_BIO import derive_transition_supervision

from_labels, to_labels, transition_mask = derive_transition_supervision(labels)
```

模型 transition head 可输出 `[batch, length - 1, 3, 3]`，按 `from_labels` 选择对应行后，对 `to_labels` 计算三分类交叉熵。

## Python API

```python
from transformers import AutoTokenizer
from sequence_BIO import label_aligned_pair

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    use_fast=True,
)

alignment, result = label_aligned_pair(
    source_text=source_text,
    refined_text=refined_text,
    tokenizer=tokenizer,
    min_match_chars=20,
    max_adjust_chars=5,
    max_length=32768,
)

if alignment.is_accepted and result is not None and not result.truncated:
    print(alignment.status)
    print(alignment.matched_spans)
    print(result.bio_tags)
    print(result.labels)
```

## 测试

```bash
python -m unittest discover -s sequence_BIO/tests -v
```
