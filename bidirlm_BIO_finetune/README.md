# BidirLM BIO 微调

本目录使用 `sequence_BIO` 生成的最小 JSONL 数据，对本地
`BidirLM-0.6B-Base` 做 SELECT 风格的序列标注微调。

模型严格对应论文的两个输出：

- 分类头：每个 token 输出 `O/B/I` 三分类分数。
- 转移头：位置 `i` 的隐藏状态输出 `3×3` 分数，表示
  `i` 位置标签到 `i+1` 位置标签的条件转移。

训练损失为 `L_cls + L_tr`，预测使用
`log P_cls + log P_tr` 的 Viterbi 解码。BidirLM 已经是双向注意力编码器，
这里不会再次修改注意力结构。

## 输入格式

每行必须是：

```json
{"id":"doc-1","input_ids":[1,2],"attention_mask":[1,1],"labels":[1,2]}
```

三个数组长度必须相等，标签映射固定为：

```text
O=0, B=1, I=2, IGN=-100
```

脚本拒绝超过 `--max-length` 的数据，不会静默截断 BIO span。

## 安装

建议在服务器上新建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r bidirlm_BIO_finetune/requirements.txt
```

需要 Python 3.10 或更高版本。如果服务器需要固定 CUDA 11.8 的 PyTorch wheel，
可以先执行：

```bash
python -m pip install 'numpy<2'
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r bidirlm_BIO_finetune/requirements.txt
```

BidirLM 要求 `transformers>=4.57.6,<5.0.0`，并通过
`trust_remote_code=True` 加载本地模型代码。V100 使用 FP16，不使用 BF16，
默认注意力实现为 `eager`。

## 数据划分

如果一行就是一个独立网页，可以将约 3000 条数据按 8:1:1 划分：

```bash
python -m bidirlm_BIO_finetune.split_jsonl \
  --input data/accepted.audit.sft.jsonl \
  --output-dir data/bio_split
```

如果同一网页被拆成多行，不要直接使用这个脚本；应先按网页 ID 分组，确保
同一网页的所有片段只进入一个集合，避免数据泄漏。

## 单张 V100 32GB：推荐的首次实验

先用 20～100 条样本做 smoke test。完整数据的推荐命令为：

```bash
CUDA_VISIBLE_DEVICES=0 python -m bidirlm_BIO_finetune.train \
  --model-name-or-path /绝对路径/BidirLM-0.6B-Base \
  --train-file data/bio_split/train.jsonl \
  --validation-file data/bio_split/validation.jsonl \
  --output-dir outputs/bidirlm_lora_single \
  --finetuning-mode lora \
  --max-length 8192 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 8
```

有效全局 batch 为：

```text
per_device_batch × GPU 数 × gradient_accumulation_steps
```

所以上面单卡命令的有效 batch 是 16。

## 8 张 V100 32GB

使用 `torchrun` 启动 DDP。为了仍保持有效全局 batch 16，每卡 batch 1、
梯度累积设为 2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  -m bidirlm_BIO_finetune.train \
  --model-name-or-path /绝对路径/BidirLM-0.6B-Base \
  --train-file data/bio_split/train.jsonl \
  --validation-file data/bio_split/validation.jsonl \
  --output-dir outputs/bidirlm_lora_8gpu \
  --finetuning-mode lora \
  --max-length 8192 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --epochs 8
```

不需要修改代码或手工传入 rank；脚本从 `torchrun` 的环境变量读取
`RANK/LOCAL_RANK/WORLD_SIZE`。验证和 checkpoint 只由 rank 0 写入，避免冲突。

## 三种微调模式

```text
--finetuning-mode lora       默认；主干 LoRA，两个新 head 全参数训练
--finetuning-mode full       BidirLM 主干和两个 head 全参数训练
--finetuning-mode heads_only 冻结主干，只训练两个 head
```

LoRA 默认作用于 `q_proj,k_proj,v_proj,o_proj`，参数为：

```text
--lora-rank 16
--lora-alpha 32
--lora-dropout 0.05
```

默认学习率：LoRA 为 `1e-4`，full/heads-only 为 `1e-5`。可以通过
`--learning-rate` 覆盖。

## 输出与断点续训

训练目录包含：

```text
run_config.json   最终解析后的配置、数据量和可训练参数量
metrics.jsonl     每个 epoch 的训练、验证指标
best/             验证损失最优的模型
last/             最近一个 epoch 的模型和 trainer_state.pt
```

继续训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m bidirlm_BIO_finetune.train \
  --model-name-or-path /绝对路径/BidirLM-0.6B-Base \
  --train-file data/bio_split/train.jsonl \
  --validation-file data/bio_split/validation.jsonl \
  --output-dir outputs/bidirlm_lora_single \
  --finetuning-mode lora \
  --resume-from-checkpoint outputs/bidirlm_lora_single/last
```

恢复时模型结构以 checkpoint 中的 `select_config.json` 为准。

## 测试集预测与评估

```bash
CUDA_VISIBLE_DEVICES=0 python -m bidirlm_BIO_finetune.predict \
  --checkpoint outputs/bidirlm_lora_single/best \
  --base-model-name-or-path /绝对路径/BidirLM-0.6B-Base \
  --input data/bio_split/test.jsonl \
  --output outputs/bidirlm_lora_single/test_predictions.jsonl
```

程序输出 token accuracy、三标签 macro-F1、保留/删除二分类 F1 和严格匹配的
BIO span-F1。`test_predictions.jsonl` 保存 Viterbi 后的标签序列。

## 显存不足时

依次尝试：

1. 确认 `--per-device-train-batch-size 1`。
2. 保持动态 padding，并按长度分桶准备数据。
3. 将 `--max-length` 从 8192 降到 4096 或 2048，同时重新构建/过滤数据。
4. 不要关闭默认开启的 gradient checkpointing。

LoRA 会显著减少可训练参数和优化器状态，但 8192 tokens 的双向注意力本身仍然
具有平方级计算开销。

## 运行单元测试

```bash
python -m unittest discover -s bidirlm_BIO_finetune/tests -v
```
