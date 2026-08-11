# SELECT 论文调研与复现说明

## 论文身份

- 标题：*SELECting over Tokens: Curating Pre-training Data at Scale via Token Classification*
- 作者：Xin Tong, Weidong Zhang, Jiaang Li, Haibin Chen, Shilei Liu, Langming Liu, Kangtao Lv, Yujin Yuan, Wenbo Su, Bo Zheng
- 机构：Future Living Lab of Alibaba
- 会议：ACL 2026 主会 Long Paper，Anthology ID `2026.acl-long.2219`，页 48060–48085
- 主页：https://aclanthology.org/2026.acl-long.2219/
- PDF：https://aclanthology.org/2026.acl-long.2219.pdf

论文提出把网页清洗转成 BIO token classification：`B/I` 是保留的主内容 span，`O` 是删除的噪声。它用单次 encoder 前向替代 PROX-C 的生成式脚本解码。

## 完整方法与本项目边界

论文的完整流程不是单阶段：

1. 用 DeepSeek-R1 按“只能删除、禁止改写”的 prompt 清洗小规模网页。
2. 对原文与清洗文本做 longest-matching-segments 对齐与验证，构造约 400K BIO 样本。
3. 第一训练阶段：Qwen3-0.6B-Base 改为双向注意力，在 FineWeb 上做 MLM 持续预训练。
4. 第二训练阶段：在 400K BIO 数据上进行 token-classification SFT，并学习相邻标签 transition probability。
5. 用 Viterbi 解码得到最终保留/删除序列。

本项目按用户要求只复现第 3 步。最终实验中“在不同清洗语料上从头预训练 Qwen3-1.7B 40B token”是论文的效果评测，不是 SELECT refinement model 的第一训练阶段。

## 论文明确披露的一阶段配置

来源：论文 Appendix C.3。

- base model：Qwen3-0.6B-Base
- 将 causal self-attention 替换成 bidirectional self-attention
- corpus：FineWeb 100B English token
- objective：Masked Language Modeling
- maximum sequence length：32K
- batch size：512
- schedule：cosine，peak `5e-5`，最终 `1e-6`
- framework：FSDP

论文未披露：mask probability、80/10/10 与否、mask token 的构造、optimizer/betas/epsilon、weight decay、warmup、packing、document boundary attention、seed、checkpoint 频率。

## 官方资产检索结论

在 ACL Anthology 论文主页、论文 PDF、Future-Living-Lab GitHub 组织、GitHub 仓库搜索、Hugging Face 与 ModelScope 检索中，没有发现作者发布的 SELECT 代码、SELECT checkpoint 或 400K BIO 数据集。可直接核实的对应公开资产只有：

- 官方基座：https://huggingface.co/Qwen/Qwen3-0.6B-Base
- 官方 FineWeb：https://huggingface.co/datasets/HuggingFaceFW/fineweb

因此本仓库没有伪造“官方代码”或“官方 checkpoint”的说法；所有未披露选择均单独记录。

## 关键实现判断

### 1. 不能只改 attention mask

Hugging Face 的 Qwen3 decoder-only 假设同时存在于三处：

- `Qwen3Model._update_causal_mask()` 构造三角 mask。
- SDPA 在 mask 为 `None` 时自动推断 `is_causal=True`。
- `Qwen3ForCausalLM` 使用右移一位的 causal loss。

本实现同时覆盖三处：padding-only 非因果 mask、SDPA 显式 `is_causal=False`、同位置 masked CE。

### 2. 只计算 masked-position logits

8K × 151,936 的全量 BF16 logits 约 2.32GiB，CE 临时上采样还会继续增加显存。MLM 只在约 15% 位置计算 loss，因此训练时先筛选 hidden states，再过 `lm_head`。这与全量 logits 上按 `labels != -100` 计算的 CE 数值等价，单测已锁定该等价性。

### 3. Qwen tokenizer 没有 mask token

本项目增加 `<|mask|>`。Qwen3 模型 embedding table 有 151,936 行，而原 tokenizer 长度是 151,669；新增 token ID 为 151,669，仍落在现有 embedding table 内，所以不会错误缩小模型词表，也无需扩展权重矩阵。

### 4. Packing 假设

论文没有说明 document boundary masking。本项目把 FineWeb 文档以 EOS 分隔后连续 pack 到固定 8K block，并允许跨文档双向注意力。这是单卡条件下吞吐和内存最稳定的实现假设，但不应被误认为论文明确设定。

## 缩放配置

- sequence length：8,192
- train tokens：99,999,744（12,207 个完整 block）
- validation tokens：1,048,576（128 个完整 block）
- micro-batch：1
- gradient accumulation：4
- effective token batch：32,768
- optimizer steps：约 3,052
- 参数与 AdamW state：FP32；前向/反向计算：BF16 autocast
- attention backend：PyTorch SDPA
- gradient checkpointing：开启
- optimizer：fused AdamW，`β=(0.9, 0.95)`，`eps=1e-8`，`wd=0.1`
- gradient clipping：1.0
- LR：100 step warmup 后 cosine `5e-5 → 1e-6`

## 已获得的验证证据

- 13 个离线单元测试覆盖双向右文、padding mask、MLM 无 shift、masked-only CE 等价、权重键兼容、保存/重载、数据窗口与 RNG 恢复，以及 dtype 分离、resume 不变量和 Windows 本地 tokenizer 路径。
- 8K context、FP32 参数/AdamW state + BF16 autocast、全参数训练完成 2 optimizer step。
- 吞吐 4,022–4,690 token/s。
- 峰值 PyTorch allocated/reserved 为 10.92/11.79GiB。
- checkpoint 重载后：310 个权重张量可读，mask token 一致，右侧 token 改动让左侧 logits 最大变化 3.484375，masked loss 有限，数据 SHA-256 匹配。
- checkpoint 内保存严格 resume 不变量；相同配置可恢复，数据指纹或 MLM/优化器关键超参变化会被拒绝。
