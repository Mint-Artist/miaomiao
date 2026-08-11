# SELECT ACL 2026：一阶段预训练复现（4090 / V100 / Ascend 910C）

本项目复现 Alibaba Future Living Lab 的 ACL 2026 长文
[SELECting over Tokens: Curating Pre-training Data at Scale via Token Classification](https://aclanthology.org/2026.acl-long.2219/)
中的第一阶段：把 `Qwen3-0.6B-Base` 的因果注意力改为双向注意力，并在 FineWeb 上以 MLM 目标进行持续预训练。

截至 2026-08-11，论文主页没有提供官方代码、SELECT 权重或作者构造的 400K BIO 数据集。因此这里是依据论文正文和附录实现的独立复现，不是官方仓库。原论文的一阶段使用 32K 上下文和 100B token；本项目提供单张 RTX 4090 的 8K/100M 缩放复现，以及 V100/NCCL、Ascend 910C/HCCL 的 8K/10B 多机多卡配方。

## 当前已完成

- 论文原件：`papers/2026.acl-long.2219.pdf`
- 官方基座模型：`artifacts/models/Qwen3-0.6B-Base`
- FineWeb 8K 冒烟数据：`artifacts/data/fineweb_smoke_8k`
- FineWeb 正式缩放数据：`artifacts/data/fineweb_select_100m`（99,999,744 train token）
- 8K 全参数训练冒烟 checkpoint：`outputs/smoke-qwen3-0.6b-8k-fp32/final`
- 论文级关键语义：真正的双向 SDPA、同位置 MLM loss、动态 15% masking、masked-only logits 节省显存、断点状态与数据指纹
- V100 多机多卡部署：[`deploy/v100`](deploy/v100/README.md)
- Ascend 910C 多机多卡部署：[`deploy/ascend910c`](deploy/ascend910c/README.md)

冒烟实测（RTX 4090 24GB，PyTorch 2.6.0+cu118，FP32 参数/AdamW state + BF16 autocast）：

| 指标 | 结果 |
| --- | ---: |
| 上下文长度 | 8,192 |
| 模型参数 | 596,049,920 |
| 完成的训练 token | 16,384 |
| 吞吐 | 4,022–4,690 token/s |
| PyTorch peak allocated | 10.92 GiB |
| PyTorch peak reserved | 11.79 GiB |
| 两步 loss | 12.04 → 9.64 |

两步 loss 只证明训练链路可执行，不代表模型已经收敛。按实测吞吐线性估算，100M token 的纯训练时间约 6.5–7 小时，另加 checkpoint 时间。

## 与论文的对应关系

| 项目 | 论文 | 本复现 |
| --- | --- | --- |
| 基座 | Qwen3-0.6B-Base | 相同官方权重 |
| 目标 | MLM 持续预训练 | 相同 |
| 注意力 | causal → bidirectional | 相同语义，SDPA 显式 `is_causal=False` |
| 数据 | FineWeb 100B token | FineWeb `sample-10BT` 中约 100M token |
| 上下文 | 32K | 8K |
| batch size | 512 | micro-batch 1，梯度累积 4 |
| 学习率 | cosine，5e-5 → 1e-6 | 相同峰值、终值和曲线 |
| 分布式 | FSDP | 单 GPU（4090）或 DDP（V100/910C） |

论文没有披露 MLM mask 比例、替换策略、AdamW 参数、warmup 或文档 packing 细节。本项目明确采用：15% MLM、BERT 80/10/10 替换、AdamW `(β1=0.9, β2=0.95, wd=0.1)`、100-step warmup、以 EOS 分隔的连续 token stream。所有这些推断项都会写入 checkpoint 的 `stage1_manifest.json`。

## 快速验证

环境中已经有匹配版本时，不需要重新安装依赖：

```powershell
python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__)"
python -m unittest discover -s tests -t . -v
```

验证已生成的 8K checkpoint、双向语义和数据哈希：

```powershell
python scripts\verify_stage1.py `
  --checkpoint outputs\smoke-qwen3-0.6b-8k-fp32\final `
  --data-dir artifacts\data\fineweb_smoke_8k
```

## 准备 FineWeb

冒烟集（64K train token + 8K validation token）：

```powershell
python scripts\prepare_fineweb.py `
  --output-dir artifacts\data\fineweb_smoke_8k `
  --train-tokens 65536 `
  --validation-tokens 8192 `
  --progress-every 1
```

正式缩放集（99,999,744 train token + 1,048,576 validation token）：

```powershell
python scripts\prepare_fineweb.py `
  --output-dir artifacts\data\fineweb_select_100m `
  --train-tokens 99999744 `
  --validation-tokens 1048576
```

输出是定长 `uint32` token 文件和 `metadata.json`。脚本会记录进度；进程中断后执行同一命令即可续传。选择 99,999,744 是因为它恰好等于 `12,207 × 8,192`，与“约 100M”只差 256 token。

## 训练

已验证的两步冒烟配置：

```powershell
python scripts\train_stage1.py --config configs\smoke_4090_8k.json
```

完整 100M-token 配置：

```powershell
python scripts\train_stage1.py --config configs\stage1_4090_8k_100m.json
```

正式配置共有约 3,052 个 optimizer step；每 1,000 step 保存一次。正式配置保留 FP32 参数和 FP32 AdamW state，单个可恢复 checkpoint 会明显大于 BF16 冒烟 checkpoint。若只需要最终权重，可把配置中的 `save_every_steps` 改为 `0`。恢复训练时，把 `resume_from` 指向某个 checkpoint，并保持其余数据与训练配置不变。

服务器多机多卡入口分别是：

```bash
# V100 / NCCL / FP16
bash deploy/v100/launch.sh

# Ascend 910C / HCCL / BF16
bash deploy/ascend910c/launch.sh
```

两套部署均包含 2-step smoke 配置、10B 配置、环境预检、容器模板和中文操作说明。共享训练器会验证跨 rank 配置/数据身份、输出共享挂载，保存 rank-local RNG 状态，并拒绝改变 world size 的非精确恢复。

## 实现要点

- `src/select_repro/modeling_bidirectional_qwen3.py`：与原 Qwen3 state dict 键兼容的双向 MLM 模型。
- `src/select_repro/data.py`：内存映射的 8K packed dataset 和动态 MLM collator。
- `scripts/prepare_fineweb.py`：流式下载、分词、定额物化和可恢复进度。
- `scripts/train_stage1.py`：单卡 BF16、gradient checkpointing、masked-only logits、checkpoint/manifest。
- `src/select_repro/distributed.py`：torchrun/DDP、无重复 rank 分片、NCCL/HCCL、FP16/BF16、共享 checkpoint 和严格恢复。
- `deploy/v100`：V100 多机多卡环境、预检、smoke/10B 配置与启动脚本。
- `deploy/ascend910c`：Ascend 910C 多机多卡环境、预检、smoke/10B 配置与启动脚本。
- `scripts/verify_stage1.py`：权重、tokenizer、数据指纹、MLM 和右文可见性验证。
- `docs/reproduction_notes.md`：论文调研、披露项与复现假设。

注意：这一阶段的输出只是 `Qwen3-0.6B-Base-Encoder` 的近似缩放版本。要得到最终 SELECT 清洗模型，仍需论文的第二阶段——在约 400K 条 BIO 标注数据上做 token-classification SFT；该数据和官方 checkpoint 目前未公开。

## 许可

- Qwen3-0.6B-Base：Apache-2.0；项目内保留原模型 `LICENSE`。
- FineWeb：ODC-By；使用和再分发时需遵循数据集卡及原始网页内容的适用条款。
- ACL 论文：ACL Anthology 页面标注为 CC BY 4.0。
