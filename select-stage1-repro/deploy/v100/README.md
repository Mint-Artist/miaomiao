# V100 多机多卡训练

这里提供可直接复制到 Linux 服务器的 `torchrun + NCCL + DDP` 部署层。训练核心位于 `src/select_repro/distributed.py`；V100 固定使用 FP16 autocast、GradScaler、FP32 参数/AdamW state，不使用 BF16。

0.6B 模型在 32GB V100 上可以采用 DDP，每张卡保存完整模型，无需先引入 FSDP。8K 双向注意力仍建议开启 gradient checkpointing。

## 目录

- `train.py`：V100 入口。
- `launch.sh`：单机或多机 `torchrun` 启动器。
- `preflight.py`：CUDA、NCCL、GPU、模型和数据预检。
- `config-smoke.json`：8K、2 optimizer step 的端到端验证配置。
- `config-10b.json`：约 10B token 的正式配置。
- `requirements-v100.txt`、`Dockerfile`：宿主机与容器环境。

## 环境

推荐使用 Python 3.10、PyTorch 2.6 的 CUDA 11.8 构建。CUDA 11.8 对 Volta/V100 保持支持，也避免把 V100 部署绑定到更新的 GPU 架构。

PyTorch 2.6 官方发布的是 cu118/cu124/cu126 wheel，这里选择 [官方 cu118 安装项](https://pytorch.org/get-started/previous-versions/)，不使用不存在的 cu121 wheel。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0
python -m pip install -r deploy/v100/requirements-v100.txt
```

容器方式：

```bash
docker build -f deploy/v100/Dockerfile -t select-v100:latest .
docker run --gpus all --ipc=host --net=host -it \
  -v "$PWD":/workspace -v /shared:/shared select-v100:latest
```

## 路径约定

启动器支持三个路径变量；不设置时使用仓库内默认位置：

```bash
export SELECT_MODEL_DIR=/shared/models/Qwen3-0.6B-Base
export SELECT_DATA_DIR=/shared/data/fineweb_select_10b_8k
export SELECT_OUTPUT_DIR=/shared/checkpoints/select-v100-10b
```

多机时 `SELECT_OUTPUT_DIR` 必须是所有节点可见的同一个共享挂载。训练器会在启动阶段写入探针并由所有 rank 验证；如果各节点只是同名本地目录，会在训练前直接失败。模型和数据可以是共享挂载，也可以在每个节点保留内容完全相同的本地副本。

约 10B 的 packed 数据可用仓库脚本生成；该过程支持断点继续：

```bash
python scripts/prepare_fineweb.py \
  --output-dir "$SELECT_DATA_DIR" \
  --train-tokens 9999998976 \
  --validation-tokens 1048576
```

## 先跑 2-step smoke

当前项目已有 100M 数据时，可先验证完整的多卡通信、反向和 checkpoint：

```bash
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export NNODES=1 NODE_RANK=0 GPUS_PER_NODE=8
export CONFIG_PATH="$PWD/deploy/v100/config-smoke.json"
export SELECT_DATA_DIR="$PWD/artifacts/data/fineweb_select_100m"
export SELECT_OUTPUT_DIR=/shared/checkpoints/select-v100-smoke
bash deploy/v100/launch.sh
```

`config-smoke.json` 仍会保存最终可恢复 checkpoint，因此能验证实际落盘格式，而不只是前向。

## 多机正式训练

以下示例为两台机器、每台 8 张 V100。两台机器使用完全相同的 `MASTER_ADDR`、`MASTER_PORT`、路径和配置，只改变 `NODE_RANK`。

节点 0：

```bash
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
export NNODES=2 NODE_RANK=0 GPUS_PER_NODE=8
export NCCL_SOCKET_IFNAME=ib0
bash deploy/v100/launch.sh
```

节点 1：

```bash
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
export NNODES=2 NODE_RANK=1 GPUS_PER_NODE=8
export NCCL_SOCKET_IFNAME=ib0
bash deploy/v100/launch.sh
```

以太网环境把 `NCCL_SOCKET_IFNAME` 换成实际网卡，例如 `eth0` 或 `bond0`。不要照抄不存在的网卡名。启动前可在每个节点单独运行：

```bash
python deploy/v100/preflight.py --config deploy/v100/config-10b.json
```

## batch、尾部数据与恢复

每个 optimizer step 的全局 token 数为：

```text
sequence_length × micro_batch_size × gradient_accumulation_steps × WORLD_SIZE
```

训练器先生成唯一的全局样本顺序，再按 rank 跨步切分；各 rank 不会重复取样。不能组成完整 distributed micro-batch 的最后少量序列会被统一丢弃，实际 token 数写入 `run_manifest.json` 和 checkpoint manifest。

恢复时保持 world size 和训练超参数不变：

```bash
export SELECT_RESUME_FROM=/shared/checkpoints/select-v100-10b/checkpoints/step-00005000
bash deploy/v100/launch.sh
```

rank 0 保存模型、优化器和 GradScaler；每个 rank 单独保存 masking RNG、CPU/device RNG 和数据位置。FP16 梯度溢出会降低 loss scale 并重试同一批数据，不会把跳过的更新计入 step/token。

## 已知边界

- 当前实现要求固定 world size 精确恢复；不支持改卡数后继续同一数据顺序。
- V100 不支持硬件 BF16；不要把正式配置改成 BF16。
- FlashAttention-2 官方 CUDA 路径不覆盖 V100，本实现使用 PyTorch SDPA。
- 正式 10B 任务会产生较大的 optimizer checkpoint，请预留共享存储空间；默认每 5,000 step 保存一次。
