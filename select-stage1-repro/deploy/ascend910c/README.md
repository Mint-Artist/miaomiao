# Ascend 910C 多机多卡训练

这里提供 Ascend 910C（Atlas A3）上的 `torchrun + HCCL + DDP` 部署层。共享训练核心位于 `src/select_repro/distributed.py`；910C 路径使用 BF16 autocast、FP32 参数/AdamW state，不使用 GradScaler。

## 目录

- `train.py`：先加载 `torch_npu`，再进入共享训练器。
- `launch.sh`：加载 CANN 环境并启动单机或多机 HCCL。
- `preflight.py`：检查 CANN、TorchNPU、NPU、HCCL、模型和数据。
- `configs/stage1_ascend910c_8k_smoke.json`：8K、2-step 验证配置。
- `configs/stage1_ascend910c_8k_10b.json`：约 10B token 正式配置。
- `requirements-ascend.txt`、`Dockerfile`：附加依赖和官方基础镜像模板。

## 环境原则

优先使用昇腾官方、与服务器驱动和 CANN 配套的 PyTorch 容器。`torch`、`torch_npu` 和 CANN 必须按官方版本矩阵成套选择，不能在官方镜像里随意用 CPU/CUDA PyTorch 覆盖。`requirements-ascend.txt` 因此只安装本项目的上层 Python 依赖。

公开版本与安装入口以 [Ascend TorchNPU](https://pypi.org/project/torch-npu/) 和服务器供应方提供的 CANN 镜像矩阵为准。若集群管理员给出了固定组合，可设置 `SELECT_EXPECTED_TORCH_VERSION`、`SELECT_EXPECTED_TORCH_NPU_VERSION`、`SELECT_EXPECTED_CANN_VERSION`，让 `preflight.py` 对实际环境给出版本偏差警告。

宿主机环境示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r deploy/ascend910c/requirements-ascend.txt
```

容器模板需要显式传入与你的 CANN/驱动相匹配的官方镜像：

```bash
docker build \
  --build-arg BASE_IMAGE='<官方 Ascend for PyTorch 镜像>' \
  -f deploy/ascend910c/Dockerfile \
  -t select-ascend910c:latest .
```

## 路径和共享存储

```bash
export SELECT_MODEL_DIR=/shared/models/Qwen3-0.6B-Base
export SELECT_DATA_DIR=/shared/data/fineweb_select_10b_8k
export SELECT_OUTPUT_DIR=/shared/checkpoints/select-ascend910c-10b
```

多机时 `SELECT_OUTPUT_DIR` 必须是所有节点可见的同一共享挂载。训练器会在启动时跨 rank 验证文件探针。模型和 packed 数据可以使用共享挂载，也可以在每个节点放置内容相同的本地副本。

生成约 10B packed 数据：

```bash
python scripts/prepare_fineweb.py \
  --output-dir "$SELECT_DATA_DIR" \
  --train-tokens 9999998976 \
  --validation-tokens 1048576
```

## 先跑 2-step smoke

```bash
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SELECT_DATA_DIR="$PWD/artifacts/data/fineweb_select_100m"
export SELECT_OUTPUT_DIR=/shared/checkpoints/select-ascend910c-smoke

bash deploy/ascend910c/launch.sh \
  deploy/ascend910c/configs/stage1_ascend910c_8k_smoke.json
```

预检也可独立执行：

```bash
python deploy/ascend910c/preflight.py \
  --config deploy/ascend910c/configs/stage1_ascend910c_8k_smoke.json
```

## 双机 16 卡正式训练

节点 0：

```bash
export NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=eth0
bash deploy/ascend910c/launch.sh
```

节点 1：

```bash
export NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=eth0
bash deploy/ascend910c/launch.sh
```

多机模式下 `MASTER_ADDR`、`MASTER_PORT` 和 `NODE_RANK` 是强制项，启动器不会用 loopback 静默兜底。按实际集群设置 HCCL 网卡和防火墙；不要照抄 `eth0`。

## 数据和 checkpoint 语义

训练器生成一次全局确定性顺序，再按 rank 无重复切分。不能组成完整 distributed micro-batch 的尾部序列会统一丢弃；实际执行 token 数记录在 manifest 中。

rank 0 保存模型和优化器，每个 rank 保存自己的 masking/RNG/数据位置。恢复时必须使用原 world size：

```bash
export SELECT_RESUME_FROM=/shared/checkpoints/select-ascend910c-10b/checkpoints/step-00005000
bash deploy/ascend910c/launch.sh
```

## 已知边界

- 当前 Windows/4090 开发机无法执行 HCCL 真机验证；服务器上必须先跑 preflight 和 2-step smoke。
- 第一版采用 DDP，不依赖 MindSpeed/FSDP；0.6B 模型每卡完整复制，便于部署和排错。
- CANN、驱动、固件、TorchNPU 必须匹配；如果租到的实际型号不是 910C，需要重新核对官方兼容矩阵。
- 正式配置默认每 5,000 step 保存一次，请为共享 checkpoint 预留足够空间。
