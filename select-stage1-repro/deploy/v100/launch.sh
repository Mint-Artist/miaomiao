#!/usr/bin/env bash
# torchrun entrypoint for V100/NCCL nodes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/deploy/v100/config-10b.json}"

export SELECT_MODEL_DIR="${SELECT_MODEL_DIR:-$ROOT_DIR/artifacts/models/Qwen3-0.6B-Base}"
export SELECT_DATA_DIR="${SELECT_DATA_DIR:-$ROOT_DIR/artifacts/data/fineweb_select_10b_8k}"
export SELECT_OUTPUT_DIR="${SELECT_OUTPUT_DIR:-$ROOT_DIR/outputs/v100-stage1-8k-10b}"

: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${NNODES:?NNODES is required}"
: "${NODE_RANK:?NODE_RANK is required}"
: "${GPUS_PER_NODE:=8}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python "$ROOT_DIR/deploy/v100/preflight.py" --config "$CONFIG_PATH"

exec torchrun \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --nproc-per-node "$GPUS_PER_NODE" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "$ROOT_DIR/deploy/v100/train.py" \
  --config "$CONFIG_PATH"
