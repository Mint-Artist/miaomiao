#!/usr/bin/env bash
# torchrun entrypoint for Ascend/HCCL nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/stage1_ascend910c_8k_10b.json}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if (( NNODES > 1 )); then
  : "${NODE_RANK:?NODE_RANK is required when NNODES > 1}"
  : "${MASTER_ADDR:?MASTER_ADDR is required when NNODES > 1}"
  : "${MASTER_PORT:?MASTER_PORT is required when NNODES > 1}"
fi
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export SELECT_MODEL_DIR="${SELECT_MODEL_DIR:-${REPO_ROOT}/artifacts/models/Qwen3-0.6B-Base}"
export SELECT_DATA_DIR="${SELECT_DATA_DIR:-${REPO_ROOT}/artifacts/data/fineweb_select_10b_8k}"
export SELECT_OUTPUT_DIR="${SELECT_OUTPUT_DIR:-${REPO_ROOT}/outputs/ascend910c-stage1-8k-10b}"

export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT OMP_NUM_THREADS
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  LAST_INDEX=$((NPROC_PER_NODE - 1))
  export ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 "${LAST_INDEX}")"
fi

if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

python3 "${SCRIPT_DIR}/preflight.py" --config "${CONFIG_PATH}"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train.py" \
  --config "${CONFIG_PATH}"
