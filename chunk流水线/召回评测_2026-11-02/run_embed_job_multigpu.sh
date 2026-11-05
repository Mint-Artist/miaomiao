#!/usr/bin/env bash
# 任务队列入口（多卡版，调用 embed_job_multigpu.py）。用法：bash run_embed_job_multigpu.sh <方法名...> [其他 embed_job_multigpu.py 参数]
#
# 一台 8 卡机器，一个任务算完三种方法（GPUS="all"，每张卡一个进程，按分片大小均衡分配）：
#   bash run_embed_job_multigpu.sh content_br md_struct html_struct
# 单卡任务（GPUS=""），每种方法提交一个：
#   bash run_embed_job_multigpu.sh md_struct
# 多台 8 卡机器分工：每台提交一个任务，再追加
#   bash run_embed_job_multigpu.sh content_br md_struct html_struct --num-shards 2 --shard-id 0   # 第 1 台
#   bash run_embed_job_multigpu.sh content_br md_struct html_struct --num-shards 2 --shard-id 1   # 第 2 台
#
# 下面的配置必须与 notebook（recall_pipeline.py 第 1 段）一致；已经推理过一部分时，照 OUT_ROOT/方法名/emb/meta.json 填。
# 填错了任务会打印旧值 / 新值并以退出码 1 退出，不会写出错误的向量。
set -euo pipefail

OUT_ROOT="/path/to/output/chunk_eval"   # notebook 的 OUT_ROOT（须是任务机器能读写的共享存储）
MODEL_PATH="/path/to/your/embedding/model"
LOADER="sentence_transformers"          # notebook 的 LOADER
POOLING="cls"                           # notebook 的 POOLING（sentence_transformers 时不参与计算，但要一致）
DOC_PREFIX=""                           # notebook 的 DOC_PREFIX
QUERY_PREFIX=""                         # notebook 的 QUERY_PREFIX
MAX_LENGTH=512                          # notebook 的 MAX_LENGTH
BATCH_SIZE=256                          # 只影响速度，不影响结果
GPUS="all"                              # "all" = 任务能看到的全部卡；"0,1,2,3" = 只用其中几张；"" = 单卡（用 cuda:0）

METHODS=()
while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
  METHODS+=("$1")
  shift
done
if [ ${#METHODS[@]} -eq 0 ]; then
  echo "用法：bash $0 <方法名...> [其他 embed_job_multigpu.py 参数]，例如 bash $0 content_br md_struct html_struct" >&2
  exit 2
fi
GPU_ARGS=()
if [ -n "$GPUS" ]; then
  GPU_ARGS=(--gpus "$GPUS")
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/embed_job_multigpu.py" \
  --method "${METHODS[@]}" \
  --out-root "$OUT_ROOT" \
  --model-path "$MODEL_PATH" \
  --loader "$LOADER" \
  --pooling "$POOLING" \
  --doc-prefix "$DOC_PREFIX" \
  --query-prefix "$QUERY_PREFIX" \
  --max-length "$MAX_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  "$@"
