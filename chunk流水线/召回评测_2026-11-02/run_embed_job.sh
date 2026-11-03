#!/usr/bin/env bash
# 任务队列入口：一个任务推理一种切分方法。用法：bash run_embed_job.sh <方法名> [其他 embed_job.py 参数]
#
# 每种方法提交一个任务（各占一张卡）：
#   bash run_embed_job.sh content_br
#   bash run_embed_job.sh md_struct
#   bash run_embed_job.sh html_struct
# 某种方法太慢想拆给两张卡：再追加分片参数，分别提交
#   bash run_embed_job.sh md_struct --num-shards 2 --shard-id 0
#   bash run_embed_job.sh md_struct --num-shards 2 --shard-id 1
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

if [ $# -lt 1 ]; then
  echo "用法：bash $0 <content_br|md_struct|html_struct> [其他 embed_job.py 参数]" >&2
  exit 2
fi
METHOD="$1"
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/embed_job.py" \
  --method "$METHOD" \
  --out-root "$OUT_ROOT" \
  --model-path "$MODEL_PATH" \
  --loader "$LOADER" \
  --pooling "$POOLING" \
  --doc-prefix "$DOC_PREFIX" \
  --query-prefix "$QUERY_PREFIX" \
  --max-length "$MAX_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  "$@"
