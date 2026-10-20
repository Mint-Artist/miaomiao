#!/usr/bin/env bash
# 跑一轮实验：每轮一个全新的 pi 进程、全新上下文。
# 用法：pi/run_once.sh [提示词]。环境变量：PI_BIN、NPV_AR_MODEL（如 intranet/模型名）、PI_CODING_AGENT_DIR。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NPV_AR_ROOT="$ROOT"
export NPV_AR_PYTHON="${NPV_AR_PYTHON:-python3}"
export NPV_AR_MAX_TURNS="${NPV_AR_MAX_TURNS:-30}"
export PI_OFFLINE=1 PI_TELEMETRY=0
PI_BIN="${PI_BIN:-pi}"
MODEL="${NPV_AR_MODEL:?请设置 NPV_AR_MODEL，例如 intranet/模型名}"
GUARD="${NPV_AR_GUARD:-$ROOT/pi/guard.ts}"   # 正式环境请把 guard.ts 放到 agent 写不到的目录
PROMPT="${1:-开始一轮实验：先读 results.tsv 最后 20 行、journal.md 和 ideas.md，然后按 program.md 选一个假设去验证。}"
cd "$ROOT"
exec timeout "${NPV_AR_ROUND_TIMEOUT:-900}" "$PI_BIN" --mode json --no-session -ne --no-mcp -nc --no-approve \
  --model "$MODEL" --tools read,write,run_eval,show_errors \
  -e "$GUARD" --append-system-prompt "$ROOT/program.md" "$PROMPT" < /dev/null
# 注意：json 模式下 stdin 若保持打开，pi 会一直等待，必须重定向 /dev/null
