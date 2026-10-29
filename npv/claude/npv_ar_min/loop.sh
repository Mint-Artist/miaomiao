#!/usr/bin/env bash
# 无人值守：反复调用 run_once.sh。用法：./loop.sh [会话数=20]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs"
for ((i = 1; i <= ${1:-20}; i++)); do
  [ -f "$ROOT/STOP" ] && { echo "发现 STOP 文件，停止"; break; }
  log="$ROOT/logs/session_$(date +%Y%m%d_%H%M%S).log"
  echo "== 会话 $i  $(date '+%F %T')  日志 $log"
  "$ROOT/run_once.sh" > "$log" 2>&1 || echo "   退出码 $?"
  tail -n 3 "$ROOT/results.tsv" 2>/dev/null || true
done
