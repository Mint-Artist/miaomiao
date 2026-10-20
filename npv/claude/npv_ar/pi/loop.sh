#!/usr/bin/env bash
# 无人值守循环：反复调用 run_once.sh，直到出现 STOP 文件、达到轮数上限或连续 N 轮没有保留。
# 用法：pi/loop.sh [最大轮数=50] [连续未保留上限=15]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_ROUNDS="${1:-50}"; MAX_STALE="${2:-15}"
mkdir -p "$ROOT/logs"
stale=0
for ((i = 1; i <= MAX_ROUNDS; i++)); do
  [ -f "$ROOT/STOP" ] && { echo "发现 STOP 文件，停止"; break; }
  before=$(grep -c $'\tkeep\t' "$ROOT/results.tsv" 2>/dev/null || echo 0)
  log="$ROOT/logs/round_$(date +%Y%m%d_%H%M%S).jsonl"
  echo "== 第 $i 轮  $(date '+%F %T')  日志 $log"
  "$ROOT/pi/run_once.sh" > "$log" 2>&1 || echo "   pi 退出码 $?"
  tail -n 1 "$ROOT/results.tsv" 2>/dev/null | cut -f1,4,5,6,7,17,19 || true
  after=$(grep -c $'\tkeep\t' "$ROOT/results.tsv" 2>/dev/null || echo 0)
  if [ "$after" -gt "$before" ]; then stale=0; else stale=$((stale + 1)); fi
  [ "$stale" -ge "$MAX_STALE" ] && { echo "连续 $stale 轮没有保留，停止"; break; }
done
