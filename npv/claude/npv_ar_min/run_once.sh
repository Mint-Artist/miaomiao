#!/usr/bin/env bash
# 薄版本：用 opencode（内网叫 codeagent）跑一轮。每轮一个新进程。
# 环境变量：OPENCODE_BIN（默认 codeagent）、OPENCODE_MODEL（如 provider/model，可空）、ROUND_TIMEOUT（秒，默认 1800）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${OPENCODE_BIN:-codeagent}"
PROMPT="${1:-读 program.md，按它的规矩跑：先确认基线是否已记录，然后连续做实验，直到完成 ${ROUNDS_PER_SESSION:-10} 次 eval 为止。}"
cd "$ROOT"
git config user.name  >/dev/null 2>&1 || git config user.name npv_ar_min
git config user.email >/dev/null 2>&1 || git config user.email npv_ar_min@localhost
args=(run)
[ -n "${OPENCODE_MODEL:-}" ] && args+=(-m "$OPENCODE_MODEL")
exec timeout "${ROUND_TIMEOUT:-1800}" "$BIN" "${args[@]}" "$PROMPT" < /dev/null
