#!/usr/bin/env bash
# push_to_github.sh — 把当前目录里的代码推到 GitHub 远端 (合并模式)
#
# 用法:
#   ./push_to_github.sh                 # 用默认 commit 信息
#   ./push_to_github.sh "自定义提交说明"  # 自定义
#
# 行为:
#   1. 若当前目录还不是 git 仓库，自动 git init
#   2. 写入 / 保留 .gitignore（忽略 .claude/ .omc/ 等本地状态）
#   3. 绑定 remote 到 Mint-Artist/miaomiao
#   4. 拉远端 main 合并（保留远端历史）
#   5. 暂存 + 提交本地新增/修改
#   6. push 到 origin/main

set -euo pipefail

REPO_URL="https://github.com/Mint-Artist/miaomiao.git"
BRANCH="main"
COMMIT_MSG="${1:-upload local code from $(hostname -s) @ $(date +%F_%H:%M)}"

cd "$(dirname "$0")"

log()  { printf "\033[1;36m[%s]\033[0m %s\n" "$(date +%T)" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*" >&2; exit 1; }

# 1) 确保是 git 仓库
if [ ! -d .git ]; then
    log "初始化 git 仓库"
    git init -q
    git branch -M "$BRANCH"
fi

# 2) 写入 .gitignore (若已存在则保留, 只追加缺失项)
declare -a IGNORE_ITEMS=(
    ".claude/"
    ".omc/"
    ".DS_Store"
    "*.log"
    "*.class"
    "target/"
    "build/"
    ".idea/"
    ".vscode/"
)
touch .gitignore
for item in "${IGNORE_ITEMS[@]}"; do
    if ! grep -qxF "$item" .gitignore 2>/dev/null; then
        echo "$item" >> .gitignore
    fi
done
log ".gitignore 已确认"

# 3) 绑定 remote
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi
log "remote origin -> $REPO_URL"

# 4) 拉远端并合并
log "拉取远端 $BRANCH ..."
git fetch origin "$BRANCH" 2>&1 || die "fetch 失败，检查网络或仓库权限"

if git rev-parse --verify HEAD >/dev/null 2>&1; then
    # 本地已有提交：尝试合并远端
    if ! git merge --allow-unrelated-histories --no-edit "origin/$BRANCH" 2>&1; then
        die "合并出现冲突，请手动解决后重新运行脚本"
    fi
else
    # 本地还没提交：先把远端当基线检出
    git reset --hard "origin/$BRANCH"
fi

# 5) 暂存 + 提交
git add -A
if git diff --cached --quiet; then
    log "本地没有新增/修改，跳过提交"
else
    git commit -m "$COMMIT_MSG"
    log "已提交: $COMMIT_MSG"
fi

# 6) push
log "推送到 origin/$BRANCH ..."
git push -u origin "$BRANCH"
log "完成 ✓"
