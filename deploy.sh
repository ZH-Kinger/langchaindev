#!/usr/bin/env bash
# 一键部署 AIOps Feishu Bot（Linux/macOS 版，deploy.ps1 的等价移植）。
#
# 流程：git archive <ref> → scp → 远端解压覆盖 → 同步删除 → 清 __pycache__
#       → docker compose restart bot → 轮询 /health。
#
# 为什么是"同步文件 + 重启"而不是重建镜像：docker-compose 以 `.:/app` 绑定挂载代码，
# 容器读的是宿主机上的真实文件，所以改代码 = 同步 + 重启进程（约 10 秒），无需 build。
# 只有这三类文件变更才需要 --build：requirements.txt（装依赖）、Dockerfile（镜像层）、
# .dockerignore（构建上下文）。脚本会检测并提示，不会替你决定。
#
# 服务器 /root/langchaindev **不是 git 仓库**，所以 tar 覆盖不会删掉已从仓库删除的文件 ——
# 靠服务器上的 .deployed_commit 记录上次部署的 commit，比对出删除清单再远端 rm，
# 等效 rsync --delete。
#
# 用法：
#   ./deploy.sh                      # 部署当前 HEAD
#   ./deploy.sh --ref main           # 部署指定 ref
#   ./deploy.sh --build              # 重启改为 up -d --build（改了依赖/Dockerfile 时用）
#   ./deploy.sh --dry-run            # 只打印将要做什么，不碰服务器
#   ./deploy.sh --skip-health        # 跳过部署后的 /health 轮询
#   ./deploy.sh --with-rag-assets    # 顺带补传 RAG 模型缓存+向量库（~391M，默认不传）
#   SERVER=root@<ip> ./deploy.sh     # 覆盖目标主机（默认读 ssh 别名 bot-new）
#
# 前置：本机 ssh 能免密连上 $SERVER。若用别名 bot-new，需要 ~/.ssh/config 里有对应条目：
#   Host bot-new
#       HostName <ip>
#       User root
#       IdentityFile ~/.ssh/wuji_deploy

set -euo pipefail

SERVER="${SERVER:-bot-new}"
REMOTE="${REMOTE:-/root/langchaindev}"
STAMP="$REMOTE/.deployed_commit"

REF="HEAD"
SKIP_HEALTH=0
DO_BUILD=0
DRY_RUN=0
WITH_RAG=0

C_STEP=$'\033[36m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OK=$'\033[32m'; C_OFF=$'\033[0m'
step() { printf '%s==> %s%s\n' "$C_STEP" "$*" "$C_OFF"; }
warn() { printf '%s!!  %s%s\n' "$C_WARN" "$*" "$C_OFF"; }
die()  { printf '%s✗   %s%s\n' "$C_ERR" "$*" "$C_OFF" >&2; exit 1; }
ok()   { printf '%s✓   %s%s\n' "$C_OK" "$*" "$C_OFF"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --ref)         REF="${2:-}"; [ -n "$REF" ] || die "--ref 需要参数"; shift 2 ;;
        --build)       DO_BUILD=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --skip-health) SKIP_HEALTH=1; shift ;;
        --with-rag-assets) WITH_RAG=1; shift ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
        *)             die "未知参数：$1（-h 看用法）" ;;
    esac
done

# 以仓库根为工作目录：脚本可能从任意子目录调用
REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || die "不在 git 仓库里"
cd "$REPO"

# ── 1. 解析提交、检查工作区 ──────────────────────────────────────────────────
SHA="$(git rev-parse --short "$REF")" || die "无法解析 ref: $REF"
SUBJECT="$(git log -1 --format=%s "$SHA")"
step "部署 $SHA  $SUBJECT"
step "目标   $SERVER:$REMOTE"

# 只有内容差异才算脏：这个仓库经历过 CRLF→LF 归一，git status 会因 stat 缓存误报一大片，
# 用 diff --quiet 判真实内容，避免每次部署都刷一屏假警告。
if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "工作区有未提交改动，本次只部署已提交的 $SHA（未提交内容不会上线）："
    git diff --name-only | head -10 | sed 's/^/    /'
fi

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" "$@"; }

if [ "$DRY_RUN" -eq 0 ]; then
    ssh_run true 2>/dev/null || die "连不上 $SERVER。检查 ~/.ssh/config 里的别名与私钥，或用 SERVER=user@host 覆盖。"
fi

# ── 2. 算出相对上次部署被删除的文件（等效 rsync --delete）──────────────────
LAST=""
if [ "$DRY_RUN" -eq 0 ]; then
    LAST="$(ssh_run "cat '$STAMP' 2>/dev/null" || true)"
    LAST="$(printf '%s' "$LAST" | tr -d '[:space:]')"
fi

DELETED=""
if [ -n "$LAST" ]; then
    if git cat-file -e "$LAST^{commit}" 2>/dev/null; then
        step "上次部署：$LAST，计算删除文件…"
        DELETED="$(git diff --name-only --diff-filter=D "$LAST..$SHA" || true)"
        [ -n "$DELETED" ] && printf '    将删除 %s 个文件\n' "$(printf '%s\n' "$DELETED" | grep -c .)"
    else
        warn "服务器记录的 $LAST 在本地仓库里不存在（分支没同步？），跳过删除计算"
    fi
else
    warn "服务器无部署记录，跳过删除计算"
fi

# ── 3. 检测需要 rebuild 的变更 ───────────────────────────────────────────────
# 这三类只在 docker build 时生效，restart 不会让它们起作用。
if [ -n "$LAST" ] && git cat-file -e "$LAST^{commit}" 2>/dev/null; then
    BUILD_TOUCHED="$(git diff --name-only "$LAST..$SHA" \
        | grep -E '^(requirements\.txt|Dockerfile|\.dockerignore)$' || true)"
    if [ -n "$BUILD_TOUCHED" ]; then
        warn "以下文件只在构建时生效，restart 不会让它们起作用："
        printf '%s\n' "$BUILD_TOUCHED" | sed 's/^/      /'
        if [ "$DO_BUILD" -eq 1 ]; then
            warn "已带 --build，本次会执行 up -d --build（较慢，会重建镜像）"
        else
            warn "本次仍走 restart。代码修复照常生效；上面这些等你方便时再补一次："
            warn "    ssh $SERVER 'cd $REMOTE && docker compose up -d --build'"
        fi
    fi
fi

# ── 4. 打包 ─────────────────────────────────────────────────────────────────
TAR="$(mktemp -t feishu-deploy.XXXXXX.tar)"
SH="$(mktemp -t feishu-deploy.XXXXXX.sh)"
trap 'rm -f "$TAR" "$SH"' EXIT
step "打包 → $TAR"
git archive --format=tar -o "$TAR" "$SHA"
[ -s "$TAR" ] || die "git archive 失败（产物为空）"

# ── 5. 生成远端脚本 ─────────────────────────────────────────────────────────
# 写成本地 .sh 再 scp 执行，而不是把整段塞进 ssh 的命令行：避开 argv 压平与多层引号解析。
# 这个项目在 SSH 双跳那条链上吃过嵌套引号的亏，同样的教训在这里照办。
RM_CMD=""
if [ -n "$DELETED" ]; then
    RM_CMD="$(printf '%s\n' "$DELETED" | while IFS= read -r f; do
        [ -n "$f" ] && printf "rm -f '%s/%s'\n" "$REMOTE" "$f"
    done)"
fi

if [ "$DO_BUILD" -eq 1 ]; then
    RESTART_CMD="docker compose -f '$REMOTE/docker-compose.yml' up -d --build"
else
    RESTART_CMD="docker compose -f '$REMOTE/docker-compose.yml' restart bot"
fi

cat > "$SH" <<REMOTE_EOF
set -e
cd '$REMOTE'
tar -xf /tmp/$(basename "$TAR") -C '$REMOTE'
$RM_CMD
find '$REMOTE' -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -f /tmp/$(basename "$TAR") /tmp/$(basename "$SH")
echo '$SHA' > '$STAMP'
$RESTART_CMD
echo REMOTE_SYNC_OK
REMOTE_EOF

if [ "$DRY_RUN" -eq 1 ]; then
    step "--dry-run：以下是将在服务器执行的脚本，不会真的连接"
    sed 's/^/    /' "$SH"
    exit 0
fi

# ── 6. 上传并执行 ───────────────────────────────────────────────────────────
step "上传到 $SERVER:/tmp/"
scp -q "$TAR" "$SH" "$SERVER:/tmp/" || die "scp 失败"

step "远端解压、同步、$([ "$DO_BUILD" -eq 1 ] && echo '重建镜像' || echo '重启')"
OUT="$(ssh_run "bash /tmp/$(basename "$SH")" 2>&1)" || { printf '%s\n' "$OUT"; die "远端执行失败"; }
printf '%s\n' "$OUT" | grep -q REMOTE_SYNC_OK || { printf '%s\n' "$OUT"; die "远端同步失败"; }

# ── 7. RAG 模型缓存 / 向量库缺失自动补传 ─────────────────────────────────────
# 嵌入模型(text2vec-base-chinese, ~391M) + ChromaDB 被 .gitignore 排除，git archive 永远
# 同步不了。新机上缺了不会报错，只在用户问知识库类问题时才炸（离线加载失败去连 huggingface）。
# 默认**只探测不补传**：目标镜像里 langchain-chroma 是被注释掉的（见 Dockerfile 的说明），
# RAG mode 本就不可用、这份缓存用不上，每次部署自动推 391M 纯属浪费且会拖长部署。
# 真要用 RAG 时加 --with-rag-assets 显式补传。
PROBE="$REMOTE/models/model_cache/models--shibing624--text2vec-base-chinese"
# 用 test -d 而不是 find：find 在父目录不存在时行为随实现而异（曾导致探测结果时对时错），
# test -d 的退出码是明确的；再用 exit code 而非 stdout 判定，避免 ssh 端任何输出污染判断。
if ssh_run "test -d '$PROBE'" 2>/dev/null; then
    step "RAG 嵌入模型缓存已在服务器"
elif [ "$WITH_RAG" -eq 1 ]; then
    LOCAL_MC="$REPO/models/model_cache"
    if [ -d "$LOCAL_MC/models--shibing624--text2vec-base-chinese" ]; then
        warn "补传 models/model_cache + vector_db（~391M，一次性）…"
        ssh_run "mkdir -p '$REMOTE/models' '$REMOTE/vector_db'"
        # scp 目标写父目录、源写目录本身；写成 "src/." 会被新版 OpenSSH（SFTP 后端）拒绝：
        # scp: error: unexpected filename: .
        if scp -q -r "$LOCAL_MC" "$SERVER:$REMOTE/models/"; then
            [ -d "$REPO/vector_db" ] && scp -q -r "$REPO/vector_db" "$SERVER:$REMOTE/" || true
            # 传完必须复验：曾经出现过 scp 失败却照样打 ✓ 的假成功
            if ssh_run "test -d '$PROBE'"; then ok "模型/向量库补传完成并复验通过"
            else warn "补传后复验仍未找到 $PROBE，请手动检查"; fi
        else
            warn "模型补传失败（scp 非零退出），RAG 仍不可用"
        fi
    else
        warn "本地也没有 models/model_cache（未跑过 ingest.py？），无法补传"
    fi
else
    step "服务器无 RAG 模型缓存（该镜像未装 langchain-chroma、RAG 本就不可用）。需要时加 --with-rag-assets"
fi

# ── 8. 健康校验 ─────────────────────────────────────────────────────────────
if [ "$SKIP_HEALTH" -eq 1 ]; then
    ok "已跳过 /health 校验。部署完成：$SHA"
    exit 0
fi

step "等待 /health…"
for i in $(seq 1 12); do
    sleep 3
    HEALTH="$(ssh_run "curl -s -m 5 http://localhost:8088/health" 2>/dev/null || true)"
    if printf '%s' "$HEALTH" | grep -qE '"status"[[:space:]]*:[[:space:]]*"ok"'; then
        printf '    %s\n' "$HEALTH"
        ok "部署完成并健康：$SHA  $SUBJECT"
        echo
        step "别忘了人工验一次（本项目改鉴权时的必查项）："
        echo "    · 在飞书点一次任意卡片按钮 —— 验证入站 token 门禁没把按钮打死"
        echo "    · 点一次卡分布卡的「打开实时页面」 —— 验证 GPU_DIST_TOKEN 链路正常"
        exit 0
    fi
    if [ -n "$HEALTH" ]; then printf '    [%s/12] %.80s…\n' "$i" "$HEALTH"
    else                      printf '    [%s/12] 容器启动中…\n' "$i"; fi
done

warn "部署已推送但 /health 未在 36 秒内返回 ok，请手动检查："
warn "    ssh $SERVER 'docker logs --tail 50 aiops-bot'"
exit 2
