<#
.SYNOPSIS
    一键部署 AIOps Feishu Bot 到 bot-new（代码已 bind-mount，免 rebuild）。

.DESCRIPTION
    流程：git archive 当前 HEAD → scp 到服务器 → 解压覆盖 → 清理已删文件和 __pycache__
          → docker compose restart bot → 轮询 /health 校验。

    服务器 /root/langchaindev 不是 git 仓库，代码经 docker-compose 的 .:/app 挂载直接生效，
    所以部署 = 同步文件 + 重启进程（约 10 秒），无需 docker build。
    只有 requirements.txt 变更时才需手动 `docker compose up -d --build`（脚本会检测并提示）。

    删除处理：服务器上记录 .deployed_commit，每次部署对比上次提交，把仓库里已删除的
    文件在服务器同步删掉（弥补 tar 覆盖不删文件的缺陷，等效 rsync --delete）。

.PARAMETER Ref
    要部署的 git ref，默认当前 HEAD。例：-Ref main

.PARAMETER SkipHealth
    跳过部署后的 /health 校验。

.EXAMPLE
    .\deploy.ps1
    .\deploy.ps1 -Ref main
#>
param(
    [string]$Ref = "HEAD",
    [switch]$SkipHealth
)

$ErrorActionPreference = "Stop"
# 迁移后 bot-new(8.222.149.27) 为唯一在线机；旧机 bot-server(115.191.2.86) 已停 bot 作回滚。
$SERVER   = "bot-new"
$REMOTE   = "/root/langchaindev"
$REPO     = "D:\code_python\langchaindev"
$STAMP    = "$REMOTE/.deployed_commit"

# UTF-8 无 BOM，避免中文路径/文件名经 ssh 传输乱码
$OutputEncoding = New-Object System.Text.UTF8Encoding $false

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!!  $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "✗   $msg" -ForegroundColor Red; exit 1 }

Set-Location $REPO

# ── 1. 解析提交、检查工作区 ──────────────────────────────────────────────────
$sha = (git rev-parse --short $Ref).Trim()
if (-not $sha) { Die "无法解析 ref: $Ref" }
$subject = (git log -1 --format="%s" $sha).Trim()
Step "部署 $sha  $subject"

$dirty = git status --porcelain
if ($dirty) {
    Warn "工作区有未提交改动，本次只部署已提交的 $sha（未提交内容不会上线）："
    git status --short | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
}

# ── 2. 算出相对上次部署被删除的文件（等效 rsync --delete）──────────────────
# "$(...)" 把 ssh 无输出时的 $null 归一成空串，再 Trim——否则首次部署到无 .deployed_commit 的新机会 NPE
$lastDeployed = "$(ssh $SERVER "cat $STAMP 2>/dev/null")".Trim()
$deleted = @()
if ($lastDeployed) {
    Step "上次部署：$lastDeployed，计算删除文件…"
    $deleted = git diff --name-only --diff-filter=D "$lastDeployed..$sha" 2>$null |
        Where-Object { $_ }
    if ($deleted) { Write-Host "    将删除 $($deleted.Count) 个文件" }
} else {
    Warn "服务器无部署记录（首次用本脚本），跳过删除计算"
}

# requirements.txt 变更 → 提示需要 rebuild
if ($lastDeployed) {
    $reqChanged = git diff --name-only "$lastDeployed..$sha" 2>$null |
        Where-Object { $_ -eq "requirements.txt" }
    if ($reqChanged) {
        Warn "requirements.txt 有变更：restart 不会重装依赖！"
        Warn "如新增/升级了依赖，部署后需手动：ssh $SERVER 'cd $REMOTE && docker compose up -d --build'"
    }
}

# ── 3. 打包并上传（git archive 只含已跟踪文件，天然排除 .git/pycache/数据目录）─
$tar = Join-Path $env:TEMP "feishu-deploy.tar"
Step "打包 → $tar"
git archive --format=tar -o $tar $sha
if (-not (Test-Path $tar)) { Die "git archive 失败" }

# ── 4. 远端脚本：解压 + 删文件 + 清 pycache + 记录提交 + 重启（一次执行）──────
#     写成本地 .sh（强制 LF、无 BOM）再 scp 执行：避开 stdin 管道 / argv 压平 / BOM 注入
$rmCmd = ""
if ($deleted) {
    $files = ($deleted | ForEach-Object { "'$REMOTE/$_'" }) -join " "
    $rmCmd = "rm -f $files"
}
$remoteScript = @"
set -e
cd "$REMOTE"
tar -xf /tmp/feishu-deploy.tar -C "$REMOTE"
$rmCmd
find "$REMOTE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -f /tmp/feishu-deploy.tar /tmp/feishu-deploy.sh
echo "$sha" > "$STAMP"
docker compose -f "$REMOTE/docker-compose.yml" restart bot
echo REMOTE_SYNC_OK
"@
$remoteScript = $remoteScript -replace "`r`n", "`n"
$sh = Join-Path $env:TEMP "feishu-deploy.sh"
[IO.File]::WriteAllText($sh, $remoteScript, (New-Object System.Text.UTF8Encoding $false))

Step "上传到 ${SERVER}:$REMOTE"
scp $tar $sh "${SERVER}:/tmp/"
if ($LASTEXITCODE -ne 0) { Die "scp 失败" }

Step "远端解压、同步、重启"
$out = ssh $SERVER "bash /tmp/feishu-deploy.sh"
if ($out -notmatch "REMOTE_SYNC_OK") { Die "远端同步失败：$out" }

# ── 6. 健康校验（轮询，Agent 预热 + 调度器启动需要几秒）─────────────────────
if ($SkipHealth) {
    Step "已跳过 /health 校验"
    Write-Host "✓ 部署完成：$sha" -ForegroundColor Green
    exit 0
}

Step "等待 /health…"
$ok = $false
foreach ($i in 1..12) {
    Start-Sleep -Seconds 3
    $health = ssh $SERVER "curl -s -m 5 http://localhost:8088/health" 2>$null
    if ($health -match '"status"\s*:\s*"ok"') {
        Write-Host "    $health"
        $ok = $true
        break
    }
    if ($health) { Write-Host "    [$i/12] $($health.Substring(0, [Math]::Min(80, $health.Length)))…" }
    else         { Write-Host "    [$i/12] 容器启动中…" }
}

if ($ok) {
    Write-Host "✓ 部署完成并健康：$sha  $subject" -ForegroundColor Green
} else {
    Warn "部署已推送但 /health 未在 36 秒内返回 ok，请手动检查："
    Warn "  ssh $SERVER 'docker logs --tail 50 aiops-bot'"
    exit 2
}
