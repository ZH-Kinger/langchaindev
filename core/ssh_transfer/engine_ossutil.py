"""段2 新引擎：泰国服务器 **直连**新加坡 OSS 拉取（取代「SGP rsync 转发」）。

## 为什么换

真机实测（2026-07-31，同一份 19.5 TiB 数据）：

| 路径 | 单流 | 并发 |
|---|---|---|
| SGP → 泰国 rsync（旧） | 27 MB/s | 8 流 78 MB/s |
| 泰国 ossutil 直拉新加坡 OSS（新） | 90 MB/s | **289 MB/s** |

快 10 倍，且完全不占 SGP 出口带宽、少一次中转拷贝。瓶颈原来是「SGP 那一跳」本身，
不是带宽包（SGP 实例出口上限实测 6144 Mb/s = 768 MB/s，远未跑满）。

**段1 保留不动**（杭州 OSS →(CEN)→ 新加坡 wuji-sing）：杭州出境被限速，泰国直拉杭州慢
（泰国→杭州 RTT 83ms vs →新加坡 31ms），所以中转桶这一跳是有价值的，砍掉的只是
「SGP 机器转发数据」这个动作。

## 控制面 vs 数据面

- **控制面**：bot → SGP → 泰国（双跳 SSH）。刻意如此：bot 只有到 SGP 的私钥
  （`SGP_SSH_KEY_ENC`），没有到泰国的；SGP 上早已配好到泰国的免密。走双跳 = **零新增凭证**。
- **数据面**：泰国 ↔ 新加坡 OSS 直连，**完全不经 SGP**。

## 双跳的引号地狱 → 一律 base64

bot 拼串 → SGP shell → 泰国 shell，三层解析。本次开发在这上面踩了两次
（变量被吃掉、`unexpected EOF`）。所以内层脚本**统一 base64 编码后传**：base64 只含
`A-Za-z0-9+/=`，外层一层双引号即可，从根上消灭嵌套转义问题。禁止再手写多层引号。

## marker 协议

与 `engine_ssh` 同构（pid / rc / log），但落在**泰国** `$HOME/.ossutil_jobs/<job_id>/`：
布局刻意与 2026-07-31 人工切换那一单**逐字一致**，部署后可直接认领在跑的任务、不重传。
"""
import base64
import re
import shlex

from config.settings import settings
from utils.logger import get_logger
from core.ssh_transfer import paths
from core.ssh_transfer.engine_ssh import SshTransferError, run as _run_sgp

logger = get_logger("ssh_transfer.ossutil")

# ossutil cp 的 flag 已在泰国 2.3.0 上逐个核实（查 `cp --help`，不猜）：
#   -j/--job 存在（**默认仅 3**，必须显式给）/ --jobs **不存在**（写它会 unknown flag）
#   --parallel / -u,--update / --checkpoint-dir / -e,--endpoint / --region / -f,--force 均存在
# `-u` 语义 = 跳过「已存在**且比源更新**」的文件；mtime 相等**不跳过**（实测会重下）。
# `-f` 必须给：任务是 nohup detached 跑的，任何交互确认都会让它永久挂住。
_FLAG_JOBS = "--job"
_FLAG_UPDATE = "-u"
_FLAG_FORCE = "-f"


def _thai_ssh_prefix() -> str:
    user = settings.THAI_USER or "wuji"
    host = settings.THAI_HOST
    port = int(settings.THAI_PORT or 22)
    if not host:
        raise SshTransferError("未配置 THAI_HOST（泰国服务器）。")
    return (f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            # quote 上：user/host 来自配置而非用户输入，但它们会被拼进一条经 SGP shell
            # 再解析的命令串，`.env` 里一个手滑的空格或分号就是命令注入。零成本的纵深防御。
            f"-o ConnectTimeout=15 {shlex.quote(f'{user}@{host}')}")


def run_thai(script: str, *, timeout: int = 120) -> tuple[int, str, str]:
    """在泰国跑一段 bash（bot→SGP→泰国）。内层 base64 传递，见模块 docstring。"""
    b64 = base64.b64encode(script.encode()).decode()
    inner_timeout = max(30, timeout - 30)
    cmd = f'timeout {inner_timeout} {_thai_ssh_prefix()} "echo {b64} | base64 -d | bash"'
    return _run_sgp(cmd, timeout=timeout)


def _work_dir(job_id: str) -> str:
    """泰国侧工作目录（shell 片段，`$HOME` 由远端展开，不在本地解析）。"""
    root = (settings.THAI_WORK_DIR or "$HOME/.ossutil_jobs").rstrip("/")
    return f'{root}/{job_id}'


def source_uri(source_prefix: str) -> str:
    """段2 的源 = 段1 的落点（新加坡中转桶）。"""
    bucket = (settings.SGP_OSS_BUCKET or "").strip()
    if not bucket:
        raise SshTransferError(
            "未配置 SGP_OSS_BUCKET（段1 落点 / 段2 源的新加坡中转桶，如 wuji-sing）。")
    return f"oss://{bucket}/{source_prefix}"


def _cp_flags() -> str:
    ep = (settings.THAI_OSS_ENDPOINT or "").strip()
    region = (settings.THAI_OSS_REGION or "").strip()
    try:
        jobs = max(1, int(settings.THAI_OSSUTIL_JOBS or 32))
    except (TypeError, ValueError):
        jobs = 32
    try:
        par = max(1, int(settings.THAI_OSSUTIL_PARALLEL or 8))
    except (TypeError, ValueError):
        par = 8
    parts = []
    # 显式给 endpoint/region，不吃泰国 ~/.ossutilconfig 的默认值：那个文件是人工维护的，
    # 有人把它改回杭州/加速域名，任务就会静默变慢或 403（异地 endpoint 会被 OSS 直接拒）。
    if ep:
        parts.append(f"-e {shlex.quote(ep)}")
    if region:
        parts.append(f"--region {shlex.quote(region)}")
    parts += [f"{_FLAG_JOBS} {jobs}", f"--parallel {par}", _FLAG_UPDATE, _FLAG_FORCE]
    return " ".join(parts)


def start_stage2(job_id: str, *, source_prefix: str, dest_rel: str = "") -> None:
    """在泰国起 ossutil 直拉。幂等：已有存活进程则不重复下发。"""
    src = source_uri(source_prefix)
    dest_root = (settings.THAI_DEST_ROOT or "").rstrip("/")
    if not dest_root:
        raise SshTransferError("未配置 THAI_DEST_ROOT（泰国目标根目录）。")
    # 目标目录只有一份实现（paths.dest_dir）：校验层查的必须与这里写的逐字相同，见该函数注释。
    dest = paths.dest_dir(dest_root, source_prefix=source_prefix, dest_rel=dest_rel)
    jd = _work_dir(job_id)
    # ⚠️ 安全边界的真相（别被下面的 shlex.quote 误导）：
    #    这些值被拼进 `WORK="..."` 这个**双引号赋值**里，而 shlex.quote 产生的是单引号包裹 ——
    #    单引号在双引号上下文里只是普通字符，**挡不住 $(...)、反引号、${...}**。实测
    #    `WORK="ossutil cp '/mnt/x/$(touch /tmp/PWNED)/'"` 在赋值那一刻就会执行 touch。
    #    所以 shlex.quote 在这里**不构成纵深防御**，它只在下面那些普通命令位置才真正有效。
    #
    #    这条链路上唯一真实的防线是 `paths` 层的白名单（`_SEG_RE` = [A-Za-z0-9._-]，不含
    #    $ ` { }，且显式拒 `..` 与空段）。目前构造不出可用载荷，靠的全是它。
    #
    #    **禁止放宽 `_SEG_RE`**，也不要因为"反正有 shlex.quote"而放松任何一级校验 ——
    #    这里通往泰国生产机，注入即 RCE。要彻底修，把 src/dest/flags 当位置参数传给内层
    #    `bash -s -- "$@"`，或改用 bash 数组 `"${WORK[@]}"`，别再往双引号字符串里拼。
    script = f'''
set -u
JD="{jd}"
mkdir -p "$JD/ckpt" || exit 1
if [ -f "$JD/stage2.pid" ] && kill -0 "$(cat "$JD/stage2.pid" 2>/dev/null)" 2>/dev/null; then
  echo "ALREADY_RUNNING pid=$(cat "$JD/stage2.pid")"
  exit 0
fi
rm -f "$JD/stage2.rc"
WORK="ossutil cp -r {shlex.quote(src)} {shlex.quote(dest)} {_cp_flags()} --checkpoint-dir \\"$JD/ckpt\\""
{{ nohup bash -c "$WORK; echo \\$? > \\"$JD/stage2.rc\\"" > "$JD/stage2.log" 2>&1 &
  echo $! > "$JD/stage2.pid"; }}
sleep 3
pid=$(cat "$JD/stage2.pid" 2>/dev/null || echo 0)
kill -0 "$pid" 2>/dev/null && echo "LAUNCHED pid=$pid" || echo "LAUNCH_DEAD pid=$pid"
'''
    rc, out, err = run_thai(script, timeout=90)
    if rc != 0:
        raise SshTransferError(f"泰国起 ossutil 失败(rc={rc})：{(err or out)[:300]}")
    if "LAUNCH_DEAD" in (out or ""):
        raise SshTransferError(f"泰国 ossutil 起来即退出：{(out or '')[:300]}")
    if "ALREADY_RUNNING" not in (out or "") and "LAUNCHED" not in (out or ""):
        raise SshTransferError(f"泰国下发结果无法确认（既无 LAUNCHED 也无 ALREADY_RUNNING）：{(out or '')[:300]}")
    logger.info("[SSHT-OSSUTIL] %s 段2 已下发 %s", job_id, (out or "").strip()[-80:])


def poll_stage(job_id: str) -> dict:
    """查段2 状态：{status: RUNNING|DONE|FAILED, rc, alive, error}。只读 marker，不读长日志。"""
    jd = _work_dir(job_id)
    script = f'''
set -u
JD="{jd}"
pid=$(cat "$JD/stage2.pid" 2>/dev/null || echo 0)
echo "PID=$pid"
kill -0 "$pid" 2>/dev/null && echo "ALIVE=1" || echo "ALIVE=0"
echo "RC=$(cat "$JD/stage2.rc" 2>/dev/null || echo NONE)"
[ -f "$JD/stage2.log" ] && echo "LOG=1" || echo "LOG=0"
'''
    rc, out, err = run_thai(script, timeout=90)
    if rc != 0:
        # SSH 不通不代表任务失败 —— 保持在途，交给下一轮。误判失败会触发重试、白跑几十小时。
        raise SshTransferError(f"查泰国段2 状态失败(rc={rc})：{(err or out)[:200]}")
    txt = out or ""
    alive = "ALIVE=1" in txt
    m = re.search(r"RC=(\S+)", txt)
    raw = m.group(1) if m else "NONE"
    if raw != "NONE" and raw.lstrip("-").isdigit():
        code = int(raw)
        if code == 0:
            return {"status": "DONE", "rc": 0, "alive": alive, "error": ""}
        return {"status": "FAILED", "rc": code, "alive": alive,
                "error": f"段2(泰国 ossutil) 退出码 {code}"}
    if alive:
        return {"status": "RUNNING", "rc": None, "alive": True, "error": ""}
    if "LOG=0" in txt:
        return {"status": "FAILED", "rc": None, "alive": False,
                "error": "段2 未产生日志（下发未生效或工作目录被清）"}
    return {"status": "FAILED", "rc": None, "alive": False,
            "error": "段2 进程已退出但未写退出码（被 kill / OOM / 机器重启）"}


# ossutil 进度行（`\r` 刷屏）：`... Copy... done:(553 files,40.882 GiB) skipped:(6 files,0 B), 0.210%, avg 289.4 MiB/s`
_DONE_RE = re.compile(r"done:\((\d+)\s*files?,\s*([\d.]+)\s*([KMGTP]?)i?B\)", re.I)
_SKIP_RE = re.compile(r"skipped:\((\d+)\s*files?,\s*([\d.]+)\s*([KMGTP]?)i?B\)", re.I)
_AVG_RE = re.compile(r"avg\s+([\d.]+)\s*([KMGTP]?)i?B/s", re.I)
_MULT = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4, "p": 1024 ** 5}


def _to_bytes(num: str, unit: str) -> int:
    return int(float(num) * _MULT.get(unit.lower(), 1))


def stage_progress(job_id: str) -> dict:
    """从 ossutil 日志尾部取进度。返回 {bytes_done, pct, speed_bps}，取不到的为 None。

    必须 `tr '\\r' '\\n'`：ossutil 用 `\\r` 刷屏，单次日志可达十几 MB，直接 `tail -n`
    只会拿到一整行进度条（这条教训在段1 已经踩过一次）。
    """
    jd = _work_dir(job_id)
    script = f'''
set -u
L="{jd}/stage2.log"
sz=$(wc -c < "$L" 2>/dev/null || echo 0)
t=$(tail -c 4000 "$L" 2>/dev/null | tr '\\r' '\\n')
if [ "${{sz:-0}}" -gt 4000 ]; then t=$(printf '%s\\n' "$t" | tail -n +2); fi
printf '%s\\n' "$t" | grep -a 'done:(' | tail -1
'''
    try:
        rc, out, _ = run_thai(script, timeout=75)
    except Exception:
        logger.warning("[SSHT-OSSUTIL] %s 取进度失败", job_id, exc_info=True)
        return {"bytes_done": None, "pct": None, "speed_bps": None}
    line = (out or "").strip()
    bd = spd = None
    m = _DONE_RE.search(line)
    if m:
        bd = _to_bytes(m.group(2), m.group(3))
        # 续跑时 `-u` 跳过的部分记在 `skipped:(N files,X)` 里，只算 done 会把「已就位的量」
        # 漏掉 → 进度偏低、ETA 偏长（重跑一个已完成 90% 的任务会显示成刚开始）。
        ms = _SKIP_RE.search(line)
        if ms:
            bd += _to_bytes(ms.group(2), ms.group(3))
    m = _AVG_RE.search(line)
    if m:
        spd = _to_bytes(m.group(1), m.group(2))
    # **刻意不返回 pct**：ossutil 在 Scanning 阶段的百分比是按「已扫到的量」算的分母，
    # 会虚高（真机见过 19.5TiB 的任务在扫到 16TiB 时报出偏高的百分比）。而 progress_line
    # 优先采用 job["pct"]，透出去就会把一个假百分比钉在卡片上 —— 上一轮 auditor 报的
    # 「陈旧/误导百分比」正是这一类。留 None，让上层用 bytes_done/bytes_total 算真值
    # （bytes_total 来自段1 的 ossutil du，是确定的总量）。
    return {"bytes_done": bd, "pct": None, "speed_bps": spd}


_FAIL_GREP = "Error|error|denied|Denied|refused|NoSuchBucket|InvalidAccessKeyId|SignatureDoesNotMatch|no such host|timeout"
_DETAIL_MAX = 1200
_DETAIL_LINE_MAX = 240


def failure_detail(job_id: str) -> str:
    """终态时摘一次真原因。只给「退出码 N」排障要人工翻十几 MB 日志（段1 踩过）。"""
    jd = _work_dir(job_id)
    script = f'''
set -u
L="{jd}/stage2.log"
sz=$(wc -c < "$L" 2>/dev/null || echo 0)
t=$(tail -c 20000 "$L" 2>/dev/null | tr '\\r' '\\n')
if [ "${{sz:-0}}" -gt 20000 ]; then t=$(printf '%s\\n' "$t" | tail -n +2); fi
f=$(printf '%s\\n' "$t" | grep -aiE {shlex.quote(_FAIL_GREP)} | tail -4)
if [ -n "$f" ]; then printf '%s\\n' "$f"
else printf '%s\\n' "$t" | grep -av '^[[:space:]]*$' | tail -3; fi
'''
    try:
        _, out, _ = run_thai(script, timeout=75)
    except Exception:
        logger.warning("[SSHT-OSSUTIL] %s 取失败明细失败", job_id, exc_info=True)
        return ""
    lines = []
    budget = _DETAIL_MAX
    for ln in (out or "").splitlines():
        ln = ln.strip()[:_DETAIL_LINE_MAX]
        if not ln or len(ln) + 1 > budget:
            continue
        lines.append(ln)
        budget -= len(ln) + 1
    return "\n".join(lines)


def cancel(job_id: str) -> bool:
    """停掉泰国侧在跑的 ossutil（checkpoint 保留，下次可续）。

    目前**无调用方**（没有取消按钮）。三处刻意的严格，是为了别在接按钮那天才发现：
    - pid 文件缺失/写坏时返回 **False**，不能报「已取消」——那等于什么都没杀却告诉人杀了。
    - 杀之前核对 `/proc/<pid>/cmdline` 含 ossutil：工作目录在 `$HOME` 下长期存在，
      机器重启后 pid 会被复用，盲杀���打死无关进程（这台是生产机）。
    - 杀完写 `stage2.rc=130`：不写的话「人工取消」和「进程异��死亡」在 poll 里完全同形。
    """
    jd = _work_dir(job_id)
    script = f'''
set -u
JD="{jd}"
pid=$(cat "$JD/stage2.pid" 2>/dev/null || echo "")
case "$pid" in ''|*[!0-9]*) echo BAD_PID; exit 0 ;; esac
[ "$pid" -gt 0 ] || {{ echo BAD_PID; exit 0; }}
if ! kill -0 "$pid" 2>/dev/null; then echo NOT_RUNNING; exit 0; fi
if ! tr '\\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | grep -q ossutil; then
  echo PID_REUSED; exit 0
fi
pkill -P "$pid" 2>/dev/null || true
kill "$pid" 2>/dev/null || true
sleep 3
kill -9 "$pid" 2>/dev/null || true
if kill -0 "$pid" 2>/dev/null; then echo STILL_ALIVE; else echo 130 > "$JD/stage2.rc"; echo KILLED; fi
'''
    try:
        _, out, _ = run_thai(script, timeout=75)
    except Exception:
        logger.warning("[SSHT-OSSUTIL] %s 取消失败（SSH 不可用）", job_id, exc_info=True)
        return False
    out = out or ""
    if "KILLED" in out or "NOT_RUNNING" in out:
        return True
    logger.warning("[SSHT-OSSUTIL] %s 取消未成功：%s", job_id, out.strip()[:120])
    return False
