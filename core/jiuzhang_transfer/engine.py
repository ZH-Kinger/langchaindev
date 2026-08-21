"""九章拉取引擎：bot --SSH--> 九章，在九章上跑 ossutil 从杭州 OSS 直拉。

执行模型与 core.ssh_transfer 同构（marker 模式，见那边的说明）：
    起任务 = 一条**短** SSH 命令后台化，pid/rc/log 落在九章本地
    轮询   = 只读 marker，**从不通过 SSH 读长输出**（paramiko 读大输出会死锁；
             ossutil 用 \\r 刷屏，单次日志可达十几 MB）

与泰国链的两处实质差异：
  · 单跳，所以脚本不必 base64 传递（那是为了穿三层 shell 解析）。仍然只在
    普通命令位置用 shlex.quote —— 真正的防线是 paths 层白名单。
  · 目的盘 GPFS 支持并发写，**不加**泰国那套单分片 flag（那是 ossfs2/FUSE 的补丁）。
"""
import base64
import io
import re
import shlex

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("jiuzhang_transfer.engine")

STAGE_PULL = "pull"


class JiuzhangError(RuntimeError):
    """九章迁移调用失败，消息面向用户。"""


# ── 连接层 ────────────────────────────────────────────────────────────────────

def _private_key():
    """Fernet 密文 → paramiko key（**只在内存**）。默认复用 SGP 那把（同一把 key 已在九章
    的 authorized_keys2 里），配 JIUZHANG_SSH_KEY_ENC 可单独换。"""
    import paramiko
    from utils.crypto import decrypt
    enc = (getattr(settings, "JIUZHANG_SSH_KEY_ENC", "") or settings.SGP_SSH_KEY_ENC or "").strip()
    if not enc:
        raise JiuzhangError("未配置 JIUZHANG_SSH_KEY_ENC / SGP_SSH_KEY_ENC（bot→九章 私钥）。")
    try:
        pem = decrypt(enc)
    except Exception as e:
        raise JiuzhangError(f"九章私钥解密失败：{e}")
    if not pem or "-----BEGIN" not in pem:
        raise JiuzhangError("九章私钥解密结果不是 PEM（检查是否用 BOT_CREDS_ENCRYPTION_KEY 加密过）。")
    for cls in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey):
        try:
            return cls.from_private_key(io.StringIO(pem))
        except Exception:
            continue
    raise JiuzhangError("九章私钥格式无法解析（试过 RSA/Ed25519/ECDSA）。")


def _client():
    """连九章。host key 固定 + RejectPolicy —— **禁 AutoAdd，fail-closed**，同 SGP 那条链。"""
    import base64
    import paramiko
    host = (getattr(settings, "JIUZHANG_HOST", "") or "").strip()
    port = int(getattr(settings, "JIUZHANG_PORT", 0) or 22)
    if not host:
        raise JiuzhangError("未配置 JIUZHANG_HOST。")
    hk = (getattr(settings, "JIUZHANG_HOST_KEY", "") or "").strip()
    if not hk:
        raise JiuzhangError(
            "未配置 JIUZHANG_HOST_KEY，拒绝连接以防中间人。"
            f"用 `ssh-keyscan -p {port} {host}` 取，填任意一行。")
    parts = hk.split()
    if len(parts) < 2:
        raise JiuzhangError("JIUZHANG_HOST_KEY 格式非法，应形如 `ssh-ed25519 AAAA...`。")
    if parts[0] not in ("ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256"):
        parts = parts[1:]                       # 容错 ssh-keyscan 输出的「主机名 类型 值」三列
    ctor = {"ssh-ed25519": paramiko.Ed25519Key, "ssh-rsa": paramiko.RSAKey,
            "ecdsa-sha2-nistp256": paramiko.ECDSAKey}.get(parts[0])
    if ctor is None:
        raise JiuzhangError(f"不支持的 host key 类型 `{parts[0]}`。")
    c = paramiko.SSHClient()
    name = f"[{host}]:{port}" if port != 22 else host
    c.get_host_keys().add(name, parts[0], ctor(data=base64.b64decode(parts[1])))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        c.connect(hostname=host, port=port,
                  username=getattr(settings, "JIUZHANG_USER", "root") or "root",
                  pkey=_private_key(), timeout=20, banner_timeout=20, auth_timeout=20,
                  allow_agent=False, look_for_keys=False)
    except Exception as e:
        raise JiuzhangError(f"连接九章({host}:{port}) 失败：{e}")
    return c


def run(cmd: str, *, timeout: int = 30) -> tuple:
    """在九章上跑一条**短**命令。drain 完 stdout/stderr 再取 exit code（避死锁）。"""
    c = _client()
    try:
        _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err
    finally:
        try:
            c.close()
        except Exception:
            pass


# ── 工作目录 / marker ────────────────────────────────────────────────────────

def _work_dir(job_id: str) -> str:
    root = (getattr(settings, "JIUZHANG_WORK_DIR", "") or "$HOME/.jiuzhang_jobs").rstrip("/")
    return f"{root}/{job_id}"


def dest_dir(source_prefix: str, dest_rel: str = "") -> str:
    """目的目录 —— **与校验层共用同一个实现**（paths.dest_dir）。

    为什么必须只有一份：传输器写哪、校验查哪靠它算出同一个串；漂移即「校验了个空目录」，
    而空目录表现为「目的端缺 N 万个」，运维的合理反应是重传。
    """
    from core.ssh_transfer import paths
    root = (getattr(settings, "JIUZHANG_DEST_ROOT", "") or "/root/nas").rstrip("/")
    return paths.dest_dir(root, source_prefix=source_prefix, dest_rel=dest_rel)


def _cp_flags() -> str:
    """ossutil flags。九章目的盘是 GPFS（并发写没问题），**不加**泰国那套单分片降速 flag。

    flag 名已在九章的 ossutil 2.3.0 上核实：`-j/--job`（单数，默认仅 3 必须显式给）、
    `--parallel`、`-u`、`-f`（detached 跑必须给，否则交互提示会永久挂住）、`-e`/`--region`。
    """
    def _int(name, default):
        try:
            return max(1, int(getattr(settings, name, "") or default))
        except (TypeError, ValueError):
            return default
    ep = (getattr(settings, "JIUZHANG_OSS_ENDPOINT", "") or "").strip()
    region = (getattr(settings, "JIUZHANG_OSS_REGION", "") or "").strip()
    parts = []
    # 显式给 endpoint/region，**不吃九章 ~/.ossutilconfig 的默认值** —— 那文件人工维护，
    # 被改成别的地域会静默变慢，异地 endpoint 更会被 OSS 直接 403。
    if ep:
        parts.append(f"-e {shlex.quote(ep)}")
    if region:
        parts.append(f"--region {shlex.quote(region)}")
    parts += [f"--job {_int('JIUZHANG_OSSUTIL_JOBS', 32)}",
              f"--parallel {_int('JIUZHANG_OSSUTIL_PARALLEL', 8)}", "-u", "-f"]
    return " ".join(parts)


def start_pull(job_id: str, *, source_bucket: str, source_prefix: str, dest_rel: str = "") -> None:
    """在九章起 ossutil 直拉（后台化）。幂等：已有存活进程则不重复下发。

    内层命令走 **base64 传递**，不手写嵌套引号 —— 代码库对此有明令（泰国链在双跳上
    栽过两次：变量被吃掉、unexpected EOF）。这里虽是单跳，但 `nohup bash -c '...'` 里
    还要再套一层 `echo $? > rc` 的重定向，照样是多层解析，同一个坑。
    """
    src = f"oss://{source_bucket}/{source_prefix}"
    dst = dest_dir(source_prefix, dest_rel)
    jd = _work_dir(job_id)
    inner = (f"mkdir -p {shlex.quote(dst)} && "
             f"ossutil cp -r {shlex.quote(src)} {shlex.quote(dst)} {_cp_flags()} "
             # ⚠️ 不能用 shlex.quote 包 —— 工作目录含 `$HOME`，单引号会让它**不被展开**，
             # 结果建出一个字面量叫 `$HOME` 的目录、断点续传永远命中不了。
             # 改用外层 export 的 $JD，由内层 shell 展开。
             f'--checkpoint-dir "$JD/ckpt"')
    b64 = base64.b64encode(inner.encode("utf-8")).decode("ascii")
    script = f'''
set -u
JD="{jd}"
mkdir -p "$JD/ckpt" || exit 1
if [ -f "$JD/{STAGE_PULL}.pid" ] && kill -0 "$(cat "$JD/{STAGE_PULL}.pid" 2>/dev/null)" 2>/dev/null; then
  echo ALREADY_RUNNING; exit 0
fi
rm -f "$JD/{STAGE_PULL}.rc"
export JD                       # 内层 bash -c 是子进程，不 export 拿不到 $JD
WORK=$(echo {b64} | base64 -d)
{{ nohup bash -c "$WORK; echo \\$? > \\"$JD/{STAGE_PULL}.rc\\"" > "$JD/{STAGE_PULL}.log" 2>&1 &
  echo $! > "$JD/{STAGE_PULL}.pid"; }}
sleep 3
# ⚠️ 先看 rc 再看进程活没活。小任务可能**比这个 sleep 还快跑完**（实测 66 MiB / 11 对象
# 只用了 0.9 秒），那时进程已正常退出、`kill -0` 必然失败 —— 只按存活判定会把
# 「已经成功」误报成「起来即死」。而小任务正是大家用来验证链路的那种。
if [ -f "$JD/{STAGE_PULL}.rc" ]; then
  echo "ALREADY_DONE rc=$(cat "$JD/{STAGE_PULL}.rc")"
elif kill -0 "$(cat "$JD/{STAGE_PULL}.pid" 2>/dev/null)" 2>/dev/null; then
  echo LAUNCHED
else
  echo LAUNCH_DEAD
fi
'''
    rc, out, err = run(script, timeout=90)
    if rc != 0:
        raise JiuzhangError(f"九章起拉取失败(rc={rc})：{(err or out)[:300]}")
    out = out or ""
    if "LAUNCH_DEAD" in out:
        raise JiuzhangError(f"九章 ossutil 起来即退出：{out[:300]}")
    if not any(k in out for k in ("LAUNCHED", "ALREADY_RUNNING", "ALREADY_DONE")):
        # 既无成功也无失败标记 = 结果不可知。当失败处理 ——「不知道起没起」比「以为起了」好排查。
        raise JiuzhangError(f"九章下发结果无法确认：{out[:300]}")
    if "ALREADY_DONE" in out:
        # 跑得比下发校验还快。不当失败 —— 让 poll() 去读 rc 定成败，那是唯一的权威来源。
        logger.info("[JZ] %s 下发即完成（小任务）%s", job_id, out.strip()[-40:])
    else:
        logger.info("[JZ] %s 已下发 %s → %s", job_id, src, dst)


def poll(job_id: str) -> dict:
    """查状态：{status: RUNNING|DONE|FAILED, rc, error}。只读 marker。"""
    jd = _work_dir(job_id)
    script = f'''
set -u
JD="{jd}"
pid=$(cat "$JD/{STAGE_PULL}.pid" 2>/dev/null || echo 0)
kill -0 "$pid" 2>/dev/null && echo ALIVE=1 || echo ALIVE=0
echo "RC=$(cat "$JD/{STAGE_PULL}.rc" 2>/dev/null || echo NONE)"
[ -f "$JD/{STAGE_PULL}.log" ] && echo LOG=1 || echo LOG=0
'''
    rc, out, err = run(script, timeout=60)
    if rc != 0:
        # SSH 不通 ≠ 任务失败。抛出去让上层保持在途 —— 误判失败会触发重传。
        raise JiuzhangError(f"查九章状态失败(rc={rc})：{(err or out)[:200]}")
    txt = out or ""
    alive = "ALIVE=1" in txt
    m = re.search(r"RC=(\S+)", txt)
    raw = m.group(1) if m else "NONE"
    if raw != "NONE" and raw.lstrip("-").isdigit():
        code = int(raw)
        # ossutil：0 成功；24 = 源文件传输中消失，非致命（与泰国链一致）
        ok = code in (0, 24)
        return {"status": "DONE" if ok else "FAILED", "rc": code,
                "error": "" if ok else f"拉取退出码 {code}"}
    if alive:
        return {"status": "RUNNING", "rc": None, "error": ""}
    if "LOG=0" in txt:
        return {"status": "FAILED", "rc": None, "error": "未产生日志（下发未生效或工作目录被清）"}
    return {"status": "FAILED", "rc": None, "error": "进程已退出但没写退出码（被 kill / OOM / 机器重启）"}


# ── 进度 / 失败明细 ──────────────────────────────────────────────────────────
# ossutil 进度行：`... done:(553 files,40.882 GiB) skipped:(6 files,0 B), 0.21%, avg 289.4 MiB/s`
_DONE_RE = re.compile(r"done:\((\d+)\s*files?,\s*([\d.]+)\s*([KMGTP]?)i?B\)", re.I)
_SKIP_RE = re.compile(r"skipped:\((\d+)\s*files?,\s*([\d.]+)\s*([KMGTP]?)i?B\)", re.I)
_AVG_RE = re.compile(r"avg\s+([\d.]+)\s*([KMGTP]?)i?B/s", re.I)
_MULT = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4, "p": 1024 ** 5}


def _to_bytes(num: str, unit: str) -> int:
    return int(float(num) * _MULT.get(unit.lower(), 1))


def progress(job_id: str) -> dict:
    """从日志尾部取进度。**必须 `tr '\\r' '\\n'`** —— ossutil 用 \\r 刷屏，直接 tail -n
    只会拿到一整行进度条（泰国链踩过）。

    **刻意不返回 pct**：ossutil 在 Scanning 阶段的百分比分母是「已扫到的量」、会虚高。
    让上层用 bytes_done/bytes_total 算真值。
    """
    jd = _work_dir(job_id)
    script = f'''
set -u
L="{jd}/{STAGE_PULL}.log"
sz=$(wc -c < "$L" 2>/dev/null || echo 0)
t=$(tail -c 4000 "$L" 2>/dev/null | tr '\\r' '\\n')
if [ "${{sz:-0}}" -gt 4000 ]; then t=$(printf '%s\\n' "$t" | tail -n +2); fi
printf '%s\\n' "$t" | grep -a 'done:(' | tail -1
'''
    try:
        _, out, _ = run(script, timeout=60)
    except Exception:
        logger.warning("[JZ] %s 取进度失败", job_id, exc_info=True)
        return {"bytes_done": None, "objects_done": None, "speed_bps": None}
    line = (out or "").strip()
    bd = objs = spd = None
    m = _DONE_RE.search(line)
    if m:
        objs, bd = int(m.group(1)), _to_bytes(m.group(2), m.group(3))
        ms = _SKIP_RE.search(line)
        if ms:      # 续跑时 -u 跳过的量记在 skipped 里，不算进去会让进度偏低、ETA 偏长
            objs += int(ms.group(1))
            bd += _to_bytes(ms.group(2), ms.group(3))
    m = _AVG_RE.search(line)
    if m:
        spd = _to_bytes(m.group(1), m.group(2))
    return {"bytes_done": bd, "objects_done": objs, "speed_bps": spd}


_FAIL_GREP = ("Error|error|denied|Denied|refused|NoSuchBucket|InvalidAccessKeyId|"
              "SignatureDoesNotMatch|no such host|timeout|FinishWithError")


def failure_detail(job_id: str) -> str:
    """失败时摘一次真原因。只给「退出码 N」的话排障要人工翻十几 MB 日志。"""
    jd = _work_dir(job_id)
    script = f'''
set -u
L="{jd}/{STAGE_PULL}.log"
sz=$(wc -c < "$L" 2>/dev/null || echo 0)
t=$(tail -c 20000 "$L" 2>/dev/null | tr '\\r' '\\n')
if [ "${{sz:-0}}" -gt 20000 ]; then t=$(printf '%s\\n' "$t" | tail -n +2); fi
f=$(printf '%s\\n' "$t" | grep -aiE {shlex.quote(_FAIL_GREP)} | tail -4)
if [ -n "$f" ]; then printf '%s\\n' "$f"
else printf '%s\\n' "$t" | grep -av '^[[:space:]]*$' | tail -3; fi
'''
    try:
        _, out, _ = run(script, timeout=60)
    except Exception:
        return ""
    lines, budget = [], 1200
    for ln in (out or "").splitlines():
        ln = ln.strip()[:240]           # 逐行截断：先保结论，别让几条长汇总行把根因挤没
        if ln and len(ln) + 1 <= budget:
            lines.append(ln)
            budget -= len(ln) + 1
    return "\n".join(lines)


# 正则锚定到具体那一行。松散写法会先命中表头、再跨行吞进 object count，
# 把 22MB 读成 3B —— 那会直接绕过审批门（泰国链踩过）。
_DU_SIZE_RE = re.compile(r"total\s+(?:object\s+sum\s+size|du\s+size)\s*[:：]?\s*(\d+)", re.I)
_DU_COUNT_RE = re.compile(r"total\s+object\s+count\s*[:：]?\s*(\d+)", re.I)


def estimate_source(source_bucket: str, source_prefix: str) -> tuple:
    """在九章上跑 ossutil du 估源大小，返回 (字节, 对象数, ok)。

    `ok=False` 时上层 **fail-safe 当作需审批** —— 不放行未知大小的迁移。
    """
    ep = (getattr(settings, "JIUZHANG_OSS_ENDPOINT", "") or "").strip()
    rg = (getattr(settings, "JIUZHANG_OSS_REGION", "") or "").strip()
    src = f"oss://{source_bucket}/{source_prefix}"
    cmd = (f"ossutil du {shlex.quote(src)}"
           + (f" -e {shlex.quote(ep)}" if ep else "")
           + (f" --region {shlex.quote(rg)}" if rg else "") + " 2>&1 | tail -30")
    try:
        _, out, _ = run(cmd, timeout=int(getattr(settings, "JIUZHANG_ESTIMATE_TIMEOUT", 600) or 600))
    except Exception:
        logger.warning("[JZ] 估算 %s 失败", src, exc_info=True)
        return 0, 0, False
    blob = out or ""
    m = _DU_SIZE_RE.search(blob)
    if not m:
        return 0, 0, False
    mc = _DU_COUNT_RE.search(blob)
    return int(m.group(1)), int(mc.group(1)) if mc else 0, True
