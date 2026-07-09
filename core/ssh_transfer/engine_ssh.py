"""SSH 迁移引擎：paramiko 连到新加坡 ECS，遥控段1(ossutil)/段2(rsync)。

执行模式（避 paramiko 大输出死锁 + 容器重启丢 channel）：
    起任务 = SSH 一条短命令：`nohup bash -c '<work>; echo $? > rc' > log 2>&1 & echo $! > pid`
    轮询   = SSH 短命令 `cat rc` / `kill -0 pid`，不读长输出。
每个 job 一个工作目录 `{SGP_WORK_DIR}/{job_id}/`，内含 `<stage>.pid/.rc/.log`。

私钥：Fernet 密文 `SGP_SSH_KEY_ENC` → 运行时 decrypt() → RSAKey.from_private_key(内存)，绝不落盘。
host key：固定 `SGP_SSH_HOST_KEY`（禁 AutoAddPolicy，fail-closed）。
"""
import io
import logging
import re
import shlex

from config.settings import settings

logger = logging.getLogger(__name__)

STAGE1 = "stage1"   # ossutil: 杭州 OSS → SGP 本地挂载盘
STAGE2 = "stage2"   # rsync: SGP 挂载盘 → 泰国服务器

# rsync 成功退出码：0=完全成功，24=源文件传输中消失（非致命）。其余非 0 视失败。
_RSYNC_OK = (0, 24)


class SshTransferError(RuntimeError):
    """SSH 迁移调用失败，消息面向用户。"""


# ── 连接层 ────────────────────────────────────────────────────────────────────

def _load_private_key():
    """Fernet 解密 SGP_SSH_KEY_ENC → paramiko RSAKey（只在内存，不落盘/打印）。"""
    import paramiko
    from utils.crypto import decrypt
    enc = settings.SGP_SSH_KEY_ENC
    if not enc:
        raise SshTransferError("未配置 SGP_SSH_KEY_ENC（bot→SGP 私钥，Fernet 密文）。")
    try:
        pem = decrypt(enc)
    except Exception as e:
        raise SshTransferError(f"SGP 私钥解密失败：{e}")
    # decrypt() 对非 Fernet 密文会明文透传、解密失败可能返回空 → 明确校验是 PEM，避免误报"格式非法"
    if not pem or "-----BEGIN" not in pem:
        raise SshTransferError(
            "SGP 私钥解密结果不是 PEM（检查 SGP_SSH_KEY_ENC 是否为用 BOT_CREDS_ENCRYPTION_KEY "
            "加密后的私钥密文；勿直接填明文私钥）。")
    try:
        return paramiko.RSAKey.from_private_key(io.StringIO(pem))
    except Exception as e:
        raise SshTransferError(f"SGP 私钥格式非法（应为 PEM RSA）：{e}")


def _add_host_key(client):
    """固定 host key：禁 AutoAddPolicy，未配置则 fail-closed 报错。"""
    import base64
    import paramiko
    hk = (settings.SGP_SSH_HOST_KEY or "").strip()
    if not hk:
        raise SshTransferError(
            "未配置 SGP_SSH_HOST_KEY（SGP 主机公钥指纹），拒绝连接以防中间人。"
            "请填 ssh-keyscan 得到的那行，如 `ssh-ed25519 AAAA...`。")
    parts = hk.split()
    if len(parts) < 2:
        raise SshTransferError("SGP_SSH_HOST_KEY 格式非法，应形如 `ssh-ed25519 AAAA...`。")
    keytype, b64 = parts[0], parts[1]
    blob = base64.b64decode(b64)
    ctor = {"ssh-ed25519": paramiko.Ed25519Key, "ssh-rsa": paramiko.RSAKey,
            "ecdsa-sha2-nistp256": paramiko.ECDSAKey}.get(keytype)
    if ctor is None:
        raise SshTransferError(f"不支持的 host key 类型 `{keytype}`。")
    keyobj = ctor(data=blob)
    host = settings.SGP_SSH_HOST
    port = int(settings.SGP_SSH_PORT or 22)
    # paramiko 非 22 端口按 `[host]:port` 查 host key，22 按裸 host。两种都登记，改端口不会误拒。
    for name in ({host, f"[{host}]:{port}"} if port != 22 else {host}):
        client.get_host_keys().add(name, keytype, keyobj)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())


def _client():
    """建立到 SGP ECS 的 paramiko 连接。调用方负责 close()。"""
    import paramiko
    key = _load_private_key()
    c = paramiko.SSHClient()
    _add_host_key(c)
    try:
        c.connect(
            hostname=settings.SGP_SSH_HOST, port=int(settings.SGP_SSH_PORT or 22),
            username=settings.SGP_SSH_USER or "root", pkey=key,
            timeout=20, banner_timeout=20, auth_timeout=20,
            allow_agent=False, look_for_keys=False,
        )
    except Exception as e:
        raise SshTransferError(f"连接 SGP({settings.SGP_SSH_HOST}) 失败：{e}")
    return c


def run(cmd: str, *, timeout: int = 30) -> tuple[int, str, str]:
    """在 SGP 上跑一条**短**命令，drain 完 stdout/stderr 再取 exit（避死锁）。返回 (rc, out, err)。

    仅用于起任务/轮询这类秒回命令；长任务一律 nohup 后台化 + marker 轮询，不经此读长输出。
    """
    c = _client()
    try:
        _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        try:
            c.close()
        except Exception:
            pass


# ── 工作目录 / marker ─────────────────────────────────────────────────────────

def _job_dir(job_id: str) -> str:
    root = (settings.SGP_WORK_DIR or "/var/run/ssh_transfer").rstrip("/")
    return f"{root}/{job_id}"


def _marker(job_id: str, stage: str, ext: str) -> str:
    return f"{_job_dir(job_id)}/{stage}.{ext}"


# ── 起任务 ────────────────────────────────────────────────────────────────────

def _launch(job_id: str, stage: str, work_cmd: str) -> None:
    """把 work_cmd 后台化跑在 SGP：写 pid/rc/log marker。SSH 秒回，不等任务完成。"""
    job_dir = _job_dir(job_id)
    rc_path = _marker(job_id, stage, "rc")
    pid_path = _marker(job_id, stage, "pid")
    log_path = _marker(job_id, stage, "log")
    # work_cmd 跑完把退出码写进 rc；nohup 后台化；$! 记 pid。整条 redirection 在远端。
    inner = f"{work_cmd}; echo $? > {shlex.quote(rc_path)}"
    remote = (
        f"mkdir -p {shlex.quote(job_dir)} && "
        f"rm -f {shlex.quote(rc_path)} && "
        f"nohup bash -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 & "
        f"echo $! > {shlex.quote(pid_path)}"
    )
    rc, out, err = run(remote, timeout=30)
    if rc != 0:
        raise SshTransferError(f"SGP 起 {stage} 任务失败(rc={rc})：{err or out}")
    logger.info("[SSHT] %s %s 已后台起 pid_marker=%s", job_id, stage, pid_path)


def start_stage1(job_id: str, *, source_bucket: str, source_prefix: str) -> None:
    """段1：ossutil cp 杭州 OSS → SGP 挂载盘（增量 -u + 断点 checkpoint）。"""
    mount = (settings.SGP_OSS_MOUNT or "/mnt/sgp_oss").rstrip("/")
    jobs = int(settings.SGP_OSSUTIL_JOBS or 30)
    src = f"oss://{source_bucket}/{source_prefix}"
    dst = f"{mount}/{source_prefix}"
    ckpt = f"{_job_dir(job_id)}/ckpt1"
    work = (f"ossutil cp {shlex.quote(src)} {shlex.quote(dst)} "
            f"-r --jobs {jobs} -u --checkpoint-dir {shlex.quote(ckpt)}")
    _launch(job_id, STAGE1, work)


def start_stage2(job_id: str, *, source_prefix: str, dest_rel: str = "") -> None:
    """段2：rsync SGP 挂载盘 → 泰国服务器（方案 A 免 sudo；B 走 --rsync-path=sudo rsync）。

    源尾斜杠=拷贝目录内容；dest_rel=相对 THAI_DEST_ROOT 的目标子目录（空则镜像源前缀）。
    dest_rel 已在 paths 层走白名单校验（防泰国 ssh 双跳注入）。
    """
    mount = (settings.SGP_OSS_MOUNT or "/mnt/sgp_oss").rstrip("/")
    local_src = f"{mount}/{source_prefix}"                      # 尾斜杠：拷贝目录内容
    dest_root = (settings.THAI_DEST_ROOT or "").rstrip("/")
    if not dest_root:
        raise SshTransferError("未配置 THAI_DEST_ROOT（泰国目标根目录）。")
    remote_dest = f"{dest_root}/{dest_rel or source_prefix}"    # 尾斜杠
    thai_user = settings.THAI_USER or "wuji"
    thai_host = settings.THAI_HOST
    thai_port = int(settings.THAI_PORT or 22)
    ssh_e = (f"ssh -p {thai_port} -o BatchMode=yes "
             f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15")
    flags = ["-a", "--info=progress2"]
    if settings.THAI_RSYNC_BWLIMIT:
        flags.append(f"--bwlimit={shlex.quote(str(settings.THAI_RSYNC_BWLIMIT))}")
    if str(settings.THAI_RSYNC_SUDO).lower() == "true":        # 方案 B（默认 false=方案 A）
        flags.append('--rsync-path=sudo rsync')
    flags_s = " ".join(flags)
    target = f"{thai_user}@{thai_host}:{remote_dest}"
    # 先在泰国建目标目录（多级前缀时 rsync 不一定自动建父目录），再 rsync。
    mkdir = (f"ssh -p {thai_port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
             f"{thai_user}@{thai_host} mkdir -p {shlex.quote(remote_dest)}")
    work = (f"{mkdir} && rsync {flags_s} -e {shlex.quote(ssh_e)} "
            f"{shlex.quote(local_src)} {shlex.quote(target)}")
    _launch(job_id, STAGE2, work)


# ── 轮询 ──────────────────────────────────────────────────────────────────────

def _rc_ok(stage: str, rc: int) -> bool:
    return rc in _RSYNC_OK if stage == STAGE2 else rc == 0


def poll_stage(job_id: str, stage: str) -> dict:
    """查一段状态：{status: RUNNING|DONE|FAILED, rc, alive, error}。只读 marker，不读长 log。"""
    rc_path = _marker(job_id, stage, "rc")
    pid_path = _marker(job_id, stage, "pid")
    # 一次 SSH 把 rc / pid / 存活 都取回，减少往返
    probe = (
        f"if [ -f {shlex.quote(rc_path)} ]; then echo RC=$(cat {shlex.quote(rc_path)}); "
        f"elif [ -f {shlex.quote(pid_path)} ] && kill -0 $(cat {shlex.quote(pid_path)}) 2>/dev/null; "
        f"then echo ALIVE; else echo DEAD; fi"
    )
    rc_code, out, err = run(probe, timeout=30)
    line = (out or "").strip()
    if line.startswith("RC="):
        try:
            rc = int(line[3:].strip())
        except ValueError:
            rc = 1
        return {"status": "DONE" if _rc_ok(stage, rc) else "FAILED", "rc": rc,
                "alive": False, "error": "" if _rc_ok(stage, rc) else f"{stage} 退出码 {rc}"}
    if line == "ALIVE":
        return {"status": "RUNNING", "rc": None, "alive": True, "error": ""}
    # DEAD 且无 rc：进程异常退出（OOM/被杀/机器重启），当失败
    return {"status": "FAILED", "rc": None, "alive": False,
            "error": f"{stage} 进程异常退出（无退出码 marker）"}


def estimate_source(source_bucket: str, source_prefix: str) -> tuple[int, int, bool]:
    """用 SGP 上已配好的 ossutil du 估源前缀大小，返回 (字节, 对象数, ok)。

    ok=True 仅当确实解析出大小行（含 0 字节的空前缀也算已知）；SSH 不通/正则没命中→ok=False，
    由上层 fail-safe 当作需审批（不 fail-open 放行大迁移）。
    """
    src = f"oss://{source_bucket}/{source_prefix}"
    _, out, _ = run(f"ossutil du {shlex.quote(src)} 2>&1 | tail -30", timeout=180)
    text = out or ""
    b = n = 0
    mb = (re.search(r"sum\s*size[^\d]*([\d,]+)", text, re.I)
          or re.search(r"total[^\n]*?size[^\d]*([\d,]+)", text, re.I))
    if not mb:
        return 0, 0, False   # 没解析出大小 → 未知
    b = int(mb.group(1).replace(",", ""))
    mn = (re.search(r"object\s*count[^\d]*([\d,]+)", text, re.I)
          or re.search(r"total\s*count[^\d]*([\d,]+)", text, re.I))
    if mn:
        n = int(mn.group(1).replace(",", ""))
    return b, n, True


def tail_log(job_id: str, stage: str, lines: int = 20) -> str:
    """取某段日志尾部（排障用，非完成判定）。"""
    log_path = _marker(job_id, stage, "log")
    _, out, _ = run(f"tail -n {int(lines)} {shlex.quote(log_path)} 2>/dev/null || true", timeout=30)
    return out


# rsync --info=progress2 末行：`  1,234,567  73%   45.67MB/s    0:12:34`
_RSYNC_PROG = re.compile(r"([\d,]+)\s+(\d+)%\s+([\d.]+)\s*([KMGT]?)i?B/s", re.I)
# ossutil 2.x 进度里的速率（如 `123.4 MiB/s` / `12.3MB/s`），字节数格式不稳，只稳取速率
_RATE = re.compile(r"([\d.]+)\s*([KMGT]?)i?B/s", re.I)
_MULT = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def _speed_bps(num: str, unit: str) -> int:
    return int(float(num) * _MULT.get(unit.lower(), 1))


def stage_progress(job_id: str, stage: str) -> dict:
    """从日志尾部解析进度/速率（best-effort，只 tail 不 du，快）。返回 {bytes_done, pct, speed_bps}，
    解析不到的字段为 None。段2(rsync --info=progress2)最准；段1(ossutil)只稳取瞬时速率。"""
    text = tail_log(job_id, stage, lines=3)
    bytes_done = pct = speed_bps = None
    if stage == STAGE2:
        for m in _RSYNC_PROG.finditer(text):   # 取最后一次匹配（最新进度）
            bytes_done = int(m.group(1).replace(",", ""))
            pct = int(m.group(2))
            speed_bps = _speed_bps(m.group(3), m.group(4))
    else:
        for m in _RATE.finditer(text):         # 段1 只取最新速率
            speed_bps = _speed_bps(m.group(1), m.group(2))
    return {"bytes_done": bytes_done, "pct": pct, "speed_bps": speed_bps}
