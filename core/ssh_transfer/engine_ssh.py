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

# 段1 目的端 /mnt/sgp_oss 是 ossfs2(FUSE)，**只支持顺序写**。ossutil 对超过 100MiB 的对象默认
# 切分片、并发 pwrite 到不同 offset → 在该挂载点上必然 `invalid argument`(EINVAL)。
# 真机实证（19.527 TiB / 75850 对象那单）：≤100MiB 的 33484 个全成功、>100MiB 的 42366 个全失败，
# 边界与分片阈值严格吻合，且失败报告 42366 行全是同一个 EINVAL。
# 修法=强制单分片顺序写：--part-size 顶到 ossutil 上限 5Gi（合法区间 100Ki~5Gi）+ --parallel 1
# （文件内不并发）。跨文件并发 --job 保留：不同文件各自顺序写互不干扰，实测 job 30 稳定 148MiB/s
# （比出错那次的 75MiB/s 还快一倍），故不为此降并发。
# 残留限制：单个对象 >5Gi 无法压成单分片、仍会 EINVAL —— failure_detail() 会把这点直接讲给用户。
_OSSFS2_PART_SIZE = "5Gi"


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
    # 注意 `&` 优先级低于 `&&`：必须把「后台起任务 + 写 pid」用 { ...; } 组起来，否则
    # `mkdir && rm && nohup ... & echo $!>pid` 会把整段 mkdir&&rm&&nohup 一起丢后台、
    # echo $!>pid 不等 mkdir 就先跑 → 目录未建、写 pid 失败(rc=1)。分组确保 mkdir&&rm 先完成。
    inner = f"{work_cmd}; echo $? > {shlex.quote(rc_path)}"
    remote = (
        f"mkdir -p {shlex.quote(job_dir)} && "
        f"rm -f {shlex.quote(rc_path)} && "
        f"{{ nohup bash -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 & "
        f"echo $! > {shlex.quote(pid_path)}; }}"
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
    # 跑之前先删目标前缀下残留的 `*.temp`（上一次中断留下的半成品）。**这是必须的，不是打扫卫生**：
    # ossfs2 只能顺序写，ossutil 见到已存在的 `.temp` 会当成续传、从非零 offset 往里写 → EINVAL。
    # 真机实证（同一对象、同一 flags）：留着旧 .temp → rc=4 且 0.6 秒就失败；删掉 → rc=0 完整落地。
    # 不清的话，段1 只要被中断过一次（kill/容器重启/网络抖动），**之后每一次重试都必然失败**。
    # `.temp` 是 ossutil 自己的中间文件、永远不是有效数据；源桶里也不存在以 .temp 结尾的对象（已核）。
    # 用 find -name 精确按后缀删，绝不用通配符递归删，避免误伤已传好的正式文件。
    purge = (f"[ -d {shlex.quote(dst)} ] && "
             f"find {shlex.quote(dst)} -type f -name '*.temp' -delete 2>/dev/null; true")
    # ossutil 2.x 并发 flag 是 `-j/--job`（单数），非 `--jobs`（复数会报 unknown flag、段1挂）。
    # --parallel 1 + --part-size 5Gi：目的端 ossfs2 只能顺序写，见 _OSSFS2_PART_SIZE 处的实证说明。
    work = (f"{purge}; ossutil cp {shlex.quote(src)} {shlex.quote(dst)} "
            f"-r --job {jobs} --parallel 1 --part-size {_OSSFS2_PART_SIZE} "
            f"-u --checkpoint-dir {shlex.quote(ckpt)}")
    _launch(job_id, STAGE1, work)


def start_stage2(job_id: str, *, source_prefix: str, dest_rel: str = "") -> None:
    """段2：rsync SGP 挂载盘 → 泰国服务器（方案 A 免 sudo；B 走 --rsync-path=sudo rsync）。

    源尾斜杠=拷贝目录内容；dest_rel=相对 THAI_DEST_ROOT 的目标子目录（空则镜像源前缀）。
    dest_rel 已在 paths 层走白名单校验（防泰国 ssh 双跳注入）。

    THAI_RSYNC_STREAMS>1 时按**源一级目录**切分并行跑（见 _stage2_parallel_work 的实测依据），
    每个目录一条独立 rsync（各自 log/rc），外层聚合退出码。=1 或源下无子目录时退回单流。
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
    # -s(--secluded-args)：目的路径不经远端 shell 解析。并行版的路径含来自文件系统的目录名，
    # 没有它一个畸形目录名就能在泰国生产机上执行命令（rsync 3.2.7 起支持；单流也一并加固）。
    flags = ["-a", "-s", "--info=progress2"]
    if settings.THAI_RSYNC_BWLIMIT:
        flags.append(f"--bwlimit={shlex.quote(str(settings.THAI_RSYNC_BWLIMIT))}")
    if str(settings.THAI_RSYNC_SUDO).lower() == "true":        # 方案 B（默认 false=方案 A）
        # 必须整体引起来：裸展开会被 shell 拆成 `--rsync-path=sudo` + `rsync` 两个词，
        # 后者被当成第二个源路径 → 方案 B 一开就必挂（既存缺陷，并行版会把它复制进每条 worker）。
        flags.append(shlex.quote("--rsync-path=sudo rsync"))
    flags_s = " ".join(flags)
    target = f"{thai_user}@{thai_host}:{remote_dest}"
    # 先在泰国建目标目录（多级前缀时 rsync 不一定自动建父目录），再 rsync。
    mkdir = (f"ssh -p {thai_port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
             f"{thai_user}@{thai_host} mkdir -p {shlex.quote(remote_dest)}")
    single = (f"rsync {flags_s} -e {shlex.quote(ssh_e)} "
              f"{shlex.quote(local_src)} {shlex.quote(target)}")

    # 清理上一轮的 unit marker，**必须在 mkdir 之前、且两条分支都做**：
    #   - 放在 `mkdir &&` 之后：泰国侧 mkdir 挂了会短路跳过清理 → 卡片拿旧 unit rc 报「失败分片
    #     shard_003…」并去 grep 上一轮日志，把真原因(rc=255)完全盖住。
    #   - 单流分支不清：并行跑过一次后把 STREAMS 调回 1（或触发白名单回退），此后每次单流失败都
    #     被误报成分片失败；残留 unit 日志还会让 _stage2_parallel_progress 误判成并行、
    #     把进度冻结在上一轮的字节数。
    # _launch 已先 `mkdir -p job_dir`，所以这里目录一定存在。
    d = _job_dir(job_id)
    cleanup = (f"rm -f {shlex.quote(d)}/stage2.unit-*.rc {shlex.quote(d)}/stage2.unit-*.log "
               f"{shlex.quote(d)}/stage2.units")

    streams = _stage2_streams()
    if streams <= 1:
        work = f"{cleanup}; {mkdir} && {single}"
    else:
        # `{ ...; }` 分组是必须的：并行体是多条以 `;` 分隔的命令，而 `A && B; C` 里 `&&` 只管到
        # 第一个 `;` —— 不分组的话泰国侧 mkdir 失败后，find/xargs 照样往下跑（段1 也踩过同一个坑）。
        par = _stage2_parallel_work(job_id, local_src=local_src, target=target,
                                    flags_s=flags_s, ssh_e=ssh_e,
                                    streams=streams, single=single)
        work = f"{cleanup}; {mkdir} && {{ {par}; }}"
    _launch(job_id, STAGE2, work)


# 并发上限 10：泰国侧 sshd 用默认 `MaxStartups 10:30:100`（真机确认该行是注释状态=取默认），
# 限的是**未认证并发连接数**。8 条 rsync 同时握手 + 偶发 mkdir 已到 9，再往上会被随机拒连
# （rc=255，看起来像网络抖动、极难归因）。要调高必须先同步改泰国 sshd。
_STAGE2_MAX_STREAMS = 10


def _stage2_streams() -> int:
    """并行流数。非法/空值一律归一到 1（退回单流），绝不让配置错误把 bot 拖挂。"""
    raw = getattr(settings, "THAI_RSYNC_STREAMS", 1)
    try:
        n = int(str(raw).strip() or 1)
    except (TypeError, ValueError):
        logger.warning("[SSHT] THAI_RSYNC_STREAMS=%r 非法，退回单流", raw)
        return 1
    return max(1, min(n, _STAGE2_MAX_STREAMS))


# unit（源一级目录名）白名单：只允许 字母/数字/`.`/`_`/`-`。任一条目不合规 → 整批退回单流。
# 为什么是「整批退回」而不是「跳过坏名字」：跳过会**静默漏传**那个目录，用户拿到的是「成功」
# 却少数据；退回单流只是慢，不会错。名字来自对象存储 key（外部可控），必须当不可信输入。
_UNIT_SAFE_GREP = "^[A-Za-z0-9._-]+$"        # 给远端 grep -zE 用（-z 下 ^$ 锚 NUL 记录）
# Python 侧用 fullmatch + \A\Z：`$` 会放过结尾换行（paths.py 当年就是因此从 `$` 改成 `\Z`）。
_UNIT_SAFE_PY = re.compile(r"\A[A-Za-z0-9._-]+\Z")


def _stage2_parallel_work(job_id: str, *, local_src: str, target: str,
                          flags_s: str, ssh_e: str, streams: int, single: str) -> str:
    """生成「按源一级目录切分、xargs -P 并行跑 N 条 rsync」的远端 shell。

    为什么并行：SGP→泰国 30ms RTT，单条 TCP 被窗口/整形卡在 ~26-35MB/s。真机实测聚合吞吐
    随流数近线性上涨（1流 26MB/s → 4流 43MB/s → 8流 78MB/s，且是在另一条 rsync 已占 27MB/s
    之外测得，链路远未饱和）。19.5TiB 单流要 ~9 天，8 流约 2.5 天。

    切分粒度取源一级目录：本例 50 个 shard_NNN 各 430-480GB，天然均匀；xargs -P 做工作窃取，
    不需要预先按大小装箱。**不用 --files-from**：它不隐含 -r、语义易踩坑，且要先在 FUSE 上
    列全量文件（19.5TiB/7.5万对象，慢且没必要）。

    幂等：rsync -a 按 size+mtime 跳过已传好的文件（真机 dry-run 实测 shard_000 的 1684 个文件
    只有 372 个待传），所以从单流切到并行、或中断重跑，都不会重传已完成数据。
    """
    d = _job_dir(job_id)
    dq = shlex.quote(d)
    units = f"{d}/stage2.units"
    uq = shlex.quote(units)
    # 每 unit 一条 rsync，各自 log/rc。`$1` 是 xargs 传进来的目录名，**只做变量展开、
    # 不拼进任何 shell 字符串**，配合上面的白名单与 rsync -s，注入面被封死。
    worker = (
        'u="$1"; '
        f'rsync {flags_s} -e {shlex.quote(ssh_e)} '
        f'{shlex.quote(local_src.rstrip("/") + "/")}"$u"/ {shlex.quote(target.rstrip("/") + "/")}"$u"/ '
        f'> {dq}/stage2.unit-"$u".log 2>&1; '
        f'echo $? > {dq}/stage2.unit-"$u".rc'
    )
    # 聚合：**以 units 清单为准逐条核对**，不是遍历"存在的 rc 文件"。
    # 后者有个致命洞：50 个 unit 只落了 40 个 rc（xargs 被 OOM killer 挑走、sh fork 失败、
    # 重定向建 log 失败…）且都为 0 时，聚合出 0 → 报成功 → 10 个分片从未传输、零告警。
    # 缺 rc 一律按失败。同时把 xargs 自己的退出码折进来（123/124/125 正是它报"有子命令失败/
    # 被杀"的方式，原先被 `;` 整个丢弃）。
    # rc 内容为空/非数字（worker 被杀在建文件与写入之间）也按失败——否则 `[ "$agg" -eq 0 ]`
    # 会因非数字参数报错，行为不可控。
    # **绝不能用 `exit`** —— _launch 把 work_cmd 拼成 `work; echo $? > rc`，一 exit 就跳过写 rc，
    # 轮询侧会看到「进程没了又没 rc」而误判。用 `(exit $agg)` 只设 $? 不终止 shell。
    verify = (
        'agg=0; miss=0; '
        'while IFS= read -r -d "" u; do '
        f'  f={dq}/stage2.unit-"$u".rc; '
        '  if [ ! -f "$f" ]; then miss=$((miss+1)); agg=1; continue; fi; '
        '  r=$(cat "$f" 2>/dev/null); '
        '  case "$r" in 0) ;; 24) [ "$agg" -eq 0 ] && agg=24 ;; '
        '    ""|*[!0-9]*) agg=1 ;; *) agg="$r" ;; esac; '
        f'done < {uq}; '
        '[ "$miss" -gt 0 ] && echo "[stage2] 有 $miss 个分片没留下退出码(未跑/被杀)，按失败处理"; '
        '{ [ "$agg" -eq 0 ] || [ "$agg" -eq 24 ]; } && [ "${xrc:-0}" -ne 0 ] && agg="$xrc"; '
        'true'
    )
    # 收尾全量核对：分片全绿后再跑一趟**单流全量** rsync。两个作用：
    #   1) 补传源顶层的散文件/符号链接等**非目录**条目 —— 并行只枚举一级目录，不做这一步的话
    #      源里有个 manifest.json 就会静默不传，而 rc 仍是 0（"报成功但少数据"，比慢严重得多）。
    #   2) 兜住一切"分片以为传完了其实没传"的情形。
    # 代价很小：数据已在对端，rsync 按 size+mtime 全部命中跳过，只是一趟元数据扫描。
    # 用 --info=stats1 而非 progress2：输出短，且 "files transferred" >0 本身就是"并行漏了东西"
    # 的告警信号。分片有硬失败时**不跑**这趟（否则会串行重传整批，可能拖上几天）。
    sweep_flags = flags_s.replace("--info=progress2", "--info=stats1")
    sweep = (f'rsync {sweep_flags} -e {shlex.quote(ssh_e)} '
             f'{shlex.quote(local_src)} {shlex.quote(target)}')
    finish = (
        '{ [ "$agg" -eq 0 ] || [ "$agg" -eq 24 ]; } && { '
        '  echo "[stage2] 分片全部完成，收尾全量核对(补顶层散文件+兜底漏跑)"; '
        f'  {sweep}; s=$?; '
        '  case "$s" in 0) ;; 24) [ "$agg" -eq 0 ] && agg=24 ;; *) agg="$s"; '
        '    echo "[stage2] 收尾核对失败 rc=$s" ;; esac; '
        '}; '
        '(exit "$agg")'
    )
    return (
        # units 用 NUL 分隔（`-printf '%f\0'`）：`%f\n` 遇到**名字里带换行**的目录会把一条打成
        # 两行，两半各自都能过白名单 → 不触发回退；若两半恰好都是真实目录名，就是真目录静默
        # 漏传 + rc=0。NUL 分隔从根上避免（配 grep -z / xargs -0）。
        # 另外用条数交叉核对：`-printf 'x\n'` 的计数与名字无关、天然换行安全，对不上就退回单流。
        # 也顺带兜住 find 部分失败（那会产出部分 units 而不报错，又是静默少传）。
        f'find {shlex.quote(local_src)} -mindepth 1 -maxdepth 1 -type d -printf "%f\\0" '
        f'2>/dev/null > {uq}; frc=$?; '
        f'nd=$(find {shlex.quote(local_src)} -mindepth 1 -maxdepth 1 -type d -printf "x\\n" '
        f'2>/dev/null | wc -l); '
        f'nu=$(tr -cd "\\0" < {uq} 2>/dev/null | wc -c); '
        # 无子目录（纯散文件源）/ find 报错 / 不合规目录名 / 条数不符 → 退回单流，宁慢勿错。
        # frc 必须单独看：find 部分失败（读目录出错）会产出**部分**清单却不报错到管道外，
        # 而两次 find 会失败得一样、条数照样相等 → 交叉核对抓不到，只有退出码能抓。
        f'if [ ! -s {uq} ] || [ "${{frc:-1}}" -ne 0 ] || [ "${{nd:-0}}" -ne "${{nu:-0}}" ] || '
        f'LC_ALL=C grep -zqvE {shlex.quote(_UNIT_SAFE_GREP)} {uq}; then '
        f'  echo "[stage2] 源下无子目录/find出错(rc=$frc)/含不安全目录名/条数不符($nd vs $nu)，退回单流"; '
        f'  {single}; '
        f'else '
        f'  echo "[stage2] 并行 {streams} 流，切分 $nu 个一级目录"; '
        f'  xargs -0 -a {uq} -P {streams} -n1 sh -c {shlex.quote(worker)} _ ; xrc=$?; '
        f'  {verify}; '
        f'  {finish}; '
        f'fi'
    )


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


# 失败时值得摘出来的行：ossutil 汇总/报告路径、ossutil 裸 `Error:` 行、rsync 错误行。
# `Error:` 必须在列：源桶不在杭州那种失败（前一单 sgp-841b88a7b0dd，rc=2）日志里只有
# `Error: operation error ListObjectsV2 ... AccessDenied`，没有 FinishWithError/report，
# 漏了它明细就是空的、卡片又退回「退出码 2」。（grep 带 -i，故不必再列小写变体。）
_FAIL_GREP = "FinishWithError|Error occurs|See more information|rsync error|rsync:|Error:"
_DETAIL_MAX = 1200          # 进飞书卡片，掐总长
_DETAIL_LINE_MAX = 240      # 单行上限：逐行截，避免前面几条长汇总行把后面的根因/说明挤没
# report 路径来自远端日志内容（外部数据）。虽已 shlex.quote，仍按白名单收窄，避免被日志里
# 精心构造的对象 key 骗着去 grep 任意文件、再把内容贴进飞书群（读取 oracle）。
# 必须「先切出整个空白分隔 token，再 fullmatch」——直接对整段 search 会从
# `/root/x.report_evil` 里截出 `/root/x.report`、从 `relative/x.report` 里截出 `/x.report`，
# 等于放过了构造串（tester 抓到）。另禁 `..`：字符集含 `.` 和 `/`，否则 `/a/../../etc/x.report` 能过。
_REPORT_PATH_RE = re.compile(r"/[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*\.report")
_TRIM_PUNCT = ".,;:)]}'\"`"             # 真机日志里路径可能被标点/引号/括号裹着


def _safe_report_path(blob: str) -> str:
    """从日志片段里取第一个可信的 report 路径；取不到返回 ""。

    白名单：整个 token fullmatch「绝对路径 + 仅 [A-Za-z0-9._-] 与 / + 精确 .report 结尾」且不含 `..`。
    这封死了注入面（空格/分号/反引号/`$()`/相对路径/后缀不闭合全拒）。

    **刻意不再额外限制目录**：曾加过「必须在 job 目录或 /ossutil_output/ 下」，能把残留的窄读取
    oracle（指向 SGP 上某个真实存在的 .report、泄露其一行 cause）也封掉，但 ossutil 的报告目录是可变的，
    一旦与硬编码不符，最有价值的「首条根因」会**静默消失**——而那正是本函数存在的意义。
    权衡：注入已封死，残留泄漏面需要攻击者先能写源桶对象名、再猜中真实存在的 .report 路径，仅泄漏一行；
    代价却是核心诊断能力时不时失灵。故只保结构白名单。（auditor 将目录限制列为可选低危。）
    """
    for raw in blob.split():
        tok = raw.rstrip(_TRIM_PUNCT)      # 去尾部标点，否则真机 `(/root/x.report)` 会静默摘不到
        if tok.endswith(".report") and ".." not in tok and _REPORT_PATH_RE.fullmatch(tok):
            return tok
    return ""


def _clip(line: str) -> str:
    return line[:_DETAIL_LINE_MAX]


def failure_detail(job_id: str, stage: str) -> str:
    """失败后摘一段人可读原因：ossutil/rsync 汇总行 + 报告路径 + 失败条数 + 首条根因。

    卡片原来只给「stage1 退出码 4」，排障得手工翻十几 MB 日志才知道是写入 EINVAL —— 而 ossutil
    其实早把明细写进了自己的 report。这里把它捞回来。best-effort：任何一步失败就返回已拿到的部分。

    注意：ossutil 进度用 `\\r` 刷屏（单次任务日志可达十几 MB），必须 `tr '\\r' '\\n'` 再筛，
    直接 `tail -n` 只会抓到一整行进度条、看不见尾部真错误（本次排障踩过）。
    """
    log_path = _marker(job_id, stage, "log")
    pre_key: list[str] = []
    if stage == STAGE2:
        # 并行段2：真错误在某个 unit 的日志里，stage2.log 只有外层那两行编排输出。
        # 挑第一个失败 unit 的日志来 grep，并把失败分片清单提到结论行——不然卡片上只会写
        # 「退出码 N」，运维不知道 50 个分片里是哪几个挂了。
        failed = _stage2_failed_units(job_id)
        if failed:
            shown = ", ".join(failed[:6]) + (" …" if len(failed) > 6 else "")
            pre_key.append(_clip(f"失败分片 {len(failed)} 个：{shown}"))
            log_path = f"{_job_dir(job_id)}/stage2.unit-{failed[0]}.log"
    # 取 20KB 尾窗：汇总行之后 ossutil 还会刷若干输出，窗口太小会把 report 路径那行挤出去。
    # `tail -n +2` 丢掉第一行——按字节切必然切在行中/多字节中，留着会是乱码碎片。
    # 一条命中都没有时（错误形态没见过）兜底回尾部原文：噪声也比卡片上只有「退出码 N」强。
    # 兜底必须在远端做——过滤后的输出为空时，本地已经没有原文可退回了。
    probe = (
        f"L={shlex.quote(log_path)}; sz=$(wc -c < \"$L\" 2>/dev/null || echo 0); "
        f"t=$(tail -c 20000 \"$L\" 2>/dev/null | tr '\\r' '\\n'); "
        # 只有确实被 -c 截断（日志 >20KB）时才丢首行。无条件丢会把「整个日志只有一行」的快速失败
        # 吞成空明细——正是异地桶 rc=2 那种形态（日志只有一行 `Error: ... AccessDenied`），
        # 明细一空卡片就又退回「退出码 N」。auditor 本地 bash 实测到的，单测桩掉 run 抓不到。
        f"if [ \"${{sz:-0}}\" -gt 20000 ]; then t=$(printf '%s\\n' \"$t\" | tail -n +2); fi; "
        f"f=$(printf '%s\\n' \"$t\" | grep -aiE {shlex.quote(_FAIL_GREP)} | tail -3); "
        f"if [ -n \"$f\" ]; then printf '%s\\n' \"$f\"; "
        f"else printf '%s\\n' \"$t\" | grep -av '^[[:space:]]*$' | tail -3; fi"
    )
    try:
        _, out, _ = run(probe, timeout=20)
    except Exception:
        logger.warning("[SSHT] 取 %s %s 失败明细失败（SSH 不可用或超时）", job_id, stage, exc_info=True)
        return ""
    blob = out or ""
    # lead = 日志原文摘出来的行；key = 我们二次加工出的关键结论（条数/根因/说明）。
    # 分开是为了最后按预算拼：key 必须完整保留，超预算只削 lead——否则几条长汇总行就把根因挤没了。
    lead = [_clip(ln.strip()) for ln in blob.splitlines() if ln.strip()]
    key: list[str] = list(pre_key)

    # ossutil 把逐个失败对象写进 report 文件；取条数 + 首条 cause（cause 才是真原因）
    report = _safe_report_path(blob)
    if report:
        try:
            _, out2, _ = run(
                f"echo COUNT=$(grep -ac 'cause:' {shlex.quote(report)} 2>/dev/null || echo 0); "
                f"grep -aom1 'cause: .*' {shlex.quote(report)} 2>/dev/null | cut -c1-240",
                timeout=25)
        except Exception:
            logger.warning("[SSHT] 读 %s 失败报告 %s 失败", job_id, report, exc_info=True)
            out2 = ""
        cm = re.search(r"COUNT=(\d+)", out2 or "")
        if cm and int(cm.group(1)):
            key.append(_clip(f"失败对象 {cm.group(1)} 个，明细报告：{report}"))
        cause = re.search(r"cause:\s*(.+)", out2 or "")
        if cause:
            key.append(_clip(f"首条根因：{cause.group(1).strip()}"))
        blob += out2 or ""

    # EINVAL 是 ossfs2 顺序写限制的签名错误。只有段1 往 ossfs2 挂载点写，段2 是 rsync 到泰国
    # 本地盘、同样的错另有原因，别把段1 的结论硬套上去。
    if "invalid argument" in blob.lower():
        if stage == STAGE1:
            key.append(
                f"说明：目的端 ossfs2(FUSE) 不支持随机偏移写（段1 已强制单分片顺序写 "
                f"--parallel 1 --part-size {_OSSFS2_PART_SIZE}）。优先核查目标前缀下是否残留上次中断的 "
                f".temp（会被当成偏移续写、必失败），以及是否存在 >{_OSSFS2_PART_SIZE} 的单个对象。")
        else:
            key.append("说明：目的端写入被拒(EINVAL)，请复查泰国侧目标目录与挂载点。")

    # 拼装：key（我们提炼的结论：失败条数/首条根因/说明）必须完整保住，剩余预算才轮到 lead
    # （日志原文行）。反过来先拼 lead 再整体截，几条长汇总行就能把根因和说明全挤掉。
    budget = _DETAIL_MAX - len("\n".join(key)) - (1 if key else 0)
    head: list[str] = []
    for ln in lead:
        if len(ln) + 1 > budget:
            break
        head.append(ln)
        budget -= len(ln) + 1
    return "\n".join(head + key)[:_DETAIL_MAX]


def _stage2_failed_units(job_id: str) -> list[str]:
    """并行段2：列出退出码不在 {0,24} 的 unit 名。非并行/取不到 → 空列表（调用方退回原逻辑）。"""
    d = _job_dir(job_id)
    probe = (
        f'd={shlex.quote(d)}; '
        'for f in "$d"/stage2.unit-*.rc; do [ -f "$f" ] || continue; '
        'r=$(cat "$f" 2>/dev/null); case "$r" in 0|24) ;; *) '
        'u=${f##*/stage2.unit-}; echo "${u%.rc}"; ;; esac; done'
    )
    try:
        _, out, _ = run(probe, timeout=20)
    except Exception:
        logger.warning("[SSHT] %s 列失败分片失败", job_id, exc_info=True)
        return []
    # 远端输出是外部数据：只收白名单内的名字，防被构造的目录名带进后续的 log 路径拼接。
    # fullmatch（模式已用 \A..\Z）：re.match + `$` 会放过结尾换行。
    return [ln.strip() for ln in (out or "").splitlines()
            if ln.strip() and _UNIT_SAFE_PY.fullmatch(ln.strip())]


def estimate_source(source_bucket: str, source_prefix: str) -> tuple[int, int, bool]:
    """用 SGP 上已配好的 ossutil du 估源前缀大小，返回 (字节, 对象数, ok)。

    ok=True 仅当确实解析出大小行（含 0 字节的空前缀也算已知）；SSH 不通/正则没命中→ok=False，
    由上层 fail-safe 当作需审批（不 fail-open 放行大迁移）。
    """
    src = f"oss://{source_bucket}/{source_prefix}"
    _, out, _ = run(f"ossutil du {shlex.quote(src)} 2>&1 | tail -30", timeout=180)
    text = out or ""
    b = n = 0
    # ossutil 2.2.2 du 汇总行（真机反查，TAB 分隔、纯字节整数、无 MB 后缀/无逗号）：
    #   total object count: 3 \t total object sum size: 23208637
    #   total du size:23208637         （冒号后可能无空格）
    # 必须锚定含 total 的**汇总串**抓其后数字。旧版 `sum size[^\d]*([\d,]+)` 会先命中**表头**
    # `storage class\tobject count\tsum size`，且 `[^\d]*` 跨行一路吞到数据行第一个数字
    # （object count=3）→ 把 22MB 误读成 3B。故这里锚定 `total object sum size`/`total du size`，
    # 分隔用 `[:\s]+`（不再用会跨行乱吞的 `[^\d]*`）。
    mb = (re.search(r"total\s+object\s+sum\s+size[:\s]+(\d[\d,]*)", text, re.I)
          or re.search(r"total\s+du\s+size[:\s]+(\d[\d,]*)", text, re.I))
    if not mb:
        return 0, 0, False   # 没解析出大小 → 未知
    b = int(mb.group(1).replace(",", ""))
    mn = re.search(r"total\s+object\s+count[:\s]+(\d[\d,]*)", text, re.I)
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


def _stage2_parallel_progress(job_id: str) -> dict | None:
    """并行段2：把各 unit 日志的最后一个 progress2 计数求和。无 unit 日志 → None（走单流解析）。

    只回 bytes_done，**不回 pct/speed_bps**：
    - pct：每条 rsync 的百分比是它自己那个 unit 的，跨 unit 没有意义；交给上层用
      bytes_done/bytes_total 算全局真值。
    - speed_bps：rsync 单行速率抖动极大（同一次传输里 100kB/s 与 22MB/s 交替出现，
      排障时我就被它误导过），上层 _sample_progress 在 speed 缺省时会用相邻两次字节采样差
      算 60s 平均，那个稳得多。
    """
    d = _job_dir(job_id)
    # 取尾窗 + 只在确实截断时丢首行（按字节切必然切在行中，半截数字会让求和少算一大截）。
    # awk 只做去逗号后**原样打印字符串**，不做算术：mawk 对 3e11 这种大数会输出科学计数法，
    # 一旦被当成数字打印就彻底错了。64 位加法交给 shell 的 $(( )).
    probe = (
        f'd={shlex.quote(d)}; tot=0; n=0; nrc=0; '
        'for f in "$d"/stage2.unit-*.log; do [ -f "$f" ] || continue; n=$((n+1)); '
        'sz=$(wc -c < "$f" 2>/dev/null || echo 0); '
        "t=$(tail -c 8000 \"$f\" 2>/dev/null | tr '\\r' '\\n'); "
        'if [ "${sz:-0}" -gt 8000 ]; then t=$(printf "%s\\n" "$t" | tail -n +2); fi; '
        "v=$(printf '%s\\n' \"$t\" | grep -aE '[0-9]+%' | tail -1 | "
        "awk '{gsub(/,/,\"\",$1); print $1}' | grep -aoE '^[0-9]+'); "
        '[ -n "$v" ] && tot=$((tot+v)); done; '
        'for f in "$d"/stage2.unit-*.rc; do [ -f "$f" ] && nrc=$((nrc+1)); done; '
        'echo "BYTES=$tot UNITS=$n DONE=$nrc"'
    )
    try:
        _, out, _ = run(probe, timeout=45)
    except Exception:
        logger.warning("[SSHT] %s 取并行段2 进度失败", job_id, exc_info=True)
        return None
    m = re.search(r"BYTES=(\d+)\s+UNITS=(\d+)\s+DONE=(\d+)", out or "")
    if not m or int(m.group(2)) == 0:
        return None                       # 没有 unit 日志 → 这次是单流跑的
    return {"bytes_done": int(m.group(1)), "pct": None, "speed_bps": None,
            "units_total": int(m.group(2)), "units_done": int(m.group(3))}


def stage_progress(job_id: str, stage: str) -> dict:
    """从日志尾部解析进度/速率（best-effort，只 tail 不 du，快）。返回 {bytes_done, pct, speed_bps}，
    解析不到的字段为 None。段2(rsync --info=progress2)最准；段1(ossutil)只稳取瞬时速率。"""
    if stage == STAGE2:
        par = _stage2_parallel_progress(job_id)
        if par is not None:
            return par
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
