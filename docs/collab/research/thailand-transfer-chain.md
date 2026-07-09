# 泰国迁移链路调研（杭州 OSS → SGP ECS 中转 → 泰国服务器）

> researcher 产出，结论带出处。可信度标注：【文档】官方明说 / 【实测】真机反查 / 【推测】未证实需核。
> 结论面向 dev 的实现决策。**本文只写 docs/collab/research/，不碰源码。**

## 0. 一句话结论

这条链路**不是云迁移 API**（MGW/DMS），而是 **shell 编排 + SSH 远程执行 + 轮询状态文件**。可以照搬
`core/transfer/orchestrator.py` 的状态机 / Redis job / `run_to_completion` 轮询 / `on_update` 推卡骨架，
但把"提交云任务 + poll 云 API"两个引擎调用换成"SSH 到 SGP ECS 起后台命令 + SSH 轮询 marker/日志文件"。
两段命令天然对应两个 stage，幂等和失败重跑靠 **ossutil `-u` + rsync `-a` 本身的增量语义 + 每段落地 done/fail marker**。

链路（两段都在 SGP ECS 43.98.203.59 上执行）：
```
段1 DOWNLOAD:  oss://wuji-data-tran/<prefix>/  --ossutil cp -r--> /mnt/sgp_oss/<prefix>/   (杭州→新加坡走 CEN 专线)
段2 PUSH:      /mnt/sgp_oss/<prefix>/           --rsync 远程 sudo--> 泰国服务器:/dest/<prefix>/
```
bot 容器(115.191.2.86) 通过 SSH 驱动这两段并监控，自己不搬数据。

---

## 1. ossutil cp 语义、幂等、断点、进度

### 1.1 三个 flag
- `-r` / `--recursive`：递归整个目录/前缀。【文档】
- `--jobs`（题目写 `--job 30`）：**多文件间并发数**——同时处理多少个对象。另有 `--parallel`
  控**单个大文件内**分片并发；批量大文件实际并发 ≈ jobs × parallel。资源紧张时调低到 ≤100，过高会
  性能下降甚至 EOF 报错，建议从小往上调。【文档】
  出处：https://www.alibabacloud.com/help/en/oss/developer-reference/view-options 、
  https://www.alibabacloud.com/blog/oss-open-source-tool-ossutil-upload-performance-tuning_594122
- `-u` / `--update`：**增量/跳过已存在且不更旧**。下载方向语义="目的地本地已存在同名文件且其
  last-modified ≥ 源对象时，跳过"，即**只下缺的或源更新的**。→ **可安全重跑，幂等**。
  代价：`-u` 对每个对象至少发一次 HEAD 请求（有请求费/耗时，数据很少变时是纯开销）。【文档】
  出处：https://www.alibabacloud.com/help/en/oss/developer-reference/download-objects-5 、
  upload-objects-6（"Uploads only if the destination object is missing or older than the source file"）

> **对本链路的含义**：段1 `-u` = 天然增量，段1 中断后重跑只补没下完的对象，安全幂等。可重跑 == 失败恢复的基石。

### 1.2 断点续传（大文件）
- cp 按文件大小自动选简单/断点续传：默认 ≥100MB（`--bigfile-threshold` 可调）走分片断点续传。【文档】
- 失败信息记在**本地 `.ossutil_checkpoint` 目录**（默认当前目录，`--checkpoint-dir` 可指定）；重跑读
  checkpoint 从上次分片续传，成功后自动删除该目录。**小文件不支持续传**，失败则整文件重下。【文档】
  出处：https://www.alibabacloud.com/help/en/oss/developer-reference/breakpoint-file-resumable
- 坑：若用 `rm` 删了未完成的 multipart 任务，下次续传会 `NoSuchUpload` 报错——删掉 checkpoint 文件重来即可。【文档】
- **`--snapshot-path`**（区别于 checkpoint）：记录每个文件 lastModifiedTime 到本地目录，跳过未变文件，
  加速**重复批量**下载。适合大批量；小批量/网络好时官方建议直接用 `-u`。可与 `-u` 叠加（先查 snapshot 再查 update）。
  ossutil **不自动清** snapshot，要定期删。【文档】同上 breakpoint-file-resumable

### 1.3 完成判定：退出码 + 报告文件
- **退出码**：非 0 = 出错。ossutil 2.0 明确"返回码标准化，非 0 表示错误，具体码标识出错阶段，便于程序化处理"。
  【文档】https://www.alibabacloud.com/help/en/oss/developer-reference/ossutil-2-0-new-features
  → **判完成的首选信号：`echo $?` == 0**。这是编排里段落成败的权威依据。
- **报告文件**：批量操作过程中 ossutil 把出错对象详情写进报告文件（batch report），命令行同时输出
  succeed/error 计数摘要。**注意**：若在进入批量迭代**之前**就出错（命令拼错、桶不存在、AK 错误
  导致鉴权失败），**不生成报告文件**，直接屏幕报错退出。【文档】
  出处：https://www.alibabacloud.com/help/en/oss/developer-reference/faq-9 、breakpoint-file-resumable
- 【推测/需真机核】报告文件精确路径与命令行摘要行的确切格式（形如 `Succeed: Total num..., OK num...` /
  `Error num: N`）随版本变化，1.x 默认落 `ossutil_output/` 目录。**建议真机 `ossutil64 help cp` +
  实跑一次小目录取证**（只读式：拿真桶一个小 prefix 跑一次看 `$?`、屏幕摘要、有无 output 目录）。

### 1.4 进度
- ossutil 跑批量会在**屏幕滚动**打印进度（已传/总量、速率），非结构化。可解析性一般。【文档/推测】
- **本链路推荐不解析滚动进度**，而是：把 stdout/stderr 重定向到日志文件 → 轮询时 `tail` 日志给个"活着"信号 +
  结束时以 `$?` 落 marker 定成败。真正的量化进度可选：跑前用 OSS 侧 `_prefix_size` 估总量（本仓
  `tools/aliyun/oss._prefix_size` 已有），跑中 `du -sb /mnt/sgp_oss/<prefix>` 对比出百分比——比解析 ossutil 屏幕稳。

### 1.5 ossutil 1.x vs 2.x 差异（影响命令拼装）
- **2.0 必须配 region + 升级到签名 V4**：AK ID / AK Secret / **region ID 三件套**，region 漏配直接失败。
  这是迁移最常见坑。【文档】https://www.alibabacloud.com/help/doc-detail/2804546.html 、configure-ossutil
- 2.0 命令改**分层结构**（`ossutil api ...` / 高级命令），退出码标准化，新增 `--dry-run/-n`、
  `--output-format json/yaml/xml`、`ls/cp/rm` 支持按路径/大小/时间/元数据过滤、支持 stdin 管道。【文档】
  ossutil-2-0-new-features
- 2.0 **默认不开断点续传**，要显式带 `--checkpoint-dir` 激活（查 `.ucp` 文件判断已传分片）；1.x 默认开、
  用 `.ossutil_checkpoint`。【文档】breakpoint-file-resumable
- `-r` / `-u` / `--jobs` / `--parallel` **两版都在**，语义一致。
- **行动项**：先在 SGP ECS 上 `ossutil version` 确认装的是 1.x 还是 2.x（命令 `-u` 写法一致，但 2.x 要保证
  已配 region，否则段1 直接鉴权失败）。题目命令 `-u`/`--job`/`-r` 两版通用。

---

## 2. rsync 远程 sudo（段2）

### 2.1 正确写法
```bash
rsync -a --info=progress2 --rsync-path="sudo rsync" /mnt/sgp_oss/<prefix>/ user@thai-host:/dest/<prefix>/
```
- `--rsync-path="sudo rsync"`：远端用 sudo 起 rsync，从而有 root 权限写需 sudo 的目标目录（chown 等需 uid=0）。【文档】
- `-a`（archive）= 递归 + 保留权限/属主/时间/软链等，**本身就是增量**（只传大小/mtime 变了的），可安全重跑=幂等。【文档】
- 无 TTY 时 sudo 默认要密码 → 报 `sudo: no tty present and no askpass program specified`，故**必须**在泰国侧配 NOPASSWD。
  出处：https://www.systutorials.com/how-to-write-to-remote-side-with-sudo-in-rsync/ 、
  https://raimue.blog/2016/09/25/how-to-run-rsync-on-remote-host-with-sudo/

### 2.2 泰国侧 sudoers 最小化 NOPASSWD
`/etc/sudoers.d/rsync-bot`（`visudo -f` 校验，权限 0440），用**绝对路径**：
```
# 先确认泰国机 rsync 绝对路径：which rsync  （多为 /usr/bin/rsync）
transferuser ALL=(root) NOPASSWD: /usr/bin/rsync
```
可进一步收敛到 server 模式（rsync over ssh 远端实际以 `--server` 起）：
```
transferuser ALL=(root) NOPASSWD: /usr/bin/rsync --server *
```
【文档】raimue.blog / strugglers.net。注意 `secure_path`：sudo 默认清环境变量，路径丢失会诡异失败，
sudoers 里保留 `secure_path` 即可。

### 2.3 安全隐患（务必写进上线清单）—— 这是本链路最大风险点
- **`sudo rsync` 不限参数 == 等于给该用户经 rsync 的完整 root 写权限**：能覆盖任意文件，包括
  `/etc/sudoers.d/` 自身 → 限制形同虚设。凡能 SSH 成该用户或攻破源机(SGP ECS)者即可拿泰国机 root。
  "compromise of host A is also a compromise of host B"。【文档】
  出处：https://strugglers.net/~andy/mothballed-blog/2021/04/10/rsync-and-sudo-without-x-forwarding/ 、
  https://realtechtalk.com/rsync_run_as_root_sudo_without_password-1888-articles
- **收敛建议（按性价比排序）**：
  1. **目标目录改为 transferuser 直接可写**（chown 给它 / 放它的属主目录），则**根本不需要 sudo**——首选，直接消除风险。
  2. 若必须 sudo：`authorized_keys` 里给这把 key 加 `from="43.98.203.59",command="..."` 限定来源 IP + 只能跑
     指定命令；sudoers 尽量限到 `--server *`。
  3. rsync 目标目录用独立分区/子树，别让 transferuser 能碰系统关键路径。
  4. 别用 `(ALL)`，用 `(root)`；别开 `ALL=(ALL) NOPASSWD: ALL`。

### 2.4 rsync 进度 / 退出码
- `--info=progress2`：整体进度（%、速率、剩余）；但它是回车刷新式，**适合交互看，不适合日志解析**——
  编排里重定向到日志只取"活着"信号，成败看退出码。【文档】
- 退出码（编排必须区分）：`0` 成功；`23` 部分传输因错误（**真失败，要当错误**）；`24` 源文件在扫描时消失
  （多为良性）；`255` 通常是 **SSH 连接层失败**（rsync 透传 ssh 退出码）；`127` = 远端没装 rsync。【文档】
  出处：https://lxadm.com/rsync-exit-codes/ 、https://bluebones.net/2007/06/rsync-exit-codes/
- **建议**：段2 判成功 = 退出码 ∈ {0, 24}；{23,255,127,其它非0} = 失败。

---

## 3. bot(Python) 经 SSH 跑远端长任务并监控

### 3.1 paramiko vs subprocess-ssh
| | paramiko | subprocess 调 ssh 客户端 |
|--|--|--|
| 依赖 | 需 `pip install paramiko`（**当前 requirements.txt 无**，装它要 `docker compose up -d --build`，见 CLAUDE.md 部署节） | 容器内需装 openssh-client（多数基础镜像没有，也要 rebuild） |
| 私钥 | 直接从内存/字符串加载 `RSAKey.from_private_key(io.StringIO(...))`——**能配合 Fernet 解密后不落盘** | 需 key 落到文件（`-i keyfile`），或用 ssh-agent，落盘更难避免 |
| 长任务 | `exec_command` 拿 channel，可 `recv_exit_status()`(阻塞)/`exit_status_ready()`(非阻塞轮询) | 每次 `ssh host 'cmd'` 一发一收，天然无状态 |
| 坑 | **大输出前调 `recv_exit_status` 会死锁**（远端写满 window 等你读、你等 exit）→ 必须先 drain stdout/stderr 或开线程读；`exit_status_ready` 偶发永不 true（坏 server）| 无死锁问题（进程级） |

出处（paramiko 死锁/退出码）：https://docs.paramiko.org/en/latest/api/channel.html 、
https://github.com/paramiko/paramiko/issues/1208

**推荐：paramiko + "fire-and-forget 后台命令 + 轮询 marker 文件"模式**（不长期挂 channel）：
- 不要用一条长挂的 `exec_command` 阻塞等 TB 级任务几小时（channel 断=状态丢、还有死锁风险）。
- 而是每次 SSH 只发**短命令**：起任务时发 `nohup ... &`（秒回），监控时发 `cat marker`（秒回）。
  → 完全绕开 paramiko 长任务死锁 + 容器重启 channel 丢失两个坑，也和本仓"3s 回调只读、后台线程干活"的既有纪律契合。

### 3.2 远端后台起任务 + marker（核心模式）
关键：**redirection 必须在远端发生**——`ssh host 'cmd > log 2>&1'` 整条要**加引号**，否则 log 落到本地/无效。
【文档】nohup 单独不后台化，要配 `&`；`2>&1` 别写成 `&2>1`。
出处：https://groups.google.com/g/comp.os.linux.misc/c/9wqVvfO0zvM

段1 起任务（一条 SSH 发出即返回）：
```bash
RUN=/var/run/tr/<job_id>            # 每 job 一个工作目录（远端）
mkdir -p $RUN
nohup bash -c '
  ossutil cp -r -u --jobs 30 --checkpoint-dir '"$RUN"'/ckpt \
    oss://wuji-data-tran/<prefix>/ /mnt/sgp_oss/<prefix>/ > '"$RUN"'/stage1.log 2>&1
  echo $? > '"$RUN"'/stage1.rc          # 落退出码 = 完成 marker（原子：先写 tmp 再 mv 更稳）
' >/dev/null 2>&1 &
echo $! > $RUN/stage1.pid               # 记 pid 供"还在跑吗"探活
```
监控（每次 SSH 短命令，秒回）：
```bash
# 完成？   有 stage1.rc 即已结束，内容即退出码
cat $RUN/stage1.rc 2>/dev/null           # 空=还在跑；"0"=成功；非0=失败
# 还活着？ 无 rc 且 pid 在 → 在跑；无 rc 且 pid 不在 → 异常死亡（当失败）
kill -0 $(cat $RUN/stage1.pid) 2>/dev/null && echo RUNNING || echo DEAD
```
段2 同构（rsync + `echo $? > stage2.rc`）。

> 用 **pid 文件 + rc marker 文件**（不用 tmux/screen）：容器重启后 bot 无状态照样能 SSH 去
> `cat rc` / `kill -0 pid` 重新判定，完全不依赖任何本地进程或长连接。这正对应本仓
> `orchestrator.refresh()` 的"重启后点查询即自愈"设计。

### 3.3 SSH 私钥安全供给（不硬编码）
本仓已有 Fernet（`utils/crypto.py`）+ 环境变量配置约定（`config/settings.py`）。推荐：
- 私钥内容 Fernet 加密后放 env（如 `SGP_SSH_PRIVATE_KEY_ENC`）或 Redis，运行时 `decrypt()` → paramiko
  `RSAKey.from_private_key(io.StringIO(pem))` **只在内存**，绝不落盘、绝不打印。与用户 AK 走 `user:ak:*`
  同套路。【本仓模式】utils/crypto.py::decrypt / encrypt_strict
- 新增 config 字段（`SGP_SSH_HOST=43.98.203.59` / `SGP_SSH_PORT` / `SGP_SSH_USER` / `SGP_SSH_KEY_*` /
  `THAI_HOST`/`THAI_USER`/`THAI_DEST_ROOT`）要登进 `Config._REQUIRED_FIELDS` 带 impact 串（CLAUDE.md 约定）。
- **known_hosts / host key**：paramiko 别用 `AutoAddPolicy`（MITM 风险）；预置 SGP ECS 的 host key 或
  首次固定指纹。【推测/最佳实践】
- 私钥、AK、任何凭证：**绝不写进本文件、绝不打印到日志**（本项目已有多起凭证泄漏事件，见 MEMORY）。

---

## 4. 可照搬的本仓骨架 + 与云 API 迁移的差异

### 4.1 能几乎逐字照搬的部分（参考 `core/transfer/orchestrator.py`）
- **状态机**：`NEW → STAGE1(下载) → STAGE2(推送) → DONE | FAILED`（对应现有 `NEW→SINKING→CROSSING→DONE|FAILED`
  的两段式，语义一模一样：段1≈沉降、段2≈跨云）。
- **Redis job 记录**：`ssh:transfer:job:{job_id}`（新命名空间，别复用 `transfer:*`），30 天 TTL，
  `get_job/_save`（每次刷 `updated_ts`——auditor 曾抓过 bucket_transfer 漏刷导致对账 stale 门失效，务必带上）。
- **幂等 job_id**：`hash(source, dest, 当天)`（`_job_id` 现成写法，前缀改 `sgp-` 或 `th-`；查询按 ID 前缀分派处
  `_h_query_progress_by_id` 要加这个新前缀，和 `tr-`/`cpfs-`/`vepfs-` 并列）。
- **`run_to_completion(job, on_update, poll_interval, max_polls)`**：起段1→轮询 stage1.rc→段1成功起段2→
  轮询 stage2.rc→终态。`on_update` 每次 stage 变化推卡，逐字照搬现有循环结构。
- **卡片四件套**（`core/transfer/cards.py`）：`entry_card`(源/目的/同名策略) / `confirm_card`(带审批阈值,
  超阈值 orange + 仅 admin) / `progress_card_v2`(**schema 2.0 纯展示**，避 200830) / `result_card`(带 retry 按钮)。
- **actions handler 四件套**（`core/feishu_bot/actions.py`）：
  - confirm：`transfer:launch:{job_id}` NX 锁(ex=120) + `launched` 持久标记防连点起多线程；起后台 `Thread(daemon=True)`
    跑 `run_to_completion`；同步路径 3s 内**只回卡不做网络**，2.0确认卡→回 `progress_card_v2`(2.0→2.0)。
  - query：`_async_refresh_and_push` 后台真 refresh；sync 只读 `get_job` 秒回（**绝不在 3s 回调里 SSH 探云/远端**，
    同 auditor HIGH-1 的教训——超时会被飞书重投+15s 去重吞成 `{}`）。
  - retry：重置 `stage=NEW`/`launched=False`/`error=""`，清 `ssh:transfer:launch:{job_id}` + `dataflow:notified:{job_id}` 两键。
  - 终态推送统一走 `dsw_scheduler._claim_dataflow_notify(job_id)`（Redis `SET NX dataflow:notified:{job_id}`）
    跨"在线线程 vs 对账线程"去重，谁先到谁推、只推一次。
- **对账兜底**：容器重启后 `dsw_scheduler` 扫在途 job（stage∈{STAGE1,STAGE2}）续跑/续判——照搬 `_transfer_loop`/reconcile。

### 4.2 与云 API 迁移（MGW/DMS）的本质差异（实现时要改的点）
| 维度 | 云 API 迁移 (现有) | 本链路 (SSH shell 编排) |
|--|--|--|
| 执行主体 | 云迁移服务端跑 | SGP ECS 上的 ossutil/rsync 进程跑，bot 只 SSH 遥控 |
| "提交" | `create_job/update_job` OpenAPI | `ssh 'nohup ... & echo pid'` |
| "poll" | `get_job` 拿云端 status/bytes | `ssh 'cat rc'` + `kill -0 pid` + 可选 `du -sb` 估进度 |
| 终态枚举 | `IMPORT_JOB_FINISHED/INTERRUPTED` 等云状态串 | **进程退出码**（ossutil 0/非0；rsync 0,24 成功 / 23,255,127 失败） |
| 幂等 | 云 job_name 幂等 + NX 锁 | ossutil `-u`+checkpoint / rsync `-a` 增量 + rc marker + NX 锁 |
| 凭证 | STS role / 静态 TOS AK | **SSH 私钥**(Fernet) + SGP ECS 上 ossutil 自己配的 AK + 泰国 sudoers |
| 进度量 | 云返回 bytes/objects | 无结构化回传 → 自算(prefix_size vs du) 或只给活着/百分比粗值 |
| 失败面 | 云侧 | 多了：SSH 断连(255)、专线抖动、磁盘满、sudo/权限、泰国机不可达 |

> **没有云迁移 OpenAPI**，所以 `engine_*.py` 换成一个 `engine_ssh.py`（封装 paramiko 连接 + 起任务 + 读 marker），
> orchestrator 的 `start_cross/poll_once` 换成 `start_stage/poll_stage`，其余（状态机/Redis/卡片/handler/对账/NX 闸门）
> 几乎不动。

---

## 5. 幂等与失败恢复

- **job_id**：`sgp-<hash(source_oss_uri, thai_dest, 当天)>`。当天重复提交命中同一 job（同现有）。
- **段1完成、段2失败只重跑段2**：状态机记 `stage1_rc=0` 落盘后才进 STAGE2；retry 时若 `stage1_rc==0`
  **跳过段1直接重起段2**（rsync `-a` 增量，只补没推完的）。别无脑从 NEW 重来——那会重下段1（虽然 `-u`
  也安全但浪费专线流量/时间）。这是与现有 retry（整个重来）唯一要加的分支逻辑。
- **段1本身重跑安全**：ossutil `-u` + `--checkpoint-dir` → 只补缺的 + 大文件断点续。
- **容器重启后判定远端在跑/已完成**：bot 无状态，靠 SSH `cat $RUN/stageN.rc`：
  - 有 rc 文件 → 已结束，内容即成败 → 直接落终态；
  - 无 rc 且 `kill -0 pid` 成功 → **仍在跑** → 继续轮询（不重起！否则起第二个 ossutil）；
  - 无 rc 且 pid 已不在 → 异常死亡 → 判失败、可 retry。
  这套完全对应现有 `refresh()`"点查询即自愈"。**关键**：起任务前也要 NX 锁 + 查 rc/pid，避免重启后
  对账线程和用户手点同时重起。

---

## 6. 上线前踩坑 / 安全清单

1. **[安全·最高] `sudo rsync` 无限参 = 泰国机完整 root**。首选把目标目录 chown 给 transferuser、**彻底不用 sudo**；
   否则 sudoers 限到 `--server *` + authorized_keys 限 `from=<SGP IP>,command=...`。
2. **SSH 私钥** Fernet 加密存、内存加载、绝不落盘/打印；paramiko 别用 AutoAddPolicy，固定 host key。
3. **ossutil 版本**：2.x 必须配 region（V4 签名），否则段1 直接鉴权失败；确认 SGP ECS 上 `ossutil version` + AK 配置。
4. **`-u` 的 HEAD 请求成本**：海量小文件时每对象一次 HEAD，有请求费+耗时；大批量重复跑可加 `--snapshot-path` 提速。
5. **退出码语义**：rsync `24` 当成功、`23/255/127` 当失败；ossutil 非 0 即失败；别只看日志文本。
6. **remote redirection 要引号**：`ssh host 'cmd > log 2>&1 &'`，否则 log 落错地方 / 任务不后台化。
7. **不在 3s 飞书回调里做 SSH**（会超时→重投→去重吞掉→卡死），同 auditor HIGH-1 教训；一律后台线程。
8. **paramiko 大输出死锁**：本模式用短命令 + marker 规避；若确要读长输出，先 drain stdout/stderr 再取 exit。
9. **专线/CEN 抖动、SGP 磁盘满、泰国机不可达** 都要能落 FAILED + 可 retry；`du` 估进度别把探测失败当迁移失败。
10. **别复用 `transfer:*` Redis 命名空间**，用独立 `ssh:transfer:*`，job_id 前缀 `sgp-` 并入 `_h_query_progress_by_id` 分派。

---

## 7. 待用户/dev 提供的前置信息清单

- **SGP ECS**：SSH user、port、bot→ECS 的私钥（Fernet 加密后给）；ECS 上 ossutil 版本 + 已配的 AK/region；
  `/mnt/sgp_oss` 挂载方式（ossfs？直接是 wuji-sing 桶的挂载点？影响段1"完成"判定——若目标是 ossfs 挂载，
  写完 ≠ 上传完，需确认 ossfs flush 语义）。**这条要重点核**。
- **泰国服务器**：host/user、bot 或 SGP 能否直连、目标目录路径、是否真的需要 sudo（能否 chown 给 transferuser 免 sudo）、
  rsync 绝对路径、SGP→泰国的私钥。
- **源**：`oss://wuji-data-tran` 所在 region、bot 侧是否需要 prefix 白名单/审批阈值（复用 `TRANSFER_APPROVAL_TB`?）。
- **网络**：SGP↔泰国走公网还是专线？带宽？影响 rsync `--bwlimit` 要不要设。
- 段1目标到底是本地盘还是 wuji-sing 桶的挂载——决定链路是"两跳对象存储"还是"OSS→本地→远端"，语义差异大。

---

### 附：关键出处清单
- ossutil cp/-u/--jobs：alibabacloud.com/help/en/oss/developer-reference/{upload-objects-6, download-objects-5, view-options}
- ossutil 断点/snapshot：alibabacloud.com/help/en/oss/developer-reference/breakpoint-file-resumable
- ossutil 1.0 vs 2.0：alibabacloud.com/help/doc-detail/2804546.html ; .../ossutil-2-0-new-features ; .../configure-ossutil
- ossutil 报告文件/退出：alibabacloud.com/help/en/oss/developer-reference/faq-9
- rsync sudo 安全：systutorials.com/how-to-write-to-remote-side-with-sudo-in-rsync/ ; strugglers.net/~andy/mothballed-blog/2021/04/10/ ; raimue.blog/2016/09/25/
- rsync 退出码：lxadm.com/rsync-exit-codes/ ; bluebones.net/2007/06/rsync-exit-codes/
- paramiko 长任务/死锁：docs.paramiko.org/en/latest/api/channel.html ; github.com/paramiko/paramiko/issues/1208
- nohup/后台/远端重定向：groups.google.com/g/comp.os.linux.misc/c/9wqVvfO0zvM
