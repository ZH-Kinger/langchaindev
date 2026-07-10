# 开发方案：杭州 OSS → SGP 中转 → 泰国服务器（SSH 迁移链）

> planner 产出。事实源：researcher `docs/collab/research/thailand-transfer-chain.md`（结论带出处）
> + dev 真机验证（前置全绿）。本文只写 `docs/collab/planning/`，不碰源码/测试。
> 落地由 dev 执行；tester 补测、auditor 过闸后方可 commit（notes.md 提交闸门硬规）。

---

## 0. 一句话

新增第 5 条搬运链：飞书 bot(容器) 用 **paramiko** SSH 到新加坡 ECS，遥控两段 shell（段1 `ossutil` 拉杭州
OSS 到本地挂载盘、段2 `rsync` 推到泰国服务器），**bot 自己不搬数据**。照搬 `core/transfer/` 状态机 + Redis
job + `run_to_completion` 轮询 + `on_update` 推卡骨架，把"提交云 API + poll 云 status"换成"SSH 起 `nohup`
后台命令 + SSH 轮询 pid/rc marker 文件"。

链路（两段都在 SGP ECS `43.98.203.59` 上跑）：
```
段1 STAGE1: oss://wuji-data-tran/<prefix>/ --ossutil cp -r -u--> /mnt/sgp_oss/<prefix>/   (杭州→SGP 走 CEN)
段2 STAGE2: /mnt/sgp_oss/<prefix>/         --rsync--> wuji@203.156.3.194:40002:<泰国目标目录>/<prefix>/
```

状态机：`NEW → STAGE1 → STAGE2 → DONE | FAILED`（对应现有 `NEW→SINKING→CROSSING→DONE|FAILED` 两段式）。
Redis 独立命名空间 `ssh:transfer:*`；job 前缀 `sgp-`；三入口（飞书卡片=主 + Agent 工具 + CLI），与现有迁移一致。

---

## 1. 架构决策（已定，dev 无需再问）

- **传输方式**：paramiko（纯 Python SSH），私钥 Fernet 加密、运行时 `decrypt()` → `RSAKey.from_private_key(io.StringIO)`
  只在内存加载、绝不落盘/打印。**不用** subprocess 调 ssh（容器无 openssh-client，且 key 要落盘）。
- **执行模式**：fire-and-forget `nohup ... &` + pid 文件 + rc marker 文件 + 轮询短命令。避 paramiko 大输出
  死锁、避容器重启丢 channel。3s 飞书回调只读、后台线程干活（同 auditor HIGH-1 教训——**绝不在 3s 回调里 SSH**）。
- **完成判定**：进程退出码写进 rc marker。ossutil 非 0 = 失败；rsync 0/24 = 成功，23/255/127/其它非 0 = 失败。
- **幂等/续跑**：段1 `ossutil -u` + `--checkpoint-dir` 增量+断点；段2 `rsync -a` 增量。job_id=`sgp-hash(源,目的,当天)`。
  容器重启后 bot 无状态，靠 SSH `cat rc` / `kill -0 pid` 重判（对应 `refresh()` 点查询即自愈）。
- **段1 完成后段2 失败只重跑段2**：状态机记 `stage1_rc==0` 落盘后才进 STAGE2；retry 时若 stage1 已成功则跳过段1
  直接重起段2（省 CEN 流量）。这是与现有 retry（整体重来）唯一要加的分支。
- **200830 规避**：确认后原地替换卡做成 schema 2.0 纯展示（`progress_card_v2`），retry 结果卡 1.0→1.0，照 #33。
- **终态去重**：统一走 `dsw_scheduler._claim_dataflow_notify(job_id)`（`SET NX dataflow:notified:{job_id}`），
  在线线程 / 对账 / 按 ID 查询三路径谁先到谁推、只推一张。
- **created_by**：job 存发起人 open_id，进度/结果卡优先推 `created_by`、空则回落配置频道（照 #41/#45）。

---

## 2. 待定项（dev 补齐再定稿——方案已按参数/分支留口）

| # | 待定 | 方案处理 |
|---|------|---------|
| Q1 | 泰国**目标目录**（原需 sudo 的"指定目录"）具体路径 | 配置项 `THAI_DEST_ROOT`；段2 落 `<THAI_DEST_ROOT>/<prefix>/` |
| Q2 | 段2 **rsync 完整命令/参数**（`-a`/`--info=progress2`/`--bwlimit`/是否 `--rsync-path`） | engine_ssh 里参数化，默认 `rsync -a --info=progress2`，`--bwlimit`/`--rsync-path` 走配置开关 |
| Q3 | **sudo 处理二选一** | **方案 A（dev 推荐）**：泰国目标目录 chown 给 `wuji`、彻底免 sudo，最安全。**方案 B（次选）**：`--rsync-path="sudo rsync"` + 泰国 NOPASSWD sudoers，须 `from=<SGP IP>` + `command="/usr/bin/rsync --server *"` 收紧。engine_ssh 用 `THAI_RSYNC_SUDO` 开关切换两条路 |
| Q4 | `/mnt/sgp_oss` 挂载语义 | researcher 已核为 ossfs2 挂 wuji-sing 桶；**若挂载点，段1 写完 ≠ 上传完**，需确认 ossfs flush 语义（真机验收 T10 覆盖）。dev 已真机验证挂载存在，flush 语义在 T10 观测 |
| Q5 | 审批阈值是否复用 `TRANSFER_APPROVAL_TB` | 建议新增独立 `SSH_TRANSFER_APPROVAL_TB`（默认 1 TB），语义/单位与现有一致但可独立调 |

> Q1/Q2 阻塞 T10（真机验收），不阻塞骨架编码（T1–T3 用占位默认值即可 dry-run）。Q3 决策影响泰国侧一次性配置
> 与 engine_ssh 段2 命令，dev 拍 A 则 T3 免 sudo 分支、无泰国 sudoers 步骤。

---

## 3. 配置项清单（settings + `.env.example` + `_REQUIRED_FIELDS`）

新增 `config/settings.py` 字段，凡 required 的登进 `Config._REQUIRED_FIELDS` 带 impact 串（CLAUDE.md 约定）：

| 字段 | 示例/默认 | 说明 |
|------|-----------|------|
| `SSH_TRANSFER_ENABLED` | `false` | 总开关，默认关（同其它迁移链 `*_ENABLED` 惯例） |
| `SGP_SSH_HOST` | `43.98.203.59` | 新加坡 ECS |
| `SGP_SSH_PORT` | `22` | |
| `SGP_SSH_USER` | `root` | |
| `SGP_SSH_KEY_ENC` | (Fernet 密文) | bot→SGP 私钥（`/root/bot_sgp_rsa` 内容 Fernet 加密后入 env/Redis）；**绝不明文** |
| `SGP_SSH_HOST_KEY` | (可选 known_hosts 指纹) | paramiko 固定 host key，**禁用 AutoAddPolicy** |
| `SGP_OSS_MOUNT` | `/mnt/sgp_oss` | 段1 本地落地根（ossfs2 挂 wuji-sing） |
| `SGP_WORK_DIR` | `/var/run/tr` | 每 job 的 pid/rc/log marker 工作目录根 |
| `SGP_OSSUTIL_JOBS` | `30` | ossutil `--job` 并发（2.2.2 是单数） |
| `SOURCE_OSS_BUCKET` | `wuji-data-tran` | 源桶（也可由用户输入路径带出） |
| `SOURCE_OSS_REGION` | `cn-hangzhou` | 源桶 region（估算/校验用；ossutil 的 AK/region 在 ECS 上已配） |
| `THAI_HOST` | `203.156.3.194` | 泰国服务器 |
| `THAI_PORT` | `40002` | |
| `THAI_USER` | `wuji` | |
| `THAI_DEST_ROOT` | (Q1 待定) | 段2 目标根目录 |
| `THAI_RSYNC_SUDO` | `false` | Q3 开关：false=免 sudo(方案 A) / true=`--rsync-path="sudo rsync"`(方案 B) |
| `THAI_RSYNC_BWLIMIT` | (可选) | rsync `--bwlimit`，SGP↔泰国带宽受限时设 |
| `SSH_TRANSFER_APPROVAL_TB` | `1` | 审批阈值（Q5，建议独立于 `TRANSFER_APPROVAL_TB`） |
| `SSH_TRANSFER_CHAT_ID` | (可选) | 进度/结果卡配置频道，空则回落 `FEISHU_CHAT_ID` |

> 凭证复用：SGP ECS 上 ossutil 自己配的 AK/region（不经 bot）。bot 只持有 SSH 私钥。泰国侧无凭证（方案 A）或 sudoers（方案 B）。

---

## 4. 部署差异（重要 —— 非普通 deploy）

加 `paramiko` 到 `requirements.txt` → **必须** `docker compose up -d --build` 重建镜像，
**不是** `deploy.ps1` 的普通 restart（deploy.ps1 会检测 requirements 变化并 warn，但重建要手动执行）。
改 `.env` 新增上述字段后还需 `up -d --force-recreate`（restart 不重载 env_file，见 MEMORY deploy_env_reload_gotcha）。

---

## 5. Epic → Task（有序，带验收/依赖/owner/工作量/优先级）

Epic：**SSH 迁移链（杭州 OSS→SGP→泰国）** — 骨架先跑通段1 dry-run，再段2，再卡片，最后收尾真机验收。

> 工作量：S≈半天 / M≈1天 / L≈2天+。优先级：P0 骨架必需 / P1 核心功能 / P2 收尾增强。
> owner：dev 编码，tester 补测，auditor 复审（每个编码 task 完成后走标准流水线）。

### T0 — paramiko 依赖 + 私钥 Fernet 注入 + engine 连接层
- **owner**: dev
- **产出**: `requirements.txt` 加 `paramiko`；`core/ssh_transfer/engine_ssh.py` 的连接层：`_client()`
  从 `SGP_SSH_KEY_ENC` Fernet `decrypt()` → `RSAKey.from_private_key(io.StringIO)` 内存加载，
  固定 host key（禁 AutoAddPolicy），封装 `run(cmd) -> (rc, stdout, stderr)`（短命令、先 drain 再取 exit）。
- **验收**:
  - `pip install` 后 `import paramiko` 通；本地/容器 `engine_ssh._client()` 能连 SGP 并 `run("echo ok")` 返 `(0,"ok\n","")`。
  - 私钥任何路径都不落盘、不进日志（grep 源码/日志无 PEM、无明文 key）。
  - 缺 `SGP_SSH_KEY_ENC` 时明确报错，不静默降级。
- **依赖**: 无
- **工作量**: M
- **优先级**: P0

### T1 — engine_ssh 段1/段2 起任务 + marker 轮询
- **owner**: dev
- **产出**: `engine_ssh` 的 `start_stage1/start_stage2`（`ssh 'nohup bash -c "... > log 2>&1; echo $? > rc" & echo $! > pid'`，
  **redirection 全程引号在远端**）+ `poll_stage(job_id, stage) -> {status, rc, alive, bytes?}`（`cat rc` / `kill -0 pid` /
  可选 `du -sb` 估进度）。段1=ossutil `cp -r -u --job N --checkpoint-dir`；段2=rsync（Q2/Q3 参数化）。
- **验收**:
  - dry-run（`--dry-run`/占位小 prefix）：起任务 SSH 秒回、pid/rc/log 三 marker 在 `SGP_WORK_DIR/<job_id>/` 生成。
  - `poll_stage`：无 rc+pid 在→RUNNING；rc=0→DONE；rc≠0→FAILED；无 rc+pid 亡→FAILED（异常死亡）。
  - rsync 退出码映射正确：0/24→成功，23/255/127→失败（单测覆盖映射表）。
  - 重复起任务前查 rc/pid，不起第二个进程。
- **依赖**: T0
- **工作量**: L
- **优先级**: P0

### T2 — paths + orchestrator 状态机
- **owner**: dev
- **产出**: `core/ssh_transfer/paths.py`（解析用户输入 `oss://wuji-data-tran/<prefix>/` → `Plan(source_prefix, thai_dest, ...)`，
  dir-only 尾斜杠校验）；`core/ssh_transfer/orchestrator.py`（照搬 transfer：`STAGE_NEW/STAGE1/STAGE2/DONE/FAILED`、
  `ssh:transfer:job:{job_id}` 30天TTL、`get_job/_save`（**每次刷 `updated_ts`**——auditor 曾抓 bucket_transfer 漏刷）、
  `job_id=sgp-hash(源,目的,当天)`、`create_job_record`（幂等分支回填 `created_by`，照 #45）、
  `estimate_source`（复用 `tools/aliyun/oss._prefix_size`）、`needs_approval`、`start_stage/poll_stage`、
  `refresh`、`run_to_completion(on_update)`：起段1→轮 rc→段1成功(stage1_rc=0 落盘)起段2→轮 rc→终态）。
- **验收**:
  - CLI `python -m core.ssh_transfer.cli plan oss://wuji-data-tran/foo/` 打印 source/dest/估算/是否超阈值，不建 job。
  - job_id 当天幂等：同源同目的两次 `create_job_record` 命中同一条。
  - `run_to_completion` 两段推进：段1 rc=0 才进 STAGE2；段1 rc≠0 直接 FAILED 不进段2。
  - `refresh` 在 stage∈{STAGE1,STAGE2} 时重查 marker 落库（重启自愈）。
  - retry 分支：stage1_rc==0 时重跑只从 STAGE2 起，不重下段1。
- **依赖**: T1
- **工作量**: L
- **优先级**: P0

### T3 — CLI 入口
- **owner**: dev
- **产出**: `core/ssh_transfer/cli.py`（`plan`/`apply`/`status`，dry-run 默认，`--force` 越阈值，`--overwrite` 同名策略），照 transfer/cli.py。
- **验收**: `plan` 只估算不建 job；`apply` 建 job + `run_to_completion` 打印各 stage；`apply` 超阈值无 `--force` 退出提示审批；
  `status <sgp-xxx>` 打印 stage/源目的/大小/error。
- **依赖**: T2
- **工作量**: S
- **优先级**: P0（骨架端到端验证入口，先于卡片）

### T4 — cards 卡片四件套
- **owner**: dev
- **产出**: `core/ssh_transfer/cards.py`：`entry_card`（源 OSS 路径输入 + 同名策略 + Q1 若需选目标目录）、
  `confirm_card`（估算回显 + 审批阈值，超阈值 orange + 仅 admin）、`progress_card_v2`（**schema 2.0 纯展示**，避 200830，
  含 job_id + "发送 `查询进度 sgp-xxx`" 提示行，照 #49）、`result_card`（DONE/FAILED，FAILED 带 `retry_ssh_transfer` 按钮）。
- **验收**: 各卡 dict 结构合法（可复用 `test_dataflow_progress_card_v2` 式断言）；progress_card_v2 schema 2.0 无 form/button；
  缺可选字段不 KeyError；confirm 超阈值染色 + admin 门。
- **依赖**: T2
- **工作量**: M
- **优先级**: P1

### T5 — messages 意图识别
- **owner**: dev
- **产出**: `core/feishu_bot/messages.py`：新增 `_is_ssh_transfer_intent`（泰国迁移动作词 + `oss://wuji-data-tran` 路径，
  与 `_is_transfer_intent` 区分，避免误抢跨云迁移）→ `_handle_ssh_transfer_intent` 发 `cards.entry_card`。
  `_h_query_progress_by_id` 的 `_JOB_ID_RE` 并入 `sgp-` 前缀（与 tr-/cpfs-/vepfs- 并列）。
- **验收**: 泰国迁移话术命中新意图、不误命中现有 tos→oss 跨云迁移；非泰国路径不误触；`sgp-xxx` 按 ID 查询路由到本链。
- **依赖**: T4
- **工作量**: M
- **优先级**: P1

### T6 — actions handler（confirm/query/retry）
- **owner**: dev
- **产出**: `core/feishu_bot/actions.py` 注册三 handler：
  - `_h_submit_ssh_transfer`（entry→confirm_card）
  - `_h_confirm_ssh_transfer`：`ssh:transfer:launch:{job_id}` NX 锁(ex=120) + `launched` 标记防连点；起 `Thread(daemon)`
    跑 `run_to_completion`；同步 3s 内**只回卡不做 SSH**；2.0 确认卡→回 `progress_card_v2`(2.0→2.0)；超阈值仅 admin。
  - `_h_query_ssh_transfer`：`_async_refresh_and_push` 后台真 refresh，sync 只读 `get_job` 秒回。
  - `_h_retry_ssh_transfer`：重置 `stage=NEW`（或 stage1_rc==0 时 stage=STAGE2）/`launched=False`/`error=""`，
    清 `ssh:transfer:launch:{job_id}` + `dataflow:notified:{job_id}` 两键（照 #37 四链 retry 教训）。
  - `on_update` 终态推送经 `_claim_dataflow_notify` 去重，推 `created_by`→配置频道。
  - `_cfg_ssh_chat()` 读 `SSH_TRANSFER_CHAT_ID` 回落 `FEISHU_CHAT_ID`。
- **验收**: 连点确认只起一个后台线程（NX+launched）；确认卡 2.0→2.0 不触 200830；查询 3s 内无 SSH（sync 只读）；
  retry 清两键 + stage1_rc==0 只重段2；终态卡至多一张（NX 闸门）；进度/结果卡优先推发起人。
- **依赖**: T4, T5, T2
- **工作量**: L
- **优先级**: P1

### T7 — Agent 工具
- **owner**: dev
- **产出**: `manage_ssh_transfer` 工具（`plan`/`apply`/`status`），注册进 `tools/__init__.py` `ALL_TOOLS` + `TOOL_GROUPS`
  新组（如 `ssh_transfer`）+ `INTENT_DESCRIPTIONS` 对应键；子类 `BaseOpsTool`。
- **验收**: `TOOL_GROUPS` 键在 `ALL_TOOLS`（import 时校验不抛 ValueError）；`INTENT_DESCRIPTIONS` 键与 `TOOL_GROUPS` 一致；
  工具三动作与 CLI 行为一致；Agent 路由能命中。
- **依赖**: T2
- **工作量**: M
- **优先级**: P2

### T8 — 对账兜底（scheduler reconcile）
- **owner**: dev
- **产出**: `dsw_scheduler` 的在途对账并入 `ssh:transfer:*`：扫 stage∈{STAGE1,STAGE2} 的 job，`refresh` 续判，
  终态经 `_claim_dataflow_notify` 补推（照 `_reconcile_dataflow_once`）。
- **验收**: 容器重启后在途 job 被对账续跑/续判并补推终态卡；与在线线程/查询三路径共用闸门只推一张。
- **依赖**: T2, T6
- **工作量**: M
- **优先级**: P2

### T9 — 泰国侧一次性配置（部署/运维步骤，非源码）
- **owner**: dev（拍 Q3 决策后执行；可请 auditor 复核安全收紧）
- **产出**: 方案 A：泰国 `THAI_DEST_ROOT` chown 给 `wuji`，`wuji` 自有目录、免 sudo。
  方案 B（若拍 B）：`/etc/sudoers.d/rsync-bot`（`visudo -f` 校验，0440，绝对路径 `/usr/bin/rsync --server *`）+
  authorized_keys `from="43.98.203.59",command=...` 限来源 IP。
- **验收**: 方案 A：SGP `rsync` 到 `THAI_DEST_ROOT/<prefix>/` 无 sudo 成功、属主 wuji。方案 B：NOPASSWD 生效、限到 `--server *`、
  非 SGP IP 拒。
- **依赖**: Q3 决策
- **工作量**: S
- **优先级**: P1（阻塞 T10 真机段2）

### T10 — 真机验收（端到端）
- **owner**: dev（tester 协助断言，auditor 复审安全）
- **产出**: 镜像重建（paramiko）+ `.env` 填真值（Q1/Q2 定稿）+ 部署后小 prefix 端到端跑一遍：飞书发起→确认→段1→段2→完成卡。
- **验收**:
  - 段1：小 prefix ossutil 拉到 `/mnt/sgp_oss/<prefix>/`，rc=0；**确认 ossfs flush 语义**（写完=上传完，Q4）。
  - 段2：rsync 推到泰国 `THAI_DEST_ROOT/<prefix>/`，退出码 0/24；泰国侧文件校验（数量/大小）。
  - 幂等：重跑命中同 job；段1 完成后段2 失败 retry 只重段2。
  - 重启自愈：跑中重启容器，点查询/对账续判正确。
  - 终态卡至多一张、推发起人。
- **依赖**: T6, T9, 全部编码 task + tester/auditor 过闸
- **工作量**: M
- **优先级**: P1（功能上线闸门）

---

## 6. 建议开发顺序

1. **骨架跑通段1 dry-run**：T0（依赖+连接层）→ T1（起任务+marker，先段1）→ T2（状态机）→ T3（CLI）。
   到此 CLI `apply` 能端到端 dry-run 段1、观测 marker/rc/自愈——**最小可验证闭环**。
2. **补段2**：T1/T2 里段2 打通（rsync + 退出码映射 + 段1成功才进段2 + retry 只重段2）。CLI 端到端两段跑通。
3. **卡片入口**：T4（卡片）→ T5（意图）→ T6（handler，含 NX/launched/200830/created_by/闸门）。飞书主入口打通。
4. **收尾**：T7（Agent 工具）、T8（对账兜底）、T9（泰国侧配置）→ T10（真机验收）。

> 并行：T4 卡片 可与 T3 CLI 并行（都只依赖 T2）。T7 Agent 工具 可在 T2 后随时插。T9 泰国配置 只等 Q3 决策、可提前做。
> 串行硬依赖：T0→T1→T2 是骨架主链，不可跳。T10 必须全部编码 + 过闸后。

---

## 7. 风险 / 安全清单（researcher 10 条 + sudo 决策）

| # | 风险 | 缓解 | 归属 task |
|---|------|------|----------|
| R1 | **[最高] `sudo rsync` 无限参 = 泰国机完整 root**；攻破 SGP 即得泰国 root | **首选方案 A（chown 免 sudo）**；若 B 则 sudoers 限 `--server *` + authorized_keys `from=<SGP IP>,command=` | T9/Q3 |
| R2 | SSH 私钥泄漏 | Fernet 加密存、内存加载、绝不落盘/打印；禁 AutoAddPolicy、固定 host key | T0 |
| R3 | ossutil 2.x 漏配 region（V4 签名）→ 段1 鉴权失败 | ECS 上 ossutil 已配 region+endpoint（真机已验 2.2.2）；T10 复核 | T1/T10 |
| R4 | `-u` 海量小文件每对象一次 HEAD（请求费+耗时） | 大批量重复跑可加 `--snapshot-path` 提速 | T1 |
| R5 | 退出码语义误判 | rsync 0/24 成功、23/255/127 失败；ossutil 非 0 失败；**不看日志文本判成败** | T1 |
| R6 | remote redirection 落错地方/任务不后台化 | `ssh 'cmd > log 2>&1; echo $? > rc &'` 整条引号在远端 | T1 |
| R7 | 3s 飞书回调里做 SSH → 超时被重投 → 15s 去重吞成 `{}` → 卡死 | 3s 回调只读 Redis、SSH 一律后台线程（auditor HIGH-1 教训） | T6 |
| R8 | paramiko 大输出死锁 | 短命令 + marker 模式规避；确要读长输出先 drain stdout/stderr 再取 exit | T0/T1 |
| R9 | CEN 抖动 / SGP 磁盘满 / 泰国不可达 | 落 FAILED + 可 retry；`du` 估进度探测失败不当迁移失败 | T1/T2 |
| R10 | 复用 `transfer:*` 命名空间串键 | 独立 `ssh:transfer:*`，job_id 前缀 `sgp-` 并入 `_h_query_progress_by_id` 分派 | T2/T5 |
| R11 | **ossfs 挂载 flush 语义**：段1 写完 ≠ 上传完（Q4） | T10 真机观测；若非同步 flush 需在 poll_stage 加 flush 校验 | T10/Q4 |
| R12 | 重启后对账线程 + 用户手点同时重起 → 起第二个进程 | 起任务前 NX 锁 + 查 rc/pid（对应 `refresh` 只读自愈） | T2/T6/T8 |

---

## 8. 开放问题（交 dev 去问用户/拍板）

1. **Q3 sudo 决策**：拍方案 A（chown 免 sudo，planner/researcher 推荐）还是 B（NOPASSWD sudoers）？决定 T9 步骤与 engine_ssh 段2 命令。
2. **Q1 泰国目标目录** 具体路径（`THAI_DEST_ROOT`）？
3. **Q2 rsync 参数**：是否要 `--bwlimit`（SGP↔泰国带宽/是否专线）？是否保留 `--info=progress2`？
4. **Q4 `/mnt/sgp_oss` flush 语义**：ossfs2 写完是否即上传完成、还是异步 flush？影响段1"完成"判定是否要额外校验。
5. **Q5 审批阈值**：复用 `TRANSFER_APPROVAL_TB` 还是独立 `SSH_TRANSFER_APPROVAL_TB`？（建议独立）
6. 源桶是否需 prefix 白名单/限定只能迁 `wuji-data-tran` 某些前缀？

---

## 9. 落板建议（dev 与用户确认后建 task）

以上 T0–T10 建议逐条建成任务板 task，owner=dev（tester/auditor 各 task 完成后接力），
blockedBy 按第 5 节依赖列填。P0（T0–T3）先开工；P1（T4–T6、T9、T10）次之；P2（T7、T8）收尾。
Q1–Q5 建议先建成 pending 的"待用户确认"条目，解了再解锁对应 task。
