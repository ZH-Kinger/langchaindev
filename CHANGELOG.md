# Changelog

所有版本变更记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased]

### Fixed

- **临时 AK：只勾「上传」的凭证拿不到桶信息，客户端探桶即 403**（`core/temp_ak_issuance/policy.py`）

  线上单（主体「元客」）：策略里只有一条 `PutObject` 语句，连 `GetBucketInfo` 都没有。
  ossutil / SDK / 控制台在上传前普遍会先探一次桶 → 403 → 现场表现成「凭证发了但什么都
  干不了、策略里看不到任何路径」，被误判成「权限策略没建好」。

  根因不是写错了，是**那个边界从未被验证过**：桶信息语句的触发条件是 `aa879f4`
  「对齐权威模板」时推出来的，而那份模板（`tempak-nuoyiteng-7df6a7`）是 read+write+download
  三项全勾的，**根本不含 write-only 这个场景**。元客是第一单 write-only。

  改成「勾了任何一项就给桶信息」。正交性仍然守住：桶信息只含三个只读元数据动作，
  **不含 `ListObjects`** —— 只勾上传的外部方看不到桶里有什么。实测 7 种 caps 组合里
  6 种生成的策略逐字节不变，只有 write 单选变了。

- **临时 AK：桶名填成目录时静默发出废凭证**（`core/temp_ak_issuance/`）

  线上单（主体「maxinsights」）：申请人填 `third-party-data/maxinsights/`，而
  `third-party-data` 是 `wuji-bucket-hangzhou` 里的一个**目录**、不是桶。解析层把第一段
  当成桶名，于是策略指向一个不存在的桶 —— 凭证是废的，但流程一路绿灯、评论正常下发。
  （同类问题在 SSH 迁移链也出现过三次，见 `sgp-841b88a7b0dd` 等。）

  新增 `bucket_missing_reason()`，**只在能确定「桶不存在」（`NoSuchBucket`）时拦**，
  并在报错里直接点出最常见成因。权限不足 / 网络抖动 / 没配 AK 一律放行 —— 这条拦的是
  **笔误**不是越权，拦错了会挡住正常发放，默认方向与 caps/审批那些门禁相反。

  同时把 `_probe_region_once` 改成模块**唯一的出网点**（返回 `(region, 错误码)`），
  地域探测与桶存在性检查共用 —— 多一个出网入口就多一个会被漏桩的坑，而漏桩的后果是
  单测拿生产 AK 打真 API。

  > ⚠️ **流程说明（留痕，别删）**：上面这两处改动实际包含在提交 `af3397d` 里，
  > 而那条提交的信息只讲「卡片回调取不到操作人身份」，**完全没提 temp_ak 的改动**。
  > 原因是当时用了 `git add -A` 而非逐文件挑，把不相干的改动裹了进去，
  > 并因此**跳过了本项目「改动须经 auditor 审计通过才能 commit」的闸门**、直接随部署上线。
  > 事后已补做审计。查 git 历史时请注意：`af3397d` 的实际内容大于它的提交信息。

### Security

- **临时 AK 凭证：桶不在映射表时实时探测地域**（`core/temp_ak_issuance/orchestrator.py`）

  线上现象是「凭证发了但用不了，像是权限策略没建好」。实际不是：策略是好的 —— RAM policy 的
  ARN 是 `acs:oss:*:*:{bucket}`，**region 位本就是通配符，地域根本不参与策略构造**。

  真实因果链：桶 `wuji-rl-dataset` 既不在 `permsync.BUCKET_MAP`（只有 4 个桶）也不在
  `TEMP_AK_*_BUCKET_MAP` → `resolve_bucket()` 兜底返回空 region → 凭证正文的
  「地域 / 外网 Endpoint / 桶域名」三行退化成「未知」→ 使用方随手用了默认 endpoint →
  OSS 回 403 `must be addressed using the specified endpoint`。

  **要记住的是这一条**：这个 403 与「没有权限」的 403 长得一模一样，所以现场第一反应必然是
  查策略、查 RAM、查时间窗 —— 方向整个跑偏。以后再遇到「凭证发了但 403」，先看正文里的
  Endpoint 对不对，再查权限。

  修法：映射表查不到时用**该账号自己的凭证**实时探一次 `GetBucketLocation`。新桶不必先维护
  映射表。几条刻意的设计：

  - **探不到一律返回空，绝不回退默认地域。** 一个自信的错地域比「未知」更坏 —— 使用方会照着
    连，再拿到那个与「没权限」同形的 403。返回空则正文如实写「未知，请按控制台自查」
  - **不复用 `tools.aliyun.oss.detect_bucket_region`**：它按 `open_id` 走 client factory 取默认
    账号凭证（读桶地域的权限是按账号授的，拿 A 的 AK 问 B 的桶只会 403），且失败会静默回退默认
    地域 —— 两条都与上面的原则冲突
  - **探测异常绝不让发放失败**：地域只影响正文展示，policy/issuer 完全不用它
  - 成功缓存 24h、**失败只缓存 300s**：grant 一旦落库 region 就写死了，不能让一次瞬时抖动把某个
    桶永久钉成「未知」
  - 失败日志只记 `status/code/request_id`，**绝不记 `e.body`** —— `SignatureDoesNotMatch` 的
    body 里带 AccessKeyId 与 StringToSign
  - 探测前过桶名正则、探测后过地域正则：前者挡掉中文展示名（`oss2.Bucket()` 构造器会在 try 之外
    抛 ClientError），后者挡掉 `region_from_endpoint` 对意外域名吐出的垃圾串（`data.example.com`
    → `data`，正文会写成「地域：data」）

- **单测网络兜底：桩掉桶地域探测**（`tests/conftest.py`）

  上面那个改动会让 `resolve_bucket()` 在 map-miss 时发真实 OSS 请求，而**多个既有用例正好走
  map-miss**。原有的 `requests.post/get` 兜底**挡不住它** —— oss2 走的是
  `requests.Session().request(...)`，不经过被桩的模块级函数。而 `.env` 里是真 AK，于是单测会拿
  生产凭证对真桶发签名请求。正反对照实测：停用兜底时 `resolve_bucket("wuji-rl-dataset")` 返回
  `("cn-hangzhou", ...)`（真出网了），启用后返回 `("", ...)`。

  桩的是最内层的 `_probe_region_once`，缓存/校验/账号隔离这些逻辑仍被测到。同时清模块级缓存
  —— 它跨文件残留会导致顺序相关的偶发失败。

### Security

一轮安全加固，覆盖入站鉴权、构建产物、管理员门禁与数据保留期。原则是把散落的判定收敛成单一入口，并把默认失败方向统一改成 fail-closed。

- **`/feishu/card_action` 补上验证 token 校验**：`/feishu/event` 一直有这道门，卡片回调路由没有。抽出 `_extract_request_token()` / `_token_verified()` 供两条入站路由共用，避免以后再长出第三个没有校验的路由。`hmac.compare_digest` 两侧先 `encode("utf-8")` —— 该函数对含非 ASCII 的 str 会抛 `TypeError`，而 token 取自请求体，不编码就会让畸形输入变成 500；配合 `data` / `header` / `token` 的 `isinstance` 兜底，畸形请求体一律走正常 403
- **新增 `.dockerignore`**：`Dockerfile` 的 `COPY . .` 会把构建上下文整个拷进镜像的一层，而部署目录同时是 build context 且含 `.env` 与日志。排除表覆盖 `.env*`（含 `.envrc` 这类无点变体）、`*.pem` / `*.key` / `id_rsa*` / `.ssh/` / `.aws/`、`logs/` / `sessions/` / `.git/` / 虚拟环境。运行时不受影响（compose 以 `.:/app` 绑定挂载，容器读的是宿主机真实文件）
- **token 不再跨用途回退**：`/api/ram/user` 与 `/gpu/distribution` 原本在缺专用 token 时会逐级回退到 `FEISHU_VERIFICATION_TOKEN`，等于让不同的暴露面共用同一把钥匙。三处改为只认各自专用 token、无回退；`RAM_QUERY_API_TOKEN`、`GPU_DIST_TOKEN` 一并加进 `Config._REQUIRED_FIELDS`，未配置时启动即告警。`dist_url()` 在缺 token 时返回空串，摘要卡不再渲染一个点进去必然 403 的按钮；GPU 页面加 `referrer: no-referrer`，避免带 query 的 URL 经 Referer 外泄给外链 CDN
- **管理员门禁 fail-closed**：新增 `actions._is_admin()` 收敛 5 处散落判定 —— 原写法在 `ADMIN_FEISHU_OPEN_ID` 未配置时会因 `"" != ""` 为假而全员放行。`tools/pfs_transfer`、`tools/temp_ak_issuance` 两处同款兜底
- **缺管理员配置时不再自动批准 GPU 工单**：改为拒绝并评论提示。该分支必须过 `_mark_approval_notified` NX 闸门 —— 它不写工单状态、每轮轮询都会重新捞到该工单，而轮询间隔只有 20 秒，不去重会刷成每天数千条 Jira 评论
- **`ram_approval:instance:{code}` 加 90 天保留期**：该记录在失败分支会写入申请人姓名/邮箱/手机号，此前 `r.set` 无 `ex=`、永不过期，是项目里唯一没有保留期的命名空间（其余 7~30 天）
- `REDIS_PASSWORD` 加进 `Config._REQUIRED_FIELDS`（redis-py 对空串不发 AUTH，此前缺失时启动零提示）
- `Dockerfile` 移除 `pip config set global.trusted-host`：index-url 本就是 https，该行只是对该源关闭证书校验，且会写进镜像 pip.conf 长期生效
- `core/ssh_transfer/engine_ossutil.py` 修正一处**与实现不符的注释**：原注释宣称参数经 `shlex.quote` 构成纵深防御，但这些值被拼进**双引号赋值**，单引号在该上下文只是普通字符。真正的防线是 `paths` 层白名单，注释已改为事实并标注禁止放宽 `_SEG_RE`
- `requirements.txt` 补 `matplotlib`：`utils/chart_builder.py` 直接 import 但从未声明，此前依赖传递安装
- 测试：新增 143 例安全回归（token 门禁 67 / API token fail-closed 35 / 管理员门禁 28 / 调度器审批门 13），全量 **2231 passed**。每条拒绝用例均装哨兵断言危险动作未被执行——只断言状态码不作数

> **部署提醒**：本批含 `.dockerignore` 与 `requirements.txt` 变更 → 必须 `docker compose up -d --build`，普通 restart 式部署不会生效。上线前先配置 `RAM_QUERY_API_TOKEN` 并 force-recreate（`restart` 不重载 `env_file`），否则 `/api/ram/user` 将全量 403。

### Added
- **第二阿里云主账号（`1949`）接入临时 AK/SK 发放**（`core/temp_ak_issuance/accounts.py`，`2808296`）：新增 `AccountProfile` 档案注册表，一个 Bot 给两个主账号发凭证。**不是复制一份代码**——延长/撤销审批被两账号共用同一个 definitionCode，同一 code 只能有一个处理器认领，副本必然一个抢到另一个永远收不到，故必须由单一处理器按「凭证ID → grant → 账号」分派、grant 带账号维度。隔离项：凭证（各账号自己的 RAM 可写 AK，显式传参）/ Redis（`temp_ak:` vs `temp_ak_1949:`）/ 凭证ID（`tak-` vs `tak1949-`）/ RAM 登录名（`tempak-` vs `tempak-1949-`）/ 显示名后缀 / 桶表 / 表单字段映射 / 内部回执群。新审批「数据访问凭证申请（产线）」`0133C4FC-…`（使用人名称 / 权限设置 / DateInterval / 访问目录 / 备注，无「平台」单选＝恒阿里云）；延长/撤销复用 `E9333E62-…` 无需另建
- 临时凭证正文补**地域 / 外网 Endpoint / 桶域名**三行：深圳的桶用杭州 endpoint 会被 OSS 回 403 `must be addressed using the specified endpoint`，使用方会以为凭证无效
- CPFS/NAS 数据流动支持**临时 DataFlow**（`engine_nas.create_dataflow`/`delete_dataflow`）：先 `resolve_dataflow` 找可复用的现有绑定（命中则**绝不删**，那是别人的绑定），找不到才临建并标 `dataflow_ephemeral=True`，`run_to_completion` 末尾无论成功/失败/超时都删掉自己临建的那条（不长期占用「10 条/CPFS」上限）。「CreateDataFlow 会清空 Fileset」的风险仍在，故临建**只对智算版（`bmcpfs-`）开放**，通用版直接拒绝
- SSH 迁移链失败卡带**失败明细**（`engine_ssh.failure_detail`）：ossutil 报告路径 + 失败对象条数 + 首条根因。ossutil 用 `\r` 刷屏（日志可达十几 MB），必须 `tr '\r' '\n'` 再筛；report 路径来自远端日志，过结构白名单防被构造的对象 key 骗去读任意文件
- **临时 AK/SK 发放**（`core/temp_ak_issuance`）：飞书审批「数据外采访问凭证申请」通过 → 给**外部方**发一组时限 OSS 凭证。权限 `read`/`download`/`write` 三者正交（read=列不下载 / download=才给下载 / write=上传无删除）；policy 内嵌生效 + 到期时间窗，服务端逐调用判时间、到期自动失效。按有效期分流：`到期−now ≤ TEMP_AK_STS_MAX_SECONDS`（默认 12h）走 **STS 单发**（含 Token 到点自灭），超出走 **方案 B**（RAM 长期 AK + policy 时间窗 + 到期硬删）；线上置 `0` 全走方案 B。延长/撤销二合一审批（`TEMP_AK_EXTEND_APPROVAL_CODE`）；企业名→拼音登录名 + 中文显示名；凭证走审批评论、以管理员身份下发，secret 不落盘/日志（`temp_ak:grant` 只存 ak_id）；全局审批白名单只处理指定审批 code。三入口：飞书审批（**唯一发凭证路径**）+ Agent 工具 `manage_temp_ak`（`plan`/`status`/`revoke`）+ CLI；调度器 `temp-ak-cleanup` 每日硬删到期凭证。已上线阿里 OSS，火山云 TOS P1 规划中
- PFS 跨云直传（`core/pfs_transfer`）：vePFS↔CPFS 之间物理无直连 → 三段链编排（源 PFS 沉降 → 跨云对象存储迁移 → 目的 PFS 预热），复用三个现成 orchestrator、零改引擎。状态机 `NEW→SINKING→CROSSING→PREHEATING→DONE/FAILED`，段级「跳过已成功段」续跑，`pfs:transfer:job:{id}`（`xpfs-` 前缀）。三入口（飞书向导卡 / `manage_pfs_transfer` / CLI），**卡片确认与工具 `apply` 双路都要管理员确认**，查询进度不再能把未确认任务跑起来
- SSH 迁移链 杭州 OSS → 新加坡 → 泰国（`core/ssh_transfer`）：paramiko 遥控新加坡 ECS 跑段1 `ossutil`（OSS→本地挂载盘）+ 段2 `rsync`（→泰国）；起任务 `nohup` 后台化 + 只读 `rc`/`pid` marker 轮询（不经 SSH 读长输出）。私钥 Fernet 密文只进内存、host key 固定禁 AutoAdd；桶/前缀/目标子目录过严格白名单防 ssh 双跳注入与路径穿越；估算不确定时 fail-safe 当作需审批。入口：飞书「数据迁移（泰国H200）」+ CLI，`ssh:transfer:job:{id}`（`sgp-` 前缀）
- RAM / 火山 IAM 子账号审批建号（`core/ram_approval`）+ 只读账号查询（`core/ram_query`、`core/volcano_iam_query`）：飞书审批通过 → 建子用户、开控制台、入组、建 AK，凭证走审批评论下发
- 对话**滚动摘要记忆**（`core/agent`）：逐字保留最近 20 条，更早对话由 GLM 压成滚动摘要存 `agent:chat_summary:{sid}`（上限 1200 字），下轮作前缀注入，长对话不再硬截断。压缩批量异步（约每 5 轮一次）在回复发出后收尾，不拖慢感知；失败保留旧摘要
- 数据流动/迁移**在途任务对账**（`dsw_scheduler` `dataflow-reconcile` 循环，每 2 分钟）：后台轮询线程随容器重启会死，对账线程随容器复活兜底——重启后任务完成也自动补推结果卡（跑完必通知）。在线推送与对账推送共用 Redis `SET NX dataflow:notified:{job_id}` 闸门，跨线程只推一次
- 火山 vePFS/TOS 数据流动（`core/vepfs_dataflow`）：**预热**（TOS→vePFS）/**沉降**（vePFS→TOS）。无持久 DataFlow 对象，提交任务直接带桶/前缀，方向由地址类型自动判断；与 CPFS 共用三步级联向导卡（选云→选地区→表单）。Agent 工具 `manage_vepfs_dataflow` + CLI 三入口，`vepfs:dataflow:job:{id}` 状态机
- 阿里 CPFS/NAS 数据流动（`core/cpfs_dataflow`）：NAS DataFlow **预热**（OSS→CPFS）/**沉降**（CPFS→OSS）。查现有 DataFlow + 提交任务，按目标子目录匹配最长前缀绑定（临时 DataFlow 见上方本轮条目），智算版/通用版自动分支；飞书选择器从发现的 CPFS↔OSS 绑定里选。Agent 工具 `manage_cpfs_dataflow` + CLI 三入口
- 同云桶间迁移（`core/bucket_transfer`）：同账号跨 region/桶，阿里 `oss://→oss://`、火山 `tos://→tos://`；OSS 自动探测源/目的 region，跨 region 走公网。与跨云迁移独立命名空间、仅复用引擎，混合 scheme 拒绝并提示改用跨云迁移。真机验收通过
- GPU 卡分布大盘（`tools/aliyun/gpu_distribution`）：地区×卡型分布 + 每用户在算卡数 + 近 N 小时趋势，实时 HTML 页面 `/gpu/distribution`（token 门禁），15s 陈旧后台单飞刷新；飞书问「谁在用卡/卡分布」回摘要卡 + 链接
- 数据集大盘定时维护（`core/dataset_dashboard`，`dataset-dashboard` 循环）：遍历飞书「数据集大盘」多维表格现有行，按 uri 扫对象存储回填脚本负责列（状态/云/厂商/时长/数据集类型），只填数据、不改表结构、不碰人工分析列
- 跨云数据迁移（`core/transfer`，一期 TOS→OSS）：用户给路径自动解析方向/推导目的，阿里在线迁移服务（hcs_mgw）建址→建任务→启动→轮询；飞书确认表单卡（同名策略单选）+ 受理/进度/终态卡、Agent 工具 `manage_transfer`、CLI（`python -m core.transfer.cli plan|apply|status`）三入口共享核心；>1TB 走管理员审批。方向判断已含双向，二期补火山引擎 OSS→TOS、三期接 CPFS/VePFS 沉降段
- OSS 权限同步（`core/oss_perm`）：飞书多维表格 → 每人最小权限 RAM 策略，桶级/目录级两档粒度，飞书表单卡选择性下发（粒度单选 + 成员多选默认全选 + 一键确认）
- 集群算力效率（MFU）日报工具 `cluster_mfu`：多区域、交互式区域切换卡片、24h 快照、按钮回调秒回
- 容量巡检结果写入飞书多维表格（`capacity_bitable`，巡检快照→厂家总量→批次明细 三表关联）
- 每日早报：实例汇总 + 集群 MFU 双卡（北京 9:00）
- `deploy.ps1` 一键部署脚本（bind-mount 免 rebuild）
- GitHub Actions：PR Check / Release / Sync-to-Public 三条流水线
- Jira 工作流查询工具（jira_workflow_tool）
- GitHub 工作流查询工具（github_workflow_tool）

### Changed
- SSH 迁移链段1 强制**单分片顺序写**（`e4eef16`）：目的端 `/mnt/sgp_oss` 是 ossfs2(FUSE)、只支持顺序写，`--parallel 1 --part-size 5Gi` 压成单分片（跨文件并发 `--job 30` 保留，实测 148MiB/s）
- SSH 迁移链 `run_to_completion` 轮询上限 **48h → 7 天**：19.5TiB 的段1 就要约 38h，原上限会在任务仍正常运行时误判「轮询超时」
- `permsync.make_ram_client(ak, sk)` 支持显式传参且**只用传入的那对**（多主账号隔离的唯一安全入口）；只给一半直接抛错，不再静默回落到全局凭证
- 启动自检（`settings.print_validate`）打印**已注册的临时 AK 账号档案**（`.env` 改动后 force-recreate 是否生效的客观依据），并对 `ALIBABA_CLOUD_ACCESS_KEY_*` 环境变量告警——它会被 RAM client 零参路径优先采用，把所有账号的建号请求劫持到同一个账号
- RAM 建号失败提示补上「子用户可能已建」（`5c9079e`）：密码是建号链最后一步，撞策略时子用户往往已创建，但未入组、无权限、无 AK；用**相同登录名**重新提交会自动复用补齐，勿手动删除或改名
- 屏蔽 DSW **到期自动关机**：新增 `DSW_IDLE_STOP_ENABLED`（默认 `false`），到期前 15 分钟警告卡 + 到期自动停止均包进开关，利用率关机交给阿里云工作空间；**GPU 空转提醒不受开关影响、始终生效**。置 `true` 恢复旧行为
- 数据流动/迁移进度卡+结果卡**优先推发起人**（`job.created_by`），发起人为空才降级配置频道；在线推送、对账、查询三处目标一致，NX 闸门仍只推一次
- 指标趋势图**按需附带**（`feishu_bot/messages`）：仅当意图含监控/集群或问题命中指标词才附 Prometheus 趋势图，其余纯文本回复，省掉每条回复的云查询+渲染+上传
- Agent executor 加**迭代/超时上限**（`max_iterations=8` / `max_execution_time=60`），防工具调用循环烧钱或挂住；飞书对话执行器 `verbose=False`
- 会话历史 Redis **7 天空闲 TTL**（`agent:chat_history` 每轮续期），不再无限堆积
- OSS 权限对账卡移除「孤儿策略（建议回收）」展示节（命令行 `--audit` 仍输出）

### Fixed
- SSH 迁移链段1 跑前清残留 `*.temp`（`e4eef16`）：ossutil 见到已存在的 `.temp` 会当续传、从非零 offset 写 → ossfs2 拒绝。**不清的话，段1 只要被中断过一次，之后每次重试都必然失败**。真机实证：19.527TiB/75850 对象那单，≤100MiB 的 33484 个全成功、>100MiB 的 42366 个全失败，与分片阈值严格吻合；修复后零 EINVAL
- RAM 建号失败**不再重复评论**（`8ae8a26`/`680d89e`）：`_is_instance_done` 现在也认「确定性失败」终态（`error_terminal`），后续事件在入口短路、不再重跑；`_claim_failure_notice` 让同实例同错误只评论一次（换错误仍播报，Redis 不可用则放行）。终态 marker **刻意只留四个窄词**（`invalidpassword`/`password policy`/`密码不符合`/`invalidloginname`）——曾放过 `invalidparameter`/`malformed`，宽泛子串会把瞬时错误（网关 5xx、限流）永久标成终态、失去自愈
- RAM/IAM 建号**两级审批被绕过**：事件里的节点级 `PASS` 被当成通过 → 第一级审批人点同意就建号。改为只认回拉实例详情后的**顶层实例级 `status == "APPROVED"`** 单值（`PASS`/`AGREE`/`DONE` 等节点/timeline 词一律不认），无实例号 fail-safe 不建
- 全局审批白名单：Bot 只处理配置内的审批 code（RAM 建号 + 各账号发放 + 延长/撤销），其余含未带 code 的组织审批一律记日志丢弃，早于所有审批处理器
- 临时凭证 endpoint 拼装归一两套地域写法（`8ae8a26`）：`permsync.BUCKET_MAP` 存的带 `oss-` 前缀、`TEMP_AK_*_BUCKET_MAP` 存裸 region，不判重会拼出 `oss-oss-ap-southeast-1.aliyuncs.com`；`resolve_bucket` 增加**按真实桶名反查**地域（申请人常直接填真实桶名，否则连接信息退化成「未知」）
- 临时 AK 到期清理 `sweep_expired` 的异常捕获改为 **per-grant**：原来按账号包住整个循环，一条脏记录会中断该账号剩余全部清理且每轮都断 → 到期号永久清不掉
- 临时 AK policy **桶信息动作独立成条**（`aa879f4`）：`GetBucketInfo/Stat/Acl` 原与 `ListObjects` 同条、整条带 `oss:Prefix` 条件，而桶级请求不带 prefix → 被条件卡死，外部方拿到凭证访问不了桶
- `deploy.ps1` 首次部署自动补传 RAG 模型缓存 + 向量库（`ba6cd57`）
- vePFS `DescribeDataFlowTasks` **补分页**（`page_number=1`/`page_size=100`）：不传分页时火山返 `total_count>0` 但空列表 → 任务永远卡 `RUNNING`，现终态可读；火山错误 JSON 友好化（抽 `Error.Code/Message`，已知码给人话，不再甩原始 JSON）
- CPFS/vePFS 数据流动**多套一层 `cpfs`/`vepfs` 目录**：`make_plan` 的 `fs_id` 直达分支未剥挂载前缀，导致预热落到 `/mnt/data/cpfs/...`。两条解析路径统一 `_strip_mount`（新增 `VEPFS_MOUNT_PREFIX`，默认 `/vepfs`）
- 飞书卡片回调**不再阻塞 3s 死线**：查询进度 sync 只读 Redis 秒回当前卡、后台线程 refresh 后推更新卡；GPU/AK 提交的网络调用（RAM 关联、取用户名）移入后台线程，先即时回卡再异步补推
- `tools/aliyun/oss.list_objects` 改 `itertools.islice(ObjectIterator, max_keys)` 首页截断，不再 `list()` 枚举整桶 → 百万级对象桶不再挂线程/OOM
- 数据流动/迁移对账去重加固：修 auditor 复审发现的重复推卡竞态——`bucket_transfer` 每次 `_save` 刷 `updated_ts`（补齐 stale 门），终态推送先落 `notified` 标记再抢闸门，`reconcile` 修正前缀切片
- 跨云迁移 submit/confirm 幂等：`transfer:launch` NX + MGW job_name 幂等，止住连点起多线程 + 刷卡风暴；确认后原地替换为「进行中」卡、重复解析回进度卡
- vePFS `CreateDataFlowTask` 参数对齐真机：`DataStorage` 用裸桶名，`DataStoragePath`/`SubPath` 非空须首尾带斜杠

### 已知缺口（未修，勿当已解决）
- SSH 迁移链 `start_stage1`/`estimate_source` **不带 `-e/--endpoint`**，完全依赖新加坡机上写死杭州的 `~/.ossutilconfig` → **源桶不在杭州则段1 必挂**（rc=2，403 `must be addressed using the specified endpoint`）
- PFS 跨云直传（`core/pfs_transfer`）线上已开启（`PFS_TRANSFER_ENABLED=true`）但**整链尚未真机验证**；每个 PFS 与其中转桶必须同地域，预热目标须落在该 CPFS 已有 DataFlow 绑定的目录之下
- 延长/撤销审批为两账号共用模板 ⇒ 第二账号的审批人也能批准默认账号凭证的延长/撤销（只能延长既有范围、不能扩权）。属飞书审批权限层面，代码侧无法拆
- 临时 AK 的火山 TOS 分支：引擎骨架（`issuer_volcano`/`policy_volcano`/`cleanup_volcano`）已在包内，但审批入口仍硬拒 `platform=火山云`，实际不可用（P1）

---

## [1.0.0] - 2026-04-23

### Added
- 飞书 Bot Webhook 服务（/feishu/event）
- GPU 资源申请卡片流程（Jira 工单 + DSW 实例自动创建）
- Prometheus 监控工具、GPU 训练建议工具
- 集群健康看板工具
- LangChain Agent 多工具路由