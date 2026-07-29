# Changelog

所有版本变更记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased]

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