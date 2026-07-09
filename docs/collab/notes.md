# 协作看板 — dev / tester / auditor

三个 Claude 协作开发本项目（AIOps 飞书 Bot）。**每次开工先读本文件 + 根目录 `CLAUDE.md`。**

## 角色与硬边界（防止两个 Claude 改同一文件互相覆盖）

| 角色 | 谁 | 能写什么 | 不能干什么 |
|------|----|---------|-----------|
| **dev** | 主会话（编排者） | `core/** config/** tools/**` 源码、部署 | — |
| **tester** | 队友 | 只写 `tests/**` | 不碰源码；要改源码 → 交给 dev |
| **auditor** | 队友 | 什么都不写（只读） | 不 Edit/Write/提交；发现 → SendMessage 给 dev |
| **researcher** | 队友 | 只写 `docs/collab/research/**` | 不碰源码；查阿里/火山/飞书/LangChain 文档+取证，结论带出处 SendMessage 给 dev |
| **planner** | 队友 | 只写 `docs/collab/planning/**` | 不碰源码/测试；维护路线图+backlog+任务板，拆 epic→task（带验收+owner+依赖），排优先级 |
| **docwriter** | 队友 | 写 `README.md`/`CHANGELOG.md`/`docs/**`（不含 `docs/collab/**`） | 不碰源码/测试；以 CLAUDE.md 为事实源更新用户/项目文档，新功能靠前 |

源码的唯一改写者是 dev。永不并发编辑同一文件。

## 标准流水线（每个功能都走，不跳步）
**researcher 查资料 → planner 计划 → dev 开发 → tester 测试 → auditor 审计。**
> **提交闸门（硬规）**：任何改动必须 **auditor 审计通过、无阻塞项** 才能 `git commit`。dev 不得跳过审计直接提交。文档类改动（docwriter）也要过一遍事实/准确性核验再提交。
- **第一步永远是查资料**：researcher 先上各大平台查有没有现成项目/官方文档/API 规格/踩坑，产出带出处的结论（写 `docs/collab/research/`）。别一上来就写代码。
- 再由 planner 据此拆任务、排优先级（写 `docs/collab/planning/`），dev 才动手；tester 补测、auditor 复审收口。
- 例外：改动已有充分调研+规划的既有条目（如 backlog 里已成熟的项），dev 可从开发接续，但要在日志注明"研究/规划已在 XX 完成"。

## 运作模式：常驻团队（成本不设限）
六个角色（dev + tester/auditor/researcher/planner/docwriter）常驻在岗、保留各自上下文记忆，dev 唤醒同一实例续用（不清空重开）。各站岗职责：
- **auditor**：dev 每次提交/改动 → 只读复审（正确性+安全）。
- **tester**：源码每变 → 跑相关 pytest + 补覆盖。
- **docwriter**：功能/代码变 → 同步 README/CHANGELOG/docs。
- **planner**：持续维护路线图/backlog/任务板。
- **researcher**：随时待命查规格/取证。
无新活时待命，由 dev 在每个节点派发。

### 队友上下文管理（回收重建 + 磁盘持久记忆）
- 每个队友有独立上下文窗口；快满时 harness 自动压缩其自身历史（不崩、丢细节）。dev **无法**在线清理别的 agent 的上下文。
- dev 的唯一杠杆 = **回收重建**：某队友上下文涨大（看每次完成回报的 `subagent_tokens`），dev 退休该实例、新拉同角色干净实例，让它读本文件 + 任务板重载背景。
- **持久记忆放磁盘、不放 agent 脑子**：本文件（日志/决策）+ 任务板 + `planning/` + `research/` 是团队长期记忆 → 任何队友随时可回收重建而不丢项目知识，只丢无所谓的临时工作记忆。
- dev 盯 token 用量，接近窗口上限即主动回收，不等它触压缩/崩。

## 队友在岗与任务分派表（dev 实时维护，与 `/tasks` 任务板互为镜像）
> 状态：🟢运行中 / 🟡待命(未拉起，无常驻实例) / ✅本轮已交付。实例列为空=当前无活、已结束进程，可带记忆重新唤醒。

| 角色 | 当前实例 | 当前任务 | 状态 | 最近产出 |
|------|---------|---------|------|---------|
| **dev** | 本会话 | 收口 #43 | ✅ | 本轮提交：#37 282163a/#38 ad57f6c/#40+41 508f43a/#42 edf918f/#43 ea2ecc6（均未部署） |
| **tester** | — | #43 二轮补测 | ✅ | 24 例重写，全量 694 passed |
| **auditor** | — | #43 二轮复审 | ✅ | 通过无阻塞（2 Low 记账） |
| **researcher** | — | #42-ext DstDirectory/失败重试语义 | ❌ | **API abort 挂了、无结论**（deploy-B 卡在此，待重拉） |
| **planner** | — | #39 5/6 选型框架 | ✅ | `planning/conversation-opt-5-6.md`（CONV-* 待落板） |
| **docwriter** | — | — | 🟡 | README/CHANGELOG 功能重排（未提交） |

## 通信协议
- **实时**：`SendMessage` 按名字互发。用户 → dev → 派活给 tester/auditor → 回报 dev → dev 汇总给用户。
- **任务板**：`TaskList`/`TaskCreate`/`TaskUpdate`，字段 owner(dev/tester/auditor)/status/blockedBy —— 谁在干啥的单一事实源。
- **留痕**：重要决策/发现在本文件追加一行，格式 `[YYYY-MM-DD HH:MM] [DEV|TESTER|AUDITOR] 内容`。

## #36 全项目扫描发现（待 A/C 齐后汇总去重、交用户定优先级；未开修）
**B 域（bot+调度器）** [2026-07-09][AUDITOR]：
- **HIGH-1** 四个「查询进度」sync handler 在 3s 回调里同步 `refresh()`→在途时走 poll_once 打云 API（数十秒）→超时被飞书重投、重投被 15s 去重吞成 `{}`→原地卡永远刷不出、只见失败。点位：actions.py:579/802/914/1070、_h_query_progress_by_id 956/964/977。修：sync 只读 Redis get_job 先回当前卡，后台线程 refresh 后 `_send_card` 推更新。
- **HIGH-2** 3s 死线内同步网络：`_h_submit_ak_register`(60-72，RAM ListUsers+2次飞书HTTP) / `_h_submit_gpu_request`(147-148，_get_user_name 在起线程前)。修：整块挪后台线程，先即时回卡、异步补推。
- **MED-3** `tools/feishu/notify._get_access_token`(66-80) **每次都取 token、无缓存**，与 CLAUDE.md/notes「token cached」矛盾；早报 N 用户=N 次取 token，端点有限流、历史踩过坑。修：模块级缓存+提前5min刷新+防并发锁。
- **MED-4** cpfs/vepfs/bucket 三个 retry 不删各自 launch NX 锁（TTL30s）→快速失败后 30s 内重试被「已下发」静默吞。点位：811-819/923-931/1079-1088。修：对齐 transfer(602)，_save(stage=NEW)后 delete 锁。
- **MED-5** `capacity_bitable._list_records`(107-120)/`_field_names`(86-104) 不校验 resp code→列举瞬时失败返回[]→upsert 退化成 insert 造重复行。修：code!=0 抛错、write_scan 放弃本次写。
- **LOW**：_is_duplicate_event 内存兜底溢出500直接 clear()（messages.py:42）窄窗双处理；card_action_is_duplicate Redis异常不去重（1152，非真隐患）；_TTL_NOTIFY 定义顺序略脆（上轮已记）。

**C 域（凭证/基础设施/工具层）** [2026-07-09][AUDITOR]：
- **HIGH-C1** `tools/aliyun/oss.py:109` `list_objects` = `list(oss2.ObjectIterator(...))[:max_keys]`，已核 oss2 2.19.1：max_keys 只是每页大小，会枚举**整桶**再切片→百万级桶挂线程/OOM（本项目桶动辄 1.6M 对象）。修：`itertools.islice(ObjectIterator(...), max_keys)` 提前 break。
- **MED-C2（安全）** `utils/aliyun_client_factory.py:241-249` 用户 STS 失败**静默降级全局 AK** 执行用户动作→越权+审计归属错人+破坏多租户。修：open_id 非空时 STS 失败返 None 明确报错，全局 AK 仅限无 open_id 后台任务。
- **MED-C3** `config/settings.py:310-326` `_REQUIRED_FIELDS` 漏 `BOT_CREDS_ENCRYPTION_KEY`（缺→绑 AK 静默失败）+ `ALIYUN_BOT_MASTER_AK_ID/SECRET`（缺→STS 全挂静默走全局 AK）。修：补进必填校验。
- **MED-C4** `utils/redis_client.py:18-25` 只设 connect timeout 无 `socket_timeout`（读超时）→Redis 卡住时请求/轮询线程无限阻塞；连带 STS 缓存失效狂刷打爆 QPS。修：加 socket_timeout=5+health_check+retry_on_timeout。
- **MED-C5** `utils/aliyun_client_factory.py:109-116` `get_oss_bucket` 不探测桶地域→跨地域桶 list/put/delete 打错 endpoint 失败（与 oss.py `_resolve_bucket` 行为不一致）。修：走 _resolve_bucket 或传 region。
- **MED-C6** `tools/aliyun/ram.py:45-71` `list_ram_users_api` 无分页→截断当全量（permsync 的 marker 循环是对的）。修：照 permsync 补分页。
- **MED-C7** = B 的 MED-3 同一条（notify token 无缓存）。
- **LOW**：STS 无并发锁惊群(aliyun_sts.py:156)；crypto legacy 明文回退(crypto.py:98)；`FEISHU_RAM_APPROVAL_CODE` 硬编码默认 UUID(settings.py:49)；`get_tos_client` 不复用泄漏(volcano_client_factory.py:20)。核实：无 AK/SK/密码明文写日志。

**A 域（数据搬运四链）** [2026-07-09][AUDITOR]：
- **基线纠偏**：200830「三条同坑」只对两条——cpfs(actions.py:791-794)/vepfs(905-907) 中招；**bucket(1061-1062) 没中招**（只回 toast、不返回 card、无原地替换）。且 cpfs/vepfs 的 _on_update 对 RUNNING 会另推新卡→用户仍见进度，属「按钮不消失+多张卡」非「无反馈」。
- **HIGH-A1（四链全中，最该修）** 重试后完成通知被永久吞：`dataflow:notified:{job_id}`(dsw_scheduler.py:850-857) 30天TTL、**只set不delete**，job_id=hash(源,目的,当天)跨重试不变→首次失败已占闸门→重试成功时终态 on_update 抢闸门返 False→**成功卡不发**，reconcile 因 DONE 非 active 也救不回。transfer retry 只删 transfer:launch 没删 notified；cpfs/vepfs/bucket retry 啥都不删。修：四链 retry 重置 stage=NEW 时 `delete(dataflow:notified:{job_id})` + 清 job["notified"]。
- **MED-A2** cpfs/vepfs 确认卡 200830 no-op（bucket 不用改）：照 transfer 加 progress_card_v2(2.0)。
- **MED-A3** vePFS 终态子串匹配 `"success" in "unsuccessful"`=True 会把失败判 DONE（engine_vepfs.py:46-71，poll 先判 done 后判 fail）。修：先 is_failed 后 is_done / 精确匹配 + 真机反查终态串。
- **LOW**：火山 DMS 无重复提交幂等(engine_tos.py:174-206，仅 Redis 降级窗口双建任务)；cpfs 临时 DataFlow 泄漏窗口(create成功→submit前崩)；reconcile 漏 SINKING(dsw_scheduler.py:832，Phase3 未上线)；cpfs/vepfs retry 不落盘 nit；claim-before-send 无回滚孤儿(与 A1 同源，可合并：send 失败 delete 闸门键)。

## #37 高危修复（dev 已改，待 tester+auditor，未提交）
- **HIGH#1（四链 retry）**：transfer/cpfs/vepfs/bucket 四个 retry handler 在重置 stage=NEW 时 `delete` 各自 launch 锁 + `dataflow:notified:{job_id}`（否则重试成功后结果卡永不推）；cpfs/vepfs 补 `_save` 落盘（原来不落盘靠 Redis 旧值侥幸过 guard）。顺带修 MED-A/B 的「retry 不清 launch 锁 30s 被吞」。
- **HIGH#2（oss 列举）**：`tools/aliyun/oss.py:109` 改 `itertools.islice(ObjectIterator, max_keys)` 第一页截断，不再 `list()` 枚举整桶。
- **HIGH#3（查询不阻塞 3s）**：新增 `_async_refresh_and_push` 助手；四个 `_h_query_*` + `_h_query_progress_by_id` 改为 sync 只读 `get_job` 秒回当前卡、后台线程 refresh 后推更新卡（阶段推进才推、终态经 notified 闸门去重）。schema 不变（1.0→1.0 / 各自原样），不触 200830。
- **HIGH#4（提交不阻塞 3s）**：`_h_submit_gpu_request` 的 `_get_user_name` 移进 `_do()` 线程；`_h_submit_ak_register` 的 RAM 映射块整体挪 `_link_ram` 线程，立即回卡（RAM 用户「后台关联中」），匹配到再异步补推一条文本。
- 两文件编译通过。tester 注意：query handler 由 refresh→get_job，测试 mock 需相应改（sync 不再 poll）。
- [2026-07-09][AUDITOR] #37 四高危复审**全部通过、无阻塞**。键名逐对对齐、islice 首页截断正确、查询 1.0→1.0 不触 200830、终态经共享 NX 闸门单推、提交异步闭包正确。4 条 LOW：①cpfs/vepfs retry `_save` 注释与守卫不符(FAILED 本就放行，属防御性)；②`_h_query_progress_by_id` 后台推送目标被从「私信 open_id」改成「chat/群」需确认；③非终态阶段跃迁进度卡可能双推(无害)；④重启后 reconcile 已推终态时用户当次查询见原地进行中、需二次点击自愈(窄窗)。informational：`query_transfer_progress` 已是死处理器。
- [2026-07-09][DEV] 据审计修 LOW①(注释改准)+LOW②(还原 by_id 后台推送为 `_send_card(open_id, chat)` 私信优先)；LOW③④记账不改(窄窗可自愈)。已重编译，请 auditor 快速复核这两处后随 tester 一起过闸提交。

## #38 对话功能优化 1–4 快赢批（dev 已改，待 tester+auditor，未提交）
（#37 四高危已提交：源码 282163a + 测试 0bd5ad3。）
- **1 指标图按需**（`messages.py`）：新增 `_wants_metrics_chart(text, intents)`；仅当 intents 含 monitor/cluster 或问题命中指标词才附 Prometheus 趋势图，否则纯文本回复。省掉每条回复的云查询+渲染+上传，且不给知识库/工单/迁移回复塞无关曲线。
- **2 verbose→False**（`agent.py` `_build_executor`）：飞书对话执行器不再把 Agent 推理打进日志。
- **3 迭代/超时上限**（`agent.py`）：两个 AgentExecutor 加 `max_iterations=8` + `max_execution_time=60`，防工具调用循环烧钱/挂住。
- **4 历史 TTL**（`agent.py` `_save_turn`）：pipeline 加 `expire(key, 7天)`，会话历史空闲 7 天自动过期，不再无限堆积 Redis。
- 两文件编译通过。tester 注意：`_wants_metrics_chart` 真值表 + 附图分支只在命中时走；executor 现有 iter/time 上限；`_save_turn` 每轮 expire 续期。
- [2026-07-09][AUDITOR] #38 复审通过、无阻塞。intents 恒 list 对 `[]` 安全、漏附/误附无退化、gate else 与 except 都纯文本回复不吞、TTL 为空闲滑动续期、max_execution_time 是步骤间软兜底(不中断单工具，已提醒)。tester#37 观察被 `card_action_is_duplicate` 内容哈希 SET NX 15s 全局覆盖，无需加去重。LOW(可选)：regex 裸 `gpu` 会给"gpu工单"附图。
- [2026-07-09][DEV] 采纳 LOW：去掉 regex 裸 `gpu`（留 显存/利用率/使用率等），"查我的gpu工单"→不附图、"gpu利用率/显存/集群状态/cpu趋势"→仍附图（真值表已验）。请 auditor 确认这行 delta、tester 补一条"gpu工单不附图"用例并重跑，再过闸提交。

## #40 屏蔽 DSW 到期自动关机（dev 已改，待 tester+auditor，未提交）
用户要求：阿里云工作空间已有利用率关机，我们的 DSW 到期自动停止是重复造轮子。用户选「只停自动关机 + 去掉到期警告，保留 GPU 空转提醒」。
- 新增 `settings.DSW_IDLE_STOP_ENABLED`（默认 False）。
- `dsw_scheduler._check_running_instances`：把「到期警告卡 + 到期自动 `manage_pai_dsw(stop)`」两段包进 `if settings.DSW_IDLE_STOP_ENABLED`（默认不执行）；**GPU 空转提醒段保留、不受开关影响**（flag 关时 warned 恒 False → 空转段照跑）。
- 更新模块 docstring + 启动日志（默认不再打「DSW 超时监控」）+ CLAUDE.md 描述。编译过，flag 默认 False 已验。
- tester：flag=False 时无警告卡、不调 manage_pai_dsw stop（即便 remaining<=15min 或 warned+超时），GPU 空转仍提醒；flag=True 恢复旧行为。auditor：确认空转段在 flag 关时仍可达、无死代码、stop 仅 flag 下可达、默认 False。

## #41 数据流动进度/结果卡优先推发起人（dev 已改，待 tester+auditor，未提交）
用户反馈：CPFS/迁移进度卡都推到管理员(`_cfg_*_chat`)、发起人收不到。改为优先推 `job["created_by"]`（发起人 open_id），空则降级配置频道。
- **字段是 `created_by` 不是 `open_id`**（全项目 `open_id=job.get("created_by")` 模式；用错会静默取空=没改）。
- 改点（四链在线 `_on_update` 结果+进度卡、错误文本、查询异步助手 `_async_refresh_and_push`、对账 `_reconcile_dataflow_once`）：`_send_card("", _cfg_x(), …)` → `_send_card(j.get("created_by",""), _cfg_x(), …)`。`_send_card(target,chat)` 内部 `target or chat`，故 created_by 空自动回落配置频道。
- **关键不变量**：在线线程、对账、查询助手三处必须同目标（都 created_by），否则 NX 闸门只认一方、另一方漏卡。
- 未动：oss_perm 审批推送（316/321/…，本就该进管理员频道）、`_h_query_progress_by_id`（手动按 ID 查，推给查询者本人，gate-free）。编译过。
- [2026-07-09][AUDITOR] 复审 #40/#41 通过、无阻塞。#40：flag 关时 stop 不可达、GPU 空转段仍可达非死代码、旧行为完整、默认 False、docstring/日志/CLAUDE.md 一致。#41：四 orchestrator 均存 `created_by`、三处终态目标一致（NX 闸门按 job_id 抢锁与目标无关，同目标即无漏卡）、空则回落配置频道、oss_perm/by_id 未误改、无串人。两条 Low（皆预期）：①flag 曾开过再关的过渡窗对 warned=True 的老 ticket 会抑制空转提醒（默认从没开过 True，实际不触发，7天TTL自愈）；②查询按钮异步��� created_by 与按 ID 查推查询者目标不同（#41 本意）。

## #42 修 CPFS/vePFS 数据流动多套一层 cpfs/vepfs 目录（真机暴露的真 bug，待 tester+auditor）
真机现象：预热 `/cpfs/xiaoxiong/sft_data_v1_le/` 数据落到 `/mnt/data/cpfs/xiaoxiong/...`（多一层 `cpfs/`），本该 `/mnt/data/xiaoxiong/...`。
- 根因：`make_plan` 的 `if fs_id:` 分支（卡片向导直达都走这条）用 `normalize_dir(cpfs_path)` **不剥挂载前缀**，而 `_parse_cpfs` 那条会剥 → `/cpfs` 被当真目录拼进 DstDirectory。
- 修：抽 `_strip_mount`（cpfs=CPFS_MOUNT_PREFIX、vepfs=新增 VEPFS_MOUNT_PREFIX 默认 `/vepfs`），**两条解析路径都剥**（`_parse_*` + `make_plan` fs_id 分支）。vePFS 的 `plan_from_addresses` 最终汇 `make_plan`，自动覆盖。
- 已验：`/cpfs/xiaoxiong/sft_data_v1_le/`→`/xiaoxiong/sft_data_v1_le/`；make_plan(fs_id) 的 dst_directory 不再带 /cpfs；`/vepfs/wzh/`→`/wzh/`；无前缀路径不变。编译过。
- 注意：**已落错位置的老数据不会自动搬**，部署后重发一遍预热会落对；或真机手动 mv。tester���覆盖 fs_id 分支剥前缀 + 裸目录 + cpfs:// scheme + 无前缀不变 + 双向(preheat/sink)。auditor：确认不会误剥真实以 cpfs/vepfs 开头的目录名、两侧对称、plan_from_addresses 被 make_plan 兜住。
- [2026-07-09][AUDITOR] 复审 #42 通过、无阻塞。根因堵住（卡片向导恒走 fs_id 分支、现已剥）、追全 make_plan 全调用链无绕过（guided/plan_from_addresses/CLI/Agent tool 都被剥或经 _parse_* 剥）、不误剥 `/cpfsdata`、方向正确（只剥 FS 侧不碰 OSS/TOS）、job_id 变新键=预期。Low nit：cpfs `plan_from_addresses`(216-220) 内联了一段与 `_strip_mount` 逐字相同的剥前缀，现已冗余（make_plan 会再剥），可替换为 `_strip_mount` 单一实现；vepfs 侧则不内联靠 make_plan 兜——两文件不对称，纯整洁性无功能影响。dev 记账不改。
- [2026-07-09][DEV] 延伸调研派出（researcher #42-ext，`research/cpfs-dataflow-path-and-retry.md`）：查阿里云 DstDirectory 相对 fs 根 vs fileset FileSystemPath（决定要不要再剥一层）+ DataFlow 部分失败重试机制。结论回来再定是否追加修复。

## #43 对话优化项5：滚动摘要记忆（dev 已改，待 tester+auditor，未提交）
用户要"更长的记忆"，压缩方式 dev 选 5-B 自管滚动摘要。**注意：edge 模型已废弃，实际在用 GLM（get_cloud_llm）——摘要压缩用 get_cloud_llm，不用 get_edge_llm。**
- `core/agent.py`：逐字窗口保留最近 20 条（MAX_HISTORY），更早的滚进新 Redis key `agent:chat_summary:{sid}`。
- `_load_history`：有摘要则置顶一条 `SystemMessage`「早前对话摘要」+ 最近 20 条原文。
- `_save_turn`：rpush 后若 llen>20，把即将移出的最旧消息 `_fold_into_summary`（`_compress_summary` 用 get_cloud_llm temp0 把 旧摘要+被移出对话 压成新摘要，上限 SUMMARY_MAX_CHARS=1200）再 ltrim；两 key 都 7 天续期。压缩在回复已发出后收尾阶段跑，不拖慢感知；失败保留旧摘要。
- `_clear_history`：一并删摘要 key。编译过。
- **待核**：① SystemMessage 注入 chat_history 对 GLM（OpenAI 兼容）是否 OK（多个 system 消息）；② 溢出窗口取值 `total-MAX_HISTORY-1` 无 off-by-one；③ CLI run() 内存历史不经 _load_history、只 bot 路径每轮加载摘要（可接受）。tester：溢出触发压缩+load 带摘要+clear 删两 key+压缩失败保留旧+redis down 降级。
- [2026-07-09][AUDITOR] #43 复审：核心逻辑对（模型用对/off-by-one/失败守卫全 PASS），但 3 Med 建议改后再提交。[TESTER] 11 例全绿但 GLM 多 system 兼容单测覆盖不了。
- [2026-07-09][DEV] 据审计改 3 处（agent.py/llm_factory.py/messages.py，已编译）：
  - **MED-1** 压缩改**批量+异步**：`FOLD_TRIGGER=MAX_HISTORY+10`（约每5轮1次而非每轮触发），`_fold_into_summary` 丢 `threading.Thread(daemon)`，ltrim 同步、压缩不挡回复。
  - **MED-2** `get_cloud_llm` 加可选 `timeout`（传则一并 max_retries=0）；`_compress_summary` 用 `timeout=SUMMARY_TIMEOUT=20`。
  - **MED-3** 摘要**不再作第二条 SystemMessage**：`_load_history` 只返原文、新增 `_load_summary`，`messages._process_message` 把摘要作前缀拼进 `agent_input`（GLM 100% 安全，绕开多 system 兼容风险）。移除 SystemMessage 导入。
  - → tester 注意：既有 test_conversation_summary 的断言要改（_load_history 不再含 SystemMessage；触发阈值由 >20 变 ≥30；压缩异步——测 _fold_into_summary 本身或桩掉线程；load 摘要改测 _load_summary + messages 拼接）。
- [2026-07-09][AUDITOR] #43 二轮复审**通过、无阻塞**。MED-1 非阻塞+off-by-one(30→evict10留20)+顺序(lrange在ltrim前)+线程异常不外泄 全 PASS；MED-2 timeout 仅传入时生效、既有7处调用逐字不变、lru_cache键不失效；MED-3 摘要两分支后统一拼、写回用原文、SystemMessage 全仓零残留、GLM 只收1条system。2 Low 记账：并发折叠 last-write-wins 偶丢摘要细节(按open_id隔离+5轮/次概率极低)、CLI 写个自己不读的 chat_summary:cli-default（无害）。可随 tester 绿灯过闸。

## #44 vePFS 数据流动终态查不到 + 友好错误（真机反查，dev 已改，待 tester+auditor）
真机反查(SSH prod, 只读)锤定：`DescribeDataFlowTasks` **不传分页** → 火山返 `total_count=1` 但 `data_flow_tasks=[]` → `query_task` 永远 task-None → 空 status → **所有 vePFS 任务永远卡 RUNNING**。带 `page_number=1,page_size` 后返 `status='Finished'`（`_DONE_HINTS` 已含 finished，终态判定本身没问题）。
- **P1** `engine_vepfs.query_task`：`DescribeDataFlowTasksRequest` 加 `page_number=1,page_size=100`；空��表但 total_count>0 打 warning。（核心修复，已真机验证）
- **P2** `orchestrator.poll_once`：终态判定**先 is_failed 后 is_done**（避歧义子串）。
- **P4** `engine_vepfs._err`：解析火山错误 JSON 抽 Error.Code/Message，已知码给人话（`BucketMaybeNotExist`→提示漏填桶名等），不再甩原始 JSON。
- 编译过。部署后 refresh()/对账会自愈历史卡 RUNNING 的任务。
- **待办(未纳入本批)**：P5 向导流程 `created_by` 丢失(该失败 job created_by=""→错误卡落管理员，根因在 `_h_submit_vepfs_dataflow` 拿到的 open_id 为空，待查上游 card_action 分发)；P0.5 CPFS `engine_nas.query_task` 是否同类分页 bug；P3 vePFS 是否支持自动建目标目录。
- 真机确认：用户那两条——vepfs-3dd16c139445 其实 Finished(0文件,源空,成功空操作)；vepfs-1261d7c57d94 FAILED 是**用户漏填桶名**(tos_bucket 存成 "xiaoxiong"，实为目录；真桶 wuji-dc-shanghai)。
- [2026-07-09][AUDITOR] #44 通过、无阻塞。P1 分页/P2 顺序/P4 _err 全 PASS；据 Low 已加 `unsuccess` 进 _FAIL_HINTS（Unsuccessful→FAILED）+ 清 rstrip 冗余，delta 已二次确认。
- [2026-07-09][RESEARCHER] P0.5/P3 结论（`research/dataflow-pagination-and-createdir.md`），**两条都不用改代码**：
  - **P0.5** 阿里 NAS DescribeDataFlowTasks 用 NextToken/MaxResults（默认20、非空列表），无 vePFS 那种「不传分页→空列表」坑；单 TaskId 过滤必命中，CPFS 不卡 RUNNING。可选防御性 `MaxResults:100`（非当前 bug）。
  - **P3** vePFS CreateDataFlowTask **默认自动建目标目录**（官方文档：目录不存在则直接创建），无、也不需要 CreateDirIfNotExist 开关；`submit_task` 无需改。前置：绑定作用域目录/Fileset 上层需预先存在，作用域内子目录才自动建。

## #45 P5：向导流程 created_by 丢失（dev 已改，待 tester+auditor，未提交）
根因（general-purpose 已定）：`create_job_record` 幂等分支原样返回旧记录、**从不回填 created_by**；同日同 (op/fs/子目录/tos) 的 job_id 被无 open_id 入口（Agent 工具 LLM 没注入 / CLI 无 open_id）先建成空 → 永久为空 → 进度/结果卡 `_send_card("",chat)` 回落配置频道（管理员），不私信发起人。CPFS 额外：created_by 还驱动云调用 STS 身份 → 空=退 master AK（越权+审计错人）。
- 修①（治本）：vepfs+cpfs `create_job_record` 幂等分支 `if open_id and not existing.get("created_by"): existing["created_by"]=open_id; _save(existing)`。
- 修②（兜底）：`_h_confirm_cpfs/vepfs_dataflow` 在 `job["stage"]=RUNNING` 前 `if open_id and not job.get("created_by"): job["created_by"]=open_id`（确认者 open_id 是可靠真值，随后既有 _save 落盘）——堵双投递竞态先建空的情况。
- 未改去重键（routes.py 老式分支解析），属次因、非稳定必现，暂不动。
- tester：幂等分支带真 open_id 回填空 created_by、不覆盖已有非空、无 open_id 不写空；confirm handler 两分支回填后 _save。auditor：确认不会把已有真实 created_by 覆盖、CPFS STS 身份现取到真值、vepfs/cpfs 对称。

## #46 安全 MED 批（dev 已改，待 tester+auditor，未提交；C2 用户明确指示不修）
- **C3** settings `_REQUIRED_FIELDS` 补 `BOT_CREDS_ENCRYPTION_KEY`/`ALIYUN_BOT_MASTER_AK_ID`/`ALIYUN_BOT_MASTER_AK_SECRET`（字段名已核对与 settings 属性一致）。
- **C4** `redis_client` 加 `socket_timeout=5`+`health_check_interval=30`+`retry_on_timeout=True`（读超时，Redis 卡住不再无限阻塞）。
- **C5** `get_oss_bucket` region 留空时走 `tools.aliyun.oss._detect_region_endpoint` 探测桶地域（跨地域桶不再打错 endpoint）；传 region 路径不变；延迟导入避环；探测失败回退默认地域。
- **C6** `ram.list_ram_users_api` 加 marker 分页循环（is_truncated/marker），子用户 >100 不再截断当全量。
- **B3/C7** `notify._get_access_token` 模块级缓存（到期前 5min 刷新 + 双检锁），修与「token cached」说法矛盾、防高频取 token 打限流。**（researcher 指出这也是流式回复的前置）**
- **B5** `capacity_bitable._list_records` code!=0 抛 RuntimeError（不再静默返空→upsert 退化 insert 造重复行）；`write_scan` 两处列举加 try→失败返 False 放弃本次写。
- **A2** cpfs/vepfs 新增 `progress_card_v2`(schema 2.0 纯展示)；`_h_confirm_*` 加 keyword-only `reply_v2=True`：确认卡(2.0)→回 2.0 进度卡（同家族避 200830）、重试(1.0 结果卡)→`reply_v2=False` 回 1.0（照 transfer #33）。
- 11 文件全 AST/py_compile 过（actions.py 的 U+FEFF BOM 是既存、Python 正常处理）。
- tester 注意：cpfs/vepfs confirm 现默认回 progress_card_v2（patch progress_card 的既有用例要改，同 #33）；token 缓存命中/过期刷新/并发单飞；ram 分页；redis 新 kwargs；capacity list 抛错→write_scan 返 False；get_oss_bucket 传/不传 region 两路。

- [2026-07-09] [AUDITOR] #45/#46 复审**通过、无阻塞**。#45 四处回填守卫一致（open_id 非空且旧值空才补、绝不覆盖真值）、CPFS STS 身份链路确认回填在 _save/线程前拿到真 open_id 不再退 master AK；A2 分发器 4 位置参→reply_v2 默认 True 生效、2.0→2.0 / 重试 1.0→1.0 避 200830；C3 字段名逐字一致、C4 redis-py 7.4.0 三 kwargs 合法、C5 探测签名一致避环、C6 双终止防死循环、B3 双检锁无竞态坏 token 不入缓存、B5 code!=0 抛错+legit 空表仍正常+快照单行自愈无重复。2 Low 记账：ram max_items=100 可对齐 1000（dev 已改注释措辞，值保留 100）、ram body.users 未守空但 try 吞成 []（无回归）。
- [2026-07-09] [DEV] 据审计 Low 修 ram.py 注释措辞（去掉「上限 100」误述）；max_items 保留 100（功能正确、避免动 tester 断言）。

## #49 跨云迁移(tr-)查不到进度 — 按 task_id 查询实时化（dev 已改，待 tester+auditor，未提交）
真机：确认 tos→oss 迁移(206GB/119384对象)后查不到进度。根因：确认后原地卡是 progress_card_v2（2.0 纯展示、无按钮，加按钮会 200830），在线线程只在**终态**推结果卡→中间态无卡可点；唯一查法是发「查询进度 tr-xxxx」文本，但 `messages._handle_progress_query` 三条分支都只 `get_job`（读旧 Redis）、**不 refresh**→容器重启后台轮询线程死了就永远停在旧 CROSSING、查不到真实进度。（对比：卡片按钮 `_h_query_*` 走 `_async_refresh_and_push` 会 refresh；文本按 ID 查这条没 refresh。）
- 修①：`_handle_progress_query` 三分支 `get_job(jid)` → `o.refresh(jid) or o.get_job(jid)`（refresh 在 stage 仍在途且有 task_id 时重查云端并落库自愈；跑在消息后台线程、无 3s 限制，可阻塞轮询）。
- 修②：tr- 消息补 `bytes_done / bytes_total` 实时进度（原来只显示 bytes_total）。
- 修③：`transfer.cards.progress_card_v2` 加提示行「发送 `查询进度 <任务ID>`」，让无按钮的 2.0 卡可发现文本查法（卡里本就显示 job_id）。
- 两文件 py_compile 过。cpfs/vepfs 的 progress_card_v2 同样无按钮（同坑），本轮先只按用户所指改 transfer，hint 可后续照搬。
- tester：三分支 refresh 被调用（mock refresh 返回在途/终态两态）、refresh 返 None 回退 get_job、tr- 消息含 bytes_done/total、progress_card_v2 含 job_id 提示行。auditor：refresh 在消息线程阻塞可接受（非 3s 回调）、不误触 200830（未给 2.0 卡加按钮）、三分支一致。

- [2026-07-09] [TESTER] #49 补测完成（本地全量 **777 passed / 1 skipped / 9 deselected / 0 failed**，新增 20 例）。
  新文件 `tests/unit/test_progress_query_by_id.py`：
  ① 三前缀(vepfs-/cpfs-/tr-)按 ID 查都调 `o.refresh(jid)` 且 refresh 命中真值时短路不再 get_job（in-flight+终态两态，参数化×2=6 例，**本次核心断言**）；
  ② refresh 返 None → 回退 get_job：都 None→「未找到任务」、get_job 有旧值→用旧值出文案不误报（参数化×2=6 例）；真实 `orchestrator.refresh` 内部 try 吞 poll_once 异常仍返可用 job、stage 维持原样（三链参数化 3 例）；
  ③ tr- 消息：有 bytes_done→「done / total」、无→只 total 且不出现「 / 」、FAILED 带 error（3 例）；
  ④ `transfer.cards.progress_card_v2` info_md 含 job_id + 「查询进度」提示行（且提示行带 job_id）、schema 2.0 无 form/button（1 例）；
  ⑤ 无 ID→弹 query_input_card 回归（1 例）。
  既有 `test_cpfs_feishu.py`/`test_transfer_confirm_idempotent.py` 均无「_handle_progress_query 调 get_job」断言（前者 test_progress_query_by_cpfs_id 用真实 refresh、无 task_id 不 poll，仍绿），无需改。**未发现源码问题**：三分支改动一致、refresh 短路/回退/吞错逻辑与 progress_card_v2 提示行均与断言吻合。

- [2026-07-09] [AUDITOR] 复审 #49：MEDIUM——query-path refresh 首次观测终态后只回文本、不推结果卡/不占 notified NX，导致对账(active 仅含在途态)永久跳过该 job→重启场景富结果卡(尤其 FAILED 的重试按钮)丢失。建议终态分支加 `_claim_dataflow_notify` 门控的 result_card 推送(复用 _async_refresh_and_push 思路)。余 5 点通过：非 3s 路径/三分支一致/不触 200830/bytes 安全/在线并发不双推。
- [2026-07-09] [DEV] 据审计修 MEDIUM：新增 `messages._push_terminal_result_card(chain,jid,job)`——三分支查询把 job 刷成终态后，经 `_claim_dataflow_notify` NX 闸门推 result_card 到 created_by/配置频道(复用 actions._cfg_*_chat + 各链 result_card)，与在线线程/对账去重只推一次。跑消息后台线程可同步推。已编译。请 auditor 复核 delta、tester 补测终态补推(命中 NX→推、被占→不推、FAILED 带重试按钮)。

- [2026-07-09] [AUDITOR] #49 MEDIUM 修复 delta 复核**通过、无阻塞、满足 commit gate**。①NX 闸门+目标频道三链逐一比对与在线线程/对账一致(created_by→_cfg_*_chat)，谁先到谁推、连查两次被 NX 挡；②lazy import 无顶层环(messages/actions 顶层互不 import)；③o.STAGE_* 终态判定 o 单分支绑定正确、chain↔o 成对不错链；④原 updated_ts LOW 已实质消解(终态当场补推、仅真·在途 ≤180s 有界延迟自愈)。bucket(bkt-)不在 _JOB_ID_RE、非缺口。

- [2026-07-09] [TESTER] #49 MEDIUM（终态补推结果卡）补测完成（本地全量 **789 passed / 1 skipped / 9 deselected / 0 failed**，同文件 20→32 例，+12）。
  在 `tests/unit/test_progress_query_by_id.py` 加 `push_spy` fixture（桩 `dsw_scheduler._claim_dataflow_notify`(默认放行)/`_send_card` + `actions._cfg_{vepfs,cpfs,}_chat` + 三链 `result_card` 哨兵），并：
  ① 直测 `_push_terminal_result_card`：闸门未占→推对应链 result_card、首参=job.created_by、次参=该链配置频道（3）；created_by 缺→首参 ""（3）；闸门被占(_claim 返 False)→抢闸门但不推（不双推，3）；
  ② FAILED 走真实 result_card→含各链重试按钮(retry_{transfer,cpfs,vepfs}_dataflow)（3）；
  ③ 更新既有终态用例：DONE 查询现「既回文本又补推卡」（文本仍在 + push 到 created_by），in-flight(RUNNING/CROSSING)不推也不抢闸门，tr- FAILED 文本用例补断言 push。
  未发现源码问题：`_push_terminal_result_card` 的 NX 闸门去重、created_by→配置频道回落、失败卡重试按钮均与在线线程/对账一致。

## 项目要点（历史踩坑，审计/测试重点）
- STS/RAM 多租户凭证；`ADMIN_FEISHU_OPEN_ID` 必须是本 app(cli_a962...) 的 open_id。
- 飞书卡片回调**至少投递一次**且用户会连点 → 幂等锁 / `launched` 标记 / `SET NX` 去重。
- 后台轮询线程随容器重启而死 → `refresh()` 手动自愈 + 调度器 `_reconcile_dataflow_once` 周期对账兜底。
- 跨云/数据流动参数格式（真机反查）：vePFS `DataStorage`=裸桶名、`DataStoragePath`/`SubPath` 非空须首尾带斜杠。
- 部署：`deploy.ps1`（git archive HEAD → scp → 容器 restart）；改 .env 需 `up -d --force-recreate`。测试多在服务器容器 `aiops-bot` 里跑。

## 本会话已完成（待 tester 覆盖 / auditor 复审）
- `d323e07` fix(vepfs)：CreateDataFlowTask 参数对齐真机（裸桶名 + 路径首尾斜杠）。
- `c3c538c` fix(transfer)：跨云迁移 submit/confirm 幂等，止连点起多线程+刷卡。
- `4219cc3` fix(transfer)：确认后原地替换为"进行中"卡 + 重复解析回进度卡。
- `ca67c30` feat(scheduler)：数据流动/迁移在途任务对账，重启后完成也自动通知。

## 日志
- [2026-07-09] [TESTER] #45 + #46 批 补测完成（本地全量 **757 passed / 1 skipped / 9 deselected / 0 failed**，新增约 49 例）。
  改 `test_cpfs_feishu.py::test_confirm_handler_launches`：确认默认 reply_v2=True 现回 progress_card_v2（schema 2.0、无按钮/表单），改断言由 `query_cpfs_progress` 按钮→2.0 纯展示卡（同 #33 改法）。
  新文件：
  `test_dataflow_created_by_backfill.py`（#45，cpfs+vepfs 参数化）：幂等分支 空created_by+真open_id→回填并_save / 已有非空→不覆盖且不_save / open_id空→不写空不_save / FAILED→走新建分支 created_by=本次open_id；confirm handler 空→回填后_save(RUNNING)、已有非空→不改。
  `test_dataflow_progress_card_v2.py`（#46-A2，cpfs+vepfs）：progress_card_v2 schema2.0 无按钮/表单、缺可选字段不 KeyError；confirm 默认回 2.0、retry(reply_v2=False) 回 1.0（无 schema 键）。
  `test_settings_validate.py`（C3）：三新字段登记+缺失时列出+齐全不列。
  `test_redis_client_timeouts.py`（C4）：收集期抓原 get_redis，断言构造传 socket_timeout=5/health_check_interval=30/retry_on_timeout=True + 既有参数不回归 + fakeredis 兼容。
  `test_get_oss_bucket_region.py`（C5）：传region→oss-<region> 不探测 / 不传→调 _detect_region_endpoint(auth,bucket) / 探测抛错→回退默认地域 / 无 auth→None。
  `test_ram_pagination.py`（C6）：两页 is_truncated+marker 合并全量+首页无marker二页带M1 / 单页一次调用 / truncated但无marker不死循环 / 异常返[]。
  `test_notify_token_cache.py`（B3）：首取→缓存命中不再HTTP / 过期重取 / expire=7200 提前≥5min / code!=0 抛错不污染缓存（requests.post 计次）。
  `test_capacity_bitable_list_guard.py`（B5）：_list_records code!=0 抛 RuntimeError / code==0 空返[] / 分页 has_more；write_scan 快照阶段列举抛错→False零写、厂家/批次阶段抛错→False零厂家批次写、正常空表→True 有 create。
  **未发现源码 bug。** 一条 Low informational 已发 dev：`retry_on_timeout=True` 在 redis-py 6.0+ 已 deprecated（prod=7.4.0，仍功能可用但客户端构造时会 emit DeprecationWarning），非阻塞，现代写法是 Retry 对象。
- [2026-07-09] [TESTER] #44 vePFS 终态查不到+友好错误 补测完成（本地 18 passed）。新文件 `tests/unit/test_vepfs_dataflow_pagination.py`：
  P1 `query_task` 4 例——请求带 `page_number=1/page_size=100`+解析终态计数(3dd16c 形态 Finished/0文件) / 带进度计数抽取 / `total_count>0` 但列表空→打 warning 不抛返空 status(旧 bug 现场) / 列表空且 total_count=0→静默不 warn。
  P2 `poll_once` 5 例——Finished→DONE / Failed→FAILED(含 error) / Running→保持 RUNNING / 歧义串 `FinishedWithFailures`+`CompletedWithErrors`+`Unsuccessful` 三参→FAILED(先判 failed 的意义)。
  P4 `_err` 8 例——BucketMaybeNotExist/SourceStoragePrefix/BucketName 三码给人话+附码 / Error 直挂顶层也解析 / 未知码 `Code：Message` / 无 msg 只回 Code / 非 JSON 原样 / 超长截 300 / 无 body 回退 str(e)。
  **注意**：起初本地反查 `is_failed('Unsuccessful')=False`（`_FAIL_HINTS` 无匹配项）→'Unsuccessful' 会被误判 DONE，正欲上报；随后发现 dev 已在 `_FAIL_HINTS` 补 `'unsuccess'`（diff 可见，对齐 MED-A3），该串现两边都命中→换序后正确判 FAILED。**修复完整，无源码问题。**`tests/unit/test_conversation_summary.py` 重写 11→24 例：
  MED-3 `_load_history` 从不注入摘要(即便有摘要key也只返原文·无SystemMessage) + `_load_summary` 取文本/无key""/redis down"" /
  MED-1 直接测 `_fold_into_summary`(压缩成功写摘要+7天TTL·滚动合并旧摘要·失败保留旧·LLM空串不写·坏JSON/空noop·超长截SUMMARY_MAX_CHARS) +
  `_save_turn` 桩 `threading.Thread`(run_sync=False验仅调度daemon线程+同步ltrim到20+压缩未跑；run_sync=True验摘要落库·压缩失败保留旧) /
  MED-2 `_compress_summary` 走 get_cloud_llm(timeout=SUMMARY_TIMEOUT·temperature=0)+失败返"" /
  off-by-one 改按 FOLD_TRIGGER(30)：29不折叠(llen29不ltrim)·30移出10条(首=OLD10)·32移出12条(首=OLD12)，验 total-MAX_HISTORY-1 /
  messages `_process_message` 摘要作前缀拼进 agent_input(有摘要含前缀+正文+原始消息且顺序在前；无摘要=纯base_input) / clear删两key / redis down降级。
  **改 `test_history_ttl.py::test_truncation_unchanged`**：FOLD_TRIGGER=30 后 15 轮(30条)触发折叠且用真 `threading.Thread`，加同步假线程桩(+get_cloud_llm桩)避免真起后台线程在 teardown 后跑/走网络 flaky。
  全量 pytest **694 passed / 1 skipped / 9 deselected / 0 failed**。未发现新源码问题；三处改动逻辑与断言一致。
- [2026-07-09] [TESTER] #43 滚动摘要记忆 补测完成（本地跑）。新文件 `tests/unit/test_conversation_summary.py`（11 例全绿）：
  不溢出零压缩+load无SystemMessage / 溢出触发压缩(get_cloud_llm恰1次·入参含被移出最旧q0a0·摘要写入·ltrim到20·两key≈7天TTL) /
  load带摘要首元素SystemMessage含摘要文本其后原文 / 压缩失败保留旧摘要不抛仍ltrim / clear删两key / redis down降级(load[]·save·clear静默) /
  边界(正好20不压缩、21移出1条、22移出2条，验 total-MAX_HISTORY-1 无 off-by-one)。
  **改 `test_history_ttl.py::test_truncation_unchanged`**：#43 后 15 轮溢出会真调 get_cloud_llm，加桩避免真调 LLM（原测无桩会走网络，虽被 try 吞但慢）。
  全量 pytest **681 passed / 1 skipped / 9 deselected / 0 failed**。off-by-one 正确、确用 get_cloud_llm（非 edge）。未发现新源码 bug；两条 informational 已发 dev。
- [2026-07-09] [TESTER] #40+#41 补测完成，本地全量 **649 passed / 1 skipped / 9 deselected**（新增 14 例，0 fail）。
  #40 新文件 `test_dsw_idle_stop_toggle.py`(6)：默认 flag=False；flag 关时 将到期(warned=False)→无警告卡+不 stop+GPU 空转仍提醒 / warned+超时→仍零 stop 零流转 / GPU 忙→静默；
    flag 开时 将到期→发警告卡+落 warned 不 stop / warned+超时→`manage_pai_dsw(stop)`+「已自动停止」文本+流转完成+清 redis。
    **注意 GPU 空转段有 `if not warned` 前门**：flag 关时该函数永不置 warned 故空转照跑，warned=True 与空转提醒天然互斥，故按场景拆开构造���任务描述把两条件写在一句里，实际不能同 state 共存）。
  #41 三处目标一致（都 `created_by`，空则 "" 降级配置频道）：`test_dataflow_reconcile.py`+2（对账终态首参=created_by / 缺省=""）、
    `test_query_async_refresh.py`+3（`_async_refresh_and_push` 终态+进度两分支首参=created_by / 缺省=""）、
    `test_transfer_confirm_idempotent.py`+3（在线 `_on_update` 结果卡+错误文本首参=created_by / 缺省=""）。均断言 `_send_card`/`_send_text` call_args[0][0]。
  未发现新源码问题。cpfs/vepfs/bucket 在线 `_on_update` 未逐链补（改法与 transfer 逐字相同，已由 transfer+reconcile+async 三处覆盖证明「三处一致」）。
- [2026-07-09] [TESTER] #38 补测完成，本地 venv 全量 **635 passed / 1 skipped / 9 deselected**（新增 42 例）。
  三文件：`test_wants_metrics_chart.py`(24：真值表+_process_message 不命中走纯文本不触云查询/命中发带图卡)、
  `test_agent_executor_limits.py`(6：两 executor iter=8/time=60/verbose=False + 常量 + 上限跟随常量)、
  `test_history_ttl.py`(4：expire 7天/每轮续期/截断不变/redis down 静默)。
  **注意源码与任务里贴的 diff 有出入（非 bug，是 dev 更晚的细化）**：`_METRICS_CHART_RE` 已删掉裸 `gpu`
  （源码注释：「查我的 gpu 工单」是工单问句，不该塞监控图）；已按现行为写 case 并锁死此设计。
- [2026-07-08] [DEV] 建立协作机制：拉起 tester/auditor 队友，定角色边界与通信协议。
- [2026-07-09] [TESTER] 基线 521 passed/1 failed(test_gpu_distribution，预存、与四修复无关)/9 deselected。对四修复+2e5f01c 补 20 条对抗/回归用例（bucket refresh 6 / 对账+NX闸门+specs契约 7 / confirm 幂等·审批·SINKING·confirmcard锁·_on_update只推一次 7），容器内 3 文件 43 passed。
- [2026-07-08] [AUDITOR] 复审 ca67c30/c3c538c/4219cc3。高危：bucket_transfer 无 updated_ts→对账 stale 门失效重复推卡；
  中：poll hang 超阈值双推、2.0→1.0 原地卡未真机验证；低：launched-但-NEW 孤儿不救、specs 导入不隔离。
- [2026-07-08] [TESTER] 基线 521 passed / 1 failed（`test_gpu_distribution::test_dist_url_and_summary_card`，
  与本次四修复无关，dist_url 改真实签名 token 后 mock 未同步）/ 9 deselected。给出四修复覆盖缺口清单。
- [2026-07-08] [DEV] 已修审计发现（提交 `2e5f01c`，已部署，23 相关测试绿）：
  ① bucket `_save` 每次刷 `updated_ts`（修高危）；
  ② 新增 `dsw_scheduler._claim_dataflow_notify`（Redis `SET NX dataflow:notified:{job_id}`，30天TTL）——
     **在线线程 on_update 终态推送 + 对账推送共用同一把闸门，谁先到谁推，跨线程只推一次**；
     四个在线 handler(transfer/cpfs/vepfs/bucket)终态分支都改走它；对账里 notified 标记先落盘、推送前抢闸门。
  ③ reconcile `key[len(prefix):]`。
  → **tester 注意：写测试请对齐 `2e5f01c` 后的新行为**（bucket 有了 `refresh()`+`updated_ts`；终态推送受 NX 闸门约束：
     同 job_id 第二次推被吞）。`test_gpu_distribution` 不在本轮范围，先别动。
  遗留：中危 200830（2.0→1.0 原地替换）需真机点一次确认；低危 launched-但-NEW 记录在案暂不改。
- [2026-07-08] [AUDITOR] 复审 `2e5f01c` 通过，NX 闸门无新双推竞态，不用回滚。剩余（低，非阻塞）：
  ① claim-before-send 无回滚：抢到闸门方 `_send_card` 抛异常→闸门已占、卡没发、对账因终态跳过→该完成卡永久丢
     （孤儿路径本就如此，非新增；用户仍可手动“查询”自愈）。可选：send 失败时 `delete` 闸门键放开。
  ② cpfs hang>180s 边界下对账与在线各调一次 `_cleanup_ephemeral`（不同 dict）→ 第二次删已删的 dfid，try/except 吞，安全。
  ③ nit：`_TTL_NOTIFY` 定义在 `_claim_dataflow_notify` 之后，运行期无碍但顺序略脆。
  审计对两遗留的建议：
  - 200830：**把确认后的“进行中”替换卡也做成 schema 2.0**（去掉 submit 按钮），同家族替换从根上避开降级拒绝，
    比“真机点一次核验”更可靠——这是唯一没被单测覆盖的点。
  - launched-但-NEW：可加兜底但**对账绝不能自己 submit**（保持只读）；只 `清 launched=False`，重发仍靠
    `transfer:launch` NX + MGW job_name 幂等；但确认卡已被原地换成无按钮的进度卡→需在进度卡上留“重新下发”按钮，或接受用户重新发起。窄窗口，排后。
- [2026-07-09] [RESEARCHER] 查证 #30（报告 `docs/collab/research/feishu-card-inplace-replace.md`）：
  **飞书 200830 = 2.0 卡不能原地更新为 1.0 卡；原地替换 data 的 schema 必须与最初那张卡一致**
  （出处 open.feishu.cn/document/feishu-cards/card-callback-communication）。→ 四条链路的确认 handler
  「2.0 确认卡→1.0 进度卡」原地替换**全部命中拒绝、静默失败**（业务不中断故线上看着没事）。
  代码内早已自相矛盾：`_h_query_progress_by_id` 注释已说明这点、查询路绕开了，确认路没绕。
  正解：①确认后的“进行中”卡也做成 schema 2.0（去按钮，同家族替换，可单测，最佳）；②或回 toast+另推 1.0 新卡（同查询路，改动最小）。
- [2026-07-09] [DEV] 据此确认：4219cc3 的确认卡原地替换实为 no-op（被 200830 拒），且我同时删了中间态 progress 推送
  → **确认后到完成前用户没有”进行中”反馈**（真 bug，非纯外观）。连点仍安全（launched+NX 锁挡住）。待定：按①修，范围 transfer or 四条全改。
- [2026-07-09] [DEV] 修 P-1（先 transfer 一条）：按方案①加 `core/transfer/cards.py::progress_card_v2`
  （schema 2.0 纯展示卡、无 form/按钮）；`_h_confirm_transfer` 新增 `reply_v2=True`，确认路径(2.0确认卡)回
  progress_card_v2（2.0→2.0），`_h_retry_transfer` 传 `reply_v2=False` 保持 1.0→1.0（结果卡是 1.0）。
  只改这两处，幂等/launch 锁/NX 闸门逻辑不动。→ tester 补测（confirm 回 schema 2.0 / retry 回 1.0 / 既有 patch progress_card 的用例需改）；auditor 复审。cpfs/vepfs/bucket 三条同坑，本轮先不动，待 transfer 真机验后照搬。
- [2026-07-09] [AUDITOR] 复审 #33 transfer 200830 修复通过：2.0→2.0 / 1.0→1.0 同家族闭合，reply_v2 keyword-only 不可污染，early-return 皆纯 toast 无换卡，v2 字段访问与旧卡同源无新 KeyError，幂等/NX/线程未动。提醒：v2 卡若日后加查询按钮须回 2.0（现 query handler 回 1.0，会再触 200830）。cpfs/vepfs/bucket 三条同坑仍未修（本轮不动）。
- [2026-07-09 15:10] [TESTER] #37 完成（本地跑，容器仍旧源码）。新增 3 文件 + 改 1 用例，全绿：
  `test_dataflow_retry_reset.py`（HIGH#1，10）：四链 retry 清 launch 锁 + `dataflow:notified` 闸门（fakeredis 断言）、
    job 不存在不误删、cpfs/vepfs 确有 `_save(stage=NEW)`（此前不落盘）；
  `test_oss_list_islice.py`（HIGH#2，3）：注入超大 ObjectIterator，断言只 next max_keys 次、迭代器未耗尽（还能取第 11 个）；
  `test_query_async_refresh.py`（HIGH#3，24）：四链 sync 只读 get_job 秒回、sync 内 refresh 抛错守卫证明不 poll、
    终态不起线程；`_async_refresh_and_push` 无变化不推/推进推/终态经 `_claim_dataflow_notify` 去重/refresh None 不推；
    `_h_query_progress_by_id` 非法前缀 error toast、合法前缀「正在查询」+ 后台推卡、8s 双击去重；
  `test_submit_async.py`（HIGH#4，8）：gpu/ak 提交 sync 无网络调用（recorder=0 + 换成抛异常仍成功回卡）、
    AK 卡两分支渲染不引用已删 `display`；配额/缺字段/坏 AK 是同步 error 不起线程。
  改 `test_cpfs_feishu.py::test_query_progress_by_id_routes_to_cpfs`：由同步 refresh→异步，加同步线程桩、断言 toast「正在查询」。
  全量 `pytest` 593 passed/1 skipped/9 deselected/0 failed（含 test_gpu_distribution，已不再红）。未发现新源码问题。
- [2026-07-09 14:35] [TESTER] #33 完成。改 `tests/unit/test_transfer_confirm_idempotent.py`：
  ① 修 `test_confirm_return_card_sinking_stage`——confirm 默认 reply_v2=True 走 progress_card_v2，改 patch/断言 `progress_card_v2`（原 patch progress_card 已失效）。
  ② 新增 7 条回归：confirm 成功返回 `card.data.schema=="2.0"` 且无 form/button；retry 路（reply_v2=False）返回卡无 `schema` 键（1.0）；progress_card_v2 对 SINKING/CROSSING/未知 stage（参数化3）+ 缺 same_name_policy 均不抛 KeyError；confirm 六个 early-return（缺job_id/任务不存在/stage非法/已launched/超阈值非管理员/NX锁被占）全部只回 toast 不带 card 键。
  容器为已部署旧源码（无 dev 未提交的 progress_card_v2 改动），故本地跑：该文件 17 passed；连同 test_bucket_transfer/test_dataflow_reconcile 三文件 50 passed。未发现新源码问题。
- [2026-07-09] [TESTER] #42 补测完成（本地跑）。新文件 `tests/unit/test_dataflow_mount_strip.py`（21 例，全 passed）：
  CPFS `_strip_mount`（含前缀/无前缀/`/cpfs`本身→`/`/`/cpfsdata`不误剥/尾斜杠配置归一）；
  **make_plan(fs_id=) 关键回归**——preheat `/cpfs/a/b/`→cpfs_dir & dst_directory 均 `/a/b/`（不含 /cpfs 不双套）、
  sink 方向 directory 同样剥净、`/cpfs`本身→`/`、`/cpfsdata/x/` 不误剥；无 fs_id 的 cpfs:// scheme + 裸 /cpfs/ 全路径都剥净；
  vePFS `_strip_mount('/vepfs/wzh/')=='/wzh/'` + 不误剥 `/vepfsdata`；make_plan(fs_id=) 双向 + plan_from_addresses 源/目的两方向
  经 make_plan 兜住、sub_path 均不含 /vepfs；settings 默认 `CPFS_MOUNT_PREFIX=/cpfs`、`VEPFS_MOUNT_PREFIX=/vepfs`。
  回归确认既有 test_cpfs_dataflow_orchestrator + test_vepfs_dataflow 45 passed。未发现新源码问题。
