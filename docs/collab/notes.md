# 协作看板 — dev / tester / auditor

三个 Claude 协作开发本项目（AIOps 飞书 Bot）。**每次开工先读本文件 + 根目录 `CLAUDE.md`。**

## 角色与硬边界（防止两个 Claude 改同一文件互相覆盖）

| 角色 | 谁 | 能写什么 | 不能干什么 |
|------|----|---------|-----------|
| **dev** | 主会话（编排者） | `core/** config/** tools/**` 源码、部署 | — |
| **tester** | 队友 | 只写 `tests/**` | 不碰源码；要改源码 → 交给 dev |
| **auditor** | 队友 | 什么都不写（只读） | 不 Edit/Write/提交；发现 → SendMessage 给 dev |
| **researcher** | 队友 | 只写 `docs/collab/research/**` | 不碰源码；**研究**(查阿里/火山/飞书/LangChain 文档+取证)+**侦察**(只读探查线上机器/环境/基建现状:SSH 盘点服务/容器/配置/依赖、连通性探测、环境取证)，结论带出处/证据 SendMessage 给 dev |
| **planner** | 队友 | 只写 `docs/collab/planning/**` | 不碰源码/测试；维护路线图+backlog+任务板，拆 epic→task（带验收+owner+依赖），排优先级 |
| **docwriter** | 队友 | 写 `README.md`/`CHANGELOG.md`/`docs/**`（不含 `docs/collab/**`） | 不碰源码/测试；以 CLAUDE.md 为事实源更新用户/项目文档，新功能靠前 |

源码的唯一改写者是 dev。永不并发编辑同一文件。

## 标准流水线（每个功能都走，不跳步）
**researcher 查资料 → planner 计划 → dev 开发 → tester 测试 → auditor 审计。**
> **提交闸门（硬规）**：任何改动必须 **auditor 审计通过、无阻塞项** 才能 `git commit`。dev 不得跳过审计直接提交。文档类改动（docwriter）也要过一遍事实/准确性核验再提交。
> **researcher 不只是"查资料的"**：摸清某台机/某套环境/某个线上系统的真实状态（迁移前盘点、排障取现场、上线前核对）也派 researcher 当**只读探针**（SSH 盘点、连通性探测、环境取证），产出"必搬/必改/风险/待确认"给 planner。

### 验收流程（Definition of Done — 每个改动"完成"须全绿，tester 主责、dev 收口）
1. **单测绿**：目标测试 + 全量 `pytest` 全通过（0 failed）；新增覆盖边界/对抗/回归（幂等/并发/重启/错误分支/边界值）。
2. **既存失败甄别**：与本次无关的既存失败（如缺 matplotlib、gpu_distribution）单独指出并基线复现，不混入、不甩锅。
3. **端到端验收**：有运行时面的改动用 `verify`/`run` 驱动真实流程（本项目常在容器 `aiops-bot` 里验），不只看单测。
4. **审计过闸**：auditor 复审无阻塞（= 上面提交闸门硬规）。
5. **回报可核**：通过数 + 覆盖点 + 未覆盖项/已知风险；bug 给 `file:line + 复现 + 期望/实际`。
任一未过 → 任务留 `in_progress`、说清卡点，别标 completed、别提交。
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

## #50 查询按 ID 推结果卡与自动完成通知重复（真机：两张一样的完成卡，dev 已改，待 tester+auditor，未提交）
真机：用户按 task_id 查询已完成的 tr- 迁移，收到**两张一模一样的完成卡**。根因：`_h_query_progress_by_id._bg` 终态推卡 `_send_card` **未走 `dataflow:notified` NX 闸门**（只有防同一次点击双投递的 `cpfs:query:{open_id}:{jid}` 8s 锁）；而在线线程 on_update / 对账 / 我 #49 的文本查询 `_push_terminal_result_card` 都走了那把闸门。于是「自动完成通知」+「查询回复」各推一张相同结果卡。（该任务部署重启后在线线程死→对账补推完成通知，用户又正好点查询→两张撞一起。）
- 修：`_bg` 终态分支 `if _claim_dataflow_notify(jid): _send_card(result_card)` else 只 `_send_text` 回执（见上方）；进度卡(非终态)不去重照推。与 `_async_refresh_and_push`/文本查询路径行为对齐——全链路对同一 job_id 终态结果卡「至多推一张」。
- actions.py 编译过。tester：终态查询 NX 未占→推 result_card、已占→只回文本不推卡；进度态→照推 progress_card 不占闸门；双投递仍被 cpfs:query 8s 锁挡。auditor：确认四路径（在线/对账/文本查询/表单查询）现共用同一闸门、终态至多一张；查询者≠发起人时表单查询 gate-blocked 仅回文本（可接受）。
- [2026-07-09] [TESTER] #50 delta 补测（据审计 Low 改后：else 文案改 error/完成两态 + Low-1 send 失败放开闸门）：全量 **815 passed / 1 skipped / 9 deselected / 0 failed**，同文件 22→26 例。case 2 文案断言放宽为「含 job_id + stage」（不锁旧措辞）；新增 2b（FAILED 带 error 的回执含 error）、2c（claim True 后 `_send_card` 抛异常→`delete(dataflow:notified:{jid})` 放开闸门、异常不冒泡、resp 仍 success）。claim/delete 均用 `job["job_id"]`（桩 job job_id==jid，断言无碍）。下方原 22 例明细为改前版本、仍适用其余分支。
  新文件 `tests/unit/test_query_by_id_notify_gate.py`，桩同步线程 + `dsw_scheduler._claim_dataflow_notify`/`_send_card`/`_send_text` + 三链 `orchestrator.refresh`/`result_card`/`progress_card` 哨兵：
  ① 终态(DONE/FAILED)×三前缀+claim True→`_send_card` 推 result_card、首参=open_id、不发文本、闸门被抢一次（参数化 6）；
  ② 终态×三前缀+claim False（已被别��径推）→抢闸门但**不推卡**、改 `_send_text` 回执含 jid（参数化 6）；
  ③ 非终态(CROSSING/RUNNING)×三前缀→推 progress_card、**从不调 `_claim_dataflow_notify`**、不发文本（3）；
  ④ refresh 返 None×三前缀→`_send_text`「未找到」含 jid、不推卡不碰闸门（3）；
  ⑤ FAILED 用真实 transfer.result_card→抢到闸门推的卡含 `retry_transfer` 按钮（1）；
  ⑥ 双投递被 `cpfs:query:{open_id}:{jid}` 8s NX 锁挡：第二次同点击只回「查询中…」info toast、不进 _bg（卡数不变、闸门只抢一次），不同 jid 不误挡（1）；
  ⑦ 前缀非法→error toast / 空 jid→{} noop 回归（2）。
  **既有 `test_cpfs_feishu.py::test_query_progress_by_id_routes_to_cpfs` 无需改**：它用 RUNNING（非终态）job 断言 `_send_card` 推 progress_card，恰是新逻辑非终态分支的行为，仍绿；无任何既有用例断言 `_bg` 无条件终态 `_send_card`。
  **未发现源码问题**：`_bg` 终态经 `_claim_dataflow_notify` NX 闸门单推、抢不到只文本回执、非终态不碰闸门，与在线线程/对账/`_push_terminal_result_card` 四路径行为一致。

- [2026-07-09] [AUDITOR] #50 复审通过、无阻塞、满足 commit gate。五处终态推卡共用 dataflow:notified 闸门、同步原地替换卡非新推无重复、8s 锁挡同点双投、进度态不占闸门。3 Low+1nit 记账：①_bg 终态 send 无 try 兜底(claim-before-send 无回滚孤儿)；②跨用户查询推 open_id 却抢 created_by 同闸门→发起人兜底通知被吞(自查 net 正确,可接受)；③else 文案「见上方」跨会话/send失败误导；nit claim 用 jid 可改 job["job_id"]。
- [2026-07-09] [DEV] 据审计修 Low-1(终态 claim+send 包 try、send 失败 delete dataflow:notified 放开闸门)+Low-3(else 改为回执实际结果 stage/error、不再假设「见上方」)+nit(claim 改用 job["job_id"])。Low-2 跨用户不对称按 notes 判断接受不改。已重编译，请 auditor 复核 delta、tester 据 else 文案改动更新断言。

## #51 SSH 迁移链 杭州OSS→新加坡→泰国（T0–T3 骨架，dev 已改，待 tester+auditor，未提交）
按 planner `planning/thailand-transfer-chain.md`。**所有前置真机验证全绿**（bot→SGP root@43.98.203.59:22 密钥通/ossutil2.2.2 配好能读源桶/rsync 有；SGP→泰国 wuji@203.156.3.194:40002 免密通；泰国目标 `/mnt/data04/296834/Wuji-Algorithm@wuji.tech/data` 已 chown wuji、免 sudo=**方案 A 锁定**）。
- 新包 `core/ssh_transfer/`：`paths.py`(解析 oss://bucket/prefix/→Plan)、`engine_ssh.py`(paramiko 连接层：私钥 Fernet decrypt 内存加载 RSAKey、host key 固定 SGP_SSH_HOST_KEY 禁 AutoAdd fail-closed；`run` 短命令 drain 后取 exit；`start_stage1/2` nohup+pid/rc/log marker、redirection 全远端、shlex.quote 全参数；`poll_stage` 一次 SSH 取 RC=/ALIVE/DEAD→status；rc 映射 stage1==0 / stage2∈{0,24}；`estimate_source` 走 SGP ossutil du 容错解析)、`orchestrator.py`(状态机 NEW→STAGE1→STAGE2→DONE/FAILED、Redis `ssh:transfer:*`、job 前缀 `sgp-`、`_save` 每次刷 updated_ts、create_job_record 幂等+回填 created_by 照 #45、段1 rc=0 落盘后起段2、run_to_completion 两段轮询、refresh 自愈、retry 段1已成功只重段2)、`cli.py`(plan/apply/status)。
- settings 加整块 SSH_TRANSFER_*/SGP_*/THAI_*（真机值做默认，私钥 SGP_SSH_KEY_ENC 仅 env、Fernet 密文）；SGP_SSH_KEY_ENC 进 _REQUIRED_FIELDS。requirements 加 paramiko>=3.4.0（**部署要 up -d --build 重建镜像**，非普通 deploy）。
- 6 文件 py_compile 全过。**私钥 /root/bot_sgp_rsa 在 bot 宿主机，SGP_SSH_KEY_ENC 密文待 T10 生成注入 .env。**
- tester：paths 解析/规整/拒非 oss；orchestrator job_id 幂等+created_by 回填+needs_approval 阈值+poll_once 状态转移(mock engine)+run_to_completion 两段(段1 DONE→段2 / 段1 FAILED→不进段2)+retry 段1已成功只起段2；engine `_rc_ok` 映射、start_stage1/2 生成命令含预期片段(mock run 捕获)、poll_stage 解析 RC=/ALIVE/DEAD、estimate 解析、缺 SGP_SSH_KEY_ENC/HOST_KEY 报错。**paramiko 未装→测试须 mock engine_ssh.run/_client（engine 内 lazy import paramiko，导入模块不需 paramiko）**。auditor：私钥绝不落盘/入日志、host key 固定禁 AutoAdd、shlex.quote 防命令注入(source_bucket/prefix 全经 quote)、nohup/marker 正确性、rc 语义、settings 默认无敏感、方案A 段2 免 sudo。

- [2026-07-09] [TESTER] #51 SSH 迁移链 T0–T3 骨架补测完成（本地全量 **879 passed / 1 skipped / 9 deselected / 0 failed**，新增 3 文件 64 例）。paramiko 未装，全程 mock（命令类桩 `engine_ssh.run` 捕命令串、守卫路径注入空 stub paramiko），Redis 走 fakeredis。
  `test_ssh_transfer_paths.py`(19)：合法目录/自动补尾斜杠/单级前缀/根前缀空/裸桶名空前缀/去空白/scheme 大小写；拒 tos/cpfs scheme、缺 bucket(`oss://`/`oss:///`)、空串/纯空白/None/无 scheme；SshPathError 继承 ValueError；build_plan 委派+拒非法。
  `test_ssh_transfer_orchestrator.py`(27)：`_job_id` 同 plan 同天一致+sgp- 前缀+异 prefix 异 id；create_job_record 首建 NEW/created_by/落库、幂等返旧不复位、空 created_by 回填并落库、非空不覆盖、空 open_id 不写空、FAILED 走新建复位；needs_approval 默认 1TB 边界+自定阈值+坏值回退 1TB；poll_once 段1DONE→起段2/段1FAILED→不起段2/段2DONE→DONE/RUNNING 保持/终态 noop/引擎抛错保持在途；refresh 在途重查+终态不查+缺 job 返 None；run_to_completion 两段推进(on_update 见 STAGE1/STAGE2/DONE)+段1FAILED 不进段2+起段1即抛→FAILED+retry(stage1_rc==0)只起段2 不调 start_stage1。
  `test_ssh_transfer_engine.py`(18)：`_rc_ok`(段1只0/段2 0&24/23&255 失败)；start_stage1 命令含 ossutil cp+oss URI+--jobs+-u+--checkpoint-dir+nohup+rc/pid/log marker+SGP 挂载 dst；start_stage2 含 rsync+-a+--info=progress2+mkdir -p+`wuji@host:`+sudo true/false 切 --rsync-path+bwlimit 设置+缺 THAI_DEST_ROOT 报错；poll_stage RC=0→DONE/段2 RC=23→FAILED/RC=24→DONE/ALIVE→RUNNING/DEAD→FAILED/RC=非数字兜底1；estimate_source du 解析/无法解析→(0,0)/空→(0,0)；`_load_private_key` 缺 SGP_SSH_KEY_ENC 报错、`_add_host_key` 缺/畸形 SGP_SSH_HOST_KEY 报错。
  **未发现源码问题**：paths 规整/拒非 oss、orchestrator 幂等回填/两段推进/retry 跳段1、engine rc 语义与命令片段均与断言吻合。

- [2026-07-09] [AUDITOR] #51 T0–T3 复审：**HIGH 命令注入阻塞**——engine_ssh 泰国 ssh 双跳的 prefix 未过滤（`oss://b/foo; rm -rf x/` 经 ssh 双层 shell 在泰国生产机 RCE，shlex.quote 只护 SGP 单层）；MED-2 路径穿越(..)；MED-3 估算失败(0,0)→审批 fail-open；6 Low 记账。核心逻辑(私钥内存/host key fail-closed/rc/状态机/幂等/settings)PASS。
- [2026-07-09] [DEV] 据审计修：**HIGH-1+MED-2** `paths.parse_source` 加严格白名单（桶名 OSS 规范正则；prefix 每级 `^[A-Za-z0-9._\-]+$`、禁 `..`/`.`/空格/shell 元字符）——从源头挡住注入+穿越，已自证 `; rm`/`..`/空格/`$(id)` 全被拒、正常路径通过。**MED-3** estimate_source 返 `(b,n,ok)`，ok=False(SSH不通/没解析出大小)→`needs_approval(...,size_known=False)` fail-safe 当需审批；job 存 estimate_ok；CLI 相应改。Low：_load_private_key 校验解密结果含 `-----BEGIN` 否则明确报错(修 decrypt 静默透传误导)；host key 同时按裸 host 和 `[host]:port` 登记(改端口不误拒)；bwlimit 值 shlex.quote；orchestrator fmt_ts/fmt_duration 加 __all__ 声明(供 cards)。4 文件编译过。请 auditor 复核 delta、tester 补注入用例 + estimate 3 元组断言。
- [2026-07-09] [TESTER] #51 HIGH 命令注入 + MED-3 审批 fail-open 修复同步补测（全量 **911 passed / 1 skipped / 9 deselected / 0 failed**，3 文件 64→96 例，+32）。
  paths 安全白名单（新增）：拒 prefix 注入（`; rm -rf`/`$(id)`/反引号/管道/空格/`&`/`>`/引号/反斜杠/换行）、路径穿越（`..`/`.` 整段/中段）、非法桶名（大写/<3/单字符/首尾非字母数字/下划线/点）；真机风格 `oss://wuji-data-tran/wangyuran/AgiBotWorld-Beta/` 仍通过+prefix 规整、段内 `.`/`_`/`-` 合法。
  estimate_source 改 3 元组 `(bytes,objects,ok)`：engine 解析出 size→ok=True（含 size=0 空前缀 ok=True）/没匹配 size 行→`(0,0,False)`/空输出→`(0,0,False)`；orchestrator 透传+异常→`(0,0,False)`。
  needs_approval(bytes, size_known)：size_known=False→恒 True（与字节数无关）/默认 True 按阈值/边界/坏值回退 1TB。create_job_record 新增 size_known 参+落 `estimate_ok`（默认 True，size_known=False 落盘 False）。
  Low：_load_private_key decrypt 结果非 PEM/空串→SshTransferError（桩 utils.crypto.decrypt）。
  **未发现源码问题**：白名单正则收严不误伤真机路径、estimate 3 元组与 fail-safe 审批语义一致、PEM 校验挡明文透传。

- [2026-07-09] [AUDITOR] #51 修复复核：白名单+审批 fail-safe 基本到位，唯正则尾锚 `$` 放行结尾换行(`aaa\n/` 溜过)，一字符 `\Z`/fullmatch 修后过闸。
- [2026-07-09] [DEV] 据审计把 `_BUCKET_RE`/`_SEG_RE` 尾锚 `$`→`\A...\Z`（`$` 会在结尾换行前匹配）。自证：审计的确切绕过 `oss://b/aaa\n/`、`buck\net`、`a\nrm -rf x/` 全被拒；`_SEG_RE.match('aaa\n')`/`_BUCKET_RE.match('abcd\n')` 均 None；安全不变量「凡被接受的输入输出 bucket/prefix 只含 `[A-Za-z0-9._/-]`（无换行/元字符）」成立（含 `.strip()` 把纯尾换行归一成干净路径）。请 auditor 复核这一字符 delta。
- [2026-07-09] [TESTER] #51 尾锚 `$`→`\Z` 补 2 例回归：段内尾换行 `oss://bucket/aaa\n/`→SshPathError、桶名尾换行 `oss://bucket\n/aaa/`→SshPathError（两者旧 `$` 版会被误放行，`\Z` 收严后拒）。全量 **913 passed / 1 skipped / 9 deselected / 0 failed**（+2）。`\Z` 只收严、未破坏任何既有断言。

- [2026-07-09] [AUDITOR] #51 T0–T3 复审两轮收口：白名单 \A...\Z 闭合换行绕过、注入/穿越全拒、审批 fail-safe、Low 全修，过闸无阻塞。

## #51 续：T4–T6 飞书入口 + 目标子目录 + 进度/速度（dev 已改，待 tester+auditor，未提交；骨架 82de239 已提交）
用户确认触发词「数据迁移（泰国H200）」+ 首单 `oss://wuji-data-tran/ossutil_output/`→`.../data/test/`（**B 语义：内容铺进 test/，不套源目录名**）。两边目录都自动建（段1 ossutil 写挂载盘自动建→ossfs 建桶前缀；段2 rsync 前 ssh mkdir -p）。
- **目标子目录**（新需求）：`paths.Plan` 加 `dest_subdir` + `dest_rel()`（空=镜像源前缀）；`build_plan(source, dest_subdir)` **dest 也走同一道白名单**（`_norm_dest_subdir`，防泰国 ssh 双跳注入/穿越）；engine.start_stage2 用 `dest_rel`；orchestrator job 存 dest_subdir/dest_rel、job_id 纳入 dest_rel、dest_uri 用 dest_rel；cli 加 `--dest`。已自证 B 语义 + dest 白名单挡 `; rm`/`../`/空格/`$()`。
- **进度+速度**：engine `stage_progress`（tail 日志解析：段2 rsync `--info=progress2` 取字节+%+速率、段1 ossutil 取瞬时速率，`_RSYNC_PROG`/`_RATE`/`_speed_bps`）；orchestrator `_sample_progress`（poll_once 每轮采样，日志无速率则用相邻字节采样差算）+ `progress_line`（`已传X/Y(nn%)·速率nn/s·剩余约mm`，`_fmt_eta`）。进度卡显示它。
- **T4 cards**：entry_card(2.0 表单：源+目标子目录) / confirm_card(2.0，估算+审批门，超阈值 red+仅 admin) / progress_card_v2(2.0 纯展示含 progress_line，无按钮避 200830) / result_card(1.0，FAILED 带 retry_ssh_transfer)。
- **T5 messages**：`_is_ssh_transfer_intent`（泰国+迁移/h200；**排在跨云 transfer 意图前**，「数据迁移」会被 transfer 入口误抢）；`_JOB_ID_RE` 加 `sgp-`；`_handle_progress_query` 加 sgp- 分支（refresh+progress_line+终态经闸门补推）；`_push_terminal_result_card` 加 "ssh" 链。
- **T6 actions**：`_cfg_ssh_chat`；`_h_submit_ssh_transfer`(后台解析+估算→confirm_card) / `_h_confirm_ssh_transfer`(NX launch 锁+launched+created_by 回填+审批门+后台 run_to_completion+on_update 终态经 `_claim_dataflow_notify` 去重推 created_by；确认卡 2.0→progress_card_v2、retry reply_v2=False 只回 toast 避 200830) / `_h_retry_ssh_transfer`(重置+清 launch/notified 键+保留 stage1_rc 只重段2)。注册进 _ACTION_HANDLERS。
- 7 文件编译过；smoke：意图不误撞跨云、四卡 schema 正确、progress_line 出「已传/速率/剩余」、handlers 注册齐。**待部署（paramiko 需 up -d --build 重建镜像）+ SGP_SSH_KEY_ENC 注入 .env 才能真跑。**
- tester：dest_subdir B 语义+dest 白名单注入用例；stage_progress 解析 rsync progress2/ossutil 速率、_speed_bps 单位、progress_line 各分支+_fmt_eta；_sample_progress 字节差算速率；意图 `_is_ssh_transfer_intent`(命中泰国H200 变体/不误撞跨云)；_JOB_ID_RE sgp-；三 handler(submit 后台估算/confirm NX+审批门+created_by 回填+reply_v2 分支/retry 清键保留 stage1_rc)；cards 四件套结构。auditor：dest_subdir 经白名单（新用户输入面！注入复核）、confirm 审批门+admin、200830 规避（confirm 2.0→2.0 / retry 只 toast）、终态闸门去重、created_by 回填、意图排序不误抢跨云。

- [2026-07-09] [TESTER] #51 续（T4–T6 + 目标子目录 + 进度/速度）补测完成，全量 **1021 passed / 1 skipped / 9 deselected / 0 failed**（+108 例，paramiko 全程 mock、fakeredis）。新增 4 文件 + 1 例入 reconcile：
  `test_ssh_transfer_dest_subdir.py`(19)：build_plan('...ossutil_output/','test')→源前缀不变/dest_subdir='test/'/dest_rel()='test/'(B 语义)、空 dest 镜像源前缀、显式空==缺省、多级 'a/b'→'a/b/'、去首尾斜杠空白；dest 白名单拒 `test; rm`/`../etc`/`a b`/`a$(id)`/反引号/管道/&/>/引号/反斜杠/换行/中段穿越/单点(14 参数化)、段内 `.`/`_`/`-` 合法；job_id 纳入 dest_rel(异 dest 异 id、空==显式镜像、自定≠镜像)；create_job_record 存 dest_subdir/dest_rel/dest_uri(自定+镜像两态)。
  `test_ssh_transfer_progress.py`(30)：engine.stage_progress——段2 rsync progress2 取 bytes/pct/speed+取最后一次匹配、段1 ossutil 只稳取速率(bytes/pct=None)、无匹配全 None；_speed_bps 单位 K/M/G/T/空+大小写(7 参数化)、MiB/s==MB/s(4 参数化)；start_stage2 用 dest_rel(mkdir -p+rsync target 落 `<root>/test/`、源仍源前缀、空 dest_rel 镜像)；orchestrator._sample_progress(直接速率/字节差算速率含 time.time 桩/首采样无速率/set pct/引擎抛错 noop)；progress_line(全字段/有total无速率/spd=0 不除零/仅 bytes 无 total/完成无 ETA/空→采集中)；_fmt_eta(s/m/h/负数归零)。
  `test_ssh_transfer_feishu.py`(33)：_is_ssh_transfer_intent 命中「数据迁移（泰国H200）」全半角/泰国H200/迁到泰国/迁移到泰国服务器(6)、不命中「把 tos://a 迁到 oss://b」跨云话术/普通迁移/查询进度(5)；_JOB_ID_RE 认 sgp-(≥6hex)不认 <6；_handle_progress_query sgp- 分支路由到 ssh orchestrator.refresh+文本含 progress_line、终态 claim True 补推 result_card 到 created_by、claim False 抢闸门不推、refresh None「未找到」；_h_submit(空源 error toast/后台估算推 confirm 2.0/非法路径 _send_text)、_h_confirm(NX 锁 ex=120 首下发回 progress_card_v2·连点第二次 info 且只真下发一次/超阈值非 admin 拒且锁未占/created_by 空回填不覆盖/reply_v2=False 只 toast/缺job_id·不存在·stage 非NEW-FAILED early-return)、_h_retry(清 launch+notified 键·stage=NEW·保留 stage1_rc·转 confirm reply_v2=False/缺 job error)、handlers 注册齐。
  `test_ssh_transfer_cards_v2.py`(13)：entry(2.0+source/dest/submit 回调)、confirm(2.0 orange/审批 red+管理员字样/估算失败标/缺可选不 KeyError)、progress_card_v2(2.0 无 form/button+含 progress_line+缺进度「采集中」+各 stage 不炸)、result_card(DONE 无 schema 键·无重试/FAILED 无 schema·含 retry_ssh_transfer 按钮/缺可选不 KeyError/缺 error 显「未知」)。
  `test_dataflow_reconcile.py`+1：`_dataflow_reconcile_specs()` 含 "ssh"、active=={STAGE_STAGE1,STAGE_STAGE2}、cards 有 result_card、o 有 refresh/get_job/_save/_KEY_PREFIX/STAGE_DONE/STAGE_FAILED、chat 可调用。
  **未发现源码问题**：dest 白名单复用段级校验挡注入/穿越、B 语义 dest_rel、job_id/落库纳入 dest_rel、进度解析(rsync/ossutil/单位/采样差)、意图排序(SSH 排跨云前)、三 handler NX/审批/回填/reply_v2/清键、四卡 schema 与 200830 规避均与断言吻合。

- [2026-07-09] [AUDITOR] #51续 T4-T6 复审通过、无阻塞（无 HIGH）；7 重点全 PASS（dest_subdir 走同一白名单封死泰国 ssh 双跳、200830 规避、审批门、意图排序不误抢跨云、终态闸门去重+created_by 回填、进度/速度无除零、NX/后台/3s）。1 MED：ssh 链漏进 `_dataflow_reconcile_specs`→sgp- 任务重启无对账兜底补终态卡（数据不丢、仅自动通知）；建议同批补。5 Low 记账（confirm 后双推进度卡/retry authz 在 mutation 后/MB按1024/段1 bd=0 显示/重复 import）。
- [2026-07-09] [DEV] 据审计补 MED（T8）：`dsw_scheduler._dataflow_reconcile_specs` 加 ssh 链（active={STAGE_STAGE1,STAGE_STAGE2}、chat=SSH_TRANSFER_CHAT_ID||FEISHU_CHAT_ID、cleanup=None），docstring 注明 ssh 两段链 refresh 会自动起段2（重启续跑，靠 stale 门+NX 无双起、rsync 幂等）。已验 5 链齐（transfer/cpfs/vepfs/bucket/ssh）、result_card 齐。5 Low 记账不改（与既有链一致）。请 auditor 复核这一 spec delta、tester 加「ssh 在 specs + active 两段」用例。

- [2026-07-09] [AUDITOR] #51续 T4-T6 复审通过、无阻塞、满足 commit gate。7 重点全 PASS（dest_subdir 白名单/200830/审批门/意图序/终态去重/进度速度/3s）；MED（reconcile 漏 ssh 链）已补 spec 并复核正确（active 双段/result_card/refresh/chat/cleanup 齐、docstring 例外注解准、5 链齐、去重闸门同目标）；6 Low 记账（L1 双进度卡·L2 retry authz-after-mutation·L3 MiB/MB·L4 段1已传0B·L5 重复 import·**L6 ssh refresh 触发段2 与手动查询窄窗并发可选加 NX 锁**），皆与既有链一致、非阻塞。
- [2026-07-09] [DEV] L6 记为已知可选硬化（`_start_stage(STAGE2)` 复用 `ssh:transfer:launch` NX 锁），当前 rsync -a 幂等+自纠、极低概率，不churn 已过闸批次，留后续。

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

[2026-07-10 TESTER] JIRA_ENABLED/SCHEDULER_ENABLED 两开关补测：新文件 `tests/unit/test_jira_scheduler_switches.py`（17 例，本地 .venv 全绿）。覆盖 ① settings 默认(reload+dotenv 桩验 JIRA_ENABLED 默认 false/SCHEDULER_ENABLED 默认 true、TRUE/false 解析) + validate()(停用不列 JIRA_URL/PAT、启用列、非 Jira 字段如 API_KEY 不受开关影响)；② dsw_scheduler.start()(桩 Thread：停用不起 jira-poll 且 _ticket_thread 保持 None、dsw-check+morning-report 照常、stop() 不 NPE、日志「(Jira 停用)」；启用三线程全起+日志「Jira 轮询」；_running 已 True 早返回)；③ routes.health()(停用→jira=="disabled" 且 status ok 不 degraded、启用+缺配置→not_configured 不 degraded、启用+配齐+桩 requests 200→ok)；④ actions._h_submit_gpu_request()(停用→info 停用 toast·不建工单·不起线程·排在字段校验前、空 form noop、启用走原逻辑回处理中卡)。**改既有 `test_submit_async.py`**：autouse fixture 补 `JIRA_ENABLED=True`（否则 gpu 用例因新默认 false 撞进停用分支）。**注意 settings 默认测用 importlib.reload——已在 finally 恢复 config.settings.settings 原单例身份，防止 reload 换新实例污染其它模块的 `from config.settings import settings` 引用（health 首版正因此挂）。** 本地全量 **1042 passed / 1 skipped / 9 deselected / 0 failed**。未发现源码问题。
[2026-07-10 TESTER] GPU 申请第二入口（文本路径）JIRA 停用 gate 补测（auditor 发现漏 gate、dev 补两处后）：追加进 `tests/unit/test_jira_scheduler_switches.py`（+5 例）。① messages._process_message 发「申请GPU」文本：停用→回停用提示、不调 gpu_flow._send_gpu_card/_send_ak_register_card/_set_gpu_state、且早于 _is_registered 返回；启用+已注册→弹 GPU 卡+清 state；启用+未注册→弹 AK 卡+置 pending_gpu state。② gpu_flow._handle_gpu_request：停用→回停用提示、create_gpu_ticket 0 次、不起线程；启用→回「正在提交申请」+起后台线程，跑线程体确认建工单。桩 messages/gpu_flow 的 threading.Thread 避免 _auto_map_user/建工单真跑网络；mock_feishu_send 捕 messaging._feishu_reply。本地全量 **1047 passed / 1 skipped / 9 deselected / 0 failed**。未发现源码问题。
[2026-07-09 TESTER] SSH迁移录入卡 entry_card 补测：tests/unit/test_ssh_transfer_cards_v2.py 新增4例(泰国目标根 user@host:root/ 拼接+placeholder+结构回归+空/None不炸)。目标文件 21 passed。全量 1024 passed / 2 failed。两处 failed 与本改动无关且在 /app 基线同样复现：test_gpu_distribution::test_dist_url_and_summary_card、test_wants_metrics_chart（utils/chart_builder.py 缺 matplotlib，容器未装该模块）。已告知 dev。
[2026-07-10 TESTER] ssh_transfer 段1 ossutil 并发 flag `--jobs`→`--job`（真机 bug #2）补测：dev 已修 engine_ssh.py:163-164（`-r --job {jobs}`）。改 test_ssh_transfer_engine.py::test_start_stage1_command 旧断言 `'--jobs' in cmd`→`'--jobs' not in cmd` + `'--job ' in cmd`（带空格避误匹配复数）+ 补 `-r` 回归；新增 test_start_stage1_concurrency_flag_singular_not_plural（词边界正则 `--job(?!s)\s+\d+` 断言独立单数 token 后跟并发数、`--jobs` 恒不出现、复用配置 30）。其余回归断言（ossutil cp/oss URI/-u/--checkpoint-dir/SGP 挂载 dst/marker）原样保留。目标文件 35 passed（34→35，+1）；本地 .venv 全量 **1058 passed / 1 skipped / 9 deselected / 0 failed**。matplotlib 相关两例本地 .venv 装了 matplotlib 故通过（容器基线缺 matplotlib 会红，与本改动无关）。未发现源码问题。

[2026-07-10 TESTER] 收口全量 pytest 通过：**1060 passed / 0 failed / 1 skipped(h5py 缺,既存) / 9 deselected(integration)**，含 ssh_transfer 两修复(--job 单数 0b75a90 + estimate_source du 解析锚定 total 汇总行)。本地 .venv(pytest 9.0.3, matplotlib 3.10.8) 跑，跑两遍(随机序+no:randomly)均稳定。补 du 回归用例于 tests/unit/test_ssh_transfer_engine.py(核心断言 bytes==23208637 而非 3;兜底 total du size;空前缀(0,0,True);防回归表头+数据行无total→(0,0,False);空/不可解析→(0,0,False))。matplotlib 两测(test_gpu_distribution 7 passed / test_wants_metrics_chart) 本地绿——容器无 matplotlib 时才 red,属既存基线。注:服务器容器 aiops-bot 停机(Exited 137/OOM)已重启,但 .deployed_commit=7050f65 陈旧(engine_ssh 仍 --jobs),故未在容器跑本轮——建议 dev 部署 0b75a90 后容器复核。

[2026-07-10 DEV] ssh_transfer 首单验收收口 + 部署 bot-new。真机首单(oss://wuji-data-tran/ossutil_output/→泰国 .../data/test/ option B)抓出并修 4 处：`0b75a90` 段1 ossutil `--jobs`→`--job`(2.2.2 单数)、`aba09a1` estimate_source du 解析(旧正则命中表头 sum size 后 [^\d]* 跨行吞 object count=3→22MB 读成 3B、绕审批门；改锚 total object sum size/total du size + [:\s]+；researcher 真机反查 du 逐字输出见 research/ossutil_du_output_format.md)、`27b485a` deploy.ps1 目标 bot-server→bot-new(迁移后唯一在线机)、`bd1406f` deploy.ps1 首次部署无 .deployed_commit 新机 NPE("$()".Trim() 兜 null)。两 app 修复经 tester(1060 passed)+auditor(无阻塞)过闸；两 deploy.ps1 改动属运维脚本(一为用户直接指示、一为部署当场实测修复)未走流水线。**已部署 27b485a 到 bot-new(8.222.149.27)并 live 验证**：health status:ok/jira:disabled、容器内 --job+新 du 正则在位、paramiko 5.0.0+ssh_transfer 全模块 import OK(迁移+OOM 后完好)、du 端到端走部署代码 estimate_source→22.13MB/3 对象/ok=True(非 3B)。旧机 bot-server 仍停机作回滚;bot-new 1.8G 内存偏紧、aiops-bot 曾 OOM(Exited137)需留意。

[2026-07-10 DEV] paramiko 烤入镜像坑根治 + Dockerfile 纳入 git。Dockerfile 原只在服务器 `/root/langchaindev/Dockerfile` 未版本化，现纳入仓库(commit `a701c61`)并调层序：torch 安装 RUN 块挪到 `COPY requirements.txt` 之前作独立缓存层(torch 版本写死、不依赖 requirements → 改 requirements 不再连累 torch 重下 200MB+)。auditor 逐行对比服务器原版：除 torch 块位置+2 行注释外无改动、构建语义不变、无阻塞。**已在 bot-new 上 `docker compose up -d --build --no-deps bot` 重建验证通过**：容器用新镜像(ID 一致)、`docker run --rm --entrypoint python langchaindev-bot -c 'import paramiko'`→5.0.0(镜像铁证、无 bind-mount 也 import 得到)、ssh_transfer 全模块 import OK、health 全绿、torch 用缓存未重下。以后 force-recreate/--build 不再丢 paramiko。**插曲(已妥善处理)**：首次 `git add Dockerfile && commit` 因工作区已有他人预暂存的文件(infra/pulumi、3 篇 collab 文档、main.py、pytest.ini)被一并带入 commit 633d714；其中 main.py 有 `kimport sys` 损坏(疑似 IDE 误触，未进过生产)。已 soft-reset 撤销、修好 main.py 损坏(py_compile OK)、还原 pytest.ini、重新只提交 Dockerfile(a701c61)；infra/docs 保持未跟踪原样。构建首次因 aliyun mirror 拉 aliyun-python-sdk-core tarball 读超时(15s)失败一次，重试(torch 已缓存)即过。见 MEMORY [[deploy-paramiko-and-torch-layer]]（已更新为"已解决"）。
