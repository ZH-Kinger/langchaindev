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

## 分派优先级：默认分派、留判断
dev 的默认反射是**先拉子 agent**、不是自己扛。下面的活**默认分派**给对应队友；只有琐碎（几行内、一眼可答）或分派开销明显大于任务本身时 dev 才自己做。清单之外自行判断。**能并行就并行**：互不依赖的活在同一条消息里一次拉多个 agent 并发，有依赖才按流水线串行；派出去的别自己重复做。

| 触发信号 / 活儿 | 默认派给 | 说明 |
|----------------|---------|------|
| 改了任何源码（core/config/tools/…） | **tester + auditor** | tester 补测+全量 pytest、auditor 过闸——提交闸门硬规，缺一不可 commit |
| 阿里云/火山/飞书/LangChain 规格、API 用法、现成方案 | **researcher（研究）** | 别猜，查官方文档 + 取证，带出处 |
| 摸清某台机/环境/线上系统现状 | **researcher（探针）** | 迁移盘点、排障取现场、上线前核对，只读侦察 |
| 多步 / 需求含糊 / 排顺序拆任务 | **planner** | 先规划再动手 |
| 功能变了、README/CHANGELOG/docs 要更新 | **docwriter** | 文档同步，别过时 |

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

[2026-07-10 TESTER] RAM 建号门禁安全修复补测（core/ram_approval.py:174 附近，只认回拉实例详情顶层 status）。`tests/unit/test_ram_approval.py` +9 例（25→34），全量 **1069 passed / 1 skipped / 9 deselected / 0 failed**（本地 .venv，matplotlib 已装故 gpu_distribution/wants_metrics 两例本地绿；容器缺 matplotlib 才 red，属既存基线，非本改动）。哨兵桩 `create_accounts_for_platforms` 探测"有没有建号"。覆盖：① 核心回归 事件节点 status=PASS 但实例 PENDING→不建号(created==[])、reason=instance_status=PENDING、确以实例码回拉详情（锁死绕过第二级 bug 1C854E42）；② 实例 APPROVED→建号恰一次；③ 无 instance_code→ignore(no_instance_code)、绝不建号、压根不回拉；④ REJECTED→不建；⑤ 实例顶层缺 status→归一 'none' 不建；⑥ 小写 'approved'→.upper() 后建；⑦ 首尾空白 '  APPROVED  '→strip 后建；⑧ 深层嵌套多节点 PASS/APPROVED 但实例 PROCESSING→不建（事件 status 仅早期过滤用）；⑨ dry-run 亦受实例级门禁约束。**既有用例甄别**：`test_handle_event_dedupes_processed_instance` 已 mock fetch_approval_instance 返回 status=APPROVED（新逻辑下仍建号，无需改）；`test_handle_event_ignores_cc_or_node_event_without_status` 走 no_status_no_instance 早退、不触门禁（无需改）——**无既有用例需补 mock**。**做了变异验证**：临时把 PENDING 塞进 APPROVED_STATUSES 模拟旧 bug→核心回归用例如期 FAIL（created==['twolevel']），证明非空断言。未发现源码问题。未覆盖项：真实 fetch_approval_instance 的 HTTP/飞书审批 GET（integration 面，单测 mock）、多平台聚合建号在门禁后的分支（既有其它用例覆盖）。
[2026-07-10 TESTER] RAM 建号门禁 delta 收口（源码收紧 `instance_status != "APPROVED"`，弃用 APPROVED_STATUSES 做实例级判据）。`tests/unit/test_ram_approval.py` 34→42（+8），全量 **1077 passed / 1 skipped / 9 deselected / 0 failed**。新增：① 参数化 6 例锁死收紧价值——实例 status ∈ APPROVED_STATUSES 但 != 'APPROVED'（PASS/PASSED/AGREE/AGREED/COMPLETED/APPROVE，皆节点级/timeline 词）→ 不建号（旧 `in APPROVED_STATUSES` 版会误建号，收紧后拒绝；含前置健全性断言"确为成员"）；② DONE 防御性回归（事实核对 'DONE' 本不在 APPROVED_STATUSES，新旧都拒，reason=instance_status=DONE）；③ CANCELED 补齐官方枚举负路径。原 9 例 create 路径皆用 'APPROVED'(+小写/空白归一)、skip 路径用 PENDING/REJECTED/PROCESSING/none，不受收紧影响仍绿。未发现源码问题。
[2026-07-10 DEV] paramiko 烤入镜像坑根治 + Dockerfile 纳入 git。Dockerfile 原只在服务器 `/root/langchaindev/Dockerfile` 未版本化，现纳入仓库(commit `a701c61`)并调层序：torch 安装 RUN 块挪到 `COPY requirements.txt` 之前作独立缓存层(torch 版本写死、不依赖 requirements → 改 requirements 不再连累 torch 重下 200MB+)。auditor 逐行对比服务器原版：除 torch 块位置+2 行注释外无改动、构建语义不变、无阻塞。**已在 bot-new 上 `docker compose up -d --build --no-deps bot` 重建验证通过**：容器用新镜像(ID 一致)、`docker run --rm --entrypoint python langchaindev-bot -c 'import paramiko'`→5.0.0(镜像铁证、无 bind-mount 也 import 得到)、ssh_transfer 全模块 import OK、health 全绿、torch 用缓存未重下。以后 force-recreate/--build 不再丢 paramiko。**插曲(已妥善处理)**：首次 `git add Dockerfile && commit` 因工作区已有他人预暂存的文件(infra/pulumi、3 篇 collab 文档、main.py、pytest.ini)被一并带入 commit 633d714；其中 main.py 有 `kimport sys` 损坏(疑似 IDE 误触，未进过生产)。已 soft-reset 撤销、修好 main.py 损坏(py_compile OK)、还原 pytest.ini、重新只提交 Dockerfile(a701c61)；infra/docs 保持未跟踪原样。构建首次因 aliyun mirror 拉 aliyun-python-sdk-core tarball 读超时(15s)失败一次，重试(torch 已缓存)即过。见 MEMORY [[deploy-paramiko-and-torch-layer]]（已更新为"已解决"）。

## #55 🔴 ram_approval 两级审批被绕过：第一级过就建号（真机确认，dev 已修，过闸）
真机报错 `[ram_approval] result notification failed` 追出更严重的主 bug。researcher 只读侦察+爆炸半径坐实（`research/ram-approval-premature-creation.md`）：
- **根因**：`routes.py` 把飞书 `approval_instance`(整单) + `approval_task`(节点)两类事件通吃；`handle_approval_event` line151 `_extract_status(payload)` 经 `_deep_first` 深搜抓到**节点级** PASS，line175 `status = status or ...` 短路 → 回拉了实例却不复核整体状态 → line206 建号。第一级审批人一点同意就建号，绕过第二级。平台无关（阿里 RAM + 火山 IAM 都中）。
- **铁证**：Redis 出现 `result_status=success` + `approval_status=PENDING` 矛盾体。实例 1C854E42(jichuan 火山)整单至今 PENDING、第二级从未批，账号已建=越权。爆炸半径：4/4 两级审批全部提前建号（另 3 个第二级事后补批侥幸合规）。
- **修复**（dev，line 174-192）：建号判据改为只认【回拉实例详情顶层实例级 `status`】，且收紧为 `!= "APPROVED"` 单值（不用 APPROVED_STATUSES——PASS/AGREE/DONE 等节点/timeline 词是帮凶）；无 instance_code → fail-safe 不建。权威判据出处见 researcher 报告（飞书审批 v4 实例级枚举 PENDING/APPROVED/REJECTED/CANCELED/DELETED）。
- **闸门**：tester 42 例(+8，含 6 个"成员但非 APPROVED"参数化锁死收紧+非空验证)、全量 1077 passed；auditor 两轮无阻塞（确认无绕过路径、锁 finally 正常释放使第二级事件能重新 claim 建号、单调收紧不误伤）。
- **待办**：①jichuan 越权号处置（撤/催批，待用户定）；②部署 bot-new（待用户授权）+ 真机走一条两级全过验证正常落号；③次 bug `_account_delivery_text` account_id(int) 拼文本 TypeError 致结果通知失败（非阻断，未纳入本次，待修）。

[2026-07-15 TESTER] #55 次 bug `_account_delivery_text` account_id(int) TypeError 补测完成（dev 已修 line 527/539 `str(...)`）。`tests/unit/test_ram_approval.py` 42→46（+4），全量 **1081 passed / 1 skipped / 9 deselected / 0 failed**（本地 .venv）。覆盖点：① create 分支（line 539）火山 SDK create_user 返 int account_id(2111674479)→result.account_id 为 str "2111674479"；② existing 分支（line 527）get_user 命中已存在用户 int account_id 同样规范成 str + skipped 含 volcano_iam_user_exists；③ **锁死 bug** 走 create_volcano_iam_account 拿含数字主账号 ID 的 result→`_account_delivery_text` 返回 str、含 "2111674479" + "所属主账号ID" 行 + AccessKey 段未被 join 中断；④ account_id 为空("")→`if item.account_id:` 跳过、不出现"所属主账号ID"行、不崩。**非空验证（变异复现）**：直接构造 int account_id=2111674479 调 `_account_delivery_text` 精确复现 `TypeError: sequence item 9: expected str instance, int found`（index 9 即 account_id 值），证明 join 是崩点、str() 回退即 FAIL。matplotlib 两例（test_gpu_distribution/test_wants_metrics_chart）本地 .venv 绿（装了 matplotlib）；容器缺 matplotlib 才 red，属既存基线、非本改动。未发现新源码问题。

## #55 续 次 bug：_account_delivery_text account_id(int) 拼文本 TypeError（dev 已修，过闸）
建号成功后结果通知 `[ram_approval] result notification failed`：`_account_delivery_text` 的 `"\n".join(section_lines)` 里火山主账号 ID `item.account_id` 是 int（SDK 返回数字）→ `TypeError: sequence item 9: expected str instance, int found`（非阻断，`_safe_notify_result` 吞异常，账号照建但凭证没下发到）。修复：`create_volcano_iam_account` line 527/539 两处 `result.account_id = getattr(...) or ""` 外包 `str()`，字段（声明 str）真为 str，下游 join/URL/落库均安全。tester 46 例(+4，含变异复现锁定 index 9)、全量 1081 passed；auditor 无阻塞。Low 待办（可选纵深）：`_account_delivery_text` 内 join 前再 str() 兜一层，防未来直接构造 int account_id 的入口。

[2026-07-16 TESTER] 认领 P0-9：PFS 跨云直传（core/pfs_transfer/）P0 骨架补测。开工。范围：paths/orchestrator/cards/tool/feishu/reconcile 六面单测。本机 Python3.14 langchain 不兼容（StructuredTool import 失败），走服务器容器 aiops-bot（已从 Exited137 重启）。注意：新 pfs 代码未部署，容器 /app 无 core/pfs_transfer——需先把未提交源码同步进容器才能跑。

[2026-07-16 TESTER] PFS 跨云直传 P0 骨架补测完成（6 文件 153 例，容器 aiops-bot 内 python3.11 全绿）。文件：tests/unit/test_pfs_transfer_{paths,orchestrator,cards,tool,settings,feishu}.py。
覆盖：① paths P1/P2 方向判定+两 staging 推导(bucket/prefix/region/dataflow)+缺map/region/bucket/坏JSON拒+拒同云/对象存储/非法+子目录白名单拒注入穿越(14参)+裸挂载剥前缀用默认fs(无默认fs报错、不误剥/vepfsdata)；② orchestrator _job_id 幂等/xpfs-前缀/异plan异id、create首建/幂等回填created_by/不覆盖/FAILED复位/staging prefix含chain_id、needs_approval(size_known=False恒True+边界+自定阈值)、run三段推进(P1 vepfs→transfer→cpfs / P2 cpfs→transfer→vepfs)+某段FAILED不进下一段+段级跳过(sink_done/双跳过)+子run抛错→FAILED、_make_sub方向×段派发+OP+staging uri+created_by透传、refresh定位段续推/RUNNING不动/子FAILED反映/终态noop/缺job；③ **HIGH-1 锁死**：_resume_async 未launched守卫——NEW+launched=False经refresh/_resume_async零起线程不续跑(绕过审批门)、launched=True放行；④ cards 四卡schema(confirm/progress 2.0、result 1.0)、200830规避、缺字段不KeyError、FAILED含retry_pfs_transfer；⑤ tool plan只读不gate、**MED-2 apply admin门**(非管理员force=true也拦不建job不起线程/管理员force起链/管理员无force提示)、status、路径错回错、TOOL_GROUPS注册；⑥ feishu 意图命中(vepfs+cpfs/pfs直传词)+不误命中(单沉降/普通迁移)+排序(pfs在ssh前、关时落ssh分支不推pfs卡)、_JOB_ID_RE认xpfs-、query xpfs-分支refresh+终态经闸门补推、**MED-2 confirm非管理员一律拦**(不论自报量、不占锁不起线程)/admin正常NX+progress_card_v2、created_by回填/不覆盖、reply_v2分支、retry清launch+notified保留段完成标记+非管理员经confirm门被拦、handler注册；⑦ settings 默认+validate启用空map/坏JSON/默认'{}'告警、关闭/有map不告警、reconcile spec含pfs+active三段+契约齐。
既存失败甄别：全量 1174 passed / 5 failed——2 例 matplotlib(test_gpu_distribution/test_wants_metrics_chart，容器无 matplotlib 模块，环境基线、��本改动)；3 例 test_submit_async 是**测试容器陈旧**artifact(容器 .deployed_commit=7050f65 的 test_submit_async.py 缺 JIRA_ENABLED=True autouse fixture，而 actions.py 已同步为 HEAD+pfs)——同步当前测试文件后 8/8 全绿，证非 PFS 回归。注：本机 py3.14 langchain 不兼容，全部在服务器容器 aiops-bot 跑；pfs 源码尚未部署，我把未提交源+新测试同步进容器 /root/langchaindev 才跑起（dev 部署时 deploy.ps1 会正式覆盖）。

[2026-07-16 TESTER] PFS 补测收口：同步当前 test_submit_async.py(HEAD 版含 JIRA fixture)后全量 clean run = **1177 passed / 2 failed / 9 deselected**。2 failed 全是 matplotlib 缺模块基线(test_gpu_distribution::test_dist_url_and_summary_card + test_wants_metrics_chart::test_process_message_attaches_chart_for_metrics)——容器无 matplotlib、任何 commit 都复现、与 PFS 无关。PFS 153 例全绿、零回归。待 dev 部署 pfs 源 + auditor 过闸。

## #56 vePFS↔CPFS 跨云直传 P0 骨架（新包 core/pfs_transfer/，dev 已过闸，未部署）
物理无直连，3 段链编排：源 PFS 沉降→跨云→目的 PFS 预热。加法式新包，复用 vepfs/transfer/cpfs 三段现成 orchestrator + NX 闸门，**零改现有引擎**。方案 planning/pfs-transfer-chain.md、调研 research/vepfs-cpfs-direct-transfer.md + pfs-staging-buckets.md。
- 状态机 NEW→SINKING→CROSSING→PREHEATING→DONE/FAILED；段级"跳过已成功段"续跑（sink/cross/preheat_done）；refresh 定位当前段+NX 锁续推。xpfs- 前缀、当天幂等、Redis pfs:transfer:job。
- 三入口：飞书向导卡（意图 _is_pfs_transfer_intent 同时提 vepfs+cpfs 判据、排 ssh/transfer 前）+ Agent 工具 manage_pfs_transfer + CLI。对账加 pfs 链（6 链齐）。配置 PFS_TRANSFER_ENABLED/PFS_STAGING_MAP/APPROVAL_TB/STAGING_CLEANUP/CHAT_ID。纯 Python、普通 deploy 免 rebuild。
- P1=vePFS→CPFS（三段引擎全真机验过，先做）；P2=CPFS→vePFS 前置阻塞（跨云 OSS→TOS 未真机验+DMS 桶级落盘），暂不开工。
- 审计抓两个审批绕过口均已堵：HIGH-1（查询进度经 refresh→_resume_async 起未确认任务，加 launched 守卫）；MED-2（用户拍板收紧：卡片 confirm + Agent 工具 apply 双路"非管理员一律需管理员确认"，自报量只作显示）。LOW-1 补 validate 启用空 map 告警。
- tester：6 文件 153 例、全量 1177 passed（2 matplotlib 既存基线）。auditor：全 launch 向量清点无残留 fail-open，无阻塞。
- 待办：P1 真机（服务器填 PFS_STAGING_MAP 真值+CPFS段+mgw 配置+vePFS→TOS 服务授权）；小缺口 _h_query_progress_by_id 表单查白名单未含 xpfs-/sgp-（文本查已支持）。

## #57 审批制临时 AK/SK 发放（外部/卖数据）P0 骨架（新包 core/temp_ak_issuance/，dev 已写，待 tester+auditor，未提交）
方案 planning/temp-ak-issuance.md、调研 research/temp-ak-issuance.md。用户拍板「按有效期窗口分流」。加法式新包，复用 ram_approval 门禁 + permsync policy 生成 + SMTP 下发，零改现有引擎，纯 Python 免 rebuild。
- **分流**（issuer.classify_mode）：`expire−now ≤ TEMP_AK_STS_MAX_SECONDS(默认12h=43200)` → **STS 单发**（含 SecurityToken，到点自灭，assume_role_with_policy 现场 session policy 收窄，无需清理）；否则 → **方案 B**（RAM 子用户+长期 AK+policy 时间窗条件+到期硬删）。用 expire−now 非 expire−not_before（正确处理「3天后生效短窗」→方案B）。
- **policy.py**（纯逻辑）：build_policy_with_window / build_session_policy（≤2048 校验）→ 每条 statement 注入 DateGreaterThan(生效)/DateLessThan(到期)（acs:CurrentTime ISO8601 **+08:00**，同 statement AND）+ 可选 IpAddress(acs:SourceIp)；桶级读补 **GetBucketAcl**；对象写收紧**无 DeleteObject**（外部防删）；List 带 oss:Prefix。复用 permsync._obj_arn/动作集。
- **issuer.py**：STS 走 utils.aliyun_sts.**assume_role_with_policy**(新增,传 Policy+duration,不缓存不复用,与 assume_role_for_user 隔离)；方案 B 走 permsync.make_ram_client 建 user(无控制台/无组)+建/附时间窗 policy+建 AK。
- **orchestrator.py**：grant 状态机 NEW→ISSUED→REVOKED/FAILED，Redis temp_ak:grant:{id} 30d TTL，grant_id=hash(实例) 幂等（已 ISSUED/REVOKED 短路不重发），独立 NX 锁 temp_ak:lock。**记录绝不含 secret/token**（只存 ak_id 供方案B 到期定位）。桶解析 TEMP_AK_BUCKET_MAP→permsync.BUCKET_MAP→原样。
- **approval.py**：should_handle_event **精确匹配 TEMP_AK_APPROVAL_CODE**（不做 no-code 兜底，与 ram_approval 按 code 互斥不误抢）；handle_temp_ak_event **复用 #55 硬化门禁**——只认回拉实例详情顶层实例级 status=="APPROVED"、无 instance_code fail-safe 不发；表单解析（桶/读写前缀/生效/到期/收方邮箱/出口IP/原因），前缀注入白名单拒控制字符，时间容错解析(ms/s/ISO)。
- **cleanup.py**：方案 B 到期硬删序列（停用 AK→删 AK→解绑 policy→删 policy 清版本→删 user，照 jichuan 经验，幂等）；sweep_expired 扫 ISSUED 且 expire<now。STS 自灭不清理。
- **delivery.py**：外部 SMTP 邮件（含 AK/SK[/Token]，一次性提示）；内部飞书回执卡（**脱敏，不含 secret**）。secret/token 只在邮件正文出现一次。
- **三入口**：飞书审批（主，routes.py 按 TEMP_AK_APPROVAL_CODE 分发，gated on TEMP_AK_ENABLED）；Agent 工具 manage_temp_ak（**只有 plan/status/revoke，无 issue**——杜绝绕审批发凭证；revoke 管理员 gate）；CLI（plan/status/revoke/sweep，dry-run 默认，**无 issue**）。TOOL_GROUP temp_ak。
- **调度**：dsw_scheduler._temp_ak_cleanup_loop 每日北京 TEMP_AK_CLEANUP_HOUR:35 sweep，start() 按 TEMP_AK_ENABLED+CLEANUP_ENABLED 起。
- **配置**：settings 加整块 TEMP_AK_*（ENABLED/APPROVAL_CODE/OSS_ROLE_ARN/STS_MAX_SECONDS/BUCKET_MAP/CLEANUP_*/CHAT_ID/FIELD_*）；validate() 启用时校验 APPROVAL_CODE+OSS_ROLE_ARN+ALIYUN_ACCESS_KEY_ID+SMTP_HOST。16 文件 py_compile 全过。
- **安全自查**：发放路径唯一=审批（工具/CLI 无 issue）；真发放二次校验实例 APPROVED；secret/token 不入 Redis/日志（grant 记录+内部卡+CLI status 均脱敏）；STS session policy 2048 上限硬校验；方案 B 对象权限无 DeleteObject。
- **tester 重点**：policy 时间窗(ISO8601+08:00/AND/session≤2048/GetBucketAcl/无DeleteObject)、classify 边界(≤12h/跨天/未来生效短窗)、orchestrator 幂等+不存 secret+桶解析、approval 门禁 fail-safe(实例非APPROVED不发/无instance_code不发/节点级PASS不误发)+表单解析+dt容错+前缀注入拒、cleanup 删序+sweep、tool 无 issue+revoke admin gate、settings validate、routes 与 ram_approval code 互斥、assume_role_with_policy 传 policy+duration(mock STS)。全量 pytest（matplotlib 2 例既存基线甄别）。
- **auditor 重点**：清点所有 launch 向量确认无绕审批发凭证口（工具/CLI/routes/查询）；secret/token 全链路不落盘不入日志；policy 时间条件正确性（+08:00/AND/DateLessThan 到期拒）；删号顺序幂等；classify 边界；STS 2048 守卫；approval code 精确匹配不误抢 RAM 建号流。
- **待办（P1 真机，用户/ops 提供）**：① 用户稍后给 TEMP_AK_APPROVAL_CODE（飞书新建独立审批模板 + 表单字段）；② 控制台建 STS 宽 OSS 角色(信任 Master AK+MaxSessionDuration=43200)填 TEMP_AK_OSS_ROLE_ARN；③ researcher 取证(GetBucketAcl 动作名/CreatePolicy 时间条件 JSON 校验/RAM 可写 AK 全套权限/STS policy 参数真机可用)；④ 真机发 ≤12h STS + 跨天方案 B + 到期清理验证。

- [2026-07-21][AUDITOR] #57 过闸：无 HIGH 阻塞。发放唯一经审批+严格实例级 APPROVED 门禁、全 launch 向量(工具/CLI/scheduler/reconcile)无绕审批发凭证口、secret/token 全链路脱敏、policy 时间窗(+08:00/AND/每条statement/无DeleteObject)、删号序幂等、classify 边界、幂等双投递 均 PASS。MED-1 routes 与 ram_approval no-code 兜底耦合(edge 丢发放非误发,建议 temp_ak 严格判上移)；LOW: STS_MAX>43200 clamp、FAILED 重投重入、RAM 邮件失败 creds 丢失告警。均非阻塞。
- [2026-07-21][DEV] 据审计修 MED-1 + LOW-2 + LOW-4（3 文件已编译）：**MED-1** routes.py 把 temp_ak 严格 code 分发**上移到 ram_approval 分发之前**（temp_ak 严格 code==target 不误抢 ram，先判杜绝极端下 no-code 兜底吞发放）；**LOW-2** issuer 加 `STS_HARD_CAP=43200` + `_sts_limit()=min(配置,43200)`，classify_mode/`_sts_duration` 都用它（误配 >12h 不再把长窗口误判 STS）；**LOW-4** delivery.deliver 方案B 外部邮件失败(secret 不可恢复)时发醒目内部告警 `_alert_creds_undelivered` 提示管理员 revoke+重审。**LOW-3（FAILED 重投重入）记为可接受不改**：_issue_ram 中 create_access_key 是最后一步，其后仅 issue_grant 置 ISSUED，AK 创建后再失败几无可能；且失败重试是期望特性、残留 AK 由 cleanup 按 user 兜底删。请 tester 覆盖这三处 delta（routes 顺序、classify >43200 clamp 边界、RAM 邮件失败告警）。

- [2026-07-21][DEV] **真机审批模板拉取 → 解析层重构（用飞书审批 API 只读拉 approval 定义，approval_code=5B4A3105-1EF9-4645-99D2-CCF69FE75D06「数据外采访问凭证申请」）**。真实 5 控件与 P0 原假设差异大，已重构 approval 解析 + 下发 + 卡片（**policy/issuer/orchestrator 状态机/cleanup/STS 安全核心未动，auditor 过的就是这些**）：
  - 真字段：**平台**(radioV2 火山云/阿里云) · **使用企业名称**(input) · **权限设置**(checkboxV2 read/write/read&write，对同一目录) · **DateInterval**(dateInterval，生效+到期合一) · **申请目录**(input)。**无邮箱字段、无独立桶字段、无独立读/写前缀字段**。
  - `approval._FIELDS` 改为这 5 项（别名含真机 widget id，名字改了也命中）；`parse_temp_ak_request` 重写：`_parse_platform`(火山→volcano/其余→aliyun)、`_parse_perm`(read/write/read&write→{read,write})、`_parse_date_interval`(容错 dict{start/end 多键}/list/单值)、`_parse_directory`(oss://桶/前缀、tos://…、桶/前缀、裸桶→(bucket,prefix))；单目录 + perm 派生 read_prefixes/write_prefixes=[prefix]。
  - `_validate_spec`：**平台=火山云 → 抛「暂未支持，请改阿里云」**（P0 只做阿里 OSS；火山 TOS 临时凭证留 P1 取证）；缺桶/未选读写/到期已过/生效≥到期 各拒。
  - **下发改造（无邮箱）**：`delivery.deliver` 改为**凭证私信审批发起人**(grant.requester open_id，一次性)——由其转交外采企业；无发起人 open_id 则拒发（不进群防泄漏）+ 告警。新增 `cards.credential_card`(含 secret，仅私信发起人)；`receipt_card`/`status_card` 脱敏进内部群，加显示 企业/平台。移除 SMTP 依赖，settings.validate 去掉 SMTP_HOST 校验。
  - settings：`TEMP_AK_FIELD_*` 改为 PLATFORM/ENTERPRISE/PERM/DATE_INTERVAL/DIRECTORY；grant 记录加 platform/enterprise。
  - 本地 .venv（可 import langchain）自证：包 import OK + 工具注册 OK；smoke 全过（申请目录三形态解析、平台/权限/时间区间解析、classify 6h→sts/20h·3d→ram、policy 含 +08:00/DateGreaterThan+LessThan/无 DeleteObject/GetBucketAcl/SourceIp、session policy 675 字符 ≤2048）。
  - **secret 通道变更(auditor 需复核 delta)**：secret/token 现只出现在 `credential_card`（私信 grant.requester 一次），receipt/status 群卡与日志均脱敏。其余安全面(门禁/无 issue 口/classify/cleanup/policy 时间条件)未变。
  - 待办：① 火山云分支(P1，先 researcher 取证火山临时凭证能力)；② DateInterval 真机 value 形状 P1 用真实提交反查确认(现容错解析);③ 延期入口(P0.5，独立审批 id，用户将另给)。
- [2026-07-21][AUDITOR] #57 delta 复核：维持过闸无 HIGH。secret 通道(私信发起人,无 open_id 拒发不入群,群卡/日志脱敏)、火山云门禁(_validate_spec 抛错早于建号)、解析注入面(prefix _KEY_RE+json.dumps 转义+ARN 字面前缀无穿越+单 base 无跨桶放大)、routes temp_ak 严格 code 上移闭合 MED-1 均 PASS。新增 LOW: _send_card 不校验响应(送达盲点)、_parse_platform 未知默认 aliyun、bucket 未过 _KEY_RE。均非阻塞。
- [2026-07-21][DEV] 据 delta 审计修 3 LOW（approval.py+delivery.py 已编译+smoke 过）：**LOW-A** 凭证私信改用带响应校验的 `_feishu_send_interactive_checked`(飞书返非 0 即抛→触发 `_alert_creds_undelivered`，杜绝"以为送达实则没送"；异常只含 code/msg 不含卡片 body/secret)，不动共享 `_send_card`；**LOW-B** `_parse_platform` 未知平台→"unknown"（不再默认 aliyun），`_validate_spec` 增 `platform!="aliyun"→拒`；**LOW-C** `_parse_directory` bucket 也过 `_KEY_RE`。tester 请补：私信失败(飞书 code!=0/status!=200)→触发 undelivered 告警、未知平台拒发、bucket 控制字符拒。
- [2026-07-22][TESTER] #57 补测完成：全量 **1396 passed / 1 skipped / 9 deselected / 1 xfailed**（本地 .venv py3.11），8 文件 163 例，对齐 5 控件模型 + 全部 3 批 delta，门禁做了变异验证。发现 1 源码 bug（cleanup.py:36 DeletePolicyVersionRequest 传了非法 policy_type→TypeError 被吞→非默认 policy 版本永不删、孤儿 policy 泄漏），strict-xfail 锁定，非阻塞（单默认版本不触发+时间窗兜底）。matplotlib 2 例既存基线。
- [2026-07-22][DEV] 修 tester 报的 cleanup bug：`cleanup.py` `DeletePolicyVersionRequest` 去掉 `policy_type="Custom"`（该类签名仅 policy_name/version_id，误传构造即抛）。已 venv 验证请求可构造。tester 去 xfail 重跑确认 XPASS→全绿后过闸提交。
- [2026-07-22][DEV] **#57 P0 已提交 `fac5dec`**（26 文件 3635 插入，只 #57 相关，未 push/部署）。

## #58 全局审批白名单 + P0.5 临时AK延期入口（dev 已写，待 tester+auditor，未提交）
接 #57(P0 fac5dec)。用户拍板：提交 P0 + 要白名单 + 开做 P0.5。
- **白名单硬门（用户安全要求「只收我给你的审批」）**：routes.py `_approval_allowlist()` = {FEISHU_RAM_APPROVAL_CODE + 启用时 TEMP_AK_APPROVAL_CODE / TEMP_AK_EXTEND_APPROVAL_CODE}；审批事件 approval_code 不在白名单（**含未带 code 的**）→记日志丢弃，早于所有审批处理器。**顺带收紧 ram_approval 无-code 兜底**（用户接受）。
- **P0.5 延期**：独立审批 `TEMP_AK_EXTEND_APPROVAL_CODE=E9333E62-37D3-4644-9117-C994D6035EFD`「访问凭证延长申请」3 控件（凭证ID/使用企业信息/DateInterval，别名含真机 widget id）。routes 加 `should_handle_extend_event` 分发（排 ram 前）；`handle_temp_ak_extend_event` 复用 #55 实例级 APPROVED 门禁 → 按凭证ID load grant → `_verify_enterprise`（企业不符拒，防串企业）→ `extend_grant`。
- **extend_grant**（orchestrator）：仅 ISSUED 可延；**方案B → `issuer.rewrite_ram_window` 改写 policy 时间窗（同 AK、不重发，creds=None）**；**STS → 按新窗重签发（classify，新窗 >12h 自动转方案B，creds=新凭证）**；REVOKED/FAILED 拒。幂等：`extend_instances` 记已应用实例 + NX 锁。
- **下发**：`delivery.deliver_extend`——creds(重签发)→私信新凭证卡(含 secret 一次)；None(同AK)→`cards.extended_card`「有效期已延长至 X」(无 secret) + 内部回执。
- settings 加 `TEMP_AK_EXTEND_APPROVAL_CODE` + `TEMP_AK_FIELD_GRANT_ID`。7 文件编译 + venv smoke 过（allowlist 含三 code、extend 精确匹配、企业不符拒、方案B 同AK creds=None+改写窗+历史、STS>12h 转方案B+ak 更新、REVOKED 拒）。
- **tester**：白名单（命中过/非白名单含无 code 丢弃/ram 事件仍过/非审批消息不受影响）、should_handle_extend、parse_extend(3 字段 by_id+by_name)、_verify_enterprise(不符拒/宽松包含容错/任一空跳过)、extend_grant(方案B creds=None+改写窗、STS 重签发+>12h转B、REVOKED/FAILED 拒、幂等 extend_instances 二次不重复应用)、rewrite_ram_window(mock RAM 断言 create_policy_version set_as_default)、deliver_extend(creds→凭证卡 / None→extended_card + 内部回执脱敏)。全量 pytest。
- **auditor**：白名单硬门无绕过（无 code 也丢、早于处理器）；延期复用 APPROVED 门禁不绕审批（无 instance_code/非 APPROVED 不动）；企业校验防串；extend 不泄 secret（方案B 不重发/STS 重发经私信 checked、群卡脱敏）；REVOKED 拒；幂等。
- [2026-07-22][AUDITOR] #58 批2 过闸无 HIGH。白名单硬门闭合(丢弃门早于所有 handler、含空 code、同一 code 提取口径、ram 兜底被前置拦、无第二路)；延期复用 #55 APPROVED 硬门+严格 code 无旁路(extend_grant 仅审批 handler 可达)；extend 不泄 secret(extended_card 无密钥、creds 走 checked 私信、群卡/日志脱敏)；仅 ISSUED 可延+extend_instances 幂等+NX 锁+rewrite 用 RAM 可写 AK。MED-1：_verify_enterprise 延期表单企业名为空时跳过校验→防串失守(引用别家 grant_id+留空企业可绕)；LOW：企业名宽松包含、STS 重签发重复 AK。
- [2026-07-22][DEV] 据审计修 MED-1（防串失守，approval.py `_verify_enterprise` 已编译+smoke）：改为 fail-safe——原凭证有企业名而延期表单**留空**→拒（无法核实归属）；仅原凭证本身无企业名才跳过。venv 验证：orig 有+ent 空→reject、orig 空+ent 空→ok、匹配→ok、不符→reject。LOW-1(宽松包含)/LOW-2(STS 重签发重复 AK，同 #57 LOW-3)记账不改（人工 APPROVED+白名单兜底/cleanup 按 user 删兜底）。请 tester 补：orig 有值 ent 空→拒。
- [2026-07-22][DEV] 用户放弃 STS 快路（嫌建角色麻烦），改为全走方案B：`.env` 设 `TEMP_AK_STS_MAX_SECONDS=0` 即所有申请走方案B、不建角色（classify_mode limit=0→恒 RAM，无需改码）。顺带 settings.validate 收紧：`TEMP_AK_OSS_ROLE_ARN` 仅在 `TEMP_AK_STS_MAX_SECONDS>0` 时才校验/告警（STS 关时不再唠叨缺角色）。venv 验证：STS off 不告警角色、STS on 告警。折进批2。请 tester 在 test_temp_ak_settings 补两例（STS=0 不列角色 / STS>0 列角色）；auditor 瞄一眼这 4 行 validate 条件 delta（纯启动日志、无流程/安全面变化）。

## #58 续 权限模型重构：read/download/write 三者正交（用户改了审批「权限设置」，dev 已改，待 tester+auditor）
用户把发放审批「权限设置」选项改成 `[read, write, download, read&write]`（加了 **download**），并拍板：**read 只给 List（列文件、不能下载）、只有勾 download 才给 GetObject（下载）**、两者正交。重构 policy + 解析（触及已提交 P0 的 policy.py/approval.py/orchestrator.py/issuer.py + 工具/CLI，全在 temp_ak 包内）：
- `policy.py`：动作集拆成三块——`LIST_ACTIONS`(ListObjects+GetBucketInfo/Stat/Location/Acl)、`DOWNLOAD_ACTIONS`(GetObject+GetObjectAcl)、`WRITE_ACTIONS`(PutObject/Abort/ListParts，**无 DeleteObject**)。`build_policy_with_window(bucket, *, prefix, caps, ...)` 按 caps⊆{read,download,write} 出语句：read→List(带 oss:Prefix)、download→GetObject 对象语句、write→上传对象语句；每条叠时间窗。`build_session_policy` 同签名 + ≤2048。**旧的 read_prefixes/write_prefixes 参数取消**。
- `approval._parse_perm` → 返回 caps list（download/write/read 子串判定，read&write→read+write）；`parse_temp_ak_request` spec 改存 `prefix`+`caps`（不再 read_prefixes/write_prefixes）；`_validate_spec` 改为「caps 空→拒」。
- `orchestrator.create_grant_record` grant 存 `prefix`+`caps`；`scope_line` 显示「目录+权限[列/下载/上传]」。
- `issuer` plan/_issue_ram/_issue_sts/rewrite_ram_window 全改用 `prefix=`+`caps=`。工具 manage_temp_ak plan 参数 read_prefixes/write_prefixes→prefix/caps；CLI plan --read/--write→--prefix/--cap。
- venv smoke 全过：read-only 无 GetObject 有 ListObjects、download-only 有 GetObject 无 List、write-only 无 delete 无 get、combo 三合一无 delete、时间窗 +08:00/Date* 在。
- **tester**：temp_ak 现有测试（policy/approval/issuer/orchestrator/extend/tool）大量引用旧 read_prefixes/write_prefixes 模型 → **需按 caps 模型重写**。重点：_parse_perm 五种输入→caps（read/download/write/read&write/多选）；build_policy_with_window 每种 cap 精确动作集（**read 无 GetObject、download 有 GetObject 无 List、write 无 delete**）+ 空 caps 空策略 + 整桶 prefix；session ≤2048；parse→spec(prefix,caps)；scope_line；issuer 四处用 prefix+caps；工具/CLI plan。全量 pytest。
- **auditor**：核心安全——**download（GetObject）只在 caps 含 download 时才出现**（read/write/read&write 都不给下载）；write 永无 DeleteObject/任何删除；正交无越权（read 不含 get、单勾 download 不含 list）；时间窗每条 statement 都在；STS session ≤2048 硬校验。
- [2026-07-22][AUDITOR] #58 续 caps 权限模型 过闸无 HIGH/MED。GetObject 当且仅当 download∈caps（read/write/read&write 均无下载）；WRITE_ACTIONS 无任何 delete；三者正交(read=List桶级无get / download=GetObject对象级无list / write=Put无read / caps空=空策略+validate拒)；每条 statement 叠 +08:00 时间窗 AND、session≤2048 硬校验；_parse_perm 子串判定 download 不误触 read/write；extend rewrite_ram_window 沿用原 caps 无放大、门禁/白名单/幂等未回退。LOW：_parse_perm 依赖选项英文 token(fail-closed 安全)、tests 待 tester 更新到 caps 模型。均非阻塞。
- [2026-07-22][DEV] 审批「权限设置」选项已定为 `[read, write, download]`（去 read&write，caps 解析天然支持三原子项，源码未改）。auditor LOW（选项英文 token 依赖）记为 **P1 真机核对项**：首条真实审批时反查提交值是否含 read/download/write 英文串（同 DateInterval 反查），若为中文再补别名映射。待 tester 按 caps 模型重写测试全绿后，本批（白名单+延期+权限模型）过闸提交。
- [2026-07-22][TESTER] 权限模型重写全绿 **1456 passed / 0 failed**（temp_ak 222 例迁 prefix+caps）。
## #59 命名规范化 + 下发改审批评论(B)（dev 已改，待 tester+auditor，未提交）
真机首单成功后用户两点反馈：
- **命名规范化**：`orchestrator._ascii_slug`（企业名→RAM 登录名 slug：中文走 pypinyin，未装降级 ASCII/否则 ext；只留 [a-z0-9]、截 20）→ `_derive_user_name`=`tempak-<slug>-<6hex>`；新增 `display_name_for(grant)`=`<企业>-临时外采用户`（中文 OK，控制台可辨），`issuer._issue_ram` 建号 display_name 用它。新依赖 **pypinyin**（requirements.txt，**部署需 --build**；lazy import + try 兜底，非硬依赖）。
- **下发改审批评论（B，用户拍板）**：`delivery.py` 重写——凭证（含 secret）作为**评论**贴到审批实例（复用 `ram_approval._send_approval_comment`），不再私信；发放贴 `instance_code`、延期贴 `extend_instances[-1]`；内部群回执卡脱敏；secret 只在评论正文、不入日志/Redis/群卡；评论失败+ram→`_alert_creds_undelivered`。移除 DM 路径（_send_creds_to_requester/_feishu_send_interactive_checked）。venv smoke 全过。
- 白名单实测验证（真机）：bot app 收全组织审批事件，白名单正确只放行 3 个 code、丢弃无关审批（补卡 C3F0FA20 等），无 RAM 建号误伤（补卡非 RAM）。
- 待 tester（重写 delivery 为评论 + 命名测试）+ auditor（secret 入评论=用户选择/不入日志Redis群卡、评论贴对实例、slug 无注入、pypinyin 降级）过闸 → commit → **--build 部署**。
- [2026-07-22][AUDITOR] #59 过闸无 HIGH/MED。B 审批评论下发：secret 仅在 credential_text→评论正文，_extended_text/群卡/日志/Redis 全脱敏，_send_approval_comment 无 logger、失败只回显 data.msg 不含 body→secret 不入日志；评论贴对实例(发放=instance_code/延期=extend_instances[-1]，缺则 raise)；失败+ram→alert。命名：_ascii_slug 过滤后仅[a-z0-9]→user_name 无注入、pypinyin lazy+try 不崩、display_name_for 接线 issuer:107 非安全面。LOW：credential_card/extended_card 死代码(含secret但永不执行)建议清、delivery 旧私信测试待 tester 更新。
- [2026-07-22][DEV] 据审计 LOW-1 清死代码：cards.py 删 `credential_card`（含 secret 渲染，改评论下发后无调用方）+ `extended_card`，只留脱敏 receipt_card/status_card；receipt_card note 改「已通过审批评论下发」。编译+import 验证：两卡已移除、receipt/status 在。请 tester 一并把引用这两卡的测试删/改。DateInterval 真机格式坐实：飞书给 `{start,end:ISO8601+08:00,...}`，_parse_dt %z 解析正确→存 epoch=北京时刻，fmt_ts(tz北京)渲染回 15:52→16:52 与提交一致，校验无误（之前诊断显示 07:52 是裸 fromtimestamp 按容器 UTC 的锅、非 bug）。

## #60 延期审批加撤销分支（延长/撤销二合一）+ 去内部回执卡（dev 已改，待 tester，auditor 已过）
- 用户把延期审批模板改「访问凭证延长/撤销 申请」：加 `撤销/延长` 单选(widget17847115181700001，选项文本 撤销/延长)+DateInterval 转非必填。
- dev：`approval._EXTEND_FIELDS` 加 action；`_parse_extend_action`(撤销/吊销/revoke→revoke，否则 extend，默认延长向后兼容)；`parse_temp_ak_extend_request` 返回 **5 元组**(action,grant_id,enterprise,not_before,expire)；`handle_temp_ak_extend_event` 分叉——**revoke**：防串在分叉前→`cleanup.revoke_grant`(删AK+policy+user)+幂等 `revoke_instances`+已REVOKED短路+`_notify_internal_action`；**extend**：不变，expire 校验从 parse 挪到 handler。settings 加 `TEMP_AK_FIELD_EXTEND_ACTION`。
- delivery delta（用户要求）：**删内部回执卡推送**（deliver/deliver_extend 不再调 `_send_internal_receipt`/`receipt_card`，函数删）；**凭证ID 写进凭证评论正文**（延长/撤销回填用）。信息只在审批评论。
- venv smoke 全过（撤销/延长解析、5元组、去卡零推送、grant_id 入评论）。普通 deploy（无新依赖）��
- [2026-07-22][AUDITOR] #60 撤销分支 + delivery delta 过闸无 HIGH/MED/LOW。撤销只经审批门禁(APPROVED+无instance_code不动)、防串对撤销分支同样生效(分叉前 _verify_enterprise)、revoke_instances 幂等+NX、无工具/CLI/查询旁路触发撤销；延长分支无回退。delivery：删回执卡=暴露只减不增，secret 分发点仍唯一=credential_text→审批评论；grant_id 入评论=非 secret 任务句柄不扩可见面；日志/Redis/告警脱敏未变；receipt_card 变死代码(脱敏无害)。

## #61 评论作者身份修正（真机发现：多级审批下错挂审批人）
- 用户真机复验发现：凭证评论挂在**审批人**名下、非申请人/管理员。根因：`delivery._comment_user_id` 用 `grant["requester"]`，多级审批下 requester 解析成审批人。用户要求「与 RAM 建号审批评论身份一致」。
- 修：`_comment_user_id(grant)` 改为**直接复用** `ram_approval._approval_comment_user_id()`（req=None → FEISHU_RAM_APPROVAL_COMMENT_USER_ID → ADMIN_FEISHU_OPEN_ID，requester 分支不可达），与 RAM 建号审批完全同款；撤掉临时加的 TEMP_AK_COMMENT_USER_ID。venv 自测：temp_ak 评论身份 == ram_approval 评论身份、≠ requester。
- [2026-07-22][AUDITOR] #61 评论作者定稿=复用 ram_approval._approval_comment_user_id()（配置→ADMIN_FEISHU_OPEN_ID，与 RAM 建号审批一致）：req=None→requester 分支不可达、彻底去 requester；撤掉 TEMP_AK_COMMENT_USER_ID；secret 面零变化(只改 user_id)；皆空→空 user_id 守卫早于 body→fail-closed 不泄漏+ram 告警。无新泄漏面，过闸维持。作者 open_id 须同发评论 app(否则 99992361 cross-app,fail-closed)。

- [2026-07-22][DEV] **批2+3 提交 `821a93a`（21 文件 +1214/-238）并已上线 bot-new**：deploy.ps1 部署 HEAD → force-recreate 重载 .env（服务器 .env 已填 TEMP_AK_ENABLED=true + 发放 5B4A…/延期 E9333E… code + TEMP_AK_STS_MAX_SECONDS=0，全走方案B无角色）→ 容器内验证 TEMP_AK 全就位、工具注册、validate temp_ak 零缺失、/health ok。**功能 live**（发放+延期+白名单）。白名单已变活（只处理 RAM建号+temp_ak code，收紧无-code 兜底）。待办：①第一条真审批 E2E 验收（真建号+下发+反查 DateInterval/权限选项提交值 + 验 RAM 可写 AK 建号权限，fail-closed）；②火山云分支 P1；③docwriter 同步文档。顺带发现（非 temp_ak）：服务器 validate 报 ALIYUN_BOT_MASTER_AK_ID/SECRET 缺失（用户态 STS 调云的 Master AK，疑轮换后未回填，待用户确认）。

[2026-07-22 TESTER] #57 临时 AK 发放 P0 补测完成（本地 .venv py3.11，全量默认 pytest **1396 passed / 1 skipped / 9 deselected / 1 xfailed**）。8 文件 163 例：test_temp_ak_{policy(16)/issuer(20)/orchestrator(18)/approval(60)/cleanup(9+1xfail)/tool(17)/settings(7)/delivery(15)}.py。已对齐 dev 真机模板重构（5 控件模型）+ 3 批 delta（MED-1 routes 顺序 / LOW-2 STS>43200 clamp / LOW-4 邮件→私信改造 / LOW-A checked 私信 / LOW-B unknown 平台拒 / LOW-C 目录桶名过滤）。
  覆盖点：① policy 时间窗（+08:00、DateGreaterThan/LessThan 每条 statement AND、GetBucketAcl、无 DeleteObject、source_ips→IpAddress、空读/写不出 statement、''=整桶无 oss:Prefix、session≤2048 + PolicyTooLargeError）；② classify 边界（≤12h→sts / >上限→ram / 3天后短窗→ram / 误配>43200 硬顶封 43200，20h 仍 ram）+ plan dry-run 不调云 + _issue_sts 透传 policy+duration + assume_role_with_policy 断言 AssumeRoleRequest.policy(json)/duration_seconds 被设并 clamp[900,43200]；③ orchestrator grant_id 幂等/tak-前缀、create 记录无 secret/token 键、幂等返旧不复位、issue_grant 已 ISSUED/REVOKED 短路不重发+RAM 落 ak_id、桶解析三路、claim NX；④ **approval 门禁（安全核心）**：实例级 status≠APPROVED（PENDING/PROCESSING）不发、无 instance_code fail-safe 不回拉不发、实例 APPROVED 发放恰一次、code 不匹配 ignore、already_issued 短路、REJECTED 早退、dry-run 不发；**变异验证**：事件携带节点级 PASS(∈APPROVED_STATUSES) 但门禁只认实例顶层 status→仍拦，证非空断言；⑤ 新表单解析 _parse_directory(oss/tos scheme 剥、桶/前缀、裸桶、控制字符桶名+前缀均拒)/_parse_perm(list/read&write/单选)/_parse_platform(火山→volcano/阿里→aliyun/未知→unknown)/_parse_date_interval(dict{ms+ISO 多键}/list/单值)+parse 整体(name 键与 widget-id 键两路参数化)+_validate_spec(火山拒/unknown拒/aliyun过/缺桶·缺读写·到期过·生效≥到期各拒/空生效→now/无邮箱不再拦)；⑥ cleanup 删序相对顺序+STS 只翻状态不调云+幂等吞删已删+sweep 只挑 ISSUED&expire<now(边界 expire==now 不扫)；⑦ tool 无 issue 动作(schema/dispatch/源码静态断言)+revoke admin gate+plan/status 不调云；⑧ settings validate 启用缺三字段(APPROVAL_CODE/OSS_ROLE_ARN/ALIYUN_ACCESS_KEY_ID)各告警+SMTP_HOST 已移除、关闭不告警；⑨ delivery 凭证私信发起人(checked HTTP,code!=0/status!=200 抛→ram 告警,异常信息不含 secret)+receipt/status 卡与 grant 记录脱敏+无 requester 不发不泄漏、STS 失败不告警。
  **发现 1 个源码 bug（已 SendMessage dev，未自行改）**：`core/temp_ak_issuance/cleanup.py:36` `DeletePolicyVersionRequest(policy_type="Custom", ...)` —— 该 SDK 请求类无 `policy_type` 形参（仅 `policy_name/version_id`），构造即抛 TypeError 被内层 `except Exception: pass` 吞 → 方案 B policy 的非默认版本永不删除；若 policy 曾被 create_policy_version 加过版本，后续 DeletePolicy 会因残留版本失败→自定义 policy 泄漏（残留 artifact）。用 xfail(strict) 锁在 test_temp_ak_cleanup.py::test_revoke_deletes_nondefault_policy_versions，dev 去掉 policy_type 后应 XPASS→转红提示解除 xfail。非阻塞（新建 policy 只有单默认版本时不触发；且时间窗到期已服务端拒调用，安全面不受影响）。
  既存基线甄别：本地 .venv 装了 matplotlib(3.10.8)，test_gpu_distribution/test_wants_metrics_chart 本地绿；**容器 aiops-bot 无 matplotlib 时这两例 red，属既存环境基线、与 #57 无关**。未覆盖/风险：真机飞书审批 GET/私信 HTTP（integration 面，单测 mock）；DateInterval 真机 value 形状待 P1 真实提交反查（现容错解析）；火山云分支 P1（当前 validate 直接拒）。

[2026-07-22 TESTER] #57 cleanup bug 修复收口：dev 已删 `DeletePolicyVersionRequest` 的误传 policy_type（cleanup.py:38，只留 policy_name/version_id）。去掉 test_revoke_deletes_nondefault_policy_versions 的 strict-xfail，改成正常回归断言（非默认版本 v2 被删、在 delete_policy 前、且请求带正确 version_id 无 policy_type）。全量默认 pytest **1397 passed / 1 skipped / 9 deselected / 0 failed / 0 xfailed**。#57 补测全绿，可过闸。

[2026-07-22 TESTER] #58 批2（全局审批白名单 + P0.5 延期）补测完成（本地 .venv py3.11，全量默认 pytest **1442 passed / 1 skipped / 9 deselected / 0 failed**，+45 例）。2 新文件：
  `test_temp_ak_allowlist.py`(11)：`_approval_allowlist()` 启用含 {RAM+TAK+EXT}/停用仅 {RAM}/RAM 空→空集；feishu_event（app.test_client + FakeThread 捕获 target）：RAM code→ram_approval.handle_approval_event、TAK→handle_temp_ak_event、EXT→handle_temp_ak_extend_event 各命中且不误起他者；**非白名单 code 丢弃（无线程）**、**无 code 审批事件丢弃**、TEMP_AK_ENABLED=False 时 TAK 也丢弃；非审批消息(is_approval_like False)不受白名单影响（image 消息早退、不误起审批处理器 + 直测 is_approval_like False）。
  `test_temp_ak_extend.py`(34)：should_handle_extend（==EXT 命中/发放 TAK 不命中/无 code/未配）；parse_extend（by_id+by_name 参数化、带 start、缺凭证ID/缺到期各抛）；**_verify_enterprise MED-1 fail-safe**（orig 有值+ent 空→抛、orig 空→跳过、精确匹配 ok、宽松包含 ok、不符抛）；handle_extend 门禁（实例 PENDING 不动/无 instance_code fail-safe/grant 不存在抛/REVOKED 抛「已清理」/企业���符抛/happy 恰一次/同 extend 实例二投 extend_already_applied 不重复）；extend_grant（方案B creds=None+改窗+AK 不变+extends 追加+extend_instances、STS≤12h 重签发 sts、STS>12h 转 ram+ak_id 更新+补 policy_name、stage∈{NEW,REVOKED,FAILED} 拒、新到期已过拒、生效≥到期拒）；rewrite_ram_window（mock RAM 断言 create_policy_version set_as_default+policy_name+含 +08:00、缺 policy_name 抛）；deliver_extend（重签发→私信凭证卡含 secret+群回执脱敏、同AK→extended_card 无 secret+群回执、extended_card 不含 secret/token）。
  已纳入 auditor MED-1 修复（_verify_enterprise 留空拒绝防串）。未发现新源码问题。matplotlib 两例本地绿（.venv 装了）、容器无则 red 属既存基线。批2 全绿可过闸。

[2026-07-22 TESTER] 🔴 测试卫生事故修复（真实飞书发送泄漏）：temp_ak 错误分支用例调真实 handler→approval._notify_internal_failure→dsw_scheduler._send_text，跑在真 .env 上真往用户飞书群发消息、重跑刷屏。**只动 tests/**：在 `tests/conftest.py` 加 autouse(function) fixture `_no_real_feishu_or_network`，全局兜底桩 `core.dsw_scheduler._send_text`/`_send_card`（空操作+记录）+ `requests.post`/`requests.get`（返 200/{"code":0} 假响应，不出网）。**未桩 notify._get_access_token**（test_notify_token_cache 要测真函数；requests 兜底已挡住取 token 的真实 HTTP，网络安全不依赖桩它）。测试自带的 monkeypatch（send_spy 等）在 autouse 之后跑、覆盖桩，断言不受影响。加桩后全量默认 pytest **1442 passed / 1 skipped / 9 deselected / 0 failed**，运行时 16s→8s（旁证此前确有真实网络调用、现已全停）；token-cache 真函数用例仍绿；temp_ak 208 例全绿。零真实发送。

[2026-07-22 TESTER] #58 续 权限模型重构（read/download/write 三者正交）补测重写完成（本地 .venv py3.11，全量默认 pytest **1456 passed / 1 skipped / 9 deselected / 0 failed**）。把 temp_ak 全部测试从旧 read_prefixes/write_prefixes 模型迁到 prefix+caps 模型（222 例）：
  · policy 重写（21 例）：**read→有 ListObjects+GetBucketAcl、无 GetObject**；**download→有 GetObject/GetObjectAcl、无 ListObjects**；**write→PutObject/Abort/ListParts、无 DeleteObject、无 GetObject**；combo 三合一无 delete；read+write（无download）仍无 GetObject；空 caps→空 Statement；模块动作集交叉污染断言（LIST 无 GetObject、DOWNLOAD 无 List、WRITE 无 delete）；Resource（read=桶级+oss:Prefix、download/write=对象 ARN）；整桶 prefix="" 无 oss:Prefix；每条 statement 时间窗 +08:00/Date* AND + IpAddress；session≤2048 + 超限抛。
  · _parse_perm→caps list（真实三原子项 read/write/download：["read"]→[read]、["download"]→[download]、["write"]→[write]、多选 read+download 合并、全三；空→[]；防御性 'read&write'→{read,write} 注明已非真实选项）。
  · parse_temp_ak_request→spec(prefix+caps，无 read_prefixes/write_prefixes，含负断言)；read-only/download-only/write-only 派生；_validate_spec caps 空→拒（替换旧「缺读写前缀」）。
  · orchestrator create_grant_record 存 prefix+caps + platform/enterprise；scope_line 含目录+权限中文（列/下载/上传）+ 整桶。
  · issuer plan/_issue_sts/assume_role_with_policy 用 prefix+caps；tool plan（bucket+prefix+caps，新增 caps→policy 精确反映 read+download 有 List+GetObject 无 delete、read-only 无 GetObject 两例）；delivery/extend/cleanup grant 夹具全迁 prefix+caps。
  未发现新源码问题（源码 _parse_perm 按每勾选项独立判 caps，符合三原子项）。matplotlib 两例本地绿、容器无则 red 属既存基线。#58 续全绿可过闸。

[2026-07-22 TESTER] #59（命名规范化 + 下发改审批评论）补测完成（本地 .venv py3.11，全量默认 pytest **1470 passed / 1 skipped / 9 deselected / 0 failed**）。
  A 命名（新文件 test_temp_ak_naming.py，16 例）：_ascii_slug（TEST→test、纯中文无pypinyin→ext、混合Acme科技→acme、空/空白/None→ext、截20、只留[a-z0-9]；**注入假 pypinyin 模块** monkeypatch sys.modules 测中文→拼音分支 moumou/acmekeji）；_derive_user_name（tempak-<slug>-<6hex> 确定性、中文企业→ext-slug、只 RAM 安全字符）；display_name_for（<企业>-临时外采用户 中文OK / 空→临时外采用户 / 截128）；issuer._issue_ram 建号 create_user 传 display_name=display_name_for(grant)（假 RAM client 断言）。
  B 下发改审批评论（重写 test_temp_ak_delivery.py，16 例）：deliver(creds)→ram_approval._send_approval_comment(instance_code=发放实例, text 含 secret)+内部群脱敏 receipt_card；comment_user_id 用 requester、空→回退 _approval_comment_user_id；评论失败+ram→_alert_creds_undelivered(不含 secret)、STS 失败不告警；creds=None 只推回执。deliver_extend 贴**延期实例**(extend_instances[-1]，无则回退 instance_code)：creds→凭证评论含 secret、None→_extended_text(无 secret)评论。credential_text 含 secret（STS 带 Token 行/RAM 无）、_extended_text 无 secret、receipt/status 卡脱敏、grant 记录无 secret 键。
  据 auditor LOW-1 + dev：**cards.credential_card/extended_card 已删**（评论下发后无调用方、credential_card 曾渲染 secret）→ 删掉 test_temp_ak_extend.py 里引用 extended_card 的用例（deliver_extend 评论行为已在 delivery 测试覆盖）。旧 DM 私信/credential_card/_feishu_send_interactive_checked 用例全部移除。
  temp_ak 全套 236 例绿。conftest 兜底桩在、零真实发送。未发现新源码问题（pypinyin 降级、评论 instance_code 定向、脱敏均与断言吻合）。matplotlib 两例本地绿、容器无则 red 属既存基线。#59 全绿可过闸。

[2026-07-22 TESTER] #60（延期审批「延长/撤销二合一」）补测完成（本地 .venv py3.11，全量默认 pytest **1493 passed / 1 skipped / 9 deselected / 0 failed**）。temp_ak 全套 259 例绿。
  改 test_temp_ak_extend.py（+_parse_extend_action、+撤销分支 + 5元组）：_ext_detail 加 撤销/延长 控件(widget17847115181700001，by_id+by_name)；parse 现返 **5 元组** (action,grant_id,enterprise,not_before,expire) — 旧 4 元组解包全改；_parse_extend_action（撤销/吊销/撤回/revoke(大小写)→revoke；延长/空/其它/None→extend 默认向后兼容）；parse 缺到期不再抛（改断言 exp==0，校验挪 handler）。撤销分支 handler：APPROVED→防串→cleanup.revoke_grant 被调(桩)且不走 extend/deliver、res.action=revoke/ok=True + 内部通知；落 revoke_instances 幂等；同撤销实例二投→revoke_already_applied 不重复删；grant 已 REVOKED→already_revoked 短路+通知；部分失败(revoke_grant False)→ok=False+通知人工核查；防串对撤销同样生效(乙公司拿甲公司 grant_id→拒)；grant 不存在→拒；撤销实例 PENDING→门禁拦。延长分支：缺到期→handler 抛（新增）。_notify_internal_action 真函数→dsw_scheduler._send_text（含 grant_id/enterprise、无群配置 noop）。
  **对��� dev 又一处未在任务描述里的 delivery delta**：#60 起 `delivery.deliver` **不再推内部群回执卡**（源码注释「信息只在审批评论里，不单独推群/推人」，`_send_internal_receipt` 已删）→ 改 test_temp_ak_delivery.py 三例（deliver 有/无 creds 均 spy["cards"]==[]；延期同 AK 不再断言群回执）。receipt_card/status_card 卡函数仍在、脱敏单测保留。
  未发现新源码问题（撤销幂等/防串/门禁、5 元组、action 解析均与断言吻合）。matplotlib 两例本地绿、容器无则 red 既存基线。#60 全绿可过闸。

[2026-07-22 TESTER] #60.1（凭证评论身份 bug：多级审批下 requester 解析成审批人→评论错挂）补测完成（全量默认 pytest **1495 passed / 1 skipped / 9 deselected / 0 failed**）。**注意 dev 二次改法**：最终不是新增 TEMP_AK_COMMENT_USER_ID，而是 `delivery._comment_user_id(grant)` 直接复用 `ram_approval._approval_comment_user_id()`（FEISHU_RAM_APPROVAL_COMMENT_USER_ID → ADMIN_FEISHU_OPEN_ID，忽略 grant.requester），与 RAM 建号审批评论身份一致。
  改 test_temp_ak_delivery.py：spy 桩 `ram_approval._approval_comment_user_id`→ou_admin；旧「user_id==requester」用例改成 `test_deliver_comment_user_id_is_admin_not_requester`（grant.requester=ou_bob 但评论 uid==ou_admin≠ou_bob）；_comment_user_id 直测三例（忽略 requester 用 admin / FEISHU_RAM_APPROVAL_COMMENT_USER_ID 优先 / 两者空→""）。**撤掉**先前按 coordinator 首版消息写的 TEMP_AK_COMMENT_USER_ID settings 字段用例（该字段最终未落地，源码无此属性）。deliver 断言 _send_approval_comment 的 user_id 为管理员身份、非 requester。未发现新源码问题。temp_ak 全套 261 例绿。

## #62 policy.py 桶信息条独立成条修复线上「拿凭证访问不了桶」bug（dev 已改，tester 已补测）
- 根因：原 `build_policy_with_window` 把桶级动作 GetBucketInfo/Stat/Acl 与 ListObjects 塞进同一条 read 语句、整条带 oss:Prefix StringLike 条件；GetBucket* 桶级操作请求不带 prefix → 被 oss:Prefix 条件卡死 → 拒绝。dev 拆成独立语句对齐用户权威模板（temp-ak-auto-tempak-nuoyiteng-7df6a7）：桶信息条(无 Condition/无 Prefix) + List 条(带 Prefix+时间窗) + 下载条(GetObject) + 写条(无 delete)。
- [2026-07-23][TESTER] 重写 `tests/unit/test_temp_ak_policy.py`（21→30 例）对齐新权威模板。**既存 5 failed 甄别**：全部由本次 working-copy 改动引入（旧文件锁的是 pre-bug 模型：`GetBucketAcl in LIST_ACTIONS`、download 含 `GetObjectAcl`、`每条 statement 都有 Condition`），已随源码语义更新，非无关既存失败。覆盖：read 单勾=桶信息(无Condition)+List(Prefix+时间窗) / download 单勾=桶信息+GetObject(Resource=桶/前缀*) / write 单勾=只写条无delete且**不出桶信息条** / 全勾四条顺序=桶信息/List/下载/写；**关键回归**（参数化）桶信息条绝不带 oss:Prefix/任何 Condition + GetBucket* 三件套不在带 Prefix 的 List 语句里；caps 空/None→空 Statement；整桶 prefix=""→List 无 oss:Prefix；时间窗只作用于 List/下载/写三条(桶信息条无)；session≤2048 校验 + PolicyTooLargeError。全量 pytest **1504 passed / 1 skipped(h5py 既存) / 9 deselected(integration) / 0 failed**（本地 .venv py3.11，装了 matplotlib 故 gpu_distribution/wants_metrics 本地绿；容器无 matplotlib 才 red，既存基线、非本改动）。未发现新源码问题。
