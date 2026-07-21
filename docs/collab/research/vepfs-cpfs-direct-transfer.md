# 研究 + 侦察：vePFS ↔ CPFS 直接传输（跨云 PFS 直传编排层）

> researcher，2026-07-16。给 planner 拆方案用。只读代码调研 + 一次 Web 交叉验证。
> 结论标注：【文档】官方明说 /【实测】本仓真机反查沉淀 /【代码】读源码所得 /【推测】未证实。

## 0. 一句话结论

物理上 **没有** vePFS↔CPFS 直连（Web 交叉验证确认，见 §6），必须走 **3 段链**：
`源PFS --沉降Export--> 源云对象存储 --跨云迁移--> 目的云对象存储 --预热Import--> 目的PFS`。

**三段的引擎积木本仓全部已存在且大多真机验过**（`engine_vepfs` / `engine_nas` / `engine_mgw` / `engine_tos`）。
缺的是 **把三段串起来的"链式"上层编排** —— 现有 `core/transfer/` 的 Phase-3 **只到 2 段**（沉降+跨云，落在对方**对象存储**），
**没有第三段预热进目的 PFS**，且 vePFS 侧沉降**根本没 wire**。这是一个纯**加法式新建编排包**的活，不改现有引擎。

---

## 1. 三段积木的现成能力与接口（读代码，file:line 证据）

### 1.1 vePFS 数据流动 `core/vepfs_dataflow/`（火山，沉降 Export + 预热 Import）

- **引擎 `engine_vepfs.py`**：`submit_task(fs_id, task_action, tos_bucket, tos_prefix, sub_path, fileset_id, region, same_name_policy, data_type)` → 返回 `data_flow_task_id`（engine_vepfs.py:116-150）。方向只由 `task_action∈{Import,Export}` 决定，**源/目的字段不反转**（engine_vepfs.py:35-36, 121-123）。`query_task(task_id, fs_id, region)` 必带 `page_number/page_size` 分页（engine_vepfs.py:159-161，#44 真机锤定不带分页会死轮询）。终态判定 `is_failed`/`is_done` 子串归类、**先判 failed**（engine_vepfs.py:67-74）。
- **凭证**：静态 `TOS_ACCESS_KEY/SECRET`，火山无 STS（engine_vepfs.py:79-82）。
- **无 CreateDataFlow 绑定**：直接 submit 带 TOS 桶/前缀 + vePFS `sub_path`/`fileset_id`（engine_vepfs.py 模块 docstring:16-18）。比阿里省掉 resolve/临建/临删。
- **orchestrator.py**：状态机 `NEW→RUNNING→DONE|FAILED`（:36-39），Redis `vepfs:dataflow:job:{job_id}`（:32），`run_to_completion(job, on_update, poll_interval=60, max_polls=1440)` 阻塞轮询、每次阶段变化回调 `on_update`（:344-366）。`make_plan`/`plan_from_addresses`（源/目的地址自动定方向，:160-222）。job 结构见 create_job_record（:263-287）：`operation/action/fs_id/region/sub_path/tos_bucket/tos_prefix/task_id/stage/bytes_*/created_by`。
- **成熟度**：【实测】双向真机跑过（notes #44：vepfs-3dd16c Finished、vepfs-1261 FAILED 用户漏填桶名）。字段格式（裸桶名、路径首尾斜杠）真机反查沉淀在 `docs/vepfs-dataflow-sdk-usage.md`。

### 1.2 CPFS/NAS 数据流动 `core/cpfs_dataflow/`（阿里，沉降 Export + 预热 Import）

- **引擎 `engine_nas.py`**：通用 tea-openapi `call_api`，产品 NAS 2017-06-26（:4, :26）。核心：`resolve_dataflow`（选已有 DataFlow，:249-291）、`submit_task(fs_id, data_flow_id, action, directory, dst_directory, conflict_policy, ...)`（:296-336）、`query_task`（:339-379）。**三期沉降复用薄封装已就绪**：`submit_sink_job(fs_id, cpfs_dir, oss_bucket, oss_prefix, region, open_id)` → 返回 `'TaskId@DataFlowId'`（:394-407）、`poll_sink(fs_id, sink_ref, region)`（:410-416）。
- **关键前置**：CPFS 侧必须**已有** DataFlow 绑定（或 orchestrator 临时建 `create_dataflow`+用完删 `_cleanup_ephemeral`，见 orchestrator.py:310-361, 400-412）。engine 层的 `submit_sink_job` 走 `resolve_dataflow`，**要求 DataFlow 已存在**，否则抛错（engine_nas.py:262-291）。
- **orchestrator.py**：状态机 `NEW→RUNNING→DONE|FAILED`（:33-36），Redis `cpfs:dataflow:job:{job_id}`（:29），`run_to_completion`（:427-450，终态删临时 DataFlow）。`make_plan`/`plan_from_addresses`（:136-226）。
- **成熟度**：【实测】双向真机跑过（notes #42 预热多套一层 /cpfs 的 bug 就是真机预热暴露；#44 P0.5 确认 CPFS `DescribeDataFlowTasks` 无 vePFS 那种分页坑）。智算版(`bmcpfs-`)/通用版(`cpfs-`)分支（engine_nas.py:49-51, 322-325）。

### 1.3 跨云迁移 `core/transfer/`（对象存储↔对象存储，方向定引擎）

- **`paths.py`**：`_ENGINE_BY_DEST = {"oss":"mgw", "tos":"tos_mig"}`（:31）——**引擎由目的端决定**（目的端拉取）。`build_plan` 已支持 cpfs/vepfs 源（:107-133），会推导 `sink_target` + `dest`（跨云对象存储），但 **dest 止于对象存储**。
- **`engine_mgw.py`（进 OSS = 阿里在线迁移 hcs_mgw）**：`submit_cross_job(job_name, src_bucket, src_prefix, src_scheme, ..., dest_bucket, dest_prefix, dest_region, ...)`（:227-289），`poll_job`（:292-303）。源 TOS 用静态 AK（`create_tos_source_address` address_type="tos"，:69-81）、目的 OSS 用 RAM role（:90-104）。
  - **成熟度**：【实测】Phase 1 **TOS→OSS 真机全 SDK 验过**（`wuji_il` 路径，CLAUDE.md）。这是**唯一真机验过的跨云方向**。
- **`engine_tos.py`（进 TOS = 火山 DMS 1.0）**：`submit_cross_job(..., src_is_tos=False)` 建 `create_data_migrate_task`（:67-114, 174-206），`query_task`/`poll_job`（:140-217）。
  - **成熟度**：【实测】**桶间 TOS→TOS 真机验过**（`StorageVendorTOS` task 553276，bucket_transfer 用）；【推测/待验】**跨云 OSS→TOS**（`StorageVendorOSS`）虽字段照真机金标准 task 552548 配置，但 CLAUDE.md 明说 "Phase 2 pending Volcano migration OpenAPI verification"——**跨云 OSS→TOS 未真机端到端验证**。
  - **已知硬限制**：【实测】DMS 目的**只能到桶级**，对象保持源 key 原样落入目的桶，**dest_prefix 不可靠**（engine_tos.py:186-187, CLAUDE.md "Known limitation"）。→ 影响 CPFS→vePFS 方向的落盘路径可控性（见 §4/§5）。

---

## 2. Phase 3 现状：wire 到哪一步，缺口在哪（file:line）

**结论：transfer 的 Phase 3 是"2 段链、落对象存储"，不是"3 段链、落 PFS"；且只 wire 了 CPFS 沉降，vePFS 沉降没 wire。**

- `orchestrator.start_sinking`（transfer/orchestrator.py:286-317）：**只支持 `src.scheme == "cpfs"`**，`if src.scheme != "cpfs": FAILED "vepfs 沉降暂未实现"`（:295-300）。调 `engine_nas.submit_sink_job`。→ **vePFS→TOS 沉降段完全没接进 transfer**（缺口①）。
- `poll_sink_once`（:320-340）：只调 `engine_nas.poll_sink`（写死 CPFS）。→ vePFS 沉降轮询也没有（缺口①同源）。
- `run_to_completion`（:385-437）：`needs_sink` 时先 `start_sinking`→轮询 `poll_sink_once` 到 `sink_done`→再 `start_cross`。**结束即 `start_cross` 的跨云段终态（dest=对象存储），没有第三段"预热进目的 PFS"**（缺口②，最关键）。
- `paths.build_plan`（paths.py:99-133）：cpfs/vepfs 源 → `dest = _CROSS_TARGET[...]` 对方**对象存储**。**Plan 里没有"最终目的 PFS"这一维**——它把 `vepfs://` 当"目的是阿里、落 oss"，把 `cpfs://` 当"目的是火山、落 tos"（paths.py:16-17 注释即如此设计）。→ **数据模型层就没有目的 PFS 概念**（缺口②的根）。
- 对账 `dsw_scheduler._dataflow_reconcile_specs`（dsw_scheduler.py:847-859）：transfer 的 `active` 只含 `{STAGE_CROSSING}`，**不含 `STAGE_SINKING`**（:848）——Phase-3 沉降段孤儿重启后不会被对账兜底（既存 LOW，Phase 3 未上线故未补）。

**小结：** transfer 现有的三期骨架对本功能只能贡献"沉降+跨云"两段的一半参考（且沉降只 CPFS）。**它不能被直接复用来做 PFS→PFS**，因为它的 Plan/状态机根本不建模"目的 PFS + 预热段"。

---

## 3. 编排层：能不能链式串三段 orchestrator

**能，而且很自然。** 每个 orchestrator 的 `run_to_completion(job, *, on_update, poll_interval, max_polls)` 都是**自包含、阻塞到终态、每阶段回调 on_update** 的（vepfs:344 / cpfs:427 / transfer:385）。上层"链编排器"只需：

```
在一个后台线程里顺序调用（伪代码）：
  1) 源云沉降段:   {vepfs|cpfs}_orchestrator.run_to_completion(sink_job, on_update=chain_cb)
                   → 成功再继续，失败则整链 FAILED
  2) 跨云段:       transfer.start_cross + 轮询（或直接复用 transfer 的对象存储→对象存储那半段）
  3) 目的云预热段: {cpfs|vepfs}_orchestrator.run_to_completion(preheat_job, on_update=chain_cb)
```

**要点 / 注意：**
- 三段各自 `run_to_completion` 是**阻塞**的（内部 `time.sleep(poll_interval)` 循环），串起来总时长 = 三段之和，必须跑在**后台线程**（本项目一贯做法，见各 `_h_confirm_*`）。
- 每段有独立 job_id 命名空间（`vepfs-*`/`cpfs-*`/`tr-*`）与独立 Redis 记录 → 天然可分别 refresh/查询。**链任务本身需要新的第 6 个命名空间**（如 `pfs2pfs:job:{chain_id}` 或 `xpfs-*`），存三段各自子 job_id + 当前段指针 + 汇总进度。
- **续跑/重启**：现有 5 条链靠 `refresh()`（点查询自愈）+ 对账 `_reconcile_dataflow_once`（stale 门 180s + NX 闸门去重，dsw_scheduler.py:865-878）。链编排器要跑通"重启续跑"，需把自己也加进 `_dataflow_reconcile_specs`（:847），并暴露 `_KEY_PREFIX/get_job/refresh/_save/result_card`（:836 契约）。链的 `refresh` 要能判断"卡在第几段"并续推该段（比现有单段 refresh 复杂——这是本活最需要设计的点）。

### 3.1 中转对象存储（staging 桶+前缀）怎么定 —— **需要新配置**

三段链要 **两个** staging 对象存储位置：
- **源云 staging**（沉降落点）：vePFS→**TOS 桶**（须与 vePFS 同 region，VEPFS_REGION）；或 CPFS→**OSS 桶**（须与 CPFS 同 region，且**有 DataFlow 绑定**或临建）。
- **目的云 staging**（跨云落点 = 预热源）：跨云段的 `dest_bucket` 必须 == 目的云预热段读的那个桶。CPFS 预热要求该 OSS 桶**有 DataFlow 绑定**；vePFS 预热要求该 TOS 桶**同 region + 服务授权**。

现有 `TRANSFER_BUCKET_MAP`（settings.py:216-220）键 `'<scheme>://<bucket>'`、值目的桶，**不适配**——它是"对象桶→对象桶"映射，没有"PFS fs-id → 本云 staging 桶(+region+prefix)"的概念。**建议 planner 设计新配置**，例如 `PFS_STAGING_MAP`：`{"vepfs://<fs>": {"tos_bucket":..., "region":..., "prefix":...}, "cpfs://<fs>": {"oss_bucket":..., "region":..., "dataflow_id":...}}`。或复用 cpfs/vepfs discovery（`core/cpfs_dataflow/discovery.py`、`core/vepfs_dataflow/discovery.py` 已能发现 PFS↔对象存储绑定）来自动定位源云 staging 桶。

### 3.2 job 状态怎么表达三段进度

建议链 job 存：`chain_stage ∈ {NEW, SINKING, CROSSING, PREHEATING, DONE, FAILED}` + 三个子 job_id（`sink_job_id`/`cross_job_name`/`preheat_job_id`）+ 当前段字节/对象进度透传。汇总进度卡显示"第 X/3 段 + 该段进度"。

---

## 4. 方向对称性：两个方向的引擎组合 + 成熟度

| 段 \ 方向 | **vePFS → CPFS** | **CPFS → vePFS** |
|---|---|---|
| ① 沉降(源云 PFS→对象存储) | vePFS→TOS：`vepfs_dataflow` Export ✅【实测】 | CPFS→OSS：`cpfs_dataflow` Export ✅【实测】 |
| ② 跨云(对象存储→对象存储) | TOS→OSS：`transfer.engine_mgw`（进 OSS）✅【实测 Phase1 wuji_il】 | OSS→TOS：`transfer.engine_tos`（进 TOS，DMS）⚠️【跨云未真机验，且目的只到桶级】 |
| ③ 预热(对象存储→目的云 PFS) | OSS→CPFS：`cpfs_dataflow` Import ✅【实测】 | TOS→vePFS：`vepfs_dataflow` Import ✅【实测】 |
| **整链成熟度** | **三段引擎全真机验过** → **建议先做** | 卡在②跨云 OSS→TOS 未验 + 落盘只到桶级 → 依赖 Phase-2 先收口 |

**关键判断（给 planner 排序）：**
- **vePFS→CPFS 方向的每一段引擎都真机验过**（沉降 vepfs、跨云 mgw、预热 nas），是**风险最低、应先落地**的方向。
- **CPFS→vePFS 方向**的跨云段 `engine_tos` 跨云 OSS→TOS 从未真机端到端验证（只验过同云 TOS→TOS），且 DMS 目的**只到桶级、dest_prefix 不可靠**（engine_tos.py:186）——落进 vePFS 后目录结构可能与源不一致，需先把 Phase-2 跨云 OSS→TOS 真机验收 + 解决落盘前缀，才谈得上这个方向。

---

## 5. 风险 / 待确认清单

1. **【最高】跨云 OSS→TOS 未验（影响 CPFS→vePFS）**：engine_tos 跨云路径 Phase-2 pending verification（CLAUDE.md）。且 DMS **目的只到桶级、无法指定子目录**（engine_tos.py:186-187）——CPFS→vePFS 时对象以源 TOS-staging 的完整 key 落进目的 TOS 桶，再预热进 vePFS，落点目录可能非预期。→ 需 researcher 真机反查 / dev 真机验收，或对该方向的 staging 桶做"专桶专用 + 前缀预整形"。
2. **区域约束（硬）**：【文档/实测】CPFS↔OSS 必须同 region（`CPFS_REGION`，settings.py:230）；vePFS↔TOS 必须同 region（`VEPFS_REGION`，settings.py:254）。跨云段可跨 region。→ 链的 staging 桶 region 由各自 PFS region 决定，planner 配置要显式带 region。
3. **CPFS 侧 DataFlow 前置**：CPFS 的沉降/预热要求 OSS 桶与 CPFS 有 DataFlow 绑定；`resolve_dataflow` 找不到会抛错，orchestrator 层可临建+用完删（cpfs orchestrator.py:310-361），但 engine 层 `submit_sink_job` 不临建。→ 链编排应走 cpfs **orchestrator**（含临建逻辑），别直接调 engine 的 `submit_sink_job`。
4. **中转数据清理**：三段链在两个 staging 桶留下全量副本。是否/何时清 staging？沉降+预热都是"同名跳过/不覆盖"语义（安全），但 staging 长期堆积会涨容量。→ planner 决策：保留（幂等续跑友好）还是终态后清（需额外删对象逻辑，本仓无现成"删桶前缀"编排）。**建议先保留、不自动删**（对齐现有"永不覆盖"保守风格）。
5. **审批门**：三段各有独立 `needs_approval`（transfer TB 级 / cpfs·vepfs GB 级，阈值不同：TRANSFER_APPROVAL_TB / CPFS_APPROVAL_GB / 无 vepfs 阈值）。链应**在链入口按预估源大小审批一次**，别三段各弹一次。源大小预估：vePFS/CPFS 侧本仓**无现成 PFS 目录测大小工具**（transfer.estimate_source 只测对象存储，:118-149，全闪源明说"暂不探测"），可能要 sink 完拿 staging 桶大小、或走各 SDK。→ 待确认预估手段。
6. **幂等/续跑/失败重试衔接**：每段 orchestrator 幂等 job_id=hash(...当天)、retry 重置 stage=NEW + 清 launch/notified 键（notes #37/#50 大量修过）。链层要设计"从失败段续跑"（沉降成功、跨云失败 → 重试只重跑跨云+预热，别重跑沉降）。这是链编排的核心状态机设计，参考 `core/ssh_transfer` 两段链的 retry（"段1已成功只重段2"，notes #51）——**ssh_transfer 是本仓最接近的多段链先例，强烈建议 planner/dev 参照它的骨架**（paths/engine/orchestrator/cards/messages/actions 六件套 + 段级 rc + retry 跳过已成功段）。
7. **火山无 STS**：vePFS/TOS 全程静态 AK（`TOS_ACCESS_KEY`）；阿里 CPFS/OSS/MGW 走 STS/master AK。链跨两云，凭证来源不同，`created_by`(open_id) 只对阿里段有身份意义。
8. **终态通知去重**：链有三段、每段 orchestrator 自己也会 on_update。链编排若复用各段 `run_to_completion` 的 on_update，要避免"每段都推一张完成卡"——应由链 cb 统一收敛，只在**整链**终态推结果卡，走 `_claim_dataflow_notify` NX 闸门（dsw_scheduler.py:868）。

---

## 6. 有没有官方"跨云 PFS 直传"能省掉中间段

**没有。**【文档】Web 交叉验证：vePFS 与 CPFS 是两家竞品并行文件系统，各自的"数据流动"只对接**本厂对象存储**（vePFS↔TOS、CPFS↔OSS），**无任何直连 connector**。官方与第三方分析均确认"借助数据流动模块打通并行文件存储↔对象存储"是唯一模式，跨云必须以对象存储为桥。→ 3 段链是**不可省的物理下限**，本功能就是编排这 3 段。

来源：
- [实现 vePFS 与对象存储 TOS 之间数据流动 - 火山引擎](https://www.volcengine.com/docs/6645/1175846)
- [数据流动概述 - 文件存储 vePFS - 火山引擎](https://www.volcengine.com/docs/6645/1536432)
- [高性能计算并行文件存储 CPFS - 阿里云](https://help.aliyun.com/zh/cpfs/)
- [将 vePFS 中的数据定时备份到对象存储 - 火山引擎](https://www.volcengine.com/docs/6645/1174720)

---

## 7. 给 planner 的"复用 vs 新建"清单

**直接复用（不改）：**
- `core/vepfs_dataflow/orchestrator.py` 的 `make_plan`/`create_job_record`/`run_to_completion`/`refresh` —— 当沉降段(Export)与预热段(Import)引擎。
- `core/cpfs_dataflow/orchestrator.py` 同上（含临建 DataFlow 逻辑）。
- `core/transfer/engine_mgw.submit_cross_job/poll_job`（TOS→OSS 跨云，已验）；`engine_tos`（OSS→TOS 跨云，待验）。
- `dsw_scheduler._claim_dataflow_notify` NX 去重闸门、`fmt_size/fmt_ts/fmt_duration`。
- 六件套模式先例：**`core/ssh_transfer/`（两段链）**是最佳参照。

**新建（加法，不碰现有引擎）：**
- **新编排包 `core/pfs_transfer/`（暂名）**：`paths.py`（解析源 PFS + 目的 PFS 两端地址、定方向、推导两个 staging 位置）、`orchestrator.py`（新链状态机 `NEW→SINKING→CROSSING→PREHEATING→DONE|FAILED` + 新 Redis 命名空间 + 三段子 job 编排 + 段级续跑/重试）、`cards.py`/`cli.py`。
- **新配置**：`PFS_STAGING_MAP`（PFS fs-id → 本云 staging 桶+region+prefix[+dataflow_id]），或复用两个 discovery 自动发现。链开关 `PFS_TRANSFER_ENABLED`、审批阈值、`PFS_TRANSFER_CHAT_ID`。
- **接入 3 入口**（照本仓惯例）：飞书卡片（新意图，须排在跨云 transfer 意图**之前**避免误抢）、Agent tool、CLI。
- **接入对账**：把新链加进 `_dataflow_reconcile_specs`，`active` 含三个在途段。
- **transfer Phase-3 的 `start_sinking`/`paths` 不必强行改**——本功能用独立链包更干净（transfer 那套"落对象存储"语义与"落 PFS"不同，硬塞会互相污染，同 bucket_transfer 当初与 transfer 独立的决策）。

**先做哪个方向：vePFS→CPFS（三段引擎全验过），CPFS→vePFS 待 Phase-2 跨云 OSS→TOS 真机收口后再上。**
