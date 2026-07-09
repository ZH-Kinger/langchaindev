# AIOps 飞书 Bot — 现状盘点 + 路线图 / Backlog（#31 初稿）

维护者：planner。事实源：`CLAUDE.md` / `docs/collab/notes.md` / git log / 代码结构。
本文只做规划，不改任务板；下列「建议 task」待 dev 过目后再落板 + 派 owner。
更新：2026-07-09。

---

## 一、现状盘点（已有能力 / 在做 / 未做）

### A. 已有能力（已合入、多数真机验收过）

运行形态
- 五种模式：rag / agent / hybrid / collab / bot（`main.py` RUNNERS 分发）。
- 飞书 bot：Flask `/feishu/event` + `/feishu/card_action` + `/health`，DSW 调度后台线程。
- 三级工具路由（快路 → LLM 意图路由 → 关键词兜底），`TOOL_GROUPS` import 期校验。

多租户 / 安全
- STS AssumeRole 按 open_id 取用户角色，master AK 只 STS+只读；临时凭证 Redis 缓存自动续期。
- 用户 AK Fernet 加密入 Redis；`encrypt_strict` 写路径强约束。

工具层（`tools/`）
- 阿里：PAI DSW / ECS / OSS（含 dir_sizes/tree/自动探测地域）/ SLS / RAM / Prometheus / GPU 顾问 / 集群健康 / 集群 MFU（多区域日报+交互卡）。
- 火山：TOS 容量盘点。
- 飞书：notify / cards 原语。Jira / GitHub / K8s / 本地 RAG。

调度与巡检（`core/dsw_scheduler.py`）
- Jira 工单轮询建 DSW、空闲/超时看护、容量监控（OSS+TOS，写飞书多维表）、每日晨报（DSW+MFU）、OSS 权限审计推送。

数据搬运 / 流动（四条链路 + 对账兜底）
- 跨云迁移 `core/transfer/`：一期 TOS→OSS 全 SDK 打通（在途参数待回填真值）。
- 同云桶间迁移 `core/bucket_transfer/`：OSS→OSS + TOS→TOS，真机验收通过。
- 阿里 CPFS/NAS DataFlow `core/cpfs_dataflow/`：预热/沉降，发现+可选卡。
- 火山 vePFS DataFlow `core/vepfs_dataflow/`：预热/沉降；参数已对齐真机（裸桶名+首尾斜杠）。
- 统一三步向导卡（选云→选地区→表单，`core/dataflow_cards.py`），阿里/火山共用。
- 幂等加固：submit/confirm 锁 + `SET NX` 去重；重启后在途任务 `_reconcile_dataflow_once` 周期对账，NX 闸门保证终态跨线程只推一次。

OSS 权限最小化（`core/oss_perm/`）
- 从飞书多维表生成按人最小权限 RAM 策略，两级粒度（桶级/目录级），CLI + bot 表单卡下发，审计对账（多授/少授/孤儿）。

GPU 卡分布（`tools/aliyun/gpu_distribution.py`）
- 地区×卡型 + 每用户在算卡数 + 趋势快照，实时 HTML 页 `/gpu/distribution`（token 门禁），飞书摘要卡。

IaC 脚手架（`infra/pulumi/`，未合入 git）
- Pulumi-Python 纳管 RAM/STS 地基骨架：EPIC.md + STORY.md + config/master/roles/groups/create_governance 模块 + README runbook。**代码框架在，尚未 import 对齐、未真机 apply。**

### B. 在做 / 半成品（有骨架，缺真值或验收）

- **跨云 transfer 二期 OSS→TOS**：`engine_tos.py` 在，待火山迁移 OpenAPI 真机核实；OSS→TOS 目标前缀落盘限制已知（DMS 只到桶级）。
- **跨云 transfer 三期全链路**：CPFS→OSS→TOS sink 已接阿里 CPFS；火山 vePFS→TOS 沉降接入 transfer 三期 `start_sinking` 仍是遗留 follow-up。
- **transfer 一期上线前真值回填**：`wuji_il` 的 TOS domain/bucket、OSS role/bucket/region、transfer/overwrite 模式、`TRANSFER_BUCKET_MAP`（`CLAUDE.md` 明列「go-live 前必填」）。
- **vePFS 四项真机反查**：`DataStorage` 桶串格式、task status 终态枚举、`vepfs:*` IAM 动作名、智算版是否必须 FilesetId（`CLAUDE.md` 列「待真机验证」）。
- **IaC INFRA-1 / INFRA-1B**：脚手架已写，卡在前置条件（管理员 AK、OSS state 桶、活体策略导出、CREATOR_GROUP 组名）+ import 对齐 no-op 未做。

### C. 未做 / 待议

- IaC 后续：INFRA-2 OSS 桶、INFRA-3 CPFS+DataFlow 绑定、INFRA-5 ACK、INFRA-6 监控、INFRA-7 火山（PAI INFRA-4 用户明确顺延）。
- 凭证轮换（安全事件遗留，见 MEMORY）：Master AK + 三把飞书凭证 + 服务器 root/Redis + Fernet key + PAI_DSW AK Secret 全部待轮换。**属最高安全优先级，但落地在 dev/运维，非本项目代码。**
- 无 lint / type-check；测试仅 pytest。

---

## 二、当前零散在飞项 → 收敛为可执行 task（建议，待 dev 过目）

> 编号用 `P-`（proposal）临时前缀，避开现有任务板 #28-32；dev 认可后转正式 task 并派 owner。

### P-1 · transfer 200830 原地替换失效（真 bug，方案② 落地）
- **背景**：researcher 查证飞书 200830 = 2.0 确认卡不能原地更新为 1.0 进度卡，四条链路确认 handler 全部命中静默拒绝；且 `4219cc3` 同时删了中间态进度推送 → 确认后到完成前用户无「进行中」反馈。连点安全（launched+NX 挡住）。
- **决策（dev 已定）**：方案②——确认后回 toast + 另推一张 1.0 新卡（同查询路做法，改动最小），不做原地替换。
- **产出**：四条确认 handler（`_h_confirm_transfer` L503 / `_h_confirm_cpfs_dataflow` L745 / `_h_confirm_vepfs_dataflow` L862 / `_h_confirm_bucket_transfer` L1019，`core/feishu_bot/actions.py`）改为 toast+新推进度卡。
- **验收**：① 确认后用户立即收到独立「进行中」卡（非原地、无 200830 日志）；② 完成/失败卡照常（NX 闸门只推一次）；③ 连点仍不起多线程/不刷卡风暴。
- **owner**：dev 实现 → tester 覆盖（四链路确认→进度卡推送断言）→ auditor 复审。
- **工作量/风险**：中/中。**开放问题**：范围是先修 transfer 单条、还是四条一起改？（见开放问题 Q1）

### P-2 · 待提交产物落库（测试 + 协作文档 + infra 脚手架）
- **背景**：`git status` 三个测试文件 modified 未提交（test_bucket_transfer / test_dataflow_reconcile / test_transfer_confirm_idempotent），`docs/collab/`、`infra/` 全 untracked。
- **产出**：dev 分类提交（测试一提交、协作文档一提交、infra 脚手架一提交或单列分支）。
- **验收**：`git status` 干净；容器内相关 pytest 绿。
- **owner**：dev（tester 已在 notes 报告 43 passed）。
- **工作量/风险**：低/低。**注意**：infra/ 是否纳入主仓提交见 Q2。

### P-3 · test_gpu_distribution 预存红修复
- **背景**：`test_dist_url_and_summary_card`（L148-155）断言 `dist_url()==".../gpu/distribution?token=TK"`，但 dist_url 已改真实签名 token → mock 未同步，1 failed 长期挂红。与四修复无关。
- **产出**：确认源码签名 token 行为为最终态后，同步测试 mock/断言（若行为待定则先冻结）。
- **验收**：`pytest tests/unit/test_gpu_distribution.py` 全绿，基线回到 0 failed。
- **owner**：tester 改测试（若需改源码签名逻辑则 dev）。
- **工作量/风险**：低/低。**开放问题**：签名 token 是既定最终行为吗？（Q3）

### P-4 · 审计低危遗留（记录在案，非阻塞，可排后）
- 来自 notes 的 auditor 复审：
  - **P-4a** claim-before-send 无回滚：抢到 NX 闸门方 `_send_card` 抛异常 → 闸门已占+卡没发+对账跳过 → 完成卡永久丢。缓解：send 失败时 `delete` 闸门键放开。owner dev，tester 补异常路径用例。
  - **P-4b** launched-但-NEW 孤儿：窄窗口，对账绝不能自 submit（保持只读），只清 `launched=False` + 进度卡留「重新下发」按钮。owner dev。
  - **P-4c** nit：`_TTL_NOTIFY` 定义顺序 / specs 导入不隔离 / cpfs 双 cleanup（已 try/except 吞，安全）——清理级，低优。
- **工作量/风险**：低/低。合并成一条「低危收尾」task，价值低，排在 P-1~P-3 之后。

---

## 三、transfer / dataflow go-live 收尾（上线闸门）

### P-5 · transfer 一期真值回填 + 上线核验
- **产出**：填 `wuji_il` 真值（TOS domain/bucket、OSS role/bucket/region、模式、`TRANSFER_BUCKET_MAP`）→ 真机跑一次 TOS→OSS 小样。
- **验收**：confirm→CROSSING→DONE 全绿，进度/结果卡正常，审批闸门按阈值触发。
- **owner**：dev（真值）+ researcher（如需核实火山/阿里迁移服务规格）。依赖：P-1（进度卡反馈修好后再上线体验才完整）。
- **工作量/风险**：中/中（依赖外部真值与真机）。

### P-6 · vePFS 四项真机反查 + 二期 OSS→TOS 核实
- **产出**：反查 vePFS DataStorage 桶串/status 枚举/IAM 动作名/FilesetId；核实火山迁移 OpenAPI（OSS→TOS 二期）。
- **验收**：dry-run 结论被真机确认，`engine_vepfs`/`engine_tos` 若有偏差 dev 修正。
- **owner**：researcher 取证 → dev 修正 → tester 回归。
- **工作量/风险**：中/中。

### P-7 · vePFS→TOS 接入 transfer 三期全链路
- **产出**：`core/transfer` 三期 `start_sinking` 支持 `vepfs→tos`，复用 `core/vepfs_dataflow`。
- **验收**：full-chain（vePFS→TOS→OSS 或反向）状态机走通。
- **owner**：dev。依赖：P-6。工作量/风险：中-大/中。

---

## 四、IaC（`infra/pulumi/`）—— 独立 epic，建议单列跟踪

已有完整 EPIC.md / STORY.md（INFRA-1 ~ INFRA-7，PAI INFRA-4 顺延）。规划成熟，缺前置条件与真机执行。建议：

- **P-8 IaC 前置条件就位**：管理员 AK（≠master）、OSS state 桶、`BotRole-*` 活体信任/权限策略导出、`CREATOR_GROUP` 真实组名。owner dev/运维（用户提供真值）。**这是 INFRA-1/1B 的唯一阻塞。**
- **P-9 INFRA-1 import 对齐 no-op**：`pulumi preview` 0 变更（AC3 核心闸门）。依赖 P-8。
- **P-10 INFRA-1B 创建权限收归**：Deny/Allow 双面策略 apply。依赖 P-9。

INFRA-2/3/5/6/7 按 EPIC 里程碑排（M2 存储 → M3 K8s → M4 收口），暂不细拆，待 M1 落地再展开。

---

## 五、建议优先级（价值 × 风险 × 成本）

| 序 | task | 理由 | 可并行? |
|----|------|------|---------|
| 1 | **P-1** transfer 200830 方案② | 真 bug，影响所有搬运链路用户反馈；dev 已定方案，可立即开工 | 与 P-2/P-3 并行 |
| 2 | **P-3** test_gpu 修红 | 基线长期挂红掩盖新回归，低成本速清 | 并行 |
| 3 | **P-2** 待提交产物落库 | 未提交 = 协作产物易丢、无法复审基线 | 并行 |
| 后续 | P-5 transfer 上线核验 | 依赖 P-1 + 外部真值 | 串行于 P-1 |
| 后续 | P-6/P-7 vePFS 真机+全链路 | 依赖 researcher 取证 | 串行 |
| 后续 | P-4 低危收尾 | 价值低、非阻塞 | 见缝插针 |
| 独立线 | P-8→P-9→P-10 IaC M1 | 高价值（凭证/权限地基），但卡外部前置 | 独立于 bot 主线 |

**须串行**：P-5←P-1；P-7←P-6；P-9←P-8←（用户真值）；P-10←P-9。
**可并行**：P-1 / P-2 / P-3 三者互不依赖，可同时开。

---

## 六、开放问题（交 dev 转问用户 / 团队定夺）

- **Q1**：P-1 范围——先只修 transfer 单条链路，还是 transfer/cpfs/vepfs/bucket 四条确认 handler 一次改齐？（四条同源问题，一次改齐省重复工；但改动面大、回归面广。建议一次改齐 + tester 四链路覆盖。）
- **Q2**：`infra/pulumi/` 是否纳入本仓库提交？它是独立 IaC epic（自带 EPIC/STORY/README），含 state/AK 红线。建议：脚手架代码入仓（`.gitignore` 已排 state/机密），但作为独立跟踪线，不混进 bot 功能里程碑。请 dev 确认提交策略（同仓子目录 vs 独立仓）。
- **Q3**：`dist_url()` 的「真实签名 token」是既定最终行为吗？若是 → P-3 改测试；若签名方案还会变 → 先冻结该用例避免反复。
- **Q4**：transfer 一期上线时间点 / `wuji_il` 真值谁提供？（P-5 阻塞在此。）
- **Q5**：凭证轮换（Master AK 等，安全事件遗留）要不要纳入本团队跟踪的 backlog，还是纯运维侧另行处理？（代码侧仅 Fernet key 轮换会影响历史 AK 绑定解密，需评估迁移。）

---

## 附：任务板说明

本会话 planner 未启用 `TaskList/TaskCreate` 工具，无法直接读写任务板（#28-32）。
上述 P-1~P-10 为建议粒度，待 dev 认可后由能操作任务板的一方落板 + 派 owner + 标 blockedBy。
