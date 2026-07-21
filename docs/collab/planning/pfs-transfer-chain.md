# 开发方案：vePFS ↔ CPFS 直接传输（3 段跨云 PFS 直传链）

> planner，2026-07-16。据 researcher `docs/collab/research/vepfs-cpfs-direct-transfer.md` + 用户拍板决定。
> **本文是给 dev 的落地方案；dev 需拿去请用户最终确认（尤其 §8 待定项）后才开工。**
> 硬事实：物理上无 vePFS↔CPFS 直连，必走 3 段（源 PFS→源云对象存储→跨云→目的云对象存储→目的 PFS）。三段引擎积木本仓全有，本功能是**加法式新建"链式上层编排包"**，不改现有引擎。

---

## 1. 范围与分期（用户已拍板）

| | 方向 | 段引擎成熟度 | 决定 |
|---|---|---|---|
| **P1（先做）** | vePFS → CPFS | 三段引擎**全真机验过** | 骨架 → dry-run → 真机跑通一条 |
| **P2（后做）** | CPFS → vePFS | 卡在②跨云 OSS→TOS **未真机验** + DMS 目的只到桶级 | **前置阻塞**：先收口跨云 OSS→TOS，见 §7-EPIC-P2 |

- **两个方向都要**，方向由源/目的地址类型**自动判断**（源 vePFS→P1，源 CPFS→P2）。
- **中转 staging 数据**：可配置、**默认保留**（`PFS_TRANSFER_STAGING_CLEANUP` 默认 false；开开关才在整链 DONE 后清中转副本）。
- **审批**：**整条链入口审批一次**（不每段重复）。
- **架构**：独立新包 `core/pfs_transfer/`（不塞进 `core/transfer/`——"落对象存储" vs "落 PFS" 语义互污，同当初 bucket_transfer 与 transfer 独立的决策）。参照 `core/ssh_transfer/` 的多段链六件套 + 段级"跳过已成功段"retry。

---

## 2. 3 段链状态机 + 段→引擎映射矩阵

### 2.1 状态机

```
NEW → SINKING(源 PFS→源云对象存储) → CROSSING(跨云对象存储→对象存储)
    → PREHEATING(目的云对象存储→目的 PFS) → DONE | FAILED
```

- 链在**后台线程**里顺序驱动三段：每段复用对应现成 orchestrator 的 `run_to_completion(sub_job, on_update=chain_cb, poll_interval, max_polls)`（自包含、阻塞到该段终态、逐阶段回调）。某段失败 → 整链 FAILED，不进下一段。
- **段级续跑**（照 ssh_transfer `stage1_rc` 先例）：链 job 记录每段的**子 job 终态**；retry / 重启续跑时，已 DONE 的段**跳过**，只从第一个未完成段起。语义：沉降成功、跨云失败 → 重试只重跑跨云 + 预热，不重跑沉降。

### 2.2 段→引擎映射矩阵

| 段 | **P1: vePFS→CPFS** | **P2: CPFS→vePFS** |
|---|---|---|
| ① SINKING（源 PFS→源云对象存储） | vePFS→TOS：`core/vepfs_dataflow` **Export** ✅ | CPFS→OSS：`core/cpfs_dataflow` **Export** ✅ |
| ② CROSSING（跨云对象存储→对象存储） | TOS→OSS：`core/transfer.engine_mgw`（进 OSS，阿里在线迁移）✅真机验过 | OSS→TOS：`core/transfer.engine_tos`（进 TOS，火山 DMS）⚠️跨云未真机验 + 目的只到桶级 |
| ③ PREHEATING（目的云对象存储→目的 PFS） | OSS→CPFS：`core/cpfs_dataflow` **Import** ✅ | TOS→vePFS：`core/vepfs_dataflow` **Import** ✅ |

- **① / ③ 复用 orchestrator 层**（不是直调 engine）——CPFS 段必须走 `cpfs_dataflow.orchestrator`（含临建/用完删 DataFlow 逻辑；engine 层 `submit_sink_job` 不临建，`resolve_dataflow` 找不到会抛错）。vePFS 段走 `vepfs_dataflow.orchestrator`（无 DataFlow 绑定，直接 submit）。
- **② CROSSING** 可复用 `transfer.engine_mgw`/`engine_tos` 的 `submit_cross_job`/`poll_job`（对象存储↔对象存储那半段），或复用 `transfer.orchestrator` 的 `start_cross`/轮询。**不复用 transfer 的 Plan/整机**（它不建模"目的 PFS + 预热段"）。

---

## 3. 数据模型 / Plan

`core/pfs_transfer/paths.py`（纯逻辑，参照 ssh_transfer/transfer paths）：

- **输入**：源 PFS 地址 + 目的 PFS 地址（如 `vepfs://<fs>/wzh/data/` → `cpfs://<fs>/team/data/` 或裸挂载路径 `/vepfs/...`→`/cpfs/...`）。目录级、尾斜杠规整；剥挂载前缀（复用 `VEPFS_MOUNT_PREFIX`/`CPFS_MOUNT_PREFIX`，照 #42 两条解析路径都剥）。
- **方向自动判断**：源 `vepfs://`（或 vePFS 挂载路径）→ P1；源 `cpfs://` → P2。混合非法组合（如源目的同云、或含对象存储 scheme）**拒绝并提示**。
- **推导两个 staging 位置**（查 `PFS_STAGING_MAP`，见 §4）：
  - 源云 staging（沉降落点 = 跨云源）：P1=源 vePFS 的 fs → TOS 桶+前缀；P2=源 CPFS 的 fs → OSS 桶+前缀（+ DataFlow 绑定）。
  - 目的云 staging（跨云落点 = 预热源）：P1=目的 CPFS 的 fs → OSS 桶+前缀；P2=目的 vePFS 的 fs → TOS 桶+前缀。
- **中转前缀命名**（§8 待定，建议）：staging prefix = `<配置基前缀>/<chain_id 或镜像源子路径>/`，专桶/专前缀避免多链撞车 + 便于清理定位。
- **产出** `Plan`：`direction(P1/P2)`、`src_pfs{fs_id,region,sub_path}`、`dst_pfs{fs_id,region,sub_path}`、`src_staging{scheme,bucket,prefix,region[,dataflow_id]}`、`dst_staging{...}`、三段各自参数（供三个子 orchestrator 的 `make_plan`/`plan_from_addresses` 消费）。
- **区域硬约束**（在 paths 校验，不满足即拒）：CPFS↔OSS 必须同 region；vePFS↔TOS 必须同 region；跨云段②可跨 region。

---

## 4. 中转 staging 桶配置（新配置，现有 `TRANSFER_BUCKET_MAP` 不适配）

现有 `TRANSFER_BUCKET_MAP`（`<scheme>://<bucket>`→目的桶）是"对象桶→对象桶"，缺"PFS fs → 本云 staging 桶+region+prefix[+dataflow]"维度。**新增** `PFS_STAGING_MAP`（JSON）：

```jsonc
{
  "vepfs://<fs-id>": { "region": "cn-beijing",  "tos_bucket": "wuji-dc-beijing",  "tos_prefix": "pfs-staging/" },
  "cpfs://<fs-id>":  { "region": "cn-hangzhou", "oss_bucket": "wuji-il-hz",       "oss_prefix": "pfs-staging/", "dataflow_id": "df-xxx" }
}
```

- key = `<pfs-scheme>://<fs-id>`；value 带 region（须与 PFS 同区，paths 校验）+ 桶 + 基前缀；CPFS 侧可选 `dataflow_id`（省 `resolve_dataflow`，留空则 orchestrator 临建）。
- **回退发现**：留空的 fs 可复用现成 `cpfs_dataflow/discovery.py`、`vepfs_dataflow/discovery.py` 自动定位 PFS↔对象存储绑定桶（`CPFS_FILE_SYSTEM_IDS`/`VEPFS_FILE_SYSTEM_IDS`）。方案：**先手填 `PFS_STAGING_MAP` 保证 P1 可跑，discovery 自动填作后续增强**。

---

## 5. Redis 命名空间 / 幂等 / 对账 / 清理 / 审批门

- **新命名空间（第 6 条链）**：`pfs:transfer:job:{chain_id}`，30 天 TTL。job 前缀 `xpfs-`（与 `tr-/cpfs-/vepfs-/bkt-/sgp-` 并列）。
- **链 job 结构**：`chain_id`、`direction`、源/目的 PFS、两个 staging、`chain_stage∈{NEW,SINKING,CROSSING,PREHEATING,DONE,FAILED}` + 三个子 job 引用（`sink_job_id`/`cross_job_ref`/`preheat_job_id`）+ 各段完成标记（照 ssh `stage1_rc`：`sink_done`/`cross_done`/`preheat_done`）+ 当前段进度透传（`bytes_done/total`、files）+ `created_by`/`created_ts`/`updated_ts`（每次 `_save` 刷，对账 stale 门依赖）/`finished_ts`/`error`/`launched`。
- **幂等 job_id** = `xpfs-` + hash(direction, 源 PFS, 目的 PFS, 当天)（照现有链"当天"幂等）。`create_job_record` 幂等复用旧记录 + 回填 `created_by`（照 #45）。
- **NX 去重闸门**：确认下发用 `pfs:transfer:launch:{chain_id}` NX 锁（连点/双投递去重）；retry 重置 `chain_stage=NEW` 时 **delete launch 锁 + `dataflow:notified:{chain_id}`**（照 #37 HIGH#1，否则重试成功结果卡永不推）。整链终态结果卡经 `dsw_scheduler._claim_dataflow_notify` NX 闸门去重（在线线程 / 对账 / 按 ID 查询共用，至多推一张，照 #49/#50）。
- **终态通知收敛**：三段各自 orchestrator 的 `on_update` 只喂链 cb 更新进度，**不各自推卡**；只在**整链**终态推一张结果卡（chain cb 统一收敛，走上面 NX 闸门）。
- **对账兜底**：把新链加进 `dsw_scheduler._dataflow_reconcile_specs`，`active={SINKING,CROSSING,PREHEATING}`，暴露 reconcile 契约（`_KEY_PREFIX`/`get_job`/`refresh`/`_save`/`STAGE_DONE`/`STAGE_FAILED`/`result_card`）。
  - **核心设计点（本活最难）**：链 `refresh(chain_id)` 要能判断"卡在第几段"并续推。建议：refresh 读链 job 定位当前段 → 调该段子 orchestrator 的 `refresh(sub_job_id)`（单次 poll 云端并自愈）→ 若当前段刚 DONE 且非末段 → 经 `pfs:transfer:launch` NX 锁在后台线程续跑剩余段的 `run_to_completion`（重启续跑；NX 锁防 refresh 与手动查询窄窗并发双起，照 ssh L6 记账）。已 DONE 段跳过（子 job_id 幂等 + 段完成标记）。
- **审批门（入口一次）**：链创建时按预估源大小判 `needs_approval`，超 `PFS_TRANSFER_APPROVAL_TB`（默认 1TB）→ confirm 卡 red + 仅 `ADMIN_FEISHU_OPEN_ID` 可确认。**PFS 目录暂无现成测大小工具**（`transfer.estimate_source` 只测对象存储）→ 大小估算手段见 §8 待定；未知大小时 **fail-safe 当作需审批**（照 ssh_transfer `size_known=False`→恒需审批）。
- **中转清理开关**：`PFS_TRANSFER_STAGING_CLEANUP` 默认 false（保留，幂等续跑友好、对齐"永不覆盖"保守风格）。开则整链 DONE 后删两个 staging 前缀副本——**本仓无现成"删桶前缀"编排**，需新写 helper（P1 先不做自动删、只占位开关，见 §7-EPIC-X）。

---

## 6. 三入口（照 ssh_transfer/transfer 先例）

1. **飞书向导卡**（`core/pfs_transfer/cards.py` + 接进 `feishu_bot/messages.py`+`actions.py`，由 dev 写）：
   - 新意图 `_is_pfs_transfer_intent`（如"PFS 直传 / vePFS 迁 CPFS / 跨云 PFS 搬运"等词）——**须排在跨云 transfer 与 ssh 意图之前/之间避免误抢**（"迁移"泛词会被 transfer/ssh 入口抢；参照 #51 意图排序）。
   - 向导：录入源 PFS 地址 + 目的 PFS 地址（+ 可选大小预估）→ confirm 卡（估算 + 审批门，2.0）→ progress_card_v2（2.0 纯展示含 `progress_line`，无按钮避 200830）→ result_card（1.0，FAILED 带 `retry_pfs_transfer`）。
2. **Agent 工具** `manage_pfs_transfer`（`plan`/`apply`/`status`，新 TOOL_GROUP `pfs_transfer`，注册进 `tools/__init__.py`）。
3. **CLI** `python -m core.pfs_transfer.cli plan|apply|status`（dry-run 默认，`--force` 过阈值）。

---

## 7. 配置项清单 / _REQUIRED_FIELDS / 部署

### 7.1 新增配置项
| 配置 | 默认 | 说明 |
|---|---|---|
| `PFS_TRANSFER_ENABLED` | false | 功能总开关 |
| `PFS_STAGING_MAP` | `{}` | PFS fs → 本云 staging 桶+region+prefix[+dataflow_id]（§4） |
| `PFS_TRANSFER_APPROVAL_TB` | 1 | 整链入口审批阈值（TB） |
| `PFS_TRANSFER_STAGING_CLEANUP` | false | 整链 DONE 后清中转副本（默认保留） |
| `PFS_TRANSFER_CHAT_ID` | "" | 结果/进度卡回落频道（空回落 `FEISHU_CHAT_ID`） |

### 7.2 复用现有配置（不新增）
`VEPFS_REGION`/`CPFS_REGION`、`VEPFS_MOUNT_PREFIX`/`CPFS_MOUNT_PREFIX`、`TOS_ACCESS_KEY/SECRET`（火山无 STS）、`TRANSFER_OSS_ROLE`（OSS 目的 RAM role）、`MGW_ENDPOINT/REGION/USER_ID`、`CPFS_CONFLICT_POLICY_DEFAULT`/`VEPFS_CONFLICT_POLICY_DEFAULT`、`CPFS_FILE_SYSTEM_IDS`/`VEPFS_FILE_SYSTEM_IDS`（discovery）、`ADMIN_FEISHU_OPEN_ID`。

### 7.3 _REQUIRED_FIELDS
补 `PFS_STAGING_MAP`（impact："PFS↔PFS 直传无法定位中转桶，功能不可用"）。其余复用字段各自已在必填校验。

### 7.4 部署
**纯 Python 编排，复用已装 SDK**（volcengine / tea-openapi / hcs_mgw / tos）——**无新依赖，普通 `deploy.ps1` 即可，不需 `up -d --build` 重建镜像**（对比 ssh_transfer 因 paramiko 需 rebuild）。

---

## 8. 待定项（dev 去和用户敲定，敲定前不开工 P1 真机段）

1. **审批阈值判据 / 源大小估算手段**：PFS 目录无现成测大小工具。选项：(a) 向导卡让用户填预估量（最简，推荐）+ 未知则 fail-safe 需审批；(b) 先跑沉降段、拿源云 staging 桶大小再在跨云前审批（破坏"入口一次"语义，不推荐）；(c) 固定经验阈值。**建议 (a)**，请用户确认 + 定 `PFS_TRANSFER_APPROVAL_TB` 值。
2. **staging 桶具体取值**：`PFS_STAGING_MAP` 每个 fs 的真实 (region, 桶, 基前缀[, dataflow_id])——P1 至少要源 vePFS fs 的 TOS 桶 + 目的 CPFS fs 的 OSS 桶+DataFlow。用户/researcher 提供。
3. **中转前缀命名规则**：`<基前缀>/<chain_id>/` vs 镜像源子路径。定了才能保证多链不撞 + 清理可定位。
4. **失败中转数据（默认保留）下的重试语义**：确认"retry 复用同 staging（沉降/预热同名跳过、幂等）+ 从失败段续跑（已成功段不重跑）"符合预期；以及失败后残留 staging 是否需手动清指引。
5. **P2 前置阻塞**：跨云 OSS→TOS（`engine_tos`）真机端到端收口——谁做（researcher 真机反查 + dev 真机验收）、何时；且 **DMS 目的只到桶级、`dest_prefix` 不可靠**（预热进 vePFS 后目录结构可能非预期）如何处理（专桶专用 / 源 OSS key 预整形 / 迁移后 TOS copy-rename）。**P2 在此收口前不开工。**
6. **中转清理实现范围**：P1 是否先只占位 `PFS_TRANSFER_STAGING_CLEANUP` 开关（默认 false、不实现删逻辑），删桶前缀 helper 留 EPIC-X 后续？（推荐先占位）

---

## 9. Epic → Task（有序，带验收/owner/依赖/工作量/优先级）

> owner 待 dev 与用户对齐后在任务板正式分派；此处为建议。工作量：S≤半天 / M≤2天 / L>2天。

### EPIC P0 — 骨架 + P1 dry-run（vePFS→CPFS）｜优先级 P0

| # | Task | 验收 | owner | 依赖 | 量 |
|---|---|---|---|---|---|
| P0-1 | `paths.py`：解析源/目的 PFS 地址、方向自动判断、剥挂载前缀、推导两 staging、区域硬约束校验、拒非法组合 | 单测：P1 地址→正确 Plan（两 staging + 三段参数）；混合/同云/对象 scheme 被拒；跨 region PFS↔对象存储被拒；挂载前缀剥净（照 #42） | dev | 待定#2#3 | M |
| P0-2 | `settings` 加 §7.1 五配置 + `PFS_STAGING_MAP` 进 `_REQUIRED_FIELDS` | 单测：默认值、JSON 解析、validate 缺失列出 | dev | — | S |
| P0-3 | `orchestrator.py`：链状态机 + Redis `pfs:transfer:job` + 幂等 job_id + `create_job_record`(回填 created_by) + `needs_approval`(fail-safe) + `run_to_completion`(顺序驱三段、段级跳过已成功段) + `refresh`(定位当前段续推) | 单测：job_id 幂等/前缀；三段推进（①DONE→②→③→DONE、任一段 FAILED→不进下一段）；retry 跳过已成功段（照 ssh `stage1_rc`）；needs_approval size_known=False 恒 True；refresh 定位段+续推（mock 三子 orchestrator） | dev | P0-1,P0-2 | L |
| P0-4 | `cards.py`：entry/confirm(2.0 审批门)/progress_card_v2(2.0 含 progress_line 无按钮)/result_card(1.0 带 retry) | 单测：schema 正确、200830 规避（confirm 2.0→progress 2.0 / retry 1.0→1.0）、缺字段不 KeyError | dev | P0-3 | M |
| P0-5 | `cli.py`：plan/apply/status（dry-run 默认） | dry-run 打印 Plan + 两 staging + 三段计划、不触云；`--force` 过阈值 | dev | P0-3 | S |
| P0-6 | Agent 工具 `manage_pfs_transfer` + TOOL_GROUP `pfs_transfer` + 注册 | `TOOL_GROUPS` 导入期校验通过；plan/status 只读、apply 起链 | dev | P0-3 | S |
| P0-7 | 接飞书三入口：`_is_pfs_transfer_intent`(排序不误撞 transfer/ssh) + `_h_submit/_h_confirm/_h_retry_pfs_transfer` + `_JOB_ID_RE` 加 `xpfs-` + `_handle_progress_query`/`_push_terminal_result_card` 加 pfs 链 | 单测：意图命中/不误撞跨云ssh；confirm NX 锁+审批门+created_by 回填+reply_v2 分支；retry 清 launch+notified 保留段完成标记；按 ID 查 refresh+终态经闸门补推 | dev | P0-3,P0-4 | L |
| P0-8 | 接对账：`_dataflow_reconcile_specs` 加 pfs 链（active 三段） | 单测：specs 含 pfs、active=={SINKING,CROSSING,PREHEATING}、reconcile 契约齐（照 #51 T8） | dev | P0-3 | S |
| P0-9 | 全量补测 + 覆盖边界/对抗/重启/幂等/续跑 | 全量 pytest 0 failed；新增覆盖三段推进/段级续跑/审批 fail-safe/意图排序/NX 去重 | tester | P0-1..P0-8 | M |
| P0-10 | 审计过闸（正确性 + 安全：地址白名单防注入、区域约束、STS/静态 AK 两云凭证边界、终态去重、审批门 admin） | 无阻塞项（commit gate 硬规） | auditor | P0-9 | M |

### EPIC P1 — 真机跑通一条 vePFS→CPFS｜优先级 P0（骨架过闸后）

| # | Task | 验收 | owner | 依赖 | 量 |
|---|---|---|---|---|---|
| P1-1 | 回填真实 `PFS_STAGING_MAP`（源 vePFS fs→TOS 桶 / 目的 CPFS fs→OSS 桶+DataFlow）+ 前置核对（vePFS→TOS 服务授权+带宽>0、CPFS OSS 有 DataFlow 绑定、目的 OSS↔CPFS 同 region） | researcher 只读侦察出真实 fs/桶/region/DataFlow + 前置条件清单 | researcher | 待定#2 | M |
| P1-2 | dry-run 一条真实 vePFS→CPFS：CLI plan 打印正确三段计划 + 两 staging | plan 输出与真机资源对得上、区域约束通过 | dev | P0-*,P1-1 | S |
| P1-3 | 真机端到端跑通一条（小数据）：沉降→跨云→预热→数据落对目的 CPFS 目录 | 三段各自终态 DONE、目的 CPFS 目录数据正确、整链结果卡推发起人一张、重启后对账续跑不重复 | dev | P1-2 | L |
| P1-4 | docwriter 同步 CLAUDE.md/README/CHANGELOG（新链 + 配置 + 三入口） | 事实核验一致 | docwriter | P1-3 | S |

### EPIC P2 — CPFS→vePFS（前置阻塞，暂不开工）｜优先级 P1

**阻塞（blockedBy）**：跨云 OSS→TOS（`engine_tos`）真机端到端收口 + DMS 目的只到桶级的落盘处理（见 §8-#5）。

| # | Task | 验收 | owner | 依赖 | 量 |
|---|---|---|---|---|---|
| P2-0 | **前置**：跨云 OSS→TOS 真机验收 + DMS 桶级落盘方案定案 | researcher 真机反查 DMS 跨云 OSS→TOS 参数 + dev 真机验一条 OSS→TOS；落盘前缀处理方案（专桶/预整形/后置 rename）拍板 | researcher+dev | 待定#5 | L |
| P2-1 | paths/orchestrator 支持 P2 方向（源 CPFS→②engine_tos→③vepfs Import） | 单测：P2 方向 Plan + 三段映射；真机跑通一条 CPFS→vePFS，目的 vePFS 目录落盘符合 P2-0 定案 | dev | P2-0,EPIC-P1 | M |
| P2-2 | 补测 + 审计过闸 | 全量绿 + 无阻塞 | tester+auditor | P2-1 | M |

### EPIC X — 中转清理（可选后续）｜优先级 P2

| # | Task | 验收 | owner | 依赖 | 量 |
|---|---|---|---|---|---|
| X-1 | `PFS_TRANSFER_STAGING_CLEANUP=true` 时整链 DONE 后删两 staging 前缀副本（新写"删桶前缀" helper，OSS+TOS 两云） | 单测 + 真机：开开关跑完 staging 副本被清、关开关保留；失败链不清（保幂等续跑） | dev | EPIC-P1 | M |

---

## 10. 复用 vs 新建 速查

- **直接复用（不改）**：`vepfs_dataflow.orchestrator`（沉降/预热）、`cpfs_dataflow.orchestrator`（沉降/预热，含临建 DataFlow）、`transfer.engine_mgw`(TOS→OSS)/`engine_tos`(OSS→TOS)、`dsw_scheduler._claim_dataflow_notify` NX 闸门 + `fmt_size/fmt_ts/fmt_duration`、六件套模式先例 `core/ssh_transfer/`。
- **新建（加法）**：`core/pfs_transfer/`（paths/orchestrator/cards/cli/__init__）+ 配置 `PFS_STAGING_MAP` 等 + 三入口接线（messages/actions/tools）+ 对账 spec。**不碰任何现有引擎。**
