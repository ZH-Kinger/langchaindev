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
| **dev** | 本会话 | #33 P-1 transfer 200830 修复（全绿，待部署真机验） | ✅ | 源码 progress_card_v2 + reply_v2，已编译 |
| **tester** | — | #34 补测 transfer 200830 | ✅ | 改1+增7，本地三文件 50 passed（见日志 07-09 14:35） |
| **auditor** | 3 实例在跑 | #36 全项目扫描（A四链/B bot+调度/C 凭证基础设施） | 🟢 | #35 复审通过 |
| **researcher** | — | — | 🟡 | #30 200830 取证（`research/feishu-card-inplace-replace.md`） |
| **planner** | — | — | 🟡 | roadmap P-1~P-10 / Q1~Q5（`planning/roadmap.md`） |
| **docwriter** | — | — | 🟡 | README/CHANGELOG 功能重排（未提交） |

## 通信协议
- **实时**：`SendMessage` 按名字互发。用户 → dev → 派活给 tester/auditor → 回报 dev → dev 汇总给用户。
- **任务板**：`TaskList`/`TaskCreate`/`TaskUpdate`，字段 owner(dev/tester/auditor)/status/blockedBy —— 谁在干啥的单一事实源。
- **留痕**：重要决策/发现在本文件追加一行，格式 `[YYYY-MM-DD HH:MM] [DEV|TESTER|AUDITOR] 内容`。

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
- [2026-07-09 14:35] [TESTER] #33 完成。改 `tests/unit/test_transfer_confirm_idempotent.py`：
  ① 修 `test_confirm_return_card_sinking_stage`——confirm 默认 reply_v2=True 走 progress_card_v2，改 patch/断言 `progress_card_v2`（原 patch progress_card 已失效）。
  ② 新增 7 条回归：confirm 成功返回 `card.data.schema=="2.0"` 且无 form/button；retry 路（reply_v2=False）返回卡无 `schema` 键（1.0）；progress_card_v2 对 SINKING/CROSSING/未知 stage（参数化3）+ 缺 same_name_policy 均不抛 KeyError；confirm 六个 early-return（缺job_id/任务不存在/stage非法/已launched/超阈值非管理员/NX锁被占）全部只回 toast 不带 card 键。
  容器为已部署旧源码（无 dev 未提交的 progress_card_v2 改动），故本地跑：该文件 17 passed；连同 test_bucket_transfer/test_dataflow_reconcile 三文件 50 passed。未发现新源码问题。
