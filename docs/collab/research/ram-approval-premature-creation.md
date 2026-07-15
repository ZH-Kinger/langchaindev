# RAM/IAM 建号绕过第二级审批 — 独立复核 + 爆炸半径

调查人：researcher（只读侦察 + 取证）。日期 2026-07-15。
结论：**dev 初判成立且更严重——所有多级审批实例 100% 在第二级完成前就建了号**（不是偶发竞态）。
已发现 1 个至今仍未获第二级批准的账号（jichuan，火山 IAM），另有 3 个是靠第二级审批人事后几秒~几分钟内补批才侥幸合规。

---

## 1. 触发路径复核（代码级证据链）

### 事件如何进来
- `core/feishu_bot/routes.py:61-80`：`/feishu/event` 对**任何** `event_type` 含 "approval" 或带 `approval_code` 的事件，只要 `should_handle_event` 通过，就 `threading.Thread(target=ram_approval.handle_approval_event)`。
- `core/ram_approval.py:136-143` `should_handle_event`：仅校验「event_type 含 approval 或有 approval_code」+「approval_code == `FEISHU_RAM_APPROVAL_CODE`（或事件里无 code）」。**完全不区分事件是"实例级"还是"任务/节点级"。**

### 飞书审批事件有两类（【文档】已证实，见下 §2）
- `approval_instance`（审批实例状态变更）：`status` = **实例级**状态，仅在整单状态变化时触发，节点逐个通过时**不触发**。
- `approval_task`（审批任务状态变更）：`status` = **节点级**状态（单个审批人同意=APPROVED），**每个节点通过各触发一次，此时整单仍是 PENDING**。

bot 两类都会照单全收 → **节点通过时收到的 `approval_task` 事件（node status=APPROVED）就是误触发源**。

### 门禁为何被节点级 PASS 骗过
1. `handle_approval_event` line 151：`status = (_extract_status(payload) or "").upper()`。
   - `_extract_status`（line 1611）= `_deep_first(payload, ("status","approval_status",...))`。
   - `_deep_first`（line 1625）递归深搜第一个命中键。事件 payload 顶层是 `header`/`event`，无 `status`；递归进 `event` dict 命中 `event.status` = **节点级 APPROVED**。
2. line 152：`APPROVED_STATUSES`（line 38-46）含 `APPROVED/PASS/AGREE/DONE/COMPLETED/...`——**这些恰恰多为节点/timeline 概念**，节点 APPROVED 直接过门。
3. line 174：`detail = fetch_approval_instance(instance_code)`——回拉了完整实例（其顶层 `status` = 真实**实例级** PENDING）。
4. line 175：`status = status or (_extract_status(detail))`——**致命短路**：`status` 已是事件里的节点 APPROVED（truthy），`or` 右边**永不执行**，**从不用回拉到的实例整体状态复核**。
5. line 180-182：`status not in APPROVED_STATUSES` 为假 → 放行 → line 206 `create_accounts_for_platforms(req)` 建号。

**内部自相矛盾铁证**：line 190 `save_approval_record` 存的 `approval_status` 走的是 `_extract_status(detail)`（line 1081，独立从回拉实例算），得到的是**实例级 PENDING**；而门禁用的是事件节点 APPROVED。于是 Redis 记录里出现 `result_status=success` 且 `approval_status=PENDING` 的矛盾体——真机数据里 3 条正是如此（见 §3）。

### 平台无关
门禁在 `handle_approval_event` 通用层，`create_accounts_for_platforms`（line 437）阿里 RAM(`create_ram_account`) / 火山 IAM(`create_volcano_iam_account`) 都在其后。**阿里、火山双中**，真机数据两边都有踩中案例。

---

## 2. 权威"整单通过"信号（修复应锚定的正确判据）【文档】

飞书审批 v4：

| 层级 | 字段 | 枚举 | "全部通过"判据 |
|------|------|------|----------------|
| **实例级**（整单） | GET 实例返回体顶层 `status` | PENDING / **APPROVED** / REJECTED / CANCELED / DELETED（事件里还有 REVERTED/OVERTIME_CLOSE/OVERTIME_RECOVER） | **`status == "APPROVED"`** ← 唯一权威"所有节点都通过" |
| **节点/任务级** | `task_list[].status` | PENDING / APPROVED / REJECTED / TRANSFERRED / DONE | 单个审批人处理结果，**不能代表整单** |

- 事件回调可靠区分：`approval_instance` 事件才是整单状态；`approval_task` 事件是节点级、整单可能仍 PENDING。**只有 `approval_instance` 且 `status==APPROVED` 才应触发建号**；或忽略事件里的 status、拿 `instance_code` 回拉 GET 实例、读**顶层 `status`** 判 `== "APPROVED"`。
- 修复要点：
  1. 删掉 line 175 的 `status or` 短路——**有 instance_code 时必须以回拉实例的顶层 status 为准**（`detail["status"]`，别再 `_deep_first` 深搜以免又抓到 task_list 里的节点 status）。
  2. 收紧判据为 `status == "APPROVED"` 单值（当前 `APPROVED_STATUSES` 混入 `PASS/AGREE/DONE` 等节点/timeline 词，是把节点 PASS 当整单通过的帮凶）。
  3. 可选：`should_handle_event` / 入口按事件类型过滤，只认 `approval_instance` 那类，丢弃 `approval_task`。

出处：
- [Get instance details](https://open.feishu.cn/document/server-docs/approval-v4/instance/get)（实例 status vs task_list status 枚举）
- [Approval overview](https://open.feishu.cn/document/server-docs/approval-v4/approval-overview)（instance/task 枚举定义）
- [Approval instance status change event](https://open.feishu.cn/document/server-docs/approval-v4/event/common-event/approval-instance-event)（instance 事件 status=实例级，节点逐个通过不触发）
- [Approval events overview](https://open.feishu.cn/document/server-docs/approval-v4/event/function-introduction)（确认 approval_task 是独立于 approval_instance 的节点级事件）

---

## 3. 爆炸半径（真机只读扫描 bot-new / 容器 aiops-bot Redis + 回拉实例）【实测】

方法：容器内 `redis scan ram_approval:instance:*` → 对每条 `result_status∈{success,dry_run}` 用 `ram_approval.fetch_approval_instance()`（只读 GET）回拉真实实例状态 + task_list。**遍历显式跳过 password/secret 等键，未打印任何敏感值**（只取 status/node/end_time/时间戳）。

Redis 现存 12 条实例记录：success=8、dry_run=2、failed=2。逐条核实 10 条已建号/演练记录：

### 单节点（单级审批）——建号合规（6 条，含 test/dry_run）
建号时该唯一节点已 APPROVED = 整单已通过，无违规：
- `0FE380D9…` test（火山）、`66D0D6E0…` test、`C8D04B89…` hubohua（阿里）、`9F27FC94…` chenxinyi（阿里）；dry_run：`D7063BC7…`/`7F47693F…` wangzihan。

### 多节点（两级审批）——**全部提前建号（4/4 = 100%）**

| 实例 | 登录名 | 平台 | 建号 created_at | 建号时实例真实态 | 节点情况（建号时 vs 现在） | 判定 |
|------|--------|------|-----------------|------------------|------|------|
| **1C854E42…** | **jichuan** | **火山 IAM** | 1784116417670 | **PENDING（至今仍 PENDING）** | node A 至今 **PENDING(end=0)**；node B APPROVED(end=1784116416146，建号前 1.5s)。timeline 只有 1 个 PASS | **🔴 硬违规：第二级从未批准，账号已存在** |
| 627E4B3A… | qiaodongming | 阿里 RAM | 1783564777729 | PENDING | node A APPROVED(建号前 0.87s)；node B 当时 PENDING，**建号后 ~10s** 才 APPROVED(end=1783564787345) | 🟠 提前建号，第二级 10s 后补批，现已 APPROVED |
| AC90A9AC… | shenglijie | 阿里 RAM | 1783932383742 | PENDING | node A APPROVED(建号前 1.66s)；node B **建号后 ~10.5 分钟**才 APPROVED(end=1783933013356) | 🟠 提前建号，第二级 10 分钟后补批，现已 APPROVED |
| 78BB71CE… | zhangxiaoxiong | 火山 IAM | 1783411321332 | （存储态 None） | node A APPROVED(建号前 1.1s)；node B **建号后 ~25s**才 APPROVED(end=1783411346100) | 🟠 提前建号，第二级 25s 后补批，现已 APPROVED |

- jichuan(`1C854E42`) 即 dev 给的样例，独立复核完全吻合：实例整体 PENDING、node 一 PENDING 一 APPROVED、建号 created_at 在节点通过后 1.5s。**这是唯一一个第二级至今未批的活违规——该火山 IAM 账号应视作未授权、建议核实是否撤号。**
- 另 3 条（qiaodongming/shenglijie/zhangxiaoxiong）**流程同样被绕过**（建号时第二级还没批），只是第二审批人事后几秒~十分钟内补批，结果侥幸合规。**若当时第二级选择拒绝，就是 3 个未授权账号**。
- `627E4B3A`/`AC90A9AC` 的 Redis 记录 `approval_status=PENDING` 但 `result_status=success`——正是 §1 那处"门禁用事件节点态、save 用回拉实例态"矛盾的实锤。

### 严重性结论
- **确定性 bug，非竞态**：4 个两级审批实例，4 个都在第二级完成前建了号（100%）。只要有两级/多级审批，第一个节点一通过就会立刻建号。
- **当前口子仍开着**：代码未修，下一个多级审批仍会被绕过。
- **需处置**：jichuan 火山 IAM 账号（实例仍 PENDING）需人工核实/考虑撤号；其余 3 个虽已事后合规，建议登记备查。单节点 6 条不受影响。

---

## 附：取证脚本
`/tmp/scan_ram.py`（researcher 本地生成，`docker exec -i aiops-bot python` 只读跑），仅 SCAN + GET，跳过敏感键，未做任何写/改。
