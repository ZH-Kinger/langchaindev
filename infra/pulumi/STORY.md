# Story：用 Pulumi 纳管 RAM/STS 地基（IaC 一期）

**Epic**：AIOps 基础设施即代码（IaC）建设
**Story ID**：INFRA-1（建议归入 Jira `SWD` 项目）
**类型**：Story　**优先级**：High　**预估**：5 点（约 2–3 人日，含活体核实与 import 对齐）

---

## 用户故事

> **作为**平台运维负责人，
> **我希望**把现有阿里云的 RAM 角色、STS 信任策略、审批用户组用 Pulumi 声明式代码管起来，
> **以便**这层身份权限地基可版本化、可 review、可回滚、可审计，不再靠手点控制台——为后续 PAI 灵骏 / K8s 两套 stack 打好共同底座。

## 背景

当前 `BotRole-*` 角色、`wuji_*` 用户组、STS 信任策略全是控制台手建：无版本、无 review、无审计、误改无法回滚。这正是「Master AK 泄露待轮换」安全隐患的根因层。本 story 是 IaC Epic 的第一步，**只做最脆弱、最该声明式管理的 RAM/STS 地基**，存量资源 import-first，绝不重建。

## 范围

**纳入（IaC 管"骨架"）**
- Master 用户 `feishu-bot-master` + 其托管策略挂载（`AliyunSTSAssumeRoleAccess`、`AliyunRAMReadOnlyAccess`）。
- `BotRole-*` 角色（`BotRole-Default` + 每组一个），从 Bot 读的同一份 `ALIYUN_BOT_ROLE_MAPPING` 生成 → 单一事实源。
- 每个角色的信任策略（principal = master）+ 已挂权限策略。
- 审批白名单用户组 `wuji_Algorithm`、`wuji_Examination`。

**不纳入（Bot 管"血肉"，永不进 IaC）**
- STS 临时凭证（Redis）、按人生成的 `wuji-oss-auto-*` 策略、审批流开的 RAM 用户/AK/组成员。
- 本期不碰：OSS 桶、CPFS/DataFlow、PAI Workspace、ECS/VPC、ACK、火山 TOS/vePFS（后续 story）。

## 前置条件（阻塞项）

1. 一把有 RAM 写权限的**管理员 AK**（≠ Bot 的 master AK），仅本地/CI 用。
2. 一个 OSS state 桶（`pulumi login oss://…`）。
3. 从活体账号导出 `BotRole-*` 的信任策略与已挂权限策略原文。
4. 真实值：`ALIYUN_BOT_ACCOUNT_UID`、`ALIYUN_BOT_ROLE_MAPPING` 实际内容、各 `BotRole-*` 名称。

## 验收标准（Acceptance Criteria）

- [ ] **AC1 工程就绪**：`infra/pulumi/` 独立 venv，`pulumi-alicloud` 安装，`pulumi login oss://…` 成功，state 不落本地/不进 git。
- [ ] **AC2 声明齐全**：`ALIYUN_BOT_ROLE_MAPPING` 里每个角色 + `BotRole-Default` + Master 用户 + 两个 `wuji_*` 组均在代码中声明；角色由映射循环生成（新增组只改映射）。
- [ ] **AC3 import 对齐（核心闸门）**：全部存量资源 import 进 state 后，`pulumi preview` 输出 **no changes**（0 create / 0 update / 0 delete）。这是"没动线上"的硬证据。
- [ ] **AC4 apply 路径可用**：做一个安全可逆改动（如加 description）→ `pulumi up` → 控制台确认 → 改回 → `pulumi up`，确认能真正写 RAM 且不误伤。
- [ ] **AC5 运行时不回归**：飞书发一条触发 STS AssumeRole 的云指令（查 DSW/OSS），Bot 临时凭证链路照常工作。
- [ ] **AC6 红线自查**：`pulumi preview` 中**不出现任何** `wuji-oss-auto-*` 策略或审批创建的用户/组成员——出现即越界，必须剔除。
- [ ] **AC7 交付物**：`infra/pulumi/` 提交（不含 state/AK/机密策略原文），README 含 import runbook + 红线说明。

## 任务拆解（Sub-tasks）

1. 脚手架：`Pulumi.yaml` / `Pulumi.prod.yaml` / `requirements.txt` / `.gitignore` / venv。〔进行中〕
2. `config.py`：解析 `ALIYUN_BOT_ROLE_MAPPING` / `ROLE_DEFAULT` / `ACCOUNT_UID` / 审批组（复用 Bot 的 .env）。
3. `ram/master.py`：Master 用户 + 两个托管策略挂载。
4. `ram/roles.py`：按映射循环声明 `BotRole-*` + 信任策略 + 权限策略挂载。
5. `ram/groups.py`：`wuji_Algorithm` / `wuji_Examination`。
6. `README.md`：前置条件、`pulumi login oss`、逐资源 import 命令、no-op 闸门、红线自查。
7. 活体核实 + import + 反复修声明直到 `preview` 干净（AC3）。

## 依赖与风险

- **依赖**：前置条件 1–4 全部就绪才能跑 import（AC3）。
- **风险 · 信任策略不匹配**：活体信任策略与声明不一致 → import 后 preview 出 diff。缓解：`trustPrincipal` 可配 user/root，以活体导出原文为准。
- **风险 · 越界纳管运行时资源**：误把 `wuji-oss-auto-*` 或审批用户写进声明 → 与 Bot 抢资源、天天漂移。缓解：AC6 红线自查强制拦截。
- **风险 · 权限不足**：管理员 AK 缺 RAM 写权限 → import/apply 失败。缓解：前置条件 1 先核实。

## 完成定义（DoD）

AC1–AC7 全绿；代码合入；README runbook 可被他人复现；后续期路线图（OSS→CPFS→ACK→监控→火山；PAI 顺延）在计划文档中登记。

---

# Story：创建权限收归 —— ECS 实例 + RAM 用户/AK（IaC 一期 B）

**Story ID**：INFRA-1B　**类型**：Story　**优先级**：High　**预估**：3 点　**依赖**：INFRA-1

## 用户故事

> **作为**平台运维负责人，
> **我希望**把 ECS 实例创建、RAM 用户/AK 创建这两类权限从大多数算法组成员手里收回，只保留给现有的少数管理组，
> **以便**杜绝随意建机器、随意开子账号/发 AK 带来的成本失控与安全风险；其余人改走已有的飞书 Bot/工单代建渠道。

## 背景

当前"谁能创建"完全由云端 RAM/STS 决定、无集中管控。本 story 用 INFRA-1 的同一套 Pulumi 项目，以两条自定义策略（显式 Deny + Allow）双面（用户组 + STS 角色）落地收归。**不含 DSW/DLC，不动 PAI 平台**——日常 notebook/训练不受影响。

## 已确认决策

- **收归范围**：ECS 实例（`ecs:RunInstances/CreateInstance`）+ RAM 用户/AK（`ram:CreateUser/CreateAccessKey/…`）。
- **特权载体**：复用现有某个管理组 `CREATOR_GROUP`（组名待提供）。
- **降级渠道**：飞书 Bot/工单代建。
- **执行强度**：显式 Deny（Deny 永远压过 Allow）。
- **PAI**：暂不纳入。

## 关键设计：双执行面

Bot `manage_ecs` 走 AssumeRole 后的**角色**权限，非用户本人权限——故须同时收：① 用户面（组挂 Deny）② 角色面（非特权 `BotRole-*` 挂 Deny，特权角色挂 Allow）。

## 验收标准

- [ ] **AC1**：`wuji-deny-create`（Deny）挂到全部非特权组 + 非特权 `BotRole-*`；`wuji-allow-create`（Allow）只挂 `CREATOR_GROUP` + 特权角色。
- [ ] **AC2 用户面生效**：普通组成员个人 AK/控制台 `ecs RunInstances` 与 `ram CreateUser` 均被拒。
- [ ] **AC3 角色面生效**：普通用户经 Bot 建 ECS 失败（角色被 Deny）。
- [ ] **AC4 特权可用**：`CREATOR_GROUP` 成员建 ECS / 建子账号成功。
- [ ] **AC5 不回归**：DSW 代建照常；飞书 RAM 审批开户、OSS 权限下发（用管理 AK）照常成功——管理 AK 未被误伤。
- [ ] **AC6 preview 干净**：`pulumi preview` 仅显示新增 2 策略 + 挂载，其余 no-op；不出现任何运行时资源。

## 三条铁律（防自伤）

1. 特权成员不得同时在被 Deny 的组里（Deny 会赢）。
2. Bot 管理 AK 归属的 RAM 用户必须豁免（不在被 Deny 的组）。
3. 不 import 运行时资源（Bot 建的用户/实例/`wuji-oss-auto-*`）。

## 前置输入

`CREATOR_GROUP` 真实组名 + 成员清单；特权组→`BotRole-*` 映射；管理 AK 归属身份确认豁免；全部普通组清单。

## DoD

AC1–AC6 全绿；`ram/create_governance.py` 合入；收归前基线（谁受影响）审计留档；EPIC 路线图登记 INFRA-1B 并标 PAI 顺延。
