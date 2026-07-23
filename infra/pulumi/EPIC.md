# Epic：AIOps 基础设施即代码（IaC）建设

**Epic ID**：INFRA　**Owner**：平台运维　**状态**：进行中（一期开工）
**工具选型**：Pulumi (Python)　**建议 Jira 项目**：`SWD`

---

## Epic 目标

> 把现在靠**手点控制台**维护的阿里云基础设施，逐步收敛为**声明式代码**：可版本化、可 review、可回滚、可审计。最终形成 **PAI 灵骏** 与 **K8s(ACK)** 两套 stack，共用一层 RAM/STS 身份权限地基。

## 为什么做

- 当前 RAM 角色 / STS 信任策略 / 用户组 / 存储桶 / 文件系统全无版本无审计，误改无法回滚——「Master AK 泄露待轮换」隐患的根因层。
- CPFS DataFlow 绑定「手建、危险（CreateDataFlow 清 Fileset）、Bot 只能绕开」——正需声明式管理。
- 两套集群（灵骏 + K8s）需要一致、可复现的地基，不能各自手搭。

## 贯穿全程的核心原则（红线）

- **IaC 管"骨架"，Bot 管"血肉"**：IaC 只纳管长生命周期资源（角色/组/信任/桶/文件系统/DataFlow 绑定/Workspace/集群）；**运行时动态资源永不进 state**——STS 临时凭证、`wuji-oss-auto-*` 按人策略、审批开户、DSW/DLC 实例、迁移/DataFlow 任务。
- **import-first，绝不 big-bang**：存量活体资源一律先 import，`pulumi preview` 必须 no-op 才算对齐；确认后才允许改。
- **state 与机密隔离**：state 放 OSS 后端，不落本地/不进 git；执行用独立管理员 AK（≠ Bot master AK），机密走 secrets provider。
- **单一事实源**：`BotRole-*` 从 Bot 读的同一份 `ALIYUN_BOT_ROLE_MAPPING` 生成，消除"云上角色 vs Bot 期望"漂移。

## 范围

**纳入**：阿里云 RAM/STS、OSS、CPFS/NAS + DataFlow 绑定、PAI Workspace/资源组、ACK 集群+节点池、（监控按需）。
**不纳入**：一切 Bot 运行时动态资源（见红线）；火山引擎暂缓（provider 覆盖弱）；K8s 工作负载（用 Helm/Argo，非 IaC）。

---

## 子 Story 路线图（按风险由低到高）

| ID | Story | 目标 | 关键资源 | 预估 | 依赖 |
|----|-------|------|----------|------|------|
| **INFRA-1** | **RAM/STS 地基** 〔进行中〕 | 角色/组/信任声明式化，两套 stack 共同底座 | `BotRole-*`、`feishu-bot-master`、`wuji_Algorithm/Examination` | 5 | 管理员 AK、OSS state 桶、活体策略导出 |
| **INFRA-1B** | **创建权限收归** 〔进行中〕 | ECS 实例 + RAM 用户/AK 创建权限从多数人收回，只留特权组；显式 Deny 双面(组+角色) | `wuji-deny-create`、`wuji-allow-create`、`CREATOR_GROUP` | 3 | INFRA-1；`creatorGroup` 组名、管理 AK 豁免核实 |
| INFRA-2 | OSS 桶 | 数据桶 + 基线 policy + 生命周期纳管 | `wuji-bucket-hangzhou`、`wuji-ego-processed`、`wuji-test-data`、`wuji-sing` | 3 | INFRA-1 |
| INFRA-3 | CPFS/NAS + DataFlow 绑定 | 文件系统 + **DataFlow 绑定声明式管**（解决手建/绕开痛点，Bot 仅提交 task） | `CPFS_FILE_SYSTEM_IDS`、(fs_id,region,DataFlowId)↔(oss bucket/prefix/fs_path) | 8 | INFRA-2；AK 需 `nas:Describe*` |
| ~~INFRA-4~~ | ~~PAI 灵骏~~ **〔顺延待议〕** | 用户 2026-07-07 明确暂不动 PAI；Workspace/资源组骨架待后续再议 | `PAI_DSW_WORKSPACE_ID`、`PAI_DSW_RESOURCE_ID` | — | 暂缓 |
| INFRA-5 | K8s (ACK) | 集群本体 + 节点池 + 基础插件；工作负载走 Helm/Argo | ACK 集群、GPU 节点池、VPC/vSwitch/SG | 13 | INFRA-1 |
| INFRA-6 | 监控（可选） | 托管 Prometheus / SLS / Grafana 纳管 | `alicloud_arms_prometheus`、SLS project/logstore、Grafana datasource | 5 | 按需插入 |
| INFRA-7 | 火山 TOS/vePFS | 待 volcengine provider 逐资源核实后评估 | `wuji-egocentric-data`、vePFS fs | 待定 | provider 覆盖度评估 |

> **PAI (INFRA-4) 暂缓**——用户明确暂不动 PAI 平台。K8s (INFRA-5) 仍可在 INFRA-1 后独立推进。

## Epic 级验收 / 完成定义

- [ ] 每个子 story 达成各自 AC（核心闸门统一为 import 后 `pulumi preview` no-op + 红线自查无运行时资源）。
- [ ] 全部 IaC 资源可由他人从 README runbook 复现纳管，state 在 OSS，机密不入库。
- [ ] 两套 stack（灵骏 + K8s）成型，共用 RAM/STS 地基，无手点残留。
- [ ] Bot 运行时链路（STS AssumeRole、oss_perm、审批、DSW 调度）全程不回归。

## 关键风险（Epic 级）

- **越界纳管运行时资源** → 与 Bot 抢资源、天天漂移。每期用红线自查强制拦截。
- **信任策略/存量不匹配** → import 后 preview 出 diff。以活体导出原文为准逐个对齐。
- **火山 provider 覆盖弱** → 故意排在最后、单独评估，不阻塞阿里云主线。
- **权限不足** → 管理员 AK 缺写权限（尤其 `nas:*`）；每期前置核实。

## 里程碑

1. **M1 地基就绪**：INFRA-1 + INFRA-1B 完成，共同底座 + 创建权限收归到位。
2. **M2 存储纳管**：INFRA-2 + INFRA-3，数据桶与 CPFS/DataFlow 声明式化。
3. **M3 K8s 成型**：INFRA-5（ACK）可复现。（PAI 灵骏 INFRA-4 顺延，届时再并入。）
4. **M4 收口**：监控（INFRA-6）+ 火山评估（INFRA-7），手点残留清零。
