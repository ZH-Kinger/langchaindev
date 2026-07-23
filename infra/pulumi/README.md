# aiops-infra —— 阿里云 RAM/STS 地基（Pulumi-Python）

用 Pulumi 以 **RAM-as-code** 纳管阿里云身份权限地基。**存量资源一律 import-first，绝不重建。**

- **INFRA-1**：纳管 `BotRole-*` 角色 + 信任策略、`feishu-bot-master` 用户、`wuji_*` 用户组。
- **INFRA-1B**：创建权限收归——ECS 实例 + RAM 用户/AK 的创建权限从多数人收回，只留特权组。

> **红线**：IaC 只管"骨架"（角色/组/信任/基线策略）。运行时动态资源（STS 临时凭证、
> `wuji-oss-auto-*` 按人策略、审批开的用户/AK、DSW/DLC/迁移任务）**永不进 state**。
> **PAI 平台暂不纳入**（INFRA-4 顺延待议）。

---

## 目录

```
infra/pulumi/
  Pulumi.yaml            项目定义
  Pulumi.prod.yaml       prod stack 配置（非机密）
  config.py              单一事实源：解析 .env 里 ROLE_MAPPING/UID/组
  __main__.py            入口
  ram/
    groups.py            wuji_* 组 + 特权组
    master.py            feishu-bot-master + 托管策略
    roles.py             BotRole-*（角色本体 + 信任策略）
    create_governance.py wuji-deny-create / wuji-allow-create（INFRA-1B）
  requirements.txt
```

## 前置条件（一次性）

1. **执行凭证**：一把有 **RAM 写权限**的管理员 AK（≠ Bot 的 master AK，master 只有 STS+只读）。
   ```bash
   export ALICLOUD_ACCESS_KEY=...        # 仅本地/CI，用完即清，勿入库
   export ALICLOUD_SECRET_KEY=...
   export ALICLOUD_REGION=cn-hangzhou
   ```
2. **state 后端 = OSS**（state 不落本地、不进 git）：
   ```bash
   pulumi login oss://<your-state-bucket>
   ```
3. **真实值就位**：`Pulumi.prod.yaml` 的 `envFile` 指向项目 `.env`，其中需含真实
   `ALIYUN_BOT_ACCOUNT_UID` / `ALIYUN_BOT_ROLE_MAPPING` / `ALIYUN_BOT_ROLE_DEFAULT`。
4. **特权组名**：设 `aiops-infra:creatorGroup`（现有管理组名）：
   ```bash
   pulumi config set aiops-infra:creatorGroup <现有管理组名>
   ```
5. **活体导出**（repo 无）：把各 `BotRole-*` 的**信任策略**与**已挂权限策略**原文导出比对
   （`aliyun ram GetRole` / `ListPoliciesForRole`），确认 `config.trustPrincipal` 与活体一致。

## 安装

```bash
cd infra/pulumi
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pulumi stack select prod   # 或 pulumi stack init prod
```

## 执行：INFRA-1（import-first，目标 no-op）

逐个把存量资源 import 进 state（资源名见各模块 `resource_name`）：

```bash
# 组（每个 wuji_* / 特权组）
pulumi import alicloud:ram/group:Group group-wuji-algorithm wuji_Algorithm
pulumi import alicloud:ram/group:Group group-wuji-examination wuji_Examination
# Master 用户 + 托管策略挂载
pulumi import alicloud:ram/user:User master-user feishu-bot-master
pulumi import alicloud:ram/userPolicyAttachment:UserPolicyAttachment \
  master-attach-aliyunstsassumeroleaccess feishu-bot-master:AliyunSTSAssumeRoleAccess:System:User
# 角色（每个 BotRole-*）
pulumi import alicloud:ram/role:Role role-botrole-default BotRole-Default
```
> import ID 格式随 provider 版本，以 `pulumi import` 报错提示或 provider 文档为准。

**验收闸门（AC3）**：全部 import 后
```bash
pulumi preview
```
必须显示 **no changes**（0 create/update/delete）。有 diff → 说明声明与活体不符，
修 `config.py`/信任策略/托管策略清单直到干净。**这是"没动线上"的硬证据。**

## 执行：INFRA-1B（创建权限收归，有意变更）

INFRA-1 对齐 no-op 后，`__main__.py` 已包含 `create_governance.build(...)`：

```bash
pulumi preview   # 预期：仅新增 wuji-deny-create / wuji-allow-create + 其挂载，其余 no-op
pulumi up        # 低峰期执行
```

## 验证

- **非特权用户（用户面）**：普通组成员用个人 AK/控制台跑 `ecs RunInstances`、`ram CreateUser` → 均 `AccessDenied`。
- **非特权用户（角色面）**：该用户在飞书让 Bot 建 ECS → 其映射的 `BotRole-*` 挂了 Deny → 失败。
- **特权用户**：`creatorGroup` 成员建 ECS / 建子账号成功。
- **不回归**：① DSW 代建（`dsw_scheduler`）照常（不在收归范围）；② 飞书 RAM 审批开户、
  OSS 权限下发用管理 AK 照常成功——证明管理 AK 未被 Deny 误伤。

## 红线自查（每次 up 前）

`pulumi preview` 中**不得出现**任何 `wuji-oss-auto-*` 策略、审批创建的 RAM 用户/组成员、
DSW/DLC 实例——出现即越界纳管了运行时资源，必须从声明中剔除。

## 三条铁律（防自伤）

1. **特权成员不得同时在被 Deny 的组里**（RAM Deny 永远赢）。
2. **Bot 管理 AK 归属的 RAM 用户必须豁免**（不在被 Deny 的组）——否则审批/OSS 下发全废。
3. **不 import 运行时资源**（Bot 建的用户/实例/按人策略）——否则 Bot 一删就 state 漂移。

## 注意

- **绝不提交** state、AK、从活体导出的机密策略原文（见 `.gitignore`）。
- pulumi-alicloud 资源属性名可能随 provider 版本微调（如 `policy_name`/`policy_document`）；
  若 preview 报未知属性，按已装版本的 provider schema 调整。
