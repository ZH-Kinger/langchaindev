# 审批制临时 AK/SK 发放（外部/卖数据）— 研究报告

面向 planner。需求：飞书审批申请一组 OSS 访问凭证给**外部**，指定**生效时间 + 到期时间** + **自定义权限**（某桶某前缀的 GetBucketInfo/Stat/Acl + ListObjects(带 `oss:Prefix`) + PutObject/Abort/ListParts 等），**审批通过才创建**。用户原话要"经 ram+sts 得到含 AccessKeyId/AccessKeySecret/SecurityToken 的临时凭证"。

标注：【文档】=官方明说附 URL；【实测】=真机反查；【推测】=未证实。

---

## 0. 一句话结论

**用户原话（STS 临时凭证）与需求（生效+到期跨天/跨周）存在硬矛盾**：STS AssumeRole 单张凭证**最长 12 小时**，且外部方无法自助续签。
→ 推荐 **方案 B（RAM 子用户 + 长期 AK/SK + policy 内嵌时间区间条件 + 到期定时硬清理）**为主。它把"生效时间 + 到期时间"直接写进权限策略的 `DateGreaterThan`/`DateLessThan` 条件里，**每次 API 调用都在服务端按当前时间判定**——即使凭证泄漏，过期后一切调用自动被拒（这正是"泄漏也自动失效"的关键）。若窗口 ≤12h 且外部能消费 SecurityToken，可用纯 STS 作轻量变体（更贴合用户原话）。

---

## 1. 有效期矛盾（核心）

### STS AssumeRole 临时凭证最大时长【文档】
- `DurationSeconds`：最小 **900s**，默认 **3600s**，最大 = 角色的 `MaxSessionDuration`。
- 角色 `MaxSessionDuration` 可设范围 **3600–43200s（1–12 小时）**，经控制台或 `CreateRole`/`UpdateRole`(`NewMaxSessionDuration`) 设置。
- 即：**单张 STS 凭证绝对上限 = 12 小时**。默认只有 1 小时。
- 出处：
  - [AssumeRole API（DurationSeconds/Policy 参数）](https://www.alibabacloud.com/help/en/ram/developer-reference/api-sts-2015-04-01-assumerole)
  - [设置 RAM 角色最大会话时间](https://www.alibabacloud.com/help/en/ram/user-guide/specify-the-maximum-session-duration-for-a-ram-role)

**本项目现状**：`config/settings.py:108` `ALIYUN_STS_DURATION_SECONDS` 默认 3600；`utils/aliyun_sts.py:129` 用它作 `duration_seconds`。要拉到 12h 得先把目标角色 MaxSessionDuration 调到 43200。

### 若"生效~到期"跨天/跨周 → 单张 STS 不够，三条路子

| | 方案 A：自定义 policy 的 role + 反复 AssumeRole | 方案 B：子用户 + 长期 AK + policy 时间条件 | 方案 C：混合（role + 时间条件 policy） |
|---|---|---|---|
| **满足"生效+到期"** | 部分：到期靠停止续签+删 role；生效靠定时首发。**每张 ≤12h** | ✅ 原生：`DateGreaterThan`(生效) + `DateLessThan`(到期) 同一 statement（AND） | ✅ 时间条件写进 role 的 policy，AssumeRole 出的会话也受约束 |
| **外部方可自助续签?** | ❌ 外部无阿里云身份，无法自己调 sts:AssumeRole；须 bot 每 ≤12h 主动推新凭证给外部（不可"发完即走"） | ✅ 一次发放，跨天有效、无需续签 | ❌ 同 A：外部拿不到续签手段 |
| **泄漏也自动失效** | ✅ STS token ≤12h 天然自灭（最强） | ✅ 过期后 policy 每次调用判 `acs:CurrentTime > 到期` → 拒（服务端强制，与凭证类型无关） | ✅ 两层：STS 12h + 时间条件 |
| **凭证类型（贴合用户原话）** | ✅ 含 SecurityToken 的临时凭证 | ❌ 长期 AK/SK，**无 SecurityToken** | ✅ 含 SecurityToken |
| **复杂度** | 高（要给外部搭续签通道 / bot 定时重发） | **低**（发一次 + 到期清理定时任务） | 中高（role + policy + STS + 续签通道） |
| **复用点** | STS `_do_assume_role`+新 Policy 参数 | `build_policy`(加时间条件) + `create_ram_account` 建号建 AK | 上述两者叠加 |

### 结论（哪条最匹配）
- **需求核心是"发给外部、指定生效+到期、卖数据、泄漏也失效、发完即走"** → **方案 B 最匹配**：
  - 时间区间原生进 policy，服务端每次判定；
  - 一次交付，外部无需（也无法）续签；
  - 过期即使 AK 未删也全部拒绝调用 = 泄漏自动失效；
  - 生效时间在未来时，AK 可立即创建但 policy 判 `DateGreaterThan(生效)` → 未到点前一切调用被拒（凭证"存在但惰性"）。
- **代价**：长期 AK 无自动轮换、风险高于 STS token；靠 policy 时间条件 + 到期硬删补偿。**不含 SecurityToken**，与用户字面表述不符——需向用户说明矛盾并确认取舍。
- **轻量变体（可并存）**：若某次申请窗口 ≤12h 且外部能直接使用 `AccessKeyId/Secret/SecurityToken`（如对方脚本走 STS），则用**纯 STS 单发**（方案 A 单发版），完全贴合用户原话、且天然自灭，最安全。建议按"窗口时长"分流：≤12h→STS 单发；跨天→方案 B。

---

## 2. 生效时间（未来生效）怎么实现

**policy 能表达"生效区间"**【文档】：同一 statement 内 `DateGreaterThan`(生效) + `DateLessThan`(到期)，多条件是 **AND** 关系，即"在生效之后、到期之前"才 Allow。官方原例（[在指定时间段访问](https://help.aliyun.com/zh/ram/access-alibaba-cloud-in-a-specified-period-of-time)、[Condition 运算符](https://help.aliyun.com/zh/ram/conditional-operator)）：

```json
{
  "Version": "1",
  "Statement": [{
    "Effect": "Allow",
    "Action": "oss:PutObject",
    "Resource": "acs:oss:*:*:<bucket>/<prefix>*",
    "Condition": {
      "DateGreaterThan": {"acs:CurrentTime": "2026-08-01T00:00:00+08:00"},
      "DateLessThan":    {"acs:CurrentTime": "2026-08-08T00:00:00+08:00"}
    }
  }]
}
```
- 时间值须 ISO 8601 带时区（`+08:00`=北京）。
- **AND 语义已官方确认**：一条 statement 里多个 Condition 同时满足才生效。

**两种实现路线**：
1. **创建即生效、靠 policy `DateGreaterThan(NotBefore)`（推荐）**：审批通过后立即建号建 AK，policy 里 `DateGreaterThan=生效时间`。生效前 AK 存在但所有调用被拒；到点自动放行；`DateLessThan=到期`到点自动拒。**无需未来定时创建**，只需一个到期清理定时任务。
2. **定时到点才创建**：审批过但生效在未来 → 挂定时任务到生效时刻再建号。多一套调度、多一个"待创建"状态机，且外部拿不到凭证直到生效。**不推荐**（除非业务要求"生效前凭证根本不存在"）。

→ 推荐路线 1：policy 时间区间同时管生效与到期，代码面只需"审批建号"+"到期清理"两段。

---

## 3. 可复用积木（file:line）

### `core/ram_approval.py` —— 本功能是它的直接变体，审批流+建号几乎全可复用
- 审批事件入口 `handle_approval_event` (146)、`should_handle_event` (136)。
- **两级审批门（刚硬化过 #55）** `handle_approval_event` (187-195)：只认回拉实例详情顶层 `status=="APPROVED"` 才建号——本功能同样"审批通过才创建"，直接复用此门。
- 拉实例详情 `fetch_approval_instance` (233)；读表单字段 `parse_ram_account_request`(249)/`extract_form_values`(289)/`_field_text`/`_field_bool`——用来读"生效时间/到期时间/桶/前缀/权限项"新字段（`FIELD_CONFIG` 48-65 加映射即可）。
- 幂等 NX 锁 `_claim_instance` (1249)；Redis 记录（**不存密钥**，1082 注释）`save_approval_record`(1076)/`save_approval_result`(1122)/`_save_instance_record`(1227)——到期清理定时任务将扫这些记录。
- 建号主体 `create_ram_account` (335)：建 RAM user → profile → login profile → **AK**(`create_access_key` 421) → 加组。本功能的建号 = 建 user + 附**时间条件自定义 policy** + 建 AK（去掉控制台登录，外部不需要）。
- **凭证下发（一次性展示，符合用户"外部通道"诉求）**：`notify_result`(867)/`_account_delivery_text`(958) 已含"AccessKey Secret 只展示一次"提示(1000-1004)。`_send_approval_comment`(1042)=审批评论下发；`_send_account_email`(931)=SMTP 邮件下发（**外部方不在飞书 → 邮件是天然通道**）。
- 判断：本功能建议**新建 `core/temp_ak_issuance/` 独立模块**（如 bucket_transfer 相对 transfer 的"加法式复用"），复用 ram_approval 的审批事件解析/门禁/建号/下发函数，但独立 Redis 命名空间（如 `temp_ak:grant:{id}`）+ 独立审批模板 code，避免与 RAM 建号审批混流。

### `core/oss_perm/permsync.py` —— OSS 最小权限 policy 生成，参考 policy 正是这套
- `build_policy` (201)：已生成对象级 ARN（读/写前缀）+ **前缀限定的 ListObjects（带 `oss:Prefix` 条件，229）**——与用户参考 policy 逐条对应。
- 动作集 (53-62)：`WRITE_OBJECT_ACTIONS` 已含 PutObject/AbortMultipartUpload/ListParts/GetObjectAcl/PutObjectAcl；`LIST_BUCKET_ACTIONS` 含 GetBucketInfo/GetBucketStat/GetBucketLocation。**用户还要 GetObject-Acl/GetBucketAcl**——GetObjectAcl 已在；`oss:GetBucketAcl` 需按需补一个动作。
- `_obj_arn` (197) 生成 `acs:oss:*:*:<bucket>/<prefix>*`。
- **改造点**：`build_policy` 现无 Condition 时间条件 → 写一个变体 `build_policy_with_window(resolved, not_before, expire)`，在每条 statement（或整体）注入 `DateGreaterThan/DateLessThan`。文档 6144 字符上限（`RAM_DOC_LIMIT` 66）单桶单前缀绰绰有余。
- `make_ram_client` (323)、`apply_user` (421，建/版本化 custom policy + 幂等附加)——建号后附 policy 直接复用。

### `utils/aliyun_sts.py` —— STS 调用（用于 ≤12h 轻量变体 / 方案 A）
- `_do_assume_role` (109)：现固定角色映射 + `settings.ALIYUN_STS_DURATION_SECONDS`，**未传 Policy 参数、未按需定制 duration**。
- 变体需求：加一个"按指定 role_arn + 内联 Policy + 自定 duration 换 STS"的函数（见 §6 AssumeRole Policy 参数）。当前 `assume_role_for_user` (156) 是"按 open_id 查角色"的用户态路径，不直接适用外部发放。

### `utils/aliyun_client_factory.py`
- `_resolve_cred` (220)、`_make_ims_client` 范式、`get_nas_openapi_client`(185) 的通用 OpenAPI 调用范式——建号用 `permsync.make_ram_client`（RAM 可写 AK）即可，工厂主要作参考。

### `core/dsw_scheduler.py` —— 到期清理定时任务挂这
- 仿 `_morning_loop` (752) / `_oss_perm_loop` (817) / `_capacity_loop` (770) 的"北京时间每日对齐"循环；在 `start()` (677-724) 里按新 settings 开关（如 `TEMP_AK_CLEANUP_ENABLED`）起一条 `temp-ak-cleanup` daemon 线程。
- 清理逻辑：扫 Redis `temp_ak:grant:*`，对 `到期时间 < now` 且未清理的记录执行 §4 删除流程。
- 幂等/重启自愈参考现有 `_dataflow_reconcile_loop` (806) 对账范式。

---

## 4. 到期清理（照 jichuan 删号经验）

RAM `DeleteUser` 有依赖前置：用户存在 AK / login profile / 挂载策略 / 属组 / MFA 时删不掉（报 `DeleteConflict.User.*`）。删除顺序【文档，标准 RAM 依赖】：
1. **先失效再删（防窗口）**：`UpdateAccessKey`(Status=Inactive) 立即停用 AK（比删更快、可留痕）。
2. `DeleteAccessKey` 删所有 AK。
3. `DeleteLoginProfile`（若开了控制台，外部场景建议压根不开）。
4. `DetachPolicyFromUser` 解绑自定义 policy + `DeletePolicy`（含清所有版本，参考 permsync `apply_user` 的 rotate 逻辑）删掉这张一次性 policy。
5. `RemoveUserFromGroup` 移出所有组（外部场景建议不入组）。
6. `DeleteUser`。

**纵深防御**：policy 的 `DateLessThan` 到期后已服务端拒绝一切调用，硬删只是清理残留 artifact（避免子用户/AK 堆积）。即使定时任务某次失败，泄漏凭证也已因时间条件失效。

**定时任务**：扫 Redis 到期记录 → 执行上面顺序 → 回写记录 `status=revoked`。建议 grant 记录带 `not_before/expire/policy_name/user_name/ak_id/revoked` 字段，30 天 TTL 后由 Redis 自然清（对齐既有 `*:job:*` 30 天 TTL 惯例）。

---

## 5. 安全 / 合规

- **最小权限**：`build_policy` 已按 桶+前缀+具体动作 收敛；外部场景**建议只给 AK（不开控制台登录）、可加 `acs:SourceIp` 条件锁定收方出口 IP**（进一步防泄漏，同 statement AND）。
- **到期强制失效**：policy `DateLessThan` = 即使凭证泄漏/未及时删，过期后每次调用服务端判 `acs:CurrentTime` 直接拒——**这是外部/卖数据场景最关键的控制**。STS 变体则叠加 ≤12h 天然自灭。
- **审计留痕**：ram_approval 已把脱敏快照存 Redis（`access_key_secret_saved=False` 1147，从不存 secret）；阿里云侧 ActionTrail 记录该 AK 的全部 OSS 调用，可追溯外部用了什么。grant 记录建议加"谁申请/谁审批/生效/到期/收方"。
- **下发通道**：外部方不在飞书 → 用 `_send_account_email`(931) 邮件下发，或一次性安全链接；`_account_delivery_text` 的"只展示一次"提示照用。**secret 绝不入日志/Redis**（现有约定已满足）。
- **建号权限**：建 RAM 用户/AK/policy 需 RAM 写权限（`permsync.make_ram_client` 用 `ALIBABA_CLOUD_*` 或 `settings.ALIYUN_ACCESS_KEY_*`）。注意本项目 Master AK 只有 STS+RAMReadOnly——**方案 B 建号需要一把有 RAM 写权限的 AK**，planner 需确认凭证来源（可能复用 oss_perm 那把 RAM 可写 AK）。

---

## 6. 阿里云现成的"限时/带过期临时授权"更省机制

- **AssumeRole `Policy` 参数（session policy）**【文档】：AssumeRole 时传内联 Policy，会话权限 = **role policy ∩ 该 Policy 的交集**；`Policy` 长度 1–2048 字符。→ 可让 master AK 扮演一个"宽 OSS 角色"，AssumeRole 时用 `build_policy` 生成的窄 policy 现场收敛到具体桶/前缀。适用于 ≤12h 的 STS 变体，免为每次申请建专用 role。出处同 [AssumeRole API](https://www.alibabacloud.com/help/en/ram/developer-reference/api-sts-2015-04-01-assumerole)。
- **STS 是唯一"原生自动过期凭证"**，但上限 12h；**长期 AK 无原生 TTL** → policy 时间条件是官方推荐的"给长期凭证加时间闸"的惯用替代（§2 官方用例即为此场景）。
- **【推测/侧路】OSS 预签名 URL（presigned URL）**：若外部只是"下载/上传特定对象"而非需要一组 AK，可用 OSS 预签名 URL 完全不发凭证。但用户明确要 ram+sts 的 AK/SK/token，故仅作备选提示，未深挖其过期上限（用 STS 签的 URL 受 token ≤12h 约束）。

---

## 7. 给 planner 的落地建议（拆任务提示）

1. **确认取舍**（用户）：跨天窗口只能用长期 AK（无 SecurityToken）；接受则走方案 B；或按窗口时长分流（≤12h→STS 单发含 token，跨天→B）。
2. **新模块** `core/temp_ak_issuance/`：审批解析/门禁/建号/下发复用 ram_approval，policy 生成复用 permsync `build_policy` + 新增时间区间条件变体，独立 Redis 命名空间。
3. **飞书审批模板**新字段：生效时间、到期时间、桶、读前缀、写前缀、收方邮箱。
4. **到期清理**：dsw_scheduler 加 `temp-ak-cleanup` 每日循环 + 开关。
5. **建号 RAM 写 AK 来源**确认。
6. **前置**：若走 STS 12h 变体，目标 role 的 MaxSessionDuration 调 43200。

---

## 附：待真机取证项（researcher 后续可做，只读/dry-run）
- `oss:GetBucketAcl` 精确动作名与 `build_policy` 现有动作集的差集（补齐用户要的 Acl 读）。【推测→需核】
- 用**不存在的桶/构造性 policy** 试 `CreatePolicy` 校验时间条件 JSON 是否被接受（`acs:CurrentTime` + ISO8601 语法真机校验，不建真号）。
- 本项目那把"RAM 可写 AK"是否具备 `CreateUser/CreateAccessKey/CreatePolicy/AttachPolicyToUser/DeleteUser` 全套动作（读 oss_perm 运行现状或 dry-run 触发权限报错反推）。
