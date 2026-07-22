# 审批制临时 AK/SK 发放（外部 / 卖数据）— 开发落地方案

面向 dev（大概率新开会话开发，故本文自包含）。据 researcher `docs/collab/research/temp-ak-issuance.md` + 用户已拍板。

## 0. 需求一句话
用户经飞书**审批**，为**外部方**（传数据/卖数据）申请一组 OSS 访问凭证：指定**生效时间 + 到期时间** + **自定义权限**（某桶某前缀：GetBucketInfo/Stat/**Acl** + ListObjects(带 `oss:Prefix`) + PutObject/AbortMultipartUpload/ListParts）。**审批通过才创建。**

## 1. 用户已拍板：按有效期窗口分流
**分流边界（关键，dev 照此实现）**：`到期时刻 − 签发时刻(now) ≤ 12h` → 走 STS；否则 → 方案 B。
> 用 `expire − now`（而非 `expire − not_before`）判定：一张 STS token 从签发起最多活 12h，只能覆盖 `[now, now+12h]`。若到期在 12h 内，未来生效由 session policy 的 `DateGreaterThan` 惰性化；若到期超 12h（跨天/跨周），单张 STS 覆盖不到 → 必须方案 B。此定义同时正确处理"3 天后生效、只用 2h"这类（expire 远大于 now+12h → 方案 B）。

- **STS 单发（≤12h）**：`sts:AssumeRole` 一次性签发含 `AccessKeyId/AccessKeySecret/SecurityToken` 的临时凭证，到点自灭（最安全，贴合用户原话）。用 **AssumeRole 的 `Policy` 参数（session policy）** 现场把会话收窄到具体桶/前缀/动作（会话权限 = role policy ∩ session policy）。未来生效则在 session policy 内也写 `DateGreaterThan`。
- **方案 B（跨天/跨周）**：建 RAM 子用户 + 长期 AK/SK + **policy 内嵌时间区间**（`DateGreaterThan`(生效) + `DateLessThan`(到期)，`acs:CurrentTime` ISO8601 `+08:00`，同一 statement AND；服务端每次调用判时间 → 泄漏也自灭）+ **到期定时硬删**。生效在未来用 `DateGreaterThan` 惰性化：审批过即建号建 AK，生效前所有调用被 policy 拒，到点自动放行，**无需未来定时创建**。

## 2. 架构：新包 `core/temp_ak_issuance/`
加法式独立包（照 bucket_transfer 相对 transfer 的做法），**零改** ram_approval / oss_perm 现有引擎，只调用其函数。独立 Redis 命名空间 `temp_ak:grant:{grant_id}`、独立审批入口。建议文件：
- `policy.py`：policy 生成（复用 permsync `build_policy` + 时间条件变体 + GetBucketAcl）。纯逻辑。
- `issuer.py`：分流 + 凭证生成（STS 变体 / 方案 B 建号建 AK）。
- `orchestrator.py`：grant 状态机 + Redis 记录 + 幂等 + 下发编排。
- `cleanup.py`：方案 B 到期硬删流程。
- `cards.py`：飞书卡片（提交/确认/结果）。
- `cli.py`：`python -m core.temp_ak_issuance.cli`（dry-run 默认）。
- Agent 工具入 `tools/`（见 §11）。

## 3. 复用积木（file:line，全部只调用不改）
- **审批流** `core/ram_approval.py`：
  - `handle_approval_event`(146) + 硬化过的**两级审批门**(174-195)：只认回拉实例详情**顶层实例级 `status=="APPROVED"`** 才动作，无 `instance_code` → fail-safe 不建（#55 教训）。本功能同样"审批通过才创建" → 复用同一门禁语义。
  - `should_handle_event`(136)、`fetch_approval_instance`(174)、`_claim_instance`(幂等 NX 锁)、`extract_form_values`/`_field_text`/`_field_bool`、`FIELD_CONFIG`(48-65) 加映射即读新字段。
- **policy 生成** `core/oss_perm/permsync.py`：
  - `build_policy`(201)：已出对象级 ARN（读/写前缀）+ 前缀限定 ListObjects(带 `oss:Prefix` 条件，229）——与用户参考 policy 逐条对应。
  - 动作集(53-62)：`WRITE_OBJECT_ACTIONS` 已含 PutObject/AbortMultipartUpload/ListParts/GetObjectAcl/PutObjectAcl；`READ_OBJECT_ACTIONS` 含 GetObject/GetObjectAcl；`LIST_BUCKET_ACTIONS` 含 GetBucketInfo/Stat/Location。**改造点见 §6。**
  - `_obj_arn`(197) 生成 `acs:oss:*:*:<bucket>/<prefix>*`；`make_ram_client`(323) = RAM 可写 AK 客户端（见 §7）；`RAM_DOC_LIMIT`(65)=6144，单桶单前缀绰绰有余。
- **建号主体** `core/ram_approval.py::create_ram_account`(335)：建 user → profile →（本功能**外部方不给控制台，跳过 login profile**）→ **AK**(`create_access_key` 421)。方案 B 建号 = 建 user + 附时间条件自定义 policy + 建 AK（`console_access=False`、不入组）。**注意**：现 create_ram_account 走 groups 分支，本功能应走 `AttachPolicyToUser`（照 permsync `apply_user`）附一次性 policy，不加组。
- **STS 变体** `utils/aliyun_sts.py`：`_do_assume_role`(109) 现固定角色映射 + `settings.ALIYUN_STS_DURATION_SECONDS`、**未传 Policy 参数、未自定 duration**。需**新增**一个函数：`assume_role_with_policy(role_arn, session_policy_dict, duration_seconds, session_name)` → `AssumeRoleRequest` 传 `policy=json.dumps(...)` + `duration_seconds`（1–2048 字符上限、900–43200s）。`assume_role_for_user`(156) 是用户态路径，**不复用**（外部无 open_id）。
- **凭证下发** `core/ram_approval.py`：`_send_account_email`(931，SMTP)、`_account_delivery_text`(958，含"只展示一次"提示 1000-1004)、`notify_result`/`_send_approval_comment`（审批评论下发）。外部方不在飞书 → **邮件是天然通道**。
- **到期清理挂载** `core/dsw_scheduler.py`：仿 `_oss_perm_loop`(817) 的"北京时间每日对齐"循环 + `start()`(677-724) 里按开关起 daemon 线程；对账幂等参考 `_dataflow_reconcile_specs`(833) 范式。

## 4. 审批流（决策 + 表单字段）
- **独立审批模板 code**（不与 RAM 建号审批 `FEISHU_RAM_APPROVAL_CODE` 混流）：新 `TEMP_AK_APPROVAL_CODE`。`routes.py` 审批事件分发按 code 路由到本模块 `handle_temp_ak_event`（复用 ram_approval 门禁逻辑，独立记录/命名空间）。
  - **待定 A**：复用同一 approval_code 还是新建？→ 建议新建独立模板（隔离、字段不同、避免与建号混）。交 dev 与用户敲定。
- **表单字段**（新审批模板要收）：① OSS 桶（展示名，映射真实 bucket/region）② 前缀/子目录（读集 / 写集，可多）③ 读动作 / 写动作勾选（默认给用户参考的那套：读=GetBucketInfo/Stat/Acl+ListObjects+GetObject/Acl；写=PutObject/Abort/ListParts）④ **生效时间** ⑤ **到期时间** ⑥ **外部接收方邮箱**（下发目标）⑦ 申请原因/用途。
- **门禁**：只认实例级 `status=="APPROVED"` 才创建（复用 174-195）；审批通过后才触发 §5 凭证生成。

## 5. 凭证生成（分流实现）
- **入口**：审批通过 → `issuer.issue(grant)`：按 §1 边界选 STS 或方案 B。
- **STS 分支**：`policy.build_session_policy(resolved, not_before, expire)`（≤2048 字符）→ `aliyun_sts.assume_role_with_policy(TEMP_AK_OSS_ROLE_ARN, policy, duration=min(expire-now,43200), session_name)`。返回 AK/SK/**Token** + expire。**无需清理**（token 到点自灭）。
  - 前置：需一个"宽 OSS 角色" `TEMP_AK_OSS_ROLE_ARN`，其信任策略允许 Master AK 扮演，`MaxSessionDuration=43200`（真机配置）。
- **方案 B 分支**：`create_ram_account`(去 console/去组) 建 user + AK → `policy.build_policy_with_window(...)` 生成带时间条件 policy → `AttachPolicyToUser`（照 permsync `apply_user`）。落 grant 记录（`not_before/expire/policy_name/user_name/ak_id`）供到期清理。

## 6. Policy 改造点（在 `core/temp_ak_issuance/policy.py`，不改 permsync 源）
- **补动作**：用户要 `oss:GetBucketAcl`（读桶 ACL）——现 `LIST_BUCKET_ACTIONS` 无。新增桶级读动作集含 `GetBucketAcl`（**待真机取证**精确动作名，见 §13）。GetObjectAcl 已在 READ_OBJECT_ACTIONS。
- **时间条件变体** `build_policy_with_window(resolved, not_before, expire)`：在 `build_policy` 产出的每条 statement（或整体外层）注入 `Condition: {DateGreaterThan:{acs:CurrentTime:<not_before ISO8601+08:00>}, DateLessThan:{acs:CurrentTime:<expire ISO8601+08:00>}}`。与既有 `StringLike:{oss:Prefix}` 条件合并（同 statement 多条件 AND）。
- **session policy 变体** `build_session_policy(...)`：同上但产物 ≤2048 字符（STS Policy 参数上限）；单桶单前缀足够。
- **可选安全增强**：加 `acs:SourceIp` 条件锁外部方出口 IP（同 statement AND）——列为可选，交用户定。

## 7. 建号所需 RAM 可写 AK（researcher 风险②）
- Master AK（`ALIYUN_BOT_MASTER_AK_*`）只有 STS+RAMReadOnly → **建不了** user/AK/policy。
- **复用 oss_perm 那把 RAM 可写 AK**：`permsync.make_ram_client`(323) 用 `ALIBABA_CLOUD_ACCESS_KEY_ID/_SECRET` 或 `settings.ALIYUN_ACCESS_KEY_ID/_SECRET`。方案 B 建号/附 policy/删号统一用它。
- **待真机取证**：确认这把 AK 具备 `CreateUser/CreateAccessKey/CreatePolicy/AttachPolicyToUser/DeleteUser/DetachPolicyFromUser/DeletePolicy/UpdateAccessKey` 全套（见 §13）。STS 分支不需要这把（走 Master AK AssumeRole）。

## 8. 凭证下发外部
- 外部方不在飞书 → **SMTP 邮件**（复用 `_send_account_email` 931 / `_account_delivery_text` 958）。收方邮箱来自审批表单。
- **一次性展示**：照搬"AccessKey Secret 只展示一次"提示。审批评论 `_send_approval_comment` 作内部留痕（发给申请人/审批群，**不含 secret** 或仅一次性）。
- **脱敏 Redis**：照 ram_approval 从不存 secret（`access_key_secret_saved=False`）。grant 记录只存 `ak_id`（方案 B）/无 secret。
- **待定 B**：STS 分支的 Token 也走邮件？外部脚本能否消费 SecurityToken（否则方案 B）？交 dev 问用户/接收方。

## 9. 到期清理（方案 B）
- **STS 无需清理**（自灭）。
- **方案 B 到期硬删**（照 jichuan 删号顺序，有依赖前置）：
  1. `UpdateAccessKey(Status=Inactive)` 先停用（比删快、留痕、防窗口）；
  2. `DeleteAccessKey` 删所有 AK；
  3. `DetachPolicyFromUser` 解绑 + `DeletePolicy`（清所有版本，照 permsync rotate）；
  4.（本功能不开控制台/不入组，若历史有则 `DeleteLoginProfile`/`RemoveUserFromGroup`）；
  5. `DeleteUser`。
- **挂 `dsw_scheduler`**：仿 `_oss_perm_loop`(817) 每日北京循环 `_temp_ak_cleanup_loop`，开关 `TEMP_AK_CLEANUP_ENABLED`。扫 Redis `temp_ak:grant:*`，对 `expire < now 且 status != revoked` 执行上面顺序 → 回写 `status=revoked`。
- **纵深防御**：policy `DateLessThan` 到期已服务端拒一切调用；硬删只清残留 artifact。清理某次失败不影响安全（时间条件已失效）。幂等：删已删的 try/except 吞。

## 10. 状态 / Redis / 幂等 / 审计
- Redis `temp_ak:grant:{grant_id}`，30 天 TTL（对齐既有 `*:job:*`）。字段：`grant_id / mode(sts|ram) / bucket / read_prefixes / write_prefixes / actions / not_before / expire / recipient_email / policy_name / user_name / ak_id / status(NEW→ISSUED→REVOKED|FAILED) / requester / approver / created_ts / updated_ts`。**绝不存 secret / token。**
- **幂等**：`grant_id = hash(instance_code)`（一审批实例一凭证）；`_claim_instance` NX 锁防审批事件重投双发（复用 ram_approval）。
- **审计留痕**：grant 记录（谁申请/谁审批/生效/到期/收方/桶前缀）；阿里云侧 ActionTrail 记该 AK 全部 OSS 调用可追溯。secret/token 不入日志。

## 11. 三入口（飞书为主）
- **飞书**（主）：审批模板提交 → 审批通过事件 → 本模块建凭证 → 邮件下发 + 审批评论回执。（申请由审批发起，非卡片向导；结果卡/评论回内部群。）
- **Agent 工具** `tools/temp_ak_issuance/`（TOOL_GROUP `temp_ak`）：`manage_temp_ak`（`plan` 只读预览 policy+分流 / `status` 查 grant / `revoke` 手动吊销）。**`issue` 必须经审批门**——工具不得绕过审批直接发凭证（#55/#56 教训，见 §12）。
- **CLI** `python -m core.temp_ak_issuance.cli`（dry-run 默认，`--apply` 真发，运维/应急用；`--revoke <grant_id>` 手动清理）。

## 12. 安全（照 #55/#56 教训，任何绕过口都堵）
- **最小权限**：桶+前缀+具体动作收敛（build_policy 已具）；外部只给 AK、不开控制台、不入组。
- **到期强制失效**：policy `DateLessThan` 是外部/卖数据场景最关键控制（服务端每次判）；STS 叠加 ≤12h 天然自灭。
- **审批 admin 门**：建凭证只认实例级 `status=="APPROVED"`（复用 174-195 fail-safe）。**清点所有 launch 向量**：Agent 工具 `issue`、CLI `--apply`、查询/refresh 路都不得触发未审批发放（#56 HIGH-1 是查询 refresh 起未确认任务，#56 MED-2 是工具 apply 绕审批 → 本功能对应堵：工具/CLI 的真发放路径要么禁用、要么二次校验该 grant 对应审批实例已 APPROVED）。
- **凭证不入日志**：secret/token 绝不 log / 绝不落 Redis。
- **STS 分支的宽角色**信任策略只允许 Master AK，session policy 强收窄。

## 13. 配置项 / _REQUIRED_FIELDS / 部署
- 新 settings（`config/settings.py`）：`TEMP_AK_ENABLED`、`TEMP_AK_APPROVAL_CODE`、`TEMP_AK_OSS_ROLE_ARN`（STS 宽角色）、`TEMP_AK_BUCKET_MAP`（展示名→region/bucket，或复用 permsync BUCKET_MAP）、`TEMP_AK_CLEANUP_ENABLED`、`TEMP_AK_CLEANUP_HOUR`、`TEMP_AK_CHAT_ID`（内部回执群）、`TEMP_AK_MAX_DURATION_SECONDS`（STS 上限，默认 43200）。SMTP 复用既有 `SMTP_*`。RAM 可写 AK 复用 `ALIYUN_ACCESS_KEY_*`/`ALIBABA_CLOUD_*`。
- `_REQUIRED_FIELDS`(settings 310-326)：`TEMP_AK_ENABLED=true` 时补 `TEMP_AK_APPROVAL_CODE`、`TEMP_AK_OSS_ROLE_ARN`（STS）、`ALIYUN_ACCESS_KEY_ID/_SECRET`（方案 B 建号）、`SMTP_*`（下发）及其 impact 串。
- **部署**：纯 Python，无新依赖（RAM/STS/SMTP SDK 均已在）→ 普通 `deploy.ps1`，免 rebuild。
- **待真机取证（P1 前置，researcher 只读/dry-run）**：
  ① `oss:GetBucketAcl` 精确动作名 + 与现动作集差集；
  ② 用不存在的桶/构造性 policy dry-run `CreatePolicy`，校验时间条件 JSON（`acs:CurrentTime`+ISO8601+`+08:00`）被接受；
  ③ RAM 可写 AK 是否具备 §7 全套动作；
  ④ STS 宽角色 `MaxSessionDuration` 已调 43200 且信任 Master AK；`assume_role_with_policy` 传 `policy` 参数真机可用。

## 14. Epic → Task（有序，带验收 / owner / 依赖 / 工作量 / 优先级）
> owner 待与 dev 确认后落任务板；以下为建议。researcher 取证项可与 P0 并行。

### P0 骨架 + dry-run（价值高、风险低、无真机依赖）
- **T1 `policy.py` 生成 + 时间条件变体 + GetBucketAcl**（owner dev；deps 无；~M）
  - 验收：`build_policy_with_window` / `build_session_policy` 产出含 `DateGreaterThan/DateLessThan`(ISO8601+08:00) + 既有 `oss:Prefix` 条件；session policy ≤2048 字符；含 GetBucketAcl；单测覆盖时间条件/前缀/动作集。
- **T2 `issuer.py` 分流逻辑（dry-run）**（owner dev；deps T1；~M）
  - 验收：`expire-now≤12h→sts / else→ram` 边界正确（含"未来生效短窗""3天后短窗"用例）；dry-run 只产计划不调云；STS 分支组装 `assume_role_with_policy` 入参（duration cap 43200）、RAM 分支组装建号+附 policy 计划。
- **T3 `aliyun_sts.assume_role_with_policy` 新函数**（owner dev；deps 无；~S）
  - 验收：传 `role_arn/policy(json)/duration_seconds/session_name`；不污染既有 `assume_role_for_user`；单测 mock STS 客户端断言 `AssumeRoleRequest.policy/duration_seconds` 被设。
- **T4 `orchestrator.py` 状态机 + Redis + 幂等**（owner dev；deps T2；~M）
  - 验收：grant_id=hash(instance)、NX 幂等、状态流转 NEW→ISSUED→REVOKED/FAILED、记录不含 secret/token、30 天 TTL。
- **T5 审批接入 `handle_temp_ak_event`（复用门禁）**（owner dev；deps T4；~M）
  - 验收：只认实例级 status=="APPROVED"、无 instance_code fail-safe 不发、表单字段解析（桶/前缀/读写动作/生效/到期/收方邮箱）、routes 按 `TEMP_AK_APPROVAL_CODE` 路由不误撞 RAM 建号审批。
- **T6 cards + CLI + Agent 工具（issue 经审批门）**（owner dev；deps T4；~M）
  - 验收：工具 `plan/status/revoke` 只读或吊销、**无绕审批发放路径**；CLI dry-run 默认；TOOL_GROUPS 注册校验过。
- **T7 settings + _REQUIRED_FIELDS**（owner dev；deps T1-T6；~S）
  - 验收：新字段登记、`TEMP_AK_ENABLED` 时缺失项 validate 告警。
- **T8 tester 全面单测**（owner tester；deps T1-T7；~L）
  - 验收：分流边界/policy 时间条件/幂等/门禁 fail-safe/工具无绕过/清理顺序（mock 云）；全量 pytest 绿。
- **T9 auditor 过闸**（owner auditor；deps T8；~S）
  - 验收：无阻塞——重点查审批绕过口（工具/CLI/查询）、secret/token 不落盘不入日志、policy 时间条件正确、删号顺序幂等。

### P1 真机发放（前置真机取证，标待取证）
- **T10 researcher 真机取证**（owner researcher；deps 无，可与 P0 并行；~M）
  - 验收：§13 四项取证有结论带证据（GetBucketAcl 动作名 / CreatePolicy 时间条件校验 / RAM 可写 AK 权限 / STS 宽角色+policy 参数）。
- **T11 真机发一组 ≤12h STS**（owner dev+用户；deps T10、T2/T3；~M）
  - 验收：审批通过 → STS 单发含 Token、session policy 收窄到桶/前缀、到点自灭；外部可用（或确认 Token 消费方式）。
- **T12 真机发一组跨天方案 B**（owner dev+用户；deps T10、T4/T5；~M）
  - 验收：审批通过 → 建 user+AK+时间条件 policy、邮件下发、生效前调用被拒、生效后放行；grant 记录完整。
- **T13 到期清理真机验证**（owner dev；deps T12；~S）
  - 验收：`_temp_ak_cleanup_loop` 扫到期 grant 执行删号顺序、status=revoked、幂等重跑安全。

## 15. 待定项（dev 去和用户敲定）
- **A** 审批模板：新建独立 `TEMP_AK_APPROVAL_CODE` 模板（建议）还是复用现 RAM 建号模板？表单字段最终定义。
- **B** STS 分支的 SecurityToken：外部接收方能否消费带 Token 的临时凭证？不能 → 该场景一律走方案 B（长期 AK 无 Token）。用户接受"跨天=长期 AK 无 Token"取舍吗？
- **C** 分流边界定义确认：用 `expire − now ≤ 12h`（本方案）作 STS/方案 B 分界，OK？
- **D** 是否加 `acs:SourceIp` 条件锁外部方出口 IP（更防泄漏，需收方提供固定出口 IP）？
- **E** 下发通道：只邮件，还是邮件 + 审批评论 + 一次性安全链接？内部回执发哪个群（`TEMP_AK_CHAT_ID`）？
- **F** 建号用哪把 RAM 可写 AK：确认复用 oss_perm 的 `ALIYUN_ACCESS_KEY_*`；是否需要为外部发放单独一把权限更小的 AK？
- **G** STS 宽角色：新建专用 role（信任 Master AK、MaxSessionDuration=43200）由谁在控制台建？role ARN 填 `TEMP_AK_OSS_ROLE_ARN`。
- **H** 一个审批可否申请多桶/多前缀？（build_policy 支持多桶，但 STS session policy 2048 字符上限需留意。）
