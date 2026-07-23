# 火山引擎「临时 AK/SK 发放给外部访问 TOS」— 可行性调研

面向 dev/planner。对标已上线的阿里 OSS 版（`core/temp_ak_issuance/`，见 notes #57–#61）。目标：**尽量复用同一「审批 + 编排 + 下发」骨架，只换「建凭证 + policy 生成」引擎**。

标注：【文档】=火山官方明说附 URL；【实测】=SDK 源码/真机反查；【推测】=未证实需取证。

> **状态**：用户已拍板走**方案 B（长期 AK + 时间条件 policy + 到期硬删，支持跨天）**。方案 B 命门（TOS 是否执行 `volc:CurrentTime`）**已于 2026-07-22 真机坐实为「认」**——见文末 **§7**。

---

## 0. 一句话结论

**完全可行，且比预期省事**。火山与阿里在这块高度同构：
- 火山**有 STS**（`AssumeRole`，含 `SessionToken`，`DurationSeconds` 900–43200s，**上限 12h**，可传 inline `Policy` 现场收窄）——**与阿里几乎一比一**，可直接对标阿里的 STS 变体。（注意：CLAUDE.md / 代码里"火山无 STS"是 transfer/vePFS 图省事选静态 AK 的**局部说法，不是"火山没有 STS"**——本调研纠正之。）
- 火山 IAM 支持**时间条件**（`DateLessThan`/`DateGreaterThan` on `volc:CurrentTime`），TOS 支持**桶/前缀/动作**级 policy（`trn:tos:::bucket/prefix/*` + `StringLike:{tos:prefix}`），动作名 `tos:GetObject/PutObject/ListBucket…`。
- 项目已装 `volcengine-python-sdk`，其 `volcenginesdksts`(assume_role) + `volcenginesdkiam`(create_user/create_policy/attach_user_policy/create_access_key/delete_* /update_policy) **建凭证全套 API 齐备**【实测·SDK 源码 + 真机跑通】。
- temp_ak 框架**已经是 platform-aware**：审批表单已有「平台」字段、`_parse_platform` 已把 `火山/tos` 归 `volcano`、`_parse_directory` 已认 `tos://` scheme。当前唯一拦路 = `approval._validate_spec` 一句显式 `raise`（"平台=火山云 暂未支持"）。

**命门（决定架构）已真机坐实 ✅**：**TOS 的鉴权引擎确实执行 `volc:CurrentTime` 时间条件**——2026-07-22 在生产 `aiops-bot` 容器实测：过期条件 → TOS `head_object` 返回 **403 AccessDenied**；有效时间窗 → **200 放行**；改回过期又 → 403（可逆，证明由时间窗驱动）。UTC `Z` 与 `+08:00` 两种格式均被接受并强制。→ **方案 B 成立，"泄漏也随到期自动失效"关键控制有效**。全部细节见 **§7 命门取证结果（真机实测）**。

---

## 1. 火山临时凭证机制（STS / AssumeRole）

### 有 STS，且贴合阿里语义【文档 + 实测】
- `AssumeRole` `DurationSeconds`：**900–43200s，默认 3600，上限 = 12h**——与阿里 `AssumeRole` 上限**完全相同**。
  - 出处：[使用 AssumeRole 管理资源（Terraform）](https://www.volcengine.com/docs/6706/1223560)、[使用临时凭证访问 TOS](https://www.volcengine.com/docs/6627/102220)、[控制台内嵌 sessionDuration](https://www.volcengine.com/docs/6470/160159)。
- 火山**官方推荐**：服务端存长期 AK → 调 STS 换临时（临时 AK/SK/token）→ 下发客户端。正是本功能的形态。
- **inline 会话 Policy 支持**【实测·SDK】：`AssumeRoleRequest` 字段 = `role_trn`(注意是 **TRN** 不是 ARN)、`policy`(str, inline session policy)、`duration_seconds`(int)、`role_session_name`、`tags`。→ 可让 master AK 扮演一个"宽 TOS 角色"，AssumeRole 时用生成的窄 policy 现场收敛到具体桶/前缀，**免为每次申请建专用 role**（同阿里 session policy）。
- **返回凭证字段**【实测·SDK `CredentialsForAssumeRoleOutput`】：`access_key_id` / `secret_access_key` / **`session_token`**(火山管 SecurityToken 叫 SessionToken) / `expired_time` / `current_time`。→ 含 token、到点自灭、无需清理，最贴合"发临时凭证给外部"。
- **上限矛盾同阿里**：单张 STS ≤12h，外部无火山身份无法自助续签 → 窗口>12h（跨天/跨周）必须走长期 AK 版。分流逻辑可 100% 复用 `issuer.classify_mode`。
- **本次用户选型**：走方案 B（长期 AK），不建角色、不走 STS 快路（同阿里现状 `TEMP_AK_STS_MAX_SECONDS=0`）。STS 能力保留作未来可选变体，本期不实现。

---

## 2. 长期 AK + 时间条件 policy（方案 B，用户选型）

### 能建 IAM 子用户 + 长期 AK【实测·真机跑通】
`core/ram_approval.create_volcano_iam_account` 已能：建 user（`create_user`）→ profile → login profile → **AK**（`create_access_key`，返回 `access_key_id`/`secret_access_key`）→ 入组（`add_user_to_group`）。**但它不附任何 policy**——权限靠"入组"给。→ **temp_ak 的"每笔一策"自定义 policy 是全新引擎，不能复用 create_volcano_iam_account 的授权路径**（只复用其建 user + 建 AK 的动作）。§7 已真机验证全套建号+建策略+附加+建 AK 动作可用。

### 时间条件：IAM 层【文档】+ TOS 层【实测·已坐实 ✅】
- 通用 IAM Condition【文档】：日期运算符 `DateLessThan`/`DateLessThanEquals`/`DateGreaterThan`/`DateGreaterThanEquals`，时间条件键 **`volc:CurrentTime`**，时间格式 ISO8601（官方例 `2024-08-30T23:59:59Z`）。同一 statement 多条件 = **AND**（同阿里）。出处：[条件（Condition）](https://www.volcengine.com/docs/6257/1134320)。
  - **与阿里一处差异**：时间键 `volc:CurrentTime`（阿里 `acs:CurrentTime`）。时区两种格式（UTC `Z` / `+08:00`）**均可用**（§7 实测），建议统一发 UTC `Z`。
- **TOS 是否把 `volc:CurrentTime` 纳入鉴权判定 = 已实测「认」**（§7）。生效/到期由 TOS 服务端每次调用按当前时间强制判定，**泄漏也随到期自动失效**成立。
- **纵深防御仍保留到期硬删**：policy 时间窗服务端拒 = 主控制；cleanup 定时删 AK/user = 清残留 artifact（防子用户/AK 堆积）。二者叠加，同阿里方案 B。

---

## 3. TOS policy 限桶/限前缀/限动作【文档 + 实测】

官方示例（[使用 TOS Browser 的权限指南](https://www.volcengine.com/docs/6349/1183391)、[TOS 鉴权说明](https://www.volcengine.com/docs/6349/1183370)、[IAM 策略管理·对象存储](https://www.volcengine.com/docs/6349/102124)）：

- **Resource TRN 格式**：桶级 `trn:tos:::bucketname`；对象级 `trn:tos:::bucketname/prefix/*`。（对标阿里 `acs:oss:*:*:bucket/prefix*`）§7 实测 `trn:tos:::data-tran/*` 被正确解析并鉴权。
- **前缀限定 List**：`ListBucket` 是**桶级**动作，Resource 只能是桶本身；限目录用
  `"Condition": {"StringLike": {"tos:prefix": ["prefix/*"]}}`（对标阿里 `oss:Prefix`）。
- **动作名**（对标阿里 caps 三正交模型 read/download/write）：
  - **read（列清单，不下载）**：`tos:ListBucket`（+ `tos:prefix` 前缀条件）、`tos:HeadBucket`、`tos:GetBucketLocation`、`tos:ListBucketVersions`、`tos:GetBucketVersioning`。（对标阿里 `oss:ListObjects/GetBucketInfo/GetBucketStat/GetBucketLocation/GetBucketAcl`）
  - **download（下载）**：`tos:GetObject`、`tos:GetObjectAcl`、`tos:GetObjectVersion`。
  - **write（上传，无删除）**：`tos:PutObject`、`tos:AbortMultipartUpload`、`tos:ListMultipartUploadParts`（+ 可选 `tos:PutObjectAcl`）。**刻意排除 `tos:DeleteObject`/`tos:DeleteObjectVersion`**（同阿里 WRITE_ACTIONS 无 delete）。
- **鉴权规则**：显式 Deny 覆盖 Allow；无 Allow = 隐式拒绝（必须有 Allow 才授权）。§7 实测：时间条件不满足 = 该 Allow 不生效 = 隐式拒绝 = 403，符合此模型。
- **policy 文档结构**【实测】：**`{"Statement":[…]}`、无 `"Version"` 字段** 即被 `CreatePolicy` 接受并生效（§7 全程未加 Version，鉴权正常）。→ 火山版 `build_policy` **不要照抄阿里加 `"Version":"1"`**。

---

## 4. 项目现有火山能力可复用点【实测】

| 现有资产 | 位置 | 对本功能的价值 |
|---|---|---|
| 静态 AK | `settings.VOLCANO_ACCESS_KEY/SECRET`，回退 `TOS_ACCESS_KEY/SECRET` | 建号/建 policy/STS 的调用凭证来源。**注意生产 `VOLCANO_ACCESS_KEY` 当前为空，实际全靠 `TOS_ACCESS_KEY` 回退**（§7） |
| `volcengine-python-sdk`(已装) | `requirements.txt:36` | `volcenginesdksts.assume_role` + `volcenginesdkiam` 全套建号/策略 API |
| `make_volcano_iam_client()` | `core/ram_approval.py:603` | IAM client 工厂，直接复用。**region 用 `cn-beijing` 可正常调 IAM**（§7 实测；`cn-north-1` 亦可） |
| `create_volcano_iam_account()` | `ram_approval.py:507` | **建 user + 建 AK 逻辑可参照**，但它**不附 policy**、开 console、入组——temp_ak 要"不开 console/不入组/附每笔自定义 policy" |
| `_volcano_iam_models()` / 错误判定 `_volcano_is_not_exist/_already_exists` | `ram_approval.py:620+` | 幂等建号的 not-exist/exists 判定复用。**火山 not-found = HTTP 404 + body `Error.Code=UserNotExist`**（§7） |
| `volcano_iam_query` | `core/volcano_iam_query.py` | 只读查子用户/AK/组，cleanup 前核对可复用 |
| `get_tos_client()` / `TosClientV2(ak,sk,endpoint,region)` | `utils/volcano_client_factory.py` | 数据面校验/落盘客户端范式；外部拿长期 AK 后即用这套 |

**SDK 建凭证全套 API 已核实存在且真机跑通**【实测】：
- `volcenginesdksts`：`assume_role`、`get_caller_identity`。
- `volcenginesdkiam`：`create_user`/`get_user`/`delete_user`、`create_access_key`/`list_access_keys`/`update_access_key`(停用)/`delete_access_key`、`create_policy`/`update_policy`/`delete_policy`/`list_policies`、`attach_user_policy`/`detach_user_policy`/`list_attached_user_policies`。
- **延期(#58 P0.5)更省**：火山 `update_policy`(改整篇 `new_policy_document`) 比阿里 `create_policy_version`+rotate 简单，无版本管理。§7 实测 `update_policy` 改时间窗即时生效（~30s 传播）。

**建号所需权限（问题 5）已坐实**【实测·§7】：生产那把可写 AK（= `TOS_ACCESS_KEY` 回退）**已验证具备** `iam:CreateUser/CreateAccessKey/CreatePolicy/AttachUserPolicy/UpdatePolicy/DetachUserPolicy/DeletePolicy/DeleteUser/GetUser/ListUsers/ListPolicies` + `tos:PutObject/HeadObject/DeleteObject`。→ **方案 B 的建号+到期硬删全套动作现有 AK 够用，无需补授**。唯一待定：`DeleteAccessKey` 返回 400（见 §7，DeleteUser 已级联删除，非阻塞）。

---

## 5. 架构差异清单（能复用 vs 要新写）

**几乎全复用（审批 + 编排 + 下发骨架）**：
- `approval.py`：#55 实例级 `APPROVED` 硬门 / 白名单 / 防串 `_verify_enterprise` / 时间容错解析 / `_parse_platform`(**已识别 volcano**) / `_parse_directory`(**已认 tos://**) / caps 解析(read/download/write 正交) — **全复用**。**唯一改动**：`_validate_spec:290` 放开 `volcano` 分支 + 走火山校验（当前是硬 `raise`）。
- `orchestrator.py`：grant 状态机 NEW→ISSUED→REVOKED/FAILED、Redis `temp_ak:grant`、NX 锁、`grant_id` 幂等 hash — 全复用。**改动**：`resolve_bucket`(:95) 加 TOS 桶映射（**TOS 的 `tos://` 带不出 region**，需 `TEMP_AK_TOS_REGION` 配置或回退 `TOS_REGION`，同 transfer/bucket_transfer 的既有做法）；mode 分流按平台分派到火山 issuer。
- `delivery.py`：审批评论下发 + 全链路脱敏 + creds-undelivered 告警 — 厂商无关，**全复用**。
- `cards.py`：卡片文案个别处写"OSS"，参数化即可。
- `dsw_scheduler._temp_ak_cleanup_loop`：调度骨架复用，扫 `temp_ak:grant:*` 到期 → 分派火山 cleanup。
- 工具 `manage_temp_ak`(plan/status/revoke，无 issue) / CLI — 复用，plan/revoke 内部按 platform 分派。

**要新写（换引擎，加法式新文件，勿改阿里现有）**：
- `policy_volcano.py`（对标 `policy.py`）：
  - `build_policy_with_window` 火山版。Resource `trn:tos:::bucket[/prefix/*]`；动作集见 §3；前缀 List 用 `StringLike:{tos:prefix}`；时间窗 `DateGreaterThan/DateLessThan` on **`volc:CurrentTime`**、**UTC `Z`** 格式；可选 IP `volc:SourceIp`(需取证 TOS 是否支持，§6 #4)；**不加 `Version`**（§7 实测无需）。
- `issuer_volcano.py`（对标 `issuer.py` 的方案 B 分支）：
  - `iam.create_user`(无 console/无组) → `create_policy(policy_name, policy_document)` → `attach_user_policy(policy_name, policy_type="Custom", user_name)` → `create_access_key`。延期 = `update_policy(policy_name, new_policy_document=…)`。
- `cleanup_volcano.py`（对标 `cleanup.py`）：`update_access_key`(Inactive 先失效) → `delete_access_key` → `detach_user_policy` → `delete_policy` → `delete_user`。§7 实测 `DeleteUser` 可级联删 AK，但 `DeleteAccessKey` 单独调返 400——清理序建议**先 `update_access_key`→Inactive 再 `delete_access_key`**，或直接依赖 `delete_user` 级联（§6 #5 待补确认精确语义）。
- 配置：`TEMP_AK_VOLCANO_ENABLED`、`TEMP_AK_TOS_BUCKET_MAP`、`TEMP_AK_TOS_REGION`、复用 `VOLCANO_ACCESS_KEY/SECRET`(空则回退 `TOS_ACCESS_KEY`)。加进 `Config._REQUIRED_FIELDS` 的告警。IAM 调用 region 用 `VOLCANO_IAM_REGION`(=cn-beijing 可用)。

---

## 6. 剩余真机取证待办（方案 B 命门已闭，以下为完善项）

- ~~#1 TOS 是否执行 `volc:CurrentTime`~~ → **已坐实「认」（§7）**。
- ~~#3 写 AK 权限边界~~ → **已坐实够用（§7 OPS 全 OK）**。
- ~~#6 policy `Version` 字段 / 时间格式~~ → **已坐实：无需 Version；UTC `Z` 与 `+08:00` 均可（§7）**。
- **#2（本期不做，STS 变体才需）** STS AssumeRole inline `Policy` 交集语义 + `role_trn` 信任策略 + Policy 参数字符上限。
- **#4 TOS 是否支持 `volc:SourceIp` IP 条件**（防泄漏加固，同阿里 acs:SourceIp）——建议开发时用同 §7 探针法一测（有效 IP 放行 / 错误 IP 拒）。
- **#5 `DeleteAccessKey` 400 的精确语义**：确认是否须先 `update_access_key`→Inactive 才能 `delete_access_key`，还是 `delete_user` 级联即可（§7 已知级联可用、无残留）。影响 cleanup 引擎删除序，非阻塞。
- **#7（本期不做）** 火山 AssumeRole 角色 MaxSessionDuration 是否需从 1h 调到 12h（仅 STS 变体相关）。

---

## 7. 命门取证结果（真机实测 2026-07-22，生产 `aiops-bot` 容器）

**探针设计（方案甲，最小副作用）**：用生产可写 AK（`TOS_ACCESS_KEY` 回退，IAM region=cn-beijing），建 1 个临时子用户 + 1 把 AK + 1 条自定义 TOS policy；用 master AK 先 PUT 一个**自建的 1 字节探针对象**（`data-tran/__volc_perm_probe__/probe-*.txt`，非外采数据），再用子用户 AK 对该对象 `head_object`，以 **200(放行) vs 403(拒绝)** 作无歧义判据（绕开 404/403 存在性歧义）。全程 try/finally 自清理，**不打印任何 AK/SK/token**（源码级 scrub）。

**判据序列（每步轮询以吸收 IAM 传播延迟）**：
| # | policy 时间条件 | 期望 | 实测 | 传播耗时 |
|---|---|---|---|---|
| 1 | 无条件（基线，验管线通） | 放行 200 | **ALLOW(200)** ✅ | 1s |
| 2 | `DateLessThan volc:CurrentTime=昨天Z`（过期） | 拒绝 403 | **DENY(403 AccessDenied)** ✅ | 37s |
| 3 | `DateGreaterThan 昨天Z AND DateLessThan 明天Z`（有效窗） | 放行 200 | **ALLOW(200)** ✅ | 34s |
| 4 | `DateLessThan …+08:00`（过期，偏移量格式） | 拒绝 403 | **DENY(403)** ✅ | 34s |

**结论**：
1. **TOS 鉴权引擎确实执行 IAM policy 的 `volc:CurrentTime` 时间条件**：过期→403、有效→200、改回过期→403（可逆，排除"策略坏了/AK 失效"等替代解释，唯一变量是时间窗）。→ **方案 B「泄漏也随到期自动失效」成立**。
2. **时间格式**：UTC `Z`（`2026-07-21T11:35:53Z`）**与** `+08:00` 偏移量**均被接受并强制**。建议实现统一发 **UTC `Z`**（对齐火山官方例，避免时区歧义）。
3. **policy 文档**：`{"Statement":[…]}` 无 `"Version"` 字段即被 `CreatePolicy` 接受且鉴权正常；`trn:tos:::<bucket>/*` 资源格式正确解析。
4. **传播延迟**：policy 附加/`update_policy` 变更约 **30–40s** 生效（非即时）。→ 实现启示：**"撤销"若靠改/删 policy，非秒级生效；真正的即时止损是硬删 AK**（temp_ak 的 revoke 已是删 AK+policy+user，符合）。发放后首次可用亦需容忍数十秒传播。
5. **写 AK 权限（问题 5/§6 #3）全部够用**：本次真机成功执行 `PutObject / CreateUser / CreateAccessKey / CreatePolicy / AttachUserPolicy / UpdatePolicy(×3) / DetachUserPolicy / DeletePolicy / DeleteUser / GetUser / ListUsers / ListPolicies / DeleteObject`——**方案 B 建号+到期硬删全套现有 AK 均有权限，无需补授**。
6. **IAM 可达性**：region `cn-beijing` 与 `cn-north-1` 均可调 IAM；火山 not-found 语义 = HTTP **404 + body `Error.Code=UserNotExist`**（不是网关 404，是正常业务响应）。

**清理已核实干净 ✅**：
- 探针实体全部删除：`DetachUserPolicy` OK、`DeletePolicy` OK、`DeleteUser` OK（`get_user` 复查 = `UserNotExist`/GONE）、探针对象 `DeleteObject` OK。
- **独立只读复扫**：`list_users`(共 49) + `list_policies(query="zz-permprobe")` → **`zz-permprobe*` 残留均为 0**。
- 唯一异常：`DeleteAccessKey` 返回 **400 Bad Request**（未细究，疑似须先置 Inactive）；但 **`DeleteUser` 成功并级联删除该 AK**（子用户已 GONE，AK 为其子资源不可能孤存，复扫亦无残留）→ **无泄漏**。此点记为 §6 #5 的 cleanup 引擎设计待办（删除序：先 Inactive 或依赖 user 级联），非阻塞。

**探针方法学备注**：脚本经 `ssh bot-new "docker exec -i aiops-bot python3 -"` 以 stdin 喂入，**服务器/容器/仓库均无脚本落盘残留**；密钥仅存在于容器进程内存、随进程退出消失，全程未打印。

---

## 附：出处
- STS AssumeRole / 时长：[Terraform AssumeRole](https://www.volcengine.com/docs/6706/1223560)、[使用临时凭证访问 TOS](https://www.volcengine.com/docs/6627/102220)、[控制台内嵌](https://www.volcengine.com/docs/6470/160159)
- IAM Condition / 时间条件：[条件（Condition）](https://www.volcengine.com/docs/6257/1134320)
- TOS policy 桶/前缀/动作：[TOS Browser 权限指南](https://www.volcengine.com/docs/6349/1183391)、[TOS 鉴权说明](https://www.volcengine.com/docs/6349/1183370)、[IAM 策略管理·对象存储](https://www.volcengine.com/docs/6349/102124)
- IAM 授权 API：[AttachUserPolicy](https://www.volcengine.com/docs/6257/65029)、[策略概述](https://www.volcengine.com/docs/6257/65058)
- SDK 方法/字段【实测】：`volcengine-python-sdk` GitHub `volcenginesdksts/api/sts_api.py`、`volcenginesdkiam/api/iam_api.py`、`*/models/{assume_role_request,create_policy_request,update_policy_request,attach_user_policy_request,credentials_for_assume_role_output}.py`
- 命门真机取证【实测】：§7，2026-07-22 于生产 `aiops-bot` 容器，探针一拒一放一复原 + 独立复扫，实体已清理干净。
