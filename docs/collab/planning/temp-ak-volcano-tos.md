# 火山云 TOS 临时凭证发放（方案 B）— 开发落地方案

面向 dev。据 `docs/collab/research/temp-ak-volcano-tos.md`（命门已真机坐实，见其 §7）+ 用户拍板走**方案 B**（长期 IAM AK + `volc:CurrentTime` 时间条件 policy + 到期硬删，支持跨天）。

## 0. 需求一句话

给外采企业发一组**火山 TOS** 访问凭证（对标已上线的阿里 OSS 版 `core/temp_ak_issuance/`，见 CLAUDE.md「临时 AK/SK 发放」段 + notes #57–#61）。审批表单「平台」选**火山云** → 路由到火山引擎：建 IAM 子用户 + 长期 AK + 时间窗自定义 TOS policy；到期由 policy 服务端拒 + 定时硬删双控。**阿里云现有代码零改，加法式。**

## 1. 铁律：加法式，零改阿里

现有框架已 **platform-aware**——审批表单已有「平台」控件、`approval._parse_platform` 已把 `火山/tos/volc` 归 `volcano`、`_parse_directory` 已认 `tos://`、`orchestrator` grant 已存 `platform` 字段、`cards` 已有 `火山云 TOS` 文案。当前唯一拦路 = `approval._validate_spec:290` 一句硬 `raise`（"平台=火山云 暂未支持"）。

本方案只做两类改动：
- **A. 新写 3 个火山专属引擎文件**（`policy_volcano.py` / `issuer_volcano.py` / `cleanup_volcano.py`），不碰阿里 `policy.py`/`issuer.py`/`cleanup.py`。
- **B. 在 6 个既有分派点按 `platform` 字段做 if 分流**（阿里走原路，火山走新引擎），改动均是"加一个 elif 分支"级别，不重写既有逻辑。

审批 / 编排 / 下发 / 卡片 / 调度 / 工具 / CLI / 命名 / 白名单 / 门禁 **全复用同一骨架**，只有"建凭证 + policy + cleanup"换引擎。

---

## 2. 架构：复用 vs 新增文件清单

### 复用（不改，或仅加 `platform` 分派 elif）

| 文件 | 复用内容 | 是否需改 |
|---|---|---|
| `core/temp_ak_issuance/approval.py` | 事件入口 / #55 实例级 APPROVED 门禁 / 白名单 / `_parse_platform`(已识别 volcano) / `_parse_directory`(已认 tos://) / `_parse_perm`(read/download/write) / `_parse_date_interval` / `_verify_enterprise` 防串 / 延期+撤销分支 | **改 1 处**：`_validate_spec` 放开 volcano（见 §3.1） |
| `core/temp_ak_issuance/orchestrator.py` | grant 状态机 NEW→ISSUED→REVOKED/FAILED / Redis `temp_ak:grant:*` / NX 锁 / `grant_id` 幂等 hash / `_derive_user_name` 拼音登录名 / `display_name_for` / 展示助手 / grant 已存 `platform` | **改 2 处**：`resolve_bucket` 加 TOS 桶映射；`issue_grant`/`extend_grant` 里的 mode 与 policy_name 派生按 platform（见 §3.2） |
| `core/temp_ak_issuance/issuer.py` | `classify_mode`（方案 B 判定，火山同样 `TEMP_AK_STS_MAX_SECONDS=0` → 全 RAM_MODE） / `plan` dry-run 骨架 | **改 1 处**：`plan`/`issue` 按 platform 派发到 `issuer_volcano`（见 §3.3） |
| `core/temp_ak_issuance/cleanup.py` | `sweep_expired`（扫 Redis 到期）/ `revoke_grant` 分派入口 | **改 1 处**：`revoke_grant` 按 `grant["platform"]` 分派到 `cleanup_volcano`（见 §3.4） |
| `core/temp_ak_issuance/delivery.py` | 审批评论下发 / 脱敏 / creds-undelivered 告警 / `credential_text`（含 SecurityToken 兼容） | **零改**（厂商无关；火山无 token，`security_token` 空串照样跳过） |
| `core/temp_ak_issuance/cards.py` | `receipt_card` / `status_card`，`_PLATFORM_CN` 已含 `volcano: 火山云 TOS` | **零改** |
| `core/temp_ak_issuance/cli.py` | `plan`/`status`/`revoke`/`sweep`，dry-run 默认 | **改 1 处**：`plan` 支持 `--platform volcano`（可选，见 §7） |
| `tools/temp_ak_issuance/manage_temp_ak.py` | `plan`/`status`/`revoke`，无 issue（杜绝绕审批） | **改 0–1 处**：plan 走 platform 分派（内部复用 issuer.plan，已分派则零改） |
| `core/ram_approval.py` | `make_volcano_iam_client()`(:603) / `_volcano_iam_models()`(:620) / `_volcano_err_code`/`_volcano_is_not_exist`/`_volcano_is_already_exists`(:662+) / `_approval_comment_user_id` / `_send_approval_comment` / `fetch_approval_instance` / 实例级 APPROVED 门禁 | **零改**（只调用） |
| `core/feishu_bot/routes.py` | `_approval_allowlist`(:21) 白名单已含 temp_ak code；审批分发 `handle_temp_ak_event`/`handle_temp_ak_extend_event`(:98-111) | **零改**（同一 code，platform 在表单内分流） |
| `core/dsw_scheduler.py` | `_temp_ak_cleanup_loop`(:838) 每日北京扫 sweep | **零改**（`sweep_expired`→`revoke_grant` 内部按 platform 分派，调度层无感） |
| `config/settings.py` | 全部 `TEMP_AK_*` 字段 + `VOLCANO_*`/`TOS_*` 凭证 | **加字段**（见 §5） |

### 新增（火山专属，加法式）

| 新文件 | 对标阿里 | 职责 |
|---|---|---|
| `core/temp_ak_issuance/policy_volcano.py` | `policy.py` | 生成火山 TOS policy 文档（`volc:CurrentTime` UTC `Z` + `trn:tos:::bucket/prefix/*` + `tos:` 动作 read/download/write 三正交、无 delete、**无 `Version` 字段**） |
| `core/temp_ak_issuance/issuer_volcano.py` | `issuer.py` 的 `_issue_ram`/`rewrite_ram_window` | 方案 B 建号：`create_user`(无 console/无组) → `create_policy` → `attach_user_policy(Custom)` → `create_access_key`；延期 = `update_policy`（整篇 `new_policy_document`，无版本管理） |
| `core/temp_ak_issuance/cleanup_volcano.py` | `cleanup.py` 的 `revoke_grant` RAM 分支 | 到期硬删：`update_access_key`(Inactive) → `delete_access_key` → `detach_user_policy` → `delete_policy` → `delete_user`（火山差异见 §4.3） |

新增单测：`tests/unit/test_temp_ak_policy_volcano.py` / `test_temp_ak_issuer_volcano.py`（mock SDK）/ `test_temp_ak_cleanup_volcano.py` / `test_temp_ak_approval_volcano.py`（volcano 分支解析+校验）。

---

## 3. 平台分流接线点（file:line 级）

> 分派统一读 `grant["platform"]`（发放时由 `spec["platform"]` 落库，已在 `orchestrator.create_grant_record:168`）。约定：一个内部 helper `_is_volcano(grant) -> bool`（`grant.get("platform") == "volcano"`）放 `issuer.py` 或 `orchestrator.py`，各分派点复用。

### 3.1 `approval._validate_spec`（approval.py:288-306）— 放开 volcano 校验

现状：
```python
def _validate_spec(spec: dict) -> None:
    if spec["platform"] == "volcano":
        raise orchestrator.TempAkError("平台=火山云 暂未支持…")   # ← 拆掉这句
    if spec["platform"] != "aliyun":
        raise orchestrator.TempAkError("无法识别申请平台…")
    ...
```
改为：`volcano` 与 `aliyun` 都放行，共用后续 bucket/caps/expire/not_before 校验（这些厂商无关）。仅加一个 volcano 专属守卫：`TEMP_AK_VOLCANO_ENABLED` 为 false 时对 volcano 请求友好拒（灰度开关）。`unknown` 仍拒。伪代码：
```python
if spec["platform"] not in ("aliyun", "volcano"):
    raise TempAkError("无法识别申请平台，请在审批表单选择阿里云或火山云")
if spec["platform"] == "volcano" and not settings.TEMP_AK_VOLCANO_ENABLED:
    raise TempAkError("火山云 TOS 临时凭证未启用（TEMP_AK_VOLCANO_ENABLED=false）")
# 下面 bucket/caps/expire/not_before 校验对两平台通用，保持不动
```

### 3.2 `orchestrator`（orchestrator.py）— 桶解析 + mode/policy_name 派生

- **`resolve_bucket`(:94-111)**：现在只查阿里 `TEMP_AK_BUCKET_MAP` → `permsync.BUCKET_MAP` → 原样。火山需**独立**桶映射（TOS 展示名→真实桶）。方案：新增 `resolve_bucket_volcano(display)`，或给 `resolve_bucket` 加 `platform` 参：
  ```python
  def resolve_bucket(display, platform="aliyun"):
      if platform == "volcano":
          m = json.loads(settings.TEMP_AK_TOS_BUCKET_MAP_RAW or "{}")
          if display in m and isinstance(m[display], dict):
              return m[display].get("region",""), m[display].get("bucket") or display
          return "", display   # 表单直接填真实桶
      # ...原阿里逻辑不动
  ```
  调用点 `create_grant_record:161` 改为 `resolve_bucket(spec["bucket"], spec.get("platform","aliyun"))`。
  - **火山 region 命门**：`tos://` **带不出 region**（同 transfer/bucket_transfer 既有痛点）。real region 优先取桶映射 value 的 `region`，回退 `settings.TEMP_AK_TOS_REGION`，再回退 `TOS_REGION`。region 落 `grant["region"]` 供 issuer/cleanup 建 IAM client 用（火山 IAM 用 `cn-beijing` 即可调，但 TOS 数据面 region 需与桶一致——policy 的 `trn:tos:::` **不含 region**，故 region 只影响可选的数据面校验探针，不影响 policy 本身）。

- **`create_grant_record:180`** policy_name 派生：现 `(POLICY_PREFIX + user_name) if mode == RAM_MODE else ""`。火山方案 B 恒为 RAM_MODE（`classify_mode` 在 `TEMP_AK_STS_MAX_SECONDS=0` 下永远返回 RAM），故此行对火山天然成立，**无需改**。仅 policy 命名前缀建议区分：火山用独立前缀（见 §4.1），可在此按 platform 选 `POLICY_PREFIX` vs `policy_volcano.POLICY_PREFIX`。

- **`issue_grant`(:194-207)** 与 **`extend_grant`(:216-251)**：内部调 `issuer.issue`/`issuer.rewrite_ram_window`。分派下沉到 `issuer` 层（见 §3.3），orchestrator 这两函数**几乎不改**——只需确保 `grant["mode"]` 与 `ak_id` 回写逻辑对火山成立（火山也返回 `access_key_id`，成立）。`extend_grant` 的 STS 分支火山不会走（恒 RAM_MODE），保持不动。

### 3.3 `issuer.plan` / `issuer.issue` / `issuer.rewrite_ram_window`（issuer.py）— 派发到 volcano 引擎

在 `issue()`(:65) 顶部加 platform 派发：
```python
def issue(grant):
    if grant.get("platform") == "volcano":
        from . import issuer_volcano
        return issuer_volcano.issue(grant)     # 火山方案 B（恒长期 AK）
    mode = grant.get("mode") or classify_mode(grant["expire"])
    ...原阿里 STS/RAM 分流不动
```
`plan()`(:44) 同理派发到 `issuer_volcano.plan(grant)`（dry-run 出火山 policy 预览，不调云）。`rewrite_ram_window()`(:152) 延期同理派发到 `issuer_volcano.rewrite_window(grant)`。

> 注：也可把派发写在 `orchestrator.issue_grant`/`extend_grant` 里而非 `issuer.issue`。**推荐放 `issuer.issue`/`plan`/`rewrite_ram_window`**——因为 CLI/工具的 `plan` 也经 `issuer.plan`，一处派发覆盖三入口，且 orchestrator 状态机彻底厂商无关。dev 二选一，保持单一分派层。

### 3.4 `cleanup.revoke_grant`（cleanup.py:48）— 派发到 volcano 清理

```python
def revoke_grant(grant, *, log=None):
    if grant.get("stage") == o.STAGE_REVOKED:
        return True
    if grant.get("platform") == "volcano":
        from . import cleanup_volcano
        return cleanup_volcano.revoke_grant(grant, log=log)
    if grant.get("mode") != o.issuer.RAM_MODE:   # 阿里 STS 自灭
        ...原阿里逻辑不动
```
`sweep_expired`(:100) 无需改——它逐条调 `revoke_grant`，分派在后者内部完成。调度 `_temp_ak_cleanup_loop` 因此零改。

### 3.5 汇总：改动清单（6 处 + 3 新文件 + 配置）

| # | 文件:行 | 改动 | 风险 |
|---|---|---|---|
| 1 | `approval.py:290` | 放开 volcano 校验 + `TEMP_AK_VOLCANO_ENABLED` 守卫 | 低（阿里路径不变） |
| 2 | `orchestrator.py:94` | `resolve_bucket` 加 platform 参 + TOS 桶映射 | 低 |
| 3 | `orchestrator.py:161` | `create_grant_record` 传 platform 给 resolve_bucket；policy 前缀按 platform | 低 |
| 4 | `issuer.py:44,65,152` | `plan`/`issue`/`rewrite_ram_window` 顶部 platform 派发 | 低（阿里在 else） |
| 5 | `cleanup.py:48` | `revoke_grant` 顶部 platform 派发 | 低 |
| 6 | `settings.py` | 加 `TEMP_AK_VOLCANO_ENABLED`/`TEMP_AK_TOS_BUCKET_MAP`/`TEMP_AK_TOS_REGION` + `_REQUIRED_FIELDS` | 低 |
| N1 | `policy_volcano.py` | 新文件 | — |
| N2 | `issuer_volcano.py` | 新文件 | — |
| N3 | `cleanup_volcano.py` | 新文件 | — |

---

## 4. 三个新引擎：职责 + 关键函数签名

### 4.1 `policy_volcano.py`（对标 `policy.py`，纯逻辑不调云）

与阿里的**四点差异**（research §3/§7 坐实）：
1. 时间键 `volc:CurrentTime`（阿里 `acs:CurrentTime`），**UTC `Z` 格式**（`2026-07-21T11:35:53Z`，非 `+08:00`）。
2. Resource TRN `trn:tos:::bucket` / `trn:tos:::bucket/prefix/*`（阿里 `acs:oss:*:*:bucket/prefix*`），**无 region/account 段**。
3. 动作名 `tos:*`（见下）；List 前缀条件键 `tos:prefix`（阿里 `oss:Prefix`）。
4. policy 文档**不加 `"Version"` 字段**（阿里加 `"Version":"1"`）——`{"Statement":[…]}` 即可。

动作集（三正交，对标阿里、无 delete）：
```python
POLICY_PREFIX = "tempak-tos-auto-"     # 火山独立前缀，区别阿里 temp-ak-auto-
LIST_ACTIONS     = ["tos:ListBucket", "tos:HeadBucket", "tos:GetBucketLocation"]   # read：列清单不下载
DOWNLOAD_ACTIONS = ["tos:GetObject"]                                               # download：下载
WRITE_ACTIONS    = ["tos:PutObject", "tos:AbortMultipartUpload",
                    "tos:ListMultipartUploadParts"]                                # write：上传，无 DeleteObject
```
> 动作名精确清单以 research §3 为准；`tos:ListBucketVersions`/`GetObjectAcl` 等按需增补（P1 真机探针法一测，见 §9 遗留 #4 同法）。**刻意排除 `tos:DeleteObject`。**

关键函数（签名镜像阿里，便于分派层无脑调用）：
```python
def iso8601_utc(epoch: float) -> str:
    """epoch → '2026-07-21T11:35:53Z'（UTC Z，火山官方例格式）。"""

def _time_conditions(not_before, expire, source_ips=None) -> dict:
    """{"DateGreaterThan":{"volc:CurrentTime":<nb Z>},
        "DateLessThan":{"volc:CurrentTime":<exp Z>}}；多运算符同 statement=AND。
       source_ips 可选 IpAddress:{volc:SourceIp}（P1 取证后启用，见 §9 #4）。"""

def build_policy_with_window(bucket, *, prefix="", caps, not_before, expire, source_ips=None) -> dict:
    """单桶单目录 + caps⊆{read,download,write} + 时间窗 → TOS policy dict。
       download→GetObject 对象语句(trn .../prefix/*)；write→上传对象语句；
       read→ListBucket 桶级语句(trn ...bucket)+prefix 时 StringLike:{tos:prefix}。
       返回 {"Statement":[...]}（无 Version）。caps 空→空 Statement。"""
```
> 无 `build_session_policy`（STS 变体本期不做，见 §8）。若后续加火山 STS，再补 inline session policy 变体（research §1）。

### 4.2 `issuer_volcano.py`（对标 `issuer._issue_ram` + `rewrite_ram_window`）

用 `ram_approval.make_volcano_iam_client()` + `_volcano_iam_models()`（复用，IAM region=cn-beijing 可用）。凭证 = `VOLCANO_ACCESS_KEY/SECRET`，空则回退 `TOS_ACCESS_KEY/SECRET`（research §4 坐实生产靠 TOS 回退）。

**与阿里 `_issue_ram` 的三点 SDK 差异**（已核实 `volcenginesdkiam` 源码签名）：
- 响应取值：火山 `resp.access_key`（阿里 `resp.body.access_key`）；secret 字段 `secret_access_key`（阿里 `access_key_secret`）。
- policy 幂等：火山无 policy version 概念，延期用 `update_policy(policy_name, new_policy_document=…)` **改整篇**（阿里 `create_policy_version`+rotate）。
- 建 user：`CreateUserRequest(user_name, display_name, description=…)`，**不传** `console_access`/groups（对标 `create_volcano_iam_account` 但反过来：不开 console、不入组、附自定义 policy）。not-exist 判定复用 `_volcano_is_not_exist`（HTTP 404 + body `Error.Code=UserNotExist`）。

关键函数：
```python
def plan(grant: dict) -> dict:
    """dry-run：出火山 policy 预览，不调云。返回 {mode:"ram", user_name, policy_name, policy, has_token:False}。"""

def issue(grant: dict) -> dict:
    """真发放（火山方案 B）。返回 {access_key_id, access_key_secret, security_token:"", expire_ts, mode:"ram"}。
       1) create_user（幂等：get_user 404→create；不开 console/不入组）
       2) create_policy（不存在建）/ update_policy（存在改整篇 new_policy_document）
       3) attach_user_policy(policy_name, policy_type="Custom", user_name)（幂等，already-exists 吞）
       4) create_access_key(user_name) → resp.access_key.{access_key_id, secret_access_key}
       凭证只当次返回给下发层，绝不落 Redis/日志。"""

def rewrite_window(grant: dict) -> None:
    """延期：update_policy(policy_name, new_policy_document=新时间窗 doc)。AK/user 不变、不重发凭证。
       ~30-40s 传播（research §7），非即时。"""
```
SDK 调用样板（已按 `volcenginesdkiam` 真实 attribute 名核对）：
```python
client.create_user(models.CreateUserRequest(
    user_name=user, display_name=o.display_name_for(grant),
    description=(grant.get("reason") or "temp-ak tos external issuance")[:128] or None))
client.create_policy(models.CreatePolicyRequest(
    policy_name=pol, policy_document=json.dumps(doc, ensure_ascii=False),
    description="temp-ak tos external issuance (time-boxed)"))
client.attach_user_policy(models.AttachUserPolicyRequest(
    policy_name=pol, policy_type="Custom", user_name=user))
resp = client.create_access_key(models.CreateAccessKeyRequest(user_name=user))
ak = getattr(resp, "access_key", None)   # 注意：resp.access_key 非 resp.body
# update_policy(models.UpdatePolicyRequest(policy_name=pol, new_policy_document=json.dumps(doc)))
```
> **user_name 复用** `orchestrator._derive_user_name`（拼音 slug，火山 user_name 字符集与阿里同 `[A-Za-z0-9.@_-]`，兼容）。**display_name 复用** `display_name_for`（可中文，火山 `CreateUserRequest.display_name` 接受）。

### 4.3 `cleanup_volcano.py`（对标 `cleanup.revoke_grant` RAM 分支）

删序（照阿里，火山 SDK 名差异）：`list_access_keys` → 每个 AK `update_access_key(status="Inactive")` → `delete_access_key` → `detach_user_policy` → `delete_policy` → `delete_user`。全步幂等（删已删 try/except 吞）。

**火山差异（research §7 坐实）**：
- `DeleteAccessKeyRequest(access_key_id=, user_name=)`（阿里是 `user_access_key_id=`）；`UpdateAccessKeyRequest(access_key_id=, user_name=, status="Inactive")`。
- **`DeleteAccessKey` 单删真机返 400**（research §7 #6 未究因，疑须先 Inactive）——但 **`DeleteUser` 级联删净该 AK**（子用户 GONE → AK 子资源不可孤存 → 复扫 0 残留）。故删序建议：**先 `update_access_key`→Inactive（软失效），再尝试 `delete_access_key`（吞 400），最后依赖 `delete_user` 兜底级联**。纵深防御：policy 时间窗到期已服务端拒一切调用，硬删只清 artifact，某步失败不影响安全。
- 无 policy version → 删 policy 直接 `delete_policy(policy_name)`（阿里要先删非默认版本；`DeletePolicyRequest` 火山**只收 policy_name**，无 policy_type）。
- `detach_user_policy(policy_name, policy_type="Custom", user_name)`。
- not-found 判定复用 `_volcano_is_not_exist`。

关键函数（签名镜像阿里 `revoke_grant`/`sweep_expired` 不改）：
```python
def revoke_grant(grant: dict, *, log=None) -> bool:
    """火山方案 B 到期/手动硬删：软失效 AK → 删 AK(吞400) → 解绑 policy → 删 policy → 删 user(级联兜底)。
       翻 grant.stage=REVOKED，返回是否成功。幂等。"""

def _list_ak_ids(client, models, user_name) -> list[str]:
    """list_access_keys → resp.access_key_metadata[].access_key_id（阿里是 resp.body.access_keys.access_key）。"""
```

---

## 5. 配置项新增

### 新增（`config/settings.py`，加在 `TEMP_AK_*` 段）

```python
# 火山云 TOS 临时凭证发放（方案 B）：与阿里 OSS 版共用审批/编排/下发骨架，platform=volcano 分派火山引擎
TEMP_AK_VOLCANO_ENABLED   = os.environ.get("TEMP_AK_VOLCANO_ENABLED", "false").lower() == "true"
# TOS 展示桶名 → {region,bucket}（JSON）。留空则表单填真实桶名原样用。tos:// 带不出 region 故需此映射或下面回退
TEMP_AK_TOS_BUCKET_MAP    = os.environ.get("TEMP_AK_TOS_BUCKET_MAP", "{}")
# TOS 桶 region 回退（映射无 region 时用；再空回退 TOS_REGION）。仅影响可选数据面校验，policy trn 不含 region
TEMP_AK_TOS_REGION        = os.environ.get("TEMP_AK_TOS_REGION", "")
```

### 复用（不新增，火山引擎直接读）

| 已有字段 | 用途 |
|---|---|
| `TEMP_AK_ENABLED` | 总开关（gates routes 白名单 + 审批分发 + 调度）。火山也受它管 |
| `TEMP_AK_APPROVAL_CODE` / `TEMP_AK_EXTEND_APPROVAL_CODE` | **同一审批模板 code**（发放/延期），platform 在表单内分流，见 §6 |
| `TEMP_AK_STS_MAX_SECONDS`（=0 全走方案 B） | 火山恒方案 B（本期无火山 STS） |
| `TEMP_AK_CLEANUP_ENABLED` / `TEMP_AK_CLEANUP_HOUR` | 到期硬删调度（火山共用同一 loop） |
| `TEMP_AK_CHAT_ID` | 内部通知群 |
| `TEMP_AK_FIELD_*` | 表单字段映射（含 widget id，火山申请同模板同字段） |
| `VOLCANO_ACCESS_KEY/SECRET`（空回退 `TOS_ACCESS_KEY/SECRET`） | 建号/建 policy/cleanup 调用凭证（research §4/§7 坐实够权） |
| `VOLCANO_IAM_REGION`（默认 cn-beijing） | IAM client region（`make_volcano_iam_client` 已读） |
| `TOS_REGION` | TEMP_AK_TOS_REGION 空时的 region 回退 |

### `_REQUIRED_FIELDS`（settings.py:434 `if self.TEMP_AK_ENABLED` 块）

在 `TEMP_AK_VOLCANO_ENABLED` 为 true 时追加校验：
```python
if self.TEMP_AK_VOLCANO_ENABLED:
    ak = self.VOLCANO_ACCESS_KEY or self.TOS_ACCESS_KEY
    if not ak:
        missing.append(("VOLCANO_ACCESS_KEY/TOS_ACCESS_KEY",
                        "火山 TOS 临时凭证方案 B 建号不可用：缺可写 IAM AK"))
```
（`TEMP_AK_TOS_BUCKET_MAP` 留空合法——表单可填真实桶名；不强制。）

---

## 6. 审批：复用同一模板，按 platform 分流（推荐，最省事）

**结论：复用现有发放/延期审批模板同一 code**（`TEMP_AK_APPROVAL_CODE` / `TEMP_AK_EXTEND_APPROVAL_CODE`），不为火山另开模板。理由：

- 发放审批「平台」控件已有**火山云/阿里云两选项**（`_parse_platform` 已识别）；同一模板同一 code，platform 字段天然分流最省事。
- `routes.py:_approval_allowlist` 白名单、`should_handle_event` 严格 code 匹配、#55 实例级 APPROVED 门禁、`_verify_enterprise` 防串、延期+撤销分支——**全部厂商无关，零改复用**。
- 火山申请与阿里走**同一事件流**：`handle_temp_ak_event` → `parse_temp_ak_request`（`_parse_platform` 出 `volcano`）→ `_validate_spec`（放开 volcano）→ `create_grant_record`（落 `platform=volcano`）→ `issue_grant` → `issuer.issue`（派发火山引擎）→ `delivery.deliver`（评论下发，厂商无关）。
- 延期/撤销同理：`handle_temp_ak_extend_event` → load grant（含 platform）→ `extend_grant`/`cleanup.revoke_grant`（内部按 platform 派发）。

**唯一注意**：审批表单「申请目录」火山填 `tos://桶/前缀/`（`_parse_directory` 已认 `tos://`）。若用户在火山申请里误填 `oss://`，`_parse_platform` 靠「平台」控件定性（不靠 scheme），platform=volcano 但 bucket 从 oss:// 剥出——**加一条软校验**（可选）：volcano 平台 + directory 是 `oss://` 时告警"目录 scheme 与平台不符"。列为 §10 待定。

---

## 7. 工具 / CLI / 命名 / 白名单 / 门禁：全复用

- **manage_temp_ak 工具**：`plan`/`status`/`revoke` 复用。`plan` 若要预览火山 policy，加一个 `platform` 参（默认 aliyun）传入 grant → `issuer.plan` 分派。`status`/`revoke` 厂商无关（revoke 走 `cleanup.revoke_grant` 已分派）。**仍无 issue**（杜绝绕审批）。
- **CLI**：`plan` 加 `--platform aliyun|volcano`（默认 aliyun）；`status`/`revoke`/`sweep` 厂商无关零改。dry-run 默认。
- **命名**：user_name 复用 `_derive_user_name`（拼音 slug）；display_name 复用 `display_name_for`（中文）；grant_id 复用 `grant_id_for`（hash 实例）。policy_name 火山用独立前缀 `tempak-tos-auto-`。
- **白名单/门禁**：`routes._approval_allowlist`、#55 实例级 APPROVED、`_verify_enterprise`、NX 幂等锁——全复用零改。
- **下发**：`delivery.deliver`/`credential_text` 复用（火山 `security_token=""` → 文本自动跳过 SecurityToken 行，凭证类型文案走"长期 AccessKey"分支，天然正确）。

---

## 8. 分流边界：本期只做方案 B（火山 STS 作后续）

- 用户拍板**跨天** → 方案 B（长期 AK）。对标阿里现状 `TEMP_AK_STS_MAX_SECONDS=0` → 火山也恒走方案 B（`classify_mode` 在 0 上限下永返 RAM_MODE）。
- research §1 坐实**火山也有 STS**（`AssumeRole` ≤12h，含 SessionToken，支持 inline session policy），与阿里近乎一比一。**作为后续可选变体**：若未来要≤12h 自灭凭证，再补 `issuer_volcano._issue_sts` + `policy_volcano.build_session_policy`（inline policy 交集语义 + role_trn 信任策略待取证，research §6 #2）。**本期不实现**，`build_session_policy` 不写。
- **标注给 dev**：火山引擎的 `issue` 不做 mode 分流（恒 RAM），比阿里 `issuer.issue` 简单——但保留 `mode` 字段 = "ram" 以复用 orchestrator/cards/delivery 的 mode 相关展示（火山 cards `_MODE_CN["ram"]` 已有文案）。

---

## 9. 真机取证遗留（research §6，P1 前置/并行）

| # | 项 | 状态 | 影响 | 处理 |
|---|---|---|---|---|
| §6 #4 | TOS 是否支持 `volc:SourceIp` IP 条件（防泄漏加固） | 未证 | policy_volcano 的可选 IP 锁 | P1 用 research §7 同款探针法（有效 IP 放行/错误 IP 拒）一测。**默认不加 IP 条件**，取证通过后开 `source_ips` 通路（spec 已留 `source_ips` 字段，现恒空） |
| §6 #5 | `DeleteAccessKey` 400 精确语义（须先 Inactive？还是 delete_user 级联即可） | 部分（级联已验删净） | cleanup_volcano 删序 | P1 实测：先 Inactive 后 delete_access_key 是否 200。**不阻塞**——现设计已"软失效 + 吞 400 + delete_user 兜底级联"，安全无损 |
| §6 #2 | 火山 STS inline Policy 交集语义 + role_trn 信任策略 + Policy 字符上限 | 未证 | 仅火山 STS 变体（本期不做） | 后续变体再取证 |
| §6 #7 | 火山 AssumeRole role MaxSessionDuration 1h→12h | 未证 | 仅 STS 变体 | 后续 |
| 补 | `tos:` 动作精确名全集（ListBucketVersions/GetObjectAcl 等是否需要） | research §3 已列主集 | policy_volcano 动作集完整性 | P1 发一组真凭证后按外采方实际报错补动作（探针法） |

---

## 10. 待定项（列给用户 / dev 拍）

- **A. 审批模板**：确认复用同一 `TEMP_AK_APPROVAL_CODE`（platform 表单内分流，§6 推荐）✅ 还是火山另开模板？——**建议复用**，交用户确认。
- **B. TOS 桶映射来源**：`TEMP_AK_TOS_BUCKET_MAP`（新配置，展示名→region/bucket）还是允许表单直填真实桶名？是否复用某处既有 TOS 桶清单？——dev 与用户定 real bucket 值。
- **C. region 回退链**：`TEMP_AK_TOS_REGION` 空时回退 `TOS_REGION`（=cn-shanghai）——确认外采 TOS 桶所在 region（policy trn 不含 region，故仅影响可选数据面校验；但若加 SourceIp/数据面探针需准）。
- **D. IP 条件**：是否给火山也加 `volc:SourceIp` 锁外采方出口 IP（§9 #4 取证后）？需外采方提供固定出口 IP。
- **E. 灰度开关**：`TEMP_AK_VOLCANO_ENABLED` 独立于 `TEMP_AK_ENABLED`（阿里已上线，火山单独灰度）——确认默认 false、真机验收后再 true。
- **F. 分派层位置**：platform 派发放 `issuer.issue`/`plan`（§3.3 推荐，一处覆盖三入口）还是 `orchestrator`？——建议 issuer 层，dev 定。
- **G. scheme/platform 不符软校验**：volcano 平台填 `oss://` 目录是否告警拒（§6）？——建议软告警不硬拒（platform 控件为准），dev 定。

---

## 11. Epic → Task 拆分

> owner 为**建议**，待 dev 确认后落任务板。P0 无真机依赖可先做；P1 需真机火山凭证。

### P0：骨架 + dry-run（无真机，价值高风险低）

- **T1 `policy_volcano.py`**（owner: dev；deps: 无；~M）
  验收：`build_policy_with_window` 出 `{"Statement":[…]}`（无 Version）含 `DateGreaterThan/DateLessThan volc:CurrentTime`(UTC Z) + `trn:tos:::bucket[/prefix/*]` + read/download/write 三正交动作集 + prefix 时 `StringLike:{tos:prefix}`；无 delete 动作；单测覆盖时间格式/前缀/caps 组合/空 caps。
- **T2 `issuer_volcano.py`（dry-run + issue 骨架）**（owner: dev；deps: T1；~M）
  验收：`plan` 出火山 policy 预览不调云；`issue`/`rewrite_window` 组装正确 SDK 请求（mock `volcenginesdkiam`，断言 `create_user`/`create_policy`/`attach_user_policy(Custom)`/`create_access_key`/`update_policy` 参数 + `resp.access_key.secret_access_key` 取值路径）；凭证不落库不入日志。
- **T3 `cleanup_volcano.py`**（owner: dev；deps: 无；~S）
  验收：删序 Inactive→delete_ak(吞400)→detach→delete_policy→delete_user；mock SDK 断言调用序 + `DeleteAccessKeyRequest(access_key_id=)` 参数名；幂等（删已删吞 not-exist）；翻 stage=REVOKED。
- **T4 平台分派接线（6 处）**（owner: dev；deps: T1-T3；~M）
  验收：`approval._validate_spec` 放开 volcano+守卫；`orchestrator.resolve_bucket` TOS 桶映射；`issuer.plan/issue/rewrite_ram_window` + `cleanup.revoke_grant` 顶部 platform 派发；阿里路径**回归零变化**（既有 temp_ak 全套单测仍绿）。
- **T5 settings + `_REQUIRED_FIELDS`**（owner: dev；deps: T4；~S）
  验收：新字段登记 + `TEMP_AK_VOLCANO_ENABLED` 时缺 AK 告警；`print_validate` 输出正确。
- **T6 CLI/工具 platform 参 + cards/delivery 复用验证**（owner: dev；deps: T4；~S）
  验收：CLI `plan --platform volcano` 出火山 policy；工具 plan 分派；delivery 火山凭证文本（无 token、长期 AK 文案）；卡片火山平台展示。
- **T7 tester 全面单测**（owner: tester；deps: T1-T6；~L）
  验收：火山 policy/issuer/cleanup/approval-volcano 分支全覆盖 + 阿里回归全绿；platform 分派边界；凭证不落盘不入日志；全量 pytest 绿。
- **T8 auditor 过闸**（owner: auditor；deps: T7；~S）
  验收：无阻塞——重点查①阿里路径零回归②审批绕过口（工具/CLI 无 issue）③secret 不落 Redis/日志④policy 时间窗+无 delete⑤cleanup 幂等+级联兜底⑥volcano 守卫早于建号。

### P1：真机（发一组火山凭证 + 权限实测 + 到期清理）

- **T9 真机取证补全**（owner: dev/researcher；deps: 无，可与 P0 并行；~M）
  验收：§9 #4（SourceIp 探针）+ #5（DeleteAccessKey Inactive 语义）+ `tos:` 动作全集有结论带证据（研究已做 #1/#3/#6，剩这几项）。
- **T10 真机发一组火山方案 B 凭证**（owner: dev+用户；deps: P0、T9；~M）
  验收：审批通过 → 建 IAM user+AK+时间窗 TOS policy → 评论下发；生效前调用被拒、生效后放行（research §7 传播 30-40s 容忍）；grant 记录完整含 platform/user_name/ak_id/policy_name，无 secret。
- **T11 权限实测（外采方视角）**（owner: dev+用户；deps: T10；~S）
  验收：read 只列不下载、download 能下、write 能传不能删；超桶/超前缀/超时窗被拒；按实际报错补 `tos:` 动作（§9 补）。
- **T12 到期清理真机验证**（owner: dev；deps: T10；~S）
  验收：`_temp_ak_cleanup_loop`/CLI sweep 扫火山到期 grant → cleanup_volcano 删净（独立只读复扫 0 残留，同 research §7 方法）；stage=REVOKED；幂等重跑安全。

---

## 12. 部署

纯 Python，`volcengine-python-sdk` 已装（`requirements.txt:36`，`volcenginesdkiam` 全套 API 已核实在装环境可 import）→ 普通 `deploy.ps1`，**免 rebuild**。上线前置：
- 填 `TEMP_AK_TOS_BUCKET_MAP` 真实 TOS 桶（或确认表单直填）；确认 `VOLCANO_ACCESS_KEY`/`TOS_ACCESS_KEY` 那把 AK 具 IAM 建号 + policy + tos 权限（research §7 已在生产 AK 坐实）。
- `TEMP_AK_VOLCANO_ENABLED=true` 灰度开（P0 阶段保持 false，真机 T10 前再开）。
- 改服务器 `.env` 后 **force-recreate 容器**（restart 不重载 env_file，见 memory deploy_env_reload_gotcha）。
