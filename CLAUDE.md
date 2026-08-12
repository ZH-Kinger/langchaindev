# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIOps 智能运维助手 — a multi-mode AI operations platform built on LangChain 0.3. Five runtime modes (RAG, single agent, hybrid edge→cloud, multi-agent collab, Feishu Webhook bot) sit on top of a tool layer that wraps Aliyun (PAI DSW / ECS / OSS / SLS / RAM / NAS-CPFS / Prometheus), 火山引擎 (TOS / vePFS / IAM), Jira, GitHub, K8s, and a local ChromaDB knowledge base. Per-user Aliyun calls go through STS AssumeRole so the bot never uses a shared AK.

Two families dominate the code base beyond the agent itself: **六条数据搬运链**（跨云迁移 / 同云桶间迁移 / CPFS 预热沉降 / vePFS 预热沉降 / PFS 跨云直传 / SSH 三跳迁移链，统一「飞书卡片 + Agent 工具 + CLI」三入口 + 后台状态机 + 对账兜底）与 **审批驱动的凭证发放**（RAM/IAM 子账号建号、临时 AK/SK 外采凭证，均以飞书审批实例级 `APPROVED` 为唯一门禁）。

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
python ingest.py              # build ChromaDB knowledge base (run once)
```

Config self-check runs at startup (`config/settings.py::Config.print_validate`) and logs missing required fields with their impact — read its output before debugging mysterious failures.

Generate the Fernet key for user-AK encryption:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste into BOT_CREDS_ENCRYPTION_KEY in .env
```

## Running the Application

```bash
python main.py                  # interactive mode selector (default: agent)
python main.py --mode rag       # RAG Q&A (use --session <id> for memory isolation)
python main.py --mode agent     # single agent with three-tier tool routing
python main.py --mode hybrid    # edge (Qwen3-4B) perception → cloud (Qwen-Max) decision
python main.py --mode collab    # diagnostic expert → ops officer
python main.py --mode bot       # Feishu Webhook + DSW scheduler on :8088
```

## Tests

Tests use pytest. Configuration in `pytest.ini`: integration tests are skipped by default via `-m "not integration"`.

```bash
pytest                                # unit + tool tests (default, no external services)
pytest -m integration                 # only integration tests (need real Grafana/Jira/LLM/飞书)
pytest -m ""                          # everything including integration
pytest tests/unit/test_aliyun_sts.py  # a single file
pytest -k test_router                 # match by test name
```

`tests/conftest.py` autoloads fixtures that replace Redis with `fakeredis`, inject a test Fernet key, and provide opt-in mocks for STS, RAM API, and Feishu sending. No lint or type-check is configured.

## Deployment

```powershell
.\deploy.ps1            # 部署当前 HEAD 到 bot-server
.\deploy.ps1 -Ref main  # 部署指定 ref
```

`deploy.ps1` syncs code to the server and restarts — no image rebuild. The server's `/root/langchaindev` is **not** a git repo; code is bind-mounted into the `aiops-bot` container via docker-compose (`.:/app`), so deploy = `git archive HEAD` → scp → 解压覆盖 → 清理已删文件/`__pycache__` → `docker compose restart bot` → poll `/health` (~10s). It tracks `.deployed_commit` on the server to delete files removed since last deploy (rsync-`--delete` equivalent). Only `requirements.txt` changes need a manual `docker compose up -d --build` (the script detects and warns).

## Architecture

### Modes and Entry Points

`main.py` dispatches via `RUNNERS`:

| Mode | Entry | Notes |
|------|-------|-------|
| `rag` | `core/rag_runner.py` | Session memory via `FileChatMessageHistory` (file-backed, not Redis) |
| `agent` | `core/agent.py` | Three-tier tool routing, Redis session memory, streaming |
| `hybrid` | `core/hybrid_agents.py` | Edge model emits JSON observation → cloud model decides |
| `collab` | `core/multi_agent_system.py` | Diagnostic agent (read-only tools) → Ops agent (write tools) |
| `bot` | `core/feishu_bot/` | Flask `/feishu/event`, starts `core/dsw_scheduler` background threads |

### LLM Factory (`core/llm_factory.py`)

`get_cloud_llm(temperature, streaming)` and `get_edge_llm(temperature)` are `@lru_cache`'d — same args = same instance. Call `clear_llm_cache()` after mutating `settings.MODEL_NAME` at runtime (see `core/agent._switch_model`).

### Tool System (`tools/`)

Subpackaged by vendor / domain. Single source of truth is `tools/__init__.py`:

- `ALL_TOOLS` — flat list registered with the agent.
- `TOOL_GROUPS` — `{group_name: set[tool_name]}` used by the agent router. **Validated at import time**: any name in `TOOL_GROUPS` not present in `ALL_TOOLS` raises `ValueError` — silent routing breakage is impossible.

```
tools/
  temp_ak_issuance/  manage_temp_ak       # 临时凭证 plan/status/revoke（**无 issue**，发凭证只走审批）
  pfs_transfer/      manage_pfs_transfer  # vePFS↔CPFS 三段直传 plan/apply/status（apply 需管理员）
  cpfs/     manage_cpfs_dataflow          # 阿里 CPFS/NAS 预热/沉降 list|discover|preheat|sink|status
  transfer/ manage_transfer               # 跨云迁移 plan/apply/status
  aliyun/   pai_dsw, ecs, oss, sls, ram, prometheus, gpu_advisor,
            gpu_training_advisor, cluster_health, cluster_mfu, dsw_inspector
            # oss also: dir_sizes (各子目录大小) + tree (目录结构), 自动探测地域
            # cluster_mfu: 多区域算力效率(MFU)+容量+调度 日报，交互式飞书卡片(区域切换)，
            #   24h Redis 快照，按钮回调只读缓存秒回
            # gpu_distribution: 卡分布快照数据源 —— 不是 agent 工具，供 /gpu/distribution 页面+摘要卡
  volcano/  tos, vepfs_dataflow  # TOS 容量盘点 + 火山 vePFS 预热/沉降 (静态 AK，无 STS)
  feishu/   notify, cards       # notify: message cards, GPU progress bars, token cache
                                # cards: card-dict primitives (card/div/fields/btn/...) shared by
                                #   all card builders — NOT an agent tool, never export in __init__
  jira/     ticket, workflow    # GPU ticket CRUD + algo workflow query
  github/   workflow            # PR / commit / sprint activity
  knowledge/rag                 # ChromaDB retriever, lazy-loaded
  ops/      system, analysis, monitor, k8s    # psutil, pandas dedup, k8s restart
```

SSH 迁移链（`core/ssh_transfer`）与同云桶间迁移（`core/bucket_transfer`）**只有飞书卡片 + CLI 两个入口，没有 Agent 工具**。

`BaseOpsTool` (`tools/base_tool.py`) 提供 data-path 解析与一个 `log_operation` 日志方法。**注意它不是安全边界**：只有 `tools/ops/analysis.py` 与 `tools/ops/system.py` 各建了一个实例当路径工具用，基类里**没有任何鉴权、二次确认或审计落盘**，写工具也没有调用 `log_operation`。别以为"继承了基类就有兜底"。

### Agent Tool Routing (`core/agent.py::_select_tools`)

Three tiers, in order:

1. **Fast path** — if input contains an unambiguous proper noun (`dsw`, `jira`, `ecs`, `oss`, `sls`, `k8s`, `pod`, `prometheus`...) and no knowledge-override token (`知识库`, `文档`, `手册`...), go straight to the legacy keyword router. Zero latency, zero cost.
2. **LLM router** (`core/intent_router.py`) — `cloud_llm` at temperature 0 picks 1–2 intents from `INTENT_DESCRIPTIONS`. `@lru_cache(256)` on input text. Output is parsed with tolerance for markdown wrapping; unknown intents are filtered. Empty result → fall through.
3. **Legacy keyword fallback** — long `elif` chain in `_legacy_keyword_route`; `knowledge` is the final branch to avoid `怎么/如何` hijacking other intents.

`INTENT_DESCRIPTIONS` keys must match `TOOL_GROUPS` keys; the intent router prompt is built dynamically from these descriptions.

### Aliyun STS Multi-Tenant Credentials

> ⚠️ **下面描述的是设计目标，生效有前提**：本节的多租户隔离要求 `ALIYUN_BOT_MASTER_AK_ID/SECRET` 与 `ALIYUN_BOT_ROLE_MAPPING`（或 `ALIYUN_BOT_ROLE_DEFAULT`）都已配置；任一缺失则 `_do_assume_role` 恒返 None。此外 `aliyun_client_factory` 在 STS 失败时会回落全局 AK 且当前无开关可关 —— 所以部署新环境时**务必先确认这几项已配好并观察日志里没有降级告警**，否则「不使用共享 AK」这个前提并不成立。

The bot never calls Aliyun with a global AK for user-initiated actions. Flow:

1. User triggers a cloud action in Feishu → handler receives `open_id`.
2. `utils/aliyun_sts.assume_role_for_user(open_id)` resolves the user's role ARN:
   - Look up RAM user by Feishu `open_id` (`tools/aliyun/ram.get_ram_user_by_open_id`).
   - `ListGroupsForUser` → match first group in `ALIYUN_BOT_ROLE_MAPPING` JSON → use that role ARN.
   - Fallback: `ALIYUN_BOT_ROLE_DEFAULT`.
3. Master AK (`ALIYUN_BOT_MASTER_AK_*`, only `STSAssumeRoleAccess + RAMReadOnlyAccess`) calls `sts:AssumeRole`. Temp credentials cached in Redis under `aliyun:sts:{open_id}:{role_arn}`, auto-refreshed 5 min before expiry.
4. `utils/aliyun_client_factory.get_*_client(open_id)` returns an SDK client with the temp creds injected. **All cloud API calls must go through this factory** — do not read `settings.PAI_DSW_ACCESS_KEY_*` directly outside the factory's legacy fallback path.

User-supplied AKs (from Feishu binding cards) are encrypted with Fernet (`utils/crypto.py`) before going into Redis. `encrypt_strict()` raises `CryptoNotConfigured` when the key is missing — use it for any write path. `decrypt()` transparently passes through legacy plaintext to keep historical bindings working.

### DSW Scheduler (`core/dsw_scheduler.py`)

Background threads start with `bot` mode:
- **Ticket poller** (every 2 min): scans Jira project `JIRA_PROJECT_KEY` for new GPU tickets → creates DSW instance via `manage_pai_dsw` → records state in Redis `dsw:ticket:{key}`.
- **Idle/timeout watcher** (every 5 min): for tracked instances, sends a GPU-idle nudge when utilization stays below `GPU_IDLE_THRESHOLD_PCT`. The duration-based **auto-stop** (15-min "about to expire" warning card + auto-stop after `DSW_IDLE_STOP_MINUTES` of no response) is **disabled by default** (`DSW_IDLE_STOP_ENABLED=false`) — utilization-based shutdown is delegated to Aliyun workspace's built-in setting, not reinvented here. Set `DSW_IDLE_STOP_ENABLED=true` to restore it. The GPU-idle nudge is unaffected by the flag.
- **Capacity monitor** (`core/capacity_monitor.py`, opt-in via `CAPACITY_MONITOR_ENABLED`): every `CAPACITY_MONITOR_INTERVAL_HOURS`, scans each `CAPACITY_MONITOR_TARGETS` entry (OSS via `tools.aliyun.oss.compute_dir_sizes`, TOS via `tools.volcano.tos.compute_dir_sizes`), pushes a Feishu card with per-subdir sizes + delta-since-last-scan, reds the header if total exceeds `CAPACITY_ALERT_THRESHOLD_TB`. Snapshot in Redis `capacity:snapshot:{vendor}:{bucket}:{prefix}`. Also upserts per-vendor totals into a Feishu Bitable via `core/capacity_bitable.write_scan` (三表关联：巡检快照→厂家总量→批次明细，每次 upsert 去重旧行).
- **Morning report** (daily 北京 9:00): pushes two cards — per-user DSW instance summary + cluster MFU summary (`tools.aliyun.cluster_mfu.build_mfu_card(refresh=True)`, which also warms the snapshot cache for the card's region-switch buttons). Skipped if `PROMETHEUS_URL`/`FEISHU_CHAT_ID` unset.
- **Dataflow reconcile** (`_dataflow_reconcile_loop`, every 2 min): 在途任务对账，**六条搬运链**（transfer / bucket / cpfs / vepfs / ssh / pfs）各按 `_dataflow_reconcile_specs()` 的契约 refresh。后台轮询线程随容器重启而死，对账线程随容器复活兜底 → 重启后任务完成也会补推结果卡；与在线推送共用 `dataflow:notified:{job_id}` NX 闸门，只推一次。
- **Dataset dashboard** (`_dataset_dashboard_loop`, opt-in): 维护飞书「数据集大盘」多维表格现有行（按 uri 扫对象存储回填脚本负责列），不改表结构、不碰人工分析列。
- **Temp-AK cleanup** (`_temp_ak_cleanup_loop`, opt-in via `TEMP_AK_ENABLED` + `TEMP_AK_CLEANUP_ENABLED`, daily 北京 `TEMP_AK_CLEANUP_HOUR`:35): 逐个账号档案扫 `<前缀>grant:*`，对已到期的方案 B 凭证硬删 AK+policy+user。
- **OSS perm audit push** (`core/dsw_scheduler._oss_perm_loop`, opt-in via `OSS_PERM_PUSH_ENABLED`, daily 北京 `OSS_PERM_PUSH_HOUR`:20): see OSS Permission Sync below.

### 临时 AK/SK 发放 — 外部数据外采 / 产线访问 (`core/temp_ak_issuance/`)

飞书审批通过后，为**外部方 / 使用人**发一组时限 OSS 凭证。加法式独立包，复用 `ram_approval` 审批门 + `oss_perm.permsync` policy 生成 + `aliyun_sts`，**零改现有引擎**。已上线 bot-new，覆盖**两个阿里云主账号**（仅阿里 OSS；火山 TOS 见末尾）。

**多阿里云主账号 = 账号档案注册制 (`accounts.py`)**

- 第二主账号 UID `1339279783371949`，短标识 **`1949`**。`AccountProfile`（frozen dataclass）+ 注册表：`profiles()`（**刻意不缓存**——settings 在测试里会被 monkeypatch，缓存会让用例互相污染）/ `default()` / `by_slug()`（未注册档案**抛 `UnknownAccountError`**，绝不静默退回默认档，否则会拿错账号的 AK 去动另一个账号的用户）/ `by_issue_code()`（发放事件按审批 code 定档）/ `by_grant_id()`（**共用的延长/撤销审批按凭证ID 前缀分派**，最长前缀优先）/ `issue_codes()`（供 routes 白名单）/ `bucket_map()` / `chat_id_for()` / `subject_label_for()` / `assert_account_consistent()` / `ram_client()`。
- **为什么是注册制、不是复制一份代码**（架构决策）：延长/撤销审批被两账号**共用同一个 definitionCode**，而 routes 里三个 `should_handle_*` 都是严格等值 → 同一 code 只能有一个处理器认领，两份独立副本必然一个抢到、另一个永远收不到。所以延长/撤销必须由**单一处理器按「凭证ID → grant → 账号」分派** → grant 必须带账号维度 → 档案注册制。副作用红利：实例级 APPROVED 硬化门不用复制第三份。
- **注册条件是「任一存在」**：`TEMP_AK_1949_APPROVAL_CODE` / `ALIYUN_1949_ACCESS_KEY_ID` / `..._SECRET` 三项任一存在即注册档案，AK 缺失只在 `ram_client()` 显式抛错。刻意如此：若因缺 AK 就不注册，该账号已有 grant 会**静默失效**（`sweep_expired` 扫不到 `temp_ak_1949:grant:*`、`_key()` 又按默认档拼成 `temp_ak:` 读不到）→ 已发出去的长期 AK 和 RAM 子号永久残留云上、运维毫无察觉。发放路由不受影响：code 为空时 `by_issue_code`/`issue_codes` 本就过滤掉。
- **隔离维度**：凭证（各账号自己的 RAM 可写 AK，**显式传参**）/ Redis（`temp_ak:` vs `temp_ak_1949:`）/ 凭证ID（`tak-` vs `tak1949-`）/ RAM 登录名（`tempak-` vs `tempak-1949-`）/ 显示名后缀（`-临时外采用户` vs `-1949产线临时用户`）/ 桶映射表 / 审批表单字段映射 / 内部回执群。默认档 slug `""`，取值与档案机制引入前**逐字一致**（历史数据无账号标识）。

**安全硬门（勿放宽）**

- `issuer.classify_mode(expire, now, profile)` 对**非默认账号一律返回 RAM，绝不 STS**：STS 分支用的是默认账号的 Master AK + 默认账号的宽 OSS 角色 `TEMP_AK_OSS_ROLE_ARN`，让别的账号走这条 = 把 A 账号的数据权限发给 B 账号的申请人（申请人填一个 A 账号的桶即可）。且 `TEMP_AK_STS_MAX_SECONDS` **代码默认值是 43200（危险侧）**，不能只依赖线上 `.env` 写 0。
- `_issue_sts` 入口再拦一次（纵深）：先过 `accounts.assert_account_consistent(grant)`，**双源校验**（`grant["account"]` + 凭证ID 前缀）——只看 `account` 的话，「account 被抹掉 + 凭证ID 仍是 `tak1949-`」的畸形记录能过门（auditor 实测复现过），与 RAM 路径不对称。
- `accounts.assert_account_consistent(grant)`：两套账号真相源背离即抛。防「用 A 的 AK 删 B 的号 → `EntityNotExist` 被吞成日志 → 状态却被置成已撤销」的假成功。四条取凭证路径（issue / rewrite / revoke / sweep）全按 grant 分派。
- **绝不要设 `ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET` 环境变量**：`permsync.make_ram_client()` 的零参路径优先读它，一旦设了**所有账号的建号请求都被劫持到同一个账号**，分账号配置形同虚设且失败得很隐蔽。`settings.print_validate()` 已加检测告警（多账号启用时按 error 级），并会打印「临时 AK 已注册账号档案：`['(默认)', '1949']`」——**这行是 `.env` 改动后 force-recreate 是否生效的客观依据**（`docker compose restart` 不重载 `env_file`）。
- `permsync.make_ram_client(ak, sk)` 支持显式传参；**只给一半会抛错**，不再静默回落到全局凭证。
- `cleanup.sweep_expired` 的 try 按 **per-grant** 包，不是按账号包住整个循环——否则一条脏记录会中断该账号剩余全部清理，且每轮都断 → 到期号永久清不掉。

**审批模板与分派**

- 默认档「数据外采访问凭证申请」`5B4A3105-…`，5 控件：平台（火山云/阿里云，火山暂拒）/ 使用企业名称 / 权限设置 / DateInterval（生效+到期）/ 申请目录。
- 1949 档「数据访问凭证申请（产线）」`0133C4FC-8793-4FF3-A759-C4ECE8AC1FF9`，5 控件：使用人名称 / 权限设置（read/write/download）/ DateInterval / 访问目录 / 备注。**无「平台」单选**（`has_platform_field=False` → 恒阿里云）；主体是**使用人名称**而非企业名（`subject_is_person=True` → 延期防串校验要求**精确相等**，因为包含匹配下「张三」会通过「张三丰」）；备注进凭证评论前把 `\s+` 压成单空格（留着换行就能伪造一整行假 `AccessKey Secret：`）。widget id 已由飞书 API 实拉写死，无需真机试错。
- **延长/撤销复用现有那条** `E9333E62-…`「访问凭证延长/撤销 申请」，第二账号**不需要**另建；按凭证ID 前缀分派到对应账号。
- **已知约束**：共用模板 ⇒ 1949 的审批人也能批准默认账号凭证的延长/撤销（只能延长既有范围、不能扩权）。这是飞书审批权限层面的事，代码侧无法拆；要拆需在飞书为第二账号另建一条延长/撤销模板。
- **前置（飞书侧一次性，每个审批定义都要）**：`POST /approval/v4/approvals/{code}/subscribe` **订阅**，不订阅 bot 收不到任何事件（日志里连一条 `approval_event` 都没有）。对已订阅的 code 重复调返回 `1390007 subscription existed`，可用来探测订阅状态。评论作者 open_id 须是本 app 下的。

**发放流程**

- **按有效期分流** (`issuer.classify_mode`)：`到期−now ≤ TEMP_AK_STS_MAX_SECONDS`（默认 12h/43200s，硬顶 43200）→ **STS 单发**（含 SecurityToken、到点自灭，`aliyun_sts.assume_role_with_policy` 现场 session policy 收窄）；超出 → **方案 B**（RAM 子用户 + 长期 AK + policy 内嵌时间窗 + 到期硬删）。**设 `TEMP_AK_STS_MAX_SECONDS=0` 即全走方案 B、不需 STS 宽角色**（当前线上即此配置）。非默认账号恒方案 B（见安全硬门）。
- **权限模型 read/download/write 三者正交** (`policy.build_policy_with_window(bucket,*,prefix,caps,not_before,expire,source_ips)`)：`read`→只 `ListObjects`（不含下载）；`download`→`GetObject`（才给下载）；`write`→`PutObject/AbortMultipartUpload/ListParts`（**无任何 delete**）。桶信息动作（`GetBucketInfo/Stat/Acl`）**单独成一条无 Condition 的语句**——混进带 `oss:Prefix` 的 List 语句会被条件卡死（桶级请求不带 prefix），线上「拿了凭证访问不了桶」就是这个。其余每条 statement 叠时间窗 `DateGreaterThan`(生效)/`DateLessThan`(到期)（`acs:CurrentTime` ISO8601 +08:00，AND）——服务端逐调用判时间，泄漏也随到期自动失效。`build_session_policy` 供 STS，≤2048 字符硬校验。真机实测：下载/删除/越界前缀全 403。
- **审批门禁** (`approval.handle_temp_ak_event`)：复用 `ram_approval` 硬化门——回拉实例详情、只认**实例级 `status=="APPROVED"`**（整单全批）才发、无 `instance_code` fail-safe 不发；`should_handle_event` 精确匹配已注册档案的发放 code。
- **延长/撤销** (`handle_temp_ak_extend_event`)：凭证ID + 撤销·延长单选 + 使用方信息（防串，`_verify_enterprise` fail-safe：原凭证有主体名而表单留空 → 拒）+ DateInterval。**延长**：方案 B 改写 policy 时间窗（同 AK 不重发）、STS 重签发（新窗 >12h 自动转方案 B）；**撤销**：`cleanup.revoke_grant`（删 AK+policy+user）。`extend_instances`/`revoke_instances` 幂等。
- **命名** (`orchestrator._ascii_slug`/`display_name_for`)：RAM 登录名 `<账号前缀><主体名 ASCII/拼音，否则 ext>-<6hex>`（中文经 pypinyin，未装则降级）；显示名 `<主体>-<档案后缀>`（可中文，控制台可辨）。
- **下发**：凭证（含 secret）作为**审批评论**贴到审批实例（`delivery`，复用 `ram_approval._send_approval_comment`），评论身份 = `ram_approval._approval_comment_user_id()`（管理员，与 RAM 建号审批一致）；凭证ID 也写进评论供延长/撤销回填。**secret/token 只在评论正文出现一次，绝不落 Redis/日志/内部群卡**（grant 记录只存 `ak_id`）。
- **凭证正文带连接信息三行**（地域 / 外网 Endpoint / 桶域名）：深圳的桶用杭州 endpoint 会被 OSS 回 403 `must be addressed using the specified endpoint`，使用方会以为凭证无效。拼装要**归一两套地域写法**——`permsync.BUCKET_MAP` 存的带 `oss-` 前缀、`TEMP_AK_*_BUCKET_MAP` 存裸 region，不判重会拼出 `oss-oss-ap-southeast-1.aliyuncs.com`（`_access_lines` 按 `region.startswith("oss-")` 处理）。`orchestrator.resolve_bucket` 还支持**按真实桶名反查**地域（申请人常直接照占位符填真实桶名，查不到则三行退化成「未知」，首单已踩）。
- **到期硬删**：`dsw_scheduler._temp_ak_cleanup_loop` 每日北京 `TEMP_AK_CLEANUP_HOUR`:35 逐档扫 `<档案前缀>grant:*`，对 ISSUED 且 `expire<now` 的方案 B grant 走 `cleanup.revoke_grant`。
- **建号凭证**：STS 用 Master AK；方案 B 用 `accounts.ram_client(profile)` → `permsync.make_ram_client(ak=, sk=)`（该档自己的 RAM 可写 AK）。**发凭证唯一路径 = 审批**——Agent 工具 `manage_temp_ak`（`plan`/`status`/`revoke`，无 `issue`）+ CLI 都不能绕审批发凭证。
- **全局审批白名单** (`routes._approval_allowlist`)：bot 只处理 RAM 建号 code + 各档发放 code + 延期 code，其余（含无 code 的组织审批如补卡）一律记日志丢弃，早于所有审批处理器。
- **火山 TOS 临时凭证**：`issuer_volcano`/`policy_volcano`/`cleanup_volcano` 骨架已在包内（长期 IAM AK + 时间条件 policy），但 `approval._validate_spec` 仍硬拒 `platform=volcano`、`TEMP_AK_VOLCANO_ENABLED` 未接线 → **当前不可用，P1**。
- **Config**: `TEMP_AK_ENABLED`, `TEMP_AK_APPROVAL_CODE`, `TEMP_AK_EXTEND_APPROVAL_CODE`, `TEMP_AK_STS_MAX_SECONDS`(=0 全方案B), `TEMP_AK_OSS_ROLE_ARN`（仅 STS，`>0` 时才校验）, `TEMP_AK_BUCKET_MAP`, `TEMP_AK_CLEANUP_ENABLED/HOUR`, `TEMP_AK_CHAT_ID`, `TEMP_AK_FIELD_*`；第二账号 `ALIYUN_1949_ACCESS_KEY_ID/SECRET`, `TEMP_AK_1949_APPROVAL_CODE`, `TEMP_AK_1949_BUCKET_MAP`, `TEMP_AK_1949_CHAT_ID`, `TEMP_AK_1949_COMMENT_USER_ID`, `TEMP_AK_1949_FIELD_*`。凭证下发身份默认复用 `ADMIN_FEISHU_OPEN_ID`。

### RAM / IAM 子账号审批建号 (`core/ram_approval.py`)

飞书审批「RAM 子账号申请」通过 → 建阿里云 RAM 用户和/或火山引擎 IAM 用户（表单「平台」可多选）：建号 → 开控制台登录 → 入组 → 建 AK，凭证走审批评论下发。只读查询侧在 `core/ram_query.py`（阿里）+ `core/volcano_iam_query.py`（火山镜像，只读、不返密钥）+ `core/ram_query_cards.py`。

- **建号门禁（安全核心）**：只认**回拉实例详情的顶层实例级 `status == "APPROVED"`** 单值 —— 不用 `APPROVED_STATUSES`（`PASS`/`AGREE`/`DONE` 等是节点级/timeline 词，用它会让第一级审批人一点同意就建号、绕过第二级）；无 `instance_code` → fail-safe 不建。事件里的 status 只作早期过滤。
- **失败不再重复评论**（两层机制，各管一层）：
  - **终态标记**：`save_approval_failure` 写 `error_terminal = _is_terminal_error(exc)`，`_is_instance_done` 除 `success/dry_run` 外也认「failed + terminal」→ 后续事件在入口短路，连云 API 都不再打。`_TERMINAL_ERROR_MARKERS` **刻意只留四个窄词**（`invalidpassword` / `password policy` / `密码不符合` / `invalidloginname`）：曾放过 `invalidparameter`/`malformed`，但宽泛子串会把瞬时错误（网关 5xx、限流）永久标成终态、失去自愈。判不准一律按可重试。
  - **播报闸门**：`_claim_failure_notice(instance_code, text)` 键含错误文本 md5 签名（`ram_approval:instance:failnotice:{inst}:{sig}`，7 天 TTL）→ 同实例同错误只评论一次；换了错误内容仍播报（有信息量）；Redis 不可用**放行**（宁可重复也别漏报失败——漏报会让人以为审批成功了）。它覆盖所有失败，是瞬时错误重试那层的承重件。
- **密码策略失败提示**（`_humanize_error`）交代清楚：密码是建号链最后一步，撞策略时**子用户往往已经建出来了**，但它未入组、无权限、无 AK；用**相同登录名**重新提交会命中 `get_user` 续做、不会重复建号；**勿手动删除或改名**（那才会留下永久空壳）。
- **Redis**：`ram_approval:instance:{code}`（处理记录）、`ram_approval:lock:{code}`（600s NX，防并发/重投）、`ram_approval:instances`（zset 索引）、上面那把 failnotice 闸门。
- **已知遗留（云侧，记账未修）**：火山 IAM 可能残留空壳用户；volcano `get_login_profile` 对无登录配置的用户返回**全零 stub 而非 NotExist** → 重提审批可能误判「已存在」而跳过创建，结果账号建成却没有控制台密码、还被判成功（比重复评论更危险，修它需先真机确认该 API 语义）。

### PFS 直传 — vePFS↔CPFS 跨云 (`core/pfs_transfer/`)

两朵云的并行文件系统之间**物理无直连**，所以是 3 段链编排（第 6 条搬运链）：`源 PFS --沉降--> 源云对象存储 --跨云--> 目的云对象存储 --预热--> 目的 PFS`。加法式新包，复用 `vepfs_dataflow` / `transfer` / `cpfs_dataflow` 三个现成 orchestrator 的 `run_to_completion`，**零改现有引擎**。

- 状态机 `NEW→SINKING→CROSSING→PREHEATING→DONE|FAILED`，Redis `pfs:transfer:job:{job_id}`（30 天 TTL），job 前缀 `xpfs-`，当天幂等。**段级「跳过已成功段」续跑**：链 job 记 `sink_done`/`cross_done`/`preheat_done` + 三个子 job_id；retry/重启只重跑未完成段。
- **方向由源 scheme 定**：`vepfs→cpfs` = P1（三段引擎均已真机验过，先做）；`cpfs→vepfs` = P2（跨云段 `OSS→TOS` 未真机验 + 火山 DMS 只到桶级，**前置阻塞、未开工**）。同云 / 含对象存储 scheme / 方向不匹配一律拒。
- **`paths.py`**：显式 `vepfs://<fs-id>/<子目录>/`、`cpfs://<fs-id>/<子目录>/`，或裸挂载路径 `/vepfs/...`、`/cpfs/...`（用默认 fs）。子目录每级过白名单（禁 `..`/空格/shell 元字符）。两个中转 staging 从 `PFS_STAGING_MAP` 推导（key `<pfs-scheme>://<fs-id>`，value `{region, tos_bucket|oss_bucket, tos_prefix|oss_prefix[, dataflow_id]}`）；chain 级 staging 前缀带 `chain_id` 隔离。
- **审批门（两处都收紧）**：卡片 `confirm` + Agent 工具 `apply` 双路**非管理员一律需管理员确认**（用户自报量只作显示）；`_resume_async` 有 **`launched` 守卫** —— 否则「查询进度」触发 `refresh` 就能把一个从未确认的任务跑起来（绕过审批门）。
- **三入口**：飞书向导卡（`messages._is_pfs_transfer_intent`，同时要求 vepfs+cpfs 判据，**排在 ssh/transfer 意图之前**）+ Agent 工具 `manage_pfs_transfer` + CLI `python -m core.pfs_transfer.cli`。已进调度器对账（6 条链齐）。
- **线上配置现状（`PFS_TRANSFER_ENABLED=true`，但该链路尚未真机验证过）**：`PFS_STAGING_MAP` 配了两个 fs —— `vepfs://vepfs-cnshef4f4b647664`（`cn-shanghai`，staging `tos://data-tran`）与 `cpfs://bmcpfs-00000ub3ici1dnniit2i0`（`cn-hangzhou`，staging `oss://wuji-data-tran`）。**硬约束：每个 PFS 与它的 staging 桶必须同地域**（vePFS↔TOS 同区、CPFS↔OSS 同区；只有跨云段②可跨区）。预热段的目标目录须落在该 CPFS 上**已有 DataFlow 绑定**的目录之下——绑 `wuji-data-tran` 的三条是 `/wangyuran/`、`/chenrankou/`、`/zch/datasets/`（否则要靠 CPFS 那边的临建临删，前置更多，见 CPFS 节）。
- **Config**：`PFS_TRANSFER_ENABLED`、`PFS_STAGING_MAP`、`PFS_TRANSFER_APPROVAL_TB`（默认 1TB）、`PFS_TRANSFER_STAGING_CLEANUP`、`PFS_TRANSFER_CHAT_ID`。

### SSH 迁移链 — 杭州 OSS → 新加坡 → 泰国 (`core/ssh_transfer/`)

给泰国 H200 机房送数据，两段：**段1** `ossutil cp` 杭州 OSS → SGP 上的 ossfs2 挂载盘 `/mnt/sgp_oss`（背后是新加坡桶 `SGP_OSS_BUCKET`=`wuji-sing`）；**段2** 泰国服务器 `ossutil` **直连新加坡 OSS 拉取**。状态机 `NEW→STAGE1→STAGE2→DONE|FAILED`，Redis `ssh:transfer:job:{job_id}`（30 天 TTL），job 前缀 `sgp-`。

**段2 为什么是「泰国直拉」而不是「SGP rsync 推」**（2026-07-31 换的架构，同一份 19.5 TiB 实测）：SGP→泰国 rsync 单流 27 MB/s、8 流并行 78 MB/s；泰国 ossutil 直拉新加坡桶 **289 MB/s**（快 10 倍）。瓶颈是「SGP 那一跳」本身，不是带宽包（SGP 出口上限实测 768 MB/s）。改后还**完全不占 SGP 带宽、少一次中转拷贝**。旧的并行 rsync 引擎保留为回滚路径（`SSH_STAGE2_MODE=rsync`）。
**段1 不能砍**：杭州出境限速，泰国直拉杭州慢（泰国→杭州 RTT 83ms vs →新加坡 31ms），中转桶这一跳有价值。

- **段2 引擎 `engine_ossutil.py`**：控制面 bot→SGP→泰国**双跳 SSH**（复用现有 `SGP_SSH_KEY_ENC`，**零新增凭证**——bot 本来没有到泰国的 key，SGP 上早已配好免密）；数据面泰国↔新加坡 OSS 直连、**不经 SGP**。marker（pid/rc/log）落在**泰国** `$HOME/.ossutil_jobs/<job_id>/`。
  - **双跳的内层脚本一律 base64 传递**：bot 拼串→SGP shell→泰国 shell 三层解析，手写多层引号必崩（开发时踩了两次：变量被吃掉、`unexpected EOF`）。base64 只含 `A-Za-z0-9+/=`，外层留一层双引号即可。**禁止再手写嵌套引号。**
  - **ossutil flag 已在泰国 2.3.0 上逐个核实**（查 `cp --help`，不猜）：`-j/--job`（**默认仅 3，必须显式给**）、`--parallel`、`-u/--update`、`--checkpoint-dir`、`-e`、`--region`、`-f`（detached 跑必须给，否则交互提示会永久挂住）；**`--jobs` 不存在**。`-u` 语义 = 跳过「已存在**且比源更新**」，**mtime 相等不跳过**（会重下）。`--job` 16→32 只多 2%，已近饱和，别再往上调。
  - `stage_progress` **刻意不返回 pct**：ossutil 在 Scanning 阶段的百分比分母是「已扫到的量」、会虚高，而 `progress_line` 优先用 `job["pct"]` → 会把假百分比钉在卡片上。留 None 让上层用 `bytes_done/bytes_total` 算真值；`bytes_done` 含 `skipped:`（续跑不低估）。
- **引擎分派按 job 记录、不按当前配置**（`orchestrator.stage2_mode_of`）：`job["stage2_mode"]` → `job["handoff"]["engine"]`（人工接管标记）→ **回退 rsync**。绝不默认 ossutil，否则本次改动前建的所有 job 都会被拿错探针查 marker → 误判「进程异常退出」。任务跑起来后有人改 `SSH_STAGE2_MODE` 同理会拿错引擎。
- **端到端校验 `verify.py`（段2 报成功后必跑，不过则整链 FAILED）**：**不采信传输器的「Success」**——21 TB 迁移最坏的结局不是失败，而是「报成功但少数据」，几个月后训练读到坏文件才发现、那时源可能已清理。四层：L1 源对象是否全部覆盖 / L2 字节总量（**只统计源 key 集合内**的目的字节）/ L3 逐文件字节（L1/L2 会被「多一个少一个刚好抵消」骗过，L3 不会）/ L4 抽样从 OSS 重下 + `cmp` 逐字节。**源清单在 bot 侧列、目的清单在泰国侧列，互不采信对方汇总数。**
  - **fail-closed**：校验不过、或校验本身崩掉，都判 FAILED——「不知道对不对」必须当「不对」。
  - **判据是「源的每个对象都在且字节一致」，不是两边数量/总量相等**：目的目录是共享数据盘，有历史/别人的文件是正常的，拿数量相等当判据会让正常情况报失败、运维就学会忽略校验结果了。`extra` 只报不判失败。
  - **区分「数据问题」与「校验环境问题」**：样本全取样失败（凭证过期 / `/tmp` 放不下 blob）→ `env_issue=True`、文案写「校验环境问题（非数据不一致）」。都判不通过，但不能让人拿着「数据不一致」去重传 21 TB。
  - **并发闸门 `ssh:transfer:verify:{job_id}`（NX, TTL 2h）**：一趟校验几分钟到几十分钟、期间不刷 `updated_ts`，180s 后对账就判失联→再跑一次；多份 job dict last-write-wins **可能把 FAILED 覆盖成 DONE**（真 fail-open）。抢不到锁 → 保持 STAGE2、不给任何结论。
  - 远端清单用 `find -printf '%s\t%P\0'`（NUL 分隔、字节数在前）+ **结尾哨兵 + 查 rc**：`%P\t%s\n` 遇含换行文件名会错行；丢 rc 则 find 超时会被渲染成「目的端缺 7.5 万个」，运维的合理反应是重传。缺哨兵一律抛错「本次不给校验结论」。
  - 抽样清单（文件名来自 **OSS 对象 key，外部可控**）走 **base64 + NUL**，**绝不用 heredoc**：一个内容为结束标记的 key 就能提前终止 heredoc、让后续内容在泰国生产机上被当命令执行。
  - 校验结论**必须显示在成功卡上**（`cards.result_card`）：不显示的话，关掉 `SSH_STAGE2_VERIFY` 后卡片与以前一模一样，那个开关就成了隐形的 fail-open 后门。
- **目的目录只有一份实现** `paths.dest_dir()`：传输器写哪、校验查哪靠它算出同一个串；漂移即「校验了个空目录」，而空目录表现为「缺 7.5 万个」。
- **段下发锁** `ssh:transfer:stagelaunch:{job_id}:{stage}`（NX, TTL 180s）：段1 rc 落盘到 `stage=STAGE2` 写回 Redis 之间有最长 60s 窗口，期间任何 `refresh()`（文本查询/按钮/对账兜底）都会再起一次段2。输家**不写 Redis 但回读刷新本地 job dict**（否则调用方那份永远停在旧 stage，7 天后用陈旧对象写 FAILED、覆盖赢家的 DONE）。
- **执行模型**：起任务 = 一条短 SSH 命令 `nohup bash -c '<work>; echo $? > rc' > log 2>&1 & echo $! > pid`；轮询 = 只读 `rc`/`kill -0 pid` marker，**从不通过 SSH 读长输出**（避 paramiko 大输出死锁 + 容器重启丢 channel）。每 job 一个工作目录 `{SGP_WORK_DIR}/{job_id}/`。rc 语义：段1 只有 `0` 算成功；段2 `0` 与 `24`（源文件传输中消失）都算成功。
- **凭证与主机认证**：私钥是 Fernet 密文 `SGP_SSH_KEY_ENC`，运行时解密进内存（**绝不落盘**，解密结果不含 `-----BEGIN` 直接报错）；host key 固定 `SGP_SSH_HOST_KEY` + `RejectPolicy`（**禁 AutoAdd，fail-closed**）。
- **注入面**：源桶/前缀/目标子目录全部先过 `paths` 的**严格白名单**（桶名 OSS 规范正则；每级 `\A[A-Za-z0-9._-]+\Z`，禁 `..`、空格、shell 元字符，尾锚用 `\Z` 而非 `$` 以封住结尾换行）——`shlex.quote` 只护 SGP 那一层 shell，段2 是 `ssh` 双跳、到泰国生产机上会再解一层。
- **段1 目的端 `/mnt/sgp_oss` 是 ossfs2(FUSE)，只支持顺序写**。ossutil 有两种写法违反它：① 超过 100MiB 的对象默认切分片、并发 pwrite 到不同 offset；② 见到已存在的 `.temp` 会当续传、从非零 offset 写。所以 `start_stage1` 现在：
  - `--parallel 1 --part-size 5Gi`（`_OSSFS2_PART_SIZE`，ossutil 上限）压成单分片顺序写；跨文件并发 `--job`（**单数**，2.x 的 flag，`--jobs` 会报 unknown flag）保留 30 不降。
  - 跑前 `find <dst> -type f -name '*.temp' -delete` 清残留。**不清的话，段1 只要被中断过一次（kill/容器重启/网络抖动），之后每一次重试都必然失败**。按后缀精确删，源桶内不存在 `.temp` 结尾对象（已核）。
  - 实证（19.527 TiB / 75850 对象那单）：≤100MiB 的 33484 个全成功、>100MiB 的 42366 个全失败，边界与分片阈值严格吻合，报告里 `invalid argument` 出现 42366 次、其它 Error 0 次；同一 345.4MiB 对象改 flags 后 rc=0；`.temp` 单独实测「留着 rc=4（0.6 秒就挂）/ 删掉 rc=0」。修复后 200 对象 77.6GiB 跑出 **148MiB/s 零 EINVAL**（比出错那次的 75MiB/s 快一倍）。残留限制：单个对象 >5Gi 压不成单分片，仍会 EINVAL。
- **失败明细** `engine_ssh.failure_detail()`：失败卡带 ossutil 报告路径 + 失败对象条数 + 首条 `cause`。ossutil 用 `\r` 刷屏（单次日志可达十几 MB），**必须 `tr '\r' '\n'` 再 grep**，直接 `tail -n` 只能抓到一整行进度条。取 20KB 尾窗、逐行截断（`_DETAIL_LINE_MAX=240` / 总长 `_DETAIL_MAX=1200`，先保「结论」再填日志原文，否则几条长汇总行就把根因挤没）；report 路径来自远端日志（外部数据）→ 过结构白名单 fullmatch，防被构造的对象 key 骗去 grep 任意文件。`Error:` 必须在 grep 列表里（异地桶 rc=2 那种快速失败日志里只有一行 `Error: ... AccessDenied`）。
- **`run_to_completion` 轮询上限 48h → 7 天**（`max_polls=10080` × 60s）：19.5TiB 的段1 就要约 38h，原上限会在任务仍正常运行时误判「轮询超时」。真失败仍由 rc marker 立刻判定，不依赖这个上限。
- **估算与审批**：`estimate_source` 走 SGP 上的 `ossutil du`，正则锚定 `total object sum size` / `total du size`（旧写法会先命中表头再跨行吞到 object count，把 22MB 读成 3B、绕过审批门），返回 `(bytes, objects, ok)`；`ok=False` → `needs_approval(size_known=False)` **fail-safe 当作需审批**，不 fail-open 放行大迁移。
- **未修的已知缺口**：`start_stage1` / `estimate_source` **都不带 `-e/--endpoint`**，完全依赖 SGP 上写死杭州的 `~/.ossutilconfig` → **源桶不在杭州则段1 必挂**（rc=2，403 `must be addressed using the specified endpoint`；前一单 `sgp-841b88a7b0dd` 即此）。与 ossfs2 修复互不影响。
- **入口**：飞书发「数据迁移（泰国H200）」等意图（`_is_ssh_transfer_intent`，**必须排在跨云 transfer 意图之前**，否则「数据迁移」会被跨云入口抢走）→ 录入卡（源 + 目标子目录）→ 确认卡（估算 + 超阈值仅管理员）→ 后台 `run_to_completion` 推进度/结果卡；CLI `python -m core.ssh_transfer.cli plan|apply|status`。**没有 Agent 工具**。目标子目录语义：内容铺进该目录，不再套一层源目录名；空则镜像源前缀。
- **Config**：`SSH_TRANSFER_ENABLED`、`SGP_SSH_HOST/PORT/USER/KEY_ENC/HOST_KEY`、`SGP_OSS_MOUNT`、`SGP_OSS_BUCKET`（段1 落点=段2 源的新加坡桶，默认 `wuji-sing`，须与 `/etc/ossfs2_sgp.conf` 的 `oss_bucket` 一致）、`SGP_WORK_DIR`、`SGP_OSSUTIL_JOBS`、`THAI_HOST/PORT/USER/DEST_ROOT`、`SSH_STAGE2_MODE`（`ossutil` 默认 / `rsync` 回滚）、`THAI_OSS_ENDPOINT`+`THAI_OSS_REGION`（**显式给、不吃泰国 `~/.ossutilconfig` 的默认值**——那文件人工维护，被改回杭州或加速域名会静默变慢、异地 endpoint 更会被 OSS 直接 403）、`THAI_OSSUTIL_JOBS`/`THAI_OSSUTIL_PARALLEL`、`THAI_WORK_DIR`（含 `$HOME`，**由远端 shell 展开**）、`SSH_STAGE2_VERIFY`（默认 true，**别关**）、`SSH_STAGE2_VERIFY_SAMPLES`、`THAI_RSYNC_STREAMS`/`THAI_RSYNC_SUDO`/`THAI_RSYNC_BWLIMIT`（仅 rsync 回滚路径用）、`SSH_TRANSFER_APPROVAL_TB`、`SSH_TRANSFER_CHAT_ID`。
  性能旋钮（`THAI_RSYNC_STREAMS`/`THAI_OSSUTIL_*`/`SSH_STAGE2_VERIFY_SAMPLES`）**刻意存原始字符串、不在 import 期 `int()`**：`.env` 写成空值或非数字会让 `config.settings` 整个 import 失败、**bot 起不来**，而它们只是旋钮。转换与钳位在使用处（非法值退回安全默认）。
  依赖 `paramiko` → **部署需 `docker compose up -d --build`**，不是普通 deploy。
- **泰国侧前置（人工一次性）**：装 `ossutil`（现 2.3.0）+ 配好 `~/.ossutilconfig`（能读 `SGP_OSS_BUCKET`），**权限须 600**（曾是 664，同机其他账号可读 AK/SK）。目标盘挂载点是 `/mnt/data04/296834`（WekaFS，比 `/mnt/data04` 深一层，`df /mnt/data04` 看到的是根分区、会误判容量不足）。sshd `MaxStartups` 取默认 `10:30:100` → rsync 回滚路径的并发上限被钉在 10（`engine_ssh._STAGE2_MAX_STREAMS`）。

### Cross-Cloud Transfer (`core/transfer/`)

Migrates object-storage data across clouds from a single user-given path. The chain is inherently two-stage — no PFS can skip its own object storage to reach the other cloud: `[PFS] --沉降(dataflow)--> [本厂对象存储] --跨云迁移--> [对方对象存储]`. **Direction decides engine** (destination-pull, by service design, not a choice): into OSS uses 阿里「在线迁移服务」(hcs_mgw, `alibabacloud_hcs_mgw20240626`); into TOS uses 火山「迁移服务」.

- **Phasing**: Phase 1 (done) = `TOS→OSS` only (the proven `wuji_il` path, full SDK). Phase 2 = `OSS→TOS` (new `engine_tos.py`, pending Volcano migration OpenAPI verification). Phase 3 = full chain with CPFS/VePFS sink via `CreateDataFlowTask`. Direction判断 + 全链路抽象 already in `paths.py`; later phases only add engines/stages.
- **`paths.py`** (pure logic): parses `tos://`/`oss://`/`cpfs://`/`vepfs://` URIs (dir-only, trailing `/`), derives destination bucket via `TRANSFER_BUCKET_MAP` (`{"<scheme>://<bucket>": "<dst-bucket>"}`), mirrors source prefix, builds a `Plan(source, dest, sink_target, engine, direction)`.
- **`engine_mgw.py`**: 阿里在线迁移 call-chain — `create_address(源 tos: access_id/secret/bucket/prefix/domain)` → `verify_address` → `create_address(目的 oss: bucket/region_id/prefix/role)` → `verify` → `create_job(transfer_mode, overwrite_mode)` → `update_job(IMPORT_JOB_LAUNCHING)` → poll `get_job` until `IMPORT_JOB_FINISHED|INTERRUPTED`. OSS dest uses RAM **role** (no AK); source TOS uses static `TOS_ACCESS_KEY/SECRET`. Client via `aliyun_client_factory.get_mgw_client`.
- **Known limitation: `OSS->TOS` target prefix**: Volcano DMS 1.0 only lets this integration specify the target **bucket**. `dest_prefix` is recorded by the tool but does not reliably change the TOS landing path; objects keep their source key structure under the target bucket. Example: migrating `oss://src/team/data/file.txt` to `tos://dst/custom/prefix/` lands as `tos://dst/team/data/file.txt`, not `tos://dst/custom/prefix/file.txt`. If a fixed TOS subdirectory is required, either pre-shape the source OSS keys or add a post-migration TOS copy/rename step.
- **`orchestrator.py`**: state machine `NEW→CROSSING→DONE|FAILED` (Phase 3 adds `SINKING`). Job in Redis `transfer:job:{job_id}` (30-day TTL). Idempotent: `job_id = hash(source, dest, 当天)`. `estimate_source` probes size via TOS `_prefix_size` for the confirm card + approval gate (`needs_approval` vs `TRANSFER_APPROVAL_TB`, default 1 TB). `run_to_completion` launches + blocks-polls (60s/轮) in a background thread, calling `on_update(job)` on each stage change.
- **Three entry points share the core** (like oss_perm): Feishu card (intent in `messages._is_transfer_intent`: 迁移动作词 + `tos://`/`oss://` path → `_handle_transfer_intent` sends `cards.confirm_card`); Agent tool `manage_transfer` (`plan`/`apply`/`status`); CLI `python -m core.transfer.cli plan|apply|status` (dry-run default, `--force` past threshold).
- **Bot flow**: confirm form-card (`confirm_transfer`, `form_value{overwrite, job_id}`) → `actions._h_confirm_transfer` (>threshold requires `ADMIN_FEISHU_OPEN_ID`) launches background `run_to_completion`, pushes progress/result cards to `TRANSFER_CHAT_ID or FEISHU_CHAT_ID`. Failure card has a `retry_transfer` button (`_h_retry_transfer` resets stage→NEW).
- **Config**: `TRANSFER_ENABLED`, `MGW_ENDPOINT`/`MGW_REGION`/`MGW_USER_ID`, `TRANSFER_OSS_ROLE` (RAM role for OSS dest), `TRANSFER_MODE_DEFAULT`/`TRANSFER_OVERWRITE_DEFAULT`, `TRANSFER_APPROVAL_TB`, `TRANSFER_BUCKET_MAP`. **Before go-live, fill real `wuji_il` values**: TOS domain/bucket, OSS rolename/bucket/region, transfer/overwrite modes, and the bucket map.
- **Phase 3 SINKING is wired for 阿里 CPFS** via `core/cpfs_dataflow.engine_nas` (NAS DataFlow Export): `run_to_completion` runs `start_sinking`→`poll_sink_once` (CPFS→OSS sink_target) before `start_cross`. Standalone 火山 vePFS↔TOS 预热/沉降 now exists as `core/vepfs_dataflow/` (see below); wiring it into transfer Phase-3 `start_sinking` for `vepfs→tos` full-chain is a remaining follow-up.

### Bucket Transfer — 同云桶间迁移 (`core/bucket_transfer/`)

一次性搬运**同云**对象存储：阿里 `oss://→oss://`、火山 `tos://→tos://`（同账号、跨 region/桶）。与跨云 `core/transfer/` 完全独立（新卡片/意图/编排/job 命名空间），只**加法式复用**其引擎：`engine_mgw.submit_cross_job(src_scheme="oss")`（新增 `create_oss_source_address`，源用 RAM role）+ `engine_tos.submit_cross_job(src_is_tos=True)`（DMS 源 vendor 切 `StorageVendorTOS`+`tos-<region>.volces.com`）。混合 scheme（oss↔tos）由 `paths.build_plan` 拒绝并提示改用跨云迁移。

- **云别由 scheme 推断**：oss→oss=阿里(engine=mgw) / tos→tos=火山(engine=dms)。
- **region**：OSS 用 `tools/aliyun/oss.detect_bucket_region`（GetBucketLocation）自动探测源/目的，跨 region 源走公网 domain；TOS 从 `tos://` 无法带出 → 回退 `TRANSFER_TOS_REGION`/`TOS_REGION`。
- **火山落盘限制**：DMS 目的只到桶级，对象保持源 key 原样落入目的桶（同 OSS→TOS，见上）。
- **三入口**：飞书发「桶间迁移」→ `cards.entry_card`（源/目的/同名策略）→ `_h_submit_bucket_transfer`→`confirm_card`→`_h_confirm_bucket_transfer` 后台 `run_to_completion` 推进度/结果（`query_bucket_transfer`/`retry_bucket_transfer` 按钮）。Redis `bkt:transfer:job:{id}`。
- **Config**：`BUCKET_TRANSFER_ENABLED`、`BUCKET_TRANSFER_OSS_SRC_ROLE`（阿里源 OSS 读权限 role，留空回退 `TRANSFER_OSS_ROLE`）；复用 `MGW_USER_ID`/`TOS_ACCESS_KEY`。
- **真机验收通过**：阿里 OSS→OSS（179MB/30对象跨region）+ 火山 TOS→TOS（task 553276，`StorageVendorTOS` 枚举确认，秒级）。火山 DMS 无阿里那种分钟级 LAUNCHING 排队。

### CPFS/NAS DataFlow — 数据预热 / 沉降 (`core/cpfs_dataflow/`)

Aliyun NAS DataFlow (product `NAS`, version `2017-06-26`, RPC): **预热**(`TaskAction=Import`, OSS→CPFS, 加载) and **沉降**(`TaskAction=Export`, CPFS→OSS, 刷回). See `docs/aliyun_cpfs_oss_dataflow_api.md`.

- **No new SDK**: NAS calls go through the generic `alibabacloud-tea-openapi` `call_api` (same pattern as `ram_approval._call_ims_api`) via `aliyun_client_factory.get_nas_openapi_client`. Deploy needs no rebuild.
- **API 面**: `list_dataflows`/`resolve_dataflow` (DescribeDataFlows) + `submit_task` (CreateDataFlowTask) + `query_task` (DescribeDataFlowTasks) + `create_dataflow`/`delete_dataflow`（CreateDataFlow/DeleteDataFlow，只服务下面的「临时 DataFlow」）。已有绑定的目录通常比任务目标宽 → `resolve_dataflow` 取最长的、是目标祖先的 `FileSystemPath`。
- **临时 DataFlow（复用优先，临建即删）**: `orchestrator.start_task` 先 `resolve_dataflow` 找能覆盖该目录的**现有**绑定 —— 命中则 `dataflow_ephemeral=False`（**绝不删**，那是别人的绑定）；找不到才 `create_dataflow` 临建并标 `dataflow_ephemeral=True`。`_cleanup_ephemeral()` 在 `run_to_completion` 末尾调用，**无论成功/失败/超时都删掉自己临建的那条**（只删自己的，且不长期占用「10 条/CPFS」上限）。
- **「CreateDataFlow 会清空 Fileset」的风险依然真实**，所以策略是「优先复用现有绑定、只在找不到时才临建」，而不是随手建流：CPFS 通用版的 DataFlow 绑在 Fileset 上，Fileset 里已有数据时建流会清空并替换为 OSS 侧数据（见 `docs/aliyun_cpfs_oss_dataflow_api.md`）。因此 `create_dataflow` **只对智算版 (`bmcpfs-`) 开放**，通用版直接抛错（还差 FsetId/Throughput），绝不代人建流。
- **Edition auto-branch**: `bmcpfs-*`=智算版 (`DataType=MetaAndData` + task requires `ConflictPolicy`) / `cpfs-*`=通用版.
- **Full-path input**: users give a full path like `/cpfs/cwr/third_party_data/label`; the `CPFS_MOUNT_PREFIX` (`/cpfs`) is stripped → DataFlow FileSystemPath `/cwr/third_party_data/label`. OSS side `oss://bucket/prefix/`. `orchestrator.make_plan(operation, cpfs_path, oss)` orients Directory/DstDirectory per action (Import: OSS-side Directory→CPFS DstDirectory; Export: reverse).
- **orchestrator.py**: state machine `NEW→RUNNING→DONE|FAILED`, Redis `cpfs:dataflow:job:{job_id}` (30-day TTL), idempotent `job_id=hash(op, fs, dir, oss, 当天)`, `run_to_completion` background-polls.
- **Three entry points** (mirror transfer): Feishu (`messages._is_sink_preheat_entry_intent` → `cards.entry_card`; `actions._h_submit_cpfs_dataflow`→`confirm_card`, `_h_confirm_cpfs_dataflow` launches + pushes progress/result, `retry_cpfs_dataflow`); Agent tool `manage_cpfs_dataflow` (`list`/`preheat`/`sink`/`status`, TOOL_GROUP `cpfs`); CLI `python -m core.cpfs_dataflow.cli list|preheat|sink|status` (`--dry-run`).
- **Discovery / selectable map** (`discovery.py`): `discover` iterates `CPFS_FILE_SYSTEM_IDS` (`fs_id@region,...`), calls `engine_nas.list_dataflows` per fs, and builds a list of `{region, fs_id, data_flow_id, oss_bucket, oss_prefix, fs_path, label, value}` options — cached in Redis `cpfs:dataflow:map` (`CPFS_MAP_TTL_SECONDS`, 6h). **`cri`-prefixed OSS buckets are excluded** (镜像仓库, not data). The Feishu `entry_card` renders these as a `select_static` so users pick a CPFS↔OSS binding + 相对子目录 instead of typing full paths (free-text inputs are the fallback when the map is empty). Selection `value` is a JSON blob decoded by `discovery.decode_selection` → `make_plan(..., fs_id=, region=, data_flow_id=)` (explicit binding skips `resolve_dataflow`). Refresh via CLI `discover --refresh` or tool `action=discover`.
- **Config**: `CPFS_DATAFLOW_ENABLED`, `CPFS_REGION`, `CPFS_FILE_SYSTEM_ID` (single default), `CPFS_FILE_SYSTEM_IDS` (multi-fs discovery list), `CPFS_MAP_TTL_SECONDS`, `CPFS_MOUNT_PREFIX`, `CPFS_CONFLICT_POLICY_DEFAULT`, `CPFS_DATAFLOW_MAP` (JSON override: `oss://<bucket>` or FileSystemPath → DataFlowId), `CPFS_APPROVAL_GB`, `CPFS_CHAT_ID`.
- **文件系统来源两条路**: 显式 `CPFS_FILE_SYSTEM_IDS` (`fs_id@region,...`) 优先；留空则按 `CPFS_REGIONS` 逐地区 `engine_nas.list_filesystems`(DescribeFileSystems) 枚举。`discovery.regions()` 供三步向导卡的「选地区」步骤。
- **Prereq for the map**: the bot AK needs `nas:DescribeDataFlows` (+ `nas:DescribeFileSystems` 走枚举路时)；default master AK is RAMReadOnly+STS — may need granting.

### vePFS/TOS DataFlow — 火山数据预热 / 沉降 (`core/vepfs_dataflow/`)

火山「文件存储 vePFS」数据流动 (service `vepfs`, version `2022-01-01`): **预热**(`TaskAction=Import`, TOS→vePFS) and **沉降**(`TaskAction=Export`, vePFS→TOS). 阿里 `core/cpfs_dataflow/` 的火山镜像。

- **SDK, no new dep**: uses `volcenginesdkvepfs.VEPFSApi` — a subpackage of the already-installed `volcengine-python-sdk` (same monolith as `volcenginesdkdms` used by `core/transfer/engine_tos.py`). `engine_vepfs._api(region)` mirrors `engine_tos._api`: `Configuration(ak/sk/region)`+`ApiClient`+`VEPFSApi`, 静态 AK 复用 `TOS_ACCESS_KEY/SECRET` (火山无 STS). region 须与 vePFS 文件系统区域一致.
- **Key diff vs 阿里**: 火山**没有** `CreateDataFlow` 持久绑定对象 → `submit_task` (CreateDataFlowTask) 直接带 TOS 桶/前缀 + vePFS `SubPath`/`FilesetId`，方向只由 `TaskAction` 决定、**不反转源/目的字段**。**省掉阿里那层 `resolve_dataflow`/临建临删 DataFlow 的全部逻辑**。双向都能指定路径（不像跨云 DMS 只到桶级）。
- **engine_vepfs.py**: `submit_task`→`CreateDataFlowTask` 返回 `data_flow_task_id`；`query_task`→`DescribeDataFlowTasks` 取 `status`/`total_size`/`exec_size`/`exec_count`/`failed_count`（容错 getattr）；`is_done`/`is_failed` 按子串归类 status（SDK 里 status 是自由 str，终态串真机反查确认）。字段：`data_storage`(TOS 桶, 默认 `tos://<bucket>`)/`data_storage_path`/`sub_path`/`fileset_id`/`same_name_file_policy`(Skip/KeepLatest/OverWrite)/`data_type`(MetaAndData)。
- **orchestrator.py**: state machine `NEW→RUNNING→DONE|FAILED`, Redis `vepfs:dataflow:job:{job_id}` (30-day TTL), idempotent `job_id=hash(op, fs, sub_path, tos, 当天)`, `run_to_completion` background-polls. `make_plan(op, vepfs_addr, tos)` + `plan_from_addresses(region, source, dest)` (方向由源/目的地址类型自动判断：源 vePFS→沉降 / 源 TOS→预热). 无 DataFlow create/resolve/cleanup。
- **统一向导卡（三步级联）**：阿里 CPFS 与火山 vePFS 共用 `core/dataflow_cards.py`：**入口选云**（`entry_card`，按钮 `pick_cloud_aliyun`/`pick_cloud_volcano`）→ **选地区**（`region_card`，按钮 `pick_region_aliyun`/`pick_region_volcano`，地区来自各自 `discovery.regions()`）→ **表单卡**（`form_card`：文件系统**下拉**(该地区，级联) + **源地址/目的地址文本** + 同名策略；**无操作/桶选择器**——方向由地址自动判断、桶写在 `oss://`/`tos://` 地址里）。飞书意图 `messages._is_sink_preheat_entry_intent`（含 `vepfs沉降`/`火山预热`/`tos沉降` 等词）→ `entry_card`。提交：阿里表单→`submit_cpfs_dataflow`、火山表单→`submit_vepfs_dataflow`，各自 handler 走 `_guided_plan(cloud, fv, region_hint)`（`_orient_addresses` 定方向：源=对象存储→预热 / 源=文件系统→沉降）→ confirm/progress/result 卡（`confirm_*`/`query_*_progress`/`retry_*`）。发现（`filesystems_in`/`fs_options`）为空时文件系统回退文本输入。
- **另两入口**：Agent tool `manage_vepfs_dataflow` (`preheat`/`sink`/`status`, TOOL_GROUP `vepfs`); CLI `python -m core.vepfs_dataflow.cli preheat|sink|status` (dry-run 默认, `--apply` 执行). 按 ID 查进度支持 `vepfs-` 前缀 (与 `cpfs-`/`tr-` 并列).
- **Config**: `VEPFS_DATAFLOW_ENABLED`, `VEPFS_REGION` (vePFS 与 TOS 必须同地域), `VEPFS_FILE_SYSTEM_ID` (默认 fs), `VEPFS_FILE_SYSTEM_IDS` (多 fs, 后续下拉发现), `VEPFS_CONFLICT_POLICY_DEFAULT` (Skip), `VEPFS_CHAT_ID`; 凭证复用 `TOS_ACCESS_KEY/SECRET`.
- **前置条件（控制台一次性）**: vePFS 与 TOS 同地域; 开通 **vePFS→TOS 服务访问授权**; `ConfigDataFlowBandwidth` 带宽>0; vePFS 已建 Fileset/目标目录. 调用 AK 需 `vepfs:CreateDataFlowTask`/`DescribeDataFlowTasks` 等.
- **待真机验证 (dry-run 反查, 同当初反查 DMS `StorageVendorTOS`)**: ① `DataStorage` 桶串格式 (`tos://bucket` vs 裸名); ② task `status` 终态枚举串; ③ `vepfs:*` IAM 动作精确名; ④ 智算版是否必须 `FilesetId`.

### OSS Permission Sync (`core/oss_perm/`)

Generates a least-privilege custom RAM policy per algo-team member from a Feishu Bitable and attaches it to their RAM user. Standalone script + bot flow share the same core.

- **Source of truth**: Feishu「舞肌算法组权限统计」多维表格 — member table (姓名/账号/状态/`OSS_Bucket`/`子目录(读)`/`子目录(写)`) + bucket 对照表 (合法子目录). `BUCKET_MAP` maps display names → real `(region, bucket)`（注意这张表存的 region **带 `oss-` 前缀**，与 `TEMP_AK_*_BUCKET_MAP` 的裸 region 是两套写法）。
- **Pipeline** (`permsync.py`): `load_members` → `build_plan` (`resolve_member` per row, merge by username) → policy doc `wuji-oss-auto-<username>`. `build_policy` emits object-level ARNs scoped to read/write prefixes + a prefix-scoped `ListObjects`.
- **Two-tier granularity** (`coerce_level` + `build_plan(level=)`): `bucket` collapses every prefix to whole-bucket (`<bucket>/*`, no `oss:Prefix` condition) → `dir` keeps subdir prefixes. Roll out bucket-level first, observe, then tighten to dir-level.
- **RAM user resolution**: display-name (==姓名) → email-prefix, overridable via `ram_user_map.json` (`--build-map` regenerates, keeps manual edits).
- **CLI**: dry-run by default; `--apply` writes RAM, `--audit` read-only reconciles RAM-actual vs table-expected (多授/少授/孤儿), `--level bucket|dir`, `--only`, `--create-users`. Credentials: `ALIBABA_CLOUD_*` or `settings.ALIYUN_ACCESS_KEY_*` (needs RAM write) + Feishu app token.
- **`make_ram_client(ak, sk)`**: 显式传参时**只用传入那对**（多主账号隔离的唯一安全入口）；只给一半直接抛错；零参时才回落 `ALIBABA_CLOUD_*` env → `settings.ALIYUN_ACCESS_KEY_*`（历史行为，逐字不变）。
- **Bot flow**: scheduler pushes `cards.audit_form_card` (card JSON 2.0 form: 粒度单选默认桶级 / 成员多选默认全选 / 单个「确认下发」) to `FEISHU_CHAT_ID` when `audit_diff` finds drift. Admin (`ADMIN_FEISHU_OPEN_ID`) submits → `actions._h_approve_oss_perm_selected` reads `form_value{level, selected}`, filters the plan to selected members, `apply_all` downs them, replies `result_card` (per-member granted scope). Deselect a member = skip this round (not persisted). Legacy `audit_card`/`approve_oss_perm` (整批两按钮) kept as fallback.
- Note: orphan-policy reporting is computed by `audit_diff` (CLI `--audit` shows it) but no longer rendered on the bot card.

### Redis Usage (`utils/redis_client.py`)

Single client, `decode_responses=True`. Failures degrade silently. Key namespaces in use（新功能在前）:
- `temp_ak:grant:{grant_id}` / `temp_ak_1949:grant:{grant_id}` — 30-day TTL, 临时凭证发放记录，**按账号档案分命名空间**（`accounts.AccountProfile.redis_prefix`）。**只存 `ak_id`，绝不存 secret/token**。配套 NX 锁 `temp_ak*:lock:*`。
- `ram_approval:instance:{code}` — 90-day TTL（每次写刷新），RAM/IAM 建号审批处理记录（含 `result_status`/`error_terminal`，失败记录含申请人姓名/邮箱/手机号，故必须有保留期）；`ram_approval:lock:{code}` 600s NX 防并发重投；`ram_approval:instances` zset 索引；`ram_approval:instance:failnotice:{code}:{sig}` 7-day，同实例同错误只评论一次的播报闸门。
- `pfs:transfer:job:{job_id}` — 30-day TTL, PFS 直传三段链状态（段完成标记 `sink_done`/`cross_done`/`preheat_done` + 三个子 job_id）；`pfs:transfer:launch:*` 下发 NX 锁。
- `ssh:transfer:job:{job_id}` — 30-day TTL, 杭州→新加坡→泰国两段链状态（`stage1_rc`/进度采样/`error_detail`）。
- `vepfs:dataflow:job:{job_id}` — 30-day TTL, 火山 vePFS 预热/沉降 task state (operation/fs/sub_path/tos/task_id/progress).
- `cpfs:dataflow:job:{job_id}` — 30-day TTL, CPFS 预热/沉降 task state (operation/fs/dataflow/task_id/progress/`dataflow_ephemeral`).
- `cpfs:dataflow:map` — `CPFS_MAP_TTL_SECONDS` (6h) TTL, discovered CPFS↔OSS DataFlow binding options for the Feishu selector.
- `bkt:transfer:job:{job_id}` — 30-day TTL, 同云桶间迁移状态机。
- `transfer:job:{job_id}` — 30-day TTL, cross-cloud transfer state machine record (stage/engine/bytes/error).
- `dataflow:notified:{job_id}` — 30-day TTL `SET NX`，**六条搬运链共用**的终态推送闸门（在线线程 / 对账 / 文本查询 / 按钮查询谁先到谁推，同 job 只推一张结果卡）。
- `gpu:dist:snapshot` — 24h TTL, GPU 卡分布快照(地区×卡型 + 每用户在算卡数 + 近N小时趋势时序)，15s 陈旧后台单飞刷新；实时 HTML 页面 `/gpu/distribution` 高频读它。数据源 `tools/aliyun/gpu_distribution.py`（DLC `count by(jobUserId,regionId)` + quota `NODE_GPU_ACCELERATOR_TOTAL/REQUEST`，jobUserId→姓名走 RAM ListUsers）；飞书意图 `_is_gpu_dist_intent`（卡分布/谁在用卡…）回摘要卡+链接；路由 `routes.py:/gpu/distribution` token 门禁。
- `mfu:snapshot` — 24h TTL, full cluster-MFU snapshot (~25 PromQL/region, 1–2 min to collect); card region-switch buttons read it for instant <3s callbacks. `mfu:refresh_lock` guards the single background refresher when stale (>15 min).
- `capacity:snapshot:{vendor}:{bucket}:{prefix}` — 30-day TTL, last capacity scan per target (for delta).
- `agent:chat_history:{session_id}` — 20-message list, FIFO trimmed via pipeline `rpush + ltrim`；`agent:chat_summary:{session_id}` 滚动摘要（两者 7-day 空闲 TTL）。
- `aliyun:sts:{open_id}:{role_arn}` — STS credential cache, TTL = `ALIYUN_STS_DURATION_SECONDS - 300`.
- `feishu:user_creds:{open_id}` — encrypted user AK/SK（Fernet）; 30-day idle TTL via `USER_AK_IDLE_TTL_SECONDS`。（旧文档误记为 `user:ak:{open_id}`，实际前缀见 `utils/aliyun_user_creds.py:24`。）
- `feishu:event_dedup:{event_id}` — 1h TTL, primary mechanism for webhook idempotency (in-memory `_seen_events_fallback` set only when Redis is down).
- `dsw:ticket:{ticket_key}` — 7-day TTL, scheduler state.
- `analysis:{file_name}:{mtime}` — 5-min TTL result cache for alarm analysis (mtime in key auto-invalidates on file change).
- `alarms:dedup:{file_name}` — 24h TTL set for alarm deduplication.

### Vector Store & RAG (`core/vector_store.py`)

ChromaDB with `shibing624/text2vec-base-chinese` embeddings, persisted in `vector_db/`, populated by `ingest.py` from `data/k8s_docs/*.txt`. `get_retriever()` is `@lru_cache(1)`, top-k = 3. Model cache in `models/model_cache/` (HuggingFace offline mode is forced via `settings.setup_env()`).

### Multi-Agent Patterns

- **Hybrid** — `EdgeWatcher` (Qwen3-4B) serializes its observation as JSON, then `CloudManager` (Qwen-Max) consumes it as `input`. Edge is cheap and fast for raw signal extraction; cloud handles judgment.
- **Collab** — diagnostic expert holds read-only tools (Prometheus, RAG, system stats); ops officer holds write tools (K8s restart, Feishu notify). Expert output becomes the officer's input.

### Feishu Bot (`core/feishu_bot/`)

Flat package: `routes.py` (Flask app + `/feishu/event` `/feishu/card_action` `/health` + `run()`), `actions.py` (card-action handler registry `_ACTION_HANDLERS`; sync path must answer <3s), `messages.py` (event dedup, GPU intent, bind commands, Agent invocation), `gpu_flow.py` (GPU request cards/state/parsing), `messaging.py` (send primitives). `__init__.py` keeps the stdio UTF-8 reconfigure first, then re-exports everything for backward compat. **Convention**: intra-package calls use `from . import <mod>` + `<mod>.func(...)` module-attribute access so tests patch one module (e.g. conftest patches `core.feishu_bot.messaging`).

`/feishu/event` handles `im.message.receive_v1`（消息）与审批事件（`approval_instance`/`approval_task`）。审批事件先过 `_approval_allowlist()` 硬门（不在白名单、含无 code 的一律丢弃），再依次分发 temp_ak 发放/延期（严格 code）→ RAM 建号。

消息路径（**顺序有意义**，先命中先赢）：
1. 进度查询 `查询进度 <任务ID>`（`_JOB_ID_RE` 认 `tr-`/`cpfs-`/`vepfs-`/`sgp-`/`xpfs-`；**`bkt-` 不在其中**，桶间迁移只能点卡片按钮查）→ `refresh()` 重查云端后回文案，终态经 `dataflow:notified` 闸门补推结果卡。
2. MFU 日报 → GPU 卡分布（摘要卡 + `/gpu/distribution` 链接）。
3. 搬运链入口，顺序固定：**PFS 直传（须同时提到 vepfs+cpfs）→ SSH 迁移链（泰国 H200）→ 桶间迁移 → CPFS/vePFS 预热沉降向导 → 跨云迁移**。越通用的话术越靠后，否则「数据迁移」「迁移」会被前面的入口抢走。
4. 火山 IAM / 阿里 RAM 账号查询入口。
5. GPU intent（资源 + 动作词，或训练类话术）→ action-button card，草稿存 Redis（`JIRA_ENABLED=false` 时回停用提示）。
6. AK-binding intent → Fernet-encrypted save to `feishu:user_creds:{open_id}`。
7. 其余 → `core.agent._build_executor()`（full `ALL_TOOLS`，非流式）→ 飞书回复卡（仅命中指标词/监控意图时附趋势图）。

Event dedup uses Redis `SET NX` with TTL (`_is_duplicate_event`). App access token is cached inside `tools/feishu/notify._get_access_token`.

### Configuration (`config/settings.py`)

Reads everything from `.env` at import time. `setup_env()` (called from `main.py`) forces HuggingFace offline mode and auto-creates `sessions/`, `vector_db/`, `models/model_cache/`, `data/`.

`Config._REQUIRED_FIELDS` is the canonical list of "what breaks if missing" — when you add a new env-driven feature, add the field and its impact string here so `print_validate` warns operators at startup.
