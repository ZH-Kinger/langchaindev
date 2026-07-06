# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIOps 智能运维助手 — a multi-mode AI operations platform built on LangChain 0.3. Five runtime modes (RAG, single agent, hybrid edge→cloud, multi-agent collab, Feishu Webhook bot) sit on top of a tool layer that wraps Aliyun (PAI DSW / ECS / OSS / SLS / RAM / Prometheus), Jira, GitHub, K8s, and a local ChromaDB knowledge base. Per-user Aliyun calls go through STS AssumeRole so the bot never uses a shared AK.

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
  aliyun/   pai_dsw, ecs, oss, sls, ram, prometheus, gpu_advisor,
            gpu_training_advisor, cluster_health, cluster_mfu, dsw_inspector
            # oss also: dir_sizes (各子目录大小) + tree (目录结构), 自动探测地域
            # cluster_mfu: 多区域算力效率(MFU)+容量+调度 日报，交互式飞书卡片(区域切换)，
            #   24h Redis 快照，按钮回调只读缓存秒回
  volcano/  tos                 # 火山引擎 TOS 容量盘点 (静态 AK，无 STS)
  feishu/   notify, cards       # notify: message cards, GPU progress bars, token cache
                                # cards: card-dict primitives (card/div/fields/btn/...) shared by
                                #   all card builders — NOT an agent tool, never export in __init__
  jira/     ticket, workflow    # GPU ticket CRUD + algo workflow query
  github/   workflow            # PR / commit / sprint activity
  knowledge/rag                 # ChromaDB retriever, lazy-loaded
  ops/      system, analysis, monitor, k8s    # psutil, pandas dedup, k8s restart
```

All tools subclass `BaseOpsTool` (`tools/base_tool.py`) for audit logging and data-path resolution.

### Agent Tool Routing (`core/agent.py::_select_tools`)

Three tiers, in order:

1. **Fast path** — if input contains an unambiguous proper noun (`dsw`, `jira`, `ecs`, `oss`, `sls`, `k8s`, `pod`, `prometheus`...) and no knowledge-override token (`知识库`, `文档`, `手册`...), go straight to the legacy keyword router. Zero latency, zero cost.
2. **LLM router** (`core/intent_router.py`) — `cloud_llm` at temperature 0 picks 1–2 intents from `INTENT_DESCRIPTIONS`. `@lru_cache(256)` on input text. Output is parsed with tolerance for markdown wrapping; unknown intents are filtered. Empty result → fall through.
3. **Legacy keyword fallback** — long `elif` chain in `_legacy_keyword_route`; `knowledge` is the final branch to avoid `怎么/如何` hijacking other intents.

`INTENT_DESCRIPTIONS` keys must match `TOOL_GROUPS` keys; the intent router prompt is built dynamically from these descriptions.

### Aliyun STS Multi-Tenant Credentials

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
- **Idle/timeout watcher** (every 5 min): for tracked instances, sends a Feishu card 15 min before the requested duration ends; if no extend/stop response within `DSW_IDLE_STOP_MINUTES`, auto-stops.
- **Capacity monitor** (`core/capacity_monitor.py`, opt-in via `CAPACITY_MONITOR_ENABLED`): every `CAPACITY_MONITOR_INTERVAL_HOURS`, scans each `CAPACITY_MONITOR_TARGETS` entry (OSS via `tools.aliyun.oss.compute_dir_sizes`, TOS via `tools.volcano.tos.compute_dir_sizes`), pushes a Feishu card with per-subdir sizes + delta-since-last-scan, reds the header if total exceeds `CAPACITY_ALERT_THRESHOLD_TB`. Snapshot in Redis `capacity:snapshot:{vendor}:{bucket}:{prefix}`. Also upserts per-vendor totals into a Feishu Bitable via `core/capacity_bitable.write_scan` (三表关联：巡检快照→厂家总量→批次明细，每次 upsert 去重旧行).
- **Morning report** (daily 北京 9:00): pushes two cards — per-user DSW instance summary + cluster MFU summary (`tools.aliyun.cluster_mfu.build_mfu_card(refresh=True)`, which also warms the snapshot cache for the card's region-switch buttons). Skipped if `PROMETHEUS_URL`/`FEISHU_CHAT_ID` unset.
- **OSS perm audit push** (`core/dsw_scheduler._oss_perm_loop`, opt-in via `OSS_PERM_PUSH_ENABLED`, daily 北京 `OSS_PERM_PUSH_HOUR`:20): see OSS Permission Sync below.

### OSS Permission Sync (`core/oss_perm/`)

Generates a least-privilege custom RAM policy per algo-team member from a Feishu Bitable and attaches it to their RAM user. Standalone script + bot flow share the same core.

- **Source of truth**: Feishu「舞肌算法组权限统计」多维表格 — member table (姓名/账号/状态/`OSS_Bucket`/`子目录(读)`/`子目录(写)`) + bucket 对照表 (合法子目录). `BUCKET_MAP` maps display names → real `(region, bucket)`.
- **Pipeline** (`permsync.py`): `load_members` → `build_plan` (`resolve_member` per row, merge by username) → policy doc `wuji-oss-auto-<username>`. `build_policy` emits object-level ARNs scoped to read/write prefixes + a prefix-scoped `ListObjects`.
- **Two-tier granularity** (`coerce_level` + `build_plan(level=)`): `bucket` collapses every prefix to whole-bucket (`<bucket>/*`, no `oss:Prefix` condition) → `dir` keeps subdir prefixes. Roll out bucket-level first, observe, then tighten to dir-level.
- **RAM user resolution**: display-name (==姓名) → email-prefix, overridable via `ram_user_map.json` (`--build-map` regenerates, keeps manual edits).
- **CLI**: dry-run by default; `--apply` writes RAM, `--audit` read-only reconciles RAM-actual vs table-expected (多授/少授/孤儿), `--level bucket|dir`, `--only`, `--create-users`. Credentials: `ALIBABA_CLOUD_*` or `settings.ALIYUN_ACCESS_KEY_*` (needs RAM write) + Feishu app token.
- **Bot flow**: scheduler pushes `cards.audit_form_card` (card JSON 2.0 form: 粒度单选默认桶级 / 成员多选默认全选 / 单个「确认下发」) to `FEISHU_CHAT_ID` when `audit_diff` finds drift. Admin (`ADMIN_FEISHU_OPEN_ID`) submits → `actions._h_approve_oss_perm_selected` reads `form_value{level, selected}`, filters the plan to selected members, `apply_all` downs them, replies `result_card` (per-member granted scope). Deselect a member = skip this round (not persisted). Legacy `audit_card`/`approve_oss_perm` (整批两按钮) kept as fallback.
- Note: orphan-policy reporting is computed by `audit_diff` (CLI `--audit` shows it) but no longer rendered on the bot card.

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
- **Phase 3 SINKING is wired for 阿里 CPFS** via `core/cpfs_dataflow.engine_nas` (NAS DataFlow Export): `run_to_completion` runs `start_sinking`→`poll_sink_once` (CPFS→OSS sink_target) before `start_cross`. `vepfs→tos` sink (火山) still unimplemented (`start_sinking` fails with a clear message).

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
- **Only 查现有 + 提交任务**: `engine_nas.list_dataflows`/`resolve_dataflow` (DescribeDataFlows) + `submit_task` (CreateDataFlowTask) + `query_task` (DescribeDataFlowTasks). **No** CreateDataFlow/DeleteDataFlow (creating one clears the Fileset — dangerous). A DataFlow must already exist; its bound dir is usually broader, the task targets a sub-dir → `resolve_dataflow` picks the longest `FileSystemPath` prefix that is an ancestor of the target.
- **Edition auto-branch**: `bmcpfs-*`=智算版 (`DataType=MetaAndData` + task requires `ConflictPolicy`) / `cpfs-*`=通用版.
- **Full-path input**: users give a full path like `/cpfs/cwr/third_party_data/label`; the `CPFS_MOUNT_PREFIX` (`/cpfs`) is stripped → DataFlow FileSystemPath `/cwr/third_party_data/label`. OSS side `oss://bucket/prefix/`. `orchestrator.make_plan(operation, cpfs_path, oss)` orients Directory/DstDirectory per action (Import: OSS-side Directory→CPFS DstDirectory; Export: reverse).
- **orchestrator.py**: state machine `NEW→RUNNING→DONE|FAILED`, Redis `cpfs:dataflow:job:{job_id}` (30-day TTL), idempotent `job_id=hash(op, fs, dir, oss, 当天)`, `run_to_completion` background-polls.
- **Three entry points** (mirror transfer): Feishu (`messages._is_sink_preheat_entry_intent` → `cards.entry_card`; `actions._h_submit_cpfs_dataflow`→`confirm_card`, `_h_confirm_cpfs_dataflow` launches + pushes progress/result, `retry_cpfs_dataflow`); Agent tool `manage_cpfs_dataflow` (`list`/`preheat`/`sink`/`status`, TOOL_GROUP `cpfs`); CLI `python -m core.cpfs_dataflow.cli list|preheat|sink|status` (`--dry-run`).
- **Discovery / selectable map** (`discovery.py`): `discover` iterates `CPFS_FILE_SYSTEM_IDS` (`fs_id@region,...`), calls `engine_nas.list_dataflows` per fs, and builds a list of `{region, fs_id, data_flow_id, oss_bucket, oss_prefix, fs_path, label, value}` options — cached in Redis `cpfs:dataflow:map` (`CPFS_MAP_TTL_SECONDS`, 6h). **`cri`-prefixed OSS buckets are excluded** (镜像仓库, not data). The Feishu `entry_card` renders these as a `select_static` so users pick a CPFS↔OSS binding + 相对子目录 instead of typing full paths (free-text inputs are the fallback when the map is empty). Selection `value` is a JSON blob decoded by `discovery.decode_selection` → `make_plan(..., fs_id=, region=, data_flow_id=)` (explicit binding skips `resolve_dataflow`). Refresh via CLI `discover --refresh` or tool `action=discover`.
- **Config**: `CPFS_DATAFLOW_ENABLED`, `CPFS_REGION`, `CPFS_FILE_SYSTEM_ID` (single default), `CPFS_FILE_SYSTEM_IDS` (multi-fs discovery list), `CPFS_MAP_TTL_SECONDS`, `CPFS_MOUNT_PREFIX`, `CPFS_CONFLICT_POLICY_DEFAULT`, `CPFS_DATAFLOW_MAP` (JSON override: `oss://<bucket>` or FileSystemPath → DataFlowId), `CPFS_APPROVAL_GB`, `CPFS_CHAT_ID`.
- **Prereq for the map**: the bot AK needs `nas:DescribeDataFlows` (default master AK is RAMReadOnly+STS — may need granting); fill `CPFS_FILE_SYSTEM_IDS` with the real fs/region list. CPFS 智算版(`bmcpfs-`) 全量枚举(DescribeFileSystems)未实现,故按配置 fs 清单逐个 DescribeDataFlows。

### Redis Usage (`utils/redis_client.py`)

Single client, `decode_responses=True`. Failures degrade silently. Key namespaces in use:
- `agent:chat_history:{session_id}` — 20-message list, FIFO trimmed via pipeline `rpush + ltrim`.
- `analysis:{file_name}:{mtime}` — 5-min TTL result cache for alarm analysis (mtime in key auto-invalidates on file change).
- `alarms:dedup:{file_name}` — 24h TTL set for alarm deduplication.
- `aliyun:sts:{open_id}:{role_arn}` — STS credential cache, TTL = `ALIYUN_STS_DURATION_SECONDS - 300`.
- `feishu:event_dedup:{event_id}` — 1h TTL, primary mechanism for webhook idempotency (in-memory `_seen_events_fallback` set only when Redis is down).
- `dsw:ticket:{ticket_key}` — 7-day TTL, scheduler state.
- `capacity:snapshot:{vendor}:{bucket}:{prefix}` — 30-day TTL, last capacity scan per target (for delta).
- `mfu:snapshot` — 24h TTL, full cluster-MFU snapshot (~25 PromQL/region, 1–2 min to collect); card region-switch buttons read it for instant <3s callbacks. `mfu:refresh_lock` guards the single background refresher when stale (>15 min).
- `user:ak:{open_id}` — encrypted user AK/SK; 30-day idle TTL via `USER_AK_IDLE_TTL_SECONDS`.
- `transfer:job:{job_id}` — 30-day TTL, cross-cloud transfer state machine record (stage/engine/bytes/error).
- `cpfs:dataflow:job:{job_id}` — 30-day TTL, CPFS 预热/沉降 task state (operation/fs/dataflow/task_id/progress).
- `cpfs:dataflow:map` — `CPFS_MAP_TTL_SECONDS` (6h) TTL, discovered CPFS↔OSS DataFlow binding options for the Feishu selector.

### Vector Store & RAG (`core/vector_store.py`)

ChromaDB with `shibing624/text2vec-base-chinese` embeddings, persisted in `vector_db/`, populated by `ingest.py` from `data/k8s_docs/*.txt`. `get_retriever()` is `@lru_cache(1)`, top-k = 3. Model cache in `models/model_cache/` (HuggingFace offline mode is forced via `settings.setup_env()`).

### Multi-Agent Patterns

- **Hybrid** — `EdgeWatcher` (Qwen3-4B) serializes its observation as JSON, then `CloudManager` (Qwen-Max) consumes it as `input`. Edge is cheap and fast for raw signal extraction; cloud handles judgment.
- **Collab** — diagnostic expert holds read-only tools (Prometheus, RAG, system stats); ops officer holds write tools (K8s restart, Feishu notify). Expert output becomes the officer's input.

### Feishu Bot (`core/feishu_bot/`)

Flat package: `routes.py` (Flask app + `/feishu/event` `/feishu/card_action` `/health` + `run()`), `actions.py` (card-action handler registry `_ACTION_HANDLERS`; sync path must answer <3s), `messages.py` (event dedup, GPU intent, bind commands, Agent invocation), `gpu_flow.py` (GPU request cards/state/parsing), `messaging.py` (send primitives). `__init__.py` keeps the stdio UTF-8 reconfigure first, then re-exports everything for backward compat. **Convention**: intra-package calls use `from . import <mod>` + `<mod>.func(...)` module-attribute access so tests patch one module (e.g. conftest patches `core.feishu_bot.messaging`).

Flask endpoint `/feishu/event` handles `im.message.receive_v1`. Three message paths:
1. GPU intent (resource + action words, or training phrases) → action-button card, persists draft to Redis.
2. AK-binding intent → Fernet-encrypted save to `user:ak:{open_id}`.
3. Everything else → `core.agent._build_executor()` (full `ALL_TOOLS`, non-streaming) → Feishu reply card.

Event dedup uses Redis `SET NX` with TTL (`_is_duplicate_event`). App access token is cached inside `tools/feishu/notify._get_access_token`.

### Configuration (`config/settings.py`)

Reads everything from `.env` at import time. `setup_env()` (called from `main.py`) forces HuggingFace offline mode and auto-creates `sessions/`, `vector_db/`, `models/model_cache/`, `data/`.

`Config._REQUIRED_FIELDS` is the canonical list of "what breaks if missing" — when you add a new env-driven feature, add the field and its impact string here so `print_validate` warns operators at startup.
