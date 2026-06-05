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

## Architecture

### Modes and Entry Points

`main.py` dispatches via `RUNNERS`:

| Mode | Entry | Notes |
|------|-------|-------|
| `rag` | `core/rag_runner.py` | Session memory via `FileChatMessageHistory` (file-backed, not Redis) |
| `agent` | `core/agent.py` | Three-tier tool routing, Redis session memory, streaming |
| `hybrid` | `core/hybrid_agents.py` | Edge model emits JSON observation → cloud model decides |
| `collab` | `core/multi_agent_system.py` | Diagnostic agent (read-only tools) → Ops agent (write tools) |
| `bot` | `core/feishu_bot.py` | Flask `/feishu/event`, starts `core/dsw_scheduler` background threads |

### LLM Factory (`core/llm_factory.py`)

`get_cloud_llm(temperature, streaming)` and `get_edge_llm(temperature)` are `@lru_cache`'d — same args = same instance. Call `clear_llm_cache()` after mutating `settings.MODEL_NAME` at runtime (see `core/agent._switch_model`).

### Tool System (`tools/`)

Subpackaged by vendor / domain. Single source of truth is `tools/__init__.py`:

- `ALL_TOOLS` — flat list registered with the agent.
- `TOOL_GROUPS` — `{group_name: set[tool_name]}` used by the agent router. **Validated at import time**: any name in `TOOL_GROUPS` not present in `ALL_TOOLS` raises `ValueError` — silent routing breakage is impossible.

```
tools/
  aliyun/   pai_dsw, ecs, oss, sls, ram, prometheus, gpu_advisor,
            gpu_training_advisor, cluster_health, dsw_inspector
            # oss also: dir_sizes (各子目录大小) + tree (目录结构), 自动探测地域
  volcano/  tos                 # 火山引擎 TOS 容量盘点 (静态 AK，无 STS)
  feishu/   notify              # message cards, GPU progress bars, token cache
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
- **Capacity monitor** (`core/capacity_monitor.py`, opt-in via `CAPACITY_MONITOR_ENABLED`): every `CAPACITY_MONITOR_INTERVAL_HOURS`, scans each `CAPACITY_MONITOR_TARGETS` entry (OSS via `tools.aliyun.oss.compute_dir_sizes`, TOS via `tools.volcano.tos.compute_dir_sizes`), pushes a Feishu card with per-subdir sizes + delta-since-last-scan, reds the header if total exceeds `CAPACITY_ALERT_THRESHOLD_TB`. Snapshot in Redis `capacity:snapshot:{vendor}:{bucket}:{prefix}`.

### Redis Usage (`utils/redis_client.py`)

Single client, `decode_responses=True`. Failures degrade silently. Key namespaces in use:
- `agent:chat_history:{session_id}` — 20-message list, FIFO trimmed via pipeline `rpush + ltrim`.
- `analysis:{file_name}:{mtime}` — 5-min TTL result cache for alarm analysis (mtime in key auto-invalidates on file change).
- `alarms:dedup:{file_name}` — 24h TTL set for alarm deduplication.
- `aliyun:sts:{open_id}:{role_arn}` — STS credential cache, TTL = `ALIYUN_STS_DURATION_SECONDS - 300`.
- `feishu:event_dedup:{event_id}` — 1h TTL, primary mechanism for webhook idempotency (in-memory `_seen_events_fallback` set only when Redis is down).
- `dsw:ticket:{ticket_key}` — 7-day TTL, scheduler state.
- `capacity:snapshot:{vendor}:{bucket}:{prefix}` — 30-day TTL, last capacity scan per target (for delta).
- `user:ak:{open_id}` — encrypted user AK/SK; 30-day idle TTL via `USER_AK_IDLE_TTL_SECONDS`.

### Vector Store & RAG (`core/vector_store.py`)

ChromaDB with `shibing624/text2vec-base-chinese` embeddings, persisted in `vector_db/`, populated by `ingest.py` from `data/k8s_docs/*.txt`. `get_retriever()` is `@lru_cache(1)`, top-k = 3. Model cache in `models/model_cache/` (HuggingFace offline mode is forced via `settings.setup_env()`).

### Multi-Agent Patterns

- **Hybrid** — `EdgeWatcher` (Qwen3-4B) serializes its observation as JSON, then `CloudManager` (Qwen-Max) consumes it as `input`. Edge is cheap and fast for raw signal extraction; cloud handles judgment.
- **Collab** — diagnostic expert holds read-only tools (Prometheus, RAG, system stats); ops officer holds write tools (K8s restart, Feishu notify). Expert output becomes the officer's input.

### Feishu Bot (`core/feishu_bot.py`)

Flask endpoint `/feishu/event` handles `im.message.receive_v1`. Three message paths:
1. GPU intent (resource + action words, or training phrases) → action-button card, persists draft to Redis.
2. AK-binding intent → Fernet-encrypted save to `user:ak:{open_id}`.
3. Everything else → `core.agent._build_executor()` (full `ALL_TOOLS`, non-streaming) → Feishu reply card.

Event dedup uses Redis `SET NX` with TTL (`_is_duplicate_event`). App access token is cached inside `tools/feishu/notify._get_access_token`.

### Configuration (`config/settings.py`)

Reads everything from `.env` at import time. `setup_env()` (called from `main.py`) forces HuggingFace offline mode and auto-creates `sessions/`, `vector_db/`, `models/model_cache/`, `data/`.

`Config._REQUIRED_FIELDS` is the canonical list of "what breaks if missing" — when you add a new env-driven feature, add the field and its impact string here so `print_validate` warns operators at startup.
