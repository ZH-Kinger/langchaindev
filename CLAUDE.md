# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIOps 智能运维助手 — a multi-mode AI operations platform built on LangChain 0.3.0. It supports RAG knowledge base Q&A, single-agent tool calling, hybrid dual-model pipelines, multi-agent collaboration, and a Feishu Webhook bot.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
python ingest.py        # build ChromaDB knowledge base (run once)
```

Key `.env` fields: `DASHSCOPE_API_KEY`, `MODELSCOPE_API_KEY`, `REDIS_URL`, `PROMETHEUS_URL`, `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_VERIFICATION_TOKEN`.

## Running the Application

```bash
python main.py                  # interactive mode selector
python main.py --mode rag       # RAG Q&A
python main.py --mode agent     # single agent with tools
python main.py --mode hybrid    # edge→cloud two-stage pipeline
python main.py --mode collab    # diagnostic + execution multi-agent
python main.py --mode bot       # Feishu Webhook server on :8088
```

## Tests

```bash
python test_llm_apis.py    # LLM connectivity (cloud & edge)
python test_skills.py      # tool skills (alarm reduction, K8s restart)
python test_grafana.py     # Grafana integration
python data/generate_data.py   # regenerate test datasets
```

No lint or type-check commands are configured; no test runner (pytest/unittest) is used — tests are plain scripts.

## Architecture

### Modes and Entry Points

`main.py` dispatches to five modes:

| Mode | Entry | Description |
|------|-------|-------------|
| `rag` | `app.py` | RAG chain with `FileChatMessageHistory` session persistence |
| `agent` | `core/agent.py` | Single agent, Redis session memory, streaming output |
| `hybrid` | `core/hybrid_agents.py` | Edge (Qwen3-4B) perception → Cloud (Qwen-Max) decision |
| `collab` | `core/multi_agent_system.py` | Diagnostic Expert agent → Ops Officer agent |
| `bot` | `core/feishu_bot.py` | Flask Webhook (port 8088), event dedup, rich card replies |

### LLM Factory (`utils/llm_factory.py`)

Two cached factories (`@lru_cache`):
- `get_cloud_llm()` — Qwen-Max via DashScope (OpenAI-compatible endpoint)
- `get_edge_llm()` — Qwen3-4B via ModelScope

Same `(temperature, streaming)` combo reuses the same instance.

### Tool System (`tools/`)

All tools subclass `BaseOpsTool` (`tools/base_tool.py`) for audit logging and data-path management. Tools are registered in `tools/__init__.py` as `ALL_TOOLS`.

Agent tool routing is **keyword-based**: `core/agent.py` inspects the user input for keywords (文档/知识库, 重启/pod, cpu/prometheus, etc.) and passes only the relevant tool subset to the agent executor.

Key tools:
- `system_tool.py` — CPU/memory/disk via `psutil`, scans `data/` directory
- `analysis_skills.py` — Pandas alarm deduplication with Redis persistence (24h TTL)
- `ops_skills.py` — K8s Pod restart; protected namespace whitelist blocks `prod`, `production`, `kube-system`
- `prometheus_tool.py` — Prometheus range query, threshold detection for CPU/mem/disk/net/GPU
- `rag_tool.py` — lazy-loads ChromaDB retriever, wraps RAG chain
- `feishu_tool.py` — Feishu message cards, GPU progress bars, token exchange
- `gpu_advisor_tool.py` — GPU cluster optimization advisor

### Vector Store & RAG (`utils/vector_store.py`)

ChromaDB with `shibing624/text2vec-base-chinese` embeddings. Populated by `ingest.py` from `.txt` files in `data/k8s_docs/`. Persistent storage in `vector_db/`. Model cache in `models/model_cache/`.

### Redis Usage (`utils/redis_client.py`)

Single Redis client shared across three concerns:
- **Session history**: `agent:chat_history:{session_id}` — capped at 20 messages
- **Analysis cache**: `analysis:{file_name}:{mtime}` — avoids reprocessing unchanged files
- **Alarm dedup**: `alarms:dedup:{file_name}` — Redis set, 24h TTL

Redis failures degrade gracefully (empty history / no cache).

### Multi-Agent Patterns

- **Hybrid** (`hybrid_agents.py`): Edge Watcher serialises its observation as a JSON string and passes it as input to the Cloud Manager. The edge model is cheaper and fast; the cloud model makes final decisions.
- **Collab** (`multi_agent_system.py`): Diagnostic Expert has read-only tools (Prometheus, RAG, system stats); Ops Officer has write tools (K8s restart, Feishu notify). Expert output is fed as input to the Officer.

### Feishu Bot (`core/feishu_bot.py`)

Flask endpoint `/feishu/event` handles `im.message.receive_v1` events. Deduplication via an in-memory `_seen_events` set. Replies use Feishu card format (markdown + optional images). App access token is fetched and cached per request via `FEISHU_APP_ID` + `FEISHU_APP_SECRET`.

### Configuration (`config/settings.py`)

Reads all config from `.env` at import time. Auto-creates `sessions/`, `vector_db/`, and `models/model_cache/` directories if missing.