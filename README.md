# AIOps 智能运维助手

基于 LangChain 构建的企业级 AI 运维平台，集成多模型协作、RAG 知识库、实时监控与飞书机器人，实现智能化的基础设施运维与异常处理。

## 功能特性

- **多运行模式**：RAG 问答、单智能体、混合双模型、多智能体协作、飞书机器人
- **混合边云架构**：轻量边缘模型（Qwen 3-4B）负责感知，云端大模型（Qwen-Max）负责决策
- **多智能体协作**：诊断专家与运维执行两个角色独立运作，职责分离
- **RAG 知识库**：ChromaDB 向量数据库 + 中文 Embedding，支持运维文档语义检索
- **持久化对话记忆**：基于 Redis 的会话历史（最近 20 条），支持优雅降级
- **企业集成**：Prometheus 指标查询、Grafana 看板、飞书 Webhook 通知
- **告警降噪**：Redis 去重，防止重复告警轰炸
- **审计日志**：所有运维操作带时间戳完整记录

## 项目结构

```
langchaindev/
├── main.py                     # 主入口，运行模式选择器
├── app.py                      # RAG 模式独立入口
├── ingest.py                   # 文档向量化写入脚本
│
├── config/
│   └── settings.py             # 全局配置（路径、模型参数、API）
│
├── core/
│   ├── agent.py                # 单智能体（工具调用 + Redis 记忆）
│   ├── hybrid_agents.py        # 混合双模型工作流（边缘→云端）
│   ├── multi_agent_system.py   # 多智能体协作（诊断 + 执行）
│   ├── feishu_bot.py           # 飞书 Webhook 服务（Flask）
│   ├── chains.py               # RAG 链组装
│   └── prompts.py              # 系统提示词
│
├── tools/
│   ├── base_tool.py            # 工具基类（日志、路径管理）
│   ├── system_tool.py          # 系统监控（CPU、内存、磁盘）
│   ├── analysis_skills.py      # 告警降噪 + Pandas 数据分析
│   ├── ops_skills.py           # K8s Pod 重启运维操作
│   ├── monitor_skills.py       # CPU 趋势分析
│   ├── prometheus_tool.py      # Prometheus 指标查询与分析
│   ├── feishu_tool.py          # 飞书消息卡片推送
│   └── rag_tool.py             # 知识库查询工具
│
├── utils/
│   ├── llm_factory.py          # LLM 工厂（云端/边缘模型）
│   ├── vector_store.py         # ChromaDB + HuggingFace Embedding
│   ├── redis_client.py         # Redis 单例（会话/缓存/去重）
│   ├── aliyun_auth.py          # 阿里云鉴权
│   └── chart_builder.py        # Grafana 看板构建
│
├── data/
│   ├── generate_data.py        # 测试数据生成脚本
│   ├── k8s_events_200.csv      # K8s 事件测试数据
│   ├── cpu/memory/network_metrics_200.csv
│   ├── raw_alarms_200.csv      # 原始告警数据
│   └── k8s_docs/               # 待向量化的知识库文档
│
├── vector_db/                  # ChromaDB 持久化存储
├── models/model_cache/         # HuggingFace 模型缓存
└── sessions/                   # 对话历史文件
```

## 快速开始

### 环境要求

- Python 3.10+
- Redis（可选，用于持久化会话记忆）

### 安装依赖

```bash
# 克隆项目
git clone <repo-url>
cd langchaindev

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入以下配置：

```bash
# 云端大模型（阿里云 DashScope）
OPENAI_API_KEY=your-dashscope-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 边缘轻量模型（ModelScope Qwen 3-4B）
EDGE_API_KEY=your-modelscope-api-key
EDGE_BASE_URL=https://your-modelscope-endpoint

# 阿里云 Prometheus（SLS）
PROMETHEUS_URL=your-prometheus-url
ALIYUN_ACCESS_KEY_ID=your-access-key-id
ALIYUN_ACCESS_KEY_SECRET=your-access-key-secret

# 飞书集成
FEISHU_APP_ID=your-feishu-app-id
FEISHU_APP_SECRET=your-feishu-app-secret
FEISHU_CHAT_ID=your-feishu-chat-id
FEISHU_VERIFICATION_TOKEN=your-verification-token

# Redis（可选）
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Grafana
GRAFANA_URL=your-grafana-url
GRAFANA_API_KEY=your-grafana-api-key
```

### 初始化知识库

将运维文档放入 `data/k8s_docs/` 目录，执行向量化写入：

```bash
python ingest.py
```

### 生成测试数据（可选）

```bash
python data/generate_data.py
```

## 运行

```bash
python main.py
```

按提示选择运行模式，也可直接指定：

```bash
python main.py --mode rag      # RAG 知识库问答
python main.py --mode agent    # 单智能体（全工具集）
python main.py --mode hybrid   # 混合边云双模型
python main.py --mode collab   # 多智能体协作
python main.py --mode bot      # 飞书机器人（监听端口 8088）
```

## 运行模式说明

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| `rag` | 知识库查询、文档问答 | ChromaDB 语义检索，带对话历史 |
| `agent` | 通用运维任务 | 全工具调用，Redis 会话记忆 |
| `hybrid` | 资源受限 + 复杂决策 | 边缘感知→云端决策，两阶段流水线 |
| `collab` | 复杂故障诊断与处置 | 诊断专家 + 运维执行，角色分离 |
| `bot` | 企业团队即时响应 | 飞书 Webhook，`/feishu/event` 接口 |

## 可用工具

| 工具 | 功能 |
|------|------|
| `system_monitor` | 实时 CPU/内存/磁盘监控 |
| `alarm_reduction` | 告警降噪去重分析 |
| `data_analysis` | CSV/JSON/Excel 多格式数据分析 |
| `k8s_pod_restart` | Kubernetes Pod 重启（带审计日志）|
| `cpu_trend` | CPU 趋势分析（代理 Prometheus）|
| `prometheus_query` | Prometheus 指标查询与异常检测 |
| `feishu_notify` | 飞书消息卡片推送 |
| `knowledge_base` | 运维知识库语义查询 |

## 技术栈

- **AI 框架**：LangChain、LangGraph
- **大模型**：Qwen-Max（云端）、Qwen 3-4B（边缘）via DashScope/ModelScope
- **向量数据库**：ChromaDB + `text2vec-base-chinese` Embedding
- **数据分析**：Pandas、NumPy
- **存储**：Redis（会话记忆）、ChromaDB（向量检索）
- **监控集成**：Prometheus、Grafana
- **企业集成**：飞书 Open API
- **Web 框架**：Flask（飞书 Webhook）
- **系统监控**：psutil

## 系统架构

```
用户输入
   │
   ├── RAG 模式 ──→ 向量检索 → LLM → 回复
   │
   ├── Agent 模式 ──→ 工具调用循环 → 回复
   │                    ├── 系统监控
   │                    ├── Prometheus
   │                    ├── K8s 操作
   │                    └── 知识库查询
   │
   ├── Hybrid 模式 ──→ 边缘模型感知 → 云端模型决策
   │
   ├── Collab 模式 ──→ 诊断智能体 → 执行智能体
   │
   └── Bot 模式 ──→ 飞书 Webhook → Agent → 飞书回复
```

## 开发与测试

```bash
# 测试 LLM API 连通性
python test_llm_apis.py

# 测试工具技能
python test_skills.py

# 测试 Grafana 集成
python test_grafana.py
```

## 注意事项

- Redis 为可选依赖，未配置时自动降级为内存模式
- HuggingFace Embedding 模型首次运行自动下载，之后使用本地缓存（`models/model_cache/`）
- 飞书机器人模式需要将服务部署到公网可访问地址，并在飞书开放平台配置 Webhook 回调 URL
- K8s 运维操作（Pod 重启等）需要配置对应的 kubeconfig 权限

## 作者

**王梓涵** · 914132612@qq.com