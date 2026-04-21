# AIOps 智能运维助手

基于 LangChain 构建的企业级 AI 运维平台，集成多模型协作、RAG 知识库、实时监控与飞书机器人，实现智能化的基础设施运维与异常处理。内置 GPU 资源全生命周期管理：从申请审批、自动创建 DSW 实例，到超时告警、费用追踪、空转检测，全流程自动化。

## 功能特性

### 核心能力
- **多运行模式**：RAG 问答、单智能体、混合双模型、多智能体协作、飞书机器人
- **混合边云架构**：轻量边缘模型（Qwen 3-4B）负责感知，云端大模型（Qwen-Max）负责决策
- **多智能体协作**：诊断专家与运维执行两个角色独立运作，职责分离
- **RAG 知识库**：ChromaDB 向量数据库 + 中文 Embedding，支持运维文档语义检索
- **持久化对话记忆**：基于 Redis 的会话历史（最近 20 条），支持优雅降级
- **企业集成**：Prometheus 指标查询、Grafana 看板、飞书 Webhook 通知、Jira 工单系统
- **告警降噪**：Redis 去重，防止重复告警轰炸

### GPU 资源管理（全生命周期）
- **自助申请**：飞书交互式卡片申请 GPU 实例，支持快选（1/4/8 GPU）和自定义参数
- **身份绑定**：每位用户绑定个人阿里云 AK/SK，创建实例时以本人账号操作
- **Jira 工单**：每次申请自动创建 Jira 工单，完整记录申请信息和状态流转
- **审批流**：大规格申请（>4 GPU 或 >48h）自动发送管理员审批卡片，支持一键批准/拒绝
- **自动创建**：调度器轮询 Jira 待办工单，审批通过后 20 秒内自动调用 PAI DSW API 创建实例
- **就绪通知**：实例进入 Running 状态后，立即向申请人推送就绪卡片
- **超时预警**：实例到期前 15 分钟推送续期/停止卡片，无响应则自动停止
- **GPU 空转检测**：对接 Prometheus 监控 GPU SM 利用率，持续低于阈值时推送节省费用提醒
- **费用估算**：申请时实时计算费用预估，停止时告知实际费用
- **月度配额**：按用户统计当月 GPU·小时使用量，超额自动拦截
- **我的实例**：发送「我的实例」即可查看运行状态、费用，并在线延续或停止
- **每日早报**：每天 09:00 向有运行实例的用户推送到期汇总

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
│   ├── dsw_scheduler.py        # DSW 工单调度器（Jira轮询 + 实例监控 + 早报）
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
│   ├── rag_tool.py             # 知识库查询工具
│   ├── gpu_advisor_tool.py     # GPU 集群智能优化顾问
│   ├── pai_dsw_tool.py         # 阿里云 PAI DSW 实例管理（Python SDK）
│   └── jira_tool.py            # Jira 工单操作（GPU 申请工单 CRUD）
│
├── utils/
│   ├── llm_factory.py          # LLM 工厂（云端/边缘模型）
│   ├── vector_store.py         # ChromaDB + HuggingFace Embedding
│   ├── redis_client.py         # Redis 单例（会话/缓存/去重）
│   ├── aliyun_auth.py          # 阿里云鉴权
│   └── chart_builder.py        # 实时指标趋势图生成
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
- Redis（推荐，用于会话记忆、GPU 状态追踪和配额管理）

### 安装依赖

```bash
git clone <repo-url>
cd langchaindev

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 云端大模型（阿里云 DashScope / Qwen-Max）
OPENAI_API_KEY=your-dashscope-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 边缘轻量模型（ModelScope / Qwen 3-4B）
EDGE_API_KEY=your-modelscope-api-key
EDGE_BASE_URL=https://your-modelscope-endpoint

# 阿里云 Prometheus（SLS 托管）
PROMETHEUS_URL=https://your-workspace.cn-hangzhou.log.aliyuncs.com/...
ALIYUN_ACCESS_KEY_ID=your-access-key-id
ALIYUN_ACCESS_KEY_SECRET=your-access-key-secret

# 飞书企业自建应用
FEISHU_APP_ID=cli_your-feishu-app-id
FEISHU_APP_SECRET=your-feishu-app-secret
FEISHU_CHAT_ID=oc_your-feishu-chat-id
FEISHU_VERIFICATION_TOKEN=your-verification-token
# 飞书卡片构建器模板 ID（可选，留空则使用内置 action buttons 卡片）
FEISHU_GPU_CARD_TEMPLATE_ID=
FEISHU_AK_REGISTER_TEMPLATE_ID=

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0

# Grafana
GRAFANA_URL=http://your-grafana-host:3000
GRAFANA_API_KEY=glsa_your-grafana-api-key
GRAFANA_DATASOURCE_UID=your-datasource-uid

# Jira（GPU 工单系统）
JIRA_URL=https://jira.your-company.com
JIRA_PAT=your-jira-personal-access-token
JIRA_PROJECT_KEY=GPU
JIRA_ISSUE_TYPE=Task

# 阿里云 PAI DSW（Python SDK）
PAI_DSW_ACCESS_KEY_ID=your-aliyun-ak
PAI_DSW_ACCESS_KEY_SECRET=your-aliyun-sk
PAI_DSW_REGION_ID=cn-hangzhou
PAI_DSW_WORKSPACE_ID=your-workspace-id
PAI_DSW_RESOURCE_ID=your-resource-id
PAI_DSW_DEFAULT_IMAGE=dsw-registry-vpc.../pai/python:3.12-gpu-...

# GPU 管理参数
DSW_IDLE_WARN_HOURS=1           # 实例运行超时告警时间（小时）
DSW_IDLE_STOP_MINUTES=30        # 告警后无响应自动停止（分钟）
ADMIN_FEISHU_OPEN_ID=           # 管理员飞书 open_id（审批大规格申请）
GPU_PRICE_PER_HOUR=35.0         # 定价（元/GPU·小时）
GPU_QUOTA_HOURS_PER_MONTH=200.0 # 每人月度配额上限（GPU·小时）
GPU_IDLE_THRESHOLD_PCT=5.0      # 空转检测阈值（%）
GPU_IDLE_WARN_MINUTES=30        # 空转持续多久告警（分钟）
```

### 初始化知识库

将运维文档放入 `data/k8s_docs/` 目录后执行：

```bash
python ingest.py
```

## 运行

```bash
python main.py                  # 交互式模式选择
python main.py --mode rag       # RAG 知识库问答
python main.py --mode agent     # 单智能体（全工具集）
python main.py --mode hybrid    # 混合边云双模型
python main.py --mode collab    # 多智能体协作
python main.py --mode bot       # 飞书机器人（端口 8088）
```

## 运行模式说明

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| `rag` | 知识库查询、文档问答 | ChromaDB 语义检索，带对话历史 |
| `agent` | 通用运维任务 | 全工具调用，Redis 会话记忆 |
| `hybrid` | 资源受限 + 复杂决策 | 边缘感知→云端决策，两阶段流水线 |
| `collab` | 复杂故障诊断与处置 | 诊断专家 + 运维执行，角色分离 |
| `bot` | 企业团队即时响应 | 飞书 Webhook，自动 GPU 资源调度 |

## GPU 资源申请流程

```
用户发送「申请GPU」
       │
       ▼
是否已注册个人 AK/SK？
       │
   否 ──┤──→ 推送「绑定账号」卡片 → 用户填写 AK/SK → 保存到 Redis
       │
   是  ▼
  推送 GPU 申请卡片（快选规格 / 自定义参数）
       │
       ▼
  用户填写：实例名 / GPU数 / 时长 / 用途 / 镜像（可选）
       │
       ▼
  配额校验（当月 GPU·小时是否超上限）
       │
  超额 ─┤──→ 返回「配额不足」提示，申请终止
       │
  正常 ▼
  是否大规格（>4 GPU 或 >48h）？
       │
  是 ──┤──→ 创建 Jira 工单（needs_approval=true）
       │         │
       │         └──→ 发审批卡片给管理员
       │                   │
       │             批准 ─┤──→ Redis 写入 approved 标记
       │             拒绝 ─┘──→ 通知申请人，工单关闭
       │
  否   ▼
  创建 Jira 工单（status=待办）
       │
       ▼
  调度器（20 秒轮询）发现新工单
       │
       ▼
  调用 PAI DSW Python SDK，以申请人个人 AK 创建实例
       │
       ▼
  轮询实例状态（最多 6 分钟）
       │
  Running ──→ 推送「实例就绪」卡片给申请人
  Failed  ──→ 推送异常通知，运维人员介入
       │
       ▼
  实例运行中：每 5 分钟检查
  ├── 到期前 15 分钟：推送「续期 / 停止」卡片
  ├── 无响应超过 DSW_IDLE_STOP_MINUTES：自动停止
  └── GPU 利用率 < 阈值 且持续 > GPU_IDLE_WARN_MINUTES：推送空转提醒
```

## 飞书机器人快捷指令

| 发送内容 | 触发动作 |
|----------|----------|
| `申请GPU` / `需要GPU` / `我要训练` | 弹出 GPU 申请卡片 |
| `我的实例` / `my dsw` | 查看运行中实例，支持在线延续/停止 |
| `注册AK: AccessKeyId AccessKeySecret` | 绑定个人阿里云凭证 |
| 其他任意问题 | 通过 Agent 工具链回答，并附实时指标趋势图 |

> 在飞书开放平台配置两个回调地址：
> - 消息事件：`https://your-domain/feishu/event`
> - 卡片回调：`https://your-domain/feishu/card_action`

## 调度器架构（`core/dsw_scheduler.py`）

`DSWScheduler` 随 Bot 启动，包含三个后台线程：

| 线程 | 间隔 | 职责 |
|------|------|------|
| `jira-poll` | 20 秒 | 拉取 Jira 待办工单 → 验证注册/配额/审批 → 创建 DSW 实例 |
| `dsw-check` | 5 分钟 | 检查所有运行中实例的超时状态和 GPU 空转指标 |
| `morning-report` | 每天 09:00 | 向有实例的用户推送今日到期汇总 |

Redis Key 设计：

```
dsw:ticket:{ticket_key}       → 实例状态（instance_id / open_id / start_ts 等）
feishu:user_ak:{open_id}      → 用户个人 AK/SK（JSON）
feishu:gpu_state:{chat_id}    → 申请对话中间状态（10 分钟 TTL）
gpu:quota:{open_id}:{YYYYMM}  → 当月已用 GPU·小时数（月底到期）
dsw:approved:{ticket_key}     → 管理员已审批标记
dsw:pending_reg:{ticket_key}  → 未注册通知去重（1 小时 TTL）
```

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
| `gpu_advisor` | GPU 集群利用率/散热/调度/成本建议 |
| `pai_dsw` | PAI DSW 实例查询、启停、创建、资源发现 |
| `manage_jira` | Jira GPU 工单查询、评论、关闭 |

## 技术栈

- **AI 框架**：LangChain 0.3、LangGraph
- **大模型**：Qwen-Max（云端）、Qwen 3-4B（边缘）via DashScope/ModelScope
- **向量数据库**：ChromaDB + `shibing624/text2vec-base-chinese`
- **数据分析**：Pandas、NumPy、Matplotlib
- **存储**：Redis（会话/状态/配额）、ChromaDB（向量检索）
- **监控集成**：Prometheus（SLS）、Grafana
- **工单系统**：Jira REST API v2
- **企业集成**：飞书 Open API（消息 + 交互卡片）
- **GPU 平台**：阿里云 PAI DSW（Python SDK `alibabacloud-paistudio20220112`）
- **Web 框架**：Flask
- **系统监控**：psutil

## 系统架构

```
飞书用户消息
      │
      ▼
Flask /feishu/event
      │
      ├── AK 注册指令 ──────────────────────────────→ Redis 保存凭证
      │
      ├── GPU 申请意图 ──→ 注册检查 ──→ 申请卡片 ──→ Jira 工单
      │                                                    │
      │                                              DSWScheduler
      │                                                    │
      │                                         PAI DSW Python SDK
      │                                                    │
      │                                            飞书实例就绪推送
      │
      ├── 「我的实例」 ──→ Redis 查询 ──→ 实例状态卡片
      │
      └── 通用问题 ──→ Agent 工具链 ──→ 飞书回复（含趋势图）
                          │
                          ├── system_monitor
                          ├── prometheus_query
                          ├── k8s_pod_restart
                          ├── knowledge_base
                          ├── pai_dsw
                          └── gpu_advisor
```

## 部署（公网发布）

### 方案一：公网服务器 + nginx + HTTPS

```bash
# 1. 安装依赖并启动
pip install -r requirements.txt
python main.py --mode bot          # 监听 0.0.0.0:8088

# 2. nginx 反向代理配置（/etc/nginx/sites-available/aiops）
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location /feishu/ {
        proxy_pass http://127.0.0.1:8088;
        proxy_set_header Host $host;
    }
}

# 3. 申请 SSL 证书
certbot --nginx -d your-domain.com

# 4. systemd 后台运行
[Unit]
Description=AIOps Bot

[Service]
WorkingDirectory=/opt/langchaindev
ExecStart=/opt/langchaindev/.venv/bin/python main.py --mode bot
Restart=always

[Install]
WantedBy=multi-user.target
```

飞书开放平台配置：
1. **事件订阅 → 请求地址**：`https://your-domain.com/feishu/event`
2. **机器人 → 卡片请求网址**：`https://your-domain.com/feishu/card_action`
3. **权限管理** 开通：`im:message:receive_v1`、`im:message`、`im:message.group_at_msg`

### 方案二：内网穿透（开发测试用）

```bash
# ngrok
ngrok http 8088
# 将 https://xxxx.ngrok.io 填入飞书后台
```

### 多用户使用说明

每位用户首次使用时需绑定个人阿里云 AK/SK（**请在私聊中操作，勿在群里发送**）：

```
注册AK: LTAI5tXXXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

绑定后发送消息请务必立即撤回，避免密钥泄漏。Bot 将以各自账号在 PAI DSW 控制台创建实例，实例「创建人」显示本人 RAM 用户名。

## 安全注意事项

- **AK/SK 明文存储**：当前实现将用户 AK/SK 明文写入 Redis，适用于内部受控环境。生产环境建议：
  - 使用 AES/Fernet 加密后再存储（需自管密钥）
  - 或接入阿里云 STS AssumeRole，仅存储 RAM 角色 ARN，临时凭证按需获取
- **飞书 Verification Token**：务必配置 `FEISHU_VERIFICATION_TOKEN`，防止伪造事件请求
- **K8s 操作**：Pod 重启工具内置命名空间白名单，禁止对 `prod`、`production`、`kube-system` 执行操作
- **Jira PAT**：使用个人访问令牌（Personal Access Token）而非账号密码，定期轮换

## 开发与测试

```bash
# 测试 LLM API 连通性
python test_llm_apis.py

# 测试工具技能
python test_skills.py

# 测试 Grafana 集成
python test_grafana.py

# 验证飞书 Bot（需服务已启动）
curl -X POST http://localhost:8088/health
```

## 注意事项

- Redis 为推荐依赖，未配置时会话历史、配额追踪、GPU 状态均不可用
- HuggingFace Embedding 模型首次运行自动下载，之后使用本地缓存（`models/model_cache/`）
- PAI DSW Python SDK 需要 `alibabacloud-paistudio20220112` 包，无需安装 Node.js
- 飞书机器人模式需将服务部署到公网可访问地址，并在飞书开放平台正确配置回调 URL
- K8s 运维操作需配置对应的 kubeconfig 权限

## 作者

**王梓涵** · 914132612@qq.com
