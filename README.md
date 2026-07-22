# AIOps 智能运维助手

基于 LangChain 0.3 构建的多模式 AI 运维平台。五种运行模式（RAG / 单智能体 / 混合边云 / 多智能体协作 / 飞书 Webhook Bot）之上是一层封装了阿里云（PAI DSW / ECS / OSS / SLS / RAM / Prometheus）、火山引擎（TOS / vePFS）、Jira、GitHub、K8s 与本地 ChromaDB 的工具层。面向算法团队的日常运维：GPU 资源全生命周期、跨云/同云数据搬运、CPFS/vePFS 数据流动、OSS 权限最小化、算力效率与容量治理。每个用户的云 API 调用走 STS AssumeRole，Bot 不用共享 AK。

功能按「新→旧」排列。

## 功能特性

### 数据外采与数据搬运（最新）
- **临时 AK/SK 发放**（`core/temp_ak_issuance`）：飞书审批「数据外采访问凭证申请」通过 → 给**外部方**发一组时限 OSS 凭证（生效 + 到期时间窗，policy 内嵌、服务端逐调用判时间，泄漏也随到期自动失效 + 到期硬删）。权限 `read`/`download`/`write` 三者正交（read=列不下载 / download=才给下载 / write=上传无删除）。按有效期分流：`到期−now ≤ TEMP_AK_STS_MAX_SECONDS`（默认 12h）走 **STS 单发**（含 Token 到点自灭），超出走 **方案 B**（RAM 长期 AK + policy 时间窗 + 到期硬删）；线上 `TEMP_AK_STS_MAX_SECONDS=0` 全走方案 B。延长/撤销二合一审批；企业名自动转拼音登录名 + 中文显示名；凭证走审批评论、以管理员身份下发，secret 不落盘/日志。全局审批白名单只处理指定审批 code。三入口：飞书审批（**唯一发凭证路径**）+ Agent 工具 `manage_temp_ak`（`plan`/`status`/`revoke`，无 `issue`）+ CLI。当前仅阿里 OSS，火山云 TOS 为 P1 规划中。
- **跨云数据迁移**（`core/transfer`）：给一条路径自动判方向、推目的桶。方向决定引擎——搬入 OSS 用阿里「在线迁移服务」(hcs_mgw)，搬入 TOS 用火山「迁移服务」。一期打通 `TOS→OSS`；>1TB 走管理员审批。
- **同云桶间迁移**（`core/bucket_transfer`）：同账号跨 region/桶搬运，阿里 `oss://→oss://`、火山 `tos://→tos://`；OSS 自动探测源/目的 region，跨 region 走公网。与跨云迁移独立命名空间，仅复用其引擎。
- **CPFS/NAS 数据流动**（`core/cpfs_dataflow`）：阿里 NAS DataFlow **预热**（OSS→CPFS，Import）/**沉降**（CPFS→OSS，Export）。只查现有 DataFlow + 提交任务，按目标子目录匹配最长前缀绑定；智算版/通用版自动分支。飞书选择器从发现的 CPFS↔OSS 绑定里选。
- **火山 vePFS/TOS 数据流动**（`core/vepfs_dataflow`）：火山「文件存储 vePFS」**预热**（TOS→vePFS）/**沉降**（vePFS→TOS）。无持久 DataFlow 对象，提交任务直接带桶/前缀，方向由地址类型自动判断。与 CPFS 共用三步级联向导卡（选云→选地区→表单）。
- **数据流动/迁移在途任务对账**：调度器每 2 分钟对账在途任务。后台轮询线程随容器重启会死，对账线程随容器复活兜底——重启后任务完成也会自动补推结果卡（跑完必通知）。在线推送与对账推送共用 Redis `SET NX` 闸门，跨线程只推一次。
- **三入口共享核心**：以上迁移/流动能力均是「飞书卡片 + Agent 工具 + CLI」三入口共用一套编排。飞书发意图→确认卡→后台推进度/结果卡（优先推发起人，空则降级配置频道）；Agent 工具 `manage_transfer`/`manage_cpfs_dataflow`/`manage_vepfs_dataflow`；CLI `python -m core.<pkg>.cli`（dry-run 默认）。

### 权限、算力与容量治理
- **OSS 权限最小化同步**（`core/oss_perm`）：飞书「舞肌算法组权限统计」多维表格 → 每人一条最小权限自定义 RAM 策略并挂到其 RAM 用户。桶级/目录级两档粒度（先放桶级观察再收紧）。飞书表单卡选择性下发（粒度单选默认桶级 + 成员多选默认全选 + 一键确认）；CLI `--apply`/`--audit`（对账多授/少授/孤儿）。
- **集群算力效率（MFU）日报**（`cluster_mfu`）：多区域算力效率 + 容量 + 调度日报，交互式飞书卡片可切区域，24h Redis 快照，按钮回调只读缓存秒回。
- **GPU 卡分布大盘**（`tools/aliyun/gpu_distribution`）：地区×卡型分布 + 每用户在算卡数 + 近 N 小时趋势，实时 HTML 页面 `/gpu/distribution`（token 门禁），15s 陈旧后台单飞刷新；飞书问「谁在用卡/卡分布」回摘要卡 + 链接。
- **容量巡检**：定时扫 OSS/TOS 各目标子目录大小 + 与上次快照的增量，超阈值标红推卡；同时把每厂家总量 upsert 进飞书多维表格（巡检快照→厂家总量→批次明细 三表关联，去重旧行）。
- **数据集大盘维护**（`core/dataset_dashboard`）：定时遍历飞书「数据集大盘」多维表格现有行，按 uri 扫对象存储回填脚本负责列（状态/云/厂商/时长/数据集类型），只填数据、不改表结构、绝不碰人工分析列。
- **STS 多租户凭证**：Bot 从不用全局 AK 做用户动作。按飞书 open_id 解析 RAM 用户 → 组映射角色 ARN → Master AK `AssumeRole` 换临时凭证（Redis 缓存，到期前 5 分钟自动刷新）。用户自填 AK 经 Fernet 加密后入 Redis。

### 平台基础能力
- **多运行模式**：RAG 问答、单智能体、混合边云双模型、多智能体协作、飞书 Webhook Bot。
- **三层工具路由**（`core/agent`）：专有名词快路径（零延迟）→ LLM 意图路由（temp 0，`@lru_cache`）→ 关键词兜底，只挂选中意图的工具组。
- **混合边云架构**：轻量边缘模型（Qwen3-4B）感知、序列化 JSON 观测，云端大模型（Qwen-Max）决策。
- **多智能体协作**：诊断专家（只读工具）→ 运维执行（写工具），职责分离。
- **RAG 知识库**：ChromaDB + 中文 Embedding（`text2vec-base-chinese`），运维文档语义检索。
- **对话滚动摘要记忆**：Redis 逐字保留最近 20 条，更早对话由 GLM 压成滚动摘要（`agent:chat_summary`）作前缀注入，长对话不再硬截断；会话历史 7 天空闲过期，失败静默降级。
- **企业集成**：Prometheus 指标、Grafana 看板、飞书消息卡片、Jira 工单、GitHub PR/commit/sprint、K8s Pod 重启。
- **告警降噪**：Redis 去重（24h TTL 集合），防止重复告警轰炸。

### GPU 资源管理（全生命周期）
- **自助申请**：飞书交互式卡片申请 GPU 实例，支持快选（1/4/8 GPU）和自定义参数。
- **身份绑定**：每位用户绑定个人阿里云 AK/SK（Fernet 加密），创建实例时以本人账号（STS）操作。
- **Jira 工单**：每次申请自动创建 Jira 工单，记录申请信息与状态流转。
- **审批流**：大规格申请（>4 GPU 或 >48h）自动推管理员审批卡片，一键批准/拒绝。
- **自动创建**：调度器轮询 Jira 待办工单，审批通过后自动调用 PAI DSW 创建实例。
- **就绪通知**：实例进入 Running 后立即向申请人推送就绪卡片。
- **超时预警（默认关）**：到期自动停止默认交给阿里云工作空间的利用率关机（`DSW_IDLE_STOP_ENABLED=false`）；置 `true` 恢复到期前 15 分钟续期/停止卡片 + 无响应自动停止。
- **GPU 空转检测**：对接 Prometheus GPU SM 利用率，持续低于阈值推送节费提醒（独立于上面的开关，始终生效）。
- **费用估算与月度配额**：申请时预估、停止时结算；按用户统计当月 GPU·小时，超额拦截。
- **我的实例**：发送「我的实例」查看运行状态/费用，在线续期或停止。
- **每日早报**：每天北京 09:00 推两张卡——各用户实例汇总 + 集群 MFU 汇总。

## 项目结构

```
langchaindev/
├── main.py                     # 主入口，运行模式选择器（--mode / --session）
├── ingest.py                   # 文档向量化写入脚本
│
├── config/
│   └── settings.py             # 全局配置（路径、模型参数、API）
│
├── core/
│   ├── agent.py                # 单智能体（三层工具路由 + Redis 记忆）
│   ├── intent_router.py        # LLM 意图路由（temp 0，@lru_cache）
│   ├── rag_runner.py           # RAG 模式运行入口（会话 ID 隔离）
│   ├── hybrid_agents.py        # 混合双模型工作流（边缘→云端）
│   ├── multi_agent_system.py   # 多智能体协作（诊断 + 执行）
│   ├── feishu_bot/             # 飞书 Webhook 服务（Flask，含 routes/actions/messages/gpu_flow）
│   ├── dsw_scheduler.py        # 后台调度器（工单轮询/实例监控/早报/巡检/对账…）
│   ├── temp_ak_issuance/       # 临时 AK/SK 发放（审批门/policy 时间窗/STS 或方案B/清理）
│   ├── transfer/               # 跨云迁移（paths/engine_mgw/engine_tos/orchestrator/cli）
│   ├── bucket_transfer/        # 同云桶间迁移（oss→oss / tos→tos）
│   ├── cpfs_dataflow/          # 阿里 CPFS/NAS 数据流动（预热/沉降）
│   ├── vepfs_dataflow/         # 火山 vePFS/TOS 数据流动（预热/沉降）
│   ├── dataflow_cards.py       # CPFS/vePFS 共用三步级联向导卡
│   ├── oss_perm/               # OSS 权限最小化同步（permsync/cards/actions）
│   ├── cpfs_dataflow.py        # (Phase-3 SINKING) NAS DataFlow Export 引擎
│   ├── capacity_monitor.py     # 容量巡检
│   ├── capacity_bitable.py     # 容量结果写飞书多维表格
│   ├── dataset_dashboard.py    # 数据集大盘多维表格维护
│   ├── llm_factory.py          # LLM 工厂（云端/边缘模型，@lru_cache）
│   ├── vector_store.py         # ChromaDB + HuggingFace Embedding（@lru_cache）
│   └── prompts_*.py            # RAG / Agent / Hybrid / Collab 系统提示词
│
├── tools/                      # 按厂商/领域分子包，单一来源 tools/__init__.py
│   ├── __init__.py             # ALL_TOOLS + TOOL_GROUPS（导入期校验一致性）
│   ├── base_tool.py            # 工具基类（审计日志、数据路径解析）
│   ├── aliyun/                 # pai_dsw / ecs / oss / sls / ram / prometheus /
│   │                           #   gpu_advisor / gpu_training_advisor / dsw_inspector /
│   │                           #   cluster_health / cluster_mfu / gpu_distribution
│   ├── volcano/                # tos（容量）/ vepfs_dataflow
│   ├── temp_ak_issuance/       # manage_temp_ak
│   ├── transfer/               # manage_transfer
│   ├── cpfs/                   # manage_cpfs_dataflow
│   ├── feishu/                 # notify（消息卡片）+ cards（卡片原语）
│   ├── jira/                   # ticket + workflow
│   ├── github/                 # workflow
│   ├── knowledge/              # rag 知识库检索
│   └── ops/                    # system / analysis / monitor / k8s
│
├── utils/
│   ├── redis_client.py         # Redis 单例（会话/缓存/去重）
│   ├── logger.py               # 结构化日志（trace_id + 飞书告警回调）
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
└── sessions/                   # RAG 对话历史文件（按会话 ID 隔离）
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
DSW_IDLE_STOP_ENABLED=false     # 到期自动停止总开关（默认关，利用率关机交给阿里云工作空间）
DSW_IDLE_WARN_HOURS=1           # 实例运行超时告警时间（小时，仅开关开启时生效）
DSW_IDLE_STOP_MINUTES=30        # 告警后无响应自动停止（分钟，仅开关开启时生效）
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
python main.py                          # 交互式模式选择
python main.py --mode rag               # RAG 知识库问答（交互式选择会话 ID）
python main.py --mode rag --session ops # RAG 模式，指定会话 ID
python main.py --mode agent             # 单智能体（全工具集）
python main.py --mode hybrid            # 混合边云双模型
python main.py --mode collab            # 多智能体协作
python main.py --mode bot               # 飞书机器人（端口 8088）
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
  ├── 到期前 15 分钟：推送「续期 / 停止」卡片      （仅 DSW_IDLE_STOP_ENABLED=true）
  ├── 无响应超过 DSW_IDLE_STOP_MINUTES：自动停止   （仅 DSW_IDLE_STOP_ENABLED=true）
  └── GPU 利用率 < 阈值 且持续 > GPU_IDLE_WARN_MINUTES：推送空转提醒（始终生效）
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

`DSWScheduler` 随 `bot` 模式启动，后台线程如下（部分按配置开关）：

| 线程 | 间隔 | 职责 |
|------|------|------|
| `jira-poll` | 20 秒 | 拉取 Jira 待办工单 → 验证注册/配额/审批 → 创建 DSW 实例 |
| `dsw-check` | 5 分钟 | 检查运行中实例的超时状态和 GPU 空转指标 |
| `morning-report` | 每天北京 09:00 | 推两卡：各用户实例汇总 + 集群 MFU 汇总 |
| `dataflow-reconcile` | 2 分钟 | 对账在途迁移/数据流动任务，重启后完成也补推结果卡 |
| `capacity-monitor` | N 小时（对齐整点，opt-in） | 扫 OSS/TOS 容量 + 增量推卡 + 写多维表格 |
| `dataset-dashboard` | N 小时（对齐整点，opt-in） | 维护飞书「数据集大盘」多维表格 |
| `oss-perm-push` | 每天北京 HOUR:20（opt-in） | OSS 权限对账，把待同步权限推群（带批准按钮） |
| `temp-ak-cleanup` | 每天北京 HOUR:35（opt-in） | 扫临时外采凭证，对已到期的方案 B 凭证硬删 AK+policy+user |

对账线程随容器一起重启复活，因此后台轮询线程即使随重启死掉，任务跑完仍会被对账补推——在线推送与对账推送共用 Redis `SET NX` 闸门，跨线程只推一次。

主要 Redis Key 命名空间：

```
dsw:ticket:{ticket_key}                → 工单/实例状态（7 天 TTL）
agent:chat_history:{session_id}        → 会话历史（最近 20 条，7 天空闲 TTL）
agent:chat_summary:{session_id}        → 更早对话的滚动摘要（GLM 压缩，7 天空闲 TTL）
user:ak:{open_id}                      → 用户个人 AK/SK（Fernet 加密，30 天闲置 TTL）
gpu:quota:{open_id}:{YYYYMM}           → 当月已用 GPU·小时数
aliyun:sts:{open_id}:{role_arn}        → STS 临时凭证缓存（到期前 5 分钟刷新）
transfer:job:{job_id}                  → 跨云迁移状态机（30 天 TTL）
bkt:transfer:job:{job_id}              → 同云桶间迁移状态机
cpfs:dataflow:job:{job_id}             → CPFS 预热/沉降任务状态
vepfs:dataflow:job:{job_id}            → vePFS 预热/沉降任务状态
temp_ak:grant:{grant_id}               → 临时外采凭证发放记录（只存 ak_id，绝不存 secret）
dataflow:notified:{job_id}             → 对账/在线推送共用的 SET NX 闸门
capacity:snapshot:{vendor}:{bucket}:{prefix}  → 上次容量巡检快照（算增量）
mfu:snapshot / gpu:dist:snapshot       → MFU / GPU 卡分布快照（区域切换、大盘秒回）
feishu:event_dedup:{event_id}          → Webhook 幂等去重（1h TTL）
```

## 可用工具

Agent 通过 `TOOL_GROUPS` 按意图选挂 1–2 组工具（`tools/__init__.py` 单一来源，导入期校验一致性）。

| 工具名 | 子包 | 功能 |
|--------|------|------|
| `manage_temp_ak` | temp_ak_issuance | 临时外采凭证 plan/status/revoke（发凭证仅走审批） |
| `manage_transfer` | transfer | 跨云迁移 plan/apply/status（TOS↔OSS） |
| `manage_cpfs_dataflow` | cpfs | 阿里 CPFS/NAS 数据流动 list/preheat/sink/status |
| `manage_vepfs_dataflow` | volcano | 火山 vePFS/TOS 数据流动 preheat/sink/status |
| `manage_oss` | aliyun | OSS 桶/对象、子目录大小、目录树（自动探测地域） |
| `manage_tos` | volcano | 火山 TOS 容量盘点 |
| `manage_ecs` | aliyun | 阿里云 ECS 实例管理 |
| `manage_sls` | aliyun | 阿里云 SLS 日志查询 |
| `cluster_mfu_report` | aliyun | 多区域算力效率（MFU）+ 容量 + 调度日报卡 |
| `cluster_health_report` | aliyun | GPU 集群全局监控看板（并行巡检 + 排行表） |
| `manage_pai_dsw` | aliyun | PAI DSW 实例查询、启停、创建、资源发现 |
| `inspect_dsw_instance` | aliyun | DSW 单实例健康巡检（GPU/温度/费用/建议） |
| `advise_gpu_cluster` | aliyun | GPU 集群利用率/散热/调度/成本建议 |
| `analyze_gpu_training` | aliyun | GPU 训练行为深度分析（Roofline 模型） |
| `query_infrastructure_metrics` | aliyun | Prometheus 指标查询与异常检测 |
| `manage_ram` | aliyun | 阿里云 RAM 用户权限查询与映射 |
| `manage_jira` | jira | Jira GPU 工单查询、评论、关闭 |
| `query_jira_workflow` | jira | Jira 算法工作流查询 |
| `query_github_workflow` | github | GitHub PR/commit/sprint 活动查询 |
| `query_knowledge` | knowledge | 运维知识库语义查询 |
| `push_report_to_feishu` | feishu | 飞书消息卡片推送 |
| `system_data_manager` | ops | 实时 CPU/内存/磁盘监控 |
| `compress_system_alarms` | ops | 告警降噪去重分析 |
| `analyze_node_cpu_trend` | ops | CPU 趋势分析 |
| `restart_k8s_service` | ops | Kubernetes Pod 重启（审计日志，生产命名空间保护） |

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

### 方案一：Docker Compose（推荐）

```yaml
# docker-compose.yml
services:
  bot:
    build: .
    restart: unless-stopped
    env_file: .env
    ports:
      - "8088:8088"
    volumes:
      - ./logs:/app/logs
      - ./sessions:/app/sessions
      - ./vector_db:/app/vector_db
      - ./models:/app/models
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

```bash
docker compose up -d          # 启动
docker compose logs -f bot    # 查看日志
```

### 方案二：公网服务器 + nginx + HTTPS

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

- **用户凭证加密 + STS**：用户自填 AK/SK 经 Fernet 加密后入 Redis（`BOT_CREDS_ENCRYPTION_KEY`）；用户发起的云动作走 STS AssumeRole 换临时凭证，Bot 不用共享 AK。Master AK 仅需 `STSAssumeRoleAccess + RAMReadOnlyAccess`。
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
