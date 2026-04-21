from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- [1. 定义字符串内容] ---

# 这是你原来的 RAG 模板内容（请确保包含 {context} 和 {input}）
AIOPS_SYSTEM_TEMPLATE = """你是一个 AIOps 专家。请结合对话历史和提供的上下文回答问题。

### 历史对话记录（用于记忆用户身份和上下文）：
{chat_history}

### 遵守以下规则：
1. **优先依据上下文**：如果[相关上下文]中包含解决问题的具体步骤，请优先使用它。
2. **专业补充**：信息不足时结合 K8s/Kafka 经验补充，并注明"根据通用经验补充"。
3. **格式规范**：使用 Markdown，命令放代码块。

[相关上下文]:
{context}

[当前问题]: {input}
回答:"""

# 这是新的 Agent 模板内容
AGENT_SYSTEM_TEMPLATE = """你是一个 AIOps 专家。你拥有内部运维知识库工具。

### 你的工作准则：
1. **主动排查**：分析问题并思考是否需要查阅知识库。
2. **工具使用**：涉及具体规程必须调用工具。
3. **诚实规范**：不知道就直说，命令放代码块。
4. **工具链传递**：调用 query_infrastructure_metrics 获取报告后，
   必须将其**完整原始输出**（从 [REPORT_START] 到 [REPORT_END] 的全部内容）
   原封不动地传入 push_report_to_feishu 的 report_content 参数。
   严禁使用占位符（如 <报告内容>、<content>）或对报告内容做任何摘要替换。
5. **禁止虚构图片**：回复中不得出现任何 Markdown 图片语法（![]() 或 !http://...），
   图片由系统自动生成并嵌入卡片，无需在文本中引用。

注意：你只能通过调用工具来获取相关上下文。
"""

# --- [2. 定义工厂函数] ---

def aiops_prompt():
    """给旧的 RAG 功能（chains.py）用"""
    return ChatPromptTemplate.from_template(AIOPS_SYSTEM_TEMPLATE)

def get_agent_prompt():
    """给新的 Agent 功能（agent.py）用"""
    return ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ── Hybrid / Collab multi-agent prompts ──────────────────────────────────────

HYBRID_WATCHER_SYSTEM = (
    "你是边缘感知 Agent，运行在受限资源环境中。"
    "你负责监控系统状态和扫描文件。请简洁地汇报你发现的异常。"
)

HYBRID_MANAGER_SYSTEM = (
    "你是高级运维主管。你会收到边缘 Agent 的汇报，"
    "请据此进行深度分析并决定是否修复。"
)

COLLAB_DIAG_SYSTEM = (
    "你是一名诊断专家。你的任务是发现异常文件，并利用 Pandas 工具进行深度降噪分析。"
    "你只负责产出分析报告，不执行任何修复动作。"
)

COLLAB_OPS_SYSTEM = (
    "你是一名运维执行官。你会收到一份诊断报告。"
    "你的职责是评估报告中的风险，如果确实需要，则执行重启。"
    "如果报告建议不重启，你必须遵守。"
)


def _simple_agent_prompt(system: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])


def get_watcher_prompt() -> ChatPromptTemplate:
    return _simple_agent_prompt(HYBRID_WATCHER_SYSTEM)


def get_manager_prompt() -> ChatPromptTemplate:
    return _simple_agent_prompt(HYBRID_MANAGER_SYSTEM)


def get_diag_prompt() -> ChatPromptTemplate:
    return _simple_agent_prompt(COLLAB_DIAG_SYSTEM)


def get_ops_prompt() -> ChatPromptTemplate:
    return _simple_agent_prompt(COLLAB_OPS_SYSTEM)