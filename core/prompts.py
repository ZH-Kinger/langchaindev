from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- [1. 定义字符串内容] ---

# 这是你原来的 RAG 模板内容（请确保包含 {context} 和 {input}）
AIOPS_SYSTEM_TEMPLATE = """你是一个 AIOps 专家。请结合对话历史和提供的上下文回答问题。

### 历史对话记录（用于记忆用户身份和上下文）：
{chat_history}

### 遵守以下规则：
1. **优先依据上下文**：如果[相关上下文]中包含解决问题的具体步骤，请优先使用它。
2. **专业补充**：信息不足时结合 K8s/Kafka 经验补充，并注明“根据通用经验补充”。
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
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Agent 专属草稿纸
    ])