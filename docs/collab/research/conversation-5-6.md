# 对话优化调研 #39：滚动摘要记忆（课题5）+ 飞书卡片流式更新（课题6）

任务 #39。为 dev 的两项对话体验优化查规格、取证、给最小侵入改法。
项目现状（只读源码所得）：
- **记忆**：`core/agent.py` 自管 Redis 历史（`agent:chat_history:{session_id}`，`MAX_HISTORY=20`=10 轮，
  FIFO `ltrim` 硬截断丢最旧，7 天 TTL），`_load_history` 还原成 `HumanMessage/AIMessage` list →
  作为 `chat_history` 传给 `create_tool_calling_agent` 的 prompt `MessagesPlaceholder`，
  `AgentExecutor.invoke({input, chat_history})` **非流式**。
- **飞书回复**：`messages.py::_process_message` 先 `_feishu_reply(message_id,"🤔 正在分析…")`，
  Agent `.invoke()` 跑完后 `_feishu_reply` 一次性发整段文本（可选附指标图）。
- **版本**：`langchain==0.3.0` / `langchain-core==0.3.0`（`requirements.txt`），**无 langgraph、无 langmem**。

---

## 课题 5：滚动摘要记忆（LangChain 0.3）

### TL;DR
- 【文档】旧的 `ConversationSummaryBufferMemory`/`ConversationSummaryMemory`/`ConversationBufferMemory`
  一族 **自 0.3.1 起弃用**（removal 目标 1.0），官方迁移指向 LangGraph checkpointer + 手动 trim/summary，
  **不再推荐用现成的"摘要缓冲记忆类"**。
- 【实测】`langchain_core.messages.trim_messages` 在**本项目已装的 0.3.0 里就有**，是当前官方推荐的"裁剪"原语。
  它既能按 token 裁（`token_counter=llm`），也能**按消息条数裁**（`token_counter=len`），
  且**可当 Runnable 用**（不传 messages 时返回 `RunnableLambda`，能塞进 LCEL 管道）。
- **官方没有"开箱即用的滚动摘要"落在 langchain-core**：summary-buffer 的"超阈值把最旧压成摘要"要么迁到
  LangGraph + `langmem.short_term.summarize_messages`（需新依赖 langgraph+langmem），要么**自己写一小段**
  （本项目结构下这是最小侵入选择）。

### 关键 API 取证
- 【实测】`trim_messages` 签名（0.3.0 docstring + 真跑）：
  `trim_messages(messages, *, max_tokens, token_counter, strategy="last", allow_partial=False,
  end_on=None, start_on=None, include_system=False, text_splitter=None)` → 返回 `List[BaseMessage]`。
  - `token_counter=len` → **按"消息条数"裁**（我真跑：`max_tokens=4, strategy="last", token_counter=len`
    把 10 条裁成最后 4 条）。要按真实 token 裁则 `token_counter=<llm 实例>`（走 `get_num_tokens_from_messages`）。
  - `strategy="last"` 保留最近的；`include_system=True` 保留 index 0 的 SystemMessage；
    `start_on="human"` 保证裁完第一条是 human（对话对齐，避免以 AIMessage 开头）。
  - `trim_messages(max_tokens=..., token_counter=len, ...)`（不给 messages）→ 返回 `RunnableLambda`，可 `|` 进链。
- 【文档】`ConversationSummaryMemory` 弃用说明：deprecated since 0.3.1，removal 1.0.0，指向迁移指南。
- 【文档】`RunnableWithMessageHistory` **仍在**（未弃用），但它只做"把历史读出/写回"，**不含摘要/裁剪**；
  且已知它**不兼容**旧 memory 类（`AttributeError: ... has no attribute 'aget_messages'`，见 langchain#21069）。
  对本项目**用不上**——我们已经自管 Redis 历史 + 手动传 `chat_history`，比 RunnableWithMessageHistory 更直接。

### 在现有结构下的最小侵入改法（不引 langgraph/langmem）

**核心思路**：不动 `create_tool_calling_agent + AgentExecutor.invoke` 主链，
只在 `_load_history` 之后、喂给 executor 之前，加一层"压缩历史"。历史仍存 Redis 明细，摘要只在**读取时**产生。

#### 方案 A（最小、纯裁剪，无摘要）— 先落地
在喂 executor 前用 `trim_messages` 按 token 预算裁，替代/补充现在的 20 条硬 `ltrim`。
```
from langchain_core.messages import trim_messages
history = trim_messages(history, max_tokens=3000, token_counter=get_cloud_llm(),
                        strategy="last", start_on="human", include_system=True)
```
- 代价：**0 额外 LLM 调用**（`get_num_tokens_from_messages` 是本地估算，OpenAI 兼容模型走 tiktoken 近似）；
  延迟 ~几 ms。只是"更聪明的截断"，仍**丢最旧内容**（不保留其信息）。
- 用途：立刻防超长 prompt 爆 token；但没解决"记住早期对话要点"。

#### 方案 B（滚动摘要，推荐）— 满足课题5诉求
自管一条 running summary，与明细历史分开存 Redis。读取时：`[SystemMessage(摘要)] + 最近 N 条明细`。
只在**明细超阈值**时，把"即将被挤出的最旧几条"喂一个便宜/低温 LLM 压进摘要。骨架：
```
# _load_history 后：
recent = trim_messages(history, max_tokens=KEEP_RECENT, token_counter=len, strategy="last",
                       start_on="human")            # 保留最近 N 条明细
overflow = history[: len(history) - len(recent)]    # 被挤出的最旧几条
summary = _load_summary(session_id)                 # Redis 里的旧摘要(str，可空)
if overflow:
    summary = _summarize(get_cloud_llm(temperature=0), summary, overflow)  # 1 次 LLM 调用
    _save_summary(session_id, summary)              # 增量更新，不重复压已压过的
messages = ([SystemMessage(f"对话历史摘要：{summary}")] if summary else []) + recent
executor.invoke({"input": ..., "chat_history": messages})
```
- **省钱要点**：摘要**增量**（喂"旧摘要 + 新溢出的几条"，不是每轮重压全历史）——这正是 LangMem
  `summarize_messages` 传 `running_summary` 的思想，我们手写等价物。
- 代价：仅在**溢出发生时**多 1 次短 LLM 调用（输入=旧摘要+2~4 条溢出消息，几百 token；用 temp=0）。
  多数轮次（历史没超阈值）**零额外调用**。延迟：溢出轮 +1~2s（可异步：先用旧摘要应答、后台补更新摘要）。
- 存储：新增 Redis key（如 `agent:chat_summary:{session_id}`，随明细同 TTL 续期）。
- 侵入面：只改 `core/agent.py` 的 `_load_history`/`_save_turn` 邻域 + 加 `_summarize` 辅助；
  `_build_executor`、prompt、飞书调用点**不动**（chat_history 仍是 `list[BaseMessage]`）。

#### 不推荐：迁 LangGraph + LangMem
- 【文档】官方"长期正解"是 `create_agent`(LangGraph) + checkpointer + `langmem.short_term.SummarizationNode`
  / `summarize_messages(..., running_summary=...)`（`max_tokens` / `max_tokens_before_summary` 控制阈值）。
- 但要**新增 langgraph + langmem 依赖**、把 `create_tool_calling_agent+AgentExecutor` 整链换成 LangGraph 图，
  与本项目"自管 Redis 历史 + 工具三层路由 + 非流式 invoke"的既有结构冲突大。**不符合"最小侵入"**，列为远期可选。

### 出处（课题5）
- 【文档】迁移指南入口（0.3 弃用 memory 一族、指向 trim/summary + LangGraph）：
  https://python.langchain.com/docs/versions/migrating_memory/
  （现 308 跳转至 https://docs.langchain.com/oss/python/langchain/overview ；分页 conversation_buffer_memory /
  conversation_summary_memory 同源）
- 【文档】`trim_messages` 概念与用法：https://python.langchain.com/docs/how_to/trim_messages/
- 【文档】`ConversationSummaryBufferMemory` 弃用标注（since 0.3.1 / removal 1.0.0）：
  https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary_buffer.ConversationSummaryBufferMemory.html
- 【文档】LangMem 摘要（running_summary/max_tokens_before_summary，远期方案参考）：
  https://langchain-ai.github.io/langmem/guides/summarization/
- 【文档】RunnableWithMessageHistory 与旧 memory 不兼容（aget_messages）：
  https://github.com/langchain-ai/langchain/issues/21069
- 【实测】本机 `langchain_core 0.3.0`：`from langchain_core.messages import trim_messages` 可导入；
  `token_counter=len` 按条数裁、不传 messages 返回 `RunnableLambda`（均真跑验证，见本任务命令记录）。

---

## 课题 6：飞书卡片流式更新（打字机 / 增量文本）

### TL;DR — **技术上可行**。飞书官方有"CardKit 流式更新"，就是为 AI 打字机场景造的。
把"🤔 分析中 → 整段文本"改成"边生成边刷一张卡"在**飞书侧完全支持**。硬限制见下；主要工程量在**我们这端**
（要从非流式 `.invoke()` 改成消费 LLM 增量输出 + 走 CardKit API 发/刷卡，不是现在的 `im reply` 文本）。

### 怎么用（官方 CardKit v1 流式）
- 【文档】开启：卡片 JSON **必须是 2.0 结构**，把 `streaming_mode: true` 写进 config（建卡时开），
  或对已有卡片实体调"更新卡片配置"接口开启。可用 `streaming_config` 调频率/步长/策略：
  - `print_frequency_ms`：两次上屏间隔，默认 **70ms**（可分端 default/android/ios/pc）。
  - `print_step`：每次上屏增量字符数，默认 **1**。
  - `print_strategy`：`fast`（默认，未上屏历史文本立即刷出再打字机续新内容）/ `delay`（历史也逐字）。
  - 版本：客户端 **7.20+** 才支持流式（低于 7.20 显示升级兜底提示）；**7.23+** 才支持自定义上述参数，
    7.20~7.22 只能用默认值。
- 【文档】调用流程（与现在的 `im reply` 完全不同的一条新路）：
  1. `POST cardkit-v1/card/create` 建**卡片实体**拿 `card_id`（传 2.0 卡 JSON）。
  2. `im-v1/message/create` 发消息（`msg_type=interactive`，content 引用 `card_id`）。
  3. `PUT cardkit-v1/card-element/content`（流式更新文本接口）**每次传全量文本**，
     平台自动算增量、打字机上屏。（对 `plain_text` 元素或 `markdown` 组件。）
  4. 收尾：调"更新卡片配置"把 `streaming_mode` 关掉，再做增删组件/加评价按钮等。
- 【文档】增量逻辑：**每次传全量文本**，若旧文本是新文本的**前缀**→末尾续打字机；
  若前缀不同→整段直接上屏、无打字机效果。所以务必"只增不改"地累加输出。
- 【文档】权限：`cardkit:card:write` + `im:message:send_as_bot` + `im:message`（需开机器人能力）。

### 硬限制（务必给 dev 记牢）
- 【文档】**卡片实体只能发送一次**（`仅支持发送一次`）；发送 app 必须是建卡 app。
- 【文档】**QPS**：单卡片实体用 卡片/组件级 OpenAPI 操作上限 **10 次/秒**；
  但**流式更新文本接口"不触发该 QPS 频率限制"**——即打字机高频刷可以，别的组件操作受 10/s 约束。
- 【文档】**流式模式在"最后一次开启"后 10 分钟自动关闭**；建议主动关。→ 单条回答生成必须 < 10min（我们已有
  `AGENT_MAX_EXECUTION_TIME=60`，无忧）。
- 【文档】**流式期间不能即时响应卡片回调交互**：用户点交互组件触发回调时卡片"无法立即更新"，
  必须**先关流式再处理回调**。→ 纯文本回答卡无按钮，无影响；若卡上要放按钮，得等流式结束。
- 【文档】流式开启期间聊天列表摘要默认显示 `[生成中...]`。
- 【文档/推断】内容大小：受**标准卡片内容体积上限**约束（Card JSON 2.0 通用限制，非流式特有），
  超长回答需自行分段/截断——文档未在流式页给出精确字节数，属 markdown/卡片通用限制。【推断】

### 与我们已知 200830 / 延时更新 token(30min、2 次) 体系的关系
- **是两套独立机制，别混淆**：
  - "延时更新 token（30min 有效、同 token 最多 2 次）"是**老式卡片回调**里的延时更新（回调先回 `{}`，
    再用回调 token 补更新）——见 `research/feishu-card-inplace-replace.md`。**次数硬限 2 次**，做不了打字机。
  - **CardKit 流式**走**持久卡片实体 `card_id` + cardkit-v1 更新接口**，**没有那个 2 次限制**，
    换成"10 分钟自动关 + 流式接口不占 10/s QPS"。这才是打字机该用的路。
- **200830（2.0 卡不能被 1.0 卡原地替换）**：CardKit 流式**全程 2.0**。若采用它，聊天回答卡就进 2.0 家族，
  与那份报告"确认后进行中卡也做成 2.0"的方向一致，不会踩 200830。
- 注意：本项目现有 进度/结果卡是 **1.0**（`cards.card()` 不写 schema）。流式聊天回答是**另起 2.0 新卡路**，
  与那些 1.0 卡各自独立，互不影响。

### 落地评估 / 我们这端的真正工作量（供 dev 决策）
飞书侧可行，瓶颈在**产出增量文本**：现在飞书路是 `AgentExecutor.invoke()` **非流式**、一次拿 `output`。
要打字机需要：
1. 改用 `.stream()`/`.astream()`（CLI 已有 `_run_with_stream` 消费 chunk 的先例，但那是打印到终端）
   或 `astream_events` 累积**最终答案 token**（工具调用中间步骤不该上屏，只累加 final output 文本）；
2. 每积累若干字符 → `PUT card-element/content` 传**当前累计全量文本**（前缀增长触发打字机）；
3. 结束关流式。
- 代价：多一条 CardKit 发/刷卡链 + LLM 改流式消费；每答一次多 N 次 `card-element/content` 调用
  （不占 10/s QPS，可控）。体验收益明显（消除"🤔"到整段的空窗）。
- **次优/兜底（若不想上流式）**：**分段追加**——把长回答按段落分成 2~3 次 `im reply`/普通卡更新，
  或保留"🤔"再一次性替换。改动最小但无逐字体验。流式仍是体验最佳解，且飞书原生支持。

### 出处（课题6）
- 【文档】《流式更新卡片 - 开发指南》（streaming_mode / streaming_config / print_frequency_ms 默认 70ms /
  print_step 默认 1 / print_strategy fast|delay / 全量文本增量逻辑 / 10次/秒 与"流式不占 QPS" / 10 分钟自动关 /
  卡片实体仅发一次 / 流式期间回调不能即时更新 / 需 2.0 / 客户端 7.20+、自定义参数 7.23+ / cardkit:card:write）：
  https://open.feishu.cn/document/cardkit-v1/streaming-updates-openapi-overview?lang=zh-CN
- 【文档】Card JSON 2.0 结构：https://open.feishu.cn/document/feishu-cards/card-json-v2-structure?lang=zh-CN
- 【文档】延时更新 token（30min/2 次）与 200830 —— 见同目录 `feishu-card-inplace-replace.md` 及
  https://open.feishu.cn/document/feishu-cards/card-callback-communication?lang=zh-CN
- 【推断】内容体积上限走卡片通用限制，流式页未列精确字节数；上线前建议真机试一条超长回答验证截断行为。

---

## 给 dev 的两句话建议
- **课题5**：先上**方案 A（trim_messages 按 token 裁，0 额外调用）**止血超长 prompt；
  要"记住早期要点"再上**方案 B（自管 running summary + 增量压缩，仅溢出轮 1 次短 LLM）**。
  别为此引 langgraph/langmem（与现结构冲突、非最小侵入）。`trim_messages` 已在装好的 0.3.0 里，可直接用。
- **课题6**：飞书侧**可行**，用 CardKit 流式（2.0 卡 + `streaming_mode` + `card-element/content` 传全量文本）；
  硬限制=需 2.0、客户端 7.20+、10 分钟自动关、流式期间回调不能即时更新（纯文本答卡无按钮，无碍）。
  真正工作量在把飞书路从 `.invoke()` 改成流式消费 final token。不想上流式则退而求其次分段追加。
