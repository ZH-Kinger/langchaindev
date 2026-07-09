# 对话优化 5·6 — 现状盘点 + 选型框架（#39）

维护者：planner。事实源：`core/agent.py` / `core/feishu_bot/messages.py` / `core/prompts_agent.py` / `notes.md`「#38」。
本文只规划，不落任务板；下列 task 待 dev 过目 + researcher 结论回来后再落板派 owner。
更新：2026-07-09。前置：#38 快赢批 1–4 已改（verbose→False / iter·time 上限 / 历史 TTL / 指标图按需）。

凡涉及 **LangChain 0.3 记忆 API** 与 **飞书卡片/消息流式更新 API** 的细节，本文标「待 researcher 确认」占位，不自拍 API。

---

## 一、现状盘点

### 项 5 相关：记忆读写（`core/agent.py`）
- Redis list `agent:chat_history:{session_id}`，`MAX_HISTORY=20`（10 轮）**硬截断**：`_save_turn` 每轮 `rpush(human, ai)` → `ltrim(key, -20, -1)` → `expire(7d)`。超 10 轮的旧内容**直接丢弃**，无摘要、无压缩。
- `_load_history` 把 list 反序列化成 `HumanMessage/AIMessage` 列表。
- 喂给 Agent：`core/prompts_agent.get_agent_prompt` 用 `MessagesPlaceholder("chat_history")`，`_build_executor(tools).invoke({"input":..., "chat_history": history})`。
- `session_id = open_id or chat_id`（按用户隔离）。
- **两处消费**：CLI `run`/`_run_with_stream` + bot `_process_message`（L668-700）共用同一组 `_load_history/_save_turn`。改这两函数即全覆盖。
- 改项 5 要动的点：`_save_turn`（加摘要触发）、`_load_history`（拼摘要+近 N 条）、新增 Redis key（如 `agent:chat_summary:{session_id}`）、可能新增 summarize 辅助函数 + 选模型（`llm_factory.get_cloud_llm` / `get_edge_llm`）、prompt 如何注入摘要（系统消息 or chat_history 头部）。**不动** executor 结构、路由、bot 调用侧。

### 项 6 相关：回复流程（`core/feishu_bot/messages.py::_process_message`）
- 消息在**后台线程**处理（webhook 已 async 起线程），**不受飞书 3s 同步死线约束** —— 这点对 6 很关键：渐进更新的瓶颈是飞书**更新 API + 限流**，不是 3s。
- 现流程（L668-718）：`_load_history` → `messaging._feishu_reply(message_id, "🤔 正在分析…")`（一条独立文本消息）→ `_build_executor(tools).invoke(...)` **阻塞式**跑完 → 清洗 markdown → `_save_turn` → 末尾 `_feishu_reply(reply)` 或附趋势图卡。
- 已有流式能力但**只 CLI 用**：`agent._get_streaming_executor` + `_run_with_stream` 走 `executor.stream()`，逐 chunk 打 `actions`/`steps`/`output` 到 stdout。bot 侧没接。
- 发送原语（`messaging.py`）：`_feishu_reply`（回文本）/`_feishu_reply_card`（回卡片）/`_feishu_reply_with_chart`。**均为「新发一条」，无「更新/patch 已发消息」原语** —— 渐进反馈若要「原地刷新同一条」必须新增更新原语（依赖飞书 update/patch API，待 researcher）。
- 相关踩坑：卡片原地替换 schema 必须一致（200830，见 notes #30/#33）；连点/重投用 event_dedup + NX 锁挡。
- 改项 6 要动的点：`_process_message` 回复段（用 `stream()` 替 `invoke()`）、新增「更新已发消息/卡片」发送原语、进度呈现载体（文本 append vs 卡片 patch）。executor：bot 现用非流式 `_build_executor`，要么切 streaming executor、要么用 `.stream()` 只取 step 事件。

---

## 二、选型框架

### 项 5 · 滚动摘要记忆

| 方案 | 说明 | 工作量 | 风险 | 价值 |
|---|---|---|---|---|
| **5-A 自管滚动摘要（推荐）** | 自己在 `agent.py` 实现：新增 `agent:chat_summary:{session_id}`；`_save_turn` 后若近期 list 超阈值，调 LLM 把最旧 K 轮折进摘要、list 只留近 N 轮逐字；`_load_history` 返回 `[SystemMessage(摘要)] + 近 N 条`。不依赖任何可能被弃用的 memory 类。 | M | 低-中（自有代码可单测；成本=偶发 LLM 调用） | 高 |
| **5-B LangChain 内置摘要记忆** | 用框架的 summary/trim 记忆件（`ConversationSummaryBufferMemory` 或 0.3 的 `trim_messages`+摘要）。**0.3 是否仍支持/已弃用、推荐替代（LangGraph? `RunnableWithMessageHistory`?）待 researcher 确认**。 | S-M | 中（API 可能已 deprecated，绑定框架演进） | 中 |
| **5-C 纯 token 截断（无摘要，过渡）** | 不摘要，只把「最近 20 条」改为「按 token 预算 `trim_messages` 截断」，仍丢旧上下文但省 LLM 调用。可作 5-A 前的廉价过渡或降级路径。 | S | 低 | 低-中 |
| **5-D LangGraph 持久化 + 摘要节点** | 迁到 LangGraph checkpointer + summarization node。 | L | 高（架构级改造，当前简单 executor 不需要） | 中（长期，非现在） |

- **验收标准（5-A）**：① 超阈值后旧轮进摘要、`_load_history` 返回 = 摘要 + 近 N 条逐字；② 连续 >10 轮对话，Bot 仍能引用早期约定（如早前给定的桶名/路径），对照现状「丢失」有可见改善；③ 摘要 LLM 调用失败时降级为纯截断（不阻断回复）；④ 摘要 key 随 `_clear_history` 一并清、有 TTL；⑤ CLI + bot 两路径都生效；⑥ 单测覆盖：触发阈值/不触发/降级/清除。
- **依赖**：researcher 结论（5-B 才需，5-A 可先行不等）；开放问题 Q5-1/Q5-2（模型 + 成本）需 dev 问用户。

### 项 6 · 流式 / 渐进反馈

| 方案 | 说明 | 工作量 | 风险 | 价值 |
|---|---|---|---|---|
| **6-A 工具步骤渐进提示（推荐起步）** | 用 `executor.stream()` 取 `actions`/`steps` 事件，把「🤔 正在分析」升级为可读进度（如「正在查询 OSS 用量…」→「正在生成回答…」），最后出完整答案。呈现载体二选一：文本追发 or 卡片 patch（后者需更新原语，待 researcher）。非 token 级，但用户明显「看到在动」。 | M | 中（飞书更新 API + 限流 + 200830 schema 一致性） | 中-高 |
| **6-B 真·token 流式刷卡** | 把 LLM token 增量流式 patch 进同一张卡片，逐字增长。 | L | 高（飞书 patch 限流、半截 markdown 渲染、schema 一致性、并发刷卡风暴——本项目已多次踩「刷卡风暴」坑） | 高但重 |
| **6-C 不流式，改善感知 + 缩短响应（廉价替代）** | 不引入流式：把占位提示按已知 `intents` 具化（路由结果现成，见 `select_tools_scoped` 返回 intents），如命中 monitor 就回「正在查询监控指标…」；同时靠 #38 的 iter/time 上限 + 工具按需（指标图按需）压低真实时延。 | S | 低 | 低-中 |

- **前置事实（利好）**：bot 回复在后台线程，**不卡 3s**，6-A/6-B 时延上可行；真正门槛是飞书**消息/卡片更新 API 能力 + 限流 + schema 约束**。
- **验收标准（6-A）**：① 一次多工具调用中，用户依次看到 ≥1 条中间进度（工具阶段），最后收到完整答案；② 更新走「同一条原地刷新」或「有限条追发」，不产生刷卡风暴（复用 NX/去重经验）；③ 失败/超时降级为现状一次性回复，不比现在差；④ 单测：`stream()` 事件序列 → 进度更新调用次数/载荷断言（mock 飞书发送）。
- **依赖**：**待 researcher 确认**：飞书是否有「更新已发消息文本 / patch 卡片内容」API、是否支持增量 append、限流阈值、是否有官方「流式卡片」能力、与 200830 schema 一致性关系。6-A/6-B 的载体选型全押在此结论上；6-C 不依赖 researcher，可随时做。

---

## 三、优先级建议

| 序 | 项 | 建议 | 理由 | 可并行? |
|---|---|---|---|---|
| 1 | **5-A** 自管滚动摘要 | **先做** | 真实痛点（>10 轮丢上下文），自有代码不依赖外部 API，可单测；不必等 researcher 即可起步（选模型问题除外） | 与 6-C 并行 |
| 2 | **6-C** 具化占位 + 缩短感知 | **快赢，先上** | S 成本，intents 现成，零外部依赖，立即改善体验 | 与 5-A 并行 |
| 3 | **6-A** 工具步骤渐进 | researcher 结论回来后评估 | 中等价值，门槛在飞书更新 API 可行性 | 依赖 researcher |
| 观望 | 5-D / 6-B | 暂不做 | 架构级/高风险，投入产出比在当前阶段不划算 | — |

**核心判断**：项 5 用 5-A（务实、可测、独立）；项 6 先用 6-C 拿快赢，6-A/6-B 是否值得投入**取决于飞书更新 API 结论 + 用户对「流式体验」的价值判断**（见开放问题）。

---

## 四、开放问题（交 dev 转问用户 / 团队定夺）

- **Q5-1（模型）**：摘要用哪个模型？`get_cloud_llm`（Qwen-Max，质量高但每次调用有成本）vs `get_edge_llm`（Qwen3-4B，便宜快，摘要质量够用？）。建议默认 edge 摘要 + 可配置。
- **Q5-2（成本/触发）**：摘要成本容忍度？触发策略——每超阈值一轮就折叠（勤但费）vs 每积 K 轮批量折叠一次（省但摘要略滞后）？阈值设多少（近 N 轮逐字 + 更早进摘要）？
- **Q6-1（值不值得）**：**流式反馈是否值得投入，还是优先「直接缩短响应时间」？** 若用户主要嫌「等太久」，6-C + #38 时延优化可能已够；6-A/6-B 是「让等待可见」而非「更快」。请用户表态偏好。
- **Q6-2（载体）**：渐进反馈接受「追发多条消息」还是要求「原地刷新同一条」？后者依赖飞书更新 API 且受 200830/限流约束，成本更高。
- **Q-研究**：以下两条已建议交 researcher 并行查证（dev 派单）：
  - R5：LangChain 0.3 记忆 API 现状——`ConversationSummaryBufferMemory` 是否可用/已弃用、0.3 推荐的摘要/裁剪记忆方案（`trim_messages`/`RunnableWithMessageHistory`/LangGraph checkpointer），带出处。→ 决定 5-A 是否借用框架件 or 5-B 是否可行。
  - R6：飞书「更新已发消息/卡片」能力——更新消息文本 API、卡片增量 patch、是否有官方流式卡片、限流阈值、与 200830 schema 一致性约束，带出处。→ 决定 6-A/6-B 载体与可行性。

---

## 五、拆分 task（建议粒度，待 dev + researcher 后落板）

> 前缀 `CONV-`，避开现有 `P-*`（roadmap）。

- **CONV-5.0（研究，researcher）**：R5 LangChain 0.3 记忆 API 现状取证。**验收**：`docs/collab/research/langchain-0.3-memory.md`，明确「5-A 是否需借用框架件 / 5-B 是否可行 / 推荐方案」带出处。**blockedBy**：无。
- **CONV-5.1（dev）**：实现 5-A 自管滚动摘要（`_save_turn` 折叠 + `_load_history` 拼摘要 + `chat_summary` key + summarize 辅助 + 摘要注入 prompt + 失败降级）。**验收**：见 5-A 验收①-⑤。**blockedBy**：CONV-5.0（确认是否借框架件）+ Q5-1/Q5-2（模型/触发拍板）。
- **CONV-5.2（tester）**：覆盖 5-A——触发阈值/未触发/摘要失败降级/`_clear_history` 连清摘要/CLI+bot 双路径。**blockedBy**：CONV-5.1。
- **CONV-6.0（研究，researcher）**：R6 飞书更新/流式卡片能力取证。**验收**：`docs/collab/research/feishu-streaming-card.md`，明确「更新原语可行性 + 限流 + schema 约束 + 6-A/6-B 载体建议」带出处。**blockedBy**：无。
- **CONV-6.1（dev）**：实现 6-C——占位提示按 `intents` 具化（`_process_message` 那条「🤔 正在分析」）。**验收**：命中已知意图时占位文案对应该意图；未知意图回通用文案；不新增外部依赖。**blockedBy**：无（可立即做）。
- **CONV-6.2（dev）**：实现 6-A 工具步骤渐进（`.stream()` 取 step 事件 + 更新/追发原语）。**验收**：见 6-A 验收①-④。**blockedBy**：CONV-6.0 + Q6-1/Q6-2 拍板。
- **CONV-6.3（tester）**：覆盖 6-A——stream 事件序列→更新次数/载荷断言、降级路径。**blockedBy**：CONV-6.2。

**可立即开工（不等任何人）**：CONV-5.0、CONV-6.0（研究），CONV-6.1（6-C 快赢）。
**须串行**：CONV-5.1←CONV-5.0+Q拍板；CONV-6.2←CONV-6.0+Q拍板；各自 tester 串在 dev 后。
