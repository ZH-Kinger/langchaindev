# 流式回复可行性（项6）— 飞书流式卡片 + LangChain token 流

调研人：researcher｜日期：2026-07-09｜面向：dev 实现决策

## 结论速览

**有限可行、推荐做，但不是"改一行"**。飞书 2025 年起有官方、专为 AI 机器人设计的
**CardKit 流式更新卡片**（打字机效果），这是正解——不要用 PATCH 编辑消息那条老路。
代价：必须切到**卡片 JSON 2.0** + 一套新的 CardKit 接口（本项目现有卡片全是 1.0，
`card()` 原语和 `_feishu_reply_card` 不能直接复用），且 Agent 带工具时"只流式最终答案"要额外接线��

---

## 1. 飞书是否支持流式更新一条已发消息？

### 1a. 正解：CardKit 流式更新卡片（官方能力，2025+）【文档】
出处：https://open.feishu.cn/document/cardkit-v1/streaming-updates-openapi-overview

- 官方明确面向 **AI 机器人场景**：机器人发一张流式卡片，服务端持续推**全量文本**，
  飞书客户端自动算增量、按打字机逐字上屏。**你不用逐 token 发请求**，客户端负责插帧。
- 两个能力：① 文本组件（`plain_text`/`markdown`）打字机输出；② 组件级增删改（更新图/按钮等）。
- **前提**：
  - 卡片必须是 **JSON 2.0**（`"schema":"2.0"`），飞书客户端 **7.20+**（旧端只显示标题+升级提示）。
  - 应用需权限 `cardkit:card:write` + `im:message:send_as_bot`/`im:message`。
- **调用链**（全部 `/open-apis/cardkit/v1/...`，可直接用 requests，无需新 SDK）：
  1. `POST /open-apis/cardkit/v1/cards` — 建卡片实体，body `{type:"card_json", data: json.dumps(2.0卡)}`，返回 `card_id`。
  2. `POST /open-apis/im/v1/messages`（或 `/{message_id}/reply`）发消息，`msg_type=interactive`，
     content 引用 `card_id`（形如 `{"type":"card","data":{"card_id":...}}`）。【推测·待实现时核 reply 是否也支持 card_id】
  3. `PUT /open-apis/cardkit/v1/cards/{card_id}/settings` — 开 `streaming_mode=true`（可选 `streaming_config`）。
  4. `PUT /open-apis/cardkit/v1/cards/{card_id}/elements/{element_id}/content` — 反复推全量文本，
     每次带**严格递增的 `sequence`**（+ `uuid`）。
  5. 结束时 `settings` 关 `streaming_mode=false`。
- **关键限制**【文档】：
  - 流式模式下的更新**不触发常规 QPS 限流**（这正是它相对 PATCH 的优势）；
    非流式的组件级操作上限 **10 次/秒/卡**。
  - `streaming_mode` 开启后 **10 分钟自动关闭**；卡片实体有效期 **14 天**。
  - **流式期间用户点卡片、回调无法立即更新卡**——必须先关流式模式。
    → 若答案卡带按钮，流式没结束按钮点了没反应。
  - 聊天栏摘要默认显示 `[生成中...]`。
  - `print_strategy`：`fast`（默认，未上屏文本立即全显再续打字）/ `delay`；
    `print_frequency_ms`（默认 70ms）、`print_step`（默认 1 字）。
    客户端 7.20–7.22 只能用默认、不可自定义；7.23+ 才能调。

### 1b. 老路（不推荐）：PATCH 编辑已发消息卡片【文档】
出处：https://open.feishu.cn/document/server-docs/im-v1/message-card/patch

- **只能更新 `interactive` 卡片，不能 PATCH 纯文本消息**。
- 单条消息更新频控 **5 QPS**；卡需 `config.update_multi:true`；卡 ≤30KB；仅 14 天内消息。
- **致命点**：社区实测单条消息有**隐性编辑次数上限**（官方未公布数值），超了后编辑被静默忽略
  → 不适合长答案高频刷。CardKit 卡片实体**无编辑次数限制**，就是为解决这个而生。
  出处（社区踩坑）：https://blog.csdn.net/xiaoting451292510/article/details/158890752

### 1c. 纯文本消息能流式吗？
**不能**。PATCH 和 CardKit 都只作用于 interactive 卡片。要流式必须走卡片。

---

## 2. LangChain(GLM) token 流 → 节流批量推飞书

- GLM 走 `get_cloud_llm`（ChatOpenAI 兼容），`streaming=True` 后 `.stream()/.astream()` 出 token。
- **典型对接模式**：不要每 token 发一次请求。**累积全量文本，按节流推**——
  每 ~300–500ms 或每 ~20 字 `PUT .../content` 推一次当前**全量**文本，客户端做逐字插帧。
  几次/秒的调用量足够顺滑，也远低于限流。
- **Agent 带工具的坑**：本项目 bot 路径走 `AgentExecutor`（`_build_executor`，非流式）。
  工具执行阶段**没有 token**，流式只对"最终答案 LLM 生成"有意义。要拿到最终答案的 token 流：
  - 简单：纯 LLM 回复（无工具）直接 `get_cloud_llm(streaming=True).stream()`。
  - Agent：需自定义 callback 只截**最终答案那次 LLM 调用**的 `on_llm_new_token`，
    或用 `astream_events`（v2）过滤 final。工具跑动期间卡片停在"[生成中...]"，
    建议先推一行"正在调用工具…"状态占位，避免长时间静止。

---

## 3. 成本 / 风险 / 复用

- **限流**：CardKit 流式不吃常规 QPS，节流后基本无压力；PATCH 路 5 QPS + 隐性编辑上限，风险大——弃用。
- **卡片 schema**：本项目 `tools/feishu/cards.py::card()` 产出**全 1.0**（config/header/elements，无 `schema:2.0`）。
  流式必须 2.0。**`_feishu_reply_card` 与 `card()` 原语无法直接复用**，要新写一个 2.0 流式答案卡 builder
  （一个带固定 `element_id` 的 markdown 组件）。项目历史已因 1.0/2.0 混用踩过 200830（见
  `research/feishu-card-inplace-replace.md`），2.0 这套要独立走。
- **token 无缓存放大**：`notify._get_access_token` 目前每次取 token（notes 里的 MED-3/MED-C7 未修）。
  流式会把取 token 频次放大 → **上流式前先修 token 缓存**，否则打爆取 token 端点。
- **可复用的**：`requests` 直连方式、`_get_access_token`、后台线程模型（`routes.py:111`
  `_process_message` 已在独立线程跑，无 3s 死线问题）。

---

## 4. 推荐实现方案（最小改动路径，供 dev）

1. **先修前置**：`_get_access_token` 加模块级缓存（MED-3），否则流式放大取 token 频次。
2. **加 CardKit 原语**（`messaging.py` 新增，不动老函数）：
   `create_stream_card(card_json2)→card_id`、`send_card_entity(message_id/chat, card_id)`、
   `open_stream(card_id, seq)`、`push_text(card_id, element_id, full_text, seq)`、`close_stream(card_id, seq)`；
   `sequence` 用每卡一个自增计数器 + 锁保证严格递增。
3. **加 2.0 流式答案卡 builder**（新模块或 cards.py 里独立函数，别混进 1.0 `card()`）：
   一个 markdown 组件，固定 `element_id`（如 `"answer"`）。
4. **改 `_process_message` 的 Agent 回复分支**（messages.py:674 起）：
   建卡+发卡+开流式 → 用 `get_cloud_llm(streaming=True)` 跑，节流回调推全量文本 → 完成后 `close_stream`。
   先只覆盖**纯 Agent 文本回复**；带指标图/工具的分支可暂留非流式。
5. **降级**：建卡/开流式失败 → 回退现有 `_feishu_reply` 一次性发，保证不 regress。

## 明确踩坑点（清单）
- 卡必须 2.0，1.0 原语不能复用；1.0/2.0 别混（200830 前科）。
- `sequence` 必须严格递增，并发推要加锁/计数器，否则更新被丢。
- 流式 **10 分钟自动关**、**期间按钮回调不生效**——完成务必显式 `close_stream`，答案卡的按钮要等关流式才能用。
- 客户端 <7.20 只见标题 → 需版本降级兜底。
- 权限 `cardkit:card:write` 要在 app（cli_a962...）后台开通，否则全链 403。
- Agent 工具执行期无 token，卡片会静止在"[生成中...]"→ 建议推一行工具状态占位。
- 取 token 缓存先修，别让流式放大限流。
- reply 端点是否支持发 card_id（vs 只能用 im/v1/messages create）——实现时核一下。

## 出处
- 流式更新总览：https://open.feishu.cn/document/cardkit-v1/streaming-updates-openapi-overview
- 建卡片实体：https://open.feishu.cn/document/cardkit-v1/card/create
- 更新卡片实体配置（streaming_mode）：https://open.feishu.cn/document/cardkit-v1/card/settings
- PATCH 更新已发卡片（老路，限制）：https://open.feishu.cn/document/server-docs/im-v1/message-card/patch
- 社区落地+踩坑（编辑次数上限/序列号/节流）：https://blog.csdn.net/xiaoting451292510/article/details/158890752
