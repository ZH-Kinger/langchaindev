# 飞书卡片原地替换：schema 2.0 ↔ 1.0 能否互换（错误码 200830）

任务 #30。求证：`card.action` 回调返回 `{"card":{"type":"raw","data":...}}` 原地替换时，
能否用**老式 schema 1.0 卡**替换当前的 **schema 2.0 表单卡**（历史见过错误码 200830）。

## 结论（TL;DR）

**不能。原地替换必须同 schema 家族（2.0→2.0 或 1.0→1.0），版本要与最初发出的那张卡一致。**
用 1.0 卡去原地替换一张 2.0 卡，会被飞书拒，返回错误码 **200830**。

- 【文档】错误码 **200830** 的官方释义就是：**「JSON 2.0 结构的卡片无法更新为 JSON 1.0 结构卡片」**。
  官方给的解决办法：交互前若是 2.0 卡，交互后仍须保持 2.0 结构。
  来源：飞书开放平台《卡片回传交互回调》错误码表
  https://open.feishu.cn/document/feishu-cards/card-callback-communication?lang=zh-CN
- 【文档】立即更新（在回调响应体里返回卡片 raw data）时，`card.data` 里的卡片
  **必须与最初发送时的卡片版本一致**：1.0 发的就传 1.0 数据，2.0 发的就传 2.0 数据。
  同上来源。
- 【推测】反方向（1.0→2.0）文档未单列错误码，但既然规则是「版本须与原卡一致」，
  跨版本替换任一方向都应视为不被支持。���妥做法：**永远同家族替换**。

## 回调返回体格式（立即更新）

```json
{
  "toast": {"type": "success", "content": "..."},
  "card":  {"type": "raw", "data": { /* 与原卡同 schema 的完整卡片 JSON */ }}
}
```
- `card.type = "raw"`（用 JSON 直接回卡时必填）；`card.data` = 卡片 JSON（必填）。
- 【文档】`data` 卡片的 schema 必须匹配最初发出的卡片版本。schema 2.0 的 `data` 含
  `schema`/`config`/`body`/`header`。
- 【文档】回调须在 **3s** 内响应；无法及时更新可先回 `"{}"`，再用回调里的 `token`
  走延时更新接口（token 30 分钟有效、同一 token 最多更新 2 次）。
- 【文档】业务服务端**不可**用 3xx 重定向状态码。
  来源同上 +《配置卡片交互》
  https://open.feishu.cn/document/common-capabilities/message-card/add-card-interaction/interaction-module?lang=zh-CN

## 官方卡片回调相关错误码摘录（【文档】同一来源）

| 码 | 含义 |
|----|------|
| 200671 | 回调返回了非 HTTP 200 状态码 |
| 200672 | 回调返回的响应体格式错误 |
| 200673 | 回调返回了不正确的卡片 |
| **200830** | **JSON 2.0 卡片无法更新为 JSON 1.0 结构卡片** |
| 200530 | 表单容器内交互组件 `name` 为空 |
| 200341/200342/200343 | 回调地址超时 / 无法建 TCP / DNS 解析失败 |

（错误码仅飞书客户端 7.28+ 才回显；低版本不回码。）

## 本项目现状核对（只读源码所得）

四条数据流动/迁移链路的卡片 schema **完全一致的模式**：**确认卡 2.0 表单 + 进度/结果卡 1.0**。
1.0 是因为进度/结果卡都走 `tools/feishu/cards.py::card()` 原语，该函数**不写 `schema` 键**（即默认 1.0）。

| 模块 | confirm_card | progress/result_card | 确认后原地替换动作 |
|------|-------------|----------------------|------|
| `core/transfer/cards.py` | 2.0（`schema:"2.0"` + form） | 1.0（`card()`） | `actions.py::_h_confirm_transfer` L565-566 返回 `{"card":{"type":"raw","data":progress_card(...)}}` |
| `core/cpfs_dataflow/cards.py` | 2.0 | 1.0 | `_h_confirm_cpfs_dataflow`（同模式，CLAUDE.md 称线上在用） |
| `core/vepfs_dataflow/cards.py` | 2.0 | 1.0 | `_h_confirm_vepfs_dataflow` |
| `core/bucket_transfer/cards.py` | 2.0 | 1.0 | `_h_confirm_bucket_transfer` |

**即：这四个确认 handler 在确认后都是「2.0 确认卡 → 1.0 进度卡」的原地替换，正好命中 200830 的拒绝场景。**

### 代码内部已有的相互矛盾（重要）
`core/feishu_bot/actions.py::_h_query_progress_by_id`（L928-933）注释已明确写着：
> 「输入卡是 schema 2.0，而进度/结果卡是老式 1.0；用 1.0 卡原地替换 2.0 卡会被飞书拒（错误 200830）。
>  故这里改为**推送一张新卡** + 回 toast，不做原地替换。」

也就是说，项目在「查询进度」这条路上**已经因 200830 放弃原地替换、改推新卡**；
但「确认下发」这条路上（`_h_confirm_transfer` 等四个）**仍在做 2.0→1.0 原地替换**——同一个坑，处理不一致。

### 为什么线上「看起来能用」（【推测】，未真机点验）
即便原地替换被 200830 拒，**业务并不中断**：确认 handler 仍返回了 `toast`（成功提示照弹）、
后台迁移线程照常起、终态由 `_send_card` **另推一张结果卡**。用户可见的唯一异常是：
确认卡那条消息**没被替换成进度卡**（停在原确认卡或飘一下报错），随后单独收到结果卡。
所以「任务完成」维度看着正常，但原地替换其实是静默失败——这正是 auditor 标注的「未真机验证」脆点。

## 正解（供 dev 决策，researcher 不改源码）

与 auditor 在 `notes.md`（2026-07-08）的建议一致，二选一：
1. **把确认后的「进行中」替换卡也做成 schema 2.0**（去掉 submit 按钮的一张纯展示 2.0 卡），
   同家族替换从根上规避 200830。最干净、可单测。
2. 或**不原地替换**、改「回 toast + 另推一张 1.0 新卡」（就像 `_h_query_progress_by_id` 已做的那样）。

方案 1 更贴合「确认后按钮即消失、卡片原地变进行中」的既有交互；方案 2 改动最小、与查询路一致。

## 出处
- 【文档】《卡片回传交互回调 - 服务端 API》（含 200830 释义、raw 响应体格式、版本须一致规则、错误码表）
  https://open.feishu.cn/document/feishu-cards/card-callback-communication?lang=zh-CN
- 【文档】《配置卡片交互》（立即/延时更新、共享/独享卡片 update_multi）
  https://open.feishu.cn/document/common-capabilities/message-card/add-card-interaction/interaction-module?lang=zh-CN
- 【文档】《Card JSON 2.0 结构》
  https://open.feishu.cn/document/feishu-cards/card-json-v2-structure?lang=zh-CN
- 【文档】《通用错误码》
  https://open.feishu.cn/document/server-docs/api-call-guide/generic-error-code?lang=zh-CN
