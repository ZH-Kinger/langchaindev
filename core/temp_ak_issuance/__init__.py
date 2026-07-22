"""审批制临时 AK/SK 发放（外部 / 卖数据）。

飞书审批通过 → 按有效期窗口分流生成 OSS 访问凭证：
  · ≤12h → 纯 STS 单发（含 SecurityToken，到点自灭）；
  · 跨天/跨周 → 方案 B（RAM 子用户 + 长期 AK + policy 时间区间条件 + 到期硬删）。
加法式独立包：复用 ram_approval 门禁 / permsync policy 生成 / SMTP 下发，零改现有引擎。
凭证 secret/token 绝不落 Redis/日志——只在下发邮件里出现一次。
"""
from . import approval, cards, cleanup, delivery, issuer, orchestrator, policy

__all__ = ["approval", "cards", "cleanup", "delivery", "issuer", "orchestrator", "policy"]
