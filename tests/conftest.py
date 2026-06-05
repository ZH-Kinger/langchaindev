"""
pytest 全局 fixtures。

提供：
  - fake_redis        : 自动替换 utils.redis_client.get_redis 为 fakeredis 实例
  - fernet_key        : 自动注入测试用的 Fernet key 到 settings
  - mock_aliyun_sts   : 拦截 STS AssumeRole（不调真实阿里云）
  - mock_ram_api      : 拦截 RAM API（list_ram_users_api 等）
  - mock_feishu_send  : 拦截 _feishu_reply / _send_card（不发真消息）

放在 tests/conftest.py，pytest 会自动发现。
"""
import sys
from pathlib import Path

# 确保测试能 import 项目代码
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


# ── Redis：用 fakeredis 替换 ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """所有测试自动替换 get_redis() 为 fakeredis（解码字符串模式，匹配生产）。"""
    import fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)

    # 替换 utils.redis_client 内部单例和 get_redis 函数
    from utils import redis_client
    monkeypatch.setattr(redis_client, "_client", fake)
    monkeypatch.setattr(redis_client, "get_redis", lambda: fake)
    monkeypatch.setattr(redis_client, "is_redis_available", lambda: True)

    yield fake


# ── Fernet 加密 key：自动注入 ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fernet_key(monkeypatch):
    """注入测试用 Fernet key 到 settings，并清掉 utils.crypto 的缓存。"""
    from cryptography.fernet import Fernet
    test_key = Fernet.generate_key().decode()

    from config.settings import settings
    monkeypatch.setattr(settings, "BOT_CREDS_ENCRYPTION_KEY", test_key)

    # 清掉 lru_cache，让下一次调用读到新 key
    from utils.crypto import _fernet
    _fernet.cache_clear()

    yield test_key

    # 测试结束后再清一次
    _fernet.cache_clear()


# ── 阿里云 STS：拦截真实 API 调用 ────────────────────────────────────────────

@pytest.fixture
def mock_aliyun_sts(monkeypatch):
    """让 _do_assume_role 返回可控的临时凭证。"""
    import time
    fake_cred = {
        "access_key_id":     "STS.MOCK_AK",
        "access_key_secret": "MOCK_SK",
        "security_token":    "MOCK_TOKEN",
        "expire_ts":         time.time() + 3600,
        "role_arn":          "acs:ram::1234567890:role/BotRole-Default",
    }

    from utils import aliyun_sts
    monkeypatch.setattr(aliyun_sts, "_do_assume_role", lambda *a, **k: dict(fake_cred))
    yield fake_cred


# ── RAM API：默认空列表，测试按需覆盖 ────────────────────────────────────────

@pytest.fixture
def mock_ram_api(monkeypatch):
    """提供可注入的 RAM 用户列表和用户组查询。"""
    state = {
        "users":  [],   # [{"user_id":"...", "user_name":"...", "display_name":"..."}]
        "groups": {},   # {ram_user_name: [group_name, ...]}
    }

    from tools.aliyun import ram as ram_module
    monkeypatch.setattr(ram_module, "list_ram_users_api", lambda: state["users"])

    from utils import aliyun_sts
    monkeypatch.setattr(
        aliyun_sts, "_list_user_groups",
        lambda user_name: state["groups"].get(user_name, []),
    )

    yield state


# ── 飞书消息：默认拦截不发 ───────────────────────────────────────────────────

@pytest.fixture
def mock_feishu_send(monkeypatch):
    """捕获所有 Bot 回复，测试可断言内容。"""
    sent = []

    from core import feishu_bot
    monkeypatch.setattr(feishu_bot, "_feishu_reply",
                        lambda msg_id, text: sent.append({"type": "reply", "text": text}))
    monkeypatch.setattr(feishu_bot, "_feishu_send",
                        lambda chat_id, text: sent.append({"type": "send", "text": text}))

    yield sent
