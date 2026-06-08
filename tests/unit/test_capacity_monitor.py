"""core/capacity_monitor.py 测试。

mock 两个 compute_dir_sizes + dsw_scheduler._send_card，断言：
  - 每个 target 推送一张飞书卡片
  - 快照写入 Redis（fake_redis），第二次巡检算出"较上次"增量
  - 合计超阈值时 header 标红
"""
import json
import pytest


def _entry(name, total, count, batches, struct=""):
    return {"厂家": name, "total_bytes": total, "total_count": count,
            "struct": struct, "batches": batches}


@pytest.fixture
def patch_compute(monkeypatch):
    """注入可控的 compute_nested_sizes 返回值（按 vendor），含厂家+批次两级（4 元组）。"""
    state = {
        "oss": [_entry("aether", 100, 2, [("batch-a", 60, 1, ""), ("batch-b", 40, 1, "")])],
        "tos": [_entry("egodex", 300, 5, [("b1", 300, 5, "")])],
    }

    from tools.aliyun import oss as oss_mod
    from tools.volcano import tos as tos_mod
    monkeypatch.setattr(oss_mod, "compute_nested_sizes",
                        lambda **k: (state["oss"], "https://oss-cn-hangzhou.aliyuncs.com"))
    monkeypatch.setattr(tos_mod, "compute_nested_sizes",
                        lambda *a, **k: state["tos"])
    return state


@pytest.fixture
def capture_cards(monkeypatch):
    cards = []
    from core import dsw_scheduler
    monkeypatch.setattr(dsw_scheduler, "_send_card",
                        lambda open_id, chat_id, card: cards.append((chat_id, card)))
    return cards


@pytest.fixture
def two_targets(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CAPACITY_MONITOR_TARGETS_RAW", json.dumps([
        {"vendor": "oss", "bucket": "wuji-bucket-hangzhou", "prefix": "third-party-data/"},
        {"vendor": "tos", "bucket": "wuji-egocentric-data", "prefix": "third-party-data/"},
    ]))
    monkeypatch.setattr(settings, "FEISHU_CHAT_ID", "oc_test_chat")
    monkeypatch.setattr(settings, "CAPACITY_MONITOR_CHAT_ID", "")
    monkeypatch.setattr(settings, "CAPACITY_ALERT_THRESHOLD_TB", 0)


def _card_text(card):
    return json.dumps(card, ensure_ascii=False)


def test_scan_pushes_single_combined_card(patch_compute, capture_cards, two_targets):
    from core.capacity_monitor import run_capacity_scan
    run_capacity_scan()
    # 火山 + 阿里云合并为一张卡片，只推一次
    assert len(capture_cards) == 1
    chat_id, card = capture_cards[0]
    assert chat_id == "oc_test_chat"           # 回退到 FEISHU_CHAT_ID
    text = _card_text(card)
    assert "aether" in text and "egodex" in text   # 两个厂商的子目录都在同一张卡里
    assert "首次" in text                       # 首次无快照 → 合计列显示"首次"


def test_second_scan_computes_delta(patch_compute, capture_cards, two_targets, fake_redis):
    from core.capacity_monitor import run_capacity_scan
    run_capacity_scan()                         # 首次：写快照
    assert fake_redis.get(
        "capacity:snapshot:oss:wuji-bucket-hangzhou:third-party-data/") is not None

    capture_cards.clear()
    patch_compute["oss"] = [_entry("aether", 100 + 1024 ** 3, 3,
                                   [("batch-a", 100 + 1024 ** 3, 3, "")])]  # +1 GB
    run_capacity_scan()                         # 第二次：应算出 +1.0 GB

    oss_card = next(card for _chat, card in capture_cards
                    if "wuji-bucket-hangzhou" in _card_text(card))
    assert "+1.0 GB" in _card_text(oss_card)


def test_threshold_reds_header(patch_compute, capture_cards, two_targets, monkeypatch):
    from config.settings import settings
    patch_compute["oss"] = [_entry("aether", 5 * 1024 ** 4, 9, [("b", 5 * 1024 ** 4, 9, "")])]  # 5 TB
    monkeypatch.setattr(settings, "CAPACITY_ALERT_THRESHOLD_TB", 1.0)  # 阈值 1 TB → 必超
    from core.capacity_monitor import run_capacity_scan
    run_capacity_scan()
    assert any(card["header"]["template"] == "red" for _chat, card in capture_cards)


def test_target_failure_isolated(patch_compute, capture_cards, two_targets, monkeypatch):
    """一个 target 取数返回 None（凭证不可用）不应影响另一个。"""
    from tools.volcano import tos as tos_mod
    monkeypatch.setattr(tos_mod, "compute_nested_sizes", lambda *a, **k: None)
    from core.capacity_monitor import run_capacity_scan
    run_capacity_scan()
    assert len(capture_cards) == 1              # 只有 oss 成功推送
    assert "wuji-bucket-hangzhou" in _card_text(capture_cards[0][1])


def test_batch_cache_roundtrip(patch_compute, capture_cards, two_targets, fake_redis):
    """首次巡检把批次大小写入缓存；下次 _fetch_nested 会带上缓存。"""
    from core.capacity_monitor import run_capacity_scan, _load_batch_cache
    run_capacity_scan()
    cache = _load_batch_cache("oss", "wuji-bucket-hangzhou", "third-party-data/")
    # aether 的两个批次应已缓存（/ 不缓存），值含 struct 段
    assert cache.get("aether/batch-a") == (60, 1, "")
    assert cache.get("aether/batch-b") == (40, 1, "")


def test_batch_cache_passed_to_compute(patch_compute, capture_cards, two_targets, fake_redis, monkeypatch):
    """第二次巡检时，已缓存批次应作为 cached 传给 compute_nested_sizes。"""
    from core.capacity_monitor import run_capacity_scan
    run_capacity_scan()                       # 首次：建缓存

    seen = {}
    from tools.aliyun import oss as oss_mod
    def spy(**k):
        seen["cached"] = k.get("cached")
        return ([_entry("aether", 100, 2, [("batch-a", 60, 1, ""), ("batch-b", 40, 1, "")])],
                "https://oss-cn-hangzhou.aliyuncs.com")
    monkeypatch.setattr(oss_mod, "compute_nested_sizes", spy)
    run_capacity_scan()                       # 第二次：应带缓存
    assert seen["cached"].get("aether/batch-a") == (60, 1, "")


def test_no_targets_noop(capture_cards, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "CAPACITY_MONITOR_TARGETS_RAW", "[]")
    from core.capacity_monitor import run_capacity_scan
    run_capacity_scan()
    assert capture_cards == []
