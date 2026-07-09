"""#46 C3：settings._REQUIRED_FIELDS 补三个安全关键字段，validate() 缺失时列出、齐全时不列。

  - BOT_CREDS_ENCRYPTION_KEY   缺 → 绑 AK 静默失败（CryptoNotConfigured）
  - ALIYUN_BOT_MASTER_AK_ID    缺 → STS 全挂、静默退全局 AK（越权+审计错人）
  - ALIYUN_BOT_MASTER_AK_SECRET 同上
"""
import pytest

_NEW_FIELDS = ["BOT_CREDS_ENCRYPTION_KEY", "ALIYUN_BOT_MASTER_AK_ID", "ALIYUN_BOT_MASTER_AK_SECRET"]


def test_new_fields_registered_in_required():
    from config.settings import settings
    names = {f for f, _impact in settings._REQUIRED_FIELDS}
    for f in _NEW_FIELDS:
        assert f in names, f"{f} 未登记进 _REQUIRED_FIELDS"


@pytest.mark.parametrize("field", _NEW_FIELDS)
def test_validate_lists_field_when_missing(monkeypatch, field):
    """该字段为空 → validate() 结果里出现它（其余全填以隔离）。"""
    from config.settings import settings
    for f, _impact in settings._REQUIRED_FIELDS:
        monkeypatch.setattr(settings, f, "x", raising=False)
    monkeypatch.setattr(settings, field, "", raising=False)

    missing = dict(settings.validate())
    assert field in missing
    assert missing[field]                    # 附影响说明，非空


def test_validate_omits_new_fields_when_all_present(monkeypatch):
    """三字段齐全 → validate() 不再列出它们。"""
    from config.settings import settings
    for f, _impact in settings._REQUIRED_FIELDS:
        monkeypatch.setattr(settings, f, "filled", raising=False)

    missing = {f for f, _i in settings.validate()}
    for f in _NEW_FIELDS:
        assert f not in missing
    assert missing == set()                  # 全填 → 完全无缺失
