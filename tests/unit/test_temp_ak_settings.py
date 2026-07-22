"""#57 临时 AK 发放 —— settings.validate() 启用/关闭时的告警。

重构后无邮箱下发 → SMTP_HOST 已从 validate 移除。TEMP_AK_ENABLED 时缺
APPROVAL_CODE / OSS_ROLE_ARN / ALIYUN_ACCESS_KEY_ID 各告警；关闭时不告警。
用 monkeypatch 直改单例属性 + 调 validate()（不 reload，无污染）。
"""
import pytest

from config.settings import settings

# TEMP_AK 块的三个字段 → 各自 impact 关键片段（用于在 missing 列表里精确定位）
_CHECKS = {
    "TEMP_AK_APPROVAL_CODE": "审批模板",
    "TEMP_AK_OSS_ROLE_ARN":  "STS 分支",
    "ALIYUN_ACCESS_KEY_ID":  "方案 B",
}


@pytest.fixture
def enabled_full(monkeypatch):
    """TEMP_AK 启用、三字段全填、PFS 关（隔离噪声）。"""
    monkeypatch.setattr(settings, "TEMP_AK_ENABLED", True)
    monkeypatch.setattr(settings, "PFS_TRANSFER_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "TEMP_AK_APPROVAL_CODE", "code")
    monkeypatch.setattr(settings, "TEMP_AK_OSS_ROLE_ARN", "acs:ram::1:role/w")
    monkeypatch.setattr(settings, "ALIYUN_ACCESS_KEY_ID", "LTAI_x")


def _temp_ak_missing(missing):
    """从 validate() 返回里挑出 TEMP_AK 块的字段。"""
    return {f for f, _impact in missing if f in _CHECKS}


def test_disabled_no_temp_ak_warnings(monkeypatch):
    monkeypatch.setattr(settings, "TEMP_AK_ENABLED", False)
    # 即便四字段全空，关闭时不为它们告警
    monkeypatch.setattr(settings, "TEMP_AK_APPROVAL_CODE", "")
    monkeypatch.setattr(settings, "TEMP_AK_OSS_ROLE_ARN", "")
    missing = settings.validate()
    # APPROVAL_CODE / OSS_ROLE_ARN 是 TEMP_AK 独有，关闭时不该出现
    fields = {f for f, _ in missing}
    assert "TEMP_AK_APPROVAL_CODE" not in fields
    assert "TEMP_AK_OSS_ROLE_ARN" not in fields


def test_enabled_all_filled_no_temp_ak_warnings(enabled_full):
    assert _temp_ak_missing(settings.validate()) == set()


@pytest.mark.parametrize("field", list(_CHECKS))
def test_enabled_missing_each_field_warns(enabled_full, monkeypatch, field):
    monkeypatch.setattr(settings, field, "")
    missing = settings.validate()
    fields = {f for f, _ in missing}
    assert field in fields, f"{field} 空时应告警"
    # 校验 impact 文案对得上（避免与其它块串号）
    impact = next(imp for f, imp in missing if f == field)
    assert _CHECKS[field] in impact


def test_enabled_all_empty_warns_all(enabled_full, monkeypatch):
    for f in _CHECKS:
        monkeypatch.setattr(settings, f, "")
    assert _temp_ak_missing(settings.validate()) == set(_CHECKS)


def test_smtp_host_no_longer_validated(enabled_full, monkeypatch):
    """重构后无邮箱下发：SMTP_HOST 空也不为 temp_ak 告警。"""
    monkeypatch.setattr(settings, "SMTP_HOST", "", raising=False)
    missing = settings.validate()
    # validate 里 temp_ak 块不再含 SMTP_HOST（其它块也不因 temp_ak 引入它）
    tak_related = [imp for f, imp in missing if "临时 AK" in imp]
    assert not any("SMTP" in imp for imp in tak_related)
