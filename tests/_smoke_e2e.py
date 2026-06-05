"""端到端业务烟测（手动跑）。

覆盖 6 个场景：
  1. 飞书绑定流程：绑定 → 查看 → 解绑
  2. 意图识别准确性：6 个真实输入
  3. GPU 申请流程：Agent 路由到 pai_dsw，confirm 拦截
  4. 凭证三级降级：用户 AK / STS / 全局 切换正确
  5. 写操作 confirm 拦截：ECS/OSS/SLS 各一个写操作
  6. 读操作（如配置可用）：ECS list / OSS list_buckets / SLS list_projects

跑：PYTHONIOENCODING=utf-8 python tests/_smoke_e2e.py
依赖：.env 配 OPENAI_API_KEY（用于 LLM 路由和 Agent）。
mock：阿里云写操作 SDK、STS、Jira 创建、飞书消息推送。
"""
import sys
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── 工具函数 ─────────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
SKIP = "⊘"
INFO = "ℹ"

results = {"pass": 0, "fail": 0, "skip": 0, "details": []}


def report(scenario: str, ok: bool, detail: str = "", skipped: bool = False):
    if skipped:
        mark = SKIP
        results["skip"] += 1
    elif ok:
        mark = PASS
        results["pass"] += 1
    else:
        mark = FAIL
        results["fail"] += 1
    line = f"  {mark} {scenario}"
    if detail:
        line += f"\n     {detail}"
    print(line)
    results["details"].append({"scenario": scenario, "ok": ok, "skipped": skipped, "detail": detail})


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── 场景 1：飞书绑定流程 ────────────────────────────────────────────────────

def scenario_1_feishu_bind():
    section("场景 1：飞书 AK 绑定流程（用 fakeredis 隔离）")
    from utils import redis_client
    import fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch.object(redis_client, "_client", fake), \
         patch.object(redis_client, "get_redis", lambda: fake), \
         patch.object(redis_client, "is_redis_available", lambda: True):

        from utils.aliyun_user_creds import save_user_ak, get_user_ak, has_user_ak, get_user_ak_meta, delete_user_ak

        OPEN_ID = "ou_e2e_test"
        AK = "LTAI5tE2E_AK_FAKE"
        SK = "E2E_SECRET_FAKE_VALUE_XYZ"

        # 1.1 绑定
        ok = save_user_ak(OPEN_ID, AK, SK)
        report("save_user_ak 成功返回 True", ok)

        # 1.2 has_user_ak
        report("has_user_ak 返回 True", has_user_ak(OPEN_ID))

        # 1.3 get_user_ak roundtrip
        result = get_user_ak(OPEN_ID)
        report("get_user_ak roundtrip 正确",
               result == (AK, SK),
               f"得到 {result}")

        # 1.4 加密落地校验（仅当 BOT_CREDS_ENCRYPTION_KEY 已配置时才严格断言）
        from config.settings import settings
        raw = fake.hgetall(f"feishu:user_creds:{OPEN_ID}")
        leaked = (AK in str(raw)) or (SK in str(raw))
        if settings.BOT_CREDS_ENCRYPTION_KEY:
            report("Redis 中无明文 AK/SK（加密 key 已配置）",
                   not leaked,
                   f"密文前缀: {raw.get('ak_id_enc', '')[:20]}...")
        else:
            report("Redis 加密校验（BOT_CREDS_ENCRYPTION_KEY 未配置）",
                   False, skipped=True,
                   detail="生成命令：python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")

        # 1.5 meta 脱敏
        meta = get_user_ak_meta(OPEN_ID)
        meta_masked = meta and meta["ak_id_masked"].endswith("****")
        report("get_user_ak_meta 脱敏正确",
               bool(meta_masked),
               f"masked={meta['ak_id_masked'] if meta else None}")

        # 1.6 解绑
        deleted = delete_user_ak(OPEN_ID)
        report("delete_user_ak 成功", deleted)

        # 1.7 解绑后查询为 None
        report("解绑后 get_user_ak 返回 None",
               get_user_ak(OPEN_ID) is None)


# ── 场景 2：意图识别准确性（真实 LLM）────────────────────────────────────────

def scenario_2_intent_routing():
    section("场景 2：意图识别准确性（真实 LLM 调用）")
    from config.settings import settings
    if not settings.API_KEY:
        report("LLM 路由（OPENAI_API_KEY 未配置）", False, skipped=True)
        return

    from core.intent_router import route, clear_cache

    cases = [
        # (输入, 期望识别的意图集合中至少有一个)
        ("整个集群效率不太行",          {"advisor", "cluster"}),
        ("跑训练慢得不行",             {"training", "advisor"}),
        ("我的服务挂了快帮我看看",      {"ops", "monitor"}),
        ("成本能不能降一些",           {"advisor"}),
        ("发个飞书通知运维同事",        {"notify"}),
        ("看下昨天 PR 都合了哪些",      {"workflow"}),
    ]

    total_latency = 0.0
    for text, expected_any in cases:
        clear_cache()
        t0 = time.time()
        intents = route(text)
        elapsed_ms = (time.time() - t0) * 1000
        total_latency += elapsed_ms
        hit = bool(set(intents) & expected_any)
        report(
            f'"{text}"',
            hit,
            f"LLM 输出={intents}, 期望含其一={list(expected_any)}, 耗时 {elapsed_ms:.0f}ms"
        )

    print(f"\n  {INFO} 平均 LLM 路由延迟：{total_latency / len(cases):.0f}ms")


# ── 场景 3：GPU 申请流程（Agent + confirm 拦截）─────────────────────────────

def scenario_3_gpu_request_flow():
    section("场景 3：GPU 申请意图触达 pai_dsw 工具（不真实创建）")
    from core.agent import _select_tools

    text = "帮我申请一个 dsw 实例"
    tools = _select_tools(text)
    tool_names = {t.name for t in tools}
    report(
        '"帮我申请一个 dsw 实例" → 命中 manage_pai_dsw',
        "manage_pai_dsw" in tool_names,
        f"工具集 = {sorted(tool_names)}"
    )

    # 直接调 manage_pai_dsw 创建动作，open_id 注入但凭证缺失（拦截或友好错误）
    from tools.aliyun.pai_dsw import manage_pai_dsw

    # mock client 工厂返回 None，避免真调云
    from utils import aliyun_client_factory
    with patch.object(aliyun_client_factory, "get_pai_dsw_client", lambda *a, **k: None):
        result = manage_pai_dsw(
            action="create_json",
            create_config_json='{"instanceName":"e2e-fake"}',
            open_id="ou_e2e_fake",
        )
    report(
        "凭证不可用时 manage_pai_dsw 返回友好错误（不抛异常）",
        ("凭证不可用" in result or "❌" in result),
        f"返回：{result[:80]}..."
    )


# ── 场景 4：凭证三级降级 ──────────────────────────────────────────────────

def scenario_4_credential_priority():
    section("场景 4：凭证三级降级路径（用户 AK > STS > 全局）")
    import fakeredis
    from utils import redis_client, aliyun_client_factory
    from config.settings import settings

    fake = fakeredis.FakeRedis(decode_responses=True)

    with patch.object(redis_client, "_client", fake), \
         patch.object(redis_client, "get_redis", lambda: fake), \
         patch.object(redis_client, "is_redis_available", lambda: True):

        # 4.1 用户 AK 路径
        from utils.aliyun_user_creds import save_user_ak
        save_user_ak("ou_with_ak", "LTAI_USER_AK", "USER_SK")
        cred = aliyun_client_factory._resolve_cred("ou_with_ak")
        report(
            "用户 AK 优先（ou_with_ak）",
            cred and cred["ak"] == "LTAI_USER_AK" and not cred.get("token"),
            f"得到 ak={cred and cred['ak'][:8]}*** token={cred and cred.get('token')!r}"
        )

        # 4.2 STS 路径（mock STS 返回临时凭证）
        fake_sts = {
            "access_key_id": "STS.FAKE_TEMP",
            "access_key_secret": "STS_SECRET",
            "security_token": "STS_TOKEN_XYZ",
            "expire_ts": time.time() + 3600,
            "role_arn": "acs:ram::1234567890:role/BotRole-Default",
        }
        with patch.object(aliyun_client_factory, "assume_role_for_user",
                          lambda oid: fake_sts):
            cred = aliyun_client_factory._resolve_cred("ou_no_ak_has_sts")
            report(
                "STS 降级（无用户 AK 时）",
                cred and cred["ak"] == "STS.FAKE_TEMP" and cred["token"] == "STS_TOKEN_XYZ",
                f"得到 ak={cred and cred['ak']!r}"
            )

        # 4.3 全局兜底（无用户 AK、STS 失败）
        original_pai_ak = settings.PAI_DSW_ACCESS_KEY_ID
        original_pai_sk = settings.PAI_DSW_ACCESS_KEY_SECRET
        try:
            settings.PAI_DSW_ACCESS_KEY_ID = "LTAI_GLOBAL_AK"
            settings.PAI_DSW_ACCESS_KEY_SECRET = "GLOBAL_SK"
            with patch.object(aliyun_client_factory, "assume_role_for_user", lambda oid: None):
                cred = aliyun_client_factory._resolve_cred("ou_no_anything")
                report(
                    "全局 AK 兜底",
                    cred and cred["ak"] == "LTAI_GLOBAL_AK",
                    f"得到 ak={cred and cred['ak']!r}"
                )

            # 4.4 全都没有 → None
            settings.PAI_DSW_ACCESS_KEY_ID = ""
            settings.PAI_DSW_ACCESS_KEY_SECRET = ""
            with patch.object(aliyun_client_factory, "assume_role_for_user", lambda oid: None):
                cred = aliyun_client_factory._resolve_cred("ou_nothing")
                report("全部缺失返回 None", cred is None, f"得到 {cred}")
        finally:
            settings.PAI_DSW_ACCESS_KEY_ID = original_pai_ak
            settings.PAI_DSW_ACCESS_KEY_SECRET = original_pai_sk


# ── 场景 5：写操作 confirm 拦截 ────────────────────────────────────────────

def scenario_5_write_confirm_intercept():
    section("场景 5：ECS/OSS/SLS 写操作必须 confirm=true（不创建真实资源）")
    from tools.aliyun import ecs, oss, sls
    from utils import aliyun_client_factory

    # mock 工厂函数返回 fake client（不会真调云）
    class _FakeClient:
        def __getattr__(self, name):
            def _err(*a, **k):
                raise AssertionError(f"client.{name} 不应被调用（confirm 应已拦截）")
            return _err

    with patch.object(aliyun_client_factory, "get_pai_dsw_client", lambda *a, **k: _FakeClient()), \
         patch.object(ecs, "get_ecs_client", lambda **k: _FakeClient()), \
         patch.object(oss, "get_oss_service", lambda **k: _FakeClient()), \
         patch.object(oss, "get_oss_bucket", lambda *a, **k: _FakeClient()), \
         patch.object(sls, "get_sls_client", lambda **k: _FakeClient()):

        # ECS create 无 confirm
        result = ecs.manage_ecs(action="create",
                                  instance_type="ecs.g7.large",
                                  image_id="ubuntu_x", vswitch_id="vsw_x",
                                  security_group_id="sg_x", confirm=False,
                                  open_id="ou_e2e")
        report("ECS create 无 confirm → 拦截",
               "confirm=true" in result,
               f"返回：{result[:80]}")

        # ECS delete 无 confirm
        result = ecs.manage_ecs(action="delete", instance_id="i-test",
                                  confirm=False, open_id="ou_e2e")
        report("ECS delete 无 confirm → 拦截", "confirm=true" in result)

        # OSS create_bucket 无 confirm
        result = oss.manage_oss(action="create_bucket", bucket="e2e-test",
                                  confirm=False, open_id="ou_e2e")
        report("OSS create_bucket 无 confirm → 拦截", "confirm=true" in result)

        # OSS put_object 无 confirm
        result = oss.manage_oss(action="put_object", bucket="b", object_key="k",
                                  content="x", confirm=False, open_id="ou_e2e")
        report("OSS put_object 无 confirm → 拦截", "confirm=true" in result)

        # OSS delete_object 无 confirm
        result = oss.manage_oss(action="delete_object", bucket="b", object_key="k",
                                  confirm=False, open_id="ou_e2e")
        report("OSS delete_object 无 confirm → 拦截", "confirm=true" in result)

        # SLS create_project 无 confirm
        result = sls.manage_sls(action="create_project", project="e2e",
                                  confirm=False, open_id="ou_e2e")
        report("SLS create_project 无 confirm → 拦截", "confirm=true" in result)


# ── 场景 6：读操作真实调用（如可用）───────────────────────────────────────

def scenario_6_real_read_operations():
    section("场景 6：阿里云读操作真实调用（如配置可用）")
    from config.settings import settings

    has_global_ak = bool(settings.PAI_DSW_ACCESS_KEY_ID and settings.PAI_DSW_ACCESS_KEY_SECRET)
    has_aliyun_ak = bool(settings.ALIYUN_ACCESS_KEY_ID and settings.ALIYUN_ACCESS_KEY_SECRET)

    if not (has_global_ak or has_aliyun_ak):
        report("跳过：未配置任何阿里云 AK", False, skipped=True,
               detail="设置 PAI_DSW_ACCESS_KEY_ID/SECRET 或 ALIYUN_ACCESS_KEY_ID/SECRET 可启用")
        return

    # PAI DSW list（如配置）
    if has_global_ak and settings.PAI_DSW_WORKSPACE_ID:
        try:
            from tools.aliyun.pai_dsw import manage_pai_dsw
            t0 = time.time()
            result = manage_pai_dsw(action="list")
            elapsed = (time.time() - t0) * 1000
            ok = "❌" not in result and "Traceback" not in result
            report("PAI DSW list", ok, f"耗时 {elapsed:.0f}ms, 返回前 80 字={result[:80]}")
        except Exception as e:
            report("PAI DSW list", False, f"异常：{e}")
    else:
        report("PAI DSW list 跳过（凭证或 WORKSPACE 缺失）", False, skipped=True)

    # Prometheus（如配置）
    if settings.PROMETHEUS_URL and has_aliyun_ak:
        try:
            from tools.aliyun.prometheus import query_prometheus_metrics
            t0 = time.time()
            result = query_prometheus_metrics(query_type="discover")
            elapsed = (time.time() - t0) * 1000
            ok = "❌" not in result
            report("Prometheus discover", ok, f"耗时 {elapsed:.0f}ms, 返回前 80 字={result[:80]}")
        except Exception as e:
            report("Prometheus discover", False, f"异常：{e}")
    else:
        report("Prometheus 跳过（URL 或凭证缺失）", False, skipped=True)


# ── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 70)
    print("█  端到端业务烟测  v1.0")
    print("█  不创建真实云资源 — 写操作全部 mock，读操作按配置可选")
    print("█" * 70)

    t_start = time.time()

    try:
        scenario_1_feishu_bind()
    except Exception as e:
        print(f"\n{FAIL} 场景 1 异常：{e}")
        results["fail"] += 1
        import traceback; traceback.print_exc()

    try:
        scenario_2_intent_routing()
    except Exception as e:
        print(f"\n{FAIL} 场景 2 异常：{e}")
        results["fail"] += 1
        import traceback; traceback.print_exc()

    try:
        scenario_3_gpu_request_flow()
    except Exception as e:
        print(f"\n{FAIL} 场景 3 异常：{e}")
        results["fail"] += 1
        import traceback; traceback.print_exc()

    try:
        scenario_4_credential_priority()
    except Exception as e:
        print(f"\n{FAIL} 场景 4 异常：{e}")
        results["fail"] += 1
        import traceback; traceback.print_exc()

    try:
        scenario_5_write_confirm_intercept()
    except Exception as e:
        print(f"\n{FAIL} 场景 5 异常：{e}")
        results["fail"] += 1
        import traceback; traceback.print_exc()

    try:
        scenario_6_real_read_operations()
    except Exception as e:
        print(f"\n{FAIL} 场景 6 异常：{e}")
        results["fail"] += 1
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    section(f"总结（{elapsed:.1f}s）")
    print(f"  {PASS} 通过：{results['pass']}")
    print(f"  {FAIL} 失败：{results['fail']}")
    print(f"  {SKIP} 跳过：{results['skip']}")

    return results


if __name__ == "__main__":
    main()
