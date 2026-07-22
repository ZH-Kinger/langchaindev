"""临时 AK 发放 CLI（运维/应急）。

  python -m core.temp_ak_issuance.cli plan --bucket B --read p/ --write q/ --expire "2026-08-01 00:00:00"
  python -m core.temp_ak_issuance.cli status --grant-id tak-xxxx
  python -m core.temp_ak_issuance.cli revoke --grant-id tak-xxxx [--apply]
  python -m core.temp_ak_issuance.cli sweep [--apply]

发放（issue）**只经飞书审批**，CLI 不提供 issue（杜绝绕审批发凭证）。revoke/sweep 是服务器特权运维，
dry-run 默认，--apply 才真删。
"""
from __future__ import annotations

import argparse
import json
import time

from . import cleanup, issuer, orchestrator, policy


def _parse_dt(s: str) -> float:
    from datetime import datetime, timedelta, timezone
    s = (s or "").strip()
    if not s:
        return 0.0
    if s.isdigit():
        v = float(s)
        return v / 1000 if v > 1e11 else v
    bj = timezone(timedelta(hours=8))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=bj).timestamp()
        except ValueError:
            continue
    raise SystemExit(f"无法解析时间：{s!r}")


def _cmd_plan(a) -> None:
    now = time.time()
    nb = _parse_dt(a.not_before) or now
    exp = _parse_dt(a.expire)
    if not exp:
        raise SystemExit("--expire 必填")
    grant = {
        "grant_id": "tak-dryrun",
        "bucket": a.bucket,
        "read_prefixes": a.read or [],
        "write_prefixes": a.write or [],
        "not_before": nb,
        "expire": exp,
        "source_ips": a.source_ip or [],
        "mode": issuer.classify_mode(exp, now),
        "user_name": "tempak-dryrun",
        "policy_name": policy.POLICY_PREFIX + "tempak-dryrun",
    }
    print(f"分流：{grant['mode']}（expire-now={int(exp-now)}s，上限"
          f" {issuer.settings.TEMP_AK_STS_MAX_SECONDS}s）")
    print(f"授权：{orchestrator.scope_line(grant)}")
    print(f"有效期：{orchestrator.fmt_window(grant)}")
    p = issuer.plan(grant)
    print("policy 预览：")
    print(json.dumps(p["policy"], ensure_ascii=False, indent=2))


def _cmd_status(a) -> None:
    g = orchestrator.get_grant(a.grant_id)
    if not g:
        raise SystemExit(f"未找到 grant {a.grant_id}")
    safe = {k: v for k, v in g.items()}  # 记录本就不含 secret/token
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def _cmd_revoke(a) -> None:
    g = orchestrator.get_grant(a.grant_id)
    if not g:
        raise SystemExit(f"未找到 grant {a.grant_id}")
    print(f"将吊销：{a.grant_id} mode={g.get('mode')} user={g.get('user_name')} ak={g.get('ak_id')}")
    if not a.apply:
        print("（dry-run；加 --apply 真删）")
        return
    ok = cleanup.revoke_grant(g, log=print)
    print("✓ 已吊销" if ok else "✗ 吊销失败（见日志）")


def _cmd_sweep(a) -> None:
    if not a.apply:
        # dry-run：只列到期未吊销的 grant，不删
        from utils.redis_client import get_redis
        now = time.time()
        n = 0
        for key in get_redis().scan_iter(orchestrator._KEY_PREFIX + "*"):
            raw = get_redis().get(key)
            if not raw:
                continue
            g = json.loads(raw)
            if g.get("stage") == orchestrator.STAGE_ISSUED and float(g.get("expire", 0)) < now:
                print(f"到期待清理：{g['grant_id']} user={g.get('user_name')} "
                      f"expire={orchestrator.fmt_ts(g.get('expire'))}")
                n += 1
        print(f"共 {n} 个到期 grant（dry-run；加 --apply 硬删）")
        return
    revoked = cleanup.sweep_expired()
    print(f"已硬删 {len(revoked)} 个到期 grant：{', '.join(revoked) or '（无）'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="临时 AK/SK 发放运维 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="dry-run 预览分流 + policy")
    p.add_argument("--bucket", required=True)
    p.add_argument("--read", action="append", help="读前缀（可多次）")
    p.add_argument("--write", action="append", help="写前缀（可多次）")
    p.add_argument("--not-before", default="", dest="not_before")
    p.add_argument("--expire", required=True)
    p.add_argument("--source-ip", action="append", dest="source_ip")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("status", help="查 grant 记录")
    p.add_argument("--grant-id", required=True)
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("revoke", help="手动吊销一个 grant")
    p.add_argument("--grant-id", required=True)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=_cmd_revoke)

    p = sub.add_parser("sweep", help="扫到期 grant 硬删")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=_cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
