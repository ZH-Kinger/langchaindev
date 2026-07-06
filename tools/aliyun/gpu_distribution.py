"""GPU 卡分布：地区×卡型 总/已用 + 每个 RAM 用户 在算卡数(+卡型)。

数据源（阿里 PAI exporter / Prometheus，主账号视角，覆盖全公司）:
- 地区×卡型 容量:  sum by(regionId,nodeGpuType)(AliyunPaiquota_NODE_GPU_ACCELERATOR_TOTAL / _REQUEST)
- 用户在算卡数:     count by(jobUserId,regionId)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE)  ← jobUserId=RAM 用户
- jobUserId→姓名:   ram.list_ram_users_api()（RAM ListUsers），team_members 兜底
- DSW 实例(补充):  pai_dsw list → 按 user_name 聚合 gpu 卡数(+gputype)

卡型精确性：每个 region 只有一种卡型（杭州/北京 H20、新加坡 H200），故"用户实例所在 region → 卡型"是精确的；
若某 region 出现多卡型，退化为该 region 首个卡型（会在文档标注）。

Redis 缓存 gpu:dist:snapshot（24h 存 / 90s 陈旧→后台单飞刷新），供实时 HTML 页面高频读取。
"""
import html as _html
import json
import logging
import threading
import time

from config.settings import settings
from utils.redis_client import get_redis

logger = logging.getLogger(__name__)

_CACHE_KEY = "gpu:dist:snapshot"
_CACHE_TTL = 24 * 3600
_STALE = 90
_LOCK_KEY = "gpu:dist:refresh_lock"
_LOCK_TTL = 180

_Q_TOTAL = "sum by (regionId,nodeGpuType)(AliyunPaiquota_NODE_GPU_ACCELERATOR_TOTAL)"
_Q_REQ = "sum by (regionId,nodeGpuType)(AliyunPaiquota_NODE_GPU_ACCELERATOR_REQUEST)"
_Q_USER = "count by (jobUserId,regionId)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE)"


def _region_name(r):
    from tools.aliyun.cluster_mfu import _region_name as rn
    return rn(r)


def _gpu_name(t):
    return settings.GPU_TYPE_DISPLAY.get(t, t or "?")


# ── 数据采集 ─────────────────────────────────────────────────────────────────────

def _name_map() -> dict:
    """jobUserId(=RAM user_id) → 姓名。RAM ListUsers 优先，team_members 兜底。"""
    m = {}
    try:
        from tools.aliyun import ram
        for u in ram.list_ram_users_api():
            uid = u.get("user_id") if isinstance(u, dict) else getattr(u, "user_id", None)
            nm = (u.get("display_name") or u.get("user_name")) if isinstance(u, dict) \
                else (getattr(u, "display_name", None) or getattr(u, "user_name", None))
            if uid:
                m[str(uid)] = nm or str(uid)
    except Exception:
        logger.warning("[GPUDIST] RAM ListUsers 失败，尝试 team_members", exc_info=True)
    if not m:
        try:
            from tools.aliyun import ram
            for x in ram.list_team_members():
                uid = x.get("user_id") if isinstance(x, dict) else getattr(x, "user_id", None)
                nm = x.get("user_name") if isinstance(x, dict) else getattr(x, "user_name", None)
                if uid:
                    m[str(uid)] = nm or str(uid)
        except Exception:
            logger.warning("[GPUDIST] team_members 兜底也失败", exc_info=True)
    return m


def _merge_dsw(users: dict) -> None:
    """把 DSW 实例的占卡并入 users(按姓名聚合)。best-effort，失败不影响 DLC 数据。"""
    try:
        from tools.aliyun.pai_dsw import _client
        from alibabacloud_pai_dsw20220101 import models as dm
        c = _client()
        if c is None:
            return
        resp = c.list_instances(dm.ListInstancesRequest(
            page_size=100, page_number=1, workspace_id=settings.PAI_DSW_WORKSPACE_ID or None))
        for inst in (resp.body.instances or []):
            rr = inst.requested_resource
            if not rr:
                continue
            try:
                gpu = int(float(rr.gpu))
            except (TypeError, ValueError):
                gpu = 0
            if gpu <= 0:
                continue
            name = inst.user_name or inst.user_id or "-"
            if name == "power-application-user":
                name = "Bot代建"
            gname = _gpu_name(getattr(rr, "gputype", "") or "")
            u = users.setdefault(name, {"name": name, "total": 0, "by_type": {}, "by_region": {}})
            u["total"] += gpu
            u["by_type"][gname] = u["by_type"].get(gname, 0) + gpu
            u["_dsw"] = u.get("_dsw", 0) + gpu
    except Exception:
        logger.warning("[GPUDIST] DSW 补充失败(忽略)", exc_info=True)


def _gather() -> dict:
    from tools.aliyun.cluster_mfu import _grouped
    totals = _grouped(_Q_TOTAL, ("regionId", "nodeGpuType"))
    reqs = _grouped(_Q_REQ, ("regionId", "nodeGpuType"))

    regions = []
    region_type = {}   # regionId -> 卡型(唯一时精确)
    for (rid, gt), tot in sorted(totals.items(), key=lambda kv: (kv[0][0], -kv[1])):
        used = int(reqs.get((rid, gt), 0.0))
        tot = int(tot)
        regions.append({
            "region": rid, "region_name": _region_name(rid),
            "gpu_type": gt, "gpu_name": _gpu_name(gt),
            "total": tot, "used": used, "free": max(0, tot - used),
            "rate": round(used / tot * 100, 1) if tot else 0.0,
        })
        region_type.setdefault(rid, gt)

    names = _name_map()
    users: dict = {}
    for (uid, rid), cnt in _grouped(_Q_USER, ("jobUserId", "regionId")).items():
        if not uid:
            continue
        name = names.get(str(uid), str(uid))
        gname = _gpu_name(region_type.get(rid, ""))
        rname = _region_name(rid)
        u = users.setdefault(name, {"name": name, "total": 0, "by_type": {}, "by_region": {}})
        u["total"] += int(cnt)
        u["by_type"][gname] = u["by_type"].get(gname, 0) + int(cnt)
        u["by_region"][rname] = u["by_region"].get(rname, 0) + int(cnt)

    _merge_dsw(users)

    users_list = sorted(users.values(), key=lambda x: -x["total"])
    return {
        "gathered_at": time.time(),
        "regions": regions,
        "users": users_list,
        "total_cards": sum(r["total"] for r in regions),
        "used_cards": sum(r["used"] for r in regions),
        "active_cards": sum(u["total"] for u in users_list),
        "user_count": len(users_list),
    }


# ── 缓存 ─────────────────────────────────────────────────────────────────────────

def _load_cache():
    try:
        raw = get_redis().get(_CACHE_KEY)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _save_cache(g):
    try:
        get_redis().setex(_CACHE_KEY, _CACHE_TTL, json.dumps(g, ensure_ascii=False))
    except Exception:
        logger.warning("[GPUDIST] 写缓存失败")


def _refresh_in_background():
    r = get_redis()
    try:
        if not r.set(_LOCK_KEY, 1, nx=True, ex=_LOCK_TTL):
            return
    except Exception:
        pass

    def _run():
        try:
            _save_cache(_gather())
        except Exception:
            logger.error("[GPUDIST] 后台采集失败", exc_info=True)
        finally:
            try:
                r.delete(_LOCK_KEY)
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()


def get_distribution(refresh: bool = False) -> dict:
    """取卡分布快照。无缓存/强制刷新→同步采集；陈旧→返回旧数据并后台刷新。"""
    g = _load_cache()
    if g is None or refresh:
        g = _gather()
        _save_cache(g)
        return g
    if time.time() - g.get("gathered_at", 0) > _STALE:
        _refresh_in_background()
    return g


# ── 展示 ─────────────────────────────────────────────────────────────────────────

def _fmt_ts(ts):
    from datetime import datetime, timezone, timedelta
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def _by_type_str(u):
    return "、".join(f"{k}×{v}" for k, v in sorted(u["by_type"].items(), key=lambda x: -x[1]))


def build_html(g: dict, token: str = "", refresh_secs: int = 30) -> str:
    """自动刷新的实时卡分布页面。前10用户直接列，其余折叠在 <details> 里。"""
    e = _html.escape
    regions = g.get("regions", [])
    users = g.get("users", [])
    # 地区×卡型 表
    rrows = []
    for r in regions:
        pct = r["rate"]
        bar = (f'<div class="bar"><div class="fill" style="width:{min(pct,100):.0f}%"></div>'
               f'<span>{r["used"]}/{r["total"]}（{pct:.0f}%）</span></div>')
        rrows.append(f'<tr><td>{e(r["region_name"])}</td><td class="t">{e(r["gpu_name"])}</td>'
                     f'<td>{bar}</td><td class="n">{r["free"]}</td></tr>')
    # 用户表：前10 + 折叠
    def urow(i, u):
        return (f'<tr><td class="n">{i}</td><td>{e(u["name"])}</td>'
                f'<td class="n b">{u["total"]}</td><td class="t">{e(_by_type_str(u))}</td></tr>')
    top = users[:10]
    rest = users[10:]
    top_html = "".join(urow(i, u) for i, u in enumerate(top, 1))
    rest_html = ""
    if rest:
        body = "".join(urow(i, u) for i, u in enumerate(rest, 11))
        rest_html = (f'<details><summary>展开其余 {len(rest)} 人</summary>'
                     f'<table class="tbl"><tbody>{body}</tbody></table></details>')
    tok = f"?token={e(token)}" if token else ""
    return f"""<!doctype html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{refresh_secs}">
<title>GPU 卡分布</title>
<style>
:root{{color-scheme:light dark}}
body{{font-family:-apple-system,'Segoe UI',Roboto,'PingFang SC',sans-serif;margin:0;padding:16px;background:#f6f7f9;color:#1f2329}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#8a919f;font-size:12px;margin-bottom:16px}}
h2{{font-size:15px;margin:20px 0 8px}}
.card{{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06);max-width:820px}}
table.tbl{{width:100%;border-collapse:collapse;font-size:14px}}
.tbl th,.tbl td{{padding:7px 8px;border-bottom:1px solid #eef0f3;text-align:left}}
.tbl th{{color:#8a919f;font-weight:600;font-size:12px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} td.b{{font-weight:700}} td.t{{color:#3370ff}}
.bar{{position:relative;height:20px;background:#eef0f3;border-radius:5px;overflow:hidden;min-width:160px}}
.bar .fill{{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#3370ff,#5b8cff)}}
.bar span{{position:relative;z-index:1;font-size:12px;line-height:20px;padding-left:8px;color:#1f2329;mix-blend-mode:difference;filter:invert(1)}}
details{{margin-top:8px}} summary{{cursor:pointer;color:#3370ff;font-size:13px;padding:6px 0}}
.kpi{{display:flex;gap:20px;margin:6px 0 2px}} .kpi div{{font-size:13px;color:#4e5969}} .kpi b{{font-size:18px;color:#1f2329}}
</style></head><body>
<div class="card">
<h1>🧮 GPU 卡分布（实时）</h1>
<div class="sub">数据时间 {e(_fmt_ts(g.get("gathered_at")))} · 每 {refresh_secs}s 自动刷新</div>
<div class="kpi"><div>总卡数<br><b>{g.get("total_cards",0)}</b></div>
<div>已分配<br><b>{g.get("used_cards",0)}</b></div>
<div>在算<br><b>{g.get("active_cards",0)}</b></div>
<div>在用人数<br><b>{g.get("user_count",0)}</b></div></div>

<h2>各地区 · 卡型（已用/总）</h2>
<table class="tbl"><thead><tr><th>地区</th><th>卡型</th><th>已用/总</th><th>空闲</th></tr></thead>
<tbody>{''.join(rrows) or '<tr><td colspan=4>无数据</td></tr>'}</tbody></table>

<h2>用户卡用量（前 10）</h2>
<table class="tbl"><thead><tr><th>#</th><th>用户</th><th>卡数</th><th>卡型</th></tr></thead>
<tbody>{top_html or '<tr><td colspan=4>无在算任务</td></tr>'}</tbody></table>
{rest_html}
<div class="sub" style="margin-top:16px">刷新地址：/gpu/distribution{tok}</div>
</div></body></html>"""


def dist_url() -> str:
    """实时页面完整链接（带 token）；未配 GPU_DIST_BASE_URL 时返回空串。"""
    base = (settings.GPU_DIST_BASE_URL or "").rstrip("/")
    if not base:
        return ""
    token = settings.RAM_QUERY_API_TOKEN or settings.FEISHU_VERIFICATION_TOKEN or ""
    q = f"?token={token}" if token else ""
    return f"{base}/gpu/distribution{q}"


def summary_card(g: dict, url: str = ""):
    """飞书摘要卡：地区×卡型 + 前10用户 + 打开实时页面按钮。"""
    from tools.feishu.cards import btn, buttons, card, div, hr
    lines = ["**各地区 · 卡型（已用/总）**"]
    for r in g.get("regions", []):
        lines.append(f"- {r['region_name']} · {r['gpu_name']}　**{r['used']}/{r['total']}**"
                     f"（{r['rate']:.0f}%，空闲 {r['free']}）")
    lines.append("\n**用户卡用量（前10 · 在算）**")
    for i, u in enumerate(g.get("users", [])[:10], 1):
        lines.append(f"{i}. {u['name']}　**{u['total']}** 卡（{_by_type_str(u)}）")
    n = g.get("user_count", 0)
    if n > 10:
        lines.append(f"\n> 其余 {n - 10} 人请点下方「实时页面」展开查看。")
    elems = [div("\n".join(lines)),
             div(f"总卡 {g.get('total_cards',0)} · 已分配 {g.get('used_cards',0)} · "
                 f"在算 {g.get('active_cards',0)} · 数据 {_fmt_ts(g.get('gathered_at'))}")]
    if url:
        elems += [hr(), buttons(btn("🖥 打开实时页面", url=url))]
    return card("🧮 GPU 卡分布", elems, color="blue")
