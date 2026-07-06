"""GPU 卡分布：地区×卡型 总/已用 + 每个 RAM 用户 在算卡数(+卡型)。

数据源（阿里 PAI exporter / Prometheus，主账号视角，覆盖全公司）:
- 地区×卡型 容量:  sum by(regionId,nodeGpuType)(AliyunPaiquota_NODE_GPU_ACCELERATOR_TOTAL / _REQUEST)
- 用户在算卡数:     count by(jobUserId,regionId)(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE)  ← jobUserId=RAM 用户
- jobUserId→姓名:   ram.list_ram_users_api()（RAM ListUsers），team_members 兜底
- DSW 实例(补充):  pai_dsw list → 按 user_name 聚合 gpu 卡数(+gputype)

卡型精确性：每个 region 只有一种卡型（杭州/北京 H20、新加坡 H200），故"用户实例所在 region → 卡型"是精确的；
若某 region 出现多卡型，退化为该 region 首个卡型（会在文档标注）。

Redis 缓存 gpu:dist:snapshot（24h 存 / 15s 陈旧→后台单飞刷新），供实时 HTML 页面高频读取。
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
_STALE = 15
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


# ── 时序（近 N 小时趋势，供实时折线图）─────────────────────────────────────────────

_TS_KEY = "gpu:dist:timeseries"
_TS_TTL = 24 * 3600
_TS_STALE = 15
_TS_LOCK = "gpu:dist:ts_lock"

ALLOWED_HOURS = (1, 6, 12, 24, 72)
_STEP_BY_HOURS = {1: "60s", 6: "300s", 12: "600s", 24: "900s", 72: "3600s"}


def _ikey(d):
    return {int(round(float(k))): v for k, v in d.items()}


def _gather_timeseries(hours: int = 24) -> dict:
    """近 hours 小时趋势：per-region MFU(张量核=avg(TENSORTFLOPS)/峰值) + 集群 GPU利用率/Tensor活跃
    + 卡数(已分配/在算/空闲)。步长随范围自适应，保持 ~60-100 点。"""
    from datetime import datetime, timezone, timedelta
    from tools.aliyun.cluster_mfu import _range_series, _peak_for, _grouped
    thr = settings.GPU_ACTIVE_THRESHOLD_PCT
    step = _STEP_BY_HOURS.get(hours, "900s")
    end = time.time()
    start = end - hours * 3600
    rt = {}
    for (rid, gt) in _grouped(_Q_TOTAL, ("regionId", "nodeGpuType")):
        rt.setdefault(rid, gt)

    duty = _ikey(_range_series("avg(AliyunPaidlc_CARD_GPU_DUTY_CYCLE)", start, end, step))
    tensor = _ikey(_range_series("avg(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE)", start, end, step))
    active = _ikey(_range_series(f"count(AliyunPaidlc_CARD_GPU_PIP_TENSOR_ACTIVE > {thr})", start, end, step))
    alloc = _ikey(_range_series("sum(AliyunPaiquota_NODE_GPU_ACCELERATOR_REQUEST)", start, end, step))
    total = _ikey(_range_series("sum(AliyunPaiquota_NODE_GPU_ACCELERATOR_TOTAL)", start, end, step))

    mfu_r = {}
    for rid, gt in rt.items():
        peak = _peak_for(gt)
        # 张量核 MFU：每卡平均张量算力 / 单卡峰值（同一批卡、不会除0、空/满区都出线）
        expr = f'avg(AliyunPaidlc_CARD_GPU_TENSORTFLOPS_USED{{regionId="{rid}"}}) / {peak} * 100'
        s = _ikey(_range_series(expr, start, end, step))
        if s:
            mfu_r[f"{_region_name(rid)}·{_gpu_name(gt)}"] = s

    keys = set(duty) | set(tensor) | set(active) | set(alloc) | set(total)
    for d in mfu_r.values():
        keys |= set(d)
    grid = sorted(keys)
    bj = timezone(timedelta(hours=8))
    fmt = "%m-%d %H:%M" if hours >= 24 else "%H:%M"
    labels = [datetime.fromtimestamp(t, bj).strftime(fmt) for t in grid]

    def arr(d, nd=1):
        return [round(d[t], nd) if t in d else None for t in grid]

    idle = [round(total[t] - alloc[t]) if (t in total and t in alloc) else None for t in grid]

    return {
        "labels": labels,
        "gpu_util": arr(duty),
        "tensor": arr(tensor, 2),
        "cards": {"allocated": arr(alloc, 0), "active": arr(active, 0), "idle": idle},
        "mfu": {name: arr(s, 2) for name, s in mfu_r.items()},
        "hours": hours,
        "gathered_at": time.time(),
    }


def get_timeseries(hours: int = 24, refresh: bool = False) -> dict:
    """取趋势快照(按时间范围缓存)。无缓存/强制→同步采集；陈旧→返回旧数据并后台单飞刷新。"""
    hours = hours if hours in ALLOWED_HOURS else 24
    key = f"{_TS_KEY}:{hours}"
    try:
        raw = get_redis().get(key)
        g = json.loads(raw) if raw else None
    except Exception:
        g = None
    if g is None or refresh:
        g = _gather_timeseries(hours)
        try:
            get_redis().setex(key, _TS_TTL, json.dumps(g, ensure_ascii=False))
        except Exception:
            pass
        return g
    if time.time() - g.get("gathered_at", 0) > _TS_STALE:
        r = get_redis()
        try:
            if r.set(f"{_TS_LOCK}:{hours}", 1, nx=True, ex=180):
                def _run():
                    try:
                        r.setex(key, _TS_TTL, json.dumps(_gather_timeseries(hours), ensure_ascii=False))
                    except Exception:
                        logger.error("[GPUDIST] 趋势后台采集失败", exc_info=True)
                    finally:
                        try:
                            r.delete(f"{_TS_LOCK}:{hours}")
                        except Exception:
                            pass
                threading.Thread(target=_run, daemon=True).start()
        except Exception:
            pass
    return g


# ── 展示 ─────────────────────────────────────────────────────────────────────────

def _fmt_ts(ts):
    from datetime import datetime, timezone, timedelta
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def _by_type_str(u):
    return "、".join(f"{k}×{v}" for k, v in sorted(u["by_type"].items(), key=lambda x: -x[1]))


def build_html(g: dict, series: dict = None, token: str = "", refresh_secs: int = 15) -> str:
    """自动刷新的实时卡分布页面（居中）：近6h 趋势折线图(MFU/GPU利用率/Tensor活跃/在算卡数)
    + 地区×卡型表 + 用户表(前10+折叠) + 手动刷新按钮。

    注意：token 不写进页面正文（避免暴露）；手动刷新按钮用 JS 从地址栏取 token 发 refresh=1 请求。
    Chart.js 走 CDN，加载失败时图表区留空、表格照常（渐进增强）。
    """
    e = _html.escape
    series = series or {}
    regions = g.get("regions", [])
    users = g.get("users", [])
    rrows = []
    for r in regions:
        pct = r["rate"]
        bar = (f'<div class="bar"><div class="fill" style="width:{min(pct,100):.0f}%"></div>'
               f'<span>{r["used"]}/{r["total"]}（{pct:.0f}%）</span></div>')
        rrows.append(f'<tr><td>{e(r["region_name"])}</td><td class="t">{e(r["gpu_name"])}</td>'
                     f'<td>{bar}</td><td class="n">{r["free"]}</td></tr>')

    def urow(i, u):
        return (f'<tr><td class="n">{i}</td><td>{e(u["name"])}</td>'
                f'<td class="n b">{u["total"]}</td><td class="t">{e(_by_type_str(u))}</td></tr>')

    n = g.get("user_count", len(users))
    top, rest = users[:10], users[10:]
    top_html = "".join(urow(i, u) for i, u in enumerate(top, 1))
    rest_html = ""
    if rest:
        body = "".join(urow(i, u) for i, u in enumerate(rest, 11))
        rest_html = (f'<details><summary>展开其余 {len(rest)} 人</summary>'
                     f'<table class="tbl"><tbody>{body}</tbody></table></details>')
    # 手动刷新：读地址栏 token，请求 refresh=1 强制重采，再刷新当前页（不把 token 写进正文）
    refresh_js = ("const u=new URL(location.href);u.searchParams.set('refresh','1');"
                  "this.textContent='⏳ 采集中…';fetch(u).then(()=>location.reload())"
                  ".catch(()=>location.reload())")

    # 趋势图（渐进增强：Chart.js CDN；数据内嵌，token 不入正文）
    charts_html = ""
    chart_js = ""
    if series.get("labels"):
        cur_h = series.get("hours", 24)
        rbtns = []
        for h, lbl in ((1, "1h"), (6, "6h"), (12, "12h"), (24, "24h"), (72, "3天")):
            on = " on" if h == cur_h else ""
            rbtns.append(f'<button class="rg{on}" onclick="const u=new URL(location.href);'
                         f"u.searchParams.set('hours','{h}');location.href=u\">{lbl}</button>")
        range_html = '<div class="ranges">' + "".join(rbtns) + "</div>"
        charts_html = (
            f'<div class="hd" style="margin-top:20px"><h2 style="margin:0">趋势</h2>{range_html}</div>'
            '<div class="charts">'
            '<div class="chart"><div class="ct">张量核利用率（实测%，按地区，≈HFU）</div><div class="cv"><canvas id="c_mfu"></canvas></div></div>'
            '<div class="chart"><div class="ct">GPU 利用率 DutyCycle（%）</div><div class="cv"><canvas id="c_util"></canvas></div></div>'
            '<div class="chart"><div class="ct">Tensor Core 活跃度（%）</div><div class="cv"><canvas id="c_tensor"></canvas></div></div>'
            '<div class="chart"><div class="ct">卡数：已分配 / 在算 / 空闲</div><div class="cv"><canvas id="c_cards"></canvas></div></div>'
            '</div>')
        ts_json = json.dumps(series, ensure_ascii=False)
        chart_js = ('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'
                    '<script>const TS=__TS__;'
                    'if(window.Chart&&TS.labels&&TS.labels.length){'
                    "const PAL=['#3370ff','#00b42a','#ff7d00','#7c5cff','#f53f3f'];"
                    'const base=(ds,max)=>({type:"line",data:{labels:TS.labels,datasets:ds},'
                    'options:{responsive:true,maintainAspectRatio:false,animation:false,'
                    'elements:{point:{radius:0}},plugins:{legend:{display:ds.length>1,'
                    'labels:{boxWidth:10,font:{size:11}}}},scales:{x:{ticks:{maxTicksLimit:8,font:{size:10}}},'
                    'y:{beginAtZero:true,suggestedMax:max}},spanGaps:true}});'
                    'const mk=(id,ds,max)=>{const el=document.getElementById(id);if(el)new Chart(el,base(ds,max));};'
                    'const C=TS.cards||{};'
                    'mk("c_mfu",Object.entries(TS.mfu||{}).map((x,i)=>'
                    '({label:x[0],data:x[1],borderColor:PAL[i%PAL.length],borderWidth:2,tension:.3})),null);'
                    'mk("c_util",[{label:"GPU利用率",data:TS.gpu_util,borderColor:"#3370ff",borderWidth:2,'
                    'tension:.3,fill:true,backgroundColor:"rgba(51,112,255,.08)"}],100);'
                    'mk("c_tensor",[{label:"Tensor活跃",data:TS.tensor,borderColor:"#7c5cff",borderWidth:2,tension:.3}]);'
                    'mk("c_cards",[{label:"已分配",data:C.allocated,borderColor:"#ff7d00",borderWidth:2,tension:.3},'
                    '{label:"在算",data:C.active,borderColor:"#00b42a",borderWidth:2,tension:.3},'
                    '{label:"空闲",data:C.idle,borderColor:"#86909c",borderWidth:2,borderDash:[4,3],tension:.3}]);'
                    '}</script>').replace("__TS__", ts_json)

    # MFU 计算器（真·6D 口径，纯前端；卡数默认当前在算数；输入 localStorage 持久化，抗自动刷新）
    calc_html = (
        '<h2>MFU 计算器（真·6D：6·P·tok/s ÷ 卡数·峰值）</h2>'
        '<div class="calc">'
        '<label>模型参数量 P（B）<input id="m_p" type="number" step="0.1" placeholder="如 7"></label>'
        '<label>吞吐 tokens/s<input id="m_t" type="number" placeholder="如 12000"></label>'
        f'<label>GPU 卡数<input id="m_n" type="number" value="{g.get("active_cards", 0)}"></label>'
        '<label>单卡峰值 TFLOPS<input id="m_k" type="number" value="148"></label>'
        '<button class="refresh" onclick="calcMfu()">算 MFU</button>'
        '<div id="m_out" class="mout">填 P / tokens·s 后点「算 MFU」</div>'
        '</div>'
        '<div class="sub">单卡峰值(BF16稠密)：H20 148 · H100/H200 989 · H100(FP8) 1979 · A100 312 · H800 989</div>')
    calc_js = ('<script>'
               'function calcMfu(){'
               'var G=function(id){return parseFloat(document.getElementById(id).value)},'
               'o=document.getElementById("m_out");'
               'var P=G("m_p")*1e9,T=G("m_t"),N=G("m_n"),K=G("m_k")*1e12;'
               'if(!(P&&T&&N&&K)){o.textContent="请把 P / tokens·s / 卡数 / 峰值 都填上";return;}'
               'var m=6*P*T/(N*K)*100;'
               'o.innerHTML="MFU ≈ <b>"+m.toFixed(1)+"%</b>";'
               'try{localStorage.setItem("mfu_calc",JSON.stringify(["m_p","m_t","m_n","m_k"].map(function(i){return document.getElementById(i).value})))}catch(e){}'
               '}'
               'try{var s=JSON.parse(localStorage.getItem("mfu_calc")||"null");'
               'if(s){["m_p","m_t","m_n","m_k"].forEach(function(id,i){if(s[i])document.getElementById(id).value=s[i]});calcMfu();}}catch(e){}'
               '</script>')

    return f"""<!doctype html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{refresh_secs}">
<title>GPU 卡分布</title>
<style>
:root{{color-scheme:light dark}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,'Segoe UI',Roboto,'PingFang SC',sans-serif;margin:0;padding:24px 16px;
  background:#f6f7f9;color:#1f2329;display:flex;justify-content:center;align-items:flex-start;min-height:100vh}}
.card{{background:#fff;border-radius:12px;padding:20px 24px;box-shadow:0 2px 12px rgba(0,0,0,.08);
  width:100%;max-width:960px}}
.hd{{display:flex;align-items:center;justify-content:space-between;gap:12px}}
h1{{font-size:19px;margin:0}} .sub{{color:#8a919f;font-size:12px;margin:4px 0 14px}}
h2{{font-size:15px;margin:22px 0 8px}}
button.refresh{{background:#3370ff;color:#fff;border:0;border-radius:7px;padding:8px 14px;font-size:13px;
  cursor:pointer}} button.refresh:hover{{background:#245bdb}}
table.tbl{{width:100%;border-collapse:collapse;font-size:14px}}
.tbl th,.tbl td{{padding:8px 10px;border-bottom:1px solid #eef0f3;text-align:left}}
.tbl th{{color:#8a919f;font-weight:600;font-size:12px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} td.b{{font-weight:700}} td.t{{color:#3370ff}}
.bar{{position:relative;height:22px;background:#eef0f3;border-radius:5px;overflow:hidden;min-width:180px}}
.bar .fill{{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#3370ff,#5b8cff)}}
.bar span{{position:relative;z-index:1;font-size:12px;line-height:22px;padding-left:8px;font-weight:600}}
details{{margin-top:8px}} summary{{cursor:pointer;color:#3370ff;font-size:13px;padding:6px 0}}
.kpi{{display:flex;gap:28px;flex-wrap:wrap;margin:8px 0 2px}}
.kpi div{{font-size:13px;color:#4e5969}} .kpi b{{font-size:20px;color:#1f2329}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px}}
@media(max-width:640px){{.charts{{grid-template-columns:1fr}}}}
.chart{{background:#fafbfc;border:1px solid #eef0f3;border-radius:8px;padding:10px 12px}}
.chart .ct{{font-size:13px;color:#4e5969;margin-bottom:6px;font-weight:600}}
.chart .cv{{height:180px;position:relative}}
.ranges{{display:flex;gap:6px}}
.rg{{background:#eef0f3;border:0;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer;color:#4e5969}}
.rg.on{{background:#3370ff;color:#fff}}
.calc{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;background:#fafbfc;
  border:1px solid #eef0f3;border-radius:8px;padding:12px 14px}}
.calc label{{display:flex;flex-direction:column;font-size:12px;color:#4e5969;gap:4px}}
.calc input{{width:120px;padding:6px 8px;border:1px solid #dfe1e6;border-radius:6px;font-size:14px}}
.mout{{font-size:15px;color:#1f2329;margin-left:4px}} .mout b{{color:#3370ff;font-size:18px}}
</style></head><body>
<div class="card">
<div class="hd"><h1>🧮 GPU 卡分布（实时）</h1>
<button class="refresh" onclick="{refresh_js}">🔄 立即刷新</button></div>
<div class="sub">数据时间 {e(_fmt_ts(g.get("gathered_at")))} · 每 {refresh_secs}s 自动刷新</div>
<div class="kpi"><div>总卡数<br><b>{g.get("total_cards",0)}</b></div>
<div>已分配<br><b>{g.get("used_cards",0)}</b></div>
<div>在算<br><b>{g.get("active_cards",0)}</b></div>
<div>在用人数<br><b>{n}</b></div></div>

{charts_html}
<h2>各地区 · 卡型（已用/总）</h2>
<table class="tbl"><thead><tr><th>地区</th><th>卡型</th><th>已用/总</th><th>空闲</th></tr></thead>
<tbody>{''.join(rrows) or '<tr><td colspan=4>无数据</td></tr>'}</tbody></table>

<h2>用户卡用量（在算 · 共 {n} 人）</h2>
<table class="tbl"><thead><tr><th>#</th><th>用户</th><th>卡数</th><th>卡型</th></tr></thead>
<tbody>{top_html or '<tr><td colspan=4>当前无在算任务</td></tr>'}</tbody></table>
{rest_html}
{calc_html}
<div class="sub" style="margin-top:18px">数据每 15 秒后台更新；点「🔄 立即刷新」强制重采。张量核利用率是硬件实测(≈HFU上界)；真 MFU 请用上方计算器填 P/吞吐。</div>
</div>{chart_js}{calc_js}</body></html>"""


def dist_url() -> str:
    """实时页面完整链接（带 token）；未配 GPU_DIST_BASE_URL 时返回空串。"""
    base = (settings.GPU_DIST_BASE_URL or "").rstrip("/")
    if not base:
        return ""
    token = (getattr(settings, "GPU_DIST_TOKEN", "")
             or settings.RAM_QUERY_API_TOKEN or settings.FEISHU_VERIFICATION_TOKEN or "")
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
