"""
把容量巡检结果 upsert 进飞书多维表格（Bitable），三表关联：
  巡检快照表（单行）→ 厂家总量表（关联快照）→ 批次明细表（关联厂家总量）

upsert：每厂家/批次只留一行，每次巡检更新而非追加；本次未出现的旧行删除；
历史遗留的重复行（旧版每次追加产生）在首次 upsert 时一并去重。

写入字段（系统自动算的不写）：
  快照：巡检时间 / 备注 / 可读标识
  厂家总量：云厂商 / Bucket / 父目录 / 厂家 / 大小GB / 对象数 / 较上次GB / 数据结构 / 数据时长 / 关联巡检快照
  批次明细：批次 / 大小GB / 对象数 / 数据结构 / 数据时长 / 关联厂家总量

凭证用 tenant_access_token；需应用开 bitable 权限且机器人是该多维表格协作者。
"""
import re
import time
import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = "https://open.feishu.cn/open-apis"
_GB = 1024 ** 3
_HOURS_RE = re.compile(r"(\d+)\s*h", re.IGNORECASE)


def _gb(num_bytes: float) -> float:
    return round(num_bytes / _GB, 2)


def _parse_hours(name: str):
    """从目录名解析「数字+h」时长（如 504h/100h），无则返回 None。"""
    m = _HOURS_RE.search(name or "")
    return int(m.group(1)) if m else None


def _tenant_token() -> str:
    r = requests.post(
        f"{_BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": settings.FEISHU_APP_ID, "app_secret": settings.FEISHU_APP_SECRET},
        timeout=10,
    )
    return r.json().get("tenant_access_token", "")


def _records_url(table_id: str) -> str:
    return f"{_BASE}/bitable/v1/apps/{settings.CAPACITY_BITABLE_APP_TOKEN}/tables/{table_id}/records"


def _is_configured() -> bool:
    return all([
        settings.CAPACITY_BITABLE_APP_TOKEN,
        settings.CAPACITY_BITABLE_TABLE_SNAPSHOT,
        settings.CAPACITY_BITABLE_TABLE_VENDOR,
        settings.CAPACITY_BITABLE_TABLE_BATCH,
    ])


def _as_text(v) -> str:
    """把字段值（文本 / 查找引用列表 / dict）统一取成纯文本，用于建唯一键。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            return v[0].get("text", "")
        return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        return v.get("text", "")
    return str(v)


# ── 通用记录操作 ──────────────────────────────────────────────────────────────

def _list_records(headers: dict, table_id: str) -> list:
    """列出表内全部记录（分页）。返回 [{record_id, fields}, ...]。"""
    out, pt = [], ""
    while True:
        params = {"page_size": 500}
        if pt:
            params["page_token"] = pt
        r = requests.get(_records_url(table_id), headers=headers, params=params, timeout=20)
        d = r.json().get("data", {})
        out.extend(d.get("items", []))
        pt = d.get("page_token", "")
        if not d.get("has_more"):
            break
    return out


def _create_one(headers: dict, table_id: str, fields: dict) -> str:
    r = requests.post(_records_url(table_id), headers=headers, json={"fields": fields}, timeout=15)
    j = r.json()
    if j.get("code") != 0:
        logger.error("[Bitable] 创建失败 table=%s code=%s msg=%s", table_id, j.get("code"), j.get("msg"))
        return ""
    return j.get("data", {}).get("record", {}).get("record_id", "")


def _update_one(headers: dict, table_id: str, record_id: str, fields: dict) -> bool:
    r = requests.put(f"{_records_url(table_id)}/{record_id}", headers=headers,
                     json={"fields": fields}, timeout=15)
    j = r.json()
    if j.get("code") != 0:
        logger.error("[Bitable] 更新失败 table=%s code=%s msg=%s", table_id, j.get("code"), j.get("msg"))
        return False
    return True


def _delete_one(headers: dict, table_id: str, record_id: str) -> None:
    try:
        requests.delete(f"{_records_url(table_id)}/{record_id}", headers=headers, timeout=15)
    except Exception:
        logger.warning("[Bitable] 删除失败 table=%s rid=%s", table_id, record_id, exc_info=True)


def _index(records: list, keyfn) -> dict:
    """按 keyfn 把记录归并成 {key: [record_id, ...]}（同键多行=历史重复，待去重）。"""
    idx = {}
    for rec in records:
        idx.setdefault(keyfn(rec.get("fields", {})), []).append(rec["record_id"])
    return idx


def _upsert_snapshot(headers: dict, readable_id: str, remark: str) -> str:
    """快照表保留单行：有则更新第一行并删除多余，无则新建。返回 record_id。"""
    snap = settings.CAPACITY_BITABLE_TABLE_SNAPSHOT
    fields = {"巡检时间": int(time.time() * 1000), "备注": remark, "可读标识": readable_id}
    existing = _list_records(headers, snap)
    if existing:
        sid = existing[0]["record_id"]
        _update_one(headers, snap, sid, fields)
        for rec in existing[1:]:
            _delete_one(headers, snap, rec["record_id"])
        return sid
    return _create_one(headers, snap, fields)


# ── 主入口：upsert 写入 ────────────────────────────────────────────────────────

def write_scan(vendor_rows: list, readable_id: str, remark: str) -> bool:
    """把一次巡检 upsert 进三张表。

    vendor_rows: 每个「厂家」一项：
      {云厂商, Bucket, 父目录, 厂家, total_bytes, total_count, struct,
       delta_bytes(None=首次), batches:[(批次,bytes,count,struct)]}
    """
    if not _is_configured():
        logger.warning("[Bitable] 多维表格未配置（app_token/table_id 缺失），跳过写入")
        return False
    if not vendor_rows:
        return False

    token = _tenant_token()
    if not token:
        logger.error("[Bitable] 获取 tenant_access_token 失败，跳过写入")
        return False
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    snap_id = _upsert_snapshot(headers, readable_id, remark)
    if not snap_id:
        return False

    VEN = settings.CAPACITY_BITABLE_TABLE_VENDOR
    BAT = settings.CAPACITY_BITABLE_TABLE_BATCH

    # 现有行索引（同键多行=历史重复）
    ven_idx = _index(_list_records(headers, VEN),
                     lambda f: (_as_text(f.get("云厂商")), _as_text(f.get("Bucket")), _as_text(f.get("厂家"))))
    bat_idx = _index(_list_records(headers, BAT),
                     lambda f: (_as_text(f.get("云厂商")), _as_text(f.get("厂家")), _as_text(f.get("批次"))))

    seen_ven, seen_bat = set(), set()
    n_ven = n_bat = 0

    for row in vendor_rows:
        # 厂家级时长 = 各批次解析时长之和（全无则空）
        hrs = [_parse_hours(b[0]) for b in row["batches"]]
        fam_hours = sum(h for h in hrs if h is not None) if any(h is not None for h in hrs) else None

        vkey = (row["云厂商"], row["Bucket"], row["厂家"])
        vfields = {
            "云厂商": row["云厂商"], "Bucket": row["Bucket"], "父目录": row["父目录"], "厂家": row["厂家"],
            "大小GB": _gb(row["total_bytes"]), "对象数": row["total_count"],
            "数据结构": row.get("struct", ""), "关联巡检快照": [snap_id],
        }
        if fam_hours is not None:
            vfields["数据时长"] = fam_hours
        if row.get("delta_bytes") is not None:
            vfields["较上次GB"] = _gb(row["delta_bytes"])

        if vkey in ven_idx:
            rids = ven_idx[vkey]
            vid = rids[0]
            _update_one(headers, VEN, vid, vfields)
            for extra in rids[1:]:               # 去掉历史重复行
                _delete_one(headers, VEN, extra)
        else:
            vid = _create_one(headers, VEN, vfields)
        if not vid:
            continue
        seen_ven.add(vkey)
        n_ven += 1

        for batch_name, bsize, bcount, bstruct in row["batches"]:
            bkey = (row["云厂商"], row["厂家"], batch_name)
            bfields = {
                "批次": batch_name, "大小GB": _gb(bsize), "对象数": bcount,
                "数据结构": bstruct, "关联厂家总量": [vid],
            }
            bh = _parse_hours(batch_name)
            if bh is not None:
                bfields["数据时长"] = bh
            if bkey in bat_idx:
                rids = bat_idx[bkey]
                _update_one(headers, BAT, rids[0], bfields)
                for extra in rids[1:]:
                    _delete_one(headers, BAT, extra)
            else:
                _create_one(headers, BAT, bfields)
            seen_bat.add(bkey)
            n_bat += 1

    # 删除本次未出现的旧行（含历史重复整组）
    for k, rids in ven_idx.items():
        if k not in seen_ven:
            for rid in rids:
                _delete_one(headers, VEN, rid)
    for k, rids in bat_idx.items():
        if k not in seen_bat:
            for rid in rids:
                _delete_one(headers, BAT, rid)

    logger.info("[Bitable] upsert 完成：快照1 + 厂家%d + 批次%d", n_ven, n_bat)
    return True
