"""
把容量巡检结果写入飞书多维表格（Bitable），三表关联：
  巡检快照表 → 厂家总量表（关联快照）→ 批次明细表（关联厂家总量）

写入分工（系统自动算的不写）：
  快照：巡检时间 / 备注 / 可读标识
  厂家总量：云厂商 / Bucket / 父目录 / 厂家 / 大小GB / 对象数 / 较上次GB / 关联巡检快照
  批次明细：批次 / 大小GB / 对象数 / 关联厂家总量

凭证用 tenant_access_token；需应用开 bitable 权限且机器人是该多维表格协作者。
"""
import time
import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = "https://open.feishu.cn/open-apis"
_GB = 1024 ** 3
_BATCH_LIMIT = 500   # batch_create 单次记录上限（API 上限 1000，留余量）


def _gb(num_bytes: float) -> float:
    return round(num_bytes / _GB, 2)


def _tenant_token() -> str:
    r = requests.post(
        f"{_BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": settings.FEISHU_APP_ID, "app_secret": settings.FEISHU_APP_SECRET},
        timeout=10,
    )
    return r.json().get("tenant_access_token", "")


def _records_url(table_id: str, batch: bool = False) -> str:
    suffix = "/batch_create" if batch else ""
    return f"{_BASE}/bitable/v1/apps/{settings.CAPACITY_BITABLE_APP_TOKEN}/tables/{table_id}/records{suffix}"


def _is_configured() -> bool:
    return all([
        settings.CAPACITY_BITABLE_APP_TOKEN,
        settings.CAPACITY_BITABLE_TABLE_SNAPSHOT,
        settings.CAPACITY_BITABLE_TABLE_VENDOR,
        settings.CAPACITY_BITABLE_TABLE_BATCH,
    ])


def _create_one(headers: dict, table_id: str, fields: dict) -> str:
    """创建单条记录，返回 record_id（失败返回空串）。"""
    r = requests.post(_records_url(table_id), headers=headers,
                      json={"fields": fields}, timeout=15)
    j = r.json()
    if j.get("code") != 0:
        logger.error("[Bitable] 创建记录失败 table=%s code=%s msg=%s",
                     table_id, j.get("code"), j.get("msg"))
        return ""
    return j.get("data", {}).get("record", {}).get("record_id", "")


def _batch_create(headers: dict, table_id: str, records: list) -> list:
    """批量创建，返回 record_id 列表（与入参顺序一致）。分块发送。"""
    ids = []
    for i in range(0, len(records), _BATCH_LIMIT):
        chunk = records[i:i + _BATCH_LIMIT]
        r = requests.post(_records_url(table_id, batch=True), headers=headers,
                          json={"records": [{"fields": f} for f in chunk]}, timeout=30)
        j = r.json()
        if j.get("code") != 0:
            logger.error("[Bitable] 批量创建失败 table=%s code=%s msg=%s",
                         table_id, j.get("code"), j.get("msg"))
            ids.extend([""] * len(chunk))
            continue
        ids.extend(rec.get("record_id", "") for rec in j.get("data", {}).get("records", []))
    return ids


def write_scan(vendor_rows: list, readable_id: str, remark: str) -> bool:
    """把一次巡检写入三张表。

    vendor_rows: 每个「厂家」一项：
      {云厂商, Bucket, 父目录, 厂家, total_bytes, total_count,
       delta_bytes(None=首次), batches:[(批次,bytes,count)]}
    返回是否写入成功（至少快照+厂家成功）。
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

    # ① 巡检快照（父）
    snap_id = _create_one(headers, settings.CAPACITY_BITABLE_TABLE_SNAPSHOT, {
        "巡检时间": int(time.time() * 1000),
        "备注": remark,
        "可读标识": readable_id,
    })
    if not snap_id:
        return False

    # ② 厂家总量（关联快照），保持顺序以便回填批次的父 record_id
    vendor_fields = []
    for row in vendor_rows:
        f = {
            "云厂商": row["云厂商"],
            "Bucket": row["Bucket"],
            "父目录": row["父目录"],
            "厂家": row["厂家"],
            "大小GB": _gb(row["total_bytes"]),
            "对象数": row["total_count"],
            "关联巡检快照": [snap_id],
        }
        if row.get("delta_bytes") is not None:   # 首次不写，留空
            f["较上次GB"] = _gb(row["delta_bytes"])
        vendor_fields.append(f)
    vendor_ids = _batch_create(headers, settings.CAPACITY_BITABLE_TABLE_VENDOR, vendor_fields)

    # ③ 批次明细（关联对应厂家总量行）
    batch_fields = []
    for row, vid in zip(vendor_rows, vendor_ids):
        if not vid:
            continue
        for batch_name, size_bytes, count in row["batches"]:
            batch_fields.append({
                "批次": batch_name,
                "大小GB": _gb(size_bytes),
                "对象数": count,
                "关联厂家总量": [vid],
            })
    n_batch = 0
    if batch_fields:
        n_batch = sum(1 for x in _batch_create(headers, settings.CAPACITY_BITABLE_TABLE_BATCH, batch_fields) if x)

    n_vendor = sum(1 for x in vendor_ids if x)
    logger.info("[Bitable] 写入完成：快照1 + 厂家%d + 批次%d", n_vendor, n_batch)
    return True
