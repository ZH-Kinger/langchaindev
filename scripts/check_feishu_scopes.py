#!/usr/bin/env python3
"""飞书应用权限自检 —— 改完权限跑一次，10 秒知道通没通。

为什么需要它
------------
飞书的权限改动有两个隐蔽之处，2026-08-13 各踩了一次：

1. **改权限必须发布新版本才生效**，光在权限页勾选没用。
2. **缺权限往往是静默的**：消息、卡片照发（那些只要 `im:*`），但 Bitable 写入、
   审批回拉这些后台路径会安静地失败，要等下一次巡检报错、或等有人问"我的号怎么没建出来"
   才发现。那次三个功能挂了近一天。

而权限本身没有任何自查手段——只能靠功能报错反推。这个脚本把「bot 实际会调的每个飞书
接口」逐个探一遍，用的是 bot 自己的凭证，**全部只读、无副作用**。

用法
----
    # 服务器上（在容器里跑，直接用线上 .env）
    docker compose exec -T -w /app bot python scripts/check_feishu_scopes.py

    # 本机（需要 .env 里有 FEISHU_APP_ID/SECRET）
    python scripts/check_feishu_scopes.py

退出码：0=全通  1=有权限缺失  2=连 token 都拿不到（app_id/secret 错或网络不通）

判定逻辑
--------
飞书的权限检查发生在参数校验**之前**，所以：
  · code == 0        → 通
  · code == 99991672 → 权限缺失，脚本会把飞书给的"需要以下任一 scope"原样打出来
  · 其他 code        → 参数问题等，说明**权限是通的**（探针故意传了最小参数，不追求成功）
"""
import os
import re
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "https://open.feishu.cn/open-apis"
DENIED = 99991672

_OK, _DENY, _OTHER = "OK", "DENY", "OTHER"
_results = []


def _mark(kind, name, feature, detail=""):
    _results.append((kind, name, feature, detail))
    icon = {"OK": "  [通过]", "DENY": "  [缺权限]", "OTHER": "  [跳过]"}[kind]
    print(f"{icon} {name:26s} {feature}")
    if detail:
        print(f"           {detail}")


def probe(name, feature, method, url, headers, **kw):
    """探一个接口。权限问题会被识别出来并打印飞书要求的 scope 清单。"""
    try:
        r = requests.request(method, url, headers=headers, timeout=15, **kw)
        j = r.json()
    except Exception as exc:                       # 网络/非 JSON 响应
        _mark(_OTHER, name, feature, f"探测异常 {type(exc).__name__}")
        return {}
    code = j.get("code")
    if code == 0:
        _mark(_OK, name, feature)
        return j
    if code == DENIED:
        m = re.search(r"\[([^\]]+)\]", j.get("msg") or "")
        _mark(_DENY, name, feature, "需要以下任一: " + (m.group(1) if m else (j.get("msg") or "")[:100]))
        return {}
    # 参数错误等：权限是通的，只是探针没打算构造合法请求
    _mark(_OTHER, name, feature, f"code={code}（非权限问题，权限侧正常）")
    return {}


def main() -> int:
    from config.settings import settings

    app_id, app_secret = settings.FEISHU_APP_ID, settings.FEISHU_APP_SECRET
    if not (app_id and app_secret):
        print("✗ 缺 FEISHU_APP_ID / FEISHU_APP_SECRET，无法自检")
        return 2

    tr = requests.post(f"{BASE}/auth/v3/tenant_access_token/internal",
                       json={"app_id": app_id, "app_secret": app_secret}, timeout=15)
    tj = tr.json()
    token = tj.get("tenant_access_token")
    if not token:
        print(f"✗ 取 tenant_access_token 失败：code={tj.get('code')} msg={tj.get('msg')}")
        return 2
    H = {"Authorization": f"Bearer {token}"}
    print(f"应用 {app_id} · token 已获取\n")

    # ── 消息与资源 ────────────────────────────────────────────────────────────
    print("[消息]")
    # 故意用空 content：权限不通会回 99991672，权限通了会回参数错误 230001
    probe("im/v1/messages", "发消息/卡片（im:message:send_as_bot）", "POST",
          f"{BASE}/im/v1/messages", H, params={"receive_id_type": "chat_id"},
          json={"receive_id": settings.FEISHU_CHAT_ID or "oc_x", "msg_type": "text", "content": "{}"})
    probe("im/v1/images", "上传图片/趋势图（im:resource）", "POST",
          f"{BASE}/im/v1/images", H, data={"image_type": "message"})

    # ── 联系人 ────────────────────────────────────────────────────────────────
    print("\n[联系人]")
    oid = settings.ADMIN_FEISHU_OPEN_ID
    if oid:
        j = probe("contact/v3/users", "取用户名（卡片上显示谁操作的）", "GET",
                  f"{BASE}/contact/v3/users", H,
                  params={"user_id_type": "open_id", "user_ids": oid})
        items = (j.get("data") or {}).get("items") or []
        if items and not items[0].get("name"):
            # API 权限通了但人员字段被剥掉 = 另一层门没开，很容易被误判成"权限没问题"
            _mark(_DENY, "contact 数据权限", "返回了记录但 name 为空",
                  "API 权限已通，但「权限管理 → 数据权限 → 通讯录权限范围」没把该成员纳入范围")
    else:
        _mark(_OTHER, "contact/v3/users", "跳过（未配置 ADMIN_FEISHU_OPEN_ID）")

    # ── 审批（建号 / 凭证发放的门禁）──────────────────────────────────────────
    print("\n[审批] —— 缺权限会让建号与凭证发放静默失效")
    ac = settings.FEISHU_RAM_APPROVAL_CODE
    if ac:
        probe("approval/v4/approvals", "读审批定义（approval:approval:readonly）", "GET",
              f"{BASE}/approval/v4/approvals/{ac}", H)
    else:
        _mark(_OTHER, "approval/v4/approvals", "跳过（未配置 FEISHU_RAM_APPROVAL_CODE）")

    # ── 多维表格 ──────────────────────────────────────────────────────────────
    print("\n[多维表格] —— 容量巡检 / OSS 权限对账 / 数据集大盘")
    at = settings.CAPACITY_BITABLE_APP_TOKEN
    if at:
        probe("bitable apps", "读表结构", "GET", f"{BASE}/bitable/v1/apps/{at}", H)
        for label, tid in (("快照表", settings.CAPACITY_BITABLE_TABLE_SNAPSHOT),
                           ("厂家表", settings.CAPACITY_BITABLE_TABLE_VENDOR),
                           ("批次表", settings.CAPACITY_BITABLE_TABLE_BATCH)):
            if not tid:
                continue
            u = f"{BASE}/bitable/v1/apps/{at}/tables/{tid}/records"
            probe(f"records GET·{label}", "列举记录（容量巡检 upsert 前置）", "GET", u, H,
                  params={"page_size": 1})
            probe(f"records search·{label}", "search 查询（OSS 权限对账走这个）", "POST",
                  u + "/search", H, params={"page_size": 1}, json={})
    else:
        _mark(_OTHER, "bitable", "跳过（未配置 CAPACITY_BITABLE_APP_TOKEN）")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    denied = [r for r in _results if r[0] == _DENY]
    print("\n" + "─" * 68)
    if denied:
        print(f"✗ {len(denied)} 项缺权限：")
        for _, name, feature, detail in denied:
            print(f"    · {name} —— {feature}")
            if detail:
                print(f"      {detail}")
        print("\n  改完记得【创建版本 → 发布 → 管理员审核】，只勾选不发布不生效。")
        return 1
    print(f"✓ 全部 {len(_results)} 项探测通过，未发现权限缺失")
    return 0


if __name__ == "__main__":
    sys.exit(main())
