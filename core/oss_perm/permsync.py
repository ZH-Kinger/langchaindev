#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从飞书「舞肌算法组权限统计」多维表格读取每位成员的 OSS 读/写目录，
据此为每个 RAM 用户生成一份最小权限的自定义 RAM Policy 并附加到该用户。

默认 dry-run（只打印计划，不改动阿里云）。确认无误后加 --apply 真正执行。

环境变量：
  FEISHU_APP_ID, FEISHU_APP_SECRET                飞书自建应用凭证（应用身份）
  ALIBABA_CLOUD_ACCESS_KEY_ID / _SECRET           阿里云访问密钥（需 RAM 写权限）
可选覆盖：
  FEISHU_APP_TOKEN, MEMBER_TABLE_ID, BUCKET_TABLE_ID
"""
import argparse
import json
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# 常量 / 默认配置
# ---------------------------------------------------------------------------
FEISHU_BASE = "https://open.feishu.cn/open-apis"

DEFAULT_APP_TOKEN = "ZRzzb8RsSarWaSsJ24gcp879n0b"   # 多维表格 app_token
DEFAULT_MEMBER_TABLE = "tblr7bjmv7rLFKpO"            # 成员表
DEFAULT_BUCKET_TABLE = "tbl1FRrhCeX4KQi3"            # OSS_Bucket 对照表

ALL = "ALL/"   # 表格里表示「全部桶 / 整桶」的特殊值

# 表格里的桶展示名 -> (OSS region id, 真实 bucket 名)
# region 仅用于报告展示；策略 ARN 用 region 通配符，跨区域同样生效。
BUCKET_MAP = {
    "新加坡-wuji-sing":          ("oss-ap-southeast-1", "wuji-sing"),
    "北京-wuji-test-data":       ("oss-cn-beijing",     "wuji-test-data"),
    "杭州-wuji-bucket-hangzhou": ("oss-cn-hangzhou",    "wuji-bucket-hangzhou"),
    "杭州-wuji-ego-processed":   ("oss-cn-hangzhou",    "wuji-ego-processed"),
}

# 字段名
F_NAME = "成员姓名"
F_ACCOUNT = "账号"
F_STATUS = "账号状态"
F_BUCKET = "OSS_Bucket"
F_READ = "子目录(读)"
F_WRITE = "子目录(写)"
STATUS_ACTIVE = "在职"

# 策略动作集合
READ_OBJECT_ACTIONS = ["oss:GetObject", "oss:GetObjectAcl"]
WRITE_OBJECT_ACTIONS = [
    "oss:PutObject", "oss:GetObject", "oss:DeleteObject",
    "oss:AbortMultipartUpload", "oss:ListParts",
    "oss:GetObjectAcl", "oss:PutObjectAcl",
]
LIST_BUCKET_ACTIONS = [
    "oss:ListObjects", "oss:GetBucketInfo",
    "oss:GetBucketStat", "oss:GetBucketLocation",
]

POLICY_PREFIX = "wuji-oss-auto-"   # 自定义策略命名前缀
RAM_DOC_LIMIT = 6144               # 自定义策略文档字符上限


# ---------------------------------------------------------------------------
# 飞书读取
# ---------------------------------------------------------------------------
def feishu_token(app_id, app_secret):
    r = requests.post(
        f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取 tenant_access_token 失败: {data}")
    return data["tenant_access_token"]


def feishu_search_all(token, app_token, table_id):
    """分页拉取一张表的全部记录。"""
    url = f"{FEISHU_BASE}/bitable/v1/apps/{app_token}/tables/{table_id}/records/search"
    headers = {"Authorization": f"Bearer {token}"}
    items, page_token = [], None
    while True:
        params = {"page_size": 500}
        if page_token:
            params["page_token"] = page_token
        r = requests.post(url, headers=headers, params=params, json={}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"读取表 {table_id} 失败: {data}")
        d = data["data"]
        items.extend(d.get("items", []))
        if not d.get("has_more"):
            break
        page_token = d.get("page_token")
    return items


def cell_text(v):
    """文本/链接型字段 -> 纯文本。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in v)
    if isinstance(v, dict):
        return v.get("text", "")
    return str(v)


def cell_list(v):
    """多选型字段 -> 字符串列表。"""
    if v is None:
        return []
    if isinstance(v, list):
        return [x if isinstance(x, str) else cell_text(x) for x in v]
    return [v]


# ---------------------------------------------------------------------------
# 解析与策略生成
# ---------------------------------------------------------------------------
def build_valid_combos(bucket_rows):
    """对照表 -> {桶展示名: set(合法子目录)}。"""
    combos = {}
    for row in bucket_rows:
        f = row.get("fields", {})
        bucket = f.get(F_BUCKET)
        sub = f.get("Sub_Bucket")
        if not bucket or not sub:
            continue
        combos.setdefault(bucket, set()).add(sub)
    return combos


def parse_member(row):
    f = row.get("fields", {})
    account = cell_text(f.get(F_ACCOUNT)).strip().replace("mailto:", "")
    return {
        "name": cell_text(f.get(F_NAME)).strip(),
        "account": account,
        "username": account.split("@")[0] if account else "",
        "status": f.get(F_STATUS),
        "buckets": cell_list(f.get(F_BUCKET)),
        "read": cell_list(f.get(F_READ)),
        "write": cell_list(f.get(F_WRITE)),
    }


def _expand(subdirs, valid, disp, member, kind, warnings):
    """把子目录列表解析为前缀集合；'' 代表整桶（ALL/）。"""
    out = set()
    for s in subdirs:
        if s == ALL:
            out.add("")
            continue
        if valid is not None and s not in valid:
            warnings.append(f"  · {member['name']}：桶「{disp}」下不存在子目录「{s}」({kind})，已跳过")
            continue
        out.add(s)
    return {""} if "" in out else out


def resolve_member(member, valid_combos, warnings):
    """返回 {真实bucket名: {'read': set(prefix), 'write': set(prefix), 'display': str}}。"""
    out = {}
    displays = [ALL] if ALL in member["buckets"] else member["buckets"]
    for disp in displays:
        if disp == ALL:
            real, valid, show = "*", None, "ALL(全部桶)"
        elif disp in BUCKET_MAP:
            real, valid, show = BUCKET_MAP[disp][1], valid_combos.get(disp, set()), disp
        else:
            warnings.append(f"  · {member['name']}：未知 Bucket「{disp}」，已跳过")
            continue
        rset = _expand(member["read"], valid, disp, member, "读", warnings)
        wset = _expand(member["write"], valid, disp, member, "写", warnings)
        # 兼容：填了桶但读/写两列都空 → 默认给该桶整桶读+写
        if not member["read"] and not member["write"]:
            rset, wset = {""}, {""}
        if not rset and not wset:
            continue
        slot = out.setdefault(real, {"read": set(), "write": set(), "display": show})
        slot["read"] |= rset
        slot["write"] |= wset
    return out


def _obj_arn(base, prefix):
    return f"{base}/{prefix}*" if prefix else f"{base}/*"


def build_policy(resolved):
    """resolved -> RAM policy 文档(dict)。"""
    stmts = []
    for bucket in sorted(resolved):
        pr = resolved[bucket]
        base = f"acs:oss:*:*:{bucket}"
        reads, writes = pr["read"], pr["write"]

        if reads:
            stmts.append({
                "Effect": "Allow",
                "Action": READ_OBJECT_ACTIONS,
                "Resource": [_obj_arn(base, p) for p in sorted(reads)],
            })
        if writes:
            stmts.append({
                "Effect": "Allow",
                "Action": WRITE_OBJECT_ACTIONS,
                "Resource": [_obj_arn(base, p) for p in sorted(writes)],
            })

        # 桶级 List：限定到读写涉及的前缀，便于在该前缀内浏览
        list_stmt = {"Effect": "Allow", "Action": LIST_BUCKET_ACTIONS, "Resource": [base]}
        all_pref = reads | writes
        if all_pref and "" not in all_pref:
            conds = []
            for p in sorted(all_pref):
                conds.extend([p, p + "*"])
            list_stmt["Condition"] = {"StringLike": {"oss:Prefix": conds}}
        stmts.append(list_stmt)

    return {"Version": "1", "Statement": stmts}


def coerce_level(resolved, level):
    """按粒度档收紧 resolved。

    level='bucket' → 把每个桶的读/写前缀塌缩成整桶（''），即「限制到桶」；
    level='dir'    → 原样返回（按子目录前缀），即「限制到目录」。
    保留读/写区分与桶集合，只动前缀粒度。
    """
    if level != "bucket":
        return resolved
    out = {}
    for bucket, pr in resolved.items():
        out[bucket] = {
            "read":  {""} if pr["read"] else set(),
            "write": {""} if pr["write"] else set(),
            "display": pr["display"],
        }
    return out


# ---------------------------------------------------------------------------
# 对账：策略文档反解析 + 期望/实际差集
# ---------------------------------------------------------------------------
def _parse_obj_arn(arn):
    """build_policy._obj_arn 的逆：acs:oss:*:*:<桶>/<前缀>* -> (桶, 前缀)。

    `<桶>/*` 或桶级 ARN（无对象段）-> 前缀 ''；无法解析返回 (None, None)。
    """
    marker = "acs:oss:*:*:"
    if not isinstance(arn, str) or not arn.startswith(marker):
        return None, None
    rest = arn[len(marker):]
    if "/" not in rest:
        return rest, ""                      # 桶级 ARN，视为整桶
    bucket, _, obj = rest.partition("/")
    if obj in ("*", ""):
        return bucket, ""
    return bucket, (obj[:-1] if obj.endswith("*") else obj)


def parse_policy_doc(doc):
    """build_policy 的逆：RAM 策略文档 -> {桶: {"read":set(prefix), "write":set(prefix)}}。

    按 Action 判类型：含写动作 -> 写；否则含 GetObject -> 读；纯 List 等桶级语句忽略。
    """
    out = {}
    for stmt in (doc.get("Statement") or []):
        actions = stmt.get("Action") or []
        if isinstance(actions, str):
            actions = [actions]
        aset = set(actions)
        if "oss:PutObject" in aset or "oss:DeleteObject" in aset:
            kind = "write"
        elif "oss:GetObject" in aset:
            kind = "read"
        else:
            continue                         # ListObjects 等，不参与前缀对比
        resources = stmt.get("Resource") or []
        if isinstance(resources, str):
            resources = [resources]
        for arn in resources:
            bucket, prefix = _parse_obj_arn(arn)
            if bucket is None:
                continue
            out.setdefault(bucket, {"read": set(), "write": set()})[kind].add(prefix)
    return out


def diff_resolved(expected, actual):
    """期望 vs 实际 -> {桶: {"over":{read,write}, "under":{read,write}}}，仅保留有差异的桶。

    over = 实际有、期望无（多授，该回收）；under = 期望有、实际无（少授，该补）。
    """
    out = {}
    for bucket in set(expected) | set(actual):
        e = expected.get(bucket) or {}
        a = actual.get(bucket) or {}
        er, ew = e.get("read", set()), e.get("write", set())
        ar, aw = a.get("read", set()), a.get("write", set())
        over = {"read": ar - er, "write": aw - ew}
        under = {"read": er - ar, "write": ew - aw}
        if any(over.values()) or any(under.values()):
            out[bucket] = {"over": over, "under": under}
    return out


# ---------------------------------------------------------------------------
# 阿里云 RAM 落地
# ---------------------------------------------------------------------------
def make_ram_client():
    from alibabacloud_ram20150501.client import Client as RamClient
    from alibabacloud_tea_openapi import models as open_api_models

    ak = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    sk = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    if not ak or not sk:   # bot 内运行时复用 langchaindev 的 RAM 可写 AK
        try:
            from config.settings import settings
            ak = ak or settings.ALIYUN_ACCESS_KEY_ID
            sk = sk or settings.ALIYUN_ACCESS_KEY_SECRET
        except Exception:
            pass
    if not ak or not sk:
        raise RuntimeError("缺少 RAM AccessKey（ALIBABA_CLOUD_* 或 settings.ALIYUN_ACCESS_KEY_*）")
    cfg = open_api_models.Config(access_key_id=ak, access_key_secret=sk)
    cfg.endpoint = "ram.aliyuncs.com"
    return RamClient(cfg)


def _err_code(e):
    code = getattr(e, "code", None)
    if code:
        return code
    data = getattr(e, "data", None)
    if isinstance(data, dict):
        return data.get("Code")
    return None


def ram_username_map(client):
    """ListUsers 全量 -> (by_display={显示名: 用户名}, names=set(用户名))。

    RAM 用户名规则不统一（邮箱前缀 / 名+姓拼音 CamelCase 等），但显示名统一是中文姓名，
    故以显示名作主键匹配成员。显示名重复时取首个。
    """
    from alibabacloud_ram20150501 import models as m
    by_display, names = {}, set()
    marker, truncated = None, True
    while truncated:
        req = m.ListUsersRequest(max_items=1000)
        if marker:
            req.marker = marker
        body = client.list_users(req).body
        for u in (body.users.user if getattr(body, "users", None) else []):
            names.add(u.user_name)
            dn = getattr(u, "display_name", "") or ""
            if dn:
                by_display.setdefault(dn, u.user_name)
        truncated = bool(getattr(body, "is_truncated", False))
        marker = getattr(body, "marker", None)
    return by_display, names


def resolve_ram_username(member, by_display, names):
    """成员 -> 真实 RAM 用户名：优先「显示名==姓名」，其次邮箱前缀；都没有返回 None。"""
    if member["name"] and member["name"] in by_display:
        return by_display[member["name"]]
    if member["username"] in names:
        return member["username"]
    return None


# ── RAM 用户名映射表（邮箱前缀 -> RAM 用户名；"" 表示无 RAM 用户）────────────────
DEFAULT_MAP_FILE = "ram_user_map.json"


def load_user_map(path):
    """读取映射表 {邮箱前缀: RAM用户名}；不存在返回 {}。"""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_user_map(members, by_display, names, existing):
    """合并映射表：已有项（含手工修正）一律保留；新成员按「显示名==姓名→邮箱前缀」自动解析，
    无 RAM 用户置 ""。返回 (映射, 新增列表, 无RAM用户姓名列表)。"""
    out = dict(existing)
    added, unmatched = [], []
    for mb in members:
        pref = mb["username"]
        if not pref or pref in out:
            continue
        ram = by_display.get(mb["name"]) or (pref if pref in names else "")
        out[pref] = ram
        (added if ram else unmatched).append(mb["name"] if not ram else (pref, ram))
    return out, added, unmatched


def map_username(member, user_map, by_display, names):
    """优先用映射表；表里没有该前缀时回退到实时显示名解析。"" -> None（无 RAM 用户）。"""
    pref = member["username"]
    if pref in user_map:
        return user_map[pref] or None
    return resolve_ram_username(member, by_display, names)


def apply_user(client, username, policy_name, policy_doc, create_users, log, attach=True):
    """幂等地创建/更新策略并（可选）附加到用户。返��� True 表示成功。

    attach=False 时只建/更新策略、不附加（用于成员暂无 RAM 用户的情形）。
    """
    from alibabacloud_ram20150501 import models as m

    doc = json.dumps(policy_doc, ensure_ascii=False)

    # 1) 策略：不存在则创建，存在则新增版本并设为默认（rotate 自动清理旧版本）
    try:
        client.get_policy(m.GetPolicyRequest(policy_type="Custom", policy_name=policy_name))
        exists = True
    except Exception as e:
        if _err_code(e) in ("EntityNotExist.Policy", "EntityNotExist.CustomPolicy"):
            exists = False
        else:
            log(f"    ✗ 查询策略失败: {_err_code(e)} {e}")
            return False

    try:
        if not exists:
            client.create_policy(m.CreatePolicyRequest(
                policy_name=policy_name, policy_document=doc,
                description="auto-generated from feishu bitable"))
            log(f"    ✓ 已创建策略 {policy_name}")
        else:
            client.create_policy_version(m.CreatePolicyVersionRequest(
                policy_name=policy_name, policy_document=doc, set_as_default=True,
                rotate_strategy="DeleteOldestNonDefaultVersionWhenLimitExceeded"))
            log(f"    ✓ 已更新策略 {policy_name}（新默认版本）")
    except Exception as e:
        log(f"    ✗ 写策略失败: {_err_code(e)} {e}")
        return False

    # attach=False：成员暂无 RAM 用户，只建策略不附加
    if not attach:
        log(f"    = 策略已就绪，未附加（{username} 无 RAM 用户）")
        return True

    # 2) 用户：可选创建
    if create_users:
        try:
            client.get_user(m.GetUserRequest(user_name=username))
        except Exception as e:
            if _err_code(e) == "EntityNotExist.User":
                try:
                    client.create_user(m.CreateUserRequest(user_name=username, display_name=username))
                    log(f"    ✓ 已创建用户 {username}")
                except Exception as e2:
                    log(f"    ✗ 创建用户失败: {_err_code(e2)} {e2}")
                    return False

    # 3) 附加策略（幂等）
    try:
        client.attach_policy_to_user(m.AttachPolicyToUserRequest(
            policy_type="Custom", policy_name=policy_name, user_name=username))
        log(f"    ✓ 已附加到用户 {username}")
    except Exception as e:
        code = _err_code(e)
        if code == "EntityAlreadyExists.User.Policy":
            log(f"    = 策略已附加到 {username}")
        elif code == "EntityNotExist.User":
            log(f"    ✗ 用户 {username} 不存在（加 --create-users 自动创建）")
            return False
        else:
            log(f"    ✗ 附加失败: {code} {e}")
            return False
    return True


# ---------------------------------------------------------------------------
# 对账：拉取 RAM 现状（只读）
# ---------------------------------------------------------------------------
def ram_actual(client, username):
    """取 wuji-oss-auto-<username> 的默认版本文档并反解析；策略不存在返回 None。"""
    from alibabacloud_ram20150501 import models as m
    name = POLICY_PREFIX + username
    try:
        resp = client.get_policy(m.GetPolicyRequest(policy_type="Custom", policy_name=name))
    except Exception as e:
        if _err_code(e) in ("EntityNotExist.Policy", "EntityNotExist.CustomPolicy"):
            return None
        raise
    doc = resp.body.default_policy_version.policy_document
    return parse_policy_doc(json.loads(doc))


def list_auto_policies(client):
    """列出所有 wuji-oss-auto-* 自定义策略 -> {username: [attached_user, ...]}。"""
    from alibabacloud_ram20150501 import models as m
    names, marker, truncated = [], None, True
    while truncated:
        req = m.ListPoliciesRequest(policy_type="Custom", max_items=1000)
        if marker:
            req.marker = marker
        body = client.list_policies(req).body
        plist = body.policies.policy if getattr(body, "policies", None) else []
        names.extend(p.policy_name for p in plist if p.policy_name.startswith(POLICY_PREFIX))
        truncated = bool(getattr(body, "is_truncated", False))
        marker = getattr(body, "marker", None)

    out = {}
    for name in names:
        username = name[len(POLICY_PREFIX):]
        users = []
        try:
            ub = client.list_entities_for_policy(
                m.ListEntitiesForPolicyRequest(policy_type="Custom", policy_name=name)).body
            if getattr(ub, "users", None) and ub.users.user:
                users = [u.user_name for u in ub.users.user]
        except Exception:
            pass
        out[username] = users
    return out


def _pp(prefixes):
    """前缀集合 -> 可读串；空集合 -> '—'，空串前缀 -> '<整桶>'。"""
    return ", ".join(sorted(x or "<整桶>" for x in prefixes)) if prefixes else "—"


def _scope_str(resolved):
    """{桶:{read,write}} -> '桶A 读[..] 写[..]；桶B ...'，供结果卡列出每人实际下发范围。"""
    parts = [f"{b} 读[{_pp(resolved[b]['read'])}] 写[{_pp(resolved[b]['write'])}]"
             for b in sorted(resolved)]
    return "；".join(parts) if parts else "—"


def run_audit(client, plan, active_usernames, out_file, log=print):
    """只读对账：逐人 多授/少授/一致 + 孤儿策略。返回 (有差异人数, 孤儿dict)。"""
    log("\n" + "=" * 70)
    log("权限对账（只读：RAM 实际 vs 飞书表格期望，不改动 RAM）")
    log("=" * 70)

    diff_users, audit_rows = 0, []
    for p in plan:
        mb, username, expected = p["member"], p["member"]["username"], p["resolved"]
        actual = ram_actual(client, username)
        if actual is None:
            log(f"\n● {mb['name']} ({username})  ⚠ RAM 无该策略/未附加 → 全部少授（待同步）")
            diff_users += 1
            audit_rows.append({"username": username, "status": "missing"})
            continue
        d = diff_resolved(expected, actual)
        if not d:
            log(f"\n● {mb['name']} ({username})  ✓ 一致")
            continue
        diff_users += 1
        log(f"\n● {mb['name']} ({username})")
        for bucket in sorted(d):
            over, under = d[bucket]["over"], d[bucket]["under"]
            log(f"    桶 {bucket}")
            if any(over.values()):
                log(f"      多授(RAM有表格无→该回收): 读[{_pp(over['read'])}] 写[{_pp(over['write'])}]")
            if any(under.values()):
                log(f"      少授(表格有RAM无→该补):   读[{_pp(under['read'])}] 写[{_pp(under['write'])}]")
        audit_rows.append({"username": username, "diff": {
            b: {kind: {rw: sorted(d[b][kind][rw]) for rw in ("read", "write")}
                for kind in ("over", "under")} for b in d}})

    auto = list_auto_policies(client)
    orphans = {u: att for u, att in auto.items() if u not in active_usernames}
    log("\n— 孤儿策略（RAM 有 wuji-oss-auto-* 但其用户不在在职成员表）—")
    if orphans:
        for u, att in sorted(orphans.items()):
            log(f"  · {POLICY_PREFIX}{u}  附加于: {', '.join(att) or '（未附加）'}  → 建议回收")
    else:
        log("  无 ✓")

    log("\n— 汇总 —")
    log(f"  对账 {len(plan)} 人；{diff_users} 人有差异；{len(orphans)} 个孤儿策略")
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"audit": audit_rows, "orphans": orphans}, f, ensure_ascii=False, indent=2)
        log(f"\n→ 对账结果已写入 {out_file}")
    return diff_users, orphans


# ---------------------------------------------------------------------------
# 本地配置
# ---------------------------------------------------------------------------
def load_local_env():
    """加载脚本同目录下的 .env（KEY=VALUE，# 为注释），不覆盖已存在的环境变量。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    load_local_env()
    ap = argparse.ArgumentParser(description="飞书多维表格 -> 阿里云 OSS RAM 权限同步")
    ap.add_argument("--apply", action="store_true", help="真正写入阿里云（默认 dry-run）")
    ap.add_argument("--audit", action="store_true",
                    help="只读对账：RAM 实际 vs 表格期望，列出多授/少授/孤儿策略（不改动）")
    ap.add_argument("--create-users", action="store_true", help="RAM 用户不存在时自动创建")
    ap.add_argument("--map", default=os.environ.get("RAM_USER_MAP", DEFAULT_MAP_FILE),
                    help=f"RAM 用户名映射表文件（默认 {DEFAULT_MAP_FILE}）")
    ap.add_argument("--build-map", action="store_true",
                    help="按「RAM 显示名==成员姓名」生成/合并映射表（--map 指定路径），保留手工项后退出")
    ap.add_argument("--include-inactive", action="store_true", help="包含离职成员")
    ap.add_argument("--level", choices=["bucket", "dir"], default="dir",
                    help="权限粒度：bucket 桶级（整桶读写，先放开观察）/ dir 目录级（按子目录前缀，默认）")
    ap.add_argument("--only", help="仅处理姓名或用户名包含该子串的成员")
    ap.add_argument("--out", help="把生成的策略计划写入该 JSON 文件")
    ap.add_argument("--app-token", default=os.environ.get("FEISHU_APP_TOKEN", DEFAULT_APP_TOKEN))
    ap.add_argument("--member-table", default=os.environ.get("MEMBER_TABLE_ID", DEFAULT_MEMBER_TABLE))
    ap.add_argument("--bucket-table", default=os.environ.get("BUCKET_TABLE_ID", DEFAULT_BUCKET_TABLE))
    args = ap.parse_args()

    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        sys.exit("缺少 FEISHU_APP_ID / FEISHU_APP_SECRET")

    # 1) 读飞书
    print("→ 获取飞书 tenant_access_token …")
    token = feishu_token(app_id, app_secret)
    print("→ 读取对照表 …")
    valid_combos = build_valid_combos(feishu_search_all(token, args.app_token, args.bucket_table))
    print("→ 读取成员表 …")
    rows = feishu_search_all(token, args.app_token, args.member_table)

    members = [parse_member(r) for r in rows]
    if not args.include_inactive:
        members = [m for m in members if m["status"] == STATUS_ACTIVE]
    if args.only:
        kw = args.only
        members = [m for m in members if kw in m["name"] or kw in m["username"]]

    # 2) 解析 + 生成策略（按用户名合并：同一用户的多行取并集，避免后行覆盖前行）
    warnings, skipped = [], []
    by_user = {}   # username -> {"member": <首次出现的行>, "resolved": {...}}
    for mb in members:
        if not mb["username"]:
            skipped.append(f"  · {mb['name']}：账号为空，跳过")
            continue
        resolved = resolve_member(mb, valid_combos, warnings)
        if not resolved:
            skipped.append(f"  · {mb['name']} ({mb['username']})：无任何可解析的读写目录，跳过")
            continue
        slot = by_user.setdefault(mb["username"], {"member": mb, "resolved": {}})
        for bucket, pr in resolved.items():
            agg = slot["resolved"].setdefault(bucket, {"read": set(), "write": set(), "display": pr["display"]})
            agg["read"] |= pr["read"]
            agg["write"] |= pr["write"]

    plan = []
    for username, slot in by_user.items():
        eff = coerce_level(slot["resolved"], args.level)
        plan.append({"member": slot["member"], "policy_name": POLICY_PREFIX + username,
                     "resolved": eff, "doc": build_policy(eff), "level": args.level})

    # 2.3) 生成/合并 RAM 用户名映射表后退出
    if args.build_map:
        client = make_ram_client()
        by_display, names = ram_username_map(client)
        existing = load_user_map(args.map)
        mapping, added, unmatched = build_user_map(members, by_display, names, existing)
        with open(args.map, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"→ 映射表已写入 {args.map}：共 {len(mapping)} 条，"
              f"新增 {len(added)}，无 RAM 用户(值留空) {len(unmatched)}")
        if added:
            print("  新增匹配:", ", ".join(f"{p}→{r}" for p, r in added))
        if unmatched:
            print("  无 RAM 用户(可手工填):", ", ".join(unmatched))
        return

    # 2.5) 只读对账模式：RAM 实际 vs 表格期望
    if args.audit:
        if warnings:
            print("\n— 解析告警（桶/子目录不匹配）—")
            print("\n".join(warnings))
        active_usernames = {m["username"] for m in members if m["username"]}
        client = make_ram_client()
        diff_users, orphans = run_audit(client, plan, active_usernames, args.out)
        if diff_users or orphans:
            sys.exit(2)   # 让定时任务感知到有差异/孤儿
        return

    # 3) 打印计划
    _lvl_cn = "桶级" if args.level == "bucket" else "目录级"
    print("\n" + "=" * 70)
    print(f"计划处理 {len(plan)} 位成员 · 粒度={_lvl_cn}"
          + ("（DRY-RUN，未改动阿里云）" if not args.apply else "（APPLY）"))
    print("=" * 70)
    for p in plan:
        mb = p["member"]
        print(f"\n● {mb['name']}  用户={mb['username']}  策略={p['policy_name']}")
        for bucket in sorted(p["resolved"]):
            pr = p["resolved"][bucket]
            r = ", ".join(sorted(x or "<整桶>" for x in pr["read"])) or "—"
            w = ", ".join(sorted(x or "<整桶>" for x in pr["write"])) or "—"
            print(f"    桶 {bucket}（{pr['display']}）")
            print(f"      读: {r}")
            print(f"      写: {w}")
        size = len(json.dumps(p["doc"], ensure_ascii=False).encode("utf-8"))
        if size > RAM_DOC_LIMIT:
            print(f"    ⚠ 策略文档 {size} 字节超过 RAM 上限 {RAM_DOC_LIMIT}，需拆分")

    if skipped:
        print("\n— 跳过的成员 —")
        print("\n".join(skipped))
    if warnings:
        print("\n— 解析告警（桶/子目录不匹配）—")
        print("\n".join(warnings))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                [{"member": p["member"]["name"], "username": p["member"]["username"],
                  "policy_name": p["policy_name"], "document": p["doc"]} for p in plan],
                f, ensure_ascii=False, indent=2)
        print(f"\n→ 计划已写入 {args.out}")

    # 4) 落地
    if not args.apply:
        print("\n这是 DRY-RUN。确认无误后加 --apply 执行（如需建用户再加 --create-users）。")
        return

    print("\n→ 开始写入阿里云 RAM …")
    client = make_ram_client()
    user_map = load_user_map(args.map)
    print(f"→ 映射表 {args.map}：{len(user_map)} 条" if user_map
          else "→ 无映射表，拉取 RAM 用户按显示名实时匹配")
    by_display, ram_names = ram_username_map(client)

    ok = fail = no_user = 0
    for p in plan:
        mb = p["member"]
        real_user = map_username(mb, user_map, by_display, ram_names)
        if real_user and real_user != mb["username"]:
            print(f"\n● {mb['name']} ({mb['username']} → RAM 用户 {real_user})")
        else:
            print(f"\n● {mb['name']} ({mb['username']})")

        if real_user is None and not args.create_users:
            # 无 RAM 用户且不建用户：只建/更新策略，不附加
            no_user += 1
            apply_user(client, mb["username"], p["policy_name"], p["doc"],
                       create_users=False, log=print, attach=False)
            time.sleep(0.2)
            continue

        attach_user = real_user or mb["username"]   # 无匹配且 --create-users 时按邮箱前缀建
        if apply_user(client, attach_user, p["policy_name"], p["doc"], args.create_users, print):
            ok += 1
        else:
            fail += 1
        time.sleep(0.2)  # 轻微限速，避免触发 RAM API 频控
    print(f"\n完成：成功 {ok}，失败 {fail}，无 RAM 用户(仅建策略未附加) {no_user}。")
    if fail:
        sys.exit(2)   # 让调用方（定时任务）能感知到有失败


# ---------------------------------------------------------------------------
# 高层接口（供 langchaindev bot / 调度器调用，凭证读 settings）
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_MAP_PATH = os.path.join(_MODULE_DIR, "ram_user_map.json")


def _feishu_creds():
    aid = os.environ.get("FEISHU_APP_ID")
    sec = os.environ.get("FEISHU_APP_SECRET")
    if not aid or not sec:
        from config.settings import settings
        aid = aid or settings.FEISHU_APP_ID
        sec = sec or settings.FEISHU_APP_SECRET
    return aid, sec


def load_members(include_inactive=False):
    """读飞书 -> (members, valid_combos)。"""
    aid, sec = _feishu_creds()
    token = feishu_token(aid, sec)
    combos = build_valid_combos(feishu_search_all(token, DEFAULT_APP_TOKEN, DEFAULT_BUCKET_TABLE))
    rows = feishu_search_all(token, DEFAULT_APP_TOKEN, DEFAULT_MEMBER_TABLE)
    members = [parse_member(r) for r in rows]
    if not include_inactive:
        members = [m for m in members if m["status"] == STATUS_ACTIVE]
    return members, combos


def build_plan(members, combos, level="dir"):
    """members + 对照表 -> plan（按用户名合并多行）。

    level='bucket' 桶级（整桶读写）/ 'dir' 目录级（按子目录前缀，默认）。
    plan 里的 resolved 与 doc 均为收紧后的实际下发范围，对账/卡片据此显示。
    """
    by_user, warnings = {}, []
    for mb in members:
        if not mb["username"]:
            continue
        resolved = resolve_member(mb, combos, warnings)
        if not resolved:
            continue
        slot = by_user.setdefault(mb["username"], {"member": mb, "resolved": {}})
        for bucket, pr in resolved.items():
            agg = slot["resolved"].setdefault(bucket, {"read": set(), "write": set(), "display": pr["display"]})
            agg["read"] |= pr["read"]
            agg["write"] |= pr["write"]
    plan = []
    for u, s in by_user.items():
        eff = coerce_level(s["resolved"], level)
        plan.append({"member": s["member"], "policy_name": POLICY_PREFIX + u,
                     "resolved": eff, "doc": build_policy(eff), "level": level})
    return plan


def audit_diff(plan, active_usernames):
    """只读对账 -> {rows:[{name,username,status,diff?}], orphans, n_diff}。"""
    client = make_ram_client()
    rows, n_diff = [], 0
    for p in plan:
        mb = p["member"]
        actual = ram_actual(client, mb["username"])
        if actual is None:
            rows.append({"name": mb["name"], "username": mb["username"], "status": "missing"})
            n_diff += 1
            continue
        d = diff_resolved(p["resolved"], actual)
        if d:
            rows.append({"name": mb["name"], "username": mb["username"], "status": "diff", "diff": d})
            n_diff += 1
        else:
            rows.append({"name": mb["name"], "username": mb["username"], "status": "ok"})
    auto = list_auto_policies(client)
    orphans = {u: att for u, att in auto.items() if u not in active_usernames}
    return {"rows": rows, "orphans": orphans, "n_diff": n_diff}


def apply_all(plan, create_users=False):
    """全量下发 -> {ok, fail, no_user, lines, level}。用映射表解析真实 RAM 用户名。

    每条 line 带上该成员实际下发的读写范围（桶/前缀），供结果卡列清楚「限制了哪些权限」。
    """
    client = make_ram_client()
    by_display, names = ram_username_map(client)
    user_map = load_user_map(_MAP_PATH)
    level = plan[0].get("level", "dir") if plan else "dir"
    ok = fail = no_user = 0
    lines = []
    for p in plan:
        mb = p["member"]
        scope = _scope_str(p["resolved"])
        real_user = map_username(mb, user_map, by_display, names)
        logs = []
        if real_user is None and not create_users:
            no_user += 1
            apply_user(client, mb["username"], p["policy_name"], p["doc"],
                       create_users=False, log=logs.append, attach=False)
            lines.append(f"• {mb['name']}：仅建策略未附加（无 RAM 用户）｜ {scope}")
            continue
        attach_user = real_user or mb["username"]
        if apply_user(client, attach_user, p["policy_name"], p["doc"], create_users, logs.append):
            ok += 1
            lines.append(f"• {mb['name']} → {attach_user} ✓ ｜ {scope}")
        else:
            fail += 1
            lines.append(f"• {mb['name']} ✗ {(logs[-1].strip() if logs else '')}")
        time.sleep(0.15)
    return {"ok": ok, "fail": fail, "no_user": no_user, "lines": lines, "level": level}


if __name__ == "__main__":
    main()
