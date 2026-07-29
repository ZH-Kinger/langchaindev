"""多阿里云主账号的「账号档案」注册表 —— 临时 AK 发放的账号维度来源。

**为什么是档案注册制而不是整包复制**：延长/撤销审批（「访问凭证延长/撤销 申请」）被多个账号
**共用同一个 definitionCode**，而同一 code 只能有一个处理器认领（routes 里三个 should_handle_*
都是严格等值），两份独立副本必然一个抢到、另一个永远收不到。所以延长/撤销必须由单一处理器按
**凭证ID → grant 记录 → 账号**分派，这就要求 grant 带账号维度、引擎按档案取凭证与前缀。

隔离维度（每档一份，互不相犯）：
  · 凭证    —— 各账号自己的 RAM 可写 AK
  · Redis   —— `temp_ak:` / `temp_ak_1949:`
  · 凭证ID  —— `tak-` / `tak1949-`（共用的延长/撤销审批据此分派）
  · 云上对象 —— 用户名 `tempak-*` / `tempak-1949-*`，策略名随用户名走
  · 表单字段 —— 各账号的审批模板 widget id 不同

默认档（slug="") 就是现有主账号，**行为与本文件引入前逐字一致**——新增账号不得改变它。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SLUG = ""          # 现有主账号（历史数据无账号标识，必须保持空 slug）


@dataclass(frozen=True)
class AccountProfile:
    """一个阿里云主账号在临时 AK 发放链路上的全部账号绑定信息。"""
    slug: str                  # "" = 现有主账号；"1949" = 第二主账号
    uid: str                   # 阿里云主账号 UID（仅用于日志/回执可读性）
    issue_code: str            # 发放审批 definitionCode
    ak_id: str                 # RAM 可写 AK（建子号/建AK/挂策略/硬删都用它）
    ak_secret: str
    bucket_map_raw: str        # 展示桶名 → {region,bucket} 的 JSON 串
    chat_id: str               # 内部回执群，留空回退 FEISHU_CHAT_ID
    grant_prefix: str          # 凭证ID 前缀
    redis_prefix: str          # Redis 命名空间前缀
    user_prefix: str           # RAM 登录名前缀
    display_suffix: str        # RAM 控制台显示名后缀
    subject_label: str         # 表单里「主体」那一栏的叫法（企业名 / 使用人名），用于回执与防串校验措辞
    # 主体是自然人姓名吗？是 → 延期/撤销防串校验要求**精确相等**（包含匹配对人名太松：
    # 「张三」会通过「张三丰」的校验）。企业名可宽松（容错"有限公司"等后缀差异）。
    subject_is_person: bool = False
    comment_user_id: str = ""  # 凭证评论的发出身份；留空 = 沿用全局管理员链路（见 delivery._comment_user_id）
    fields: dict = field(default_factory=dict)          # 发放表单字段映射
    has_platform_field: bool = True                     # 表单是否有「平台」单选（无则恒为阿里云）

    @property
    def label(self) -> str:
        return f"账号{self.slug}" if self.slug else "主账号"


# ── 发放表单字段映射 ──────────────────────────────────────────────────────────
# 结构：{逻辑名: (env 覆盖项, (候选名…含真机 widget id))}。名字改了设 env 覆盖即可。

# 默认档：真机模板「数据外采访问凭证申请」(5B4A…) 5 控件
_FIELDS_DEFAULT = {
    "platform":      ("TEMP_AK_FIELD_PLATFORM",      ("平台", "云平台", "platform", "widget17846401222860001")),
    "enterprise":    ("TEMP_AK_FIELD_ENTERPRISE",    ("使用企业名称", "企业名称", "使用方", "外采企业", "enterprise", "widget17846886904010001")),
    "perm":          ("TEMP_AK_FIELD_PERM",          ("权限设置", "权限", "读写", "perm", "widget17846401501570001")),
    "date_interval": ("TEMP_AK_FIELD_DATE_INTERVAL", ("DateInterval", "有效期", "生效到期", "起止时间", "date_interval", "widget17846402309610001")),
    "directory":     ("TEMP_AK_FIELD_DIRECTORY",     ("申请目录", "目录", "路径", "directory", "widget17846402564230001")),
}

# 1949 档：真机模板「数据访问凭证申请（产线）」(0133C4FC…) 5 控件（飞书 API 实拉，widget id 已确认）。
# 与默认档三点不同：① 无「平台」单选（恒阿里云）② 主体是「使用人名称」不是企业 ③ 多一个「备注」。
# 逻辑名仍复用 "enterprise"（= 主体名称），避免解析层为改名而分叉；措辞差异走 subject_label。
_FIELDS_1949 = {
    "enterprise":    ("TEMP_AK_1949_FIELD_SUBJECT",       ("使用人名称", "使用人", "使用方", "subject", "widget17846410216400001")),
    "perm":          ("TEMP_AK_1949_FIELD_PERM",          ("权限设置", "权限", "读写", "perm", "widget17852975709640001")),
    "date_interval": ("TEMP_AK_1949_FIELD_DATE_INTERVAL", ("DateInterval", "有效期", "生效到期", "date_interval", "widget17852976459760001")),
    "directory":     ("TEMP_AK_1949_FIELD_DIRECTORY",     ("访问目录", "申请目录", "目录", "路径", "directory", "widget17852975954890001")),
    "note":          ("TEMP_AK_1949_FIELD_NOTE",          ("备注", "说明", "note", "remark", "widget17852976732260001")),
}


def _default_profile() -> AccountProfile:
    """现有主账号。所有取值与本文件引入前的硬编码逐字相同，保证零行为变化。"""
    return AccountProfile(
        slug=DEFAULT_SLUG,
        uid=getattr(settings, "ALIYUN_BOT_ACCOUNT_UID", "") or "",
        issue_code=settings.TEMP_AK_APPROVAL_CODE,
        ak_id=settings.ALIYUN_ACCESS_KEY_ID,
        ak_secret=settings.ALIYUN_ACCESS_KEY_SECRET,
        bucket_map_raw=settings.TEMP_AK_BUCKET_MAP_RAW or "{}",
        chat_id=settings.TEMP_AK_CHAT_ID,
        grant_prefix="tak-",
        redis_prefix="temp_ak:",
        user_prefix="tempak-",
        display_suffix="-临时外采用户",
        subject_label="使用企业名称",
        subject_is_person=False,     # 企业名 → 防串校验保持宽松包含匹配
        comment_user_id="",          # 默认档从不覆盖：保持与 RAM 建号审批同一个管理员身份
        fields=_FIELDS_DEFAULT,
        has_platform_field=True,
    )


def _profile_1949() -> AccountProfile | None:
    """第二主账号 1339279783371949。**任一相关配置存在即注册**（不要求 AK 齐全）。

    刻意不要求 AK：若因误删 AK 而不注册档案，该账号已有的 grant 会**静默失效**——
    `sweep_expired` 不再扫 `temp_ak_1949:grant:*`（返回空、看着像"没有到期的"），
    而 `_key()` 又会把 `tak1949-…` 按默认档算成 `temp_ak:grant:…` 从而读不到，
    于是已发出去的长期 AK 与 RAM 子号会永久残留在云上、运维毫无察觉。
    注册（前缀恒定、扫得到）+ 真要建号/删号时在 ram_client() 显式抛错，才是可观测的失败。
    发放路由不受影响：code 为空时 by_issue_code/issue_codes 本就过滤掉，不会误放行。
    """
    code = getattr(settings, "TEMP_AK_1949_APPROVAL_CODE", "") or ""
    ak = getattr(settings, "ALIYUN_1949_ACCESS_KEY_ID", "") or ""
    sk = getattr(settings, "ALIYUN_1949_ACCESS_KEY_SECRET", "") or ""
    if not (code or ak or sk):
        return None                # 三项全空 = 该账号从未接入
    return AccountProfile(
        slug="1949",
        uid="1339279783371949",
        issue_code=code,
        ak_id=ak,
        ak_secret=sk,
        bucket_map_raw=getattr(settings, "TEMP_AK_1949_BUCKET_MAP_RAW", "") or "{}",
        chat_id=getattr(settings, "TEMP_AK_1949_CHAT_ID", "") or "",
        grant_prefix="tak1949-",
        redis_prefix="temp_ak_1949:",
        user_prefix="tempak-1949-",
        display_suffix="-1949产线临时用户",
        subject_label="使用人名称",
        subject_is_person=True,      # 自然人姓名 → 防串校验要求精确相等
        comment_user_id=getattr(settings, "TEMP_AK_1949_COMMENT_USER_ID", "") or "",
        fields=_FIELDS_1949,
        has_platform_field=False,      # 该模板没有平台单选 → 恒阿里云
    )


def profiles() -> list[AccountProfile]:
    """已注册的账号档案。**每次现算不缓存**：settings 在测试里会被 monkeypatch，缓存会让用例互相污染。"""
    out = [_default_profile()]
    p = _profile_1949()
    if p:
        out.append(p)
    return out


def default() -> AccountProfile:
    return _default_profile()


def by_slug(slug: str) -> AccountProfile:
    slug = slug or DEFAULT_SLUG
    for p in profiles():
        if p.slug == slug:
            return p
    # 档案被下线（如 .env 删了 AK）但 Redis 里还有该档 grant：不静默退回默认档，
    # 否则会拿**错误账号的 AK** 去删/改另一个账号的用户。宁可显式失败。
    raise UnknownAccountError(f"未注册的账号档案 slug={slug!r}（检查该账号的 .env 配置是否被移除）")


def by_issue_code(code: str) -> AccountProfile | None:
    """发放审批事件按 definitionCode 定档；没有匹配返回 None（非本 bot 该管的审批）。"""
    code = (code or "").strip()
    if not code:
        return None
    for p in profiles():
        if p.issue_code and p.issue_code == code:
            return p
    return None


def by_grant_id(grant_id: str) -> AccountProfile | None:
    """凭证ID → 账号档案。**共用的延长/撤销审批靠它分派**。

    按前缀最长优先匹配：`tak-` 与 `tak1949-` 在字符上并不互为前缀（第 4 位一个是 `-` 一个是 `1`），
    但排序后再匹配可以让将来加 `tak19-` 之类的新档不至于被旧档抢走。
    """
    gid = (grant_id or "").strip()
    if not gid:
        return None
    for p in sorted(profiles(), key=lambda x: len(x.grant_prefix), reverse=True):
        if gid.startswith(p.grant_prefix):
            return p
    return None


def issue_codes() -> set[str]:
    """所有档案的发放审批 code（供 routes 白名单）。"""
    return {p.issue_code for p in profiles() if p.issue_code}


def subject_label_for(grant: dict | None) -> str:
    """该凭证所属账号对「主体」那一栏的叫法（使用企业名称 / 使用人名称）。

    **单一来源**：delivery / cards 都调这里，别各自复制一份 —— 同一种漂移已经在 _FIELDS 上踩过。
    """
    try:
        return by_slug((grant or {}).get("account", "")).subject_label
    except Exception:
        return "使用方"


def assert_account_consistent(grant: dict) -> str:
    """校验 grant 的两套账号真相源一致，返回归一化后的 slug；不一致抛 UnknownAccountError。

    两套源：显式字段 `grant["account"]` 与 `grant_id` 前缀。任何拿凭证去操作云资源的入口都该先过
    这道：背离时若放行，会出现「用 A 的 AK 去动 B 的对象」——轻则报 EntityNotExist 被吞成日志、
    状态却被置成已处理（假成功），重则在 A 账号里误伤同名对象。
    """
    slug = (grant or {}).get("account", "") or DEFAULT_SLUG
    gid = (grant or {}).get("grant_id", "")
    if gid:
        by_prefix = by_grant_id(gid)
        if by_prefix is not None and by_prefix.slug != slug:
            raise UnknownAccountError(
                f"grant 账号维度自相矛盾：account={slug!r} 但凭证ID 前缀指向 {by_prefix.slug!r}"
                f"（grant_id={gid}）。拒绝用可能错误的账号凭证操作。")
    return slug


def chat_id_for(grant: dict | None = None, profile: AccountProfile | None = None) -> str:
    """内部回执/告警群：该账号档案的 chat_id → TEMP_AK_CHAT_ID → FEISHU_CHAT_ID。

    有档案就必须用档案的：否则第二账号的凭证回执与失败告警会混进第一账号的运维群。
    """
    p = profile
    if p is None and grant is not None:
        try:
            p = by_slug(grant.get("account", ""))
        except Exception:
            p = None
    if p is not None and p.chat_id:
        return p.chat_id
    return settings.TEMP_AK_CHAT_ID or settings.FEISHU_CHAT_ID


def bucket_map(profile: AccountProfile) -> dict:
    try:
        m = json.loads(profile.bucket_map_raw or "{}")
        return m if isinstance(m, dict) else {}
    except Exception:
        logger.warning("[temp_ak] %s 的桶映射 JSON 非法，按空表处理", profile.label)
        return {}


def ram_client(profile: AccountProfile):
    """该账号的 RAM 可写 client。**显式传 ak/sk，绝不走 permsync.make_ram_client() 的零参数路径**。

    原因：那条路径优先读进程环境变量 `ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET`（permsync.py 里），
    一旦运维在服务器上设了这对 env，**所有账号的建号请求都会被劫持到同一个账号**，
    分账号配置形同虚设、且失败得很隐蔽（会去错误的账号里建/删用户）。
    因此这里必须走显式传参的重载。（部署机现状已核：该 env 未设置。）
    """
    if not profile.ak_id or not profile.ak_secret:
        raise UnknownAccountError(f"{profile.label} 缺少 RAM 可写 AK，无法建号")
    from core.oss_perm.permsync import make_ram_client
    return make_ram_client(ak=profile.ak_id, sk=profile.ak_secret)


class UnknownAccountError(RuntimeError):
    """账号档案缺失或未注册。"""
