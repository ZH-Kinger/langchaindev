"""
阿里云 OSS 管理工具。

凭证通过 STS AssumeRole 获取（utils.aliyun_client_factory），
按飞书 open_id 隔离权限。

写操作（create_bucket / put_object / delete_object / delete_bucket）
必须 confirm=true。
"""
import logging

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from config.settings import settings
from utils.aliyun_client_factory import get_oss_service, get_oss_bucket

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class OSSSchema(BaseModel):
    action: str = Field(
        description=(
            "操作类型：\n"
            "  'list_buckets'   - 列出所有 bucket\n"
            "  'list_objects'   - 列出 bucket 下的对象（需 bucket），可用 prefix 过滤\n"
            "  'bucket_size'    - 估算 bucket 已使用空间（需 bucket）\n"
            "  'create_bucket'  - 创建 bucket，需 bucket+region；confirm=true\n"
            "  'put_object'     - 上传文本对象，需 bucket+object_key+content；confirm=true\n"
            "  'delete_object'  - 删除对象，需 bucket+object_key；confirm=true\n"
            "  'delete_bucket'  - 删除空 bucket，需 bucket；confirm=true"
        )
    )
    bucket: str = Field(default="", description="bucket 名称")
    prefix: str = Field(default="", description="对象前缀过滤（如 logs/2026/）")
    region: str = Field(default="", description="bucket 所在 region")
    max_keys: int = Field(default=50, description="list_objects 返回上限，默认 50")

    # 写操作参数
    object_key: str = Field(default="", description="对象 Key，put_object / delete_object 用")
    content: str = Field(default="", description="put_object 内容（文本），UTF-8 编码")
    storage_class: str = Field(
        default="Standard",
        description="创建 bucket 时的存储类型：Standard / IA / Archive / ColdArchive",
    )
    acl: str = Field(
        default="private",
        description="bucket ACL：private / public-read / public-read-write",
    )

    confirm: bool = Field(default=False, description="所有写操作必须显式 true")
    open_id: str = Field(default="", description="飞书 open_id，系统自动注入")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def manage_oss(
    action: str,
    bucket: str = "",
    prefix: str = "",
    region: str = "",
    max_keys: int = 50,
    object_key: str = "",
    content: str = "",
    storage_class: str = "Standard",
    acl: str = "private",
    confirm: bool = False,
    open_id: str = "",
) -> str:
    action = action.strip().lower()

    try:
        import oss2
    except ImportError:
        return "❌ oss2 未安装，请执行：pip install oss2"

    try:
        # ── 只读：list_buckets ───────────────────────────────────────────────
        if action == "list_buckets":
            service = get_oss_service(open_id=open_id)
            if service is None:
                return "❌ OSS 凭证不可用：请确认 STS / RAM 映射已配置。"
            buckets = list(oss2.BucketIterator(service))
            if not buckets:
                return "当前账号下无 OSS bucket（或角色无 ListBuckets 权限）。"
            lines = [f"## OSS Bucket 列表（共 {len(buckets)} 个）"]
            lines.append("| 名称 | 区域 | 存储类型 | 创建时间 |")
            lines.append("|------|------|---------|---------|")
            for b in buckets:
                lines.append(
                    f"| `{b.name}` | {b.location} | {b.storage_class} | {b.creation_date} |"
                )
            return "\n".join(lines)

        # ── 只读：list_objects ───────────────────────────────────────────────
        if action == "list_objects":
            if not bucket:
                return "❌ list_objects 需要提供 bucket。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            objects = list(oss2.ObjectIterator(b, prefix=prefix or "", max_keys=max_keys))[:max_keys]
            if not objects:
                return f"bucket `{bucket}` 下无对象（prefix={prefix!r}）。"
            lines = [f"## bucket `{bucket}` 对象列表（前 {len(objects)} 条，prefix={prefix or '-'})"]
            lines.append("| 对象 Key | 大小 | 修改时间 |")
            lines.append("|---------|------|---------|")
            for obj in objects:
                size_str = _fmt_size(obj.size)
                lines.append(f"| `{obj.key}` | {size_str} | {obj.last_modified} |")
            return "\n".join(lines)

        # ── 只读：bucket_size ────────────────────────────────────────────────
        if action == "bucket_size":
            if not bucket:
                return "❌ bucket_size 需要提供 bucket。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            total_size = 0
            total_count = 0
            for obj in oss2.ObjectIterator(b, prefix=prefix or ""):
                total_size += obj.size
                total_count += 1
                if total_count >= 100000:
                    return (
                        f"⚠️ bucket `{bucket}` 对象数超过 10 万，未完整统计。\n"
                        f"已统计前 100000 个对象，累计 {_fmt_size(total_size)}。"
                    )
            return (
                f"## bucket `{bucket}` 用量\n"
                f"- 对象数：**{total_count}**\n"
                f"- 总大小：**{_fmt_size(total_size)}**"
                + (f"\n- 过滤前缀：`{prefix}`" if prefix else "")
            )

        # ── 写操作：统一拦截 confirm ────────────────────────────────────────
        if action in ("create_bucket", "put_object", "delete_object", "delete_bucket") and not confirm:
            return (
                f"⚠️ OSS `{action}` 是写操作，"
                "请明确传入 `confirm=true` 后再调用。"
            )

        if action == "create_bucket":
            if not bucket:
                return "❌ create_bucket 需要 bucket 名称。"
            from utils.aliyun_client_factory import get_oss_auth
            auth, _ = get_oss_auth(open_id)
            if auth is None:
                return "❌ OSS 凭证不可用。"
            region = region or settings.PAI_DSW_REGION_ID or "cn-hangzhou"
            endpoint = f"https://oss-{region}.aliyuncs.com"
            b = oss2.Bucket(auth, endpoint, bucket)
            # ACL 映射
            acl_map = {
                "private":          oss2.BUCKET_ACL_PRIVATE,
                "public-read":      oss2.BUCKET_ACL_PUBLIC_READ,
                "public-read-write": oss2.BUCKET_ACL_PUBLIC_READ_WRITE,
            }
            b.create_bucket(
                acl_map.get(acl, oss2.BUCKET_ACL_PRIVATE),
                oss2.models.BucketCreateConfig(storage_class=storage_class),
            )
            logger.info("[OSS] 创建 bucket %s region=%s 操作者=%s", bucket, region, open_id)
            return f"✅ bucket `{bucket}` 已创建（region={region}, storage={storage_class}, acl={acl}）。"

        if action == "put_object":
            if not bucket or not object_key:
                return "❌ put_object 需要 bucket 和 object_key。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            b.put_object(object_key, content)
            logger.info("[OSS] 上传对象 %s/%s 大小=%d 操作者=%s",
                        bucket, object_key, len(content), open_id)
            return (
                f"✅ 对象 `{object_key}` 已上传到 bucket `{bucket}`，"
                f"大小 {_fmt_size(len(content.encode('utf-8')))}。"
            )

        if action == "delete_object":
            if not bucket or not object_key:
                return "❌ delete_object 需要 bucket 和 object_key。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            b.delete_object(object_key)
            logger.warning("[OSS] 删除对象 %s/%s 操作者=%s", bucket, object_key, open_id)
            return f"⚠️ 对象 `{object_key}` 已从 bucket `{bucket}` 删除（不可恢复）。"

        if action == "delete_bucket":
            if not bucket:
                return "❌ delete_bucket 需要 bucket。"
            b = get_oss_bucket(open_id, bucket, region=region)
            if b is None:
                return "❌ OSS 凭证不可用。"
            b.delete_bucket()
            logger.warning("[OSS] 删除 bucket %s 操作者=%s", bucket, open_id)
            return f"⚠️ bucket `{bucket}` 已删除（必须为空才能删除）。"

        return (
            f"❓ 未知 action：{action}，可选 list_buckets / list_objects / bucket_size / "
            "create_bucket / put_object / delete_object / delete_bucket。"
        )

    except Exception as e:
        logger.error("[OSS] 调用失败 action=%s err=%s", action, e, exc_info=True)
        return f"❌ OSS API 调用失败：{e}"


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


# ── LangChain 工具封装 ──────────────────────────────────────────────────────────

oss_tool = StructuredTool.from_function(
    func=manage_oss,
    name="manage_oss",
    description=(
        "管理阿里云 OSS。"
        "只读：list_buckets 列 bucket、list_objects 列对象、bucket_size 估算用量。"
        "写操作（create_bucket / put_object / delete_object / delete_bucket）"
        "必须 confirm=true。"
    ),
    args_schema=OSSSchema,
)
