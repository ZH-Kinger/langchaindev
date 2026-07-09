"""SSH 迁移链路径解析（纯逻辑，无外部依赖，便于单测）。

输入只接受源对象存储目录：`oss://<bucket>/<prefix>/`（以 '/' 结尾，按前缀为单位迁移）。
目的路径由配置推导（不由用户填）：
    段1 落地  = <SGP_OSS_MOUNT>/<prefix>/            （新加坡 ECS 本地挂载盘）
    段2 落地  = <THAI_DEST_ROOT>/<prefix>/           （泰国服务器）
前缀全程镜像，保持目录结构。段1/段2 的真实根目录在 orchestrator 里拼（读 settings），
本模块只解析+校验源、保留镜像前缀。
"""
import re
from dataclasses import dataclass


class SshPathError(ValueError):
    """路径语法错误，消息面向用户。"""


# 安全白名单：prefix 会拼进"SGP→泰国"的 ssh 双跳命令（泰国 shell 会二次解析），单层 shlex.quote 不够；
# 故在解析入口就严格限死可用字符，从源头杜绝命令注入(RCE)与路径穿越(..)。
# OSS 桶名规范：小写字母/数字/连字符，首尾字母数字，3–63 位。
# 用 \A...\Z 严格锚全串（不用 $——Python 的 $ 会在结尾换行符前匹配，`abcd\n` 会溜过）。
_BUCKET_RE = re.compile(r"\A[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]\Z")
# 前缀每一级：只允许 字母/数字/点/下划线/连字符；禁空格/shell 元字符(含换行)/`..`/`.`。
_SEG_RE = re.compile(r"\A[A-Za-z0-9._\-]+\Z")


def _validate(bucket: str, prefix: str) -> None:
    if not _BUCKET_RE.match(bucket):
        raise SshPathError(
            f"桶名非法：`{bucket}`（只允许小写字母、数字、连字符，首尾为字母数字，3–63 位）")
    if prefix:  # prefix 形如 'a/b/c/'（已补尾斜杠）
        for seg in prefix.rstrip("/").split("/"):
            if not seg or seg in (".", "..") or not _SEG_RE.match(seg):
                raise SshPathError(
                    f"目录前缀含非法字符或路径穿越：`{prefix}`。每级只允许 字母/数字/`.`/`_`/`-`，"
                    f"禁 `..`、空格及特殊符号（防命令注入）。")


@dataclass
class Plan:
    source_bucket: str    # 源 OSS 桶，如 wuji-data-tran
    source_prefix: str    # 源目录前缀，保证以 '/' 结尾（根为 ''）
    dest_subdir: str = ""  # 泰国侧目标子目录（相对 THAI_DEST_ROOT，尾斜杠）；空=镜像源前缀

    def source_uri(self) -> str:
        return f"oss://{self.source_bucket}/{self.source_prefix}"

    def dest_rel(self) -> str:
        """段2 落地相对 THAI_DEST_ROOT 的子目录：自定义优先，否则镜像源前缀。"""
        return self.dest_subdir or self.source_prefix


def parse_source(raw: str) -> Plan:
    """解析 `oss://<bucket>/<prefix>/` 为 Plan。非法抛 SshPathError（消息可直接回显）。"""
    raw = (raw or "").strip()
    if not raw or "://" not in raw:
        raise SshPathError(f"路径格式错误：`{raw}`，应形如 `oss://wuji-data-tran/子目录/`")
    scheme, rest = raw.split("://", 1)
    scheme = scheme.strip().lower()
    if scheme != "oss":
        raise SshPathError(f"SSH 迁移链的源只支持 oss://（阿里对象存储），收到 `{scheme}://`")
    rest = rest.lstrip("/")
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    if not bucket:
        raise SshPathError(f"路径缺少 bucket：`{raw}`")
    # 目录语义：非空前缀补尾斜杠
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    _validate(bucket, prefix)   # 严格白名单：防命令注入(泰国 ssh 双跳)+路径穿越
    return Plan(source_bucket=bucket, source_prefix=prefix)


def _norm_dest_subdir(raw: str) -> str:
    """规整并校验用户填的目标子目录：去首尾斜杠/空白、补尾斜杠、走同一道白名单。空→空串(镜像源)。"""
    raw = (raw or "").strip().strip("/")
    if not raw:
        return ""
    sub = raw + "/"
    # 复用段级白名单校验（同样会拼进泰国 ssh 双跳，必须防注入/穿越）
    for seg in sub.rstrip("/").split("/"):
        if not seg or seg in (".", "..") or not _SEG_RE.match(seg):
            raise SshPathError(
                f"目标子目录含非法字符或路径穿越：`{raw}`。每级只允许 字母/数字/`.`/`_`/`-`，禁 `..`/空格/特殊符号。")
    return sub


def build_plan(source_raw: str, dest_subdir: str = "") -> Plan:
    """对外入口：源路径(必填) + 目标子目录(可选，默认镜像源前缀)。两者都走白名单。"""
    plan = parse_source(source_raw)
    plan.dest_subdir = _norm_dest_subdir(dest_subdir)
    return plan
