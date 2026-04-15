"""
阿里云 SLS (Simple Log Service) HMAC-SHA1 签名认证模块。
供 prometheus_tool.py 调用，为 Prometheus HTTP API 请求生成认证 headers。

参考文档：
  https://help.aliyun.com/document_detail/29012.html（SLS 请求签名）
"""
import hmac
import hashlib
import base64
from email.utils import formatdate
from urllib.parse import urlparse


def build_auth_headers(
    method: str,
    url: str,
    ak_id: str,
    ak_secret: str,
    body: str = "",
    content_type: str = "",
) -> dict:
    """
    生成阿里云 SLS 请求所需的完整认证 headers。

    Args:
        method:       HTTP 方法，如 "GET"
        url:          完整请求 URL（含 path，不含 query string 用于签名）
        ak_id:        阿里云 AccessKey ID
        ak_secret:    阿里云 AccessKey Secret
        body:         请求体字符串，GET 请求传 ""
        content_type: 请求体类型，GET 请求传 ""

    Returns:
        dict，直接 merge 到 requests.get/post 的 headers 参数中即可。
    """
    date_str = formatdate(usegmt=True)
    resource = urlparse(url).path

    # SLS 规范 headers（必须按 key 字典序排列）
    sls_headers = {
        "x-log-date":             date_str,
        "x-log-signaturemethod":  "hmac-sha1",
    }
    canonical_sls = "".join(
        f"{k}:{v}\n" for k, v in sorted(sls_headers.items())
    )

    # Content-MD5：仅 POST/PUT 有 body 时计算，GET 为空字符串
    content_md5 = (
        base64.b64encode(hashlib.md5(body.encode("utf-8")).digest()).decode()
        if body else ""
    )

    # StringToSign
    string_to_sign = "\n".join([
        method.upper(),
        content_md5,
        content_type,
        date_str,
        canonical_sls + resource,
    ])

    # HMAC-SHA1 签名
    signature = base64.b64encode(
        hmac.new(
            ak_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode()

    return {
        **sls_headers,
        "Date":          date_str,
        "Authorization": f"LOG {ak_id}:{signature}",
        "Content-MD5":   content_md5,
        "Content-Type":  content_type,
    }
