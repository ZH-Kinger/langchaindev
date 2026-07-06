"""桶间迁移（同云一次性搬运）：阿里 OSS→OSS（hcs_mgw）/ 火山 TOS→TOS（DMS 1.0）。

与 core.transfer（跨云 tos↔oss）完全独立，只复用其引擎的加法式能力：
  engine_mgw.submit_cross_job(src_scheme="oss")  —— OSS→OSS
  engine_tos.submit_cross_job(src_is_tos=True)   —— TOS→TOS
"""
