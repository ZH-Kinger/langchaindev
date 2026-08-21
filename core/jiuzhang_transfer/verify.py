"""九章端到端四层校验 —— **不采信 ossutil 报的「成功」**。

与泰国链同样的判据与 fail-closed 原则（见 core/ssh_transfer/verify.py 的完整说明）：
最坏的结局不是失败，是「报成功但少数据」——几个月后训练读到坏文件才发现，那时源可能已清理。

判据是「**源的每个对象都在目的端且字节一致**」，不是两边数量相等 —— `/root/nas` 是
共享盘，有别人的文件是正常的，拿数量相等当判据会让正常情况报失败，运维就学会忽略校验结果了。
"""
import base64
import logging
import shlex

from config.settings import settings

from . import engine

logger = logging.getLogger(__name__)

_SENTINEL = "__JZ_VERIFY_EOF__"


class VerifyError(RuntimeError):
    """校验本身没跑成（不是「校验不通过」）。调用方必须当 FAILED 处理。"""


def _list_source(bucket: str, prefix: str) -> dict:
    """在 **bot 本地**用 oss2 列源对象。刻意不在九章上列 —— 源清单与目的清单出自
    不同的机器，才谈得上交叉验证。"""
    try:
        import oss2
    except ImportError:
        raise VerifyError("校验需要 oss2：pip install oss2")
    if not (settings.ALIYUN_ACCESS_KEY_ID and settings.ALIYUN_ACCESS_KEY_SECRET):
        raise VerifyError("校验需要阿里云 AK/SK 来列举源对象。")
    ep = (getattr(settings, "JIUZHANG_OSS_ENDPOINT", "") or "oss-cn-hangzhou.aliyuncs.com")
    b = oss2.Bucket(oss2.Auth(settings.ALIYUN_ACCESS_KEY_ID, settings.ALIYUN_ACCESS_KEY_SECRET),
                    f"https://{ep}", bucket)
    out = {}
    try:
        for o in oss2.ObjectIteratorV2(b, prefix=prefix):
            if not o.key.endswith("/"):
                out[o.key[len(prefix):]] = o.size
    except Exception as e:
        raise VerifyError(f"列举源对象失败（本次不给校验结论）：{e}")
    return out


def _list_dest(dest_dir: str) -> dict:
    """在九章上列目的文件。

    `find -printf '%s\\t%P\\0'`：**NUL 分隔**（文件名可能含换行）、**字节数在前**
    （`%P\\t%s\\n` 遇含换行的文件名会错行）、**结尾哨兵 + 查 rc**（丢 rc 则 find 超时
    会被渲染成「目的端缺 N 万个」，而运维对那个结论的合理反应是重传）。
    """
    script = f'''
set -u
D={shlex.quote(dest_dir)}
if [ ! -d "$D" ]; then echo NODIR; echo "{_SENTINEL}"; exit 0; fi
find "$D" -type f -printf '%s\\t%P\\0'
rc=$?
echo ""
echo "FINDRC=$rc"
echo "{_SENTINEL}"
'''
    rc, out, err = engine.run(script, timeout=1800)
    if rc != 0:
        raise VerifyError(f"列举目的文件失败(rc={rc})，本次不给校验结论：{(err or out)[:200]}")
    if _SENTINEL not in (out or ""):
        raise VerifyError("目的清单没读到结尾哨兵（find 可能超时或被截断），本次不给校验结论。")
    if "FINDRC=" in out:
        code = out.split("FINDRC=", 1)[1].split()[0].strip()
        if code not in ("0", ""):
            raise VerifyError(f"远端 find 退出码 {code}，清单不可信，本次不给校验结论。")
    body = out.split(_SENTINEL)[0]
    if "NODIR" in body:
        return {}
    res = {}
    for rec in body.split("\0"):
        if "\t" not in rec:
            continue
        size_s, _, rel = rec.partition("\t")
        try:
            res[rel.strip("\n")] = int(size_s.strip())
        except ValueError:
            continue
    return res


def _sample_compare(bucket: str, prefix: str, dest_dir: str, keys: list) -> tuple:
    """抽样：在九章上把对象重下一份，与已落地的逐字节 `cmp`。

    ⚠️ 文件名来自**对象 key（外部可控）**，走 **base64 + NUL 传递，绝不用 heredoc** ——
    一个内容恰好是结束标记的 key 就能提前终止 heredoc，让后面的内容在生产机上被当命令执行。
    """
    if not keys:
        return 0, 0, 0
    b64 = base64.b64encode("\0".join(keys).encode("utf-8")).decode("ascii")
    ep = (getattr(settings, "JIUZHANG_OSS_ENDPOINT", "") or "").strip()
    script = f'''
set -u
D={shlex.quote(dest_dir)}
SRC="oss://{bucket}/{prefix}"
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT
same=0; diff=0; failed=0
echo {b64} | base64 -d > "$TMP/keys"
while IFS= read -r -d "" rel; do
  [ -n "$rel" ] || continue
  if ! ossutil cp "$SRC$rel" "$TMP/blob" {f'-e {shlex.quote(ep)}' if ep else ''} -f >/dev/null 2>&1; then
    failed=$((failed+1)); continue
  fi
  if cmp -s "$TMP/blob" "$D$rel"; then same=$((same+1)); else diff=$((diff+1)); fi
  rm -f "$TMP/blob"
done < "$TMP/keys"
echo "SAME=$same DIFF=$diff FAILED=$failed"
'''
    try:
        _, out, _ = engine.run(script, timeout=2400)
    except Exception:
        logger.warning("[JZ] 抽样比对失败", exc_info=True)
        return 0, 0, len(keys)
    vals = {"SAME": 0, "DIFF": 0, "FAILED": 0}
    for tok in (out or "").split():
        for k in vals:
            if tok.startswith(k + "="):
                try:
                    vals[k] = int(tok.split("=", 1)[1])
                except ValueError:
                    pass
    if sum(vals.values()) == 0:
        return 0, 0, len(keys)
    return vals["SAME"], vals["DIFF"], vals["FAILED"]


def verify_pull(job: dict, *, samples: int = 0) -> dict:
    """四层校验，返回结构化结论。清单拿不到时抛 VerifyError（调用方当 FAILED）。"""
    if not samples:
        try:
            samples = max(1, int(getattr(settings, "JIUZHANG_VERIFY_SAMPLES", 5) or 5))
        except (TypeError, ValueError):
            samples = 5
    bucket, prefix = job["source_bucket"], job["source_prefix"]
    dest = job.get("dest_dir") or engine.dest_dir(prefix, job.get("dest_rel", ""))

    src = _list_source(bucket, prefix)
    dst = _list_dest(dest)

    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    size_bad = sorted(k for k in (set(src) & set(dst)) if src[k] != dst[k])
    src_bytes = sum(src.values())
    dst_bytes = sum(dst[k] for k in src if k in dst)     # 只统计**源 key 集合内**的
    covered = not missing and not size_bad

    # 抽样只取路径干净的：对象 key 允许含 `..`，拼进 "$D$rel" 会读到目标目录之外。
    common = sorted(k for k in (set(src) & set(dst))
                    if ".." not in k.split("/") and not k.startswith("/"))
    step = max(1, len(common) // max(1, samples))
    pick = common[::step][:samples] if common else []
    same, diff, dl_fail = _sample_compare(bucket, prefix, dest, pick)

    passed = covered and src_bytes == dst_bytes and diff == 0 and same > 0
    # 区分「数据问题」与「校验环境问题」：样本全取样失败（凭证过期/临时目录满）不是数据不一致，
    # 不分开写的话会有人拿着「数据不一致」去重传几十 TB。
    env_issue = diff == 0 and same == 0 and dl_fail > 0
    if env_issue:
        l4 = f"L4 抽样内容 ✗ 校验环境问题：{dl_fail} 个样本全部取样失败（非数据不一致）"
    elif diff:
        l4 = f"L4 抽样内容 ✗ 数据不一致：{diff} 个样本与源不同（一致{same}/取样失败{dl_fail}）"
    elif not pick:
        l4 = "L4 抽样内容 — 无可抽样对象"
    else:
        l4 = f"L4 抽样内容 ✓ 一致{same}（取样失败{dl_fail}）"

    lines = [
        f"L1 对象覆盖 源{len(src)} 个，目的端缺{len(missing)} 个 "
        f"{'✓' if not missing else '✗'}（目的端另有 {len(dst)} 个文件）",
        f"L2 字节总量 源{src_bytes}/目的对应{dst_bytes} {'✓' if src_bytes == dst_bytes else '✗'}",
        f"L3 逐文件 字节不符{len(size_bad)} {'✓' if not size_bad else '✗'}",
        l4,
    ]
    if extra:
        lines.append(f"注：目的端多出 {len(extra)} 个源上没有的文件（不判失败，请人工确认）")
    for lab, lst in (("缺失", missing), ("字节不符", size_bad)):
        for k in lst[:3]:
            lines.append(f"  [{lab}] …{k[-70:]}")

    return {"passed": passed, "env_issue": env_issue,
            "src_objects": len(src), "dst_objects": len(dst),
            "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "missing": len(missing), "extra": len(extra), "size_mismatch": len(size_bad),
            "sample_same": same, "sample_diff": diff, "sample_failed": dl_fail,
            "summary": "\n".join(lines)}
