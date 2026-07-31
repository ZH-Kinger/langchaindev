"""段2 端到端校验：**不采信 ossutil 自己的「Success」**，独立比对源(OSS) vs 目的(泰国磁盘)。

为什么必须有：21 TB 迁移最坏的结局不是失败，而是「报成功但少数据/内容被截断」——
运维毫无察觉，几个月后训练读到坏文件才发现，那时源可能已经清理了。ossutil 的退出码只
反映它自己的动作，不做端到端核对。

四层，从便宜到贵，任一层不过即整体不通过：

  L1 对象数      少文件（最常见的静默失败）
  L2 总字���      整体截断
  L3 逐文件字节  **最要紧** —— L1/L2 会被「多一个少一个刚好抵消」骗过，L3 不会
  L4 抽样内容    从 OSS 重下样本到泰国本地 `cmp` 逐字节比

刻意的设计：源清单在 **bot 侧**列（凭证在 bot），目的清单在 **泰国侧**列，
两边互不采信对方��汇总数 —— 单侧统计出错时不会自己给自己背书。

L4 为什么不用 CRC64：OSS 用 CRC64/ECMA-182，泰国机上没有 C 加速库，纯 Python 逐字节
对 400MB 文件要几分钟、不可用；重下 + `cmp` 用现成工具、网络限速、结论更硬。
"""
import base64
import json
import shlex

from config.settings import settings
from utils.logger import get_logger
from core.ssh_transfer import engine_ossutil as eo

logger = get_logger("ssh_transfer.verify")

_LIST_TIMEOUT = 1800      # 7.5 万文件 find 一遍（WekaFS）几分钟量级
_CMP_TIMEOUT = 2400


def _list_source(bucket: str, prefix: str) -> dict:
    """列 OSS 源清单 {相对路径: 字节数}。目录占位对象（key 以 / 结尾）不计。"""
    import oss2
    from utils.aliyun_client_factory import get_oss_auth
    auth, _ = get_oss_auth("")
    if auth is None:
        raise RuntimeError("拿不到 OSS 凭证，无法列源清单")
    ep = (settings.THAI_OSS_ENDPOINT or "").strip() or "oss-ap-southeast-1.aliyuncs.com"
    b = oss2.Bucket(auth, "https://" + ep, bucket)
    out = {}
    for o in oss2.ObjectIterator(b, prefix=prefix):
        if o.key.endswith("/"):
            continue
        out[o.key[len(prefix):]] = o.size or 0
    return out


_SENTINEL = "__VERIFY_LIST_EOF__"


def _list_dest(dest_dir: str) -> dict:
    """列泰国侧清单 {相对路径: 字节数}。

    两处刻意的严格（都是「把校验自己挂了误报成数据缺失」的防线）：
    - **必须查 rc + 结尾哨兵**。原先丢弃 rc：find 超时或双跳中断会返回空/半截输出，
      判定层就渲染成「L1 源 75850，目的端缺 75850 ✗」。运维的合理反应是重传 19.5TiB，
      而真相只是校验没跑完。宁可抛错说「校验没做成」，也不能给出一个假的缺失结论。
    - **NUL 分隔 + 字节数在前**。`-printf '%P\\t%s\\n'` 遇到含换行的文件名会把一条打成两行
      （并行 unit 清单已因同一个坑改成 `%f\\0`，这里当时是回退）。改 `%s\\t%P\\0`：
      NUL 分隔天然安全，字节数放前面让 split 一次即可、路径里的 tab 也不影响。
    """
    script = (f'cd {shlex.quote(dest_dir)} 2>/dev/null || {{ echo DEST_MISSING; exit 1; }}\n'
              f"find . -type f -printf '%s\\t%P\\0' 2>/dev/null\n"
              f"printf '\\n{_SENTINEL}\\n'")
    rc, out, err = eo.run_thai(script, timeout=_LIST_TIMEOUT)
    out = out or ""
    if "DEST_MISSING" in out:
        raise RuntimeError(f"泰国目标目录不存在：{dest_dir}")
    if rc != 0:
        raise RuntimeError(f"列泰国清单失败(rc={rc})：{(err or out)[:200]}")
    if _SENTINEL not in out:
        raise RuntimeError("列泰国清单未跑完（无结尾哨兵，可能 find 超时或连接中断）——"
                           "本次不给校验结论，请重跑校验，勿据此重传数据")
    # rsplit：哨兵恒在最后一行；用 split 的话，目的目录里只要有个文件名含哨兵串，
    # 清单就会被从中间截断、后半段全被判成「缺失」。
    body = out.rsplit(_SENTINEL, 1)[0]
    res = {}
    for rec in body.split("\0"):
        if "\t" not in rec:
            continue
        s, p = rec.split("\t", 1)
        s = s.strip()          # strip 已含换行，不需要再 lstrip("\n")
        if s.isdigit() and p:
            res[p] = int(s)
    return res


def _sample_compare(bucket: str, prefix: str, dest_dir: str, keys: list[str]) -> tuple[int, int, int]:
    """抽样内容比对：泰国侧从 OSS 重下再 `cmp`。返回 (一致, 不一致, 取样失败)。"""
    if not keys:
        return 0, 0, 0
    # 文件名清单来自 **OSS 对象 key（外部可控）**，绝不能塞进 heredoc：只要某个 key 里出现
    # 一行恰好等于结束标记，heredoc 就会提前终止、其后的内容被 shell 当命令执行。
    # 改成 base64 载荷 + 远端 `base64 -d` 落盘：载荷只含 A-Za-z0-9+/= ，无法逃逸。
    # （文件名本身在使用处始终是双引号里的变量展开 `"$rel"`，不会被二次解析。）
    # **NUL 分隔**：仅 base64 只堵住了「key 里含结束标记 → heredoc 提前终止 → 后续内容被当命令
    # 执行」这条 RCE；换行仍会把一条 key 拆成两条（→ 两个假路径 DL_FAIL + 真文件没被抽到）。
    # NUL 分隔 + `read -r -d ''` 才是完整的修法。
    listing_b64 = base64.b64encode("\0".join(keys).encode()).decode()
    ep = (settings.THAI_OSS_ENDPOINT or "").strip()
    region = (settings.THAI_OSS_REGION or "").strip()
    opt = " ".join(filter(None, [f"-e {shlex.quote(ep)}" if ep else "",
                                 f"--region {shlex.quote(region)}" if region else ""]))
    src_base = shlex.quote(f"oss://{bucket}/{prefix}")
    script = f'''
set -u
D={shlex.quote(dest_dir)}
S={src_base}
T=$(mktemp -d) || exit 1
printf '%s' {listing_b64} | base64 -d > "$T/list"
while IFS= read -r -d '' rel; do
  [ -n "$rel" ] || continue
  timeout 600 ossutil cp "$S$rel" "$T/blob" {opt} -f >/dev/null 2>&1
  if [ ! -f "$T/blob" ]; then echo "DL_FAIL"; continue; fi
  if cmp -s "$T/blob" "$D/$rel"; then echo "SAME"; else echo "DIFF"; fi
  rm -f "$T/blob"
done < "$T/list"
rm -rf "$T"
printf '%s\\n' {_SENTINEL}
'''
    rc, out, err = eo.run_thai(script, timeout=_CMP_TIMEOUT)
    out = out or ""
    if rc != 0 or _SENTINEL not in out:
        # 同 _list_dest：抽样没跑完就不给「内容不一致」的结论，否则会把校验环境故障
        # 说成数据损坏。返回 (0,0,len(keys)) 让上层按「校验环境问题」而非「数据问题」渲染。
        logger.warning("[VERIFY] 抽样比对未跑完 rc=%s：%s", rc, (err or out)[:200])
        return 0, 0, len(keys)
    same = diff = fail = 0
    for line in out.splitlines():
        line = line.strip()
        if line == "SAME":
            same += 1
        elif line == "DIFF":
            diff += 1
        elif line == "DL_FAIL":
            fail += 1
    if diff:
        logger.error("[VERIFY] 抽样内容不一致 %d 个（源与目的字节不同）", diff)
    return same, diff, fail


def verify_stage2(job: dict, *, samples: int = 5) -> dict:
    """对一个 job 的段2 结果做四层校验。返回结构化结论（passed + 各层数字 + 明细摘要）。

    不抛异常（除拿不到清单）：校验本身失败也要能落进 job、推到卡片上。
    """
    bucket = (settings.SGP_OSS_BUCKET or "").strip()
    prefix = job["source_prefix"]
    # **必须与传输器写入的目录同一个实现**（paths.dest_dir）：算法漂移 = 校验了个空目录，
    # 而空目录的表现是「目的端缺 7.5 万个」，运维的合理反应是重传 19.5TiB。
    from core.ssh_transfer import paths
    dest_dir = paths.dest_dir(settings.THAI_DEST_ROOT or "",
                              source_prefix=prefix, dest_rel=job.get("dest_rel", ""))

    src = _list_source(bucket, prefix)
    dst = _list_dest(dest_dir)

    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    size_bad = sorted(k for k in (set(src) & set(dst)) if src[k] != dst[k])
    # 判据是「**源的每个对象都在目的端、且字节一致**」，不是「两边数量/总量相等」：
    # 目的目录是共享数据盘，可能有历史文件、别人的文件、上一批迁移的产物 —— 拿数量相等
    # 当判据会让这些完全正常的情况报成失败，运维会学会忽略校验结果，那校验就废了。
    # 所以 L2 只统计**源 key 集合内**的目的字节；extra 单独报出、不判失败。
    src_bytes = sum(src.values())
    dst_bytes = sum(dst[k] for k in src if k in dst)
    covered = not missing and not size_bad

    # 抽样只取「路径干净」的：OSS 的 key 允许含 `..`，拼进 `"$D/$rel"` 会读到目标目录之外。
    # 影响有限（`cmp` 只读、结果最多报 DIFF），但没必要给它这个机会；被跳过的照样参与
    # L1/L3 的字节比对，不影响判定完整性。
    common = sorted(k for k in (set(src) & set(dst))
                    if ".." not in k.split("/") and not k.startswith("/"))
    step = max(1, len(common) // max(1, samples))
    pick = common[::step][:samples] if common else []
    same, diff, dl_fail = _sample_compare(bucket, prefix, dest_dir, pick)

    passed = covered and src_bytes == dst_bytes and diff == 0 and same > 0
    # 区分两种「L4 没过」：`diff>0` 是**数据问题**（内容真的不一样）；`same==0 and dl_fail>0`
    # 是**校验环境问题**（样本全下不下来：凭证过期、/tmp 放不下 blob、ossutil 不在 PATH）。
    # 都判不通过（fail-closed 不放宽），但结论文案必须区分 —— 否则运维会拿着「数据不一致」
    # 去重传 19.5TiB，而真正该修的是校验环境。
    env_issue = diff == 0 and same == 0 and dl_fail > 0
    if env_issue:
        l4 = f"L4 抽样内容 ✗ 校验环境问题：{dl_fail} 个样本全部取样失败（非数据不一致）"
    elif diff:
        l4 = f"L4 抽样内容 ✗ 数据不一致：{diff} 个样本内容与源不同（一致{same}/取样失败{dl_fail}）"
    else:
        l4 = f"L4 抽样内容 ✓ 一致{same}（取样失败{dl_fail}）"

    lines = [
        f"L1 对象覆盖 源{len(src)} 个，目的端缺{len(missing)} 个 {'✓' if not missing else '✗'}"
        f"（目的端另有 {len(dst)} 个文件）",
        f"L2 字节总量 源{src_bytes}/目的对应{dst_bytes} {'✓' if src_bytes == dst_bytes else '✗'}",
        f"L3 逐文件 字节不符{len(size_bad)} {'✓' if not size_bad else '✗'}",
        l4,
    ]
    if extra:
        lines.append(f"注：目的端多出 {len(extra)} 个源上没有的文件（不判失败，请人工确认）")
    for lab, lst in (("缺失", missing), ("字节不符", size_bad)):
        for k in lst[:3]:
            lines.append(f"  [{lab}] …{k[-70:]}")
        if len(lst) > 3:
            lines.append(f"  [{lab}] 另有 {len(lst) - 3} 条")

    result = {
        "passed": passed,
        "env_issue": env_issue,      # True = 校验环境故障，不是数据问题，别去重传
        "src_objects": len(src), "dst_objects": len(dst),
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "missing": len(missing), "extra": len(extra), "size_mismatch": len(size_bad),
        "sample_same": same, "sample_diff": diff, "sample_fail": dl_fail,
        "summary": "\n".join(lines),
    }
    logger.info("[VERIFY] %s 校验%s %s", job.get("job_id"),
                "通过" if passed else "未通过", json.dumps(
                    {k: result[k] for k in ("src_objects", "dst_objects", "missing",
                                            "size_mismatch", "sample_diff")}, ensure_ascii=False))
    return result
