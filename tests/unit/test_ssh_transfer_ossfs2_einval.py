"""段1 ossfs2(FUSE) 顺序写限制修复 —— 真机 EINVAL bug 的回归钉死 + 失败明细链路。

## 这个文件在防什么（真机取证，job sgp-6796f12de0af，19.527 TiB / 75850 对象）

段1 = `ossutil cp 杭州OSS → /mnt/sgp_oss`，而 `/mnt/sgp_oss` 是 **ossfs2(FUSE) 挂载，只支持顺序写**。
ossutil 对超过 **100MiB** 的对象默认切分片、**并发 pwrite 到不同 offset** → 在该挂载点必然
`invalid argument`(EINVAL)：

  · ≤100MiB 的对象 33484 个 → 全部成功
  · >100MiB 的对象 42366 个 → 全部失败（与 ossutil report 的 done/failed 计数一字不差；
    42367 行报告里 `invalid argument` 出现 42366 次，非该错误的 Error 出现 0 次）
  · 单文件复现：现行 flags → rc=4、只留 4MiB `.temp`；换成 `--job 1 --parallel 1 --part-size 5Gi`
    → rc=0、362198872 B 完整落地
  · 高并发复验：`--job 30 --parallel 1 --part-size 5Gi` 跑 200 个大对象/77.6GiB → 164 成功、
    148MiB/s、零 EINVAL

**修法 = 强制单分片顺序写：`--parallel 1`（文件内不并发）+ `--part-size 5Gi`（顶到 ossutil 上限，
覆盖源里最大对象 0.488GiB）。跨文件并发 `--job N` 保留。**

⚠️ 任何人从 `start_stage1` 删掉 `--parallel 1` 或 `--part-size 5Gi`，>100MiB 的对象会 **100% 失败**
（本次真机就是 42366/75850 全灭）。下面 `test_start_stage1_*` 系列就是拦这件事的，别改成宽松断言。
"""
import inspect
import re

import pytest

from core.ssh_transfer import cards, engine_ssh, orchestrator as orch, paths
from core.ssh_transfer.engine_ssh import STAGE1, STAGE2
from core.ssh_transfer.orchestrator import (
    STAGE_FAILED, STAGE_STAGE1, STAGE_STAGE2, STAGE_DONE,
)


@pytest.fixture
def capture_run(monkeypatch):
    """mock engine_ssh.run，记录所有命令串 + timeout；默认返回 (0,'','')。用 box['outs'] 排队输出。"""
    box = {"cmds": [], "outs": [], "timeouts": []}

    def fake_run(cmd, *, timeout=30):
        box["cmds"].append(cmd)
        box["timeouts"].append(timeout)
        out = box["outs"].pop(0) if box["outs"] else ""
        if isinstance(out, Exception):
            raise out
        return (0, out, "")

    monkeypatch.setattr(engine_ssh, "run", fake_run)
    box["last"] = lambda: box["cmds"][-1]
    return box


# 并行段2 上线后，`failure_detail(STAGE2)` 会**先**发一条 `_stage2_failed_units` 探针
# （列 stage2.unit-*.rc 里退出码不在 {0,24} 的分片），再发原来那条 grep 探针。
# 下面这两个助手把「排在探针后面」这件事显式化，免得每个 stage2 用例各自数下标数错。
# 段1 不受影响（不发分片探针）——这也是「只在段2 多花一次 SSH」的回归钉子。
_NO_FAILED_UNITS = ""            # 分片探针无输出 = 没有失败分片 = 走原来那条 stage2.log 路径


def _stage2_outs(*outs):
    """给 STAGE2 用例排队输出：第一条固定喂给分片探针，其余按原顺序。"""
    return [_NO_FAILED_UNITS, *outs]


def _grep_cmd(box, stage):
    """取「grep 日志」那条命令（段2 时它排在分片探针之后）。"""
    return box["cmds"][1] if stage == STAGE2 else box["cmds"][0]


# ══════════════════════════════════════════════════════════════════════════════
# 一、start_stage1 flags 回归钉死（ossfs2 EINVAL 真机回归）
# ══════════════════════════════════════════════════════════════════════════════

def test_start_stage1_has_parallel_1_token(capture_run):
    """`--parallel 1` 必须作为独立 token 存在、值恰为 1（文件内不并发 → 顺序写）。

    删掉它 = 恢复文件内并发 pwrite = ossfs2 上 >100MiB 对象 100% `invalid argument`。
    """
    engine_ssh.start_stage1("sgp-abc123", source_bucket="wuji-data-tran",
                            source_prefix="team/data/")
    cmd = capture_run["last"]()
    assert re.search(r"(?<!\S)--parallel\s+1(?!\d)", cmd), cmd
    # 不能被写成 >1（那就又并发写了）
    assert not re.search(r"(?<!\S)--parallel\s+(?!1(?!\d))\d+", cmd), cmd


def test_start_stage1_has_part_size_5gi_token(capture_run):
    """`--part-size 5Gi` 必须作为独立 token 存在（顶到 ossutil 上限 → 单分片）。

    真机源里最大对象 0.488GiB（>1GiB=0、>5GiB=0），5Gi 能覆盖全部对象压成单分片。
    """
    engine_ssh.start_stage1("sgp-abc123", source_bucket="b", source_prefix="p/")
    cmd = capture_run["last"]()
    assert re.search(r"(?<!\S)--part-size\s+5Gi(?!\S)", cmd), cmd


def test_start_stage1_part_size_follows_module_constant(capture_run):
    """命令里的分片大小取自模块常量 `_OSSFS2_PART_SIZE`（改常量命令跟着改，不许两处漂移）。"""
    assert engine_ssh._OSSFS2_PART_SIZE == "5Gi"
    engine_ssh.start_stage1("sgp-abc123", source_bucket="b", source_prefix="p/")
    assert f"--part-size {engine_ssh._OSSFS2_PART_SIZE}" in capture_run["last"]()


def test_start_stage1_keeps_job_singular_with_new_flags(capture_run, monkeypatch):
    """既有 `--job`（单数）断言不被本次改动破坏：三个 flag 共存、`--jobs`(复数) 仍恒不出现。

    `--job N` = 跨文件并发（不同文件各自顺序写、互不干扰），实测 job 30 稳定 148MiB/s，
    比出错那次 75MiB/s 还快一倍 → **不为 ossfs2 降跨文件并发**。
    """
    monkeypatch.setattr(engine_ssh.settings, "SGP_OSSUTIL_JOBS", 30)
    engine_ssh.start_stage1("sgp-abc123", source_bucket="b", source_prefix="p/")
    cmd = capture_run["last"]()
    assert "--jobs" not in cmd                                   # ossutil 2.2.2 没有复数形式
    assert re.search(r"(?<!\S)--job(?!s)\s+30\b", cmd), cmd       # 跨文件并发保留
    assert re.search(r"(?<!\S)--parallel\s+1(?!\d)", cmd), cmd    # 文件内串行
    assert re.search(r"(?<!\S)--part-size\s+5Gi(?!\S)", cmd), cmd # 单分片


def test_start_stage1_all_three_flags_in_ossutil_cp_command(capture_run):
    """三个 flag 落在 ossutil cp 那条 work 命令里（不是漏在别处），且增量/断点未被挤掉。"""
    engine_ssh.start_stage1("sgp-abc123", source_bucket="b", source_prefix="p/")
    cmd = capture_run["last"]()
    i = cmd.index("ossutil cp")
    for token in ("--job ", "--parallel 1", "--part-size 5Gi", "-u", "--checkpoint-dir"):
        assert token in cmd[i:], f"{token} 不在 ossutil cp 命令段里：{cmd}"


# ══════════════════════════════════════════════════════════════════════════════
# 二、failure_detail —— 把 ossutil/rsync 真原因从十几 MB 日志里捞出来
# ══════════════════════════════════════════════════════════════════════════════

_SUMMARY = ("FinishWithError: 7 objects failed, "
            "see more information in the report file: /root/report/ossutil_20260728.report")


def test_failure_detail_summary_report_count_and_first_cause(capture_run):
    """① 正常路径：汇总行 + 报告路径 + 失败条数 + 首条 cause 都摘到。"""
    capture_run["outs"] = [
        _SUMMARY + "\n",
        "COUNT=7\ncause: connection reset by peer\n",
    ]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "FinishWithError: 7 objects failed" in detail
    assert "/root/report/ossutil_20260728.report" in detail
    assert "失败对象 7 个" in detail
    assert "首条根因：connection reset by peer" in detail
    # 非 EINVAL 时不该硬塞 >5Gi 的说明（免误导排障）
    assert "5Gi" not in detail
    # 第二条命令确实针对 report 抓条数 + 首条 cause
    second = capture_run["cmds"][1]
    assert "/root/report/ossutil_20260728.report" in second
    assert "grep -ac 'cause:'" in second
    assert "COUNT=" in second


def test_failure_detail_no_report_path_returns_summary_only(capture_run):
    """② 没有报告路径（rsync 就没有 report）→ 只回汇总行，且不发第二条命令。"""
    capture_run["outs"] = _stage2_outs("rsync error: some files could not be transferred (code 23)\n")
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert detail == "rsync error: some files could not be transferred (code 23)"
    assert len(capture_run["cmds"]) == 2, "无 .report 路径不该再发第三条命令（1=分片探针 2=grep）"
    assert "失败对象" not in detail and "首条根因" not in detail


def test_failure_detail_appends_over_5gi_note_on_einval(capture_run):
    """③ 日志含 `invalid argument` → 追加「已强制单分片、仍报错说明单对象 >5Gi」的说明。"""
    capture_run["outs"] = [
        "Error occurs: write /mnt/sgp_oss/big.bin: invalid argument\n",
    ]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "ossfs2" in detail
    assert "--part-size 5Gi" in detail and "--parallel 1" in detail
    assert ">5Gi" in detail


def test_failure_detail_einval_detected_from_report_cause(capture_run):
    """③b `invalid argument` 只出现在 report 的 cause 里（首条命令没抓到）→ 说明同样要追加。

    真机就是这个形态：进度刷屏日志尾部只有汇总行，EINVAL 全在 report 的 42366 行 cause 里。
    """
    capture_run["outs"] = [
        _SUMMARY + "\n",
        "COUNT=42366\ncause: write /mnt/sgp_oss/x.tar: invalid argument\n",
    ]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "失败对象 42366 个" in detail
    assert "invalid argument" in detail          # 首条根因原文
    assert "ossfs2" in detail and ">5Gi" in detail


def test_failure_detail_case_insensitive_einval(capture_run):
    """`Invalid Argument` 大小写变体也认（源码用 .lower()）。"""
    capture_run["outs"] = ["Error occurs: Invalid Argument\n"]
    assert "ossfs2" in engine_ssh.failure_detail("sgp-abc123", STAGE1)


def test_failure_detail_run_raises_returns_empty_and_swallows(capture_run):
    """④ 第一条命令抛异常（SSH 不通/超时）→ 返回 ""，绝不向外抛（终态判定不能被它带崩）。"""
    capture_run["outs"] = [RuntimeError("ssh down")]
    assert engine_ssh.failure_detail("sgp-abc123", STAGE1) == ""


def test_failure_detail_second_run_raises_keeps_partial(capture_run):
    """④b 第二条命令（读 report）抛异常 → best-effort 返回已拿到的汇总部分，不抛。"""
    capture_run["outs"] = [_SUMMARY + "\n", RuntimeError("report read timeout")]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "FinishWithError" in detail
    assert "失败对象" not in detail and "首条根因" not in detail


def test_failure_detail_zero_count_omits_count_line(capture_run):
    """report 里 COUNT=0（报告存在但没 cause 行）→ 不吹「失败对象 0 个」。"""
    capture_run["outs"] = [_SUMMARY + "\n", "COUNT=0\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "失败对象" not in detail


def test_failure_detail_truncated_to_detail_max(capture_run):
    """⑤ 总长 ≤ _DETAIL_MAX（要塞进飞书卡片，不能把十几 MB 日志摊上去）。

    注意真实不变量是「**≤** 上限」，不是精确等于：改成逐行截断（`_clip`）后，
    4 行 × 500 字符会先被各自截到 240 → 963 < 1200，总长兜底那刀根本用不上。
    """
    capture_run["outs"] = ["\n".join(["E" * 500] * 4) + "\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert len(detail) <= engine_ssh._DETAIL_MAX == 1200


def test_failure_detail_clips_each_line_to_line_max(capture_run):
    """逐行截断：单条超长行被截到 `_DETAIL_LINE_MAX`（240），不许一行吃掉整个预算。"""
    capture_run["outs"] = _stage2_outs("rsync error: " + "X" * 500 + "\n")
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert engine_ssh._DETAIL_LINE_MAX == 240
    assert len(detail) == engine_ssh._DETAIL_LINE_MAX
    assert detail.startswith("rsync error: ")


def test_failure_detail_long_summaries_do_not_squeeze_out_root_cause(capture_run):
    """**本次改动的真实意图**：一堆长汇总行不得把尾部最有用的三行挤没。

    旧写法（先 `join` 再整体 `[:1200]`）下，3 条 500 字符的汇总行就吃掉 1500 字符预算 →
    后面 append 的「失败对象 N 个」「首条根因」「ossfs2 说明」全被截掉，卡片上等于白给。
    逐行截断后三者都必须还在。
    """
    long_lines = [
        "FinishWithError: " + "A" * 500
        + " see more information in the report file: /root/report/x.report",
        "Error occurs: " + "B" * 500,
        "Error: " + "C" * 500,
    ]
    capture_run["outs"] = [
        "\n".join(long_lines) + "\n",
        "COUNT=42366\ncause: write /mnt/sgp_oss/x.tar: invalid argument\n",
    ]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "失败对象 42366 个" in detail, "长汇总行把失败条数挤没了"
    assert "首条根因" in detail, "长汇总行把首条根因挤没了"
    assert "ossfs2" in detail, "长汇总行把 EINVAL 说明挤没了"
    assert len(detail) <= engine_ssh._DETAIL_MAX


def test_failure_detail_probe_translates_cr_to_lf(capture_run):
    """⑥ 命令里必须有 `tr '\\r' '\\n'`（\\r→\\n）。

    这是本次排障的关键教训：ossutil 进度用 `\\r` 刷屏（单任务日志十几 MB），不 tr 的话
    `tail -n` 只能抓到一整行进度条、**看不见尾部真错误**。
    """
    capture_run["outs"] = ["\n"]
    engine_ssh.failure_detail("sgp-abc123", STAGE1)
    cmd = capture_run["cmds"][0]
    assert r"tr '\r' '\n'" in cmd, cmd
    # 且顺序是 tail → tr → grep（先归一换行再筛，不能反）
    assert cmd.index("tail -c") < cmd.index("tr '") < cmd.index("grep -aiE")


def test_failure_detail_probe_greps_expected_markers(capture_run):
    """筛选正则含 ossutil/rsync 的关键标记，且按 job/stage 定位到对应 log marker。"""
    capture_run["outs"] = _stage2_outs("\n")
    engine_ssh.failure_detail("sgp-abc123", STAGE2)
    cmd = _grep_cmd(capture_run, STAGE2)
    for token in ("FinishWithError", "Error occurs", "rsync error", "rsync:", "Error:"):
        assert token in cmd
    assert engine_ssh._marker("sgp-abc123", STAGE2, "log") in cmd


def test_failure_detail_probe_window_20kb_and_drops_partial_first_line(capture_run):
    """尾窗 20KB + `tail -n +2`。

    4KB 太小：汇总行之后 ossutil 还会刷输出，report 路径那行会被挤出窗口（原 LOW-2）。
    `tail -c` 必然切在行中/多字节中间 → 首行是乱码碎片，`tail -n +2` 丢掉它。
    """
    capture_run["outs"] = ["\n"]
    engine_ssh.failure_detail("sgp-abc123", STAGE1)
    cmd = capture_run["cmds"][0]
    assert "tail -c 20000" in cmd
    assert "tail -n +2" in cmd
    assert cmd.index("tail -c 20000") < cmd.index("tail -n +2")


def test_failure_detail_probe_has_remote_side_fallback(capture_run):
    """过滤后为空时在**远端**回退尾部非空原文（本地已拿不到原文，兜底只能在远端做）。"""
    capture_run["outs"] = ["\n"]
    engine_ssh.failure_detail("sgp-abc123", STAGE1)
    cmd = capture_run["cmds"][0]
    assert 'if [ -n "$f" ]' in cmd            # 有命中就用命中的
    assert "else" in cmd
    assert "grep -av '^[[:space:]]*$'" in cmd  # 否则回退非空原文行
    assert cmd.count("tail -3") == 2           # 命中 / 兜底 两路都只取 3 行


def test_failure_detail_ossutil_bare_error_line_is_captured(capture_run):
    """回归（前一单 sgp-841b88a7b0dd，rc=2 异地桶）：日志里只有裸 `Error:` 行也要摘到。

    那种失败没有 FinishWithError、没有 report → 原 pattern 一条都不匹配、明细为空，
    卡片又退回「stage1 退出码 2」。`Error:` 进 _FAIL_GREP 后才摘得到 AccessDenied。

    注：grep 发生在远端（这里 run 被桩掉），故「pattern 里确有 `Error:`」由
    `test_failure_detail_probe_greps_expected_markers` 断言；本例只验这类行能完整渲染进明细。
    """
    line = ("Error: operation error S3: ListObjectsV2, https response error StatusCode: 403, "
            "api error AccessDenied: Access denied")
    capture_run["outs"] = [line + "\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert "AccessDenied" in detail
    assert len(capture_run["cmds"]) == 1       # 无 report → 不发第二条命令


def test_failure_detail_run_timeouts_are_bounded(capture_run):
    """两条命令的超时收在 20s/25s 内（poll 每轮 60s，不能让取明细把轮询拖垮）。"""
    capture_run["outs"] = [_SUMMARY + "\n", "COUNT=1\ncause: x\n"]
    engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert capture_run["timeouts"] == [20, 25]


# ── report 路径白名单（日志内容是外部数据，别被骗去 grep 任意文件再贴进飞书群） ──

@pytest.mark.parametrize("bad", [
    "/root/$(id).report",          # 命令替换
    "/root/rep;rm -rf x.report",   # 分号
    "/root/a b.report",            # 空格
    "/root/`id`.report",           # 反引号
    "/root/x.report_evil",         # 后缀不闭合（.report 后还有字符 → 不是报告文件）
    "relative/x.report",           # 不以 / 开头
])
def test_failure_detail_rejects_unsafe_report_paths(capture_run, bad):
    """路径含 shell 元字符/空格/相对路径 → 不当报告路径用，绝不发第二条命令。"""
    capture_run["outs"] = [f"See more information in the report file: {bad}\n"]
    engine_ssh.failure_detail("sgp-abc123", STAGE1)
    used = " ".join(capture_run["cmds"][1:])
    assert "cause:" not in used, f"不安全路径被拿去 grep 了：{used}"


def test_failure_detail_accepts_dotted_dashed_report_path(capture_run):
    """真机形态（含 `.`/`-`/`_`/多级目录）的报告路径正常被采用。"""
    p = "/root/.ossutil_checkpoint/report/ossutil-report_2026-07-28.report"
    capture_run["outs"] = [f"see more information in the report file: {p}\n",
                           "COUNT=3\ncause: boom\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert len(capture_run["cmds"]) == 2
    assert p in capture_run["cmds"][1]
    assert "失败对象 3 个" in detail


def test_failure_detail_stage2_einval_no_ossfs2_conclusion(capture_run):
    """段2 的 EINVAL 不套段1 的 ossfs2 结论（段2 是 rsync 到泰国本地盘，原因不同）。"""
    capture_run["outs"] = _stage2_outs("rsync: write failed: invalid argument\n")
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert "ossfs2" not in detail
    assert "--part-size" not in detail
    assert "泰国" in detail and "EINVAL" in detail


def test_failure_detail_empty_log_returns_empty(capture_run):
    """日志里没有任何匹配行 → 返回空串（不编造原因）。"""
    capture_run["outs"] = ["   \n\n"]
    assert engine_ssh.failure_detail("sgp-abc123", STAGE1) == ""


# ══════════════════════════════════════════════════════════════════════════════
# 三、poll_once 终态写 error_detail（取明细失败不得带崩终态判定）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def job(monkeypatch):
    plan = paths.build_plan("oss://wuji-data-tran/team/data/")
    return orch.create_job_record(plan, open_id="u1")


def _poll_returns(monkeypatch, st):
    monkeypatch.setattr(orch.engine_ssh, "poll_stage", lambda jid, stage: st)


def test_poll_once_failed_fills_error_detail(monkeypatch, job):
    """FAILED 时 error_detail 被填进 job 并落库（卡片才有明细可显示）。"""
    job["stage"] = STAGE_STAGE1
    orch._save(job)
    _poll_returns(monkeypatch, {"status": "FAILED", "rc": 4, "error": "stage1 退出码 4"})
    seen = []
    monkeypatch.setattr(orch.engine_ssh, "failure_detail",
                        lambda jid, stage: seen.append((jid, stage)) or "失败对象 42366 个")
    out = orch.poll_once(job)
    assert out["stage"] == STAGE_FAILED
    assert out["error_detail"] == "失败对象 42366 个"
    assert seen == [(job["job_id"], STAGE1)]                      # 段号对得上
    assert orch.get_job(job["job_id"])["error_detail"] == "失败对象 42366 个"   # 真落库


def test_poll_once_failed_stage2_passes_stage2_to_detail(monkeypatch, job):
    """段2 失败时取的是 stage2 的日志（不能永远摘段1）。"""
    job["stage"] = STAGE_STAGE2
    job["stage1_rc"] = 0
    orch._save(job)
    _poll_returns(monkeypatch, {"status": "FAILED", "rc": 23, "error": "stage2 退出码 23"})
    seen = []
    monkeypatch.setattr(orch.engine_ssh, "failure_detail",
                        lambda jid, stage: seen.append(stage) or "rsync error")
    orch.poll_once(job)
    assert seen == [STAGE2]


def test_poll_once_failed_detail_exception_still_terminal(monkeypatch, job):
    """failure_detail 抛异常 → job 仍正常落 FAILED、error_detail=""、异常不外泄。

    取明细只是锦上添花，绝不能因为它失败就把整个终态判定带崩（否则任务永远卡在途）。
    """
    job["stage"] = STAGE_STAGE1
    orch._save(job)
    _poll_returns(monkeypatch, {"status": "FAILED", "rc": 4, "error": "stage1 退出码 4"})

    def boom(jid, stage):
        raise RuntimeError("ssh timeout while tailing 12MB log")
    monkeypatch.setattr(orch.engine_ssh, "failure_detail", boom)
    out = orch.poll_once(job)               # 不抛
    assert out["stage"] == STAGE_FAILED
    assert out["error"] == "stage1 退出码 4"
    assert out["error_detail"] == ""
    assert out["finished_ts"] > 0
    saved = orch.get_job(job["job_id"])
    assert saved["stage"] == STAGE_FAILED and saved["error_detail"] == ""


def test_poll_once_saves_terminal_state_before_fetching_detail(monkeypatch, job):
    """FAILED 先落库、再去摘明细（摘明细要走 SSH 最坏几十秒）。

    否则 updated_ts 停在上一轮 → 对账 stale 门认为 job 失联 → 再触发一次 refresh、白跑一遍远端 grep。
    断言方式：在 failure_detail 里回读 Redis，那时必须已经是 FAILED。
    """
    job["stage"] = STAGE_STAGE1
    orch._save(job)
    _poll_returns(monkeypatch, {"status": "FAILED", "rc": 4, "error": "stage1 退出码 4"})
    seen = {}

    def spy(jid, stage):
        seen["stage_in_redis"] = orch.get_job(jid)["stage"]
        seen["ts"] = orch.get_job(jid).get("updated_ts")
        return "detail"
    monkeypatch.setattr(orch.engine_ssh, "failure_detail", spy)
    out = orch.poll_once(job)
    assert seen["stage_in_redis"] == STAGE_FAILED, "取明细前没先落终态"
    assert out["error_detail"] == "detail"
    assert orch.get_job(job["job_id"])["error_detail"] == "detail"   # 之后再落一次带明细


def test_poll_once_done_does_not_fetch_detail(monkeypatch, job):
    """成功路径不多花一次 SSH 取明细。"""
    job["stage"] = STAGE_STAGE2
    job["stage1_rc"] = 0
    orch._save(job)
    _poll_returns(monkeypatch, {"status": "DONE", "rc": 0})
    monkeypatch.setattr(orch.engine_ssh, "failure_detail",
                        lambda *a, **k: pytest.fail("DONE 不该取失败明细"))
    assert orch.poll_once(job)["stage"] == STAGE_DONE


def test_poll_once_running_does_not_fetch_detail(monkeypatch, job):
    """在途路径同样不取明细（每 60s 一轮，不能白烧 SSH）。"""
    job["stage"] = STAGE_STAGE1
    orch._save(job)
    _poll_returns(monkeypatch, {"status": "RUNNING", "rc": None, "alive": True})
    monkeypatch.setattr(orch.engine_ssh, "failure_detail",
                        lambda *a, **k: pytest.fail("RUNNING 不该取失败明细"))
    assert orch.poll_once(job)["stage"] == STAGE_STAGE1


def test_start_stage_failure_clears_stale_detail(monkeypatch, job):
    """起任务就挂（没新日志可摘）→ 必须清掉上一轮的 error_detail。

    否则「起 stage1 失败(rc=1)」会配上上一轮那条「失败对象 42366 个 / ossfs2 说明」，
    自相矛盾、把排障带偏。
    """
    job["error_detail"] = "失败对象 42366 个（上一轮的）"
    job["error"] = "上一轮"
    orch._save(job)

    def boom(jid, **kw):
        raise orch.engine_ssh.SshTransferError("SGP 连接失败")
    monkeypatch.setattr(orch.engine_ssh, "start_stage1", boom)
    orch._start_stage(job, STAGE_STAGE1)
    assert job["stage"] == STAGE_FAILED
    assert "SGP 连接失败" in job["error"]
    assert job["error_detail"] == ""
    assert orch.get_job(job["job_id"])["error_detail"] == ""    # 落库了，卡片不会读到旧明细


def test_retry_handler_clears_stale_detail(monkeypatch):
    """重试 handler 同样清 error_detail（与 error 同步复位，新一轮失败再重新摘）。"""
    from core.feishu_bot import actions
    from core.ssh_transfer import orchestrator

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **k):
            self._t = target

        def start(self):
            self._t()

    monkeypatch.setattr(orchestrator, "run_to_completion", lambda j, **k: j)
    monkeypatch.setattr(orchestrator, "needs_approval", lambda b, ok=True: False)
    monkeypatch.setattr(actions.threading, "Thread", _SyncThread)
    stored = {"job_id": "sgp-abc123", "stage": STAGE_FAILED, "stage1_rc": 0,
              "launched": True, "created_by": "ou_c", "bytes_total": 1,
              "error": "stage1 退出码 4",
              "error_detail": "失败对象 42366 个，明细报告：/root/report/x.report",
              "source_bucket": "b", "source_prefix": "p/", "dest_rel": "p/",
              "source_uri": "oss://b/p/", "dest_uri": "wuji@h:/root/p/"}
    store = {"sgp-abc123": stored}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: store.get(jid))
    monkeypatch.setattr(orchestrator, "_save", lambda j: store.__setitem__(j["job_id"], j))

    actions._h_retry_ssh_transfer({"job_id": "sgp-abc123"}, "ou_admin", "chat", {})
    assert store["sgp-abc123"]["error"] == ""
    assert store["sgp-abc123"]["error_detail"] == ""
    assert store["sgp-abc123"]["stage1_rc"] == 0        # 段1已成功仍保留（只重段2）


# ══════════════════════════════════════════════════════════════════════════════
# 四、result_card 展示明细
# ══════════════════════════════════════════════════════════════════════════════

def _failed_job(**over):
    j = {
        "job_id": "sgp-x", "stage": "FAILED", "error": "stage1 退出码 4",
        "source_uri": "oss://wuji-data-tran/ossutil_output/",
        "dest_uri": "wuji@host:/root/data/test/",
        "created_ts": 1000, "finished_ts": 1120,
    }
    j.update(over)
    return j


def test_result_card_failed_renders_detail_lines():
    """有 error_detail → 卡里出现「明细」块，每行前缀 `· `，重试按钮仍在。"""
    detail = "失败对象 42366 个，明细报告：/root/report/x.report\n首条根因：write ...: invalid argument"
    c = cards.result_card(_failed_job(error_detail=detail))
    s = str(c)
    assert "明细" in s
    assert "· 失败对象 42366 个" in s
    assert "· 首条根因：write ...: invalid argument" in s
    assert "retry_ssh_transfer" in s           # 明细不挤掉重试按钮
    assert "schema" not in c                   # 结果卡仍是 1.0


def test_result_card_failed_without_detail_structure_unchanged():
    """无 error_detail → 卡结构与改动前一致（只多不少：元素数比有明细时少 1、无「明细」块）。"""
    plain = cards.result_card(_failed_job())
    withd = cards.result_card(_failed_job(error_detail="something"))
    assert "明细" not in str(plain)
    assert len(withd["elements"]) == len(plain["elements"]) + 1
    # 改动前的固定结构：fields / div(源+目的+失败原因) / hr / actions
    tags = [el.get("tag") for el in plain["elements"]]
    assert tags == ["div", "div", "hr", "action"]
    assert "retry_ssh_transfer" in str(plain)
    assert "stage1 退出码 4" in str(plain)


@pytest.mark.parametrize("blank", ["", "   ", "\n", "  \n \n ", None])
def test_result_card_blank_detail_treated_as_absent(blank):
    """空/空白/None 的 error_detail 不出空的「明细」块（也不 KeyError）。"""
    c = cards.result_card(_failed_job(error_detail=blank))
    assert "明细" not in str(c)
    assert len(c["elements"]) == 4


def test_result_card_detail_skips_blank_lines():
    """明细里的空行被跳过，不出 `· ` 空条目。"""
    c = cards.result_card(_failed_job(error_detail="第一行\n\n  \n第二行"))
    s = str(c)
    assert "· 第一行" in s and "· 第二行" in s
    assert "· \\n" not in s


def test_result_card_done_ignores_detail():
    """DONE 分支不显示明细（成功卡上不该出现失败细节）。"""
    c = cards.result_card({"job_id": "sgp-x", "stage": "DONE",
                           "error_detail": "失败对象 42366 个"})
    s = str(c)
    assert "明细" not in s and "42366" not in s


# ══════════════════════════════════════════════════════════════════════════════
# 五、run_to_completion 轮询上限（19.5TiB 段1 按 148MiB/s 要 ~38h，48h 会误判超时）
# ══════════════════════════════════════════════════════════════════════════════

def test_run_to_completion_max_polls_covers_seven_days():
    """max_polls 默认值 ≥ 7 天对应值（防有人手滑调回 48h=2880）。

    真机：19.5TiB 段1 按 148MiB/s 就要 ~38h，加段2 必然超 48h → 旧上限会在任务其实
    还在正常跑的时候判「轮询超时」推失败卡。
    """
    default = inspect.signature(orch.run_to_completion).parameters["max_polls"].default
    interval = inspect.signature(orch.run_to_completion).parameters["poll_interval"].default
    assert interval == 60
    assert default * interval >= 7 * 24 * 3600, f"轮询上限只有 {default * interval / 3600:.0f}h"
    assert default != 2880, "又调回 48h 了"
    assert default == 10080


def test_run_to_completion_timeout_still_reachable(monkeypatch, job):
    """上限抬高后「轮询超时」分支仍可达（显式传小 max_polls 走 for-else）。"""
    monkeypatch.setattr(orch.engine_ssh, "start_stage1", lambda jid, **kw: None)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "RUNNING", "rc": None, "alive": True})
    out = orch.run_to_completion(job, poll_interval=0, max_polls=2)
    assert out["stage"] == STAGE_FAILED
    assert "轮询超时" in out["error"]


def _timeout_error(monkeypatch, job, *, max_polls, poll_interval):
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)     # 不真睡
    monkeypatch.setattr(orch.engine_ssh, "start_stage1", lambda jid, **kw: None)
    monkeypatch.setattr(orch.engine_ssh, "stage_progress", lambda jid, st: {})
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "RUNNING", "rc": None, "alive": True})
    out = orch.run_to_completion(job, poll_interval=poll_interval, max_polls=max_polls)
    assert out["stage"] == STAGE_FAILED
    return out["error"]


def test_run_to_completion_timeout_message_uses_days_when_long(monkeypatch, job):
    """长上限的超时文案说「天」不说「168h」（用户读得懂；48h 起切换）。"""
    err = _timeout_error(monkeypatch, job, max_polls=2880, poll_interval=60)   # 48h
    assert "2天" in err
    assert "48h" not in err


def test_run_to_completion_timeout_message_uses_hours_when_short(monkeypatch, job):
    """短上限仍用小时（<48h 不硬换成 0 天）。"""
    err = _timeout_error(monkeypatch, job, max_polls=2, poll_interval=3600)    # 2h
    assert "2h" in err
    assert "天" not in err


# ── CLI status 打印明细 ───────────────────────────────────────────────────────

def test_cli_status_prints_error_detail(monkeypatch, capsys):
    """`python -m core.ssh_transfer.cli status sgp-xxx` 把明细逐行打出来（排障第三入口）。"""
    import argparse
    from core.ssh_transfer import cli

    failed = {"job_id": "sgp-abc123", "stage": STAGE_FAILED, "error": "stage1 退出码 4",
              "error_detail": "失败对象 42366 个，明细报告：/root/report/x.report\n首条根因：invalid argument",
              "source_uri": "oss://b/p/", "dest_uri": "wuji@h:/root/p/",
              "bytes_total": 0, "objects_total": 0}
    monkeypatch.setattr(cli.o, "refresh", lambda jid: failed)
    rc = cli._cmd_status(argparse.Namespace(job_id="sgp-abc123"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "明细" in out
    assert "失败对象 42366 个" in out
    assert "首条根因：invalid argument" in out


def test_cli_status_without_detail_omits_section(monkeypatch, capsys):
    """没有明细时不打空的「明细」段。"""
    import argparse
    from core.ssh_transfer import cli
    monkeypatch.setattr(cli.o, "refresh", lambda jid: {
        "job_id": "sgp-abc123", "stage": STAGE_DONE, "source_uri": "oss://b/p/",
        "dest_uri": "wuji@h:/root/p/", "bytes_total": 0, "objects_total": 0})
    cli._cmd_status(argparse.Namespace(job_id="sgp-abc123"))
    assert "明细" not in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════════
# 六、端到端串联：poll_once 写的明细最终会出现在用户收到的卡上
#     （链路：poll_once → job["error_detail"] → result_card → 查询终态补推）
# ════════════════════════════════════════════════════════════════════════════

def test_detail_flows_from_poll_once_into_result_card(monkeypatch, job):
    """闭环：段1 EINVAL 失败 → poll_once 摘明细 → result_card 上真能看到根因。"""
    job["stage"] = STAGE_STAGE1
    orch._save(job)
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "FAILED", "rc": 4, "error": "stage1 退出码 4"})
    monkeypatch.setattr(orch.engine_ssh, "failure_detail", lambda jid, st: (
        "FinishWithError: 42366 objects failed\n"
        "失败对象 42366 个，明细报告：/root/report/x.report\n"
        "首条根因：write /mnt/sgp_oss/x.tar: invalid argument"))
    out = orch.poll_once(job)
    s = str(cards.result_card(out))
    assert "明细" in s
    assert "· 失败对象 42366 个" in s
    assert "invalid argument" in s
    assert "retry_ssh_transfer" in s


def test_query_by_id_terminal_push_carries_detail(monkeypatch):
    """按 ID 查已失败的 sgp- 任务 → 补推的结果卡（真 result_card）带明细。

    真机排障入口就是这条：用户发「查询进度 sgp-xxx」，卡上直接看到 EINVAL 根因，
    不必再让人去 SGP 翻十几 MB 日志。
    """
    from core.feishu_bot import messages, messaging
    from core.ssh_transfer import orchestrator as o
    import core.dsw_scheduler as sched

    replies, pushed = [], []
    monkeypatch.setattr(messaging, "_feishu_reply", lambda mid, text: replies.append(text))
    failed = {"job_id": "sgp-aabbcc", "stage": o.STAGE_FAILED, "created_by": "ou_creator",
              "error": "stage1 退出码 4",
              "error_detail": "失败对象 42366 个，明细报告：/root/report/x.report",
              "source_uri": "oss://wuji-data-tran/ossutil_output/",
              "dest_uri": "wuji@host:/root/data/test/", "bytes_total": 0}
    monkeypatch.setattr(o, "refresh", lambda jid: failed)
    monkeypatch.setattr(sched, "_claim_dataflow_notify", lambda jid: True)
    monkeypatch.setattr(sched, "_send_card", lambda oid, chat, card: pushed.append((oid, card)))

    messages._handle_progress_query("m1", "查询进度 sgp-aabbcc", "ou_x")

    assert len(pushed) == 1
    oid, pushed_card = pushed[0]
    assert oid == "ou_creator"
    s = str(pushed_card)
    assert "· 失败对象 42366 个" in s and "明细" in s
    # 文本回执本身仍只带 error 摘要（明细走卡，避免刷屏）—— 记录现状，改动需有意为之
    assert replies and "stage1 退出码 4" in replies[0]
    assert "42366" not in replies[0]
