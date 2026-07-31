"""段2（SGP→泰国 rsync）并行多流 —— 命令生成 + 真 bash 行为 + 进度/失败明细链路。

## 为什么有这个功能（真机取证，别推翻）

SGP→泰国 RTT ~30ms，**单条 TCP 被窗口/整形卡在 26-35MB/s**；聚合吞吐随流数近线性上涨
（1流 26 / 4流 43 / 8流 78 MB/s，链路远未饱和）。在跑的 21.47TB 单子单流 27MB/s 要 9 天。
修法 = 按**源一级目录**切分，`xargs -P N -n1 sh -c <worker> _` 并行跑 N 条 rsync，
每 unit 独立 `stage2.unit-<u>.log` / `.rc`，外层聚合退出码。

## 这个文件在防什么

1. **`exit` 会吃掉 rc marker**（最阴的一条）：`_launch` 把 work 拼成 `work; echo $? > rc`，
   并行体里一旦出现裸 `exit`，`echo $? > rc` 就永远不执行 → 轮询侧看到「进程没了又没 rc」
   → `poll_stage` 判「进程异常退出」→ 一个其实只是某分片 rc=5 的任务被报成玄学崩溃。
   源码用 `(exit "$agg")`（子 shell 里 exit，只设 $? 不终止外层）。见 `test_no_bare_exit_*`。
2. **不合规目录名必须整批退回单流，不是跳过**：跳过 = 静默漏传，用户拿到「成功」却少数据。
3. **`{ ...; }` 分组**：`A && B; C` 里 `&&` 只管到第一个 `;`，不分组则泰国侧 mkdir 失败后
   find/xargs 照跑。段1 踩过同一个坑。
4. **重跑先清 `stage2.unit-*.rc`**，否则上一轮的失败码会被聚合进来、成功也报失败。
5. **单流路径（streams=1）与改动前逐字等价**（除新增的 `-s`）—— 回退开关必须是真回退。
6. 大数求和不能走 awk 算术（mawk 对 3e11 输出科学计数法）；8KB 尾窗截断不能少算。

## 怎么测的

命令串类用例 mock `engine_ssh._launch` 抓 work_cmd 做静态断言；行为类用例把同一个 work_cmd
**真的丢给本机 bash 跑**（`rsync`/`ssh` 用 PATH 桩替换，源目录/工作目录/目的根全指向 tmp），
并按 `_launch` 的原样拼成 `work; echo $? > rc` —— 这样 rc marker 语义、`&&`/`;` 优先级、
xargs 并发、grep 白名单全是真货，不是我对 shell 的想象。多处配**变异验证**（把源码那句改坏
再跑一遍，断言会挂），保证断言非空。
"""
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.ssh_transfer import engine_ssh, orchestrator as orch
from core.ssh_transfer.engine_ssh import STAGE1, STAGE2

_REPO = Path(__file__).resolve().parents[2]
NUL = bytes([0])      # NUL 字节：清单是 NUL 分隔的，但字面 NUL 会让 .py 无法解析
_BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(_BASH is None, reason="需要 bash 才能真跑生成的远端 shell")


# ══════════════════════════════════════════════════════════════════════════════
# 通用夹具
# ══════════════════════════════════════════════════════════════════════════════

def _posix(p) -> str:
    """Windows 路径 → msys/POSIX 路径（生成的 shell 只认后者）。Linux 上原样返回。"""
    s = str(p).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/{m.group(1).lower()}/{m.group(2)}" if m else s


@pytest.fixture
def capture_launch(monkeypatch):
    """抓 `_launch(job_id, stage, work_cmd)` 的 work_cmd（不真起任务）。"""
    box = {}

    def fake_launch(job_id, stage, work_cmd):
        box["job_id"], box["stage"], box["work"] = job_id, stage, work_cmd

    monkeypatch.setattr(engine_ssh, "_launch", fake_launch)
    return box


@pytest.fixture
def stage2_env(monkeypatch):
    """把段2 涉及的 settings 钉成确定值（不吃运行环境 .env）。"""
    s = engine_ssh.settings
    monkeypatch.setattr(s, "SGP_OSS_MOUNT", "/mnt/sgp_oss")
    monkeypatch.setattr(s, "SGP_WORK_DIR", "/var/run/ssh_transfer")
    monkeypatch.setattr(s, "THAI_DEST_ROOT", "/mnt/data04/thai/data")
    monkeypatch.setattr(s, "THAI_USER", "wuji")
    monkeypatch.setattr(s, "THAI_HOST", "203.0.113.9")
    monkeypatch.setattr(s, "THAI_PORT", "40002")
    monkeypatch.setattr(s, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(s, "THAI_RSYNC_BWLIMIT", "")
    monkeypatch.setattr(s, "THAI_RSYNC_STREAMS", 8)
    return s


def _work(capture_launch, monkeypatch, *, streams=8, source_prefix="team/data/", dest_rel=""):
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_STREAMS", streams)
    engine_ssh.start_stage2("sgp-abc123", source_prefix=source_prefix, dest_rel=dest_rel)
    return capture_launch["work"]


# ══════════════════════════════════════════════════════════════════════════════
# 一、_stage2_streams()：非法值 clamp（这是唯一挡住「手滑把 SGP 打满」的地方）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw,expect", [
    (8, 8),            # 默认
    (1, 1),            # 退回单流
    (2, 2),
    (10, 10),          # 上限本身（10 = 泰国 sshd 默认 MaxStartups 的未认证并发上限）
    (11, 10),          # 超上限 clamp
    (32, 10),          # 曾经的上限 32 现在也被压到 10
    (10 ** 9, 10),     # 手滑巨大值
    (0, 1),            # 0 → 单流（`or 1` 兜住）
    (-1, 1),           # 负数 → 单流
    (-999, 1),
    (None, 1),         # 未配置
    ("", 1),           # 空串
    ("0", 1),          # 字符串 0（真值但 int 后为 0）
    ("4", 4),          # 字符串数字
    (" 6 ", 6),        # 带空白
    ("abc", 1),        # 非数字 → 不炸，退单流
    ("8streams", 1),
    (4.9, 1),          # float：`int("4.9")` 抛 ValueError → fail-safe 退单流（宁慢勿错）
    ([], 1),           # 空列表（falsy）
    (["x"], 1),        # TypeError 分支
    ({"a": 1}, 1),
    (True, 1),         # bool 是 int 子类，int(True)=1
])
def test_stage2_streams_clamped(monkeypatch, raw, expect):
    """<1 归 1、>32 压 32、非数字/None 一律退单流（**绝不抛异常**：这函数在起任务路径上）。"""
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_STREAMS", raw)
    assert engine_ssh._stage2_streams() == expect


def test_stage2_streams_never_raises_on_weird_types(monkeypatch):
    """对象类型也不许把 start_stage2 带崩（起任务失败 = 整单 FAILED）。"""
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_STREAMS", object())
    assert engine_ssh._stage2_streams() == 1


# ── settings.THAI_RSYNC_STREAMS 本身（子进程里验，避免 reload 污染单例） ─────────

def _settings_probe(env_extra, expr="settings.THAI_RSYNC_STREAMS"):
    env = dict(os.environ)
    env.pop("THAI_RSYNC_STREAMS", None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(_REPO)
    return subprocess.run(
        [sys.executable, "-c",
         f"from config.settings import settings; print(repr({expr}))"],
        cwd=str(_REPO), env=env, capture_output=True, text=True)


def test_settings_streams_default_is_8():
    r = _settings_probe({})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "'8'"          # 原始字符串，理由见 type_is_str 那条


def test_settings_streams_reads_env_verbatim():
    r = _settings_probe({"THAI_RSYNC_STREAMS": "3"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "'3'"


def test_settings_streams_type_is_str_not_int():
    """刻意存**原始字符串**、不在 import 期 int()（auditor LOW-7 的修复）。

    `int(os.environ.get(...))` 在类体里求值，`.env` 写成空值或非数字会让 config.settings
    整个 import 失败 → **整个 bot 起不来**，而这只是个性能旋钮。转换与钳位下沉到
    engine_ssh._stage2_streams()（非法值退回单流）。
    """
    r = _settings_probe({"THAI_RSYNC_STREAMS": "5"},
                        expr="type(settings.THAI_RSYNC_STREAMS).__name__")
    assert r.stdout.strip() == "'str'"


def test_settings_bad_env_no_longer_breaks_import():
    """回归钉子：曾经 `.env` 里写 `THAI_RSYNC_STREAMS=abc`（或留空）会让 config.settings
    整个 import 失败、**bot 起不来**（auditor LOW-7）。现在必须只是退回单流。
    `.env` 留空是最容易手滑的形态，一并钉住。
    """
    for bad in ("abc", "", "4.9", "  "):
        r = _settings_probe({"THAI_RSYNC_STREAMS": bad})
        assert r.returncode == 0, f"env={bad!r} 让 settings import 失败了：{r.stderr}"


# ══════════════════════════════════════════════════════════════════════════════
# 二、单流路径（streams=1）与改动前逐字等价（除新增 `-s`）
# ══════════════════════════════════════════════════════════════════════════════

def _expected_cleanup(job_id="sgp-abc123", work_dir="/var/run/ssh_transfer"):
    """单流/并行两条分支都会先跑的清理前缀（auditor MED-3）。

    为什么单流也要清：并行跑过一轮后把 STREAMS 调回 1（或触发白名单回退），残留的
    unit rc 会让失败明细误报成「失败分片 shard_003…」、残留的 unit 日志会让
    `_stage2_parallel_progress` 误判成并行��把进度冻结在上一轮的字节数。
    为什么在 mkdir **之前**：放在 `mkdir &&` 之后的话，泰国侧 mkdir 挂了会短路跳过清理。
    """
    d = f"{work_dir}/{job_id}"
    return (f"rm -f {d}/stage2.unit-*.rc {d}/stage2.unit-*.log {d}/stage2.units")


def _expected_single(*, with_s: bool, flags_extra="", src="/mnt/sgp_oss/team/data/",
                     dest="/mnt/data04/thai/data/team/data/"):
    ssh_e = ("ssh -p 40002 -o BatchMode=yes "
             "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15")
    flags = "-a -s --info=progress2" if with_s else "-a --info=progress2"
    if flags_extra:
        flags += " " + flags_extra
    # 路径不带特殊字符时 shlex.quote **不会**加引号，别在期望值里硬写引号
    # （`-e '<ssh_e>'` 里的引号是真的：ssh_e 含空格）。
    mkdir = ("ssh -p 40002 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
             f"wuji@203.0.113.9 mkdir -p {dest}")
    return (f"{_expected_cleanup()}; {mkdir} && "
            f"rsync {flags} -e '{ssh_e}' {src} wuji@203.0.113.9:{dest}")


def test_single_stream_command_is_exact(stage2_env, capture_launch, monkeypatch):
    """streams=1 → 逐字等于「老命令 + `-s`」。这是回退开关是真回退的证明。"""
    work = _work(capture_launch, monkeypatch, streams=1)
    assert work == _expected_single(with_s=True)


def test_single_stream_only_delta_vs_pre_change_is_dash_s(stage2_env, capture_launch, monkeypatch):
    """把新命令里唯一那个 `-s` 抠掉，必须与改动前的命令**一字不差**。"""
    work = _work(capture_launch, monkeypatch, streams=1)
    assert work.replace(" -s ", " ", 1) == _expected_single(with_s=False)


@pytest.mark.parametrize("raw_streams", [1, 0, -3, None, "", "abc", "1"])
def test_single_stream_has_no_parallel_machinery(stage2_env, capture_launch, monkeypatch, raw_streams):
    """所有 clamp 到 1 的取值都走纯单流：无 xargs / find / 聚合 / sweep。

    注意 `stage2.unit-*` / `stage2.units` **会**出现在单流命令里 —— 那是 cleanup 前缀
    （auditor MED-3 要求单流分支也清残留），不是并行机制。所以只断言真正的并行构件。
    """
    work = _work(capture_launch, monkeypatch, streams=raw_streams)
    for token in ("xargs", "find ", "agg=", "miss=", "--info=stats1", "read -r -d"):
        assert token not in work, f"单流路径不该出现 {token}: {work}"
    # cleanup 前缀必须在，且在 mkdir 之前
    assert work.startswith("rm -f "), work
    assert work.index("rm -f ") < work.index("mkdir -p")


def test_single_stream_dest_rel_still_honoured(stage2_env, capture_launch, monkeypatch):
    """B 语义（内容铺进 dest_rel）在单流下不回归。"""
    work = _work(capture_launch, monkeypatch, streams=1,
                 source_prefix="ossutil_output/", dest_rel="test/")
    assert work == _expected_single(with_s=True, src="/mnt/sgp_oss/ossutil_output/",
                                    dest="/mnt/data04/thai/data/test/")


def test_parallel_fallback_single_is_identical_to_stream1_command(stage2_env, capture_launch,
                                                                  monkeypatch):
    """并行体里的「退回单流」那条 rsync，与 streams=1 时那条**逐字相同**（同一个 `single`）。"""
    single_work = _work(capture_launch, monkeypatch, streams=1)
    rsync_part = single_work.split(" && ", 1)[1]
    par_work = _work(capture_launch, monkeypatch, streams=8)
    assert rsync_part in par_work


# ══════════════════════════════════════════════════════════════════════════════
# 三、`-s`（--secluded-args）：两种模式都要有
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("streams", [1, 8])
def test_dash_s_present_as_standalone_token(stage2_env, capture_launch, monkeypatch, streams):
    """`-s` 必须是独立 token（不是 `-as`/`--sudo` 之类误匹配）。

    没有它，目的路径会经泰国那边的 shell 解析 —— 并行版的路径含来自文件系统的目录名。
    """
    work = _work(capture_launch, monkeypatch, streams=streams)
    assert re.search(r"rsync -a -s --info=progress2", work), work
    assert " -s " in work


def test_dash_s_in_parallel_worker_too(stage2_env, capture_launch, monkeypatch):
    """并行 worker 里那条 rsync（不是只有 fallback 那条）也带 `-s`。"""
    work = _work(capture_launch, monkeypatch, streams=8)
    worker = work[work.index("sh -c"):]
    assert "rsync -a -s --info=progress2" in worker


@pytest.mark.parametrize("streams", [1, 8])
def test_bwlimit_and_sudo_flags_propagate(stage2_env, capture_launch, monkeypatch, streams):
    """既有 flag（限速 / 方案B sudo）在两种模式下都还在，且落在 rsync flags 段。"""
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_BWLIMIT", "40000")
    monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_SUDO", "true")
    work = _work(capture_launch, monkeypatch, streams=streams)
    assert "--bwlimit=40000" in work
    # 必须整体带引号：裸展开会被 shell 拆成 `--rsync-path=sudo` + `rsync`，后者被当第二个
    # 源路径 → 方案 B 一开就必挂（既存缺陷，随并行改造一起修的）。
    assert "'--rsync-path=sudo rsync'" in work
    # 并行时 worker / 收尾 sweep / fallback 都要带（否则分片不限速、不提权）
    if streams > 1:
        assert work.count("--bwlimit=40000") == 3
        assert work.count("'--rsync-path=sudo rsync'") == 3


# ══════════════════════════════════════════════════════════════════════════════
# 四、并行命令形状（静态）
# ══════════════════════════════════════════════════════════════════════════════

def test_parallel_no_bare_exit_only_subshell_exit(stage2_env, capture_launch, monkeypatch):
    """**最重要的一条**：并行体里 `exit` 只能以 `(exit "$agg")` 出现。

    `_launch` 拼的是 `work; echo $? > rc`。一个裸 `exit` 就让 rc marker 永远不落地，
    `poll_stage` 会读到「无 rc + 进程已死」→ 报「进程异常退出」，把「某分片 rc=5」
    这种可诊断的失败变成玄学。变异验证见 `test_bash_bare_exit_loses_rc_marker`。
    """
    work = _work(capture_launch, monkeypatch, streams=8)
    hits = re.findall(r"\bexit\b", work)
    assert len(hits) == 1, f"并行体出现了 {len(hits)} 个 exit：{work}"
    assert '(exit "$agg")' in work


def test_parallel_grouping_braces_after_mkdir(stage2_env, capture_launch, monkeypatch):
    """`mkdir && { ...; }`：不分组的话 `&&` 只管到第一个 `;`，mkdir 失败也会往下跑。"""
    work = _work(capture_launch, monkeypatch, streams=8)
    assert " && { " in work
    assert work.rstrip().endswith("; }")
    # 最前面是 cleanup（auditor MED-3：放在 `mkdir &&` 之后的话 mkdir 挂了会短路跳过清理），
    # 紧接着才是 mkdir。
    assert work.startswith("rm -f ")
    assert work.index("rm -f ") < work.index("mkdir -p") < work.index(" && { ")


def test_parallel_clears_stale_unit_rc_before_find(stage2_env, capture_launch, monkeypatch):
    """重跑先清残留 marker，且必须排在 **mkdir 之前**（auditor MED-3）。

    放在 `mkdir &&` 之后：泰国侧 mkdir 挂了会短路跳过清理 → 卡片拿旧 unit rc 报
    「失败分片 shard_003…」，把真原因(rc=255)完全盖住。
    三类都要删：rc（否则旧失败码被聚合）、log（否则进度探针误判成并行、冻结在上一轮字节数）、
    units（否则聚合按上一轮的清单核对）。
    """
    d = "/var/run/ssh_transfer/sgp-abc123"
    work = _work(capture_launch, monkeypatch, streams=8)
    for pat in (f"{d}/stage2.unit-*.rc", f"{d}/stage2.unit-*.log", f"{d}/stage2.units"):
        assert pat in work, pat
    assert work.index("rm -f ") < work.index("mkdir -p") < work.index("find ") < work.index("xargs ")


def test_parallel_xargs_shape(stage2_env, capture_launch, monkeypatch):
    """xargs 形状：`-0 -a units -P N -n1 sh -c <worker> _`（`_` 占 $0，目录名才是 $1）。

    `-0` 是 MED-5 的修复配套：清单改 NUL 分隔后必须用 -0 读，否则含换行的目录名会被
    拆成两条（真目录静默漏传 + 两个幽灵路径）。
    """
    work = _work(capture_launch, monkeypatch, streams=8)
    assert re.search(r"xargs -0 -a \S+/stage2\.units -P 8 -n1 sh -c '", work), work
    assert re.search(r"' _ ; xrc=\$\?", work), work        # xargs 自身退出码要被接住


@pytest.mark.parametrize("streams,expect", [(2, 2), (8, 8), (10, 10), (32, 10), (99, 10)])
def test_parallel_p_value_follows_clamped_streams(stage2_env, capture_launch, monkeypatch,
                                                  streams, expect):
    """`-P N` 与 clamp 后的流数一致，提示文案里的流数也一致（两处不许漂移）。"""
    work = _work(capture_launch, monkeypatch, streams=streams)
    assert f"-P {expect} -n1" in work
    assert f"并行 {expect} 流" in work


def test_parallel_find_is_one_level_dirs_only(stage2_env, capture_launch, monkeypatch):
    """切分粒度 = 源一级目录（`-mindepth 1 -maxdepth 1 -type d`，`%f` 只取 basename）。"""
    work = _work(capture_launch, monkeypatch, streams=8)
    # 清单必须 NUL 分隔：`%f\n` 遇到名字里带换行的目录会把一条打成两行，两半各自都能过
    # 白名单 → 不触发回退；若两半恰好都是真实目录名，就是真目录静默漏传 + rc=0
    # （Linux 上实测可复现，auditor MED-5）。
    assert '-mindepth 1 -maxdepth 1 -type d -printf "%f\\0"' in work
    assert '-printf "x\\n"' in work                    # 条数交叉核对（与名字无关，天然换行安全）
    assert "nd=" in work and "nu=" in work
    assert "frc=" in work                              # find 退出码：部分失败会产出部分清单


def test_parallel_whitelist_regex_embedded(stage2_env, capture_launch, monkeypatch):
    r"""白名单用模块常量 `_UNIT_SAFE_GREP`（两处不许漂移），`grep -zqvE` 反向判定。

    `-z`：清单是 NUL 分隔的，且 `-z` 下 `$` 锚 NUL 不锚换行 —— 含换行的目录名会被拒
    （正是 MED-5 要的行为）。`LC_ALL=C` 求 ERE 字符类的确定性。
    Python 侧另有 `_UNIT_SAFE_PY`（`\A..\Z` + fullmatch）：`$` 会放过结尾换行。
    """
    work = _work(capture_launch, monkeypatch, streams=8)
    assert engine_ssh._UNIT_SAFE_GREP == "^[A-Za-z0-9._-]+$"
    assert f"grep -zqvE '{engine_ssh._UNIT_SAFE_GREP}'" in work
    assert "LC_ALL=C grep" in work
    assert engine_ssh._UNIT_SAFE_PY.fullmatch("shard_000")
    assert not engine_ssh._UNIT_SAFE_PY.fullmatch("shard_000\n")   # `$` 会放过它，`\Z` 不会


def test_parallel_worker_uses_variable_not_string_interpolation(stage2_env, capture_launch,
                                                                monkeypatch):
    """目录名只以 `"$u"` 变量展开出现（4 处：源/目的/log/rc），绝不被拼进任何 shell 字符串。"""
    work = _work(capture_launch, monkeypatch, streams=8)
    worker = work[work.index("u=\"$1\""):work.index("' _ ;")]
    assert worker.count('"$u"') == 4, worker
    assert "$(" not in worker and "`" not in worker    # 无命令替换面


def test_parallel_worker_writes_per_unit_log_and_rc(stage2_env, capture_launch, monkeypatch):
    """每 unit 各自 log + rc（失败明细与进度都依赖这两个 marker 的命名约定）。"""
    work = _work(capture_launch, monkeypatch, streams=8)
    assert 'stage2.unit-"$u".log 2>&1;' in work
    assert 'echo $? > /var/run/ssh_transfer/sgp-abc123/stage2.unit-"$u".rc' in work


def test_parallel_aggregate_treats_24_as_acceptable(stage2_env, capture_launch, monkeypatch):
    """聚合规则：0 忽略、24 仅在还没真失败时置位、空/非数字按失败、其余覆盖为真失败。

    且**以 units 清单逐条核对**、不是遍历"存在的 rc 文件"（auditor HIGH-2）：
    50 个 unit 只落 40 个 rc 且都为 0 时，后者会聚合出 0 → 报成功 → 10 个分片从未传输、
    零告警。缺 rc 一律 `agg=1` 并打印 miss 计数。
    """
    work = _work(capture_launch, monkeypatch, streams=8)
    assert '24) [ "$agg" -eq 0 ] && agg=24 ;;' in work
    assert '""|*[!0-9]*) agg=1 ;;' in work                 # rc 写坏也算失败
    assert 'while IFS= read -r -d "" u; do' in work        # 按清单逐条核对
    assert 'miss=$((miss+1)); agg=1' in work               # 缺 rc = 失败
    assert '(exit "$agg")' in work and "; exit " not in work   # 绝不能裸 exit（会吃掉 rc marker）
    # 与 python 侧 rc 语义一致
    assert engine_ssh._rc_ok(STAGE2, 0) and engine_ssh._rc_ok(STAGE2, 24)
    assert not engine_ssh._rc_ok(STAGE2, 5)


def test_parallel_fallback_message_mentions_single_stream(stage2_env, capture_launch, monkeypatch):
    """退回单流要在日志里说清楚（否则运维看不出这单为什么慢）。"""
    work = _work(capture_launch, monkeypatch, streams=8)
    assert "退回单流" in work


def test_parallel_dest_rel_used_in_units(stage2_env, capture_launch, monkeypatch):
    """dest_rel（B 语义）在并行路径同样生效：mkdir、worker 目的、fallback 三处都是它。"""
    work = _work(capture_launch, monkeypatch, streams=8,
                 source_prefix="ossutil_output/", dest_rel="test/")
    # shlex.quote 对无特殊字符的路径不加引号
    assert "mkdir -p /mnt/data04/thai/data/test/" in work
    assert 'wuji@203.0.113.9:/mnt/data04/thai/data/test/"$u"/' in work
    assert "/mnt/sgp_oss/ossutil_output/" in work


# ══════════════════════════════════════════════════════════════════════════════
# 五、真 bash 行为（rsync/ssh 打桩，源/工作/目的目录全在 tmp）
# ══════════════════════════════════════════════════════════════════════════════

_RSYNC_STUB = """#!/usr/bin/env bash
printf 'RSYNC\\t%s\\n' "$*" >> "$STUB_LOG"
last=""; for a in "$@"; do last="$a"; done
if [ -n "$STUB_DIR" ]; then
  mkdir -p "$STUB_DIR/live"
  : > "$STUB_DIR/live/$$"
  ls "$STUB_DIR/live" | wc -l >> "$STUB_DIR/peak"
fi
sleep "${STUB_SLEEP:-0.02}"
[ -n "$STUB_DIR" ] && rm -f "$STUB_DIR/live/$$"
IFS=',' read -ra pairs <<< "${STUB_RC:-}"
for p in "${pairs[@]}"; do
  [ -z "$p" ] && continue
  k="${p%%=*}"; v="${p##*=}"
  case "$last" in *"/$k/") exit "$v";; esac
done
exit 0
"""

_SSH_STUB = """#!/usr/bin/env bash
printf 'SSH\\t%s\\n' "$*" >> "$STUB_LOG"
exit "${STUB_SSH_RC:-0}"
"""


class _Result:
    def __init__(self, work, proc, rc_marker, job_dir, stub_log, peak):
        self.work = work
        self.proc = proc
        self.rc_marker = rc_marker          # stage2.rc 内容（"<MISSING>" = 没写）
        self.job_dir = job_dir
        self.stub_log = stub_log
        self.peak = peak                    # 观测到的最大并发 rsync 数

    @property
    def rsync_args(self):
        return [ln.split("\t", 1)[1] for ln in self.stub_log.splitlines()
                if ln.startswith("RSYNC\t")]

    @property
    def rsync_targets(self):
        """每次 rsync 的 (源, 目的) —— 用来证明「谁被传了、谁没被传」。"""
        out = []
        for a in self.rsync_args:
            parts = a.split()
            out.append((parts[-2], parts[-1]))
        return out

    @property
    def ssh_calls(self):
        return [ln.split("\t", 1)[1] for ln in self.stub_log.splitlines()
                if ln.startswith("SSH\t")]

    @property
    def unit_files(self):
        return sorted(p for p in os.listdir(self.job_dir) if p.startswith("stage2.unit-"))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """真跑生成的 shell：桩掉 rsync/ssh，把 SGP/泰国路径全指到 tmp。"""
    root = tmp_path
    (root / "bin").mkdir()
    for name, body in (("rsync", _RSYNC_STUB), ("ssh", _SSH_STUB)):
        f = root / "bin" / name
        f.write_text(body, encoding="utf-8", newline="\n")
        f.chmod(0o755)
    src_root = root / "src"
    (src_root / "team" / "data").mkdir(parents=True)
    (root / "dst").mkdir()
    (root / "work").mkdir()

    P = _posix(root)
    s = engine_ssh.settings
    monkeypatch.setattr(s, "SGP_OSS_MOUNT", P + "/src")
    monkeypatch.setattr(s, "SGP_WORK_DIR", P + "/work")
    monkeypatch.setattr(s, "THAI_DEST_ROOT", P + "/dst")
    monkeypatch.setattr(s, "THAI_USER", "wuji")
    monkeypatch.setattr(s, "THAI_HOST", "203.0.113.9")
    monkeypatch.setattr(s, "THAI_PORT", "40002")
    monkeypatch.setattr(s, "THAI_RSYNC_SUDO", "false")
    monkeypatch.setattr(s, "THAI_RSYNC_BWLIMIT", "")
    monkeypatch.setattr(s, "THAI_RSYNC_STREAMS", 8)

    box = {"work": None}
    monkeypatch.setattr(engine_ssh, "_launch",
                        lambda j, st, w: box.__setitem__("work", w))

    class SB:
        # 不能写 `root = root`：类体里同名赋值会让编译器改用 LOAD_NAME（跳过外层函数作用域），
        # 读不到闭包里的 root → NameError。引用不同名的 tmp_path 才走 LOAD_CLASSDEREF。
        root = tmp_path
        posix = P
        src = src_root / "team" / "data"

        @staticmethod
        def make(dirs=(), files=()):
            for d in dirs:
                (SB.src / d).mkdir(parents=True, exist_ok=True)
            for f in files:
                (SB.src / f).write_text("x", encoding="utf-8")

        @staticmethod
        def stale_rc(job_id="sgp-t1", **units):
            jd = root / "work" / job_id
            jd.mkdir(parents=True, exist_ok=True)
            for u, rc in units.items():
                (jd / f"stage2.unit-{u}.rc").write_text(str(rc), encoding="utf-8")

        @staticmethod
        def run(job_id="sgp-t1", *, streams=None, dest_rel="", env=None, mutate=None,
                source_prefix="team/data/"):
            if streams is not None:
                monkeypatch.setattr(engine_ssh.settings, "THAI_RSYNC_STREAMS", streams)
            engine_ssh.start_stage2(job_id, source_prefix=source_prefix, dest_rel=dest_rel)
            work = box["work"]
            if mutate:
                work = mutate(work)
            jd = root / "work" / job_id
            jd.mkdir(parents=True, exist_ok=True)
            rc_path = f"{P}/work/{job_id}/stage2.rc"
            inner = f"{work}; echo $? > {rc_path}"          # 与 _launch 的拼法一致
            e = dict(os.environ)
            e["PATH"] = P + "/bin" + os.pathsep + e["PATH"]
            e["STUB_LOG"] = P + f"/stub-{job_id}.log"
            e["STUB_DIR"] = P + f"/stubdir-{job_id}"
            e.update(env or {})
            proc = subprocess.run([_BASH, "-c", inner], env=e,
                                  capture_output=True, text=True, timeout=180)
            rcf = jd / "stage2.rc"
            marker = rcf.read_text(encoding="utf-8").strip() if rcf.exists() else "<MISSING>"
            log_p = root / f"stub-{job_id}.log"
            log = log_p.read_text(encoding="utf-8", errors="replace") if log_p.exists() else ""
            peak_p = root / f"stubdir-{job_id}" / "peak"
            peak = max((int(x) for x in peak_p.read_text().split() if x.strip()), default=0) \
                if peak_p.exists() else 0
            return _Result(work, proc, marker, str(jd), log, peak)

    return SB


# ── 5.1 正常并行 ──────────────────────────────────────────────────────────────

@needs_bash
def test_bash_five_dirs_parallel_success(sandbox):
    """5 个正常目录 + 8 流 → rc=0、rsync 恰 5 次（每目录一条）、每 unit 一 log 一 rc。"""
    dirs = [f"shard_{i:03d}" for i in range(5)]
    sandbox.make(dirs=dirs)
    r = sandbox.run()
    assert r.rc_marker == "0", r.proc.stderr
    # 5 条分片 + 1 条收尾 sweep（HIGH-1：补顶层散文件 + 兜底漏跑）
    assert len(r.rsync_args) == 6, r.rsync_args
    sweep = [a for a in r.rsync_args if "--info=stats1" in a]
    assert len(sweep) == 1, f"没跑收尾 sweep：{r.rsync_args}"
    units = [t for t in r.rsync_targets if t[0] != f"{sandbox.posix}/src/team/data/"]
    got_src = sorted(t[0] for t in units)
    assert got_src == sorted(f"{sandbox.posix}/src/team/data/{d}/" for d in dirs)
    got_dst = sorted(t[1] for t in units)
    assert got_dst == sorted(f"wuji@203.0.113.9:{sandbox.posix}/dst/team/data/{d}/" for d in dirs)
    assert r.unit_files == sorted(
        [f"stage2.unit-{d}.log" for d in dirs] + [f"stage2.unit-{d}.rc" for d in dirs])
    assert "并行 8 流，切分 5 个一级目录" in r.proc.stdout
    assert r.ssh_calls and "mkdir -p" in r.ssh_calls[0]     # 泰国侧先建目录


@needs_bash
def test_bash_units_really_run_concurrently(sandbox):
    """5 个分片必须**同时在跑**（peak 并发 = 5），不是串行 —— 整个功能的存在理由。

    用「每个桩启动时数一下还有几个桩活着」度量，比掐墙钟稳（msys 进程启动开销大）。
    """
    sandbox.make(dirs=[f"s{i}" for i in range(5)])
    r = sandbox.run(env={"STUB_SLEEP": "1.2"})
    assert r.rc_marker == "0"
    assert r.peak == 5, f"peak={r.peak}，说明没并发起来"


@needs_bash
def test_bash_p_flag_caps_concurrency(sandbox):
    """`-P 2` 真的把并发压到 2（防「N 只写进文案、实际全放开」）。"""
    sandbox.make(dirs=[f"s{i}" for i in range(6)])
    r = sandbox.run(streams=2, env={"STUB_SLEEP": "0.4"})
    assert r.rc_marker == "0"
    # 6 个分片 + 1 条收尾 sweep（sweep 是串行跑在分片之后的，不占并发）
    assert len(r.rsync_args) == 7, r.rsync_args
    assert len([a for a in r.rsync_args if "--info=stats1" in a]) == 1
    assert r.peak <= 2, f"peak={r.peak}，-P 2 没生效"


# ── 5.2 退出码聚合 ────────────────────────────────────────────────────────────

@needs_bash
def test_bash_unit_rc24_aggregates_to_24(sandbox):
    """某分片 rc=24（源文件传输中消失）→ 聚合 24 → 上层按成功处理。"""
    sandbox.make(dirs=["a", "b"])
    r = sandbox.run(env={"STUB_RC": "a=24"})
    assert r.rc_marker == "24"
    assert engine_ssh._rc_ok(STAGE2, 24) is True


@needs_bash
def test_bash_unit_rc5_aggregates_to_5(sandbox):
    """某分片 rc=5 → 聚合 5 → 整单 FAILED（不许被 0 冲掉）。"""
    sandbox.make(dirs=["a", "b"])
    r = sandbox.run(env={"STUB_RC": "a=5"})
    assert r.rc_marker == "5"
    assert engine_ssh._rc_ok(STAGE2, 5) is False


@needs_bash
@pytest.mark.parametrize("rcmap", ["a=24,b=5", "a=5,b=24"])
def test_bash_real_failure_beats_24_regardless_of_order(sandbox, rcmap):
    """24 与真失败同时出现时，无论遍历顺序都取真失败（24 只在 agg 仍为 0 时置位）。"""
    sandbox.make(dirs=["a", "b", "c"])
    r = sandbox.run(env={"STUB_RC": rcmap})
    assert r.rc_marker == "5"


@needs_bash
def test_bash_all_success_is_zero(sandbox):
    sandbox.make(dirs=["a", "b", "c"])
    assert sandbox.run().rc_marker == "0"


@needs_bash
def test_bash_stale_unit_rc_is_cleared_before_run(sandbox):
    """重跑：上一轮残留的 `stage2.unit-gone.rc=5` 必须先被清掉，否则本轮成功也被判失败。"""
    sandbox.make(dirs=["a"])
    sandbox.stale_rc("sgp-t1", gone=5)
    r = sandbox.run()
    assert r.rc_marker == "0", "上一轮的失败码漏进了本轮聚合"
    assert "stage2.unit-gone.rc" not in r.unit_files


@needs_bash
def test_cleanup_removes_all_three_marker_kinds(sandbox):
    """cleanup 必须删 rc / log / units 三类，且三类的**承重程度不同**（别想当然）。

    我一开始以为「去掉 cleanup 会让残留的旧 units 清单导致误判失败」，实测 rc=0 —— 因为
    `find ... > units` 每轮都会**覆盖**它。顺着查下去，聚合改成按 units 清单逐条核对
    （auditor HIGH-2）之后，残留的 rc 也不再会被读到（不在清单里的 rc 根本不遍历）。

    所以现在真正**承重**的只有 unit **日志**：`_stage2_parallel_progress` 和
    `failure_detail` 是按 glob 读 `stage2.unit-*.log` 的 ——
      · 并行跑过一轮后退回单流 → 残留日志让探针误判成「并行」，进度冻结在上一轮字节数；
      · 上一轮的分片日志会被求和进本轮进度（虚高）。
    rc/units 两类属纵深防御（覆盖/不遍历各自兜了一层），保留但不假装它们承重。
    """
    sandbox.make(dirs=["a"])
    jd = sandbox.root / "work" / "sgp-t1"
    jd.mkdir(parents=True, exist_ok=True)
    for stale in ("stage2.unit-gone.rc", "stage2.unit-gone.log", "stage2.units"):
        (jd / stale).write_text("stale", encoding="utf-8")

    r = sandbox.run()
    assert r.rc_marker == "0"
    # 三类残留都不该活下来（尤其 log —— 它是唯一承重的那类）
    assert "stage2.unit-gone.log" not in r.unit_files, "残留 unit 日志会污染进度/失败明细"
    assert "stage2.unit-gone.rc" not in r.unit_files

    # 变异：去掉 cleanup → 残留日志活下来（证明这条断言非空）
    for stale in ("stage2.unit-gone.rc", "stage2.unit-gone.log"):
        (jd / stale).write_text("stale", encoding="utf-8")
    r2 = sandbox.run(
        mutate=lambda w: re.sub(r"^rm -f \S+\.rc \S+\.log \S+units; ", "", w, count=1))
    assert "stage2.unit-gone.log" in r2.unit_files, "变异没生效，上面的断言等于没写"


@needs_bash
def test_bash_rerun_is_repeatable(sandbox):
    """同一 job 连跑两轮都干净成功（幂等，rsync -a 侧的跳过由真 rsync 保证）。"""
    sandbox.make(dirs=["a", "b"])
    assert sandbox.run().rc_marker == "0"
    r2 = sandbox.run()
    assert r2.rc_marker == "0"
    assert len([x for x in r2.unit_files if x.endswith(".rc")]) == 2


# ── 5.3 rc marker 必须落地（`exit` 陷阱） ──────────────────────────────────────

@needs_bash
@pytest.mark.parametrize("env", [{}, {"STUB_RC": "a=5"}, {"STUB_RC": "a=24"},
                                 {"STUB_SSH_RC": "1"}])
def test_bash_rc_marker_always_written(sandbox, env):
    """成功/24/真失败/mkdir 失败四种收场，`stage2.rc` 都必须存在（轮询侧全靠它）。"""
    sandbox.make(dirs=["a", "b"])
    r = sandbox.run(env=env)
    assert r.rc_marker != "<MISSING>", r.proc.stderr


@needs_bash
def test_bash_bare_exit_loses_rc_marker(sandbox):
    """**变异验证**：把 `(exit "$agg")` 改成裸 `exit "$agg"` → rc marker 直接没了。

    这正是源码那句注释在防的事，也证明 `test_parallel_no_bare_exit_only_subshell_exit`
    不是空断言。marker 缺失时 `poll_stage` 会走 DEAD 分支报「进程异常退出（无退出码 marker）」。
    """
    sandbox.make(dirs=["a"])
    r = sandbox.run(env={"STUB_RC": "a=5"},
                    mutate=lambda w: w.replace('(exit "$agg")', 'exit "$agg"'))
    assert r.rc_marker == "<MISSING>"
    assert r.proc.returncode == 5           # 退出码是对的，但 marker 丢了 → 轮询侧误判


# ── 5.4 白名单 / 退回单流 ─────────────────────────────────────────────────────

@needs_bash
@pytest.mark.parametrize("bad", [
    "bad name",          # 空格（xargs 会切成两个参数）
    "a;rm -rf x",        # 命令分隔符
    "a$(id)",            # 命令替换
    "a`id`",             # 反引号
    "a'q",               # 单引号
    'a"q',               # 双引号
    "a|b",               # 管道
    "a&b",               # 后台
    "a>b",               # 重定向
    "a*b",               # 通配
    # 反斜杠（xargs 会吃掉）。Windows 上 `\` 是路径分隔符，mkdir 会建成嵌套目录 `a/b`，
    # 一级目录变成合法的 `a` → 走并行，测不到本意。Linux CI 上才有效。
    pytest.param("a\\b", marks=pytest.mark.skipif(
        os.name == "nt", reason="Windows 无法创建含反斜杠的目录名（会被当路径分隔符）")),
    "中文目录",           # 非 ASCII
    "a:b",               # 冒号（rsync 会当成 host:path！）
])
def test_bash_unsafe_dirname_falls_back_to_single_stream(sandbox, bad):
    """任一目录名不合白名单 → **整批**退回单流：只跑一条覆盖整个源的 rsync，零 unit 文件。

    关键不变量是「不漏传」：绝不是跳过那个目录（跳过 = 用户拿到「成功」却少数据）。
    """
    try:
        sandbox.make(dirs=["good", bad])
    except OSError:
        pytest.skip(f"本机文件系统不允许创建目录名 {bad!r}")
    r = sandbox.run()
    assert "退回单流" in r.proc.stdout
    assert len(r.rsync_args) == 1
    src, dst = r.rsync_targets[0]
    assert src == f"{sandbox.posix}/src/team/data/"      # 整个源，一个目录都没落下
    assert dst == f"wuji@203.0.113.9:{sandbox.posix}/dst/team/data/"
    assert r.unit_files == []
    assert r.rc_marker == "0"


@needs_bash
@pytest.mark.parametrize("ok", [
    "shard_000", "a.b", "A-Z_1", "..hidden", "-rf",
    # Windows 不允许名字以点结尾，建不出这个目录（Linux CI 上才有效）
    pytest.param("....", marks=pytest.mark.skipif(
        os.name == "nt", reason="Windows 不允许目录名以点结尾")),
])
def test_bash_safe_dirname_goes_parallel(sandbox, ok):
    """白名单内的边角名字（含 `-rf`、`..hidden`）照常并行，且路径是绝对路径无穿越。"""
    sandbox.make(dirs=[ok])
    r = sandbox.run()
    assert r.rc_marker == "0"
    # 分片那条（sweep 的源是源根目录，要排除掉）
    unit = [t for t in r.rsync_targets if t[0].rstrip("/").endswith(ok)]
    assert unit, r.rsync_targets
    src, dst = unit[0]
    assert src == f"{sandbox.posix}/src/team/data/{ok}/"     # 单层 basename 拼接，不可能穿越
    assert dst.endswith(f"/dst/team/data/{ok}/")
    # 穿越要按**路径段**判，不能用子串：`..hidden`/`....` 本身就含 ".."��但它们是合法文件名
    assert ".." not in src.strip("/").split("/")[:-1] + [""], src


@needs_bash
def test_bash_no_subdirs_falls_back_to_single_stream(sandbox):
    """源下只有散文件（没有一级目录）→ 退回单流，文件不会被漏掉。"""
    sandbox.make(files=["f1.bin", "f2.bin"])
    r = sandbox.run()
    assert "退回单流" in r.proc.stdout
    assert len(r.rsync_args) == 1
    assert r.unit_files == []


@needs_bash
def test_bash_empty_source_falls_back_to_single_stream(sandbox):
    """空源目录 → 退回单流（不因空 units 文件炸掉）。"""
    r = sandbox.run()
    assert "退回单流" in r.proc.stdout
    assert len(r.rsync_args) == 1


@needs_bash
def test_bash_missing_source_dir_falls_back_not_crash(sandbox):
    """源目录压根不存在（挂载没起来）→ find 报错被 2>/dev/null 吞、退回单流，不炸。"""
    r = sandbox.run(source_prefix="no/such/dir/")
    assert "退回单流" in r.proc.stdout
    assert len(r.rsync_args) == 1
    assert r.rc_marker == "0"           # 真实 rsync 会自己失败；这里只验编排不崩


@needs_bash
def test_whitelist_grep_accepts_and_rejects_expected_names(tmp_path):
    """直接对着 `grep -qvE '^[A-Za-z0-9._-]+$'` 验白名单判定（不依赖能否创建这些目录名）。

    `grep -qv` = 「存在不匹配的行」→ 退回单流。
    """
    def judge(names):
        """按源码的真实形态判定：NUL 分隔记录 + `LC_ALL=C grep -zqvE`。
        返回 True = 判定为「含不安全名字」→ 退回单流。"""
        f = tmp_path / "units.bin"
        f.write_bytes(b"".join(n.encode() + b"\0" for n in names))
        r = subprocess.run(
            [_BASH, "-c", f"LC_ALL=C grep -zqvE '{engine_ssh._UNIT_SAFE_GREP}' {_posix(f)}"],
            capture_output=True)
        return r.returncode == 0

    assert not judge(["shard_000", "a.b", "A-Z_1", "..hidden"]), "全合规却判成了含不安全名字"
    for bad in ["bad name", "a;b", "a$(id)", "a|b", "中文", "a\tb", "", "a/b",
                "ev\nil"]:            # 含换行的名字现在也必须被拒（MED-5）
        assert judge(["good", bad]), f"{bad!r} 没被白名单拦下"


@needs_bash
def test_newline_in_dirname_no_longer_slips_through_whitelist(tmp_path):
    """回归钉子：含换行的目录名**曾经**能绕过白名单（auditor MED-5，已修）。

    旧写法 `find -printf "%f\\n"` + `grep -qvE` 会把一条 key 打成两行，两半各自都合规
    → 不触发回退。若两半恰好都是真实目录名��就是**真目录静默漏传 + rc=0**
    （Linux 上实测复现过）。现在清单是 NUL 分隔 + `grep -z`，`$` 锚 NUL 不锚换行 → 拒。
    """
    name = "ev\nil"
    nul = tmp_path / "units.bin"
    nul.write_bytes(b"good\0" + name.encode() + b"\0")
    r = subprocess.run(
        [_BASH, "-c", f"LC_ALL=C grep -zqvE '{engine_ssh._UNIT_SAFE_GREP}' {_posix(nul)}"],
        capture_output=True)
    assert r.returncode == 0, "含换行的目录名没被拦下 —— MED-5 回归了"

    # 对照：旧的按行写法确实放行（证明这个洞真实存在过，不是臆造的）
    line = tmp_path / "units.txt"
    line.write_text("good\n" + name + "\n", encoding="utf-8", newline="\n")
    r_old = subprocess.run(
        [_BASH, "-c", f"LC_ALL=C grep -qvE '{engine_ssh._UNIT_SAFE_GREP}' {_posix(line)}"],
        capture_output=True)
    assert r_old.returncode != 0, "旧写法本应放行（这条挂了说明前提变了）"


# ── 5.5 泰国侧 mkdir 失败必须整体中止 ─────────────────────────────────────────

@needs_bash
def test_bash_thai_mkdir_failure_aborts_everything(sandbox):
    """泰国侧 mkdir 失败 → rsync 一次都不许跑（`{ ...; }` 分组 + `&&` 优先级）。"""
    sandbox.make(dirs=["a", "b"])
    r = sandbox.run(env={"STUB_SSH_RC": "1"})
    assert r.rsync_args == []
    assert r.unit_files == []
    assert r.rc_marker == "1"


@needs_bash
def test_bash_mutation_without_braces_would_run_rsync_after_mkdir_failure(sandbox):
    """变异验证：去掉 `{ }` 分组 → mkdir 失败后 find/xargs 照跑（上一条断言非空）。"""
    sandbox.make(dirs=["a", "b"])

    # 把 `&&` 换成 `;`：等价于「没有短路保护」，比切字符串尾部稳
    r = sandbox.run(env={"STUB_SSH_RC": "1"},
                    mutate=lambda w: w.replace(" && { ", " ; { ", 1))
    # mkdir 失败也照跑：2 条分片 + 1 条收尾 sweep
    assert len(r.rsync_args) == 3,         f"去掉短路后本应照跑 rsync（变异没生效则上条断言无意义）：{r.rsync_args}"


@needs_bash
def test_bash_single_stream_also_aborts_on_mkdir_failure(sandbox):
    """单流路径同样是 `mkdir && rsync`（回退路径没被改坏）。"""
    sandbox.make(dirs=["a"])
    r = sandbox.run(streams=1, env={"STUB_SSH_RC": "1"})
    assert r.rsync_args == []
    assert r.rc_marker == "1"


# ── 5.6 🔴 一级目录以外的内容会被静默漏传 ─────────────────────────────────────

@needs_bash
@pytest.mark.xfail(strict=True, reason="源码 bug：切分只取一级目录，源根下的散文件/符号链接"
                                       "不属于任何 unit，会被静默漏传且整单报成功")
def test_bash_toplevel_files_are_not_silently_dropped(sandbox):
    """源根下同时有目录和散文件时，散文件也必须被传（现状：**没被传，还报成功**）。

    复现：源 = `shard_000/` + `manifest.json`。
    `find -mindepth 1 -maxdepth 1 -type d` 只挑出 `shard_000`，manifest.json 不属于任何 unit；
    退回单流的条件是「一个一级目录都没有」，这里有目录所以不触发 → manifest.json 永远不过河，
    聚合退出码 0 → 卡片报「完成」。这正是 dev 在坏目录名那条上刻意避免的「静默漏传」，
    只是从另一个入口漏进来了。

    修法建议（任选）：
      · units 里额外加一个「散文件 unit」，用 `rsync -a -f'+ /*' -f'- /*/'` 之类只传顶层文件；
      · 或先跑一条 `rsync -a --exclude='/*/'`（只顶层）再并行跑各目录；
      · 或检测到源根下有非目录条目时整批退回单流（与坏名字一致的保守做法）。
    """
    sandbox.make(dirs=["shard_000"], files=["manifest.json"])
    r = sandbox.run()
    assert r.rc_marker == "0"
    all_srcs = " ".join(t[0] for t in r.rsync_targets)
    assert "manifest.json" in all_srcs or all_srcs == f"{sandbox.posix}/src/team/data/", (
        f"顶层散文件没被任何一条 rsync 覆盖：{r.rsync_targets}")


@needs_bash
def test_bash_toplevel_files_current_behaviour_documented(sandbox):
    """回归钉子：顶层散文件**曾经**静默不传却 rc=0（auditor HIGH-1，已修）。

    并行只枚举一级目录，源里有 `manifest.json` 这类散文件时一个都不传，而聚合只看 unit rc
    → 全 0 → 报成功。修法：分片全绿后再跑一趟**单流全量 sweep**（`--info=stats1`）补齐。
    「报成功但少数据」比「失败」严重得多 —— 几个月后训练读到坏文件才发现，源可能已清理。
    """
    sandbox.make(dirs=["shard_000"], files=["manifest.json", "README.md"])
    r = sandbox.run()
    assert r.rc_marker == "0"
    # 1 条分片 + 1 条收尾 sweep
    assert len(r.rsync_args) == 2, r.rsync_args
    sweep = [a for a in r.rsync_args if "--info=stats1" in a]
    assert len(sweep) == 1, f"没跑收尾 sweep，顶层散文件会静默漏传：{r.rsync_args}"
    # sweep 的源是**源根目录**（覆盖顶层散文件），不是某个分片
    assert sweep[0].rstrip().endswith("/team/data/") or "/team/data/ " in sweep[0], sweep[0]


# ══════════════════════════════════════════════════════════════════════════════
# 六、_stage2_parallel_progress（跨 unit 求和）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_run(monkeypatch):
    """mock engine_ssh.run：记录命令、按队列返回输出（Exception 实例则抛出）。"""
    box = {"cmds": [], "outs": []}

    def _run(cmd, *, timeout=30):
        box["cmds"].append(cmd)
        out = box["outs"].pop(0) if box["outs"] else ""
        if isinstance(out, Exception):
            raise out
        return (0, out, "")

    monkeypatch.setattr(engine_ssh, "run", _run)
    return box


def test_parallel_progress_parses_big_numbers(fake_run):
    """3e11 量级求和必须是精确整数（awk 一旦参与算术会输出 3e+11，直接错三个数量级）。"""
    fake_run["outs"] = ["BYTES=300000000001 UNITS=50 DONE=12\n"]
    assert engine_ssh._stage2_parallel_progress("sgp-x") == {
        "bytes_done": 300000000001, "pct": None, "speed_bps": None,
        "units_total": 50, "units_done": 12}


def test_parallel_progress_pct_and_speed_deliberately_none(fake_run):
    """pct/speed 刻意不给：单 unit 的 % 没有全局意义、rsync 瞬时速率抖动大，交给上层算。"""
    fake_run["outs"] = ["BYTES=1 UNITS=1 DONE=0\n"]
    p = engine_ssh._stage2_parallel_progress("sgp-x")
    assert p["pct"] is None and p["speed_bps"] is None


def test_parallel_progress_zero_units_means_single_stream(fake_run):
    """UNITS=0（没有 unit 日志）→ None → 调用方回落单流解析。"""
    fake_run["outs"] = ["BYTES=0 UNITS=0 DONE=0\n"]
    assert engine_ssh._stage2_parallel_progress("sgp-x") is None


@pytest.mark.parametrize("out", ["", "garbage\n", "BYTES=abc UNITS=1 DONE=0\n",
                                 "BYTES=1 UNITS=x DONE=0\n", None])
def test_parallel_progress_unparsable_returns_none(fake_run, out):
    fake_run["outs"] = [out if out is not None else ""]
    assert engine_ssh._stage2_parallel_progress("sgp-x") is None


def test_parallel_progress_run_exception_returns_none(fake_run):
    """SSH 抛错 → None（不能把 poll_once 带崩，进度只是锦上添花）。"""
    fake_run["outs"] = [RuntimeError("ssh down")]
    assert engine_ssh._stage2_parallel_progress("sgp-x") is None


def test_parallel_progress_probe_shape(fake_run):
    """探针形状：8KB 尾窗 + 仅截断时丢首行 + `tr '\\r' '\\n'` + awk 只打印不算术 + shell 加法。"""
    fake_run["outs"] = ["BYTES=0 UNITS=1 DONE=0\n"]
    engine_ssh._stage2_parallel_progress("sgp-x")
    cmd = fake_run["cmds"][0]
    assert "tail -c 8000" in cmd
    assert '-gt 8000' in cmd and "tail -n +2" in cmd
    assert r"tr '\r' '\n'" in cmd
    assert 'gsub(/,/,"",$1); print $1' in cmd          # 只 print，不做算术
    assert "$((tot+v))" in cmd                         # 64 位加法交给 shell
    assert "stage2.unit-*.log" in cmd and "stage2.unit-*.rc" in cmd
    assert fake_run["cmds"] and "sgp-x" in cmd


@needs_bash
def test_bash_parallel_progress_sums_exactly(sandbox, monkeypatch):
    """真 bash：两条 unit 日志（一条 >8KB 且被 `\\r` 刷屏）求和精确，无科学计数法、不少算。"""
    jd = sandbox.root / "work" / "sgp-p1"
    jd.mkdir(parents=True)
    (jd / "stage2.unit-u1.log").write_text(
        "sending incremental file list\n"
        "      1,000  0%   1.00MB/s    0:00:01\n"
        "150,000,000,000  73%   45.67MB/s    0:12:34\n", encoding="utf-8", newline="\n")
    big = "".join("   %d  %d%%   1.00MB/s    0:00:01\r" % (i * 1000, i % 100)
                  for i in range(1200))                       # >8KB，全靠 \r 刷屏
    (jd / "stage2.unit-u2.log").write_text(
        big + "150,000,000,001  99%   2.00MB/s    0:00:02\n", encoding="utf-8", newline="\n")
    (jd / "stage2.unit-u3.log").write_text(
        "rsync: connection unexpectedly closed\n", encoding="utf-8", newline="\n")  # 无进度行
    (jd / "stage2.unit-u1.rc").write_text("0", encoding="utf-8")
    (jd / "stage2.unit-u2.rc").write_text("24", encoding="utf-8")

    monkeypatch.setattr(engine_ssh, "run", _bash_run)
    p = engine_ssh._stage2_parallel_progress("sgp-p1")
    assert p["bytes_done"] == 150000000000 + 150000000001     # 精确，不是 3e+11
    assert p["units_total"] == 3                              # 无进度行的 unit 也计数
    assert p["units_done"] == 2                               # 按 rc 文件数


@needs_bash
def test_bash_parallel_progress_none_when_no_unit_logs(sandbox, monkeypatch):
    """job 目录里没有 unit 日志（单流跑的）→ None。"""
    (sandbox.root / "work" / "sgp-p2").mkdir(parents=True)
    monkeypatch.setattr(engine_ssh, "run", _bash_run)
    assert engine_ssh._stage2_parallel_progress("sgp-p2") is None


def _bash_run(cmd, *, timeout=30):
    """把探针命令真的丢给 bash 跑（供上面两条真机形态用例用）。"""
    r = subprocess.run([_BASH, "-c", cmd], capture_output=True, text=True, timeout=120)
    return (r.returncode, r.stdout, r.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# 七、stage_progress 分发：并行优先，拿不到才回落单流解析
# ══════════════════════════════════════════════════════════════════════════════

def test_stage_progress_prefers_parallel_and_skips_tail_log(monkeypatch):
    par = {"bytes_done": 42, "pct": None, "speed_bps": None,
           "units_total": 3, "units_done": 1}
    monkeypatch.setattr(engine_ssh, "_stage2_parallel_progress", lambda jid: par)
    monkeypatch.setattr(engine_ssh, "tail_log",
                        lambda *a, **k: pytest.fail("并行有结果时不该再 tail 单流日志"))
    assert engine_ssh.stage_progress("sgp-x", STAGE2) == par


def test_stage_progress_falls_back_to_single_stream_parse(monkeypatch):
    """并行返回 None → 回落原来的 rsync progress2 解析（老日志形态不回归）。"""
    monkeypatch.setattr(engine_ssh, "_stage2_parallel_progress", lambda jid: None)
    monkeypatch.setattr(engine_ssh, "tail_log",
                        lambda *a, **k: "  1,234,567  73%   45.67MB/s    0:12:34\n")
    assert engine_ssh.stage_progress("sgp-x", STAGE2) == {
        "bytes_done": 1234567, "pct": 73, "speed_bps": int(45.67 * 1024 ** 2)}


def test_stage_progress_stage1_never_probes_parallel(monkeypatch):
    """段1 不碰并行探针（段1 是 ossutil，不存在分片日志；也别白烧一次 SSH）。"""
    monkeypatch.setattr(engine_ssh, "_stage2_parallel_progress",
                        lambda jid: pytest.fail("段1 不该查并行分片进度"))
    monkeypatch.setattr(engine_ssh, "tail_log", lambda *a, **k: "12.3MB/s")
    p = engine_ssh.stage_progress("sgp-x", STAGE1)
    assert p["speed_bps"] == int(12.3 * 1024 ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# 八、_stage2_failed_units（远端输出当外部数据处理）
# ══════════════════════════════════════════════════════════════════════════════

def test_failed_units_parses_names(fake_run):
    fake_run["outs"] = ["shard_001\nshard_007\n"]
    assert engine_ssh._stage2_failed_units("sgp-x") == ["shard_001", "shard_007"]


@pytest.mark.parametrize("junk", [
    "../../etc/passwd", "a;rm -rf /", "a b", "中文", "`id`", "$(id)", "a/b", "a|b",
    "a'q", 'a"q', "a\\b", "*", "..", ".", "-", "a>b",
])
def test_failed_units_filters_unsafe_remote_names(fake_run, junk):
    """远端回不合规名字一律丢弃 —— 它会被拼进 `stage2.unit-<u>.log` 去 grep（读取 oracle 面）。

    注意 `..`/`.`/`-` 本身**是**白名单字符组成的合法串，会被放行；但它们只作为单层
    basename 拼进 job 目录下的固定前缀，构不成穿越（下面 test_failed_units_path_join 钉住）。
    """
    fake_run["outs"] = [f"good_1\n{junk}\nGOOD_2\n"]
    got = engine_ssh._stage2_failed_units("sgp-x")
    if re.fullmatch(r"[A-Za-z0-9._-]+", junk):
        assert junk in got                       # 合法字符组成的怪名字：放行是预期
    else:
        assert junk not in got
    assert got[0] == "good_1" and got[-1] == "GOOD_2"


def test_failed_units_drops_blank_and_whitespace(fake_run):
    fake_run["outs"] = ["\n  \n\t\nshard_1\n\n"]
    assert engine_ssh._stage2_failed_units("sgp-x") == ["shard_1"]


def test_failed_units_strips_cr(fake_run):
    fake_run["outs"] = ["shard_1\r\nshard_2\r\n"]
    assert engine_ssh._stage2_failed_units("sgp-x") == ["shard_1", "shard_2"]


def test_failed_units_empty_output(fake_run):
    fake_run["outs"] = [""]
    assert engine_ssh._stage2_failed_units("sgp-x") == []


def test_failed_units_run_exception_returns_empty(fake_run):
    """探针抛错 → []（调用方退回原逻辑，绝不外抛把终态判定带崩）。"""
    fake_run["outs"] = [RuntimeError("ssh down")]
    assert engine_ssh._stage2_failed_units("sgp-x") == []


def test_failed_units_probe_only_treats_0_and_24_as_success(fake_run):
    fake_run["outs"] = [""]
    engine_ssh._stage2_failed_units("sgp-x")
    cmd = fake_run["cmds"][0]
    assert 'case "$r" in 0|24) ;;' in cmd
    assert "stage2.unit-*.rc" in cmd


def test_failed_units_path_join_stays_in_job_dir(fake_run, monkeypatch):
    """放行的名字只做单层拼接，落点始终在本 job 目录下（不构成任意文件读取）。"""
    monkeypatch.setattr(engine_ssh.settings, "SGP_WORK_DIR", "/var/run/ssh_transfer")
    fake_run["outs"] = ["..", "COUNT=0\n"]
    fake_run["outs"] = ["..\n", "", ""]
    engine_ssh.failure_detail("sgp-abc123", STAGE2)
    grep_cmd = fake_run["cmds"][1]
    assert "/var/run/ssh_transfer/sgp-abc123/stage2.unit-...log" in grep_cmd
    assert "/etc/" not in grep_cmd


@needs_bash
def test_bash_failed_units_reads_real_rc_files(sandbox, monkeypatch):
    """真 bash：0/24 算成功，5/23/空/垃圾算失败（空 rc = 分片没写完，必须暴露）。"""
    jd = sandbox.root / "work" / "sgp-f1"
    jd.mkdir(parents=True)
    for u, rc in (("a", "0"), ("b", "24"), ("c", "5"), ("d", "23"), ("e", ""), ("f", "oops")):
        (jd / f"stage2.unit-{u}.rc").write_text(rc, encoding="utf-8")
    monkeypatch.setattr(engine_ssh, "run", _bash_run)
    assert engine_ssh._stage2_failed_units("sgp-f1") == ["c", "d", "e", "f"]


@needs_bash
def test_bash_failed_units_empty_when_no_rc_files(sandbox, monkeypatch):
    (sandbox.root / "work" / "sgp-f2").mkdir(parents=True)
    monkeypatch.setattr(engine_ssh, "run", _bash_run)
    assert engine_ssh._stage2_failed_units("sgp-f2") == []


# ══════════════════════════════════════════════════════════════════════════════
# 九、failure_detail：先列失败分片，再 grep 第一个失败 unit 的日志
# ══════════════════════════════════════════════════════════════════════════════

def test_failure_detail_lists_failed_shards_and_greps_first(fake_run):
    """失败分片清单进结论行，且 grep 目标切到 `stage2.unit-<第一个失败>.log`。

    并行时 `stage2.log` 里只有外层那两行编排输出，真错误在各 unit 日志里。
    """
    fake_run["outs"] = ["shard_003\nshard_009\n",
                        "rsync: connection unexpectedly closed (code 12)\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert "失败分片 2 个：shard_003, shard_009" in detail
    assert "rsync: connection unexpectedly closed" in detail
    grep_cmd = fake_run["cmds"][1]
    assert "stage2.unit-shard_003.log" in grep_cmd
    assert engine_ssh._marker("sgp-abc123", STAGE2, "log") not in grep_cmd


def test_failure_detail_truncates_shard_list_at_six(fake_run):
    """>6 个失败分片只列前 6 + 省略号，但**条数是全量**（卡片塞不下 50 个名字）。"""
    units = [f"s{i:02d}" for i in range(11)]
    fake_run["outs"] = ["\n".join(units) + "\n", "rsync error: x\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert "失败分片 11 个：" in detail
    assert "s00, s01, s02, s03, s04, s05 …" in detail
    assert "s06" not in detail


def test_failure_detail_no_failed_units_keeps_stage2_log(fake_run):
    """没有失败分片（单流跑的 / rc 文件都干净）→ 行为与改动前一致，仍 grep stage2.log。"""
    fake_run["outs"] = ["", "rsync error: some files could not be transferred (code 23)\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert "失败分片" not in detail
    assert engine_ssh._marker("sgp-abc123", STAGE2, "log") in fake_run["cmds"][1]
    assert detail == "rsync error: some files could not be transferred (code 23)"


def test_failure_detail_stage1_does_not_probe_shards(fake_run):
    """段1 不发分片探针（只在段2 多花一次 SSH）。"""
    fake_run["outs"] = ["Error occurs: something\n"]
    engine_ssh.failure_detail("sgp-abc123", STAGE1)
    assert len(fake_run["cmds"]) == 1
    assert "stage2.unit-" not in fake_run["cmds"][0]


def test_failure_detail_survives_shard_probe_exception(monkeypatch):
    """分片探针抛错 → 照旧走 stage2.log 出明细（best-effort，不整体失败）。"""
    calls = {"n": 0}

    def _run(cmd, *, timeout=30):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ssh down")
        return (0, "rsync error: boom\n", "")

    monkeypatch.setattr(engine_ssh, "run", _run)
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert detail == "rsync error: boom"


def test_failure_detail_shard_line_survives_budget_squeeze(fake_run):
    """长日志行不得把「失败分片」结论挤没（它进 key 列表、优先于 lead 原文）。"""
    long_lines = "\n".join(["rsync error: " + "X" * 500] * 4)
    fake_run["outs"] = ["s1\ns2\ns3\n", long_lines + "\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert "失败分片 3 个：s1, s2, s3" in detail
    assert len(detail) <= engine_ssh._DETAIL_MAX


def test_failure_detail_shard_line_is_clipped(fake_run):
    """分片清单单行也过 `_clip`（240），不会一行吃掉整个预算。"""
    units = ["u" * 100, "v" * 100, "w" * 100]
    fake_run["outs"] = ["\n".join(units) + "\n", "rsync error: x\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    shard_line = [ln for ln in detail.splitlines() if ln.startswith("失败分片")][0]
    assert len(shard_line) <= engine_ssh._DETAIL_LINE_MAX


def test_failure_detail_stage2_einval_note_still_not_ossfs2(fake_run):
    """段2 的 EINVAL 说明仍是「泰国侧目标目录/挂载点」，没被并行改动带偏。"""
    fake_run["outs"] = ["u1\n", "rsync: write failed: invalid argument\n"]
    detail = engine_ssh.failure_detail("sgp-abc123", STAGE2)
    assert "ossfs2" not in detail and "--part-size" not in detail
    assert "泰国" in detail and "EINVAL" in detail
    assert "失败分片 1 个：u1" in detail


# ══════════════════════════════════════════════════════════════════════════════
# 十、与 orchestrator 的衔接（并行版不给速率 → 上层用字节差算）
# ══════════════════════════════════════════════════════════════════════════════

def test_sample_progress_first_sample_has_no_speed(monkeypatch):
    """首次采样只落字节（没有前一个采样点，算不出速率）。"""
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 1000, "pct": None, "speed_bps": None,
                                         "units_total": 5, "units_done": 0})
    job = {"job_id": "sgp-x"}
    orch._sample_progress(job, STAGE2)
    assert job["bytes_done"] == 1000
    assert job["_bd_sample"] == 1000 and job["_bd_ts"]
    assert "speed_bps" not in job


def test_sample_progress_second_sample_computes_speed_from_byte_delta(monkeypatch):
    """**本改动的关键衔接**：并行版 speed_bps=None → 上层必须用相邻两次字节差算速率。"""
    seq = [{"bytes_done": 1000, "pct": None, "speed_bps": None},
           {"bytes_done": 7000, "pct": None, "speed_bps": None}]
    monkeypatch.setattr(orch.engine_ssh, "stage_progress", lambda jid, st: seq.pop(0))
    t = [1000.0, 1060.0]
    monkeypatch.setattr(orch.time, "time", lambda: t.pop(0))
    job = {"job_id": "sgp-x"}
    orch._sample_progress(job, STAGE2)
    orch._sample_progress(job, STAGE2)
    assert job["bytes_done"] == 7000
    assert job["speed_bps"] == (7000 - 1000) // 60      # 60s 平均，稳过 rsync 瞬时值


def test_sample_progress_parallel_never_sets_pct(monkeypatch):
    """并行版 pct=None → job['pct'] 必须被**无条件覆盖成 None**（auditor LOW-9）。

    只在非 None 时写的话，曾单流跑过的 job 会一直挂着上一轮的陈旧百分比，而 progress_line
    优先用它 → 卡片进度永久定格在旧值（线上真见过「60% / 剩余 308h」）。
    """
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 5, "pct": None, "speed_bps": None})
    job = {"job_id": "sgp-x", "pct": 60}          # 上一轮遗留的陈旧百分比
    orch._sample_progress(job, STAGE2)
    assert job["pct"] is None, "陈旧 pct 没被覆盖"


def test_progress_line_computes_global_pct_from_bytes(monkeypatch):
    """没有 pct 时 progress_line 用 bytes_done/bytes_total 出全局百分比（并行下的真值）。"""
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 250, "pct": None, "speed_bps": None})
    job = {"job_id": "sgp-x", "bytes_total": 1000}
    orch._sample_progress(job, STAGE2)
    line = orch.progress_line(job)
    assert "(25%)" in line and "已传" in line


def test_progress_line_with_computed_speed_shows_eta(monkeypatch):
    seq = [{"bytes_done": 1000, "pct": None, "speed_bps": None},
           {"bytes_done": 7000, "pct": None, "speed_bps": None}]
    monkeypatch.setattr(orch.engine_ssh, "stage_progress", lambda jid, st: seq.pop(0))
    t = [1000.0, 1060.0]
    monkeypatch.setattr(orch.time, "time", lambda: t.pop(0))
    job = {"job_id": "sgp-x", "bytes_total": 13000}
    orch._sample_progress(job, STAGE2)
    orch._sample_progress(job, STAGE2)
    line = orch.progress_line(job)
    assert "速率" in line and "剩余约" in line


def test_sample_progress_units_counters_are_dropped(monkeypatch):
    """分片计数必须透传到 job 并显示（auditor LOW-8，已实现）。

    done 与 total 的差额是「有分片没跑完」的**唯一可见信号** —— 缺 rc 那类问题
    （HIGH-2）在卡片上本来完全看不出来。
    """
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": 1, "pct": None, "speed_bps": None,
                                         "units_total": 50, "units_done": 12})
    job = {"job_id": "sgp-x", "bytes_total": 100}
    orch._sample_progress(job, STAGE2)
    assert job["units_total"] == 50 and job["units_done"] == 12
    assert "分片 12/50" in orch.progress_line(job)
    # 分片全完成但还在途 = 正在跑收尾核对，要标出来（否则字节数几十分钟不动像卡死）
    job["units_done"], job["stage"] = 50, orch.STAGE_STAGE2
    assert "收尾核对中" in orch.progress_line(job)


def test_poll_once_stage2_samples_progress(monkeypatch):
    """poll_once 在段2 在途时确实调 stage_progress（并行进度进得了 job）。"""
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "RUNNING", "rc": None, "alive": True})
    seen = []
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: seen.append(st) or
                        {"bytes_done": 9, "pct": None, "speed_bps": None})
    job = {"job_id": "sgp-x", "stage": orch.STAGE_STAGE2}
    out = orch.poll_once(job)
    assert seen == [STAGE2]
    assert out["bytes_done"] == 9
    assert out["stage"] == orch.STAGE_STAGE2


def test_retry_reruns_stage2_only_and_reuses_parallel(monkeypatch):
    """retry（段1 已成功）只重起段2 —— 并行改动没破坏这条既有省流量路径。

    段2 默认已改「泰国 ossutil 直拉」，这条测的是 rsync 回滚路径，显式钉住模式。
    """
    monkeypatch.setattr(orch.settings, "SSH_STAGE2_MODE", "rsync", raising=False)
    started = []
    monkeypatch.setattr(orch.engine_ssh, "start_stage1",
                        lambda *a, **k: pytest.fail("段1 已成功不该重跑"))
    monkeypatch.setattr(orch.engine_ssh, "start_stage2",
                        lambda jid, **k: started.append(k))
    monkeypatch.setattr(orch.engine_ssh, "poll_stage",
                        lambda jid, st: {"status": "DONE", "rc": 0})
    monkeypatch.setattr(orch.engine_ssh, "stage_progress",
                        lambda jid, st: {"bytes_done": None, "pct": None, "speed_bps": None})
    monkeypatch.setattr(orch.time, "sleep", lambda *_: None)
    job = {"job_id": "sgp-x", "stage": orch.STAGE_FAILED, "stage1_rc": 0,
           "source_prefix": "team/data/", "dest_rel": "test/", "source_bucket": "b"}
    out = orch.run_to_completion(job, poll_interval=0, max_polls=2)
    assert started == [{"source_prefix": "team/data/", "dest_rel": "test/"}]
    assert out["stage"] == orch.STAGE_DONE
