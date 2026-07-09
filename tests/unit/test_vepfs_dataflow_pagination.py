"""#44 vePFS 数据流动回归测试：

覆盖 dev 修的三处（真根因=DescribeDataFlowTasks 缺分页 → 任务永远卡 RUNNING）：
  P1 engine_vepfs.query_task —— 请求带 page_number/page_size；total_count>0 但列表空打 warning。
  P2 orchestrator.poll_once —— 终态先判 is_failed 再判 is_done（防歧义子串）。
  P4 engine_vepfs._err —— 解析火山错误 JSON 抽 Error.Code/Message，已知码给人话。

只碰 tests/，不改源码。
"""
import logging

import pytest

from core.vepfs_dataflow import engine_vepfs, orchestrator as orch


# ── fakes ────────────────────────────────────────────────────────────────────

class _Task:
    """假 DataFlowTask 对象，字段 getattr 容错。"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    """假 DescribeDataFlowTasks 响应。"""
    def __init__(self, tasks, total_count=0):
        self.data_flow_tasks = tasks
        self.total_count = total_count


class _FakeVepfsQuery:
    """同时充当 vepfs 模块（构造 request，记录 kwargs）与 api（describe）。"""
    def __init__(self, cap, resp):
        self.cap = cap
        self._resp = resp

    def DescribeDataFlowTasksRequest(self, **kw):
        self.cap["query_req"] = kw
        return kw

    def describe_data_flow_tasks(self, req):
        self.cap["called"] = True
        return self._resp


class _FakeExc(Exception):
    """带 .body 属性的假火山 SDK 异常，喂给 _err。"""
    def __init__(self, body):
        super().__init__("boom")
        self.body = body


# ── P1: query_task 分页参数 + 空列表告警 ─────────────────────────────────────────

def test_query_task_passes_pagination_params(monkeypatch):
    """核心修复：DescribeDataFlowTasksRequest 必须带 page_number=1 / page_size=100，
    否则火山返 total_count>0 但空列表 → 永远卡 RUNNING。"""
    cap = {}
    # 真机 3dd16c139445 的形态：Finished、0 文件、源空、成功空操作
    task = _Task(data_flow_task_id=901, status="Finished",
                 total_size=0, exec_size=0, total_count=0, exec_count=0, failed_count=0)
    fake = _FakeVepfsQuery(cap, _Resp([task], total_count=1))
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))

    st = engine_vepfs.query_task("901", "vepfs-a", "cn-shanghai")

    # 请求确实带了分页参数（回归防线：dev 修复的关键）
    req = cap["query_req"]
    assert req["page_number"] == 1
    assert req["page_size"] == 100
    assert req["file_system_id"] == "vepfs-a"
    assert req["data_flow_task_ids"] == "901"
    # 解析出终态与计数
    assert st["status"] == "Finished"
    assert st["files_done"] == 0 and st["files_total"] == 0
    assert st["error"] == ""


def test_query_task_parses_progress_counts(monkeypatch):
    """带进度的任务：字节/文件计数如实抽取。"""
    cap = {}
    task = _Task(data_flow_task_id=902, status="Running",
                 total_size=2048, exec_size=1024, total_count=10, exec_count=4, failed_count=0)
    fake = _FakeVepfsQuery(cap, _Resp([task], total_count=1))
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))

    st = engine_vepfs.query_task("902", "vepfs-a", "cn-shanghai")
    assert st["status"] == "Running"
    assert st["bytes_total"] == 2048 and st["bytes_done"] == 1024
    assert st["files_total"] == 10 and st["files_done"] == 4


def test_query_task_empty_list_with_total_count_warns(monkeypatch, caplog):
    """total_count>0 但 data_flow_tasks=[] （旧 bug 现场）→ 打 warning，不抛，返回空 status。"""
    cap = {}
    fake = _FakeVepfsQuery(cap, _Resp([], total_count=1))
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))

    with caplog.at_level(logging.WARNING, logger="core.vepfs_dataflow.engine_vepfs"):
        st = engine_vepfs.query_task("903", "vepfs-a", "cn-shanghai")

    assert st["status"] == ""          # 拿不到任务 → 空 status（不再误判/崩溃）
    assert st["files_total"] == 0
    # 分页参数照样带上
    assert cap["query_req"]["page_number"] == 1 and cap["query_req"]["page_size"] == 100
    # 命中「total_count 但列表空」告警分支
    assert any("total_count 但列表空" in r.message for r in caplog.records)


def test_query_task_empty_list_no_total_count_silent(monkeypatch, caplog):
    """列表空且 total_count=0 → 正常「无此任务」，不打 warning。"""
    cap = {}
    fake = _FakeVepfsQuery(cap, _Resp([], total_count=0))
    monkeypatch.setattr(engine_vepfs, "_api", lambda region: (fake, fake))

    with caplog.at_level(logging.WARNING, logger="core.vepfs_dataflow.engine_vepfs"):
        st = engine_vepfs.query_task("904", "vepfs-a", "cn-shanghai")

    assert st["status"] == ""
    assert not any("total_count 但列表空" in r.message for r in caplog.records)


# ── P2: poll_once 终态判定顺序（先 failed 后 done）──────────────────────────────

def _running_job():
    return {
        "job_id": "j-poll-test",
        "stage": orch.STAGE_RUNNING,
        "task_id": "task-1",
        "fs_id": "vepfs-a",
        "region": "cn-shanghai",
    }


def _poll_with_status(monkeypatch, status, error=""):
    monkeypatch.setattr(engine_vepfs, "query_task",
                        lambda *a, **k: {"status": status, "error": error})
    return orch.poll_once(_running_job())


def test_poll_once_finished_to_done(monkeypatch):
    job = _poll_with_status(monkeypatch, "Finished")
    assert job["stage"] == orch.STAGE_DONE
    assert job.get("finished_ts")


def test_poll_once_failed_to_failed(monkeypatch):
    job = _poll_with_status(monkeypatch, "Failed", error="boom")
    assert job["stage"] == orch.STAGE_FAILED
    assert "boom" in job["error"]


def test_poll_once_running_stays_running(monkeypatch):
    job = _poll_with_status(monkeypatch, "Running")
    assert job["stage"] == orch.STAGE_RUNNING
    assert not job.get("finished_ts")


@pytest.mark.parametrize("status", ["FinishedWithFailures", "CompletedWithErrors", "Unsuccessful"])
def test_poll_once_ambiguous_status_prefers_failed(monkeypatch, status):
    """歧义串同时命中 done 和 fail 提示：
      'FinishedWithFailures' = 'finished'(DONE) + 'fail'(FAIL)
      'CompletedWithErrors'  = 'complete'(DONE) + 'error'(FAIL)
      'Unsuccessful'         = 'success'(DONE)  + 'unsuccess'(FAIL, dev 为此串专门补的 hint)
    先判 failed → 判为 FAILED，而非旧顺序误判 DONE。这正是 P2 换序的意义。"""
    # 前置断言：该串确实两边都命中，否则本测无意义
    assert engine_vepfs.is_done(status) and engine_vepfs.is_failed(status)
    job = _poll_with_status(monkeypatch, status)
    assert job["stage"] == orch.STAGE_FAILED


# ── P4: _err 友好错误解析 ────────────────────────────────────────────────────────

def test_err_friendly_bucket_not_exist():
    body = ('{"ResponseMetadata":{"Error":{"Code":"InvalidParameter.BucketMaybeNotExist",'
            '"Message":"bucket not found"}}}')
    out = engine_vepfs._err(_FakeExc(body))
    assert "桶不存在" in out and "漏填桶名" in out
    assert "InvalidParameter.BucketMaybeNotExist" in out   # 附原始码便于排查


def test_err_friendly_source_prefix():
    body = ('{"ResponseMetadata":{"Error":{"Code":"InvalidParameter.SourceStoragePrefix",'
            '"Message":"bad prefix"}}}')
    out = engine_vepfs._err(_FakeExc(body))
    assert "前缀非法" in out
    assert "InvalidParameter.SourceStoragePrefix" in out


def test_err_error_at_top_level():
    """Error 直挂顶层（无 ResponseMetadata 包裹）也能解析。"""
    body = '{"Error":{"Code":"InvalidParameter.BucketName","Message":"bad name"}}'
    out = engine_vepfs._err(_FakeExc(body))
    assert "桶名格式非法" in out
    assert "InvalidParameter.BucketName" in out


def test_err_unknown_code_falls_back_to_code_msg():
    body = '{"ResponseMetadata":{"Error":{"Code":"SomethingWeird","Message":"detail here"}}}'
    out = engine_vepfs._err(_FakeExc(body))
    assert out == "SomethingWeird：detail here"


def test_err_unknown_code_no_message_returns_code():
    body = '{"Error":{"Code":"OnlyCode","Message":""}}'
    out = engine_vepfs._err(_FakeExc(body))
    assert out == "OnlyCode"


def test_err_non_json_body_truncated():
    out = engine_vepfs._err(_FakeExc("plain text failure not json"))
    assert out == "plain text failure not json"


def test_err_non_json_truncated_to_300():
    out = engine_vepfs._err(_FakeExc("x" * 500))
    assert len(out) == 300


def test_err_no_body_uses_str():
    """无 body/message/reason 属性 → 回退 str(e)。"""
    out = engine_vepfs._err(ValueError("raw value error"))
    assert "raw value error" in out
