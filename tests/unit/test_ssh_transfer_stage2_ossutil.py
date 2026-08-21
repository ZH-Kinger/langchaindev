

def test_fast_stage2_not_mistaken_for_launch_failure(monkeypatch):
    """段2 跑得比下发校验里的 `sleep 3` 还快时，不能误报「起来即死」。

    九章链实测：66 MiB / 11 对象 0.9 秒跑完，进程已正常退出、rc=0、数据已落地，
    却被只按 `kill -0` 的判定报成 LAUNCH_DEAD。这个 bug 只在小任务上出现，
    而小任务正是大家用来验证链路的那种。
    """
    from core.ssh_transfer import engine_ossutil as eo
    monkeypatch.setattr(eo, "run_thai", lambda s, **k: (0, "ALREADY_DONE rc=0", ""))
    eo.start_stage2("sgp-t", source_prefix="a/")       # 不抛即通过


def test_stage2_launch_script_checks_rc_first(monkeypatch):
    from core.ssh_transfer import engine_ossutil as eo
    cap = {}
    monkeypatch.setattr(eo, "run_thai", lambda s, **k: cap.setdefault("s", s) and None or (0, "LAUNCHED", ""))
    eo.start_stage2("sgp-t", source_prefix="a/")
    s = cap["s"]
    assert s.rindex('if [ -f "$JD/stage2.rc" ]') < s.rindex("kill -0"), \
        "下发后校验必须先看 rc、再看进程存活"
