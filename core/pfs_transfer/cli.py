"""PFS↔PFS 跨云直传 CLI：plan / apply / status。dry-run 默认，--force 越审批阈值。

    python -m core.pfs_transfer.cli plan  vepfs://<fs>/dir/ cpfs://<fs>/dir/
    python -m core.pfs_transfer.cli apply vepfs://<fs>/dir/ cpfs://<fs>/dir/ [--same-name skip] [--force]
    python -m core.pfs_transfer.cli status xpfs-xxxxxxxx
"""
import argparse
import sys

from . import paths, orchestrator as o


def _print_plan(plan: paths.Plan) -> None:
    print(f"方向        : {plan.direction}")
    print(f"源  PFS     : {plan.src_pfs.scheme}://{plan.src_pfs.fs_id}/{plan.src_pfs.sub_path}  (region={plan.src_pfs.region})")
    print(f"目的 PFS    : {plan.dst_pfs.scheme}://{plan.dst_pfs.fs_id}/{plan.dst_pfs.sub_path}  (region={plan.dst_pfs.region})")
    print(f"① 沉降落点  : {plan.src_staging.scheme}://{plan.src_staging.bucket}/{plan.src_staging.base_prefix}"
          f"  (region={plan.src_staging.region})")
    print(f"② 跨云落点  : {plan.dst_staging.scheme}://{plan.dst_staging.bucket}/{plan.dst_staging.base_prefix}"
          f"  (region={plan.dst_staging.region}, dataflow={plan.dst_staging.dataflow_id or '-'})")
    print(f"③ 预热       : {plan.dst_staging.scheme}→{plan.dst_pfs.scheme}")


def cmd_plan(args) -> int:
    plan = paths.build_plan(args.source, args.dest)
    print("=== PFS 直传 计划（dry-run，不建 job、不触云）===")
    _print_plan(plan)
    _, size_known = o.estimate_source(plan)
    need = o.needs_approval(0, size_known)
    print(f"审批        : {'需管理员审批（大小未知→fail-safe）' if need else '免'}")
    return 0


def cmd_apply(args) -> int:
    plan = paths.build_plan(args.source, args.dest)
    _print_plan(plan)
    _, size_known = o.estimate_source(plan)
    if o.needs_approval(0, size_known) and not args.force:
        print("\n✗ 大小未知/超阈值，需审批。确认后加 --force 执行。")
        return 2
    job = o.create_job_record(plan, same_name=args.same_name or "")
    print(f"\njob_id: {job['job_id']}  启动三段链…")
    job = o.run_to_completion(job, on_update=lambda j: print(f"  [{j['stage']}] {o.progress_line(j)}"))
    print(f"\n终态: {job['stage']}" + (f"  error={job.get('error')}" if job.get("error") else ""))
    return 0 if job["stage"] == o.STAGE_DONE else 1


def cmd_status(args) -> int:
    job = o.get_job(args.job_id)
    if not job:
        print(f"未找到任务 {args.job_id}")
        return 1
    print(f"job_id : {job['job_id']}")
    print(f"方向   : {job.get('direction')}")
    print(f"阶段   : {job.get('stage')}")
    print(f"段完成 : sink={job.get('sink_done')} cross={job.get('cross_done')} preheat={job.get('preheat_done')}")
    print(f"进度   : {o.progress_line(job)}")
    if job.get("error"):
        print(f"错误   : {job['error']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pfs_transfer", description="vePFS↔CPFS 跨云直传（3 段链）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("source", help="源 PFS 地址，如 vepfs://<fs>/dir/")
        p.add_argument("dest", help="目的 PFS 地址，如 cpfs://<fs>/dir/")
        p.add_argument("--same-name", default="", help="同名策略 skip/keeplatest/overwrite")
        if name == "apply":
            p.add_argument("--force", action="store_true", help="越过审批阈值直接执行")
    ps = sub.add_parser("status")
    ps.add_argument("job_id")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            return cmd_plan(args)
        if args.cmd == "apply":
            return cmd_apply(args)
        if args.cmd == "status":
            return cmd_status(args)
    except paths.PfsPathError as e:
        print(f"✗ {e}")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
