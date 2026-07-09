"""SSH 迁移链 CLI：`python -m core.ssh_transfer.cli plan|apply|status`。

  plan   oss://wuji-data-tran/<prefix>/     只估算、打印计划，不建 job（dry-run 默认）
  apply  oss://wuji-data-tran/<prefix>/     建 job 并阻塞跑完两段（--force 越审批阈值）
  status sgp-xxxxxx                          实时重查并打印某 job 状态
"""
import argparse
import sys

from core.ssh_transfer import paths, orchestrator as o


def _cmd_plan(args) -> int:
    try:
        plan = paths.build_plan(args.source)
    except paths.SshPathError as e:
        print(f"路径错误：{e}")
        return 2
    b, n, ok = o.estimate_source(plan)
    print(f"源      : {plan.source_uri()}")
    print(f"段1落地 : {plan.source_prefix}  → SGP 挂载盘")
    print(f"段2落地 : {o._dest_root()}/{plan.source_prefix}  （泰国）")
    print(f"估算    : {o.fmt_size(b)} / {n} 对象" + ("" if ok else "（估算失败/未知）"))
    print(f"审批    : {'需审批' if o.needs_approval(b, ok) else '无需审批'}")
    print(f"job_id  : {o._job_id(plan)}（当天幂等，apply 时才建）")
    return 0


def _cmd_apply(args) -> int:
    try:
        plan = paths.build_plan(args.source)
    except paths.SshPathError as e:
        print(f"路径错误：{e}")
        return 2
    b, n, ok = o.estimate_source(plan)
    if o.needs_approval(b, ok) and not args.force:
        size = o.fmt_size(b) if ok else "未知（估算失败）"
        print(f"源大小 {size} 需审批，加 --force 强制下发。")
        return 3
    job = o.create_job_record(plan, bytes_total=b, objects_total=n, size_known=ok)
    print(f"job_id={job['job_id']} 已建，开始跑两段…")

    def _on(j):
        print(f"  [{j['stage']}] {o.stage_label(j['stage'])}"
              + (f"  错误：{j['error']}" if j.get("error") else ""))

    job = o.run_to_completion(job, on_update=_on)
    print(f"终态：{job['stage']}" + (f"  错误：{job.get('error')}" if job.get("error") else ""))
    return 0 if job["stage"] == o.STAGE_DONE else 1


def _cmd_status(args) -> int:
    job = o.refresh(args.job_id) or o.get_job(args.job_id)
    if not job:
        print(f"未找到任务 {args.job_id}")
        return 4
    print(f"job_id : {job['job_id']}")
    print(f"阶段   : {job['stage']}  {o.stage_label(job['stage'])}")
    print(f"源     : {job.get('source_uri')}")
    print(f"目的   : {job.get('dest_uri')}")
    print(f"大小   : {o.fmt_size(job.get('bytes_total', 0))} / {job.get('objects_total', 0)} 对象")
    if job.get("error"):
        print(f"错误   : {job['error']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ssh_transfer", description="杭州 OSS→新加坡→泰国 SSH 迁移链")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan", help="估算+打印计划，不建 job")
    sp.add_argument("source")
    sp.set_defaults(func=_cmd_plan)
    sa = sub.add_parser("apply", help="建 job 并跑完两段")
    sa.add_argument("source")
    sa.add_argument("--force", action="store_true", help="越过审批阈值强制下发")
    sa.set_defaults(func=_cmd_apply)
    ss = sub.add_parser("status", help="实时查某 job 状态")
    ss.add_argument("job_id")
    ss.set_defaults(func=_cmd_status)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
