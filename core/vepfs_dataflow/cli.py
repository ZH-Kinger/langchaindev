#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""火山 vePFS 预热/沉降 CLI。

    python -m core.vepfs_dataflow.cli preheat <vepfs://fs/dir/|/dir/> <tos://bucket/prefix/>
    python -m core.vepfs_dataflow.cli sink    <vepfs://fs/dir/|/dir/> <tos://bucket/prefix/>
    python -m core.vepfs_dataflow.cli status  <job_id>

预热=TOS→vePFS(Import)，沉降=vePFS→TOS(Export)。dry-run 默认只解析不提交。
"""
import argparse
import sys


def _utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main(argv=None):
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="vePFS DataFlow 预热/沉降 CLI")
    sub = ap.add_subparsers(dest="action", required=True)

    for name, help_ in (("preheat", "TOS→vePFS 加载"), ("sink", "vePFS→TOS 沉降")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("vepfs_path", help="vePFS 目录，vepfs://<fs>/<dir>/ 或裸目录")
        p.add_argument("tos", help="tos://bucket/prefix/")
        p.add_argument("--same-name", default="", help="同名策略 Skip/KeepLatest/OverWrite（默认 Skip）")
        p.add_argument("--apply", action="store_true", help="实际提交任务（默认 dry-run）")

    p_status = sub.add_parser("status", help="查询任务状态")
    p_status.add_argument("job_id")

    args = ap.parse_args(argv)

    from core.vepfs_dataflow import orchestrator
    from core.vepfs_dataflow.orchestrator import DataflowPathError

    if args.action == "status":
        job = orchestrator.get_job(args.job_id)
        if not job:
            sys.exit(f"job not found: {args.job_id}")
        print(f"job {args.job_id}: stage {job['stage']}  {job['operation_label']}")
        print(f"  fs:{job['fs_id']} task:{job.get('task_id','-')} region:{job['region']}")
        print(f"  vePFS:{job['sub_path']}  TOS:{job.get('tos_bucket')}/{job.get('tos_prefix','')}")
        print(f"  progress {job.get('files_done',0)}/{job.get('files_total',0)} files, "
              f"{orchestrator.fmt_size(job.get('bytes_done',0))}/{orchestrator.fmt_size(job.get('bytes_total',0))}")
        if job.get("error"):
            print(f"  error: {job['error']}")
        return

    # preheat / sink
    try:
        plan = orchestrator.make_plan(args.action, args.vepfs_path, args.tos, same_name=args.same_name)
    except DataflowPathError as e:
        sys.exit(f"path error: {e}")

    print(f"operation: {orchestrator.operation_label(plan.operation)} ({plan.action})")
    print(f"fs:        {plan.fs_id}  region:{plan.region}")
    print(f"vePFS:     {plan.sub_path}")
    print(f"TOS:       {plan.tos_bucket}/{plan.tos_prefix}")
    print(f"same-name: {plan.same_name or 'Skip'}")

    if not args.apply:
        print("\n[dry-run] 未提交任务。加 --apply 实际执行。")
        return

    job = orchestrator.create_job_record(plan)
    print(f"\n-> job {job['job_id']} 已创建，提交并轮询...")

    def _on_update(j):
        print(f"  [{j['stage']}] {orchestrator.fmt_size(j.get('bytes_done',0))} "
              f"{j.get('files_done',0)} files"
              + (f"  error: {j['error']}" if j.get("error") else ""))

    job = orchestrator.run_to_completion(job, on_update=_on_update)
    if job["stage"] == orchestrator.STAGE_DONE:
        print(f"done: {orchestrator.fmt_size(job.get('bytes_done',0))} / {job.get('files_done',0)} files")
    else:
        sys.exit(f"failed: {job.get('error') or 'unknown'}")


if __name__ == "__main__":
    main()
