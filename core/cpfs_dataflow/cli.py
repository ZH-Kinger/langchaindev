#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPFS 预热/沉降 CLI。

    python -m core.cpfs_dataflow.cli list <fs_id>
    python -m core.cpfs_dataflow.cli preheat <cpfs_dir|cpfs://fs/dir/> [oss://bucket/prefix/]
    python -m core.cpfs_dataflow.cli sink    <cpfs_dir|cpfs://fs/dir/> [oss://bucket/prefix/]
    python -m core.cpfs_dataflow.cli status  <job_id>
"""
import argparse
import sys


def _utf8_stdout():
    # Windows 控制台默认 GBK，标签含 ↔ 等字符会 UnicodeEncodeError；统一切到 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main(argv=None):
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="CPFS DataFlow 预热/沉降 CLI")
    sub = ap.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="列出文件系统下现有 DataFlow")
    p_list.add_argument("fs_id")

    p_disc = sub.add_parser("discover", help="枚举所有配置 fs 的 CPFS↔OSS 绑定映射表")
    p_disc.add_argument("--refresh", action="store_true", help="强制重新发现并刷新缓存")

    for name, help_ in (("preheat", "OSS→CPFS 加载"), ("sink", "CPFS→OSS 沉降")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("cpfs_path", help="CPFS 目录，cpfs://<fs>/<dir>/ 或裸目录")
        p.add_argument("oss", nargs="?", default="", help="可选 oss://bucket/prefix/")
        p.add_argument("--dry-run", action="store_true", help="只解析计划，不提交任务")

    p_status = sub.add_parser("status", help="查询任务状态")
    p_status.add_argument("job_id")

    args = ap.parse_args(argv)

    from core.cpfs_dataflow import engine_nas, orchestrator, discovery
    from core.cpfs_dataflow.orchestrator import DataflowPathError

    if args.action == "discover":
        opts = discovery.refresh() if args.refresh else discovery.get_options()
        if not opts:
            print("(未发现绑定；检查 CPFS_FILE_SYSTEM_IDS 配置与 nas 读权限)")
            return
        for o in opts:
            print(f"{o['label']}  [df:{o['data_flow_id']}]")
        print(f"\n共 {len(opts)} 条（cri 镜像仓库已屏蔽）")
        return

    if args.action == "list":
        try:
            flows = engine_nas.list_dataflows(args.fs_id)
        except engine_nas.NasDataflowError as e:
            sys.exit(f"error: {e}")
        if not flows:
            print("(无 DataFlow)")
            return
        for f in flows:
            print(f"{f['data_flow_id']}  {f.get('status','')}  "
                  f"{f.get('source_storage','')}  fs:{f.get('fs_path','')}")
        return

    if args.action == "status":
        job = orchestrator.get_job(args.job_id)
        if not job:
            sys.exit(f"job not found: {args.job_id}")
        print(f"job {args.job_id}: stage {job['stage']}  {job['operation_label']}")
        print(f"  fs:{job['fs_id']} df:{job.get('data_flow_id','-')} task:{job.get('task_id','-')}")
        print(f"  dir:{job['directory']} -> {job.get('dst_directory') or '(binding)'}")
        print(f"  progress {job.get('files_done',0)}/{job.get('files_total',0)} files, "
              f"{orchestrator.fmt_size(job.get('bytes_done',0))}/{orchestrator.fmt_size(job.get('bytes_total',0))}")
        if job.get("error"):
            print(f"  error: {job['error']}")
        return

    # preheat / sink
    try:
        plan = orchestrator.make_plan(args.action, args.cpfs_path, args.oss)
    except DataflowPathError as e:
        sys.exit(f"path error: {e}")

    print(f"operation: {orchestrator.operation_label(plan.operation)} ({plan.action})")
    print(f"fs:        {plan.fs_id} [{plan.edition}]  region:{plan.region}")
    print(f"cpfs dir:  {plan.cpfs_dir}")
    print(f"oss:       {plan.oss_bucket}/{plan.oss_prefix}" if plan.oss_bucket else "oss:       (由 DataFlow 绑定决定)")
    print(f"Directory: {plan.directory}  DstDirectory: {plan.dst_directory or '(default)'}")

    if args.dry_run:
        print("\n[dry-run] 未提交任务。去掉 --dry-run 实际执行。")
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
