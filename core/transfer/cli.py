#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-cloud transfer CLI."""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="cross-cloud transfer CLI")
    sub = ap.add_subparsers(dest="action", required=True)

    p_plan = sub.add_parser("plan", help="parse and estimate; no task is created")
    p_plan.add_argument("source")
    p_plan.add_argument("dest", nargs="?", default="")

    p_apply = sub.add_parser("apply", help="submit and poll until completion")
    p_apply.add_argument("source")
    p_apply.add_argument("dest", nargs="?", default="")
    p_apply.add_argument("--overwrite", choices=["skip", "overwrite", "no", "always"], default="skip",
                         help="same-name policy: skip/no or overwrite/always")
    p_apply.add_argument("--force", action="store_true", help="run over-threshold task without approval prompt")

    p_status = sub.add_parser("status", help="query task status")
    p_status.add_argument("job_id")

    args = ap.parse_args(argv)

    from core.transfer import orchestrator
    from core.transfer.paths import PathError

    if args.action == "status":
        job = orchestrator.get_job(args.job_id)
        if not job:
            sys.exit(f"job not found: {args.job_id}")
        print(f"job {args.job_id}: stage {job['stage']}")
        print(f"  {job['source']} -> {job['dest']} ({job['direction']})")
        print(f"  same-name policy {orchestrator.same_name_policy_label(job.get('same_name_policy') or job.get('overwrite_mode'))}")
        print(f"  size {orchestrator.fmt_size(job.get('bytes_total', 0))} / {job.get('objects_total', 0)} objects")
        if job.get("error"):
            print(f"  error: {job['error']}")
        return

    try:
        plan = orchestrator.make_plan(args.source, args.dest)
    except PathError as e:
        sys.exit(f"path error: {e}")

    if plan.engine not in ("mgw", "tos_mig"):
        sys.exit(f"direction {plan.direction} engine {plan.engine} is not implemented")

    bytes_total, objects_total = orchestrator.estimate_source(plan)
    policy = getattr(args, "overwrite", "skip")
    print(f"source: {plan.source.uri()}")
    print(f"dest:   {plan.dest.uri()}")
    print(f"direction: {plan.direction}  engine: {plan.engine}")
    print(f"estimate: {orchestrator.fmt_size(bytes_total)} / {objects_total} objects")
    print(f"same-name policy: {orchestrator.same_name_policy_label(policy)}")
    over = orchestrator.needs_approval(bytes_total)
    if over:
        print("over approval threshold")

    if args.action == "plan":
        print("\n[dry-run] no task created. Use apply to run.")
        return

    if over and not args.force:
        sys.exit("over approval threshold; add --force or use Feishu admin approval")

    job = orchestrator.create_job_record(
        plan, same_name_policy=args.overwrite,
        bytes_total=bytes_total, objects_total=objects_total)
    print(f"\n-> job {job['job_id']} created, submitting and polling...")

    def _on_update(j):
        print(f"  [{j['stage']}] {orchestrator.fmt_size(j.get('bytes_total', 0))}"
              + (f"  error: {j['error']}" if j.get("error") else ""))

    job = orchestrator.run_to_completion(job, on_update=_on_update)
    if job["stage"] == orchestrator.STAGE_DONE:
        print(f"done: {orchestrator.fmt_size(job.get('bytes_total', 0))} / {job.get('objects_total', 0)} objects")
    else:
        sys.exit(f"failed: {job.get('error') or 'unknown'}")


if __name__ == "__main__":
    main()
