"""Resolve one Slurm array task index to a job from manifest.py and run it.
One call = one independent experiment run; mscluster runs these in parallel
across nodes via a job array.

    python experiments/run_array_task.py --task-id $SLURM_ARRAY_TASK_ID                  # group "main"
    python experiments/run_array_task.py --group controls --task-id $SLURM_ARRAY_TASK_ID

The index is relative to the chosen group (default "main", the original 18
runs, so the existing slurm/run_fast|medium|slow.slurm keep working).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import group_jobs, CONFIG_DIR  # noqa: E402
from run_experiment import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--group", default="main")
    args = parser.parse_args()

    jobs = group_jobs(args.group)
    if not (0 <= args.task_id < len(jobs)):
        raise SystemExit(
            f"task-id {args.task_id} out of range [0, {len(jobs)}) for group {args.group!r} "
            "-- check the script's --array matches manifest.py (python experiments/check_manifest.py)"
        )

    job = jobs[args.task_id]
    print(f"[{args.group} task {args.task_id}/{len(jobs) - 1}] {job.config} mode={job.mode} seed={job.seed} data_seed={job.data_seed}")
    result = run(str(CONFIG_DIR / job.config), job.mode, seed=job.seed, data_seed=job.data_seed)
    print(f"[{args.group} task {args.task_id}] done: {result}")


if __name__ == "__main__":
    main()
