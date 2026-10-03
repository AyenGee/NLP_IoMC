"""Run a manifest group serially, in one process (default: the original 18 runs).

    python experiments/run_all.py                    # group "main": 18 runs
    python experiments/run_all.py --group controls

This is the LOCAL / fallback path. On mscluster, prefer the Slurm array
scripts in slurm/, which run the same jobs in parallel across nodes via
experiments/run_array_task.py -- see README.txt.

Each run is checkpointed to results/ as it completes, so this script (and
the array-job path) can be safely re-run; it will just overwrite whichever
runs you re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import group_jobs, CONFIG_DIR
from run_experiment import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="main")
    args = parser.parse_args()
    jobs = group_jobs(args.group)
    t_start = time.time()
    for i, job in enumerate(jobs):
        t0 = time.time()
        print(f"\n{'=' * 70}\n[run_all {args.group} {i + 1}/{len(jobs)}] {job.config} mode={job.mode} seed={job.seed} data_seed={job.data_seed}\n{'=' * 70}")
        result = run(str(CONFIG_DIR / job.config), job.mode, seed=job.seed, data_seed=job.data_seed)
        print(f"[run_all] Finished in {time.time() - t0:.1f}s: {result}")
    print(f"\n[run_all] All {len(jobs)} runs of group {args.group!r} complete in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
