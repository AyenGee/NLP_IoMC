"""Resolve one Slurm array task index to a job from manifest.py and run it (one collapse chain per task).

    python experiments/run_array_task.py --group tier1 --task-id $SLURM_ARRAY_TASK_ID

The index is relative to the group. Output: results/exact/collapse/<config stem>_s<seed>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from manifest import CONFIG_DIR, group_jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--group", default="tier1")
    ap.add_argument("--out-root", default="results/exact/collapse")
    ap.add_argument("--stagger", type=float, default=0.0, help="sleep task_id * this many seconds first (avoids start-up races on a shared filesystem)")
    args = ap.parse_args()
    jobs = group_jobs(args.group)
    if not (0 <= args.task_id < len(jobs)):
        raise SystemExit(f"task-id {args.task_id} out of range [0, {len(jobs)}) for group {args.group!r} "
                         "-- check the script's --array against experiments/manifest.py (python3 experiments/check_manifest.py)")
    if args.stagger:
        time.sleep(args.task_id * args.stagger)
    job = jobs[args.task_id]

    from collapse import CollapseConfig, run_collapse
    from utils import load_config

    kw = dict(load_config(CONFIG_DIR / job.config)["collapse"])
    name = f"{Path(job.config).stem}_s{job.seed}"
    kw.update(seed=job.seed, out_dir=str(Path(args.out_root) / name))
    cfg = CollapseConfig(**kw)
    print(f"[{args.group} task {args.task_id}/{len(jobs) - 1}] {name}: {cfg.variant} p={cfg.p_synthetic} lr={cfg.lr} "
          f"{cfg.sequences} sequences x {cfg.n_generations} generations", flush=True)
    t = time.time()
    run_collapse(cfg)
    print(f"[{args.group} task {args.task_id}] done in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
