"""Resolve one Slurm array task index to a (config, mode, seed) combination
from manifest.py and run it. One call = one independent experiment run;
mscluster runs these in parallel across nodes via a job array.

    python experiments/run_array_task.py --task-id $SLURM_ARRAY_TASK_ID

See slurm/run_experiments.slurm for how this is invoked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import all_combinations, CONFIG_DIR  # noqa: E402
from run_experiment import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()

    combos = all_combinations()
    if not (0 <= args.task_id < len(combos)):
        raise SystemExit(f"task-id {args.task_id} out of range [0, {len(combos)}) -- check --array matches manifest.py")

    config_name, mode, seed = combos[args.task_id]
    config_path = CONFIG_DIR / config_name
    print(f"[array task {args.task_id}/{len(combos) - 1}] {config_name} mode={mode} seed={seed}")
    result = run(str(config_path), mode, seed=seed)
    print(f"[array task {args.task_id}] done: {result}")


if __name__ == "__main__":
    main()
