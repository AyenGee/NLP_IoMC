"""Reproduce every core experiment in the abstract, serially, in one process.

    python experiments/run_all.py

This is the LOCAL / fallback path -- it runs all 18 (experiment x seed)
combinations from manifest.py one after another, which is simple but does
not use multiple machines. On mscluster, prefer slurm/run_experiments.slurm,
which runs the same 18 combinations as a parallel Slurm job array (one node
per combination) via experiments/run_array_task.py -- see README.txt.

Each run is checkpointed to results/ as it completes, so this script (and
the array-job path) can be safely re-run; it will just overwrite whichever
combinations you re-run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import all_combinations, CONFIG_DIR
from run_experiment import run


def main():
    combos = all_combinations()
    t_start = time.time()
    for i, (config_name, mode, seed) in enumerate(combos):
        config_path = CONFIG_DIR / config_name
        t0 = time.time()
        print(f"\n{'=' * 70}\n[run_all {i + 1}/{len(combos)}] {config_name} mode={mode} seed={seed}\n{'=' * 70}")
        result = run(str(config_path), mode, seed=seed)
        print(f"[run_all] Finished in {time.time() - t0:.1f}s: {result}")
    print(f"\n[run_all] All {len(combos)} runs complete in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
