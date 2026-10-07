"""Write slurm/t2_*.slurm (one array job per Tier 2 group) from experiments/manifest.py, so that array sizes and time limits cannot drift
from the manifest (check_manifest.py verifies them).

    python3 experiments/make_slurm_tier2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import GROUP_INFO, GROUPS, time_limit  # noqa: E402

DESC = {
    "weakgen": ("Tier 2a: weak-generator test. Generation 0 is trained for only 250/500/750/1000/1500 steps (before, during and after the\n"
                "# phase change), then ten further generations are trained on samples (p = 1, the Tier 1 settings), so compounding versus recovery\n"
                "# can be seen. 50 tasks = {base, extended} x 5 stages x 5 seeds, each a chain of 11 generations."),
    "continue20": ("Tier 2b: continue the Tier 1 chains base p=1, base p=0, extended p=1, extended p=0 (8 seeds each, 32 tasks) from\n"
                   "# generation 10 to generation 20. REQUIRES the Tier 1 output directories results/exact/collapse/exact_<cond>_s<seed>/ (at\n"
                   "# least generations.csv, run_config.json and gen0..gen9 model/pool files) to be in place; see CLUSTER_RUN.md."),
    "temperature": ("Tier 2c: sampling temperature 0.5, 2 and greedy (T = 1 is Tier 1), fully synthetic, base and extended, 5 seeds.\n"
                    "# 30 tasks = {base, extended} x {T0p5, T2, Tgreedy} x 5 seeds."),
    "accumulate": ("Tier 2d: accumulate (real pool + all earlier synthetic pools) instead of replace, fully synthetic, base and extended,\n"
                   "# 8 seeds. 16 tasks."),
    "lr1e5": ("Tier 2e: confirmation at the paper's lr 1e-5 and 1,000,000 sequences (31,250 steps) per generation:\n"
              "# base p=1, base p=0, extended p=1, extended p=0, 3 seeds, 10 generations. 12 tasks."),
    "sweep": ("Tier 2f: auxiliary-loss weight sweep on the extended model at the main collapse setting (lr 3e-4, 5,000 steps), loss weights\n"
              "# (1, w, 1), w in {1, 0.5, 0.3, 0.1}, 3 seeds. 12 single trainings."),
    "sweep1e4": ("Tier 2g: the same sweep at the earlier setting (lr 1e-4, 15,000 steps), as a second check. 12 single trainings."),
}

for g, info in GROUP_INFO.items():
    if g == "tier1":
        continue
    n = len(GROUPS[g])
    text = f"""#!/bin/bash
# {DESC[g]}
# One array task = one job of experiments/manifest.py group "{g}", one CPU thread (the code sets torch.set_num_threads(1)).
# Time limit {time_limit(g)} = 1.5 x the measured local time of the slowest task ({info['est_min']} min, 4 tasks at once on one thread each),
# rounded up to 5 minutes; if the cluster is slower, submit with a larger --time (slurm/submit_tier2.sh TIME_SCALE=...).
# Every chain is RESUMABLE: a task that timed out or was pre-empted is simply resubmitted (sbatch --array=<ids> slurm/{info['slurm']})
# and continues from its last completed generation.
#   python3 experiments/check_manifest.py                  # first: every line OK
#   sbatch slurm/{info['slurm']}
#
#SBATCH --job-name=icl-{g}
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-{n - 1}
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time={time_limit(g)}
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
# --stagger: tasks start 1 s apart (plus 0-2 s of jitter), so they do not all create the shared output directories in the same instant
python experiments/run_array_task.py --group {g} --task-id "$SLURM_ARRAY_TASK_ID" --stagger 1
"""
    (HERE.parent / "slurm" / info["slurm"]).write_text(text, newline="\n")
    print("wrote", info["slurm"], n, time_limit(g))
