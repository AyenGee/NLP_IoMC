#!/bin/bash
# Tier 2b: continue the Tier 1 chains base p=1, base p=0, extended p=1, extended p=0 (8 seeds each, 32 tasks) from
# generation 10 to generation 20. REQUIRES the Tier 1 output directories results/exact/collapse/exact_<cond>_s<seed>/ (at
# least generations.csv, run_config.json and gen0..gen9 model/pool files) to be in place; see CLUSTER_RUN.md.
# One array task = one job of experiments/manifest.py group "continue20", one CPU thread (the code sets torch.set_num_threads(1)).
# Time limit 00:50:00 = 1.5 x the measured local time of the slowest task (33 min, 4 tasks at once on one thread each),
# rounded up to 5 minutes; if the cluster is slower, submit with a larger --time (slurm/submit_tier2.sh TIME_SCALE=...).
# Every chain is RESUMABLE: a task that timed out or was pre-empted is simply resubmitted (sbatch --array=<ids> slurm/t2_continue20.slurm)
# and continues from its last completed generation.
#   python3 experiments/check_manifest.py                  # first: every line OK
#   sbatch slurm/t2_continue20.slurm
#
#SBATCH --job-name=icl-continue20
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
# --stagger: tasks start 1 s apart (plus 0-2 s of jitter), so they do not all create the shared output directories in the same instant
python experiments/run_array_task.py --group continue20 --task-id "$SLURM_ARRAY_TASK_ID" --stagger 1
