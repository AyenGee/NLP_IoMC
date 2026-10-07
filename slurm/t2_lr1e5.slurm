#!/bin/bash
# Tier 2e: confirmation of the Tier 1 conclusions at the paper's lr 1e-5 and 1,000,000 sequences (31,250 steps) per generation:
# base p=1, base p=0, extended p=1, extended p=0, 3 seeds, 10 generations. 12 tasks.
# One array task = one job of experiments/manifest.py group "lr1e5", one CPU thread (the code sets torch.set_num_threads(1)).
# Time limit 02:40:00 = 1.5 x the measured local time of the slowest task (106 min, 4 tasks at once on one thread each),
# rounded up to 5 minutes; if the cluster is slower, submit with a larger --time (slurm/submit_tier2.sh TIME_SCALE=...).
# Every chain is RESUMABLE: a task that timed out or was pre-empted is simply resubmitted (sbatch --array=<ids> slurm/t2_lr1e5.slurm)
# and continues from its last completed generation.
#   python3 experiments/check_manifest.py                  # first: every line OK
#   sbatch slurm/t2_lr1e5.slurm
#
#SBATCH --job-name=icl-lr1e5
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:40:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
# --stagger: tasks start 1 s apart (plus 0-2 s of jitter), so they do not all create the shared output directories in the same instant
python experiments/run_array_task.py --group lr1e5 --task-id "$SLURM_ARRAY_TASK_ID" --stagger 1
