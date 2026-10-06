#!/bin/bash
# Tier 1 collapse experiments (manifest group "tier1": 6 conditions x 8 seeds = 48 jobs, one 10-generation chain each).
# One CPU thread per task (parallelism comes from the array). Order of work:
#   python3 experiments/check_manifest.py        # every line OK
#   sbatch slurm/test_exact.slurm                # last line "==== FAILED: none ===="
#   sbatch slurm/run_tier1.slurm
# Output: results/exact/collapse/<condition>_s<seed>/ . Resubmit only failed tasks with
#   sbatch --array=<ids> slurm/run_tier1.slurm
#
# Cost: 10 generations x one training run each; the time limit below is generous (see README for measured speeds).
#
#SBATCH --job-name=icl-tier1
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-47
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
# --stagger: tasks start 2 s apart, so they do not all create the shared output directories in the same instant
python experiments/run_array_task.py --group tier1 --task-id "$SLURM_ARRAY_TASK_ID" --stagger 2
