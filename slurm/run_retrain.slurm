#!/bin/bash
# Retrain each FIRST FAILING generation with early stopping disabled, from its
# saved training pool, to test whether any "failure" was a delayed transition
# that early stopping cut off (see experiments/retrain_generation.py). Needs
# the gen<g>_data.pt snapshots, so run it on the cluster in the repo that holds
# results/collapse/. One array task per failing (condition, seed) case; the
# main results have 6. The array is sized for 12 cases; tasks past the actual
# number just print a message and exit, so it is safe to leave as is. Count
# the cases first:
#
#   python experiments/retrain_generation.py --list
#   sbatch slurm/run_retrain.slurm
#
# Each task trains two budgets (the original one, then 300 epochs): roughly
# 3 + 11 minutes for the extended variant, a little more for base.
#
#SBATCH --job-name=icl-retrain
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/retrain_generation.py --task-id "$SLURM_ARRAY_TASK_ID"
