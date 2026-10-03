#!/bin/bash
# Budget check for the corrected-protocol (v2) runs: retrain the FIRST FAILING
# generation of each failing v2 run from its saved pool for 450 epochs (3x the
# 150-epoch budget), to see whether any v2 "failure" is just a transition that had
# not happened yet. v2 has no early stopping, so a 150-epoch retrain would only repeat
# the original run; hence the longer budget only. (The 450-epoch schedule is
# stretched too, so this is a test of "would more training have helped", not an exact
# continuation.) Run after run_v2.slurm has finished:
#
#   python experiments/retrain_generation.py --prefix v2_ --list      # how many cases?
#   sbatch slurm/run_retrain_v2.slurm
#
# The array is sized for 20 cases; tasks past the actual number just exit, so it is
# safe to leave as is. If --list shows more than 20, raise the --array range.
# Cost: about 17 minutes per task.
#
#SBATCH --job-name=icl-retrain-v2
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-19
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
python experiments/retrain_generation.py --prefix v2_ --epochs 450 --task-id "$SLURM_ARRAY_TASK_ID"
