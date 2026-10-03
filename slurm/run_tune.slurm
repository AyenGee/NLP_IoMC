#!/bin/bash
# Hyper-parameter study (experiments/tune_hparams.py): 30 cells = 15 settings x
# {base, extended}, each trained for 100 epochs on real data with 4 seeds, scored on
# VALIDATION accuracy only. Run this FIRST, before the corrected-protocol
# experiment (run_v2.slurm), so the learning rate / weight decay / width used
# there are justified rather than assumed:
#
#   python experiments/check_manifest.py
#   sbatch slurm/run_tune.slurm
#   # when all 30 tasks have finished (squeue -u $USER is empty):
#   python experiments/tune_hparams.py --summarize
#
# --summarize prints the winner per variant and says KEEP or CHANGE for the
# settings the main experiments use (lr 1e-3, weight decay 0.1, d_model 64,
# 4 heads, batch 128). If it says CHANGE, see README Section 6i before submitting run_v2.slurm.
#
# Cost: about 15 minutes per task for the default width, up to ~40 for d_model 128;
# 30 tasks in parallel, so well under an hour of wall-clock time once nodes are free.
#
#SBATCH --job-name=icl-tune
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/tune_hparams.py --task-id "$SLURM_ARRAY_TASK_ID"
