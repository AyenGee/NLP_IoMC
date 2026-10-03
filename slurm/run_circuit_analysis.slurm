#!/bin/bash
# Per-generation circuit analysis (best-head induction / previous-token scores, offset-resolved
# induction scores, per-head and per-layer ablation) and training-pool structure statistics for
# the collapse runs, computed where the checkpoints and dataset snapshots live so that only small
# CSVs have to be copied back. See experiments/analyze_circuits.py. One array task per condition
# group (python experiments/analyze_circuits.py --list-groups): the original runs are re-analysed
# too, so that every run carries the offset-resolved columns. A group whose runs do not exist
# (yet) just prints a message and exits. Submit after the collapse jobs have finished.
#
#   sbatch slurm/run_circuit_analysis.slurm
#   # then copy results/analysis/*.csv back (bash slurm/package_results.sh light)
#
# Cost: a few minutes per run (the pool-structure statistics loop over 20,000 sequences per
# generation), at most 8 runs per group.
#
#SBATCH --job-name=icl-circuits
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-10
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/analyze_circuits.py --group-id "$SLURM_ARRAY_TASK_ID"
