#!/bin/bash
# Per-generation circuit analysis (best-head induction/previous-token scores,
# per-head and per-layer ablation) and training-pool structure statistics for
# the collapse runs, computed where the checkpoints and dataset snapshots live
# so that only small CSVs have to be copied back. See
# experiments/analyze_circuits.py. One job covers every run; it is light
# (a few minutes per run). Submit after the collapse jobs have finished.
#
#   sbatch slurm/run_circuit_analysis.slurm
#   # then copy results/analysis/*.csv back (bash slurm/package_results.sh light)
#
#SBATCH --job-name=icl-circuits
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/analyze_circuits.py
