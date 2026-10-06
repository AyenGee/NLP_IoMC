#!/bin/bash
# Training-step throughput on ONE core of a compute node (experiments/benchmark.py). Run this first
# so the compute estimates in the rebuild plan use measured numbers, not the development machine's:
#
#   sbatch slurm/benchmark.slurm
#   cat slurm/logs/icl-bench_<jobid>.out
#
#SBATCH --job-name=icl-bench
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python experiments/benchmark.py
