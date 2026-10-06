#!/bin/bash
# The CORRECTED-PROTOCOL experiment (manifest group "v2", 48 tasks): 6
# conditions x seeds 0-7, each seed with its own real-data pool. Every
# generation trains for a FIXED 150 epochs with early stopping OFF, on pools
# without duplicate sequences (one condition keeps duplicates, to separate
# the two artifacts; one is a real-data control under the same protocol). See configs/v2_*.yaml and README.txt Section 6h. This is
# the experiment that decides how much collapse is real once the training
# protocol no longer manufactures it.
#
# Expected cost: 150 epochs x ~2-3 s x 10 generations, about 1-1.3 h per task
# on one CPU (no early stopping, so every generation uses the full budget);
# 48 tasks in parallel. Check first:
#   python experiments/check_manifest.py
#   sbatch slurm/test_pipeline.slurm          # code changed since the last test
#   sbatch slurm/run_v2.slurm
# Afterwards: sbatch slurm/run_circuit_analysis.slurm
#
#SBATCH --job-name=icl-v2
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-47
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=03:30:00                 # ~2-3x the expected 1-1.3 h, in case nodes are slower than expected
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/run_array_task.py --group v2 --task-id "$SLURM_ARRAY_TASK_ID"
