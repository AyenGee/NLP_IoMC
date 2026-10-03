#!/bin/bash
# Run the end-to-end regression tests (tests/test_pipeline.py) on a compute
# node: tiny configs, ~1-2 minutes, writes only to a temporary directory. Run it
# after pulling new code and before submitting real jobs.
#
#   sbatch slurm/test_pipeline.slurm
#   cat slurm/logs/icl-test_<jobid>.out      # last line must say "FAILED: none"
#
#SBATCH --job-name=icl-test
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python tests/test_pipeline.py
