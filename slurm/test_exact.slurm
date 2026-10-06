#!/bin/bash
# Run the end-to-end tests (tests/test_exact.py) on a compute node, ALONE: it takes about a minute and writes only to a
# temporary directory. Run it after pulling new code and before submitting real jobs; do not run it while other jobs write
# to results/.
#
#   sbatch slurm/test_exact.slurm
#   tail -1 slurm/logs/icl-test_<jobid>.out        # must read "==== FAILED: none ===="
#
#SBATCH --job-name=icl-test
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python tests/test_exact.py
