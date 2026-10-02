#!/bin/bash
# Array job for the fast experiments: base_no_collapse and
# extended_no_collapse, 3 seeds each (manifest.py indices 0-5). Single
# training runs (not 10-generation collapse loops), so these are the
# quickest group -- run this before run_medium.slurm / run_slow.slurm.
#
#   sbatch slurm/run_fast.slurm
#   squeue -u $USER
#
#SBATCH --job-name=icl-fast
#SBATCH --partition=<PARTITION>        # TODO: set after test_smoke.slurm succeeds
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:45:00                 # generous vs. local ~15-20min worst case; tighten once you've timed one
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

module purge
module load python/3.11   # TODO: must match slurm/setup_env.sh
source .venv/bin/activate

# Match torch/BLAS thread count to the CPUs Slurm actually allocated (default is
# one thread per physical core on the node, which oversubscribes the allocation).
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/run_array_task.py --task-id "$SLURM_ARRAY_TASK_ID"
