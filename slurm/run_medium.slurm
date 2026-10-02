#!/bin/bash
# Array job for the medium-length experiments: extended_collapse,
# ablation_p025, ablation_p05 -- 10 generations each, 3 seeds each
# (manifest.py indices 9-17). The extended variant converges quickly per
# generation (confirmed ~8-10 min/generation locally on CPU), so 10
# generations lands around 80-100 min worst case.
#
#   sbatch slurm/run_medium.slurm
#
#SBATCH --job-name=icl-medium
#SBATCH --partition=<PARTITION>        # TODO: set after test_smoke.slurm succeeds
#SBATCH --array=9-17
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:30:00                 # generous vs. local ~80-100min worst case; tighten once timed
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
