#!/bin/bash
# Run once all three array jobs (run_fast/run_medium/run_slow) have
# completed. Loads model checkpoints and runs forward passes to build
# attention-map/PCA figures, so it's a real (if light) computation --
# submit it rather than running on the login node.
#
#   sbatch slurm/make_figures.slurm
#
#SBATCH --job-name=icl-figures
#SBATCH --partition=<PARTITION>        # TODO: same as the other scripts
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

module purge
module load python/3.11   # TODO: must match slurm/setup_env.sh
source .venv/bin/activate

# Match torch/BLAS thread count to the CPUs Slurm actually allocated (default is
# one thread per physical core on the node, which oversubscribes the allocation).
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/make_figures.py
