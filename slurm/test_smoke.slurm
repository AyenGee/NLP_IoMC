#!/bin/bash
# Single small test job -- run this FIRST, before the real array jobs, per
# HPC etiquette ("test jobs with small resource requests first"). Confirms
# the environment, partition, and code all work before committing a real
# allocation to it. Submit with:
#
#   sbatch slurm/test_smoke.slurm
#   squeue -u $USER                 # watch it run
#   cat slurm/logs/icl-smoke-test_<jobid>.out   # check it finished cleanly
#
#SBATCH --job-name=icl-smoke-test
#SBATCH --partition=<PARTITION>        # TODO: run `sinfo` and pick a short/small CPU partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
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
python src/train.py --config configs/smoke_base.yaml
