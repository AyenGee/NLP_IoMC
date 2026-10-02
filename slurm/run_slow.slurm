#!/bin/bash
# Array job for the slow experiment: base_collapse, 10 generations, 3 seeds
# (manifest.py indices 6-8). The base variant needs a much larger training
# pool (n_train=20000) to avoid a memorisation failure mode we found during
# development (see README.txt / abstract Limitations), so each generation
# is slower than the extended variant's (~15-20 min/generation locally on
# CPU) -- 10 generations lands around 150-200 min worst case. Submit this
# one on its own, ideally to a partition with a longer time limit.
#
#   sbatch slurm/run_slow.slurm
#
#SBATCH --job-name=icl-slow
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=6-8
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00                 # generous vs. local ~150-200min worst case; tighten once timed
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 (3.14) is what .venv was
# built from in slurm/setup_env.sh, and it is present on every compute node.
source .venv/bin/activate

# Match torch/BLAS thread count to the CPUs Slurm actually allocated (default is
# one thread per physical core on the node, which oversubscribes the allocation).
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/run_array_task.py --task-id "$SLURM_ARRAY_TASK_ID"
