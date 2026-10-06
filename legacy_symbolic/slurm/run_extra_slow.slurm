#!/bin/bash
# Array job for MORE SEEDS of the base collapse condition (manifest group
# "extra_slow", 5 tasks): base_collapse x seeds 3-7, each with its own
# real-data pool (data_seed = seed). The base variant is the longest job
# (~150-200 min for 10 generations on one CPU), so it is kept separate, like
# run_slow.slurm. Base failures were rare (2/27 trainings, one seed), so more
# seeds matter most here for pinning down how often it fails.
#
#   python experiments/check_manifest.py
#   sbatch slurm/run_extra_slow.slurm
#
#SBATCH --job-name=icl-extra-slow
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00                 # same as run_slow.slurm
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/run_array_task.py --group extra_slow --task-id "$SLURM_ARRAY_TASK_ID"
