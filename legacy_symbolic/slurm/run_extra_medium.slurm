#!/bin/bash
# Array job for MORE SEEDS of the extended conditions (manifest group
# "extra_medium", 15 tasks): extended_collapse (p=1.0), ablation_p05,
# ablation_p025 x seeds 3-7. Unlike seeds 0-2, which all share one real-data
# pool, each of these seeds also gets its OWN real-data pool and validation/
# test sets (data_seed = seed), so they replicate over data draws as well as
# initialisations. 8 seeds in total per condition once combined with the
# original 3.
#
#   python experiments/check_manifest.py
#   sbatch slurm/run_extra_medium.slurm
#
#SBATCH --job-name=icl-extra-medium
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:30:00                 # same as run_medium.slurm
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/run_array_task.py --group extra_medium --task-id "$SLURM_ARRAY_TASK_ID"
