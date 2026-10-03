#!/bin/bash
# Array job for the CONTROL experiments (manifest group "controls", 9 tasks):
#   control_real_resampled, control_real_fresh, extended_collapse_nodup
# x seeds 0-2. They test whether the duplicate-sequence resampling used for
# generations >= 1 (rather than synthetic data) explains (a) the faster
# learning of later generations and (b) the collapse itself. See the header
# of each configs/control_*.yaml and configs/extended_collapse_nodup.yaml.
# Each is a 10-generation extended run, like run_medium.slurm.
#
#   python experiments/check_manifest.py        # confirm --array matches the group
#   sbatch slurm/run_controls.slurm
#
#SBATCH --job-name=icl-controls
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-8
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:30:00                 # same as run_medium.slurm (extended, 10 generations)
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/run_array_task.py --group controls --task-id "$SLURM_ARRAY_TASK_ID"
