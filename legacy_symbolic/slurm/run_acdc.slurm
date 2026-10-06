#!/bin/bash
# Edge-level ACDC circuit discovery (src/acdc.py, experiments/analyze_acdc.py) on every
# generation of the corrected-protocol (v2) runs, the no-duplicates run and the real-data
# controls, one array task per condition group (python experiments/analyze_acdc.py --list-groups).
# It reads the per-generation checkpoints, so run it on the cluster after run_v2.slurm has
# finished (the controls and the no-duplicates run already exist from the first batches):
#
#   sbatch slurm/run_acdc.slurm
#   # then: bash slurm/package_results.sh light
#
# Cost: each group is about 8 runs x 10 generations x 2 thresholds x 2 corruptions at 4-5 s
# each, roughly 1 hour per task.
#
#SBATCH --job-name=icl-acdc
#SBATCH --partition=batch            # mscluster CPU partition: 6 CPUs / 30 GB per node, 1-day limit
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# No `module load`: mscluster's system python3 is what .venv was built from.
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
python experiments/analyze_acdc.py --group-id "$SLURM_ARRAY_TASK_ID"
