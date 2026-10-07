#!/bin/bash
# Submit every Tier 2 array job, in the priority order of the Gate 4 decision, and record the job IDs.
#
#   bash slurm/submit_tier2.sh --dry-run          # print what would be submitted (also checks the manifest and the inputs)
#   bash slurm/submit_tier2.sh                    # submit
#   TIME_SCALE=1.5 bash slurm/submit_tier2.sh    # time limits x 1.5 (if the smoke test showed the cluster to be slower than the local machine)
#   ONLY="weakgen temperature" bash slurm/submit_tier2.sh    # a subset, still in priority order
#
# Order (the Slurm queue is first-in first-out within a user's priority, so earlier lines start first):
#   a  weakgen      weak generator (50 chains)
#   b  continue20   generation 10 -> 20 of base p=1, base p=0, extended p=1, extended p=0 (32 chains; needs the Tier 1 output dirs)
#   c  temperature  T = 0.5, 2, greedy (30 chains)
#   d  accumulate   accumulate vs replace (16 chains)
#   e  lr1e5        confirmation at the paper's lr 1e-5 (12 chains, the longest)
#   f  sweep        auxiliary-loss weight sweep (12 trainings)
# The job IDs are appended to slurm/logs/submitted_tier2_<timestamp>.txt: send them to the person recording provenance.
# Everything is resumable: a task that times out or is pre-empted is resubmitted with
#   sbatch --array=<task ids> --time=<longer> slurm/t2_<group>.slurm        (it continues from its last completed generation)
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
ORDER=(weakgen continue20 temperature accumulate lr1e5 sweep)
declare -A SCRIPT=([weakgen]=t2_weakgen.slurm [continue20]=t2_continue20.slurm [temperature]=t2_temperature.slurm
                   [accumulate]=t2_accumulate.slurm [lr1e5]=t2_lr1e5.slurm [sweep]=t2_sweep.slurm)
SCALE="${TIME_SCALE:-1}"
SELECT=" ${ONLY:-${ORDER[*]}} "

python3 experiments/check_manifest.py || { echo "check_manifest.py failed: fix before submitting"; exit 1; }
[ -f data/omniglot_resnet18_randomized_order_s0.h5 ] || { echo "data/omniglot_resnet18_randomized_order_s0.h5 is missing (see CLUSTER_RUN.md)"; exit 1; }
[ -d .venv ] || { echo ".venv is missing: run bash slurm/setup_env.sh first"; exit 1; }
if [[ "$SELECT" == *" continue20 "* ]]; then
  have=$(ls results/exact/collapse/exact_{base_p1,base_p0,extended_p1,extended_p0}_s{0..7}/gen9_model.pt 2>/dev/null | wc -l)
  [ "$have" = 32 ] || { echo "continue20 needs the 32 Tier 1 chains' generation-9 files in results/exact/collapse/ (found $have); see CLUSTER_RUN.md step 3"; exit 1; }
fi

LOG="slurm/logs/submitted_tier2_$(date +%Y%m%d_%H%M%S).txt"
[ "$DRY" = 1 ] || { mkdir -p slurm/logs; echo "# $(date)  git $(git rev-parse --short HEAD 2>/dev/null || echo '?')  TIME_SCALE=$SCALE" > "$LOG"; }
for g in "${ORDER[@]}"; do
  [[ "$SELECT" == *" $g "* ]] || continue
  t=$(python3 experiments/manifest.py --time "$g" --scale "$SCALE")
  n=$(python3 experiments/manifest.py --size "$g")
  if [ "$DRY" = 1 ]; then
    echo "would submit: sbatch --parsable --time=$t slurm/${SCRIPT[$g]}     ($n tasks)"
  else
    id=$(sbatch --parsable --time="$t" "slurm/${SCRIPT[$g]}")
    id=${id%%;*}
    echo "$g $id slurm/${SCRIPT[$g]} tasks=$n time=$t" | tee -a "$LOG"
  fi
done
[ "$DRY" = 1 ] || { echo; echo "job IDs recorded in $LOG"; echo "progress:  squeue -u \$USER -r | awk '{print \$3}' | sort | uniq -c    |    sacct -X --format=JobID,JobName%14,State,Elapsed -S today"; }
