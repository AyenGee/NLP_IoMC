#!/bin/bash
# Pack the exact-task outputs on the cluster for download:
#
#   bash slurm/package_results_exact.sh                # results/exact/ + the Tier 2 submission records (job IDs)
#   bash slurm/package_results_exact.sh --light        # same, without the per-generation pools (gen*_pool.npz, the bulk of the size)
#   bash slurm/package_results_exact.sh --with-logs    # also slurm/logs/ (the .out/.err of every job)
#
# Writes results_exact_<timestamp>[_light].tar.gz in the repository root, then prints its size and the scp line. On your own
# machine unpack it inside the repository root:  tar xzf results_exact_*.tar.gz   (it contains paths under results/exact/ and slurm/logs/,
# so it merges into the existing tree; do not delete your local results/exact/collapse first).
# The pools are needed only to recompute pool statistics or re-sample; the analysis scripts use generations.csv and the models.
set -euo pipefail
cd "$(dirname "$0")/.."

LIGHT=0; LOGS=0
for a in "$@"; do
  case "$a" in
    --light) LIGHT=1 ;;
    --with-logs) LOGS=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done
[ -d results/exact ] || { echo "results/exact does not exist here"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="results_exact_${STAMP}$([ "$LIGHT" = 1 ] && echo _light).tar.gz"
ARGS=(results/exact)
ls slurm/logs/submitted_tier2_*.txt >/dev/null 2>&1 && ARGS+=(slurm/logs/submitted_tier2_*.txt)
if [ "$LOGS" = 1 ]; then ARGS+=(slurm/logs); fi
EXCL=()
[ "$LIGHT" = 1 ] && EXCL+=(--exclude='gen*_pool.npz')
tar czf "$OUT" "${EXCL[@]}" "${ARGS[@]}"
ls -l "$OUT"
echo "download:  scp <user>@<cluster-host>:$(pwd)/$OUT .      (then:  tar xzf $OUT  in the repository root)"
