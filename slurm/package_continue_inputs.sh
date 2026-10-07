#!/bin/bash
# RUN LOCALLY (Git Bash or any Linux/Mac shell), where the Tier 1 chains were run. Packs what the cluster needs to continue the
# 32 chains base p=1, base p=0, extended p=1, extended p=0 (seeds 0-7) from generation 10 to 20: per chain
# generations.csv, run_config.json and, for generations 0-9, the model, the pool and the training log.
#
#   bash slurm/package_continue_inputs.sh            # writes continue20_inputs.tar.gz (about 190 MB) in the repository root
#
# On the cluster, in the repository root:   tar xzf continue20_inputs.tar.gz      (creates results/exact/collapse/exact_*_s*/)
# The generation-0..8 models are needed too: the chains' sampling generator is replayed from them (the Tier 1 runs did not save it).
set -euo pipefail
cd "$(dirname "$0")/.."
FILES=()
for cond in base_p1 base_p0 extended_p1 extended_p0; do
  for s in 0 1 2 3 4 5 6 7; do
    d=results/exact/collapse/exact_${cond}_s${s}
    [ -f "$d/generations.csv" ] || { echo "missing $d/generations.csv: the Tier 1 outputs are not here"; exit 1; }
    FILES+=("$d/generations.csv" "$d/run_config.json")
    for g in 0 1 2 3 4 5 6 7 8 9; do
      for k in model.pt pool.npz train.csv; do
        [ -f "$d/gen${g}_${k}" ] || { echo "missing $d/gen${g}_${k}"; exit 1; }
        FILES+=("$d/gen${g}_${k}")
      done
    done
  done
done
tar czf continue20_inputs.tar.gz "${FILES[@]}"
ls -l continue20_inputs.tar.gz
echo "files: ${#FILES[@]}   (expected 32 x (2 + 30) = 1024)"
