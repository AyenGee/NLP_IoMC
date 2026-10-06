#!/bin/bash
# Bundle results into one tarball to copy back from the cluster.
#
#   bash slurm/package_results.sh light     # CSV/JSON/PNG + Slurm logs only (a few MB)
#   bash slurm/package_results.sh full      # ... plus every checkpoint (*.pt) and dataset snapshot
#
# "light" is what you need once experiments/analyze_circuits.py has run on the
# cluster (it turns the checkpoints/snapshots into small CSVs). "full" is only
# for working with the tensors locally (~0.5 GB for the original 12 collapse
# runs; more with the extra seeds). Then, from your own machine:
#   scp <user>@<mscluster-login-host>:<repo>/nlp_results_light.tar.gz .
#
# Run from the repository root. Safe to re-run; overwrites the tarball.

set -euo pipefail
mode="${1:-light}"
case "$mode" in light|full) ;; *) echo "usage: bash slurm/package_results.sh [light|full]" >&2; exit 1;; esac

out="nlp_results_${mode}.tar.gz"
list="$(mktemp)"
trap 'rm -f "$list"' EXIT

{
  find results -type f \( -name '*.csv' -o -name '*.json' \) 2>/dev/null
  find abstract/figures -type f -name '*.png' 2>/dev/null
  find slurm/logs -type f \( -name '*.out' -o -name '*.err' \) 2>/dev/null
  if [ "$mode" = "full" ]; then
    find results -type f -name '*.pt' 2>/dev/null
  fi
} | sort -u > "$list"

n=$(wc -l < "$list")
[ "$n" -gt 0 ] || { echo "nothing to package - are you in the repo root?" >&2; exit 1; }
tar -czf "$out" -T "$list"
echo "packaged $n files -> $out ($(du -h "$out" | cut -f1))"
