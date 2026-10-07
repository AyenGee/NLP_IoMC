#!/bin/bash
# Smoke test of every NEW Tier 2 code path on the cluster: one tiny chain of each type, driven through the same
# experiments/run_array_task.py entry point and manifest ids that the real array jobs use (40-step generations, 2-4 generations).
# It writes only to results/smoke_tier2/ (deleted at the start of each run) and never touches results/exact/.
#
#   sbatch slurm/smoke_tier2.sh                  # a compute node, about 5-10 min (about 4 min locally; the sampling of 78,400 contexts per generation dominates)
#   tail -3 slurm/logs/icl-smoke2_<jobid>.out    # the last line must read "SMOKE OK"
#
# Types checked: weak generator (generation 0 trained for fewer steps), continuation / resume (a chain run to generation 2 and then
# continued to 4 gives the same first two rows and writes two more), temperature (greedy), accumulate (the training pool grows
# R + S_1 + S_2), lr 1e-5 config, the two loss-weight sweeps ("train" jobs), plus the single-core step throughput.
#
#SBATCH --job-name=icl-smoke2
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=results/smoke_tier2
rm -rf "$OUT"
TINY=(--override steps=40 --override sequences=null --override eval_every=20)
run() { # run <group> <run name> [extra run_array_task args]
  local g=$1 n=$2; shift 2
  local id; id=$(python3 experiments/manifest.py --find "$g" "$n")
  echo; echo "=== [$g] $n (task $id) $*"; SECONDS=0
  python experiments/run_array_task.py --group "$g" --task-id "$id" --out-root "$OUT" "$@"
  echo "--- $n took ${SECONDS}s"
}
fail() { echo "SMOKE FAILED: $*"; exit 1; }
rows() { python - "$1" <<'PY'
import csv, sys
print(len(list(csv.DictReader(open(sys.argv[1])))))
PY
}

python3 experiments/check_manifest.py || fail "check_manifest.py"

# 1. weak generator: generation 0 trained for 20 steps, then 2 more generations (the real jobs run 10 further generations: 11 in all)
run weakgen exact_extended_weak750_s0 "${TINY[@]}" --override gen0_steps=20 --override n_generations=3
[ "$(rows $OUT/collapse/exact_extended_weak750_s0/generations.csv)" = 3 ] || fail "weak generator: expected 3 generations"
python experiments/run_array_task.py --group weakgen --task-id "$(python3 experiments/manifest.py --find weakgen exact_extended_weak750_s0)" --dry-run | grep -q "n_generations': 11" || fail "weak generator job should ask for 11 generations"
python - <<'PY' || fail "weak generator: generation 0 should have 20 steps and the later ones 40"
import csv
r = list(csv.DictReader(open("results/smoke_tier2/collapse/exact_extended_weak750_s0/generations.csv")))
assert [int(float(x["steps"])) for x in r] == [20, 40, 40], [x["steps"] for x in r]
PY

# 2. continuation: 2 generations, then the same task again asking for 4 (it must resume, not restart)
run continue20 exact_extended_p1_s0 "${TINY[@]}" --override n_generations=2
cp "$OUT/collapse/exact_extended_p1_s0/generations.csv" "$OUT/first_two.csv"
run continue20 exact_extended_p1_s0 "${TINY[@]}" --override n_generations=4 | tee "$OUT/resume.log"
grep -q "resuming: generations 0..1 are done" "$OUT/resume.log" || fail "continuation did not report a resume"
python - <<'PY' || fail "continuation: rows 0-1 changed or rows 2-3 are missing"
import csv
a = list(csv.DictReader(open("results/smoke_tier2/first_two.csv")))
b = list(csv.DictReader(open("results/smoke_tier2/collapse/exact_extended_p1_s0/generations.csv")))
assert len(b) == 4 and [x["generation"] for x in b] == ["0", "1", "2", "3"]
for x, y in zip(a, b):                     # the completed generations are kept untouched
    assert x == y
PY
# a completed chain is skipped when asked again
run continue20 exact_extended_p1_s0 "${TINY[@]}" --override n_generations=4 | tee "$OUT/skip.log"
grep -q "already done" "$OUT/skip.log" || fail "a finished chain was not skipped"

# 3. temperature: greedy decoding
run temperature exact_extended_Tgreedy_s0 "${TINY[@]}" --override n_generations=2
[ "$(rows $OUT/collapse/exact_extended_Tgreedy_s0/generations.csv)" = 2 ] || fail "greedy: expected 2 generations"

# 4. accumulate: the training pool is R + S_1 + S_2 = 3 x 78,400 at generation 2
run accumulate exact_extended_accum_s0 "${TINY[@]}" --override n_generations=3
python - <<'PY' || fail "accumulate: the pool did not grow as R + S_1 + ... + S_g"
import csv
r = list(csv.DictReader(open("results/smoke_tier2/collapse/exact_extended_accum_s0/generations.csv")))
sizes = [int(float(x["pool_pool_size"])) for x in r]
assert sizes == [156800, 2 * 78400, 3 * 78400], sizes
PY

# 5. lr 1e-5 config (tiny budget here)
run lr1e5 exact_lr1e5_extended_p1_s0 "${TINY[@]}" --override n_generations=2
# 6. loss-weight sweeps (lr 3e-4 / 5,000 steps and the lr 1e-4 / 15,000 steps second check): a single training each
run sweep lw3_s2w0.3_s0 --override steps=40 --override eval_every=20
[ -f "$OUT/checkpoints/lw3_s2w0.3_s0.pt" ] && [ -f "$OUT/logs/lw3_s2w0.3_s0.csv" ] || fail "sweep: no checkpoint or log written"
run sweep1e4 lw3b_lr1e-4_s2w0.3_s0 --override steps=40 --override eval_every=20
[ -f "$OUT/checkpoints/lw3b_lr1e-4_s2w0.3_s0.pt" ] || fail "sweep1e4: no checkpoint written"

# 7. throughput of one core on this node (compare with the development machine: base 119, extended 94 steps/s alone, ~51 with 4 jobs at once)
echo; echo "=== single-core throughput"; python experiments/benchmark.py
echo
echo "SMOKE OK"
