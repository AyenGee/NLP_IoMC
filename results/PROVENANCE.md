# Provenance of the exact-task runs

Records what was run, with which code and settings, and where the outputs are. (`results/` is git-ignored; this file
is the index. Cluster job IDs are added under "Cluster" as jobs are submitted.)

Code: branch `exact-singh-rebuild` of the repository; the pilot is tag `v2-symbolic-pilot` (code in `legacy_symbolic/`,
results untouched in `results/collapse`, `results/analysis`, `results/retrain`, `results/tuning`, `results/logs`,
`results/checkpoints`, `cluster_run/`). Embeddings: `data/omniglot_resnet18_randomized_order_s0.h5`
(16,638,016 bytes, from github.com/aadityasingh/icl-dynamics).

## Local runs (development machine, 1 thread per run, torch 2.13.0+cpu)

| What | Name pattern | Settings | Output |
|---|---|---|---|
| Smoke runs (Phase 1) | `smoke_base`, `smoke_extended` | seed 0, 20,000 steps, lr 1e-5 | `results/exact/smoke/` |
| Generation-0 replication (Phase 2) | `gen0_{base,extended}_s{0,1,2}` | 31,250 steps (1e6 sequences), batch 32, Adam lr 1e-5, eval every 250 steps, checkpoints at steps 5000, 7500, 10000, 12500, 15000, 20000 | `results/exact/{logs,checkpoints}/`, summary `results/exact/gen0_summary.json` |
| Extended-model diagnostics | `gen0long_extended_s{0,1,2}` (100,000 steps, lr 1e-5); `diag_extended_lr1e-4_s0` (31,250 steps, lr 1e-4) | as above | `results/exact/{logs,checkpoints}/` |
| Weighted extended runs (aux. losses 0.1; used for collapse) | `gen0w_extended_s{0,1,2}` (lr 1e-5), `tunew_extended_lr{0.00003,0.0001,0.0003}_s{0,1,2}`; weight sweep `lw_*` | 31,250 steps (sweep: 15k), loss weights (1, 0.1, 0.1) | same |
| Learning-rate tuning (Phase 3) | `tune_base_lr{0.00003,0.0001,0.0003}_s{0,1,2}`; equal-weight extended `tune_extended_lr*` (evidence only) | 31,250 steps, batch 32, eval every 250 | `results/exact/{logs,checkpoints}/`, summary `results/exact/tuning_summary.json`, `tuning_table.md` |

lr 1e-5 cells of the tuning grid are the generation-0 runs above; the extended lr 1e-4 seed 0 cell is `diag_extended_lr1e-4_s0`
(identical configuration). Runs are deterministic given the seed: the first 5,000 steps of `gen0_base_s0` reproduce `smoke_base`
to every printed digit.

## Cluster (mscluster)

Tier 1 was not run on the cluster (see "Tier 1 collapse" above). Tier 2 (Gate 4 decision of 7 Oct 2026) is submitted by the user with
`bash slurm/submit_tier2.sh` (CLUSTER_RUN.md); the job IDs are recorded here when given:

| group | what | tasks | Slurm job ID | submitted | time limit | state |
|---|---|---|---|---|---|---|
| weakgen | weak generator | 50 | (not yet submitted) | | 00:15 | |
| continue20 | Tier 1 conditions to generation 20 | 32 | (not yet submitted) | | 00:50 | |
| temperature | T = 0.5, 2, greedy | 30 | (not yet submitted) | | 00:45 | |
| accumulate | accumulate vs replace | 16 | (not yet submitted) | | 00:45 | |
| lr1e5 | paper's lr 1e-5 confirmation | 12 | (not yet submitted) | | 02:40 | |
| sweep | S2-weight sweep, 3 seeds | 12 | (not yet submitted) | | 00:10 | |
| smoke test | `slurm/smoke_tier2.sh` | 1 | (not yet run) | | 00:45 | |
| unit tests on the cluster | `slurm/test_exact.slurm` | 1 | (not yet run) | | 00:30 | |

Code state for Tier 2: branch `exact-singh-rebuild`, commit recorded in `slurm/logs/submitted_tier2_<timestamp>.txt` by `submit_tier2.sh`.
Local checks done before handing over: `tests/test_exact.py` (127 checks, including resume equivalence and manifest/Slurm consistency);
`slurm/smoke_tier2.sh` run locally; the replay of the sampling generator reproduces every saved pool of `exact_base_p1_s0` and
`exact_extended_p05_s1` (generations 1-9, 100% of rows), which is what makes the continuation to generation 20 exact.

## Generation-0 S2 head on real data (local, 7 Oct 2026)

`python experiments/analyze_gen0_s2.py` -> `results/exact/gen0_s2_quality.{json,md}` (8 Tier 1 generation-0 models and the 3 lr 1e-5 `gen0w_extended` runs).

## Tier 1 collapse (run locally, not on the cluster)

48 chains (6 conditions x 8 seeds x 10 generations; lr 3e-4, 160,000 sequences = 5,000 steps per generation; weights (1, 0.1, 0.1)),
`python experiments/run_array_task.py --group tier1 --task-id <i>` through an `xargs -P 4` queue (`results/exact/runlogs/tier1_cmds.txt`,
logs `t1_*.out`, 6 Oct 2026 evening to 7 Oct 2026 early morning, about 6 h wall). Circuit analysis (`analyze_collapse_circuits.py --group-id 0..5`,
logs `circ_g*.out`) and `analyze_exact.py --collapse`. Outputs: `results/exact/collapse/`, `results/exact/analysis/`,
`results/exact/analysis_summary_exact.json` (copy `results/analysis_summary_exact.json`), `results/RESULTS_NOTES.md`.

## Tuning gate decision (Phase 3)

Selected lr 3e-4 (base and extended, same circuit as the paper's lr in all 3 seeds, phase change 750-1,000 steps, validation within
0.02 of the best); per-generation budget 160,000 sequences = 5,000 steps. Configs: `python experiments/make_configs.py --lr 3e-4
--sequences 160000` (see the note in each `configs/exact_*.yaml`).
