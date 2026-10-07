# Running Tier 2 on the Wits mscluster (steps to do by hand)

Everything below is what you type; nothing is submitted for you. `<repo>` is the repository directory on the cluster, `<user>@<host>` your login.
Branch: `exact-singh-rebuild`. Tier 1 (48 chains) was run locally; Tier 2 is 152 jobs in six Slurm array jobs, submitted in the priority order of the Gate 4 decision.

| order | group | what | tasks | time limit per task | local single-task time | about CPU-hours |
|---|---|---|---|---|---|---|
| a | `weakgen` | weak generator: generation 0 trained 250/500/750/1000/1500 steps, then 2 generations; base and extended, 5 seeds | 50 | 00:15 | 7 min | 4.5 |
| b | `continue20` | Tier 1 chains base p=1, base p=0, extended p=1, extended p=0 (8 seeds) from generation 10 to 20 | 32 | 00:50 | 33 min | 12.6 |
| c | `temperature` | T = 0.5, 2, greedy; fully synthetic; base and extended; 5 seeds | 30 | 00:45 | 29 min | 11.8 |
| d | `accumulate` | accumulate vs replace; fully synthetic; base and extended; 8 seeds | 16 | 00:45 | 30 min | 6.3 |
| e | `lr1e5` | the paper's lr 1e-5 and 31,250 steps/generation; base p=1, p=0, extended p=1, p=0; 3 seeds; 10 generations | 12 | 02:40 | 106 min | 19.1 |
| f | `sweep` | S2 loss weight in {1, 0.5, 0.3, 0.1} (l_S2 weight 1), lr 1e-4, 15,000 steps, 3 seeds | 12 | 00:10 | 6 min | 1.0 |

Time limits are 1.5 x the measured local time (4 tasks at once, 1 thread each), rounded up to 5 minutes. About 55 CPU-hours in all; how long it
takes depends on how many of your tasks the cluster runs at once (a `QOSMaxJobsPerUserLimit` pending reason is normal, the rest simply wait).
Every chain is **resumable**: a task that times out or is pre-empted is resubmitted and continues from its last completed generation.

## 1. On your machine

```bash
git push -u origin exact-singh-rebuild                 # I have not pushed anything; the cluster needs this branch
bash slurm/package_continue_inputs.sh                  # -> continue20_inputs.tar.gz (about 190 MB): generations 0-9 of the 32 chains of group b
scp continue20_inputs.tar.gz <user>@<host>:<repo>/
scp data/omniglot_resnet18_randomized_order_s0.h5 <user>@<host>:<repo>/data/        # skip if the cluster has it, or download it there (README.txt section 1)
```

(If you do not use git on the cluster, copy the repository without `results/`, `data/*.h5`, `.venv/` and `cluster_run/` instead.)

## 2. On the cluster: code, environment, inputs

```bash
cd <repo>
git fetch origin && git checkout exact-singh-rebuild && git pull
mkdir -p data slurm/logs
tar xzf continue20_inputs.tar.gz                       # creates results/exact/collapse/exact_{base_p1,base_p0,extended_p1,extended_p0}_s{0..7}/
ls data/                                               # must list omniglot_resnet18_randomized_order_s0.h5 (16,638,016 bytes)
bash slurm/setup_env.sh                                # builds .venv with torch 2.13.0 (CPU) and h5py; safe to re-run
source .venv/bin/activate && python -c "import torch, h5py; print(torch.__version__, h5py.__version__)"     # 2.13.0 ...
python3 experiments/check_manifest.py                  # every line OK
```

Use torch 2.13.0 (the version the Tier 1 chains were run with): continuing a chain replays its sampling generator from the saved models, which
reproduced the Tier 1 pools exactly here, and that is guaranteed only for the same torch version. Different versions would still run; the continued
generations would then be statistically, not bit-for-bit, the same chain.

## 3. Tests and the smoke test (each ALONE: nothing else of yours should be running)

```bash
sbatch slurm/test_exact.slurm                          # ~5 min
tail -1 slurm/logs/icl-test_<jobid>.out                # must read: ==== FAILED: none ====
sbatch slurm/smoke_tier2.sh                            # ~5-10 min; one tiny chain of every new type, writes only results/smoke_tier2/
tail -4 slurm/logs/icl-smoke2_<jobid>.out              # must end: SMOKE OK
grep -A4 "single-core throughput" slurm/logs/icl-smoke2_<jobid>.out
```

The last command prints the node's training speed. On the development machine one thread does about 119 (base) and 94 (extended) steps/s alone and about
51 with four jobs at once. If the cluster is clearly slower (say more than 1.5 x), submit with `TIME_SCALE`: `TIME_SCALE=2 bash slurm/submit_tier2.sh`.
If the test or the smoke run fails, stop and send me the log; do not submit.

## 4. Submit everything

```bash
bash slurm/submit_tier2.sh --dry-run                   # prints the six sbatch lines, checks the manifest and that the 32 generation-9 inputs are present
bash slurm/submit_tier2.sh                             # submits a, b, c, d, e, f in that order
cat slurm/logs/submitted_tier2_*.txt                   # the six job IDs: send them to me (I record them in results/PROVENANCE.md)
```

To submit only some groups: `ONLY="weakgen temperature" bash slurm/submit_tier2.sh`.

## 5. Check progress

```bash
squeue -u $USER -r | awk '{print $3}' | sort | uniq -c           # jobs by name
sacct -X -S today --format=JobID%18,JobName%14,State,Elapsed,ExitCode | grep -v COMPLETED      # anything not completed
source .venv/bin/activate && python experiments/progress.py      # finished / partly done / not started per group, from the files on disk
python experiments/progress.py --groups lr1e5 --verbose          # one line per unfinished job
```

Resubmitting what did not finish (use the "unfinished array ids" line of `progress.py`; finished generations are kept):

```bash
sbatch --array=<ids> --time=<longer limit> slurm/t2_<group>.slurm          # e.g. sbatch --array=3,7-9 --time=01:30:00 slurm/t2_continue20.slurm
```

Logs: `slurm/logs/icl-<group>_<jobid>_<task>.out/.err`. A task's output is in `results/exact/collapse/<run name>/` (collapse chains) or
`results/exact/logs/` and `results/exact/checkpoints/` (the sweep).

## 6. Bring the results back

```bash
# on the cluster, when progress.py shows everything finished
sacct -X -S <first submission date> --format=JobID%18,JobName%14,State,Elapsed,ExitCode > results/exact/sacct_tier2.txt
bash slurm/package_results_exact.sh                    # results_exact_<timestamp>.tar.gz (add --light to leave out the per-generation pools; --with-logs adds slurm/logs)
# on your machine, in the repository root
scp <user>@<host>:<repo>/results_exact_<timestamp>.tar.gz .
tar xzf results_exact_<timestamp>.tar.gz               # merges into results/exact/ (the first 10 generations of the continued chains are identical copies)
```

Then tell me the archive name and the job IDs, and I run the analysis, update `results/RESULTS_NOTES.md`, regenerate the figures and add one figure per
new experiment.

## What each experiment writes (for reference)

| group | output | the chain's generations.csv has |
|---|---|---|
| weakgen | `results/exact/collapse/exact_{variant}_weak{steps}_s{seed}/` | 3 rows; generation 0 is the weak generator (its `steps` column is the stage) |
| continue20 | `results/exact/collapse/exact_{condition}_s{seed}/` (the Tier 1 directories) | 20 rows; `run_config.json` lists the resume |
| temperature | `.../exact_{variant}_{T0p5,T2,Tgreedy}_s{seed}/` | 10 rows |
| accumulate | `.../exact_{variant}_accum_s{seed}/` | 10 rows; `pool_pool_size` grows N, 2N, ... 10N; `pool_new_*` describe the newest synthetic pool; `gen<g>_pool.npz` is that pool |
| lr1e5 | `.../exact_lr1e5_{condition}_s{seed}/` | 10 rows |
| sweep | `results/exact/logs/lw3_s2w{w}_s{seed}.csv`, `results/exact/checkpoints/lw3_s2w{w}_s{seed}.pt` | (a training log, not a chain) |
