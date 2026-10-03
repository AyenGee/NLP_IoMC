MODEL COLLAPSE AND IN-CONTEXT LEARNING: DOES RECURSIVE TRAINING DEGRADE
INDUCTION HEADS?
========================================================================

This repository implements a small transformer trained on the synthetic
few-shot in-context-learning ("induction") task from Singh et al. (2024),
plus a model-collapse pipeline (Shumailov et al., 2023/2024) that trains a
sequence of generations, each on data partly or wholly sampled from the
previous generation's model, and interpretability tooling to track how the
induction-head circuit responds.

See abstract/main.tex for the full write-up, background, and discussion.
This file covers how to reproduce the code's results, either locally or on
the Wits mscluster HPC cluster.


1. ENVIRONMENT SETUP (LOCAL)
--------------------------------
Developed and tested with Python 3.12 on Windows, CPU-only.

    python -m venv .venv
    .venv\Scripts\activate            (Windows)   or:  source .venv/bin/activate   (Linux/Mac)

    # CPU-only PyTorch (what this repo was developed against):
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

    # OR, if you have a GPU: install the matching CUDA build of torch from
    # https://pytorch.org/get-started/locally/ FIRST, then:
    pip install -r requirements.txt

For mscluster, use slurm/setup_env.sh instead -- see Section 6.

All commands below assume your working directory is the repository root
(the directory containing this file) and that your venv is activated.


2. QUICK SANITY CHECK (~1 minute)
-----------------------------------
Confirms the induction-head phase transition reproduces on a tiny CPU-fast
config before trusting anything larger:

    python src/train.py --config configs/smoke_base.yaml

Expect training accuracy to jump from near-chance to ~100% around epoch
8-14, with the printed induction score climbing well above its printed
random-baseline value at the same point.

For a fuller check of the whole pipeline (tiny configs, ~1-2 minutes, writes
only to a temp directory, never to results/):

    python tests/test_pipeline.py        # last line must be "FAILED: none"
    sbatch slurm/test_pipeline.slurm     # same, on an mscluster compute node


3. REPRODUCING EVERY EXPERIMENT / FIGURE IN THE ABSTRACT
-----------------------------------------------------------
Every experiment is run with 3 seeds (0, 1, 2; see experiments/manifest.py)
for statistical robustness -- see Section 5 for why that matters here.
That's 6 experiment conditions x 3 seeds = 18 independent runs.

LOCAL (serial, one process, see Section 5 for timing):

    python experiments/run_all.py

MSCLUSTER (parallel, recommended -- see Section 6 for the full workflow):

    sbatch slurm/test_smoke.slurm      # 1 small job first, always
    sbatch slurm/run_fast.slurm        # base_no_collapse, extended_no_collapse (6 jobs)
    sbatch slurm/run_medium.slurm      # extended_collapse, ablation_p025, ablation_p05 (9 jobs)
    sbatch slurm/run_slow.slurm        # base_collapse (3 jobs)

Either path writes the same files to results/:
  1. configs/base_no_collapse.yaml      (train)    -> results/logs/base_no_collapse_seed{0,1,2}.csv,
                                                        results/checkpoints/base_no_collapse_seed{0,1,2}.pt
  2. configs/extended_no_collapse.yaml  (train)    -> results/logs/extended_no_collapse_seed{0,1,2}.csv, ...
  3. configs/base_collapse.yaml         (collapse) -> results/collapse/base_collapse_seed{0,1,2}/
  4. configs/extended_collapse.yaml     (collapse) -> results/collapse/extended_collapse_seed{0,1,2}/
                                                        (this is also the p=1.0 ablation point)
  5. configs/ablation_p025.yaml         (collapse) -> results/collapse/ablation_p025_seed{0,1,2}/
  6. configs/ablation_p05.yaml          (collapse) -> results/collapse/ablation_p05_seed{0,1,2}/

Each collapse run's results/collapse/<name>_seed{N}/ directory contains,
per generation g: gen{g}_model.pt (checkpoint), gen{g}_model.summary.json
(test metrics), gen{g}_data.pt (the *exact* training pool used for that
generation -- real+synthetic mix; stored compactly as stacked int32
tensors, ~4 MB each -- restore the list-of-dicts form with
`from data import load_items; items = load_items(path)`),
gen{g}_train.csv (full per-epoch
training log), plus a top-level generations.csv summarising accuracy/loss/
perplexity/distribution-drift/induction-score for every generation, and
run_config.json recording the exact config used.

To run a single (experiment, seed) combination instead of the whole suite:

    python experiments/run_experiment.py --config configs/base_no_collapse.yaml --mode train --seed 0
    python experiments/run_experiment.py --config configs/extended_collapse.yaml --mode collapse --seed 1

Omit --seed to fall back to the seed baked into the config file (0).
Every run is otherwise seeded for reproducibility. Re-running is safe --
it will just overwrite results/ for whichever (experiment, seed)
combinations you re-run.


4. GENERATING THE FIGURES FROM RESULTS
------------------------------------------
After Section 3's runs have produced results/ (for the reported run, the
mscluster output was copied from cluster_run/ into results/; cluster_run/
also keeps the raw Slurm logs, which are the provenance for the job IDs
quoted in the abstract's supplementary material):

    python experiments/analyze_results.py       # needs only the CSV/JSON files

This is the script behind every number and the two main figures in the
abstract. It writes results/analysis_summary.json (all quoted numbers),
abstract/figures/fig_main.png and abstract/figures/fig_failure_map.png.
Figures are drawn per seed, NOT as mean +- std: outcomes are bimodal (a
generation's model either learns induction, ~100% accuracy, or fails to,
~25%), so a mean describes no model that exists and its std band spills
outside [0, 1].

The qualitative figures (attention maps, embedding PCA) and the older
mean +- std plots come from the second script, which needs the model
checkpoints (*.pt) in results/:

    python experiments/make_figures.py          # local
    sbatch slurm/make_figures.slurm              # mscluster, after all 3 array jobs finish

NOTE: the *.pt checkpoints and gen{g}_data.pt dataset snapshots stayed on the
cluster when results were copied back (only CSV/JSON/PNG were retrieved), so
make_figures.py cannot be re-run locally, and the per-head ablation
(src/circuits.py) and data-property (src/attribution.py) analyses have not
been applied to the collapse runs. Copy those files from the cluster
(about 0.5 GB for all runs; extended_collapse and ablation_p05 are the
informative ones) to run them.


5. HARDWARE, RUNTIME, AND THE BASE VARIANT'S TRAINING DYNAMICS
--------------------------------------------------------------------
Developed and validated on CPU only. Approximate per-run wall-clock times
on a single CPU core, at this project's scale (K=64 symbols, L=8 labels,
n_classes=4, burstiness=3, d_model=64, 2 layers, 4 heads, n_train=20000
sequences):

    - extended variant, one training run to convergence:   ~8-10 minutes
    - extended variant, one collapse generation:            ~8-10 minutes
    - extended variant, full 10-generation collapse run:    ~80-100 minutes
    - base variant, one training run to convergence:        ~15-20 minutes
    - base variant, one collapse generation:                ~15-20 minutes
    - base variant, full 10-generation collapse run:        ~150-200 minutes
    - Full local experiments/run_all.py (18 runs, serial):  several hours

IMPORTANT -- WHY n_train IS 20000, NOT A SMALLER NUMBER (read before
touching configs/base_*.yaml): the base task's only loss is a single
final-position prediction. Trained on a SMALL, FIXED, epoch-repeated pool,
we found this is fragile in two compounding ways:
  (a) SEED SENSITIVITY. Even on perfectly clean data, some random
      initialisations permanently memorise the training sequences (100%
      train accuracy) instead of finding the general induction circuit
      (val/test accuracy stuck near chance), no matter how long you train.
      collapse.py therefore holds the model-init seed FIXED across every
      generation of a single collapse run (only the training DATA changes
      generation to generation) -- varying it per generation would
      confound "collapse effect" with "seed lottery."
  (b) NOISE SENSITIVITY. Even with a good seed, as little as ~0.4%
      query-label noise in a generation's training pool (the base
      variant's only possible corruption channel) was enough to tip
      memorisation into being the lower-training-loss solution over
      genuine induction, with a small pool (n_train=4000).
A ~5x larger pool (n_train=20000) fixes BOTH failure modes structurally --
confirmed directly: a previously-always-stuck seed converged to 100%
accuracy, and 1% injected label noise barely dented accuracy (99.6%) --
because individual sequences are repeated far fewer times per epoch,
making memorisation impractical rather than needing to be out-regularised.
weight_decay=0.1 (moderate; not the much larger values grokking papers
sometimes need) is enough at this pool size.
This is WHY we also added 3-seed replication (Section 3): we had already
empirically proven base-variant accuracy can swing from 99% to ~30% on
seed choice alone before this fix, so trusting any single run (even post-
fix) without seeing seed-to-seed spread would be scientifically weak.
If you shrink n_train or n_epochs/patience for faster iteration, re-check
against results/logs/*.csv first, ideally with patience: null for a few
hundred epochs, to see the true trajectory before trusting a shorter budget.

The extended variant's dense per-position auxiliary supervision did not
show either failure mode in any configuration we tried -- it converges
reliably within ~10-40 epochs regardless of seed, pool size, or weight
decay; its configs use a smaller, cheaper budget for that reason.

Configs use device: auto, so a CUDA-enabled torch install (see Section 1 /
slurm/setup_env.sh) picks up a GPU automatically with no code changes.


6. RUNNING ON MSCLUSTER (WITS HPC)
---------------------------------------
mscluster is a Slurm-managed cluster (~200 nodes, CPU and GPU partitions).
Jobs run on compute nodes via a queue, never directly on the login node.
See slurm/ for the job scripts used below.

    a) Connect and set up the environment (ONCE):
         ssh eazubuike@<mscluster-login-host>
         mkdir -p ~/projects && cd ~/projects
         git clone https://github.com/AyenGee/NLP_IoMC.git
         cd NLP_IoMC
         bash slurm/setup_env.sh
       setup_env.sh builds .venv from mscluster's system python3 (3.14.x;
       no `module load` needed - the same interpreter exists on every
       compute node) and installs a CPU build of torch.

    b) Partition: every slurm/*.slurm uses `batch` (CPU-only; 6 CPUs and
       30 GB per node, 1-day TIMELIMIT, usually many idle nodes). Every
       job here requests <= 4 CPUs, 4 GB and 4 h, so all of them fit.
       (`sinfo` shows the current partitions if this ever changes.)

    c) Test small first (HPC etiquette -- always do this before the real jobs):
         sbatch slurm/test_smoke.slurm
         squeue -u $USER                               # watch it queue/run
         cat slurm/logs/icl-smoke-test_<jobid>.out      # confirm it finished cleanly

    d) Submit the real work, split by expected runtime so no job
       undershoots or overshoots its partition's time limit:
         sbatch slurm/run_fast.slurm      # base_no_collapse + extended_no_collapse, 6 tasks
         sbatch slurm/run_medium.slurm    # extended_collapse + both ablations, 9 tasks
         sbatch slurm/run_slow.slurm      # base_collapse, 3 tasks (longest -- the base
                                           #   variant's larger pool, see Section 5)
       Each is a Slurm job ARRAY: every task is one independent
       (experiment config, seed) combination from experiments/manifest.py,
       so mscluster runs them in parallel across nodes rather than one
       long serial script -- this is the point of using the cluster.

    e) Monitor and clean up:
         squeue -u $USER
         scancel <jobid>                               # if something's wrong
       Check slurm/logs/*.out and *.err for each task BEFORE assuming it
       worked -- don't skip this (it's listed as one of the most common
       cluster misuses).

    f) Once all three array jobs show COMPLETED in squeue/sacct:
         sbatch slurm/make_figures.slurm
       then pull results/ and abstract/figures/ back to your machine
       (scp/rsync) to finish writing the abstract.

    g) FOLLOW-UP VALIDITY CHECKS (added after reading the first results; each
       targets a specific weakness listed in the abstract's Limitations).
       They write to NEW run names / new files, so nothing from step d) is
       overwritten. Run in this order:

       0. Get the new code onto the cluster (git pull), then:
            python experiments/check_manifest.py     # every --array matches its group
            sbatch slurm/test_smoke.slurm            # code changed since the last smoke test

       1. Analyses of the EXISTING runs (cheap; they need the *.pt checkpoints
          and gen*_data.pt snapshots that step d) left in results/ on the
          cluster, so run them before deleting anything there):
            sbatch slurm/run_circuit_analysis.slurm   # -> results/analysis/circuits_*.csv
            python experiments/retrain_generation.py --list    # how many failing cases
            sbatch slurm/run_retrain.slurm            # -> results/retrain/*.json,*.csv
          - circuit analysis: best-single-head induction score, per-head and
            per-layer ablation, and STRUCTURE statistics of every generation's
            training pool (label consistency, repeat counts, ...). Tests whether
            collapse is driven by structural corruption the marginal KL cannot see.
          - retrain: retrains each failed generation with early stopping OFF.
            Tests whether any "failure" was only a delayed transition.

       2. New experiments (about 29 array tasks; none depends on another):
            sbatch slurm/run_controls.slurm       #  9 tasks, ~1-2.5 h each
            sbatch slurm/run_extra_medium.slurm   # 15 tasks, ~1-2.5 h each
            sbatch slurm/run_extra_slow.slurm     #  5 tasks, ~3-4 h each
          - controls: p=0 with resampling, p=0 without, and the main p=1.0
            experiment without duplicate sequences. Tests whether the duplicate-
            sequence resampling in generations >= 1 explains the faster learning
            of later generations and/or the collapse itself.
          - extra seeds: seeds 3-7 for the four collapse conditions, each with its
            OWN real-data pool (data_seed = seed), giving 8 seeds per condition.
          Every new run also logs, per generation, how many sequences in its
          pool are distinct (pool_distinct_frac), pool structure statistics, and
          the best single head's induction score.

       3. Bring results back and analyse:
            bash slurm/package_results.sh light       # a few MB: CSV/JSON/PNG + Slurm logs
            scp <user>@<login-host>:<repo>/nlp_results_light.tar.gz .
          then locally: unpack into the repo and run
            python experiments/analyze_results.py
          It discovers seeds and conditions automatically, so 8 seeds and the
          control conditions are included with no changes. Then update the
          abstract's numbers and Limitations to match: several sentences there
          (failure counts, "3 seeds", the duplication caveat) are only true
          for the first 18 runs.
       ("full" instead of "light" also bundles every *.pt, ~0.5 GB or more.)

Etiquette reminders (from the cluster's own onboarding material): never
run real computation on the login node, request only the resources you
need, test small before scaling up, and clean up files you don't need.


7. REPOSITORY STRUCTURE
---------------------------
    src/
      data.py         synthetic few-shot ICL symbol task generator
      model.py        transformer (base + extended output heads)
      train.py        training loop, checkpointing, metrics logging
      collapse.py      model-collapse generation loop (fit -> sample -> refit)
      interp.py        attention maps, induction/prev-token scores, embedding PCA
      circuits.py      stretch: simplified ACDC-style per-head ablation
      attribution.py   stretch: lightweight TracIn-style influence proxy
      metrics.py       loss/accuracy/perplexity/entropy/distribution-drift utilities
      utils.py         seeding, config loading, CSV/JSON logging
    configs/          one YAML config per experiment (see Section 3)
    experiments/      manifest.py (named job groups), run_experiment.py,
                      run_array_task.py, run_all.py, check_manifest.py,
                      analyze_results.py (all report numbers + main figures),
                      analyze_circuits.py (per-head ablation + pool structure),
                      retrain_generation.py (failed generations, no early stop),
                      make_figures.py (attention maps / PCA from checkpoints)
    slurm/            mscluster job scripts (see Section 6), package_results.sh,
                      slurm/logs/
    results/          logs (CSV), checkpoints, collapse-run data/models, figures
    abstract/         extended abstract (main.tex, CCN 2-page format) + figures/
    ethics/           NeurIPS 2024 ethics checklist (placeholder, see file) +
                      Faculty of Science AI ethics statement


8. SCOPE DEVIATIONS FROM THE LITERAL PROJECT BRIEF
------------------------------------------------------
  - Exemplars are integer symbol IDs, not Omniglot character images. The
    image encoder in Singh et al. (2024) is an implementation detail for
    perceptual realism; the induction-head mechanism this project studies
    does not depend on it, and a symbolic vocabulary is far easier to
    control (exact K, exact repeat structure) and interpret. Documented in
    data.py's module docstring and in the abstract's Method section.
  - Embedding geometry is analysed with PCA rather than PCA+UMAP, to avoid
    an extra heavyweight dependency; PCA was sufficient to inspect the
    embedding structure this project needed.
  - ACDC-style circuit discovery (circuits.py) and lightweight
    influence-style data attribution (attribution.py) are stretch goals;
    see the abstract's Discussion/Limitations for what was attempted and
    the honest outcome.
