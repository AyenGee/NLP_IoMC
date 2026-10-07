REBUILD: THE EXACT SINGH ET AL. (2024) TASK  (read this first)
================================================================
The first implementation of this project (tag v2-symbolic-pilot, frozen in legacy_symbolic/) used an
ADAPTATION of the Singh et al. task: integer symbols, 4-way 3-shot, MLPs, learned positions, AdamW. It is now
a PILOT. The main implementation (src/, configs/, experiments/, tests/) follows the paper exactly (Section 2.1,
Appendices A and C) and the reference repository's settings; see "Deviations from the paper" below. The pilot's
results stay where they were written (results/collapse, results/analysis, results/retrain, results/tuning,
results/logs, results/checkpoints, cluster_run/); the sections of this file below the line "THE SYMBOLIC
PILOT" describe that earlier work. Regenerate its numbers with
    python experiments/analyze_results.py --pilot

Progress: implemented and tested (tests/test_exact.py): the exact task, the model, the training loop (Phase 1), the
generation-0 replication with head ablations and ACDC (Phase 2), the learning-rate tuning analysis (Phase 3) and the
collapse pipeline with its configs, Slurm scripts and per-generation analysis (Phase 4). Which runs have been done, and
where, is recorded in results/PROVENANCE.md.

1. Get the embeddings (data/, not in git)
   The exemplars are the precomputed Omniglot ResNet18 features from the reference repository. No image and no ResNet
   is touched. File: omniglot_resnet18_randomized_order_s0.h5 (16,638,016 bytes), dataset resnet18/224/feat,
   shape (1623, 5, 512), float32.
       mkdir -p data
       curl -L -o data/omniglot_resnet18_randomized_order_s0.h5 \
         https://github.com/aadityasingh/icl-dynamics/raw/main/omniglot_resnet18_randomized_order_s0.h5
   (If the cluster blocks outbound downloads, copy the file there with scp.) data/*.h5 is git-ignored.

2. Environment and tests
       pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
       pip install -r requirements.txt            # includes h5py
       python tests/test_exact.py                 # last line must be "==== FAILED: none ===="

3. Phase 1 commands
       python experiments/benchmark.py                         # steps/s on one core (sbatch slurm/benchmark.slurm on the cluster)
       python experiments/train_run.py --variant base --steps 20000 --eval-every 250 --name smoke_base --out-dir results/exact/smoke
       python experiments/plot_curves.py results/exact/smoke/logs/smoke_base.csv --out results/exact/smoke/smoke_base.png
   Outputs land in results/exact/: logs/<name>.csv (one row per evaluation point), checkpoints/<name>.pt.

4. What is implemented (Phase 1)
   src/data.py     the exact task: 78,400 training sequences (156,800 extended) enumerated; four evaluation sets
                   (train, val = held-out exemplars, test_exemplars = unseen classes, test_relabel = held-out label pairs)
                   plus the repository's test_class_ex0; fixed seed; documented extended-sequence design.
   src/model.py    attention-only, 2 layers, 8 heads of dim 8, RoPE, pre-LayerNorm, no MLP, no absolute positions;
                   66,629 parameters (base) / 69,879 (extended); ablation and attention-override hooks; generation.
   src/train.py    Adam, constant lr 1e-5, batch 32, no weight decay / warm-up / schedule / clipping / early stopping;
                   step-based; logs loss, accuracy, perplexity on all sets and the induction / previous-token / match
                   scores per head.
   src/metrics.py  the paper's induction strength and the plan's other attention metrics, with documented thresholds.

5. Deviations from the paper (kept up to date; every one is a decision, with its reason)
   - PyTorch instead of JAX/Equinox. Initialisation mimics the repository's (lecun-normal-scale truncated normal
     for linear layers, 0.02 x truncated normal for the label embedding); it cannot be bit-for-bit identical.
   - The exact held-out label pairs cannot be reproduced: the repository draws them with a JAX PRNG (seed 20); we
     shuffle the 10 pairs with NumPy and the same seed number. The split sizes (8 train, 2 held out) match.
   - Budget units: the reference counts train_iters in SEQUENCES (its loop is range(0, train_iters, train_bs)); 1e6
     iterations at batch 32 is 31,250 optimiser steps. src/train.py accepts either unit and logs both.
   - Evaluation: the plan's test_exemplars draws exemplars from 0-4 of the test classes; the repository's
     fsl_test_class uses exemplar 0 only. Both are logged (test_exemplars, test_class_ex0).
   - The extended model's S2 is, in real data, uniform over the episode's two classes (docstring in src/data.py);
     its Bayes-optimal accuracy is 0.5, which is not a failure.
   - EXTENDED LOSS WEIGHTS (a deviation from the plan, which says "equally weighted"): the two auxiliary losses (S2 and l_S2)
     are weighted 0.1, the query-label loss 1 (src/metrics.py EXTENDED_LOSS_WEIGHTS). With equal weights the extended model
     reaches 100% accuracy on its training sequences but does NOT form the paper's circuit: accuracy on held-out label pairs is
     0.27-0.55 and induction strength about 0, at lr 1e-5 (up to 100,000 steps), 1e-4 and 3e-4. A controlled sweep at lr 1e-4
     (seed 0, 15,000 steps) isolates the cause: the 50-way S2 loss (an in-weights task: name a training class) at weight 1.
     Held-out-pair accuracy: weights (1,0,0) 1.00; (1,0,1) 1.00; (1,0.1,0.1) 1.00; (1,0.1,1) 1.00; (1,0.3,1) 0.99; (1,0.5,1) 0.69;
     (1,1,0) 0.73; (1,1,1) 0.27-0.45. With (1,0.1,0.1) the extended model reaches the Bayes-optimal S2 accuracy (0.49-0.51 vs
     0.5) and l_S2 >= 0.99 AND the paper's circuit at lr 1e-5, 3e-5 and 1e-4. The equal-weight runs are kept as evidence
     (results/exact/tuning_table.md, block "extended_equal"). Logs: results/exact/logs/lw_*.csv.
   - The repository samples training batches online from the finite space; we draw 32 sequences uniformly with
     replacement from the enumerated pool each step (the same distribution).
   - Collapse pipeline decisions (src/collapse.py docstring has the full list): one pool size, N = 78,400, for every
     generation >= 1 of both variants (generation 0 uses the whole real space: 78,400 base / 156,800 extended); the
     synthetic share is drawn without replacement from the previous model's pool (every real context once), the real share
     without replacement from the real space; the same initialisation in every generation of a run but a different
     batch-order seed per generation. For the base variant the real space is exactly N, so the p = 0 control retrains on the
     whole real space each generation (the finite space leaves no fresh data); for the extended variant it draws a fresh
     78,400-sequence real pool each time.
   - Synthetic targets are stored in three extra integer columns of each row (lq, s2c, ls2; -1 = real data).
   - Hardware note: runs on this repository's development machine took about 70-120 steps/s per single-threaded run
     depending on how many ran at once (src/train.py); see experiments/benchmark.py and slurm/benchmark.slurm for the cluster.

6. Results so far and Tier 2 (see results/RESULTS_NOTES.md, results/PROVENANCE.md, CLUSTER_RUN.md)
   Tier 1 (48 chains, lr 3e-4, 160,000 sequences = 5,000 steps per generation, run locally):
       python experiments/run_array_task.py --group tier1 --task-id <0..47>     (the Slurm array does this: slurm/run_tier1.slurm)
       python experiments/analyze_collapse_circuits.py --group-id <0..5>        (per-generation ablations, ACDC, embeddings)
       python experiments/analyze_exact.py --collapse                           (summary JSON + figures)
       python experiments/analyze_gen0_s2.py                                    (the generation-0 S2 head on real data)
   Tier 2 (152 jobs in six Slurm array jobs, priority order a-f): weak generator, continuing four Tier 1 conditions to generation 20,
   sampling temperature, accumulate vs replace, a confirmation at the paper's lr 1e-5, and a 3-seed auxiliary-loss weight sweep.
   The jobs are listed in experiments/manifest.py; slurm/t2_*.slurm are the array scripts, slurm/smoke_tier2.sh a one-job smoke test of every
   new code path, slurm/submit_tier2.sh submits everything and records the job IDs, slurm/package_results_exact.sh packs the outputs and
   CLUSTER_RUN.md is the step-by-step list of what to do by hand. Chains are resumable (src/collapse.py): re-running a task continues from
   its last completed generation, and a 10-generation chain can be continued to 20 with the same result as one 20-generation run.
       python experiments/progress.py [--groups ... --verbose]                   finished / partly done / not started, and the ids to resubmit
   New CollapseConfig options (defaults reproduce Tier 1): mode "accumulate" (real pool + all earlier synthetic pools, Gerstgrasser et al.
   2024), gen0_steps (a weak generator). The pilot's extended abstract, tables and figures are in abstract/pilot_symbolic/ (moved, not edited).

THE SYMBOLIC PILOT (earlier work, kept for reference)
=====================================================
MODEL COLLAPSE AND IN-CONTEXT LEARNING: DOES RECURSIVE TRAINING DEGRADE
INDUCTION HEADS?
========================================================================

This repository implements a small transformer trained on the synthetic
few-shot in-context-learning ("induction") task from Singh et al. (2024),
plus a model-collapse pipeline (Shumailov et al., 2023/2024) that trains a
sequence of generations, each on data partly or wholly sampled from the
previous generation's model, and interpretability tooling to track how the
induction-head circuit responds.

See abstract/pilot_symbolic/main.tex for the full write-up, background, and discussion.
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
The first batch ("main" group) runs every experiment with 3 seeds (0, 1, 2; see
experiments/manifest.py) for statistical robustness -- see Section 5 for why that
matters here. That's 6 experiment conditions x 3 seeds = 18 independent runs.
The follow-up and corrected-protocol runs (Section 6g-6i) are further job groups
in the same manifest, which is the single authoritative list of what exists.

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

This is the script behind every number and the quantitative figures in the
abstract. It writes results/analysis_summary.json (all quoted numbers: failure
counts with intervals, the 2 x 2 protocol design, generation-0 training
tables, hyper-parameter study, circuit and ACDC summaries) and, in
abstract/pilot_symbolic/figures/: fig_main.png and fig_pools.png (original-protocol runs),
fig_v2.png (corrected-protocol runs), fig_training.png (generation-0 learning
curves), fig_failure_map.png (every model). Figures that need results which
are not present are skipped. Figures are drawn per seed, NOT as mean +- std:
outcomes are bimodal (a generation's model either learns induction, ~100%
accuracy, or fails to, ~25%), so a mean describes no model that exists and its
std band spills outside [0, 1].

The qualitative figures (attention maps, per-head offset profiles, embedding
PCA) come from the second script, which needs the model checkpoints (*.pt):

    python experiments/make_figures.py                         # default: v2_extended_p1_seed0
    python experiments/make_figures.py --run v2_extended_p1_seed3
    sbatch slurm/make_figures.slurm                             # mscluster, after the runs finish

NOTE: the *.pt checkpoints and gen{g}_data.pt dataset snapshots stay on the
cluster when results are copied back (only CSV/JSON/PNG are retrieved), so
make_figures.py cannot be re-run locally, and every analysis that needs them
(experiments/analyze_circuits.py, analyze_acdc.py, retrain_generation.py) is run
there as Slurm jobs, which write small CSV/JSON files that analyze_results.py
reads. Use `bash slurm/package_results.sh full` to copy the tensors too (about
1-2 GB with all runs) if you want to work with them locally.


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
       then pull results/ and abstract/pilot_symbolic/figures/ back to your machine
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

    h) CORRECTED-PROTOCOL RERUN (decides how much collapse is real). The
       follow-ups in step g) showed that the ORIGINAL protocol -- early
       stopping on validation loss, plus duplicate sequences in the
       generation >= 1 pools (with-replacement resampling) -- manufactured much
       of the apparent collapse: retraining first failures without early
       stopping recovered 4 of 6, and removing duplicates cut p=1 failures from
       22/27 to 1/27. The "v2" job group reruns everything with a FIXED
       150-epoch budget, no early stopping, and no duplicates. Conditions
       (configs/v2_*.yaml, regenerated by experiments/make_v2_configs.py), seeds
       0-7, each seed with its OWN real-data pool: extended p=1, p=0.5, p=0.25,
       base p=1, "v2_extended_p1_dups" (early stopping off but duplicates KEPT)
       and "v2_control_real" (p=0: fresh real data every generation under the
       same protocol, the baseline that tells synthetic-data effects apart from
       effects of retraining itself). With the original runs and the
       no-duplicates controls this gives a 2 x 2 design at p=1 (early stopping
       on/off x duplicates on/off). 48 tasks, ~1-1.3 h each.

    i) THE FINAL RUN ORDER (everything that is left to run), with what to
       check at each step. Nothing here overwrites earlier results.

       Stage 0 -- code onto the cluster and verify it there (a few minutes):
            git pull
            source .venv/bin/activate
            python experiments/check_manifest.py        # every line must say OK
            sbatch slurm/test_pipeline.slurm            # then: tail -1 slurm/logs/icl-test_<jobid>.out
                                                        #   must read "==== FAILED: none ===="

       Stage 1 -- hyper-parameter study (30 short tasks, well under an hour):
            sbatch slurm/run_tune.slurm
            # when `squeue -u $USER` is empty:
            python experiments/tune_hparams.py --summarize
          Selection uses VALIDATION accuracy only (rule fixed in the script).
          It prints KEEP or CHANGE for the settings the experiments use
          (lr 1e-3, weight decay 0.1, d_model 64, 4 heads, batch 128). On KEEP
          go straight to stage 2. On CHANGE, regenerate the v2 configs with the
          selected values, e.g.
            python experiments/make_v2_configs.py --set train.lr=0.003 --set train.weight_decay=0.01
          re-run check_manifest.py and the test, and tell whoever is writing the
          abstract: the earlier (original-protocol) runs then used different
          settings from v2, and the abstract must say so.

       Stage 2 -- the corrected-protocol experiment (48 tasks in parallel):
            sbatch slurm/run_v2.slurm
          Check slurm/logs/icl-v2_<jobid>_<task>.err for every task; resubmit
          only failed task numbers with `sbatch --array=<ids> slurm/run_v2.slurm`.

       Stage 3 -- analyses that need the checkpoints (after stage 2 finished):
            sbatch slurm/run_circuit_analysis.slurm   # per-head scores incl. offset-resolved ones, ablations, pool structure; ALL runs
            sbatch slurm/run_acdc.slurm               # edge-level ACDC circuits, 8 tasks
            python experiments/retrain_generation.py --prefix v2_ --list
            sbatch slurm/run_retrain_v2.slurm         # 450-epoch retrain of each failing v2 run's first failure (budget check)
            sbatch slurm/make_figures.slurm           # attention maps, head profiles, PCA for v2_extended_p1_seed0

       Stage 4 -- bring the results back and analyse:
            bash slurm/package_results.sh light
            scp <user>@<login-host>:<repo>/nlp_results_light.tar.gz .
          unpack into the repo, then locally:
            python experiments/analyze_results.py
          which writes results/analysis_summary.json (every number to quote)
          and the figures in abstract/pilot_symbolic/figures/, then
            python experiments/make_tex_tables.py
          which writes the abstract's tables (abstract/pilot_symbolic/tables/*.tex) from that
          summary. Rebuild abstract/pilot_symbolic/main.tex (Overleaf: upload the abstract/pilot_symbolic/
          folder and ethics/) and check the two-page limit. Keep results/ and the
          Slurm logs in the submission archive (they are not in git).

       Results of the run described in the abstract: tuning jobs 63405 (+63555),
       corrected runs 63867, circuit scores 64119, ACDC 64130, v2 retraining
       64133, figures 64134. Everything in the abstract is reproduced by stages
       0-4 above.

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
      acdc.py          edge-level ACDC circuit discovery (Conmy et al., 2023)
      circuits.py      per-head / per-layer zero-ablation (node-level)
      attribution.py   stretch: lightweight TracIn-style influence proxy (not applied to the collapse runs)
      metrics.py       loss/accuracy/perplexity/entropy/distribution-drift, induction scores
                       (classic and offset-resolved), pool-structure statistics
      utils.py         seeding, config loading, CSV/JSON logging
    configs/          one YAML config per experiment (see Section 3);
                      v2_*.yaml are generated by experiments/make_v2_configs.py
    experiments/      manifest.py (named job groups), run_experiment.py,
                      run_array_task.py, run_all.py, check_manifest.py,
                      tune_hparams.py (validation-based hyper-parameter study),
                      make_v2_configs.py (corrected-protocol configs),
                      analyze_results.py (all report numbers + quantitative figures),
                      analyze_circuits.py (per-head scores, ablation, pool structure),
                      analyze_acdc.py (ACDC circuits per generation),
                      retrain_generation.py (failed generations, no early stop),
                      make_figures.py (attention maps / head profiles / PCA from checkpoints)
    tests/            test_pipeline.py: end-to-end regression tests on tiny configs
    slurm/            mscluster job scripts (see Section 6), package_results.sh,
                      slurm/logs/
    results/          logs (CSV), checkpoints, collapse-run data/models, figures
    abstract/         pilot_symbolic/: the pilot extended abstract (main.tex, tables/, figures/); style files for the new report
    ethics/           NeurIPS 2024 ethics checklist (placeholder, see file) +
                      Faculty of Science AI ethics statement


8. SCOPE DEVIATIONS FROM THE LITERAL PROJECT BRIEF
------------------------------------------------------
  - The task is ADAPTED from Singh et al. (2024), not a copy of their setup.
    Theirs (checked against the paper): Omniglot images through a frozen
    ResNet18 (most experiments: 50 classes, 1 exemplar each), a 2-way 1-shot
    context of 2 exemplar-label pairs, a 2-layer attention-only transformer with
    8 heads per layer, RoPE, Adam at 1e-5, batch 32. Ours: K=64 integer symbols,
    L=8 labels, 4 symbols repeated 3 times (12 context pairs), a 2-layer, 4-head
    transformer with MLPs and learned positions, AdamW at 1e-3 with a cosine
    schedule, batch 128. What is preserved is the induction requirement: labels
    are redrawn per sequence, so only copying from the context solves it, and the
    loss of the base variant is on the last token only.
  - Exemplars are integer symbol IDs, not Omniglot character images. The
    image encoder in Singh et al. (2024) is an implementation detail for
    perceptual realism; the induction-head mechanism this project studies
    does not depend on it, and a symbolic vocabulary is far easier to
    control (exact K, exact repeat structure) and interpret. Documented in
    data.py's module docstring and in the abstract's Method section.
  - Embedding geometry is analysed with PCA rather than PCA+UMAP, to avoid
    an extra heavyweight dependency; PCA was sufficient to inspect the
    embedding structure this project needed.
  - Circuit discovery: acdc.py implements ACDC's graph (embedding, heads, MLPs;
    query/key/value inputs of every head), resample-ablation patching of single
    edges, and the greedy threshold search, on a batch-level KL metric. It is
    checked by two invariants (all edges kept = the model; all cut = the
    corrupted run). Differences from the paper: a corrupted batch made of other
    random task sequences (or the same symbols with redrawn labels) instead of
    hand-designed corrupted prompts, no per-position patching, one metric, and a
    small grid of thresholds. circuits.py's per-head zero-ablation is the
    coarser node-level companion.
  - Mechanistic data attribution (Chen et al., 2026) is NOT implemented; the
    lightweight gradient-influence proxy in attribution.py exists but has not
    been applied to the collapse runs, so the report does not claim it.
  - The training pools of the corrected-protocol runs hold 20,000 sequences, as
    in the original runs; hyper-parameters come from the small grid in
    tune_hparams.py (learning rate, weight decay, width, heads, batch size), not
    from an exhaustive search, and depth (2 layers) and pool size are fixed.
