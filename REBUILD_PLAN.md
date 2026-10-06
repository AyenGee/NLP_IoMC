# Rebuild plan: exact Singh et al. (2024) task + model collapse

You are working in this repository (NLP_IoMC). Read this whole file before changing anything.
It replaces the current symbolic adaptation with the exact task and architecture from
Singh et al. (2024), keeps the good experimental protocol already built, and adds new
experiments. **Do not write the extended abstract.** The report will be written after the
experiments have been run; your job is the code, the runs, the analysis, the figures and a
results-notes file.

Work in phases. **Stop at each GATE**, show me the numbers or plots it asks for, and wait for
my go-ahead before starting the next phase.

---

## 0. Context

The brief (`_NLP_2026__Mechanistic_Interp_Collapse_Project.pdf`) says: "we are going to be
using the **exact task** from Singh et al. (2024)". It also says "the base model will just
reproduce their work", and it asks for an extended model that also predicts "the next symbol
in the sequence and its corresponding value". The goal: **understand how model collapse
affects in-context learning and induction heads.** Marks come from methodology, insight and
creativity, not raw accuracy. Accuracy is used only to check the implementation is correct.

The current code deviates from the paper in several ways:
- integer symbols instead of Omniglot embeddings;
- 4-way 3-shot (12 pairs) instead of 2-way 1-shot;
- MLPs, learned absolute positions, 4 heads;
- AdamW with a cosine schedule;
- a single i.i.d. test set.

All of these must change. The protocol lessons from the existing runs stay (see §6).

The lecturer said to use the precomputed Omniglot embeddings as they are. **Do not run
ResNet or touch images.**

---

## 1. Preserve the existing work (do this first)

1. If the folder is a git repo, commit the current state and tag it `v2-symbolic-pilot`.
   Otherwise copy `src/`, `configs/`, `experiments/` and `tests/` into
   `legacy_symbolic/`.
2. **Do not delete** anything under `results/`, `cluster_run/`, `abstract/` or `slurm/logs/`.
   Move the old results under `results/pilot_symbolic/` (or leave them where they are and
   record their location in the README).
3. The old results become a **pilot study on a symbolic adaptation**. They will be cited
   briefly in the report and in the supplement, so keep `experiments/analyze_results.py`
   able to regenerate their numbers.

---

## 2. Data: exact task (`src/data.py`, rewrite)

### 2.1 Embeddings
- Source file: `omniglot_resnet18_randomized_order_s0.h5` from
  https://github.com/aadityasingh/icl-dynamics (repo root, about 16.6 MB).
  - Clone or download it into `data/`.
  - If the network blocks you, tell me and I will download it.
  - Add `data/*.h5` to `.gitignore` and document the download in the README.
- Dataset path: `resnet18/224/feat`, shape **(1623, 5, 512)**, float32.
  - The class order in the file is already randomised (`class_order` dataset).
  - Load it once into a float32 tensor.
- **Class split** (repo flag `--class_split 50 1473 100`):
  - train classes = indices `0..49`;
  - test classes = the last 100, indices `1523..1622`;
  - the 1473 in between are unused.
- **Exemplar split** (`--exemplar_split 1 4 0`):
  - training uses exemplar index 0 only (E = 1);
  - exemplars 1–4 of the training classes are held out for validation.

### 2.2 Labels
- L = 5 labels, embedded by a learned embedding.
- Label pairs: the 10 unordered pairs {i, j}, i < j, over 0..4.
  - Shuffle them with a fixed seed (document the seed). 8 pairs go to training, 2 to
    test-relabel (`--fs_relabel 5 --fs_relabel_split 8 2 0`).
  - The repo's exact pair choice comes from JAX RNG with seed 20. We cannot reproduce it
    bit-for-bit; say so in the README.

### 2.3 Sequences (2-way, 1-shot)
Token stream: `A, la, B, lb, Q`.
- A and B are embeddings of two **different** classes.
- `{la, lb}` is a label pair, randomly assigned to A and B.
- Q is a new instance of A's or B's class.
- The target is the label of Q's class.
- Labels are redrawn per sequence, which forces in-context learning.

**Enumerate the full training space** exactly as in Appendix C:

    C(50,2) classes × 8 label pairs × 2 context orders × 2 label assignments × 2 query choices
    = 1225 × 8 × 8 = 78,400 sequences

- Store each sequence as integer indices `(cA, cB, la, lb, q_is_A, exA, exB, exQ)` and look
  up the embeddings at collate time.
- With E = 1, all exemplar indices are 0 for training.

### 2.4 Evaluation sets
Each set has 1,000 sequences, generated with a fixed seed and saved to disk so every model
sees the same sets.

| Name | Classes | Exemplars (A, B, Q) | Label pairs | What it tests |
|---|---|---|---|---|
| `train_eval` | train | 0 | train | fit |
| `val` | train | random from 1–4 | train | **model selection / tuning** (paper's `fsl_train_valex`) |
| `test_exemplars` | 100 test classes | any of 0–4 | train | the **match** step generalises to unseen symbols |
| `test_relabel` | train | 0 | 2 held-out pairs | the **copy** step generalises to unseen label pairs |

This is the train/validation/test split the brief requires. Write unit tests showing that the
sets are disjoint in the sense intended: no test class appears in training, no held-out pair
appears in training, and the validation exemplars are never in training.

### 2.5 Extended sequences
The brief asks the model to also predict "the next symbol in the sequence and its
corresponding value". Implement this literally as a 7-token teacher-forced input with
3 prediction targets:

    input:   A  la  B  lb  Q  lq  S2
    targets:                lq  S2  l_S2      (predicted at positions 4, 5, 6)

- In **real** data, S2 is uniform over {A's class, B's class} (another query from the same
  episode; exemplar 0 for training). `l_S2` is its in-context label.
- Write the justification in a docstring: this is the natural continuation of a few-shot
  episode, and it keeps `l_S2` solvable by induction.
- Bayes-optimal accuracies under real data:
  - lq: 1.0;
  - S2 over 50 classes: **0.5**;
  - l_S2 given S2: 1.0.

  Log them so nobody reads 0.5 as failure.
- Enumerated extended training space = 78,400 × 2 = 156,800.

---

## 3. Model: exact architecture (`src/model.py`, rewrite)

Follow Appendix A of the paper and the repo's `models.py` / `main_utils.py`:

- **Inputs.**
  - Exemplars: `example_embed = nn.Linear(512, 64, bias=True)`.
  - Labels: `label_embed = nn.Embedding(5, 64)`.
  - Interleave the two.
  - **No absolute positional embedding.**
- **2 layers, attention-only.** No MLP blocks at all (repo `mlp_ratio=None`).
- **Each block:**
  1. pre-LayerNorm;
  2. `qkv = nn.Linear(64, 192, bias=False)`;
  3. split into **8 heads** of dimension 8;
  4. **RoPE** on q and k (θ = 10000, interleaved pairs, matching the repo's `apply_rope`);
  5. causal softmax(QKᵀ/√8);
  6. `proj = nn.Linear(64, 64, bias=False)`;
  7. residual add.
  - Write the attention module yourself (`nn.MultiheadAttention` cannot do RoPE).
  - Precompute the RoPE sin/cos tables.
- **Output.**
  - Final LayerNorm.
  - `label_head = nn.Linear(64, 5, bias=True)` read at the query position (base) or at
    positions 4 and 6 (extended).
  - The extended model also has `symbol_head = nn.Linear(64, 50, bias=True)`, read at
    position 5.
- **The forward pass must be able to return:**
  - attention patterns (B, H, T, T) per layer;
  - per-head outputs (before `proj` is summed, so heads can be ablated);
  - residual stream activations.
- **Ablation hooks:** zero any head's output, or replace its attention pattern with a given
  pattern. These are needed in §5.
- **Generation (extended).** Given a real context `A la B lb Q`:
  1. sample lq from `label_head`;
  2. sample S2's class from `symbol_head` (over the 50 training classes);
  3. feed in **exemplar 0 of that class** as the next token;
  4. sample `l_S2`.

  Use temperature T (default 1.0) and batch the whole thing.

Write a unit test that RoPE attention scores depend only on relative position.

---

## 4. Training (`src/train.py`)

- Optimiser: `torch.optim.Adam(lr=1e-5, betas=(0.9, 0.999))`, **constant** learning rate.
  - No weight decay, no warmup, no cosine schedule, no gradient clipping.
- Batch size 32. Each step draws 32 sequences uniformly from the pool (the repo samples
  online from the finite space).
- **Step-based, not epoch-based.**
  - Every 5,000 steps (configurable), log loss, accuracy and perplexity on all four
    evaluation sets plus a running train loss.
  - Log the interpretability metrics from §5.1 at the same points.
  - Save curves to CSV.
- **No early stopping.** Fixed budget (the earlier pilot showed early stopping creates fake
  collapse).
- **Loss.**
  - Base: cross-entropy on the query label only.
  - Extended: CE(lq) + CE(S2) + CE(l_S2), equally weighted. Log each term and each accuracy
    separately.
- `torch.set_num_threads(1)` by default; parallelism comes from Slurm array tasks.
- Benchmark steps per second on one cluster core and report it at GATE 1.

---

## 5. Metrics and interpretability (`src/metrics.py`, `interp.py`, `circuits.py`, `acdc.py`)

### 5.1 Logged during training and per generation
- **Induction strength** for each layer-2 head, exactly as the paper defines it: attention
  from the query to the correct in-context label minus attention to the incorrect one.
  For `A 0 B 1 A` that is attn(4→1) − attn(4→3).
- **Previous-token score** for each layer-1 head: mean attention from label positions to the
  preceding exemplar (1→0, 3→2).
- **Match score** for each layer-2 head: attention from the query to the matching exemplar
  position (offset 0).
- Attention entropy for each head.

### 5.2 Per trained model (every generation of every run)
- **Head ablations, following paper Section 3:**
  - zero each layer-2 head individually → accuracy drop;
  - keep only one layer-2 head → accuracy;
  - zero each layer-1 head;
  - cut the direct layer-1 → output path.
- **Match vs copy:** accuracy on `test_exemplars` (match) and `test_relabel` (copy),
  reported separately.
- **Phase-change timing:** the step at which val accuracy first exceeds 0.9, and the step at
  which loss leaves the ln 2 plateau (define the threshold and document it).
- **ACDC**, rebuilt for the attention-only graph (16 heads with q/k/v inputs, embeddings and
  output; no MLP nodes).
  - KL(model ‖ pruned), τ swept over {0.003, 0.01, 0.03}.
  - Two corruptions: random other sequences, and the same symbols with redrawn labels.
- **Embeddings:**
  - PCA of `example_embed(features)` for the 50 training classes and for test classes;
  - geometry of the label embeddings;
  - cosine similarity of each class's projected embedding between gen 0 and gen n.
- **Extended only, distribution metrics on each generation's training pool:**
  - P(S2 = Q's class) (true value 0.5);
  - P(S2 ∉ {A, B});
  - the S2 class histogram over 50 classes: entropy, and KL to gen 0 against a noise floor
    from two independent real pools;
  - mass on the k least frequent classes (a "tail" measure);
  - correctness rates of lq and l_S2;
  - **base**: correctness rate of the query label in each pool.

### 5.3 Statistics
- Per-seed lines in every plot.
- Exact permutation tests over seeds for condition comparisons.
- Wilson intervals for failure counts.
- State that trainings within a run are not independent.
- Reuse the existing statistics code where possible.

---

## 6. Collapse pipeline (`src/collapse.py`)

Keep the existing structure and these design decisions; they were good:
- fit → sample → refit, 10 generations;
- `p_synthetic` mixing with **fresh real data** for the real share;
- the **same model initialisation in every generation of a run** (documented reason: removes
  the initialisation lottery as a confound);
- a fixed training budget;
- per-seed data;
- save each generation's pool and checkpoint.

New pool construction:
- **Gen 0:** the real enumerated space (78,400 base / 156,800 extended).
- **Base, gen n:** every real context `A la B lb Q` appears once, with the query label
  sampled from model n−1. No duplicates by construction; say so in the README. This is the
  "reproduce their work, then collapse" condition.
- **Extended, gen n:** every real context `A la B lb Q` with `lq`, `S2`, `l_S2` sampled from
  model n−1 (S2 is fed back as exemplar 0 of the sampled class).
- Mixing: the synthetic share is drawn **without replacement** from the synthetic pool; the
  real share comes from the real space.

---

## 7. Experiments

### Tier 1: required (run first)
1. **Gen-0 replication** — base and extended, 3 seeds, 1,000,000 steps (repo
   `ih_paper_runs.sh`: `--train_iters 1000000 --train_bs 32 --lr 0.00001 --d_model 64`).
   - Expected: base loss plateaus near ln 2 ≈ 0.693, then drops sharply (the paper sees the
     phase change at roughly 2×10⁵ steps).
   - Induction strength of the layer-2 heads rises at the drop.
   - All four evaluation sets end near 1.0.
   - Also report multiple induction heads, with additive effects under ablation (paper
     Fig. 4).
2. **Hyperparameter tuning on `val`:**
   - lr ∈ {1e-5, 3e-5, 1e-4, 3e-4} × 3 seeds, for both variants;
   - selection by validation accuracy and time to the phase change.
   - If a larger lr gives the same final validation accuracy **and** the same circuit
     (induction strength, ablation profile), it may be used for the collapse runs to save
     compute. Document that choice.
   - Then fix the per-generation budget: the phase change plus a clear margin, justified
     from the gen-0 curves.
3. **Collapse:** 10 generations, **8 seeds** each (drop to 5 if compute forces it, and tell
   me first):
   - base, p = 1;
   - extended, p ∈ {1, 0.5, 0.25};
   - real-data control p = 0 for **both** base and extended (the pilot lacked a base control;
     fix that).

### Tier 2: new experiments that explain mechanism (run after Tier 1)
4. **Match vs copy under collapse.** Track `test_exemplars` and `test_relabel` accuracy, plus
   previous-token score vs induction strength, across generations. Question: which half of the
   circuit (match or copy, as the paper splits it) degrades first?
5. **Head redundancy under collapse.** The paper found 8 additive layer-2 heads. Across
   generations, track:
   - how many heads have induction strength above a threshold;
   - the accuracy when keeping only the best head.

   Question: does collapse concentrate the work in fewer heads (less redundancy, more
   fragility)?
6. **Sampling temperature:** extended, p = 1, T ∈ {0.5, 1.0, 2.0} plus greedy (argmax), 5 seeds.
   - Hypothesis: low temperature is mode-seeking and narrows the S2 distribution faster
     (tail loss); high temperature injects noise into labels.
   - Relate the distribution drift to the change in induction strength.
7. **Accumulate vs replace:** extended, p = 1, where generation n trains on the union of
   real data and all earlier synthetic pools (subsample to a fixed budget), 5 seeds.
   - Prior work (Gerstgrasser et al., 2024, "Is Model Collapse Inevitable?", arXiv
     2404.01413; **verify this citation before using it**) reports that accumulating data
     avoids collapse.
   - Test whether the induction circuit is preserved too.
8. **Weak-generator test.** The pilot's "a weak generator drives collapse" rested on one seed.
   Test it deliberately:
   - start chains from gen-0 checkpoints saved **before**, **during** and **after** the phase
     change (e.g. val accuracy ≈ 0.6, 0.8, 1.0);
   - 5 seeds each, extended p = 1.

   This turns an anecdote into a controlled result.
9. **Phase-change timing across generations.** From existing logs: does the phase change
   arrive earlier or later in later generations, and does that track the pool's label noise
   or symbol skew? This links directly to the paper's Section 5 on how data properties change
   the timing.
10. **Architecture sensitivity** (run only if compute remains): repeat extended p = 1 with an
    MLP added (`mlp_ratio = 4`), 3 seeds. Does the collapse effect depend on the paper's
    attention-only choice? This also connects back to the pilot, which had MLPs.

Do **not** start Tier 2 until GATE 3 is approved. Present Tier 2 to me as a list with a compute
estimate first, so I can choose.

---

## 8. Configs, tests, Slurm, README

- New YAML configs for every condition above, generated by a script (as
  `experiments/make_v2_configs.py` does now); name them `exact_*`.
- Update `tests/test_pipeline.py` to cover:
  - dataset size = 78,400 (base) / 156,800 (extended);
  - split disjointness (§2.4);
  - RoPE relative-position property;
  - causal mask;
  - an attention-only parameter count;
  - generation output shapes;
  - an end-to-end tiny run of 2 generations.
- Slurm scripts:
  - CPU, 1 thread per task, array jobs per seed and condition;
  - record job IDs in a `results/PROVENANCE.md` file.
  - Fix the shared-filesystem start-up race seen before (create dirs with `exist_ok`, and
    stagger the starts).
- `requirements.txt`: add `h5py`, pin versions.
- `README.txt`:
  - how to get the `.h5` file;
  - exact commands for every phase;
  - expected runtimes;
  - where outputs land;
  - a section listing every deviation from the paper and why (PyTorch instead of JAX; the
    exact label-pair split cannot be reproduced; any lr or budget change after tuning; the
    extended model's design).

---

## 9. Analysis outputs (no report writing)

- Extend `experiments/analyze_results.py` (keep the pilot analysis working) to write
  `results/analysis_summary_exact.json` with every number the report could need.
- Make figures, legible at two-column width, 300 dpi, consistent colours per condition,
  colour-blind safe:
  1. gen-0 training curves (loss on all four sets with the ln 2 line, plus induction strength
     per layer-2 head) — the reproduction of paper Fig. 3;
  2. accuracy per generation per condition, one line per seed;
  3. induction strength / previous-token score / match score per generation;
  4. match vs copy accuracy per generation;
  5. head-ablation heatmaps (heads × generations);
  6. S2 distribution drift: KL vs noise floor, P(S2 = Q), tail mass;
  7. ACDC circuit size and edges, gen 0 vs gen 9;
  8. embedding PCA;
  9. one figure per Tier-2 experiment that was run.
- Write `results/RESULTS_NOTES.md`:
  - each finding in one or two sentences, with the exact numbers, n, the test and the p-value;
  - the figure that supports it;
  - caveats;
  - **anything surprising or that contradicts a hypothesis, stated plainly.**
  - Separate "established" from "suggestive (few seeds)".

---

## 10. Gates

- **GATE 1** (after §1–§4 and tests): show me the passing tests, a parameter count, steps/sec
  on one core, and a 20k-step smoke run curve.
- **GATE 2** (gen-0 replication): show me the loss curves (is there an ln 2 plateau and a
  sudden drop?), final accuracy on all four sets, the induction strength per layer-2 head, and
  the ablation result.
  - If the phase change does not appear, **stop and diagnose** before any collapse run.
  - Check, in order:
    1. the label/query construction;
    2. the causal mask;
    3. RoPE;
    4. whether exemplar 0 is used;
    5. lr and steps.
- **GATE 3** (tuning done, budget chosen): show me the tuning table and the proposed
  per-generation budget with a total compute estimate.
- **GATE 4** (Tier 1 collapse done): show me RESULTS_NOTES for Tier 1, then the Tier-2 list
  with compute estimates.

---

## 11. Rules

- **Never fabricate or estimate a result.** Every number in the notes, tables and figures
  must be computed from logs by a script in the repo. If a run fails, say so.
- Do not copy code from `aadityasingh/icl-dynamics`. Read it for specifications only, and
  cite it as the reference for each setting you match.
- Keep code readable: docstrings saying which paper section or repo flag each setting comes
  from.
- When a design decision is not fixed by the paper or brief (for example what S2 is in real
  data, or the plateau threshold), choose sensibly, write the reason in a docstring, and list
  it in the README's deviations section. Do not ask me about small choices, but do flag big
  ones at the next gate.
- Do not touch `abstract/main.tex` except to stop it referencing files you move.
