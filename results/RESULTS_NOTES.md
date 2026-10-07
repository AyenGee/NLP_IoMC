# Results notes: exact Singh et al. task, Tier 1 (written for the author; not report text)

Everything here is read from files under `results/exact/` (indexed in `results/PROVENANCE.md`). Numbers are medians or means
over the 8 seeds of a condition unless stated; "trainings" within one chain are not independent, so per-seed lines and exact
permutation tests over seeds are used instead of pooled tests. No citation or external number is used.

## 0. Setup in one paragraph

Task: 2-way 1-shot Omniglot (precomputed ResNet18 features), sequence `A la B lb Q` (base) or `A la B lb Q lq S2` (extended);
attention-only 2-layer transformer, d=64, 8 heads, RoPE; Adam, constant lr, batch 32, no early stopping. Collapse chain: fit
(generation 0, whole real space) -> sample the query label (and, extended, the symbol S2 and its label) for every context
-> refit a fresh model from the SAME initialisation on the pool -> repeat; 10 generations, 8 seeds, 6 conditions = 48 chains.
Learning rate 3e-4 and 5,000 steps (160,000 sequences) per generation were chosen at the tuning gate (`results/exact/tuning_table.md`).

## 1. Deviations from the plan and the paper that matter for reading the numbers

1. **Auxiliary loss weights 0.1 (extended model).** With the plan's equal weights the extended model reaches ~100% training
   accuracy without the paper's circuit (accuracy on held-out label pairs 0.40-0.54, induction strength ~0, at lr 1e-5/1e-4/3e-4;
   `tuning_table.md`, variant `extended_equal`). The 50-way S2 loss at weight 1 is what prevents it (weight sweep, `lw_*` runs,
   lr 1e-4 seed 0, 15k steps, l_S2 weight 1: S2 weight 0.5 -> held-out-pair accuracy 0.69; 0.3 -> 0.99; 0.1 -> 1.00). Weights (1, 0.1, 0.1) are used throughout.
2. **Budget unit.** The plan's "1,000,000 steps" is read as the reference code's 1,000,000 *sequences* (31,250 steps at batch 32).
3. **Learning rate 3e-4, not the paper's 1e-5**, with a 5,000-step budget per generation (phase change at 750-1,000 steps).
   Same circuit as at 1e-5 in all 3 seeds (table). Collapse dynamics at the paper's lr were not run.
4. **Held-out accuracy ceiling.** Accuracy on held-out exemplars (`val`, `test_exemplars`) is ~0.85-0.87 even at generation 0
   (matching across different drawings of a character is harder than copying), not "near 1.0".
5. The pilot report `abstract/main.tex` is based on the earlier symbolic task and is stale.

## 2. Generation 0 (replication, Phase 2/3)

- Base, lr 1e-5: phase change (train acc > 0.9) at 11,000-13,500 steps; final train 0.999, held-out pairs (copy) 0.978,
  test classes 0.935, val 0.832; best induction strength 0.82; zeroing layer 2 -> 0.298 accuracy; cutting layer 1's direct
  path leaves 0.995; max single-head drop 0.065. This reproduces the paper's qualitative picture (loss plateau at ln 2, sudden drop,
  induction heads in layer 2, additive head effects).
- Extended (weights 0.1): same phase-change window (11,500-12,500), copy 0.953, induction strength 0.89.

## 3. Tier 1 results

### 3.1 Task accuracy (query label): no functional collapse, small but real decline when fully synthetic

0 of 72 later-generation trainings fall below 0.9 train accuracy in any condition (95% interval 0-0.051 per condition; "failed"
threshold 0.6, also 0).

| condition | final train | control | p | final copy (held-out pairs) | control | p | final val | control | p |
|---|---|---|---|---|---|---|---|---|---|
| base, p=1 | 0.9891 | 1.0000 | 0.0002 | 0.9821 | 0.9993 | 0.0003 | 0.8484 | 0.8464 | 0.86 |
| extended, p=1 | 0.9850 | 0.9978 | 0.0025 | 0.9789 | 0.9961 | 0.023 | 0.8510 | 0.8531 | 0.87 |
| extended, p=0.5 | 0.9993 | 0.9978 | 0.46 | 0.9971 | 0.9961 | 0.76 | 0.8547 | 0.8531 | 0.90 |
| extended, p=0.25 | 0.9989 | 0.9978 | 0.65 | 0.9977 | 0.9961 | 0.70 | 0.8630 | 0.8531 | 0.41 |

(controls: `exact_base_p0` for base, `exact_extended_p0` for extended; exact permutation tests over the 8 seeds;
no multiplicity correction; these are the only contrasts, 12 numbers.) Per-seed worst final values: base p=1 train 0.962 (seed 6),
copy 0.942 (seed 5); extended p=1 train 0.960 (seed 3), copy 0.925 (seed 3). The decline is gradual (median train accuracy
1.000 -> 0.992 base, 1.000 -> 0.990 extended over 10 generations) and still going at generation 9, so ten generations is a lower bound
on the effect, not evidence of a plateau.

### 3.2 Why so little: the target is nearly deterministic

The query label is a deterministic function of the context, and every generation's model is ~99.9% correct on it (synthetic query labels
correct 0.9996 at generation 1; 0.980 at generation 9 for base p=1; extended p=1 0.9798 at generation 9), so sampling adds almost no label
noise to propagate. This is a property of the task, not a finding about robustness of ICL in general.

### 3.3 The random target does drift (extended, S2)

S2 (the symbol after the query label) is a genuinely random target, uniform over the two context classes. Pool statistics for the
generated S2 (median over seeds; noise floor of the KL = 0.00045, from resampling a real-distribution pool):

| generation | 0 | 1 | 5 | 9 |
|---|---|---|---|---|
| p=1: P(S2 = query class) (real 0.5) | 0.500 | 0.494 | 0.429 | 0.356 |
| p=1: S2 outside the context | 0 | 0.078 | 0.286 | 0.458 |
| p=1: mass on the 10 rarest classes (uniform 0.20) | 0.200 | 0.133 | 0.097 | 0.077 |
| p=1: KL to generation 0 | 0 | 0.029 | 0.083 | 0.139 |
| p=0.5: KL to generation 0 | 0 | 0.0073 | 0.0068 | 0.0061 |
| p=0.25: KL to generation 0 | 0 | 0.0017 | 0.0015 | 0.0015 |
| p=0: KL to generation 0 | 0 | 0.0002 | 0.0002 | 0.0002 |

Fully synthetic data therefore shows a monotone loss of tail classes and a growing share of S2 symbols that are not even in the
context. With p=0.5 or 0.25 the KL is far above the noise floor (stable, roughly 7-40x the control) but does not grow: a one-off bias that
the real data holds in place rather than an accumulating one. Chain-level S2 drift in the control is below the noise floor.

### 3.4 Circuits across generations

- Phase-change timing (steps to train acc > 0.9) does not change across generations in any condition (median base 875-1,000, extended
  750 at generations 0, 1, 5, 9; no failures to reach it). With the same initialisation and learning rate the circuit forms equally fast.
- Number of layer-2 induction heads (strength > 0.2): 7-8 (base), 6-8 (extended), no trend with generation or condition.
- Redundancy shrinks slowly with fully synthetic data: median max single-head ablation drop rises 0.005 -> 0.025 (base p=1; control 0.002) and
  0.008 -> 0.026 (extended p=1; control 0.009); accuracy with only the strongest head kept falls 0.996 -> 0.964 (base p=1) and
  0.973 -> 0.914 (extended p=1) while the controls stay at 0.97-0.99. Cutting layer 1's direct path stays at 0.988-1.0 in all conditions,
  so layer 1 still reaches the output only through layer 2 (additivity of head effects holds: summed single drops 0.045 vs joint drop 0.68 for extended p=1).
- ACDC (resample corruption, tau 0.01) edge count of the recovered circuit rises mildly with generation for synthetic conditions
  (base p=1 56.5 -> 63.5; extended p=1 43.5 -> 54.5; controls 56.5 -> 65 and 43.5 -> 46); the base control rises as much as base p=1, so this is not by
  itself evidence of a collapse effect.
- Embeddings: the cosine of a class's projected embedding to its generation-0 value falls to ~0.83-0.89 already at generation 1 in the real-data controls
  (a new data order each generation), so only the difference to the control is informative: at generation 9 0.78 (base p=1) vs 0.83 (control), 0.81 (extended p=1)
  vs 0.89 (control). Effective rank of the 50 train-class embeddings falls 23.7 -> 17.8 for extended p=1 (control 23.1; p=0.25 22.3) and 14.1 -> 13.2 for base p=1
  (control 14.1): fully synthetic training shrinks the number of directions used to represent the classes.

### 3.5 Match vs copy

First generation at which accuracy on cross-drawing matching (`test_class_ex0`) vs held-out label pairs (copy) drops by more than 0.05 from generation 0:
base p=1: copy first in 2 seeds, match first in 0, neither in 6; extended p=1: copy first in 1 seed, neither in 7; no drops in any other condition. The data cannot
say which fails first; they only say neither fails much in 10 generations. Copy is the more sensitive of the two (0.9993 -> 0.982 base p=1; match 0.993 -> 0.974).

## 4. What Tier 1 does and does not show

Shows: (i) under the exact paper task the induction circuit is robust over 10 generations of fully synthetic retraining from the same initialisation; (ii) the
degradation that does appear is small, significant for fully synthetic data (train/copy accuracy), and associated with a measurable distribution drift in the
model's own random outputs (S2) and a thinning of the learned class representation; (iii) 25% real data removes the accumulating part of the drift.

Does not show: that collapse cannot occur in ICL. The protocol (same init, deterministic target, T=1 sampling, finite enumerable context space, 10 generations, lr 3e-4)
is favourable to the circuit. The Tier 2 experiments (temperature, accumulate vs replace, weak generators) are where a stronger effect, if any, would appear.

## 5. Limitations

- 8 seeds per condition; p-values are exact but uncorrected; effect sizes small.
- lr 3e-4 (see 1.3). The extended model uses auxiliary weights 0.1 (see 1.1); with equal weights the model does not form the circuit, so the S2 results are for the weighted model.
- Embedding statistics are partly seed/data-order noise (see 3.4).
- The pool for generation >= 1 is the enumerated 78,400-context space with model-resampled targets, so the synthetic data cover the same contexts as the real data; only the targets change.
- Run locally (CPU, 1 thread per run, 4 in parallel); no cluster jobs for Tier 1.

## 6. Files

`results/exact/analysis_summary_exact.json` (copy: `results/analysis_summary_exact.json`), `results/exact/figures/fig_*.png`
(accuracy by generation, head scores, match vs copy, S2 drift, ACDC, ablation heatmaps, embeddings, tuning, generation-0 curves and ablations),
`results/exact/analysis/{circuits,acdc}_<run>.csv`, `results/exact/collapse/<run>/`.
