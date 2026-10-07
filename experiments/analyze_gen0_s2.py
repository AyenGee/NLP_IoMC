"""How good is the generation-0 extended model's S2 head on real data? (Gate 4 follow-up to the "S2 outside context = 0.078" number.)

    python experiments/analyze_gen0_s2.py            # reads results/exact, writes results/exact/gen0_s2_quality.json and .md

For every generation-0 extended model (the 8 Tier 1 chains' gen0_model.pt, lr 3e-4, and the 3 weighted lr 1e-5 replication runs
gen0w_extended_s*), on REAL sequences of the training classes (the S2 head has 50 outputs, one per training class, so the test-class
sets cannot be scored): train_eval (the training exemplar), val (held-out exemplars) and test_relabel (held-out label pairs):
  mass_outside_context   mean softmax mass the S2 head puts on classes other than A and B (real data: 0)
  mass_query_class       mean softmax mass on Q's class (real data: 0.5)
  mass_other_context     mean softmax mass on the context class that is not Q's (real: 0.5)
  acc_s2                 argmax S2 accuracy against the real S2 class (the Bayes-optimal limit is 0.5: S2 is a fair coin between A and B)
  p_true_s2              mean probability of the real S2 class (Bayes-optimal: 0.5)
  acc_ls2_teacher        l_S2 accuracy when the real S2 exemplar is fed (Bayes-optimal: 1)
  sampled_*              the same quantities when S2 is SAMPLED (T = 1) and l_S2 is then sampled given it (the generation-time behaviour):
                         rate of S2 outside the context, of S2 = Q's class, and the accuracy of l_S2 when S2 is inside the context.
These numbers answer how much of the generation-1 pool's S2 statistics (pool_p_s2_outside_context at generation 1 in generations.csv)
is already a property of the generation-0 model: the generation-1 pool IS the generation-0 model's samples over all 78,400 training
contexts, so the first row below (train_eval) should reproduce it, and the val row says whether it is smaller or larger on held-out exemplars.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data as D  # noqa: E402
from train import EvalSets, load_model  # noqa: E402

SETS = ["train_eval", "val", "test_relabel"]


@torch.no_grad()
def quality(model, evals: EvalSets, feats: torch.Tensor, name: str, seed: int = 0) -> dict:
    rows, b = evals.rows[name], evals.batches[name]
    model.eval()
    out = model(b.ex, b.labels)
    p = torch.softmax(out["symbol_logits"], -1)
    cA, cB, cQ, c2 = (rows[:, D.C_A], rows[:, D.C_B], D.query_class(rows), D.s2_class(rows))
    idx = torch.arange(len(rows))
    in_ctx_mass = p[idx, cA] + p[idx, cB]
    res = {"mass_outside_context": float(1 - in_ctx_mass.mean()), "mass_query_class": float(p[idx, cQ].mean()),
           "mass_other_context": float((in_ctx_mass - p[idx, cQ]).mean()), "acc_s2": float((p.argmax(-1) == c2).float().mean()),
           "p_true_s2": float(p[idx, c2].mean()),
           "acc_ls2_teacher": float((out["label_logits_s2"].argmax(-1) == b.s2_label).float().mean())}
    gen = torch.Generator().manual_seed(seed)
    ctx = D.collate(rows, feats, False)
    lq, s2, ls2 = model.sample_extended(ctx.ex[:, :3], ctx.labels[:, :2], feats[D.TRAIN_CLASSES, 0], 1.0, False, gen)
    in_ctx = (s2 == cA) | (s2 == cB)
    true_ls2 = torch.where(s2 == cA, rows[:, D.L_A], rows[:, D.L_B])
    res.update({"sampled_outside_context": float((~in_ctx).float().mean()), "sampled_s2_equals_q_class": float((s2 == cQ).float().mean()),
                "sampled_ls2_correct_given_in_context": float((ls2 == true_ls2)[in_ctx].float().mean()) if in_ctx.any() else float("nan")})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "exact"))
    args = ap.parse_args()
    results = Path(args.results)
    feats = D.OmniglotFeatures().tensor
    evals = EvalSets(feats, True)
    groups = {"tier1_lr3e-4 (gen0 of exact_extended_p1_s*)": sorted((results / "collapse").glob("exact_extended_p1_s[0-9]*/gen0_model.pt")),
              "lr1e-5 replication (gen0w_extended_s*)": sorted((results / "checkpoints").glob("gen0w_extended_s[0-9].pt"))}
    summary = {}
    for label, paths in groups.items():
        per_seed = {}
        for i, p in enumerate(paths):
            model, _ = load_model(p)
            per_seed[p.parent.name if p.parent.name != "checkpoints" else p.stem] = {s: quality(model, evals, feats, s, seed=i) for s in SETS}
        keys = next(iter(next(iter(per_seed.values())).values())).keys()
        med = {s: {k: float(np.nanmedian([v[s][k] for v in per_seed.values()])) for k in keys} for s in SETS}
        rng = {s: {k: [float(np.nanmin([v[s][k] for v in per_seed.values()])), float(np.nanmax([v[s][k] for v in per_seed.values()]))] for k in keys} for s in SETS}
        summary[label] = {"n_models": len(per_seed), "median": med, "range": rng, "per_seed": per_seed}
    # the generation-1..9 pool statistics of the fully synthetic Tier 1 chains, for the same quantity
    pools = []
    for d in sorted((results / "collapse").glob("exact_extended_p1_s[0-9]*")):
        rows = list(csv.DictReader(open(d / "generations.csv")))
        pools.append([float(r["pool_p_s2_outside_context"]) for r in rows])
    if pools:
        n = min(map(len, pools))
        summary["tier1_p1_pool_outside_context_median_by_generation"] = np.median(np.array([p[:n] for p in pools]), axis=0).round(4).tolist()
    (results / "gen0_s2_quality.json").write_text(json.dumps(summary, indent=2))
    lines = ["| models | set | mass outside context | mass on Q's class | S2 acc (Bayes 0.5) | p(true S2) (Bayes 0.5) | l_S2 acc, teacher-forced | sampled: S2 outside context | sampled: S2 = Q's class | sampled: l_S2 acc given S2 in context |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label, v in summary.items():
        if not isinstance(v, dict) or "median" not in v:
            continue
        for s in SETS:
            m, r = v["median"][s], v["range"][s]
            f = lambda k: f"{m[k]:.3f} [{r[k][0]:.3f}, {r[k][1]:.3f}]"
            lines.append(f"| {label}, n={v['n_models']} | {s} | {f('mass_outside_context')} | {f('mass_query_class')} | {f('acc_s2')} | {f('p_true_s2')} | {f('acc_ls2_teacher')} | {f('sampled_outside_context')} | {f('sampled_s2_equals_q_class')} | {f('sampled_ls2_correct_given_in_context')} |")
    if pools:
        lines += ["", "Outside-context rate of the fully synthetic Tier 1 pools (median over chains) by generation (generation 1 = sampled by the generation-0 model): "
                  + ", ".join(f"{x:.3f}" for x in summary["tier1_p1_pool_outside_context_median_by_generation"])]
    (results / "gen0_s2_quality.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
