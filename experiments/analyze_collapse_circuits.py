"""Per-generation circuit analysis of the collapse runs (plan 5.2). Run where the checkpoints are (the cluster):

    python experiments/analyze_collapse_circuits.py                       # every run under results/exact/collapse
    python experiments/analyze_collapse_circuits.py --runs exact_extended_p1_s0 exact_extended_p1_s1
    python experiments/analyze_collapse_circuits.py --group-id 2          # one Slurm array task (one condition)
    python experiments/analyze_collapse_circuits.py --list-groups

For every generation g of a run it writes
  results/exact/analysis/circuits_<run>.csv   one row per generation:
      head ablations on the real ``train_eval`` sequences (src/circuits.py): the accuracy drop from zeroing each layer-2 head,
      the accuracy with only that head kept, zeroing each layer-1 head, whole-layer interventions, cutting layer 1's direct
      path; the number of induction heads (strength above src/circuits.INDUCTION_HEAD_THRESHOLD), the additivity of their
      effects and the accuracy with only the strongest head left; and embedding geometry (below);
  results/exact/analysis/acdc_<run>.csv       one row per (generation, corruption, tau): the ACDC circuit (src/acdc.py).

Embedding geometry (input embedding W x + b of the exemplar-0 features; label embeddings):
  emb_cos_train / emb_cos_test   mean cosine similarity, over the 50 training / 100 test classes, between a class's projected
                                 embedding at generation g and at generation 0 (same initialisation, so the spaces are comparable);
  emb_effective_rank             participation ratio (sum s)^2 / sum s^2 of the singular values of the centred train-class
                                 embeddings: how many directions the 50 classes actually span;
  label_cos_gen0                 mean cosine between each label embedding at generation g and at generation 0;
  label_mean_pair_cos            mean pairwise cosine among the 5 label embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import acdc as AC  # noqa: E402
import circuits as C  # noqa: E402
import data as D  # noqa: E402
from train import EvalSets, load_model  # noqa: E402
from utils import ensure_dir  # noqa: E402

TAUS = [0.003, 0.01, 0.03]
CORRUPTIONS = ["resample", "labels"]
N_ACDC = 128
GROUPS = [[f"exact_{c}"] for c in ["base_p1", "base_p0", "extended_p1", "extended_p05", "extended_p025", "extended_p0"]]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def embed_stats(model, ref_model, feats: torch.Tensor) -> dict:
    def proj(m, classes):
        with torch.no_grad():
            return m.example_embed(feats[classes, 0])
    tr, te = D.TRAIN_CLASSES, D.TEST_CLASSES
    e = proj(model, tr)
    s = torch.linalg.svdvals(e - e.mean(0, keepdim=True))
    lab = model.label_embed.weight.detach()
    lab_n = torch.nn.functional.normalize(lab, dim=-1)
    pair = (lab_n @ lab_n.T)[torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)].mean().item()
    return {"emb_cos_train": _cos(e, proj(ref_model, tr)), "emb_cos_test": _cos(proj(model, te), proj(ref_model, te)),
            "emb_effective_rank": float(s.sum() ** 2 / (s ** 2).sum()), "label_cos_gen0": _cos(lab, ref_model.label_embed.weight.detach()),
            "label_mean_pair_cos": pair}


def analyze_run(results: Path, run: str, evals: EvalSets, feats: torch.Tensor, taus=TAUS, n_acdc: int = N_ACDC) -> tuple[list, list]:
    run_dir = results / "collapse" / run
    cfg = json.load(open(run_dir / "run_config.json"))["collapse"]
    b = evals.batches["train_eval"]
    ex, lab, q = b.ex[:n_acdc, :3], b.labels[:n_acdc, :2], b.query_label[:n_acdc]
    half = b.ex.shape[0] // 2                      # the corrupted sequences come from the other half of the set (500 of 1000)
    cex, clab = b.ex[half:half + n_acdc, :3], b.labels[half:half + n_acdc, :2]
    corrupt = {"resample": (cex, clab), "labels": (ex, AC.relabel_corruption(lab, 5, seed=cfg["seed"] + 5))}
    rows, acdc_rows = [], []
    ref_model = None
    for g in range(cfg["n_generations"]):
        ck = run_dir / f"gen{g}_model.pt"
        if not ck.exists():
            print(f"  [{run}] gen{g}: no checkpoint, skipped", flush=True)
            continue
        model, _ = load_model(ck)
        ref_model = ref_model or model
        gen_log = next(r for r in csv.DictReader(open(run_dir / "generations.csv")) if int(r["generation"]) == g)
        strengths = {h: float(gen_log[f"ind_L2H{h}"]) for h in range(model.cfg.n_heads)}
        prof = C.ablation_profile(model, evals, sets=("train_eval",))
        base = prof["train_eval/baseline"]
        H = model.cfg.n_heads
        row = {"run": run, "generation": g, "ablation_baseline_acc": base,
               "zero_all_L1_acc": prof["train_eval/zero_all_L1"], "zero_all_L2_acc": prof["train_eval/zero_all_L2"],
               "cut_L1_direct_acc": prof["train_eval/cut_L1_direct"],
               "max_single_head_drop": max(base - prof[f"train_eval/zero_L2H{h}"] for h in range(H)),
               "best_keep_only_acc": max(prof[f"train_eval/keep_only_L2H{h}"] for h in range(H)),
               "max_zero_L1_drop": max(base - prof[f"train_eval/zero_L1H{h}"] for h in range(H))}
        for h in range(H):
            row[f"zero_L2H{h}_drop"] = base - prof[f"train_eval/zero_L2H{h}"]
            row[f"keep_only_L2H{h}_acc"] = prof[f"train_eval/keep_only_L2H{h}"]
            row[f"zero_L1H{h}_drop"] = base - prof[f"train_eval/zero_L1H{h}"]
        add = C.additivity(model, evals, strengths)
        row.update({"n_induction_heads": add["n_induction_heads"], "sum_single_drops": add["sum_single_drops"], "joint_drop": add["joint_drop"],
                    "acc_keep_only_strongest": add["acc_keep_only_strongest"]})
        row.update(embed_stats(model, ref_model, feats))
        rows.append(row)
        for corruption in CORRUPTIONS:
            for tau in taus:
                circ = AC.acdc(model, ex, lab, q, *corrupt[corruption], tau)
                acdc_rows.append({"run": run, "generation": g, "corruption": corruption, **AC.describe(circ, model.cfg.n_layers)})
        print(f"  [{run}] gen{g}: acc {base:.3f}, {add['n_induction_heads']} induction heads, strongest-only {add['acc_keep_only_strongest']:.3f}, "
              f"ACDC(resample, tau 0.01) edges {next(r['n_edges'] for r in acdc_rows[::-1] if r['corruption'] == 'resample' and r['tau'] == 0.01)}", flush=True)
    return rows, acdc_rows


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/exact")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--group-id", type=int)
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--n-acdc", type=int, default=N_ACDC, help="sequences in the ACDC discovery batch")
    args = ap.parse_args()
    results = Path(args.results)
    if args.list_groups:
        for i, g in enumerate(GROUPS):
            print(i, g)
        return
    runs = sorted(p.name for p in (results / "collapse").glob("*_s[0-9]*") if (p / "generations.csv").exists())
    if args.runs:
        runs = [r for r in runs if r in args.runs]
    if args.group_id is not None:
        if not (0 <= args.group_id < len(GROUPS)):
            raise SystemExit(f"group-id {args.group_id} out of range [0, {len(GROUPS)})")
        runs = [r for r in runs if r.rsplit("_s", 1)[0] in GROUPS[args.group_id]]
    print(f"analysing {len(runs)} run(s): {runs}", flush=True)
    if not runs:
        print("nothing to do")
        return
    feats = D.OmniglotFeatures().tensor
    ev_cache = {}
    for run in runs:
        ext = "extended" in run
        evals = ev_cache.setdefault(ext, EvalSets(feats, ext))
        rows, acdc_rows = analyze_run(results, run, evals, feats, args.taus, args.n_acdc)
        if rows:
            write_csv(rows, results / "analysis" / f"circuits_{run}.csv")
            write_csv(acdc_rows, results / "analysis" / f"acdc_{run}.csv")
    print("done")


if __name__ == "__main__":
    main()
