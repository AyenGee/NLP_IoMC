"""Per-generation CIRCUIT and DATA-STRUCTURE analysis of the collapse runs.

Run this ON THE CLUSTER, where the per-generation checkpoints
(gen<g>_model.pt) and exact training-pool snapshots (gen<g>_data.pt) already
are, so only small CSVs have to be copied back instead of ~0.5 GB of tensors.

    python experiments/analyze_circuits.py                       # every run under results/collapse
    python experiments/analyze_circuits.py --conds extended_collapse ablation_p05
    python experiments/analyze_circuits.py --conds extended_collapse --seeds 0 1 2

Writes results/analysis/circuits_<run>.csv, one row per generation, with:

  * the BEST SINGLE HEAD's induction and previous-token scores and which head
    it is. (The all-head mean logged during training is diluted by heads that
    play no part in the circuit: a 100%-accurate model scored 0.24-0.45 on it.)
  * zero-ablation of each attention head and of each whole layer, as the drop
    in accuracy on a fixed real evaluation set -- a simplified, node-level
    stand-in for ACDC (Conmy et al., 2023), see src/circuits.py. This shows
    whether the circuit is still doing the work, not just whether it looks right.
  * STRUCTURE statistics of that generation's training pool (label consistency
    for repeated symbols, repeat counts, label injectivity, query validity),
    which marginal symbol KL cannot see. Real data scores exactly 1.0 on all
    of them, so any shortfall is corruption introduced by the generating model.
    This tests the hypothesis that collapse is driven by structural errors in
    the synthetic sequences rather than by drift in symbol frequencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from circuits import head_ablation_sweep, layer_ablation_summary
from data import SymbolicICLDataset, collate_batch, load_items, TaskConfig
from interp import load_checkpoint
from metrics import attention_entropy, induction_score, prev_token_score, structure_stats


def _heads(scores: dict) -> dict:
    return {k: v for k, v in scores.items() if "_head" in k}


def analyze_run(results: Path, run_name: str, n_eval: int = 512) -> list[dict]:
    run_dir = results / "collapse" / run_name
    cfg = json.load(open(run_dir / "run_config.json"))
    tcfg = TaskConfig(**cfg["task"])
    # Same sequences as the first n_eval of the run's validation split (make_splits: val seed = data_seed + 1_000_003).
    ds = SymbolicICLDataset(tcfg, n_eval, seed=cfg["train"]["data_seed"] + 1_000_003)
    batch = collate_batch([ds[i] for i in range(len(ds))])
    sym, lab, query = batch["symbol_tokens"], batch["label_tokens"], batch["query_label"]

    rows = []
    n_gens = cfg["collapse"]["n_generations"]
    for g in range(n_gens):
        ckpt, data = run_dir / f"gen{g}_model.pt", run_dir / f"gen{g}_data.pt"
        if not ckpt.exists():
            print(f"  [{run_name}] gen{g}: no checkpoint, skipped")
            continue
        model, _, _, _ = load_checkpoint(ckpt)
        with torch.no_grad():
            out = model(sym, lab, return_attention=True)
        acc = (out.label_logits[:, -1, :].argmax(-1) == query).float().mean().item()
        ind, prev, ent = induction_score(out.attn_maps, sym), prev_token_score(out.attn_maps), attention_entropy(out.attn_maps)
        ind_h, prev_h = _heads(ind), _heads(prev)
        best_ind = max(ind_h, key=ind_h.get)
        best_prev = max(prev_h, key=prev_h.get)

        single = head_ablation_sweep(model, sym, lab, query)
        worst = max(single, key=lambda r: r.delta_acc)
        layers = layer_ablation_summary(model, sym, lab, query)

        row = {
            "run": run_name, "generation": g, "eval_acc": acc,
            "induction_mean": ind["overall_mean"], "induction_baseline": ind["baseline_mean"],
            "induction_max": ind_h[best_ind], "induction_best_head": best_ind,
            "prev_token_mean": prev["overall_mean"], "prev_token_max": prev_h[best_prev], "prev_token_best_head": best_prev,
            "attn_entropy_mean": ent["overall_mean"],
            "ablate_max_single_delta_acc": worst.delta_acc, "ablate_max_single_head": f"layer{worst.layer}_head{worst.head}",
        }
        row.update({k: v for k, v in layers.items() if k.endswith("delta_acc")})
        if data.exists():
            row.update({f"pool_{k}": v for k, v in structure_stats(load_items(data), tcfg.n_classes, tcfg.burstiness).items()})
        rows.append(row)
        print(f"  [{run_name}] gen{g}: acc={acc:.3f} best induction head {best_ind}={ind_h[best_ind]:.2f} "
              f"(mean {ind['overall_mean']:.2f}) | biggest single-head ablation drop {worst.delta_acc:.2f} ({row['ablate_max_single_head']}) "
              f"| pool fully-valid={row.get('pool_frac_fully_valid', float('nan')):.3f}")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/analysis")
    ap.add_argument("--conds", nargs="+", default=None, help="condition names, e.g. extended_collapse (default: all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=512)
    args = ap.parse_args()
    results = Path(args.results)
    out_dir = Path(args.out) if args.out else results / "analysis"

    runs = []
    for d in sorted((results / "collapse").glob("*_seed*")):
        cond, seed = d.name.rsplit("_seed", 1)
        if (args.conds is None or cond in args.conds) and (args.seeds is None or int(seed) in args.seeds):
            runs.append(d.name)
    print(f"analysing {len(runs)} run(s): {runs}")
    for name in runs:
        rows = analyze_run(results, name, args.n_eval)
        if rows:
            write_csv(rows, out_dir / f"circuits_{name}.csv")
    print("wrote CSVs to", out_dir)


if __name__ == "__main__":
    main()
