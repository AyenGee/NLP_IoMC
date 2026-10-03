"""Retrain a FAILED collapse generation with early stopping disabled, to test
whether the failure was just a delayed induction transition cut off by early
stopping.

Failed generations (test accuracy < 0.5) stopped after only 31-43 epochs,
because validation loss never beat its first epochs and patience (30/40)
ran out. Healthy runs make their transition at epoch 10-28, but a failed run
might simply need longer. This script retrains the exact saved training pool
of a failed generation (results/collapse/<run>/gen<g>_data.pt) with the same
configuration, same initialisation, and patience disabled, then reports
whether validation accuracy ever rises.

For each case it trains two budgets:
  --epochs 0    the run's ORIGINAL epoch budget and learning-rate schedule, so the
                first epochs reproduce the original run and it simply keeps going
                past the point where early stopping had cut it off;
  --epochs 300  a much longer budget (note: this also stretches the cosine LR
                schedule, so it is not an exact continuation of the original).

Run where the data snapshots live (the cluster):
    python experiments/retrain_generation.py --list
    python experiments/retrain_generation.py --task-id $SLURM_ARRAY_TASK_ID        # i-th failing case
    python experiments/retrain_generation.py --cond extended_collapse --seed 1 --gen 1

Corrected-protocol (v2) runs have no early stopping, so budget 0 would only repeat
the original run; retrain them at a longer budget instead:
    python experiments/retrain_generation.py --prefix v2_ --epochs 450 --task-id N

A "case" is the FIRST failing generation of each (condition, seed) run that has
any failure, the most informative one (a later failure may inherit garbage
data from an earlier failed generator). Writes results/retrain/*.csv (per-epoch
logs) and *.json (summary). Safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import TaskConfig, load_items
from model import ModelConfig
from train import TrainConfig, train_model

FAIL_THRESHOLD = 0.5


def failing_cases(results: Path, prefix: str | None = None, exclude_prefix: str | None = None) -> list[tuple[str, int, int]]:
    """First failing generation (>= 1) of every collapse run that has one.
    `prefix` keeps only conditions whose name starts with it; `exclude_prefix`
    drops those that do (so the original-protocol cases keep a stable list)."""
    cases = []
    for d in sorted((results / "collapse").glob("*_seed*")):
        gen_csv = d / "generations.csv"
        if not gen_csv.exists():
            continue
        cond, seed = d.name.rsplit("_seed", 1)
        if prefix and not cond.startswith(prefix):
            continue
        if exclude_prefix and cond.startswith(exclude_prefix):
            continue
        for row in csv.DictReader(open(gen_csv)):
            if int(row["generation"]) >= 1 and float(row["test_acc"]) < FAIL_THRESHOLD:
                cases.append((cond, int(seed), int(row["generation"])))
                break
    return cases


def _val_curve(log_path: Path) -> list[tuple[int, float, float]]:
    return [(int(r["epoch"]), float(r["val_acc"]), float(r["train_acc"]))
            for r in csv.DictReader(open(log_path)) if r.get("val_acc", "") != ""]


def run_case(results: Path, cond: str, seed: int, gen: int, epochs_list: list[int], out_dir: Path) -> dict:
    run_dir = results / "collapse" / f"{cond}_seed{seed}"
    cfg = json.load(open(run_dir / "run_config.json"))
    tcfg, mcfg, base = TaskConfig(**cfg["task"]), ModelConfig(**cfg["model"]), TrainConfig(**cfg["train"])
    items = load_items(run_dir / f"gen{gen}_data.pt")
    orig = next(r for r in csv.DictReader(open(run_dir / "generations.csv")) if int(r["generation"]) == gen)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "cond": cond, "seed": seed, "gen": gen,
        "original": {"test_acc": float(orig["test_acc"]), "epochs_run": int(float(orig["n_epochs_run"])),
                     "n_epochs_budget": base.n_epochs, "patience": base.patience},
        "retrained": {},
    }
    for e in epochs_list:
        n_epochs = base.n_epochs if e == 0 else e
        tag = f"ep{n_epochs}"
        log_path = out_dir / f"{cond}_seed{seed}_gen{gen}_{tag}.csv"
        tc = dataclasses.replace(base, patience=None, n_epochs=n_epochs, log_path=str(log_path), ckpt_path=None)
        res = train_model(tcfg, mcfg, tc, train_items=items, verbose=False)
        curve = _val_curve(log_path)
        best_epoch, best_val, _ = max(curve, key=lambda t: t[1])
        orig_stop = summary["original"]["epochs_run"]
        first90 = next((ep for ep, v, _ in curve if v > 0.9), None)
        summary["retrained"][tag] = {
            "n_epochs": n_epochs,
            "test_acc": res["summary"]["test_acc"], "test_loss": res["summary"]["test_loss"],
            "max_val_acc": best_val, "epoch_of_max_val_acc": best_epoch,
            "first_epoch_val_acc_gt_0.9": first90,
            "final_train_acc": curve[-1][2], "final_val_acc": curve[-1][1],
            "recovered_after_original_stop": bool(first90 is not None and first90 >= orig_stop),
            "ever_learned_induction": bool(first90 is not None),
        }
        print(f"[{cond} seed{seed} gen{gen}] {tag}: test_acc={res['summary']['test_acc']:.3f} "
              f"max_val_acc={best_val:.3f}@ep{best_epoch} first_epoch_val>0.9={first90} (original run stopped at epoch {orig_stop})")
    with open(out_dir / f"summary_{cond}_seed{seed}_gen{gen}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/retrain")
    ap.add_argument("--list", action="store_true", help="print the failing cases and exit")
    ap.add_argument("--task-id", type=int, default=None, help="index into the list of failing cases (for Slurm arrays)")
    ap.add_argument("--cond"); ap.add_argument("--seed", type=int); ap.add_argument("--gen", type=int)
    ap.add_argument("--epochs", type=int, nargs="+", default=[0, 300],
                    help="epoch budgets to train; 0 = the run's original budget (default: 0 300)")
    ap.add_argument("--prefix", default=None, help="only conditions whose name starts with this (e.g. v2_)")
    ap.add_argument("--exclude-prefix", default=None, help="skip conditions whose name starts with this")
    args = ap.parse_args()
    results = Path(args.results)
    out_dir = Path(args.out) if args.out else results / "retrain"

    cases = failing_cases(results, prefix=args.prefix, exclude_prefix=args.exclude_prefix)
    if args.list:
        for i, c in enumerate(cases):
            print(i, *c)
        print(f"{len(cases)} failing case(s)")
        return
    if args.task_id is not None:
        if args.task_id >= len(cases):
            print(f"task-id {args.task_id} >= {len(cases)} failing cases; nothing to do")
            return
        cond, seed, gen = cases[args.task_id]
    elif args.cond is not None and args.seed is not None and args.gen is not None:
        cond, seed, gen = args.cond, args.seed, args.gen
    else:
        ap.error("give --list, --task-id N, or all of --cond --seed --gen")
    run_case(results, cond, seed, gen, args.epochs, out_dir)


if __name__ == "__main__":
    main()
