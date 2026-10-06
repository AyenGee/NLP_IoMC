"""Validation-based hyper-parameter study for the induction task (no collapse).

The brief asks for hyper-parameters to be optimised on a validation split and
only the final model evaluated on test. This script is that study. It is run
BEFORE the collapse experiments so the settings they use are justified, and
it is deliberately cheap: plain training on real data, one array task per cell.

Grid (per variant, base and extended):
  * learning rate {3e-4, 1e-3, 3e-3} x weight decay {0.01, 0.1, 1.0}   (9 cells)
  * then one factor at a time around the default (lr 1e-3, wd 0.1, d_model 64,
    4 heads, batch 128): d_model {32, 128} (d_ff = 4 x d_model), heads {2, 8},
    batch size {64, 256}                                              (6 cells)
Every cell is trained for the same fixed budget (100 epochs, early stopping
OFF, as in the corrected-protocol runs) with 4 seeds, each with its own
real-data pool. These seeds (100-103) are disjoint from every seed of the main
experiments, so no validation or test sequence of a later experiment was used
for tuning. The depth (2 layers, the minimum for an induction circuit) and the
pool size (20,000; chosen in pilot runs, see the README) are not tuned here.

Selection rule, fixed before looking at any result, applied per variant to
VALIDATION accuracy only (test accuracy is recorded but never used):
  1. mean final validation accuracy over the 4 seeds (a failed seed scores
     ~0.25, so this rewards reliability, the property that mattered most);
  2. among cells within 0.01 of the best, the smallest median number of epochs
     to reach 0.9 validation accuracy (faster = cheaper);
  3. a remaining tie goes to the default cell.

    python experiments/tune_hparams.py --list
    python experiments/tune_hparams.py --task-id $SLURM_ARRAY_TASK_ID     # one cell
    python experiments/tune_hparams.py --summarize                        # after all tasks finished

--summarize prints the winner per variant, whether the settings used by the
main experiments (the default cell) are within 0.01 of it, and writes
results/tuning/summary.json, which experiments/analyze_results.py tabulates.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# torch and the project modules are imported inside run_cell, so that the cell list can be
# read (e.g. by check_manifest.py on a login node without the virtualenv) without them.

ROOT = Path(__file__).resolve().parent.parent
TUNE_SEEDS = [100, 101, 102, 103]
BUDGET_EPOCHS = 100
TOLERANCE = 0.01          # "as good as the best": within this much validation accuracy
DEFAULT = {"lr": 1e-3, "weight_decay": 0.1, "d_model": 64, "n_heads": 4, "batch_size": 128}
VARIANT_CONFIG = {"base": "base_no_collapse.yaml", "extended": "extended_no_collapse.yaml"}


@dataclasses.dataclass(frozen=True)
class Cell:
    variant: str
    lr: float = DEFAULT["lr"]
    weight_decay: float = DEFAULT["weight_decay"]
    d_model: int = DEFAULT["d_model"]
    n_heads: int = DEFAULT["n_heads"]
    batch_size: int = DEFAULT["batch_size"]

    @property
    def name(self) -> str:
        return (f"{self.variant}_lr{self.lr:g}_wd{self.weight_decay:g}_d{self.d_model}"
                f"_h{self.n_heads}_b{self.batch_size}")

    @property
    def is_default(self) -> bool:
        return all(getattr(self, k) == v for k, v in DEFAULT.items())


def build_cells() -> list[Cell]:
    cells = []
    for variant in ("base", "extended"):
        for lr in (3e-4, 1e-3, 3e-3):
            for wd in (0.01, 0.1, 1.0):
                cells.append(Cell(variant, lr=lr, weight_decay=wd))
        for d_model in (32, 128):
            cells.append(Cell(variant, d_model=d_model))
        for n_heads in (2, 8):
            cells.append(Cell(variant, n_heads=n_heads))
        for batch in (64, 256):
            cells.append(Cell(variant, batch_size=batch))
    names = [c.name for c in cells]
    assert len(set(names)) == len(names)
    return cells


CELLS = build_cells()


def _first_epoch_above(rows: list[dict], thresh: float = 0.9):
    return next((int(r["epoch"]) for r in rows if r.get("val_acc") and float(r["val_acc"]) > thresh), None)


def run_cell(cell: Cell, out_dir: Path, seeds: list[int], epochs: int = BUDGET_EPOCHS,
             n_train: int | None = None, n_val: int | None = None, n_test: int | None = None) -> dict:
    from data import TaskConfig
    from model import ModelConfig
    from train import TrainConfig, train_model
    from utils import load_config

    cfg = load_config(ROOT / "configs" / VARIANT_CONFIG[cell.variant])
    tcfg = TaskConfig(**cfg["task"])
    mcfg = ModelConfig(**{**cfg["model"], "d_model": cell.d_model, "n_heads": cell.n_heads, "d_ff": 4 * cell.d_model})
    base = TrainConfig(**cfg["train"])
    results = []
    for seed in seeds:
        log_path = out_dir / "logs" / f"{cell.name}_seed{seed}.csv"
        tc = dataclasses.replace(
            base, lr=cell.lr, weight_decay=cell.weight_decay, batch_size=cell.batch_size, n_epochs=epochs,
            patience=None, seed=seed, data_seed=seed, interp_every=1000, log_path=str(log_path), ckpt_path=None,
            n_train=n_train or base.n_train, n_val=n_val or 2000, n_test=n_test or base.n_test,
        )
        res = train_model(tcfg, mcfg, tc, verbose=False)
        rows = list(csv.DictReader(open(log_path)))
        last = [r for r in rows if r.get("val_acc")][-1]
        results.append({
            "seed": seed,
            "final_val_acc": float(last["val_acc"]), "final_val_loss": float(last["val_loss"]),
            "max_val_acc": max(float(r["val_acc"]) for r in rows if r.get("val_acc")),
            "final_train_acc": float(rows[-1]["train_acc"]),
            "epoch_val_acc_gt_0.9": _first_epoch_above(rows),
            "test_acc": res["summary"]["test_acc"],        # recorded for the record; never used for selection
            "train_seconds": res["summary"]["train_seconds"],
        })
        print(f"[{cell.name}] seed {seed}: val_acc={results[-1]['final_val_acc']:.3f} "
              f"epoch>0.9={results[-1]['epoch_val_acc_gt_0.9']} ({results[-1]['train_seconds']:.0f}s)")
    out = {"cell": dataclasses.asdict(cell), "name": cell.name, "is_default": cell.is_default,
           "epochs": epochs, "seeds": results}
    from utils import ensure_dir
    ensure_dir(out_dir)
    with open(out_dir / f"{cell.name}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def summarize(out_dir: Path, cells: list[Cell] | None = None) -> dict:
    cells = cells or CELLS
    summary = {"tolerance": TOLERANCE, "seeds": TUNE_SEEDS, "budget_epochs": BUDGET_EPOCHS, "variants": {}}
    for variant in ("base", "extended"):
        table = []
        for cell in [c for c in cells if c.variant == variant]:
            p = out_dir / f"{cell.name}.json"
            if not p.exists():
                continue
            j = json.load(open(p))
            seeds = j["seeds"]
            t90 = [s["epoch_val_acc_gt_0.9"] for s in seeds if s["epoch_val_acc_gt_0.9"] is not None]
            table.append({
                "name": cell.name, "is_default": cell.is_default,
                **{k: getattr(cell, k) for k in DEFAULT},
                "n_seeds": len(seeds),
                "mean_val_acc": sum(s["final_val_acc"] for s in seeds) / len(seeds),
                "min_val_acc": min(s["final_val_acc"] for s in seeds),
                "n_seeds_val_ge_0.9": sum(s["final_val_acc"] >= 0.9 for s in seeds),
                "median_epoch_to_val0.9": _median(t90),
                "mean_val_loss": sum(s["final_val_loss"] for s in seeds) / len(seeds),
                "mean_seconds": sum(s["train_seconds"] for s in seeds) / len(seeds),
            })
        if not table:
            continue
        best = max(r["mean_val_acc"] for r in table)
        contenders = [r for r in table if r["mean_val_acc"] >= best - TOLERANCE]
        inf = float("inf")
        contenders.sort(key=lambda r: (r["median_epoch_to_val0.9"] if r["median_epoch_to_val0.9"] is not None else inf,
                                       not r["is_default"]))
        winner = contenders[0]
        default = next((r for r in table if r["is_default"]), None)
        summary["variants"][variant] = {
            "cells": table, "best_mean_val_acc": best, "winner": winner["name"],
            "default": default["name"] if default else None,
            "default_within_tolerance": bool(default and default["mean_val_acc"] >= best - TOLERANCE),
            "n_cells": len(table), "n_cells_expected": sum(c.variant == variant for c in cells),
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary: dict) -> None:
    for variant, v in summary["variants"].items():
        print(f"\n== {variant}: {v['n_cells']}/{v['n_cells_expected']} cells finished ==")
        print(f"{'cell':44s} {'mean val':>8s} {'min val':>8s} {'ok':>5s} {'ep>0.9':>7s}")
        for r in sorted(v["cells"], key=lambda r: -r["mean_val_acc"]):
            mark = "  <- default" if r["is_default"] else ""
            e = r["median_epoch_to_val0.9"]
            print(f"{r['name']:44s} {r['mean_val_acc']:8.3f} {r['min_val_acc']:8.3f} "
                  f"{r['n_seeds_val_ge_0.9']:>3d}/{r['n_seeds']:<1d} {('-' if e is None else f'{e:g}'):>7s}{mark}")
        print(f"best mean validation accuracy {v['best_mean_val_acc']:.3f}; selected: {v['winner']}")
        if v["n_cells"] < v["n_cells_expected"]:
            print("  WARNING: some cells are missing, so this selection is incomplete.")
        if v["default_within_tolerance"]:
            print(f"  -> the settings used by the main experiments ({v['default']}) are within "
                  f"{summary['tolerance']} of the best: KEEP them.")
        else:
            print(f"  -> the settings used by the main experiments ({v['default']}) are MORE than "
                  f"{summary['tolerance']} below the best. Consider regenerating configs/v2_*.yaml with {v['winner']}'s values "
                  "before running the corrected-protocol experiment (see README, Section 6i).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--results", default="results", help="results directory; outputs go to <results>/tuning")
    ap.add_argument("--seeds", type=int, nargs="+", default=TUNE_SEEDS)
    ap.add_argument("--epochs", type=int, default=BUDGET_EPOCHS)
    ap.add_argument("--n-train", type=int, default=None, help="testing only")
    ap.add_argument("--n-val", type=int, default=None, help="testing only")
    ap.add_argument("--n-test", type=int, default=None, help="testing only")
    args = ap.parse_args()
    out_dir = Path(args.results) / "tuning"

    if args.list:
        for i, c in enumerate(CELLS):
            print(i, c.name, "(default)" if c.is_default else "")
        print(f"{len(CELLS)} cells x {len(TUNE_SEEDS)} seeds")
        return
    if args.summarize:
        print_summary(summarize(out_dir))
        return
    if args.task_id is None:
        ap.error("give --list, --summarize, or --task-id N")
    if not (0 <= args.task_id < len(CELLS)):
        raise SystemExit(f"task-id {args.task_id} out of range [0, {len(CELLS)})")
    cell = CELLS[args.task_id]
    print(f"=== tuning cell {args.task_id}/{len(CELLS) - 1}: {cell.name} ===")
    run_cell(cell, out_dir, args.seeds, epochs=args.epochs, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)


if __name__ == "__main__":
    main()
