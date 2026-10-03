"""Edge-level ACDC circuit discovery (src/acdc.py) for every generation of the
collapse runs. Run ON THE CLUSTER, next to the checkpoints, like
analyze_circuits.py:

    python experiments/analyze_acdc.py                          # every run under results/collapse
    python experiments/analyze_acdc.py --conds v2_extended_p1 v2_extended_p1_dups
    python experiments/analyze_acdc.py --group-id 0             # one Slurm array task (see ACDC_GROUPS)
    python experiments/analyze_acdc.py --list-groups

Writes results/analysis/acdc_<run>.csv, one row per (generation, corruption, tau), holding
the size of the discovered circuit, which heads and MLPs it uses, whether a
layer-0 head feeds a later head's key / query / value input (the previous-token ->
induction composition of Olsson et al. 2022 is a layer-0 -> layer-1 KEY edge),
and the accuracy of the circuit run on its own.

Two corruptions are patched in, each answering a different question:
  resample  every clean sequence is paired with a different random sequence (symbols and labels
            both change): the circuit that does the whole task, matching included;
  labels    the same symbols with freshly drawn labels: only edges that carry LABEL information
            survive, i.e. the route by which the answer reaches the output.
The discovery batch is the first `n_eval` sequences of the run's validation split; the random
sequences come from a separate seed (same task distribution).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import ACDC_GROUPS  # noqa: E402
from acdc import acdc, describe, relabel_corruption
from data import SymbolicICLDataset, collate_batch, TaskConfig
from interp import load_checkpoint

DEFAULT_TAUS = [0.01, 0.05]
CORRUPTIONS = ["resample", "labels"]
CORRUPT_SEED_OFFSET = 3_000_017      # disjoint from the train / val / test seeds of make_splits

def analyze_run(results: Path, run_name: str, taus: list[float], n_eval: int, gens: list[int] | None = None,
                corruptions: list[str] = CORRUPTIONS) -> list[dict]:
    run_dir = results / "collapse" / run_name
    cfg = json.load(open(run_dir / "run_config.json"))
    tcfg = TaskConfig(**cfg["task"])
    data_seed = cfg["train"]["data_seed"]
    ds = SymbolicICLDataset(tcfg, n_eval, seed=data_seed + 1_000_003)
    cds = SymbolicICLDataset(tcfg, n_eval, seed=data_seed + CORRUPT_SEED_OFFSET)
    b = collate_batch([ds[i] for i in range(n_eval)])
    c = collate_batch([cds[i] for i in range(n_eval)])
    corrupt = {"resample": (c["symbol_tokens"], c["label_tokens"]),
               "labels": (b["symbol_tokens"], relabel_corruption(b["symbol_tokens"], b["label_tokens"], tcfg.L, seed=data_seed + 5))}

    rows = []
    for g in (gens if gens is not None else range(cfg["collapse"]["n_generations"])):
        ckpt = run_dir / f"gen{g}_model.pt"
        if not ckpt.exists():
            print(f"  [{run_name}] gen{g}: no checkpoint, skipped")
            continue
        model, _, _, _ = load_checkpoint(ckpt)
        for corruption in corruptions:
            for tau in taus:
                circ = acdc(model, b["symbol_tokens"], b["label_tokens"], b["query_label"], *corrupt[corruption], tau)
                d = describe(circ, model.mcfg.n_layers, model.mcfg.n_heads)
                rows.append({"run": run_name, "generation": g, "acdc_corruption": corruption, **d})
                print(f"  [{run_name}] gen{g} {corruption} tau={tau}: full acc {circ.full_accuracy:.3f}, circuit acc {circ.accuracy:.3f}, "
                      f"{circ.n_live_edges}/{d['acdc_n_edges_total']} edges, K-composition={d['acdc_kcomp']}, MLP0={d['acdc_mlp0']} MLP1={d['acdc_mlp1']}")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/analysis")
    ap.add_argument("--conds", nargs="+", default=None)
    ap.add_argument("--group-id", type=int, default=None, help="index into ACDC_GROUPS (for a Slurm array)")
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--gens", nargs="+", type=int, default=None, help="generations to analyse (default: all)")
    ap.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS)
    ap.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=CORRUPTIONS)
    ap.add_argument("--n-eval", type=int, default=128)
    args = ap.parse_args()
    results = Path(args.results)
    out_dir = Path(args.out) if args.out else results / "analysis"

    if args.list_groups:
        for i, g in enumerate(ACDC_GROUPS):
            print(i, g)
        return
    conds = args.conds
    if args.group_id is not None:
        if not (0 <= args.group_id < len(ACDC_GROUPS)):
            raise SystemExit(f"group-id {args.group_id} out of range [0, {len(ACDC_GROUPS)})")
        conds = ACDC_GROUPS[args.group_id]

    runs = []
    for d in sorted((results / "collapse").glob("*_seed*")):
        cond, seed = d.name.rsplit("_seed", 1)
        if (conds is None or cond in conds) and (args.seeds is None or int(seed) in args.seeds):
            runs.append(d.name)
    print(f"ACDC on {len(runs)} run(s), taus {args.taus}, corruptions {args.corruptions}: {runs}")
    if not runs:
        print("nothing to do (no matching runs; a group whose runs have not been produced yet is skipped)")
        return
    for name in runs:
        rows = analyze_run(results, name, args.taus, args.n_eval, args.gens, args.corruptions)
        if rows:
            write_csv(rows, out_dir / f"acdc_{name}.csv")
    print("wrote CSVs to", out_dir)


if __name__ == "__main__":
    main()
