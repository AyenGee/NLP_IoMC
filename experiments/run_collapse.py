"""Run one collapse chain (fit -> sample -> refit) on the exact task.

    python experiments/run_collapse.py --variant extended --p 1.0 --seed 0 --lr 1e-5 --sequences 1000000 --name exact_extended_p1_s0
    python experiments/run_collapse.py --config configs/exact_extended_p1.yaml --seed 3

Outputs under --out-root (default results/exact/collapse)/<name>/: generations.csv (one row per generation),
gen<g>_train.csv (full per-evaluation log), gen<g>_model.pt, gen<g>_pool.npz (the exact training pool), run_config.json.
A YAML --config holds CollapseConfig fields under ``collapse:``; flags override it. Re-running a command whose output directory
already holds completed generations resumes it (see src/collapse.py), e.g. with a larger --generations.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collapse import CollapseConfig, run_collapse  # noqa: E402
from utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--name")
    ap.add_argument("--out-root", default="results/exact/collapse")
    ap.add_argument("--variant", choices=["base", "extended"])
    ap.add_argument("--p", type=float, dest="p_synthetic")
    ap.add_argument("--generations", type=int, dest="n_generations")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--sequences", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--eval-every", type=int, dest="eval_every")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--mode", choices=["replace", "accumulate"])
    ap.add_argument("--gen0-steps", type=int, dest="gen0_steps", help="train generation 0 for this many steps only (weak-generator test)")
    args = ap.parse_args()

    fields = {f.name for f in dataclasses.fields(CollapseConfig)}
    kw = dict(load_config(args.config)["collapse"]) if args.config else {}
    for key in ["variant", "p_synthetic", "n_generations", "seed", "lr", "sequences", "steps", "eval_every", "temperature", "mode", "gen0_steps"]:
        if getattr(args, key) is not None:
            kw[key] = getattr(args, key)
    if args.steps is not None:
        kw["sequences"] = None
    if args.greedy:
        kw["greedy"] = True
    name = args.name or f"{kw.get('variant', 'extended')}_p{kw.get('p_synthetic', 1.0):g}_s{kw.get('seed', 0)}"
    kw["out_dir"] = str(Path(args.out_root) / name)
    cfg = CollapseConfig(**{k: v for k, v in kw.items() if k in fields})
    print(f"=== {name}: {cfg.variant} p={cfg.p_synthetic} seed={cfg.seed} lr={cfg.lr} budget="
          f"{cfg.steps if cfg.steps else str(cfg.sequences) + ' sequences'} x {cfg.n_generations} generations ===", flush=True)
    t = time.time()
    run_collapse(cfg)
    print(f"=== done in {(time.time() - t) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
