"""Train one model on the exact Singh et al. (2024) task and write its curves.

    python experiments/train_run.py --variant base --sequences 1000000 --seed 0 --name gen0_base_s0
    python experiments/train_run.py --variant extended --steps 20000 --eval-every 250 --name smoke_ext
    python experiments/train_run.py --config configs/exact_gen0_base.yaml --seed 1

Outputs (under --out-dir, default results/exact): logs/<name>.csv (every evaluation point) and
checkpoints/<name>.pt (+ .summary.json). A YAML --config holds TrainConfig fields under ``train:``; flags override it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import TrainConfig, train_model  # noqa: E402
from utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="results/exact")
    ap.add_argument("--variant", choices=["base", "extended"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--sequences", type=int, help="budget in sequences (the paper: 1,000,000 = 31,250 steps at batch 32)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ckpt-steps", type=int, nargs="*")
    ap.add_argument("--mlp-ratio", type=float)
    ap.add_argument("--n-threads", type=int)
    args = ap.parse_args()

    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    kw = dict(load_config(args.config)["train"]) if args.config else {}
    for flag, key in [("variant", "variant"), ("steps", "steps"), ("sequences", "sequences"), ("batch_size", "batch_size"), ("lr", "lr"),
                      ("eval_every", "eval_every"), ("seed", "seed"), ("mlp_ratio", "mlp_ratio"), ("n_threads", "n_threads")]:
        if getattr(args, flag) is not None:
            kw[key] = getattr(args, flag)
    if args.steps is not None:
        kw["sequences"] = None                      # an explicit step budget replaces the default sequence budget
    if args.ckpt_steps is not None:
        kw["ckpt_steps"] = tuple(args.ckpt_steps)
    name = args.name or f"{kw.get('variant', 'base')}_s{kw.get('seed', 0)}"
    out = Path(args.out_dir)
    kw.setdefault("log_path", str(out / "logs" / f"{name}.csv"))
    kw.setdefault("ckpt_path", str(out / "checkpoints" / f"{name}.pt"))
    kw = {k: v for k, v in kw.items() if k in fields}
    cfg = TrainConfig(**kw)
    print(f"=== {name}: {cfg.variant}, {cfg.n_steps} steps x batch {cfg.batch_size} = {cfg.n_steps * cfg.batch_size} sequences, "
          f"lr {cfg.lr}, seed {cfg.seed} ===", flush=True)
    res = train_model(cfg)
    s = res["summary"]
    print(f"=== done: {s['steps']} steps in {s['seconds']:.0f}s ({s['steps_per_second']:.1f} steps/s) ===")


if __name__ == "__main__":
    main()
