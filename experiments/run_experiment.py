"""Run a single experiment (plain training or a full collapse pipeline) from
a config file, optionally overriding the seed for multi-seed replication.

Usage:
    python experiments/run_experiment.py --config configs/base_no_collapse.yaml --mode train
    python experiments/run_experiment.py --config configs/extended_collapse.yaml --mode collapse
    python experiments/run_experiment.py --config configs/extended_collapse.yaml --mode collapse --seed 1

--seed overrides both train_cfg.seed (model init, held fixed across a collapse
run's generations) and, in collapse mode, gen_cfg.seed (data-mixing
randomness) with the SAME value, and suffixes all output paths with
_seed{N} so replicate runs never collide. Omit --seed to use the seed
baked into the config file as-is (single-seed / backward compatible).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import TaskConfig
from model import ModelConfig
from train import TrainConfig, train_model
from collapse import CollapseConfig, run_collapse
from utils import load_config


def _suffix_path(path: str, seed: int) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_seed{seed}{p.suffix}"))


def run(config_path: str, mode: str, seed: int | None = None):
    cfg = load_config(config_path)
    tcfg = TaskConfig(**cfg["task"])
    mcfg = ModelConfig(**cfg["model"])
    train_cfg = TrainConfig(**cfg["train"])

    if mode == "train":
        if seed is not None:
            train_cfg = dataclasses.replace(
                train_cfg,
                seed=seed,
                log_path=_suffix_path(train_cfg.log_path, seed),
                ckpt_path=_suffix_path(train_cfg.ckpt_path, seed) if train_cfg.ckpt_path else None,
            )
        result = train_model(tcfg, mcfg, train_cfg, verbose=True)
        return result["summary"]
    elif mode == "collapse":
        gen_cfg = CollapseConfig(**cfg["collapse"])
        run_name = cfg.get("run_name", Path(config_path).stem)
        if seed is not None:
            train_cfg = dataclasses.replace(train_cfg, seed=seed)
            gen_cfg = dataclasses.replace(gen_cfg, seed=seed)
            run_name = f"{run_name}_seed{seed}"
        out_dir = run_collapse(tcfg, mcfg, train_cfg, gen_cfg, run_name=run_name, verbose=True)
        return {"out_dir": str(out_dir)}
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "collapse"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override seed for multi-seed replication")
    args = parser.parse_args()
    print(f"=== Running {args.config} [{args.mode}] seed={args.seed} ===")
    result = run(args.config, args.mode, seed=args.seed)
    print(f"=== Done: {result} ===")


if __name__ == "__main__":
    main()
