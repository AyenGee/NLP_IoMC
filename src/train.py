"""Step-based training exactly as in Singh et al. (2024), Appendix A: Adam, constant learning rate 1e-5,
batch size 32, no weight decay, no warm-up, no schedule, no gradient clipping, no early stopping, a fixed budget.

Budget units. The reference script counts ``train_iters`` in SEQUENCES (its loop is
``range(0, train_iters, train_bs)``), so its main run, ``--train_iters 1000000 --train_bs 32``, is
31,250 optimiser steps; the paper's appendix likewise says a standard run takes about 7 minutes "for 1e6
sequences seen". Set the budget in either unit: ``steps`` (optimiser steps) or ``sequences`` (steps =
sequences // batch_size). Logged rows carry both ``step`` and ``sequences``.

Every ``eval_every`` steps, the loss, accuracy and perplexity on all five evaluation sets, the running training
loss, and the interpretability metrics (data/metrics docstrings) are written to a CSV.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import metrics as M
from model import IHTransformer, ModelConfig
from utils import CSVLogger, ensure_dir, save_json


@dataclasses.dataclass
class TrainConfig:
    variant: str = "base"                   # "base" or "extended"
    steps: int | None = None                # optimiser steps ...
    sequences: int | None = 1_000_000       # ... or a budget in sequences (the paper: 1e6)
    batch_size: int = 32
    lr: float = 1e-5
    eval_every: int = 250                   # steps between evaluations (the paper: every 5,000 sequences ~ 156 steps)
    seed: int = 0                           # model initialisation AND the order of training batches
    n_threads: int = 1
    log_path: str = "results/exact/logs/run.csv"
    ckpt_path: str | None = None
    ckpt_steps: tuple[int, ...] = ()        # extra checkpoints at these steps (e.g. before / during / after the phase change)
    mlp_ratio: float | None = None
    n_heads: int = 8
    d_model: int = 64
    verbose: bool = True

    @property
    def extended(self) -> bool:
        return self.variant == "extended"

    @property
    def n_steps(self) -> int:
        if self.steps is not None:
            return int(self.steps)
        return int(self.sequences) // self.batch_size

    def model_config(self) -> ModelConfig:
        return ModelConfig(d_model=self.d_model, n_heads=self.n_heads, extended=self.extended, mlp_ratio=self.mlp_ratio)


class EvalSets:
    """The five evaluation sets, collated once (embeddings looked up) and kept as tensors."""

    def __init__(self, features: torch.Tensor, extended: bool, sets: dict[str, np.ndarray] | None = None):
        sets = sets if sets is not None else D.load_or_build_eval_sets()
        self.extended = extended
        self.rows = {k: torch.as_tensor(v, dtype=torch.long) for k, v in sets.items()}
        self.batches = {k: D.collate(v, features, extended) for k, v in self.rows.items()}


@torch.no_grad()
def evaluate(model: IHTransformer, evals: EvalSets, with_attention: bool = True) -> dict:
    """Loss / accuracy / perplexity on every evaluation set, plus the attention metrics on ``train_eval``."""
    model.eval()
    row = {}
    ext = evals.extended
    for name, b in evals.batches.items():
        out = model(b.ex, b.labels)
        terms = M.loss_terms(out, b, ext)
        acc = M.accuracies(out, b, ext)
        row[f"{name}/loss"] = terms["loss_q"].item()
        row[f"{name}/acc"] = acc["acc_q"]
        row[f"{name}/ppl"] = M.perplexity(terms["loss_q"].item())
        if ext:
            row[f"{name}/loss_s2"], row[f"{name}/loss_ls2"] = terms["loss_s2"].item(), terms["loss_ls2"].item()
            row[f"{name}/acc_s2"], row[f"{name}/acc_ls2"] = acc["acc_s2"], acc["acc_ls2"]
    if with_attention:
        b = evals.batches["train_eval"]
        out = model(b.ex[:, :3], b.labels[:, :2], return_details=True)
        am = M.attention_metrics(out["attn"], b.q_is_A)
        row.update(am)
        row.update(M.head_summary(am, model.cfg.n_heads, model.cfg.n_layers))
    return row


def train_model(cfg: TrainConfig, features: torch.Tensor | None = None, pool: np.ndarray | None = None,
                init_state: dict | None = None, evals: EvalSets | None = None) -> dict:
    """Train one model. ``pool`` (integer rows) defaults to the real enumerated training space; the collapse
    pipeline passes each generation's pool. ``init_state`` loads a given initial state (the collapse pipeline reuses
    the same initialisation in every generation)."""
    torch.set_num_threads(cfg.n_threads)
    features = features if features is not None else D.OmniglotFeatures().tensor
    pool = pool if pool is not None else D.enumerate_train_space(extended=cfg.extended)
    evals = evals if evals is not None else EvalSets(features, cfg.extended)

    torch.manual_seed(cfg.seed)
    model = IHTransformer(cfg.model_config())
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    sampler = D.TrainSampler(pool, features, cfg.extended, seed=cfg.seed + 1)
    logger = CSVLogger(cfg.log_path)
    ckpt_steps = set(cfg.ckpt_steps)

    n_steps = cfg.n_steps
    run_loss = run_acc = 0.0
    run_n = 0
    t0 = time.time()
    last_row = {}
    for step in range(1, n_steps + 1):
        model.train()
        b = sampler.next_batch(cfg.batch_size)
        out = model(b.ex, b.labels)
        terms = M.loss_terms(out, b, cfg.extended)
        opt.zero_grad(set_to_none=True)
        terms["loss"].backward()
        opt.step()
        run_loss += terms["loss_q"].item()
        run_acc += (out["label_logits_q"].argmax(-1) == b.query_label).float().mean().item()
        run_n += 1
        if step in ckpt_steps and cfg.ckpt_path:
            _save(model, cfg, Path(cfg.ckpt_path).with_name(f"{Path(cfg.ckpt_path).stem}_step{step}.pt"), step)
        if step % cfg.eval_every == 0 or step == n_steps or step == 1:
            row = {"step": step, "sequences": step * cfg.batch_size, "train_loss_running": run_loss / run_n,
                   "train_acc_running": run_acc / run_n, "seconds": time.time() - t0}
            row.update(evaluate(model, evals))
            logger.log(row)
            last_row = row
            run_loss = run_acc = 0.0
            run_n = 0
            if cfg.verbose:
                print(f"step {step:7d} | train loss {row['train_loss_running']:.4f} | val acc {row['val/acc']:.3f} "
                      f"loss {row['val/loss']:.4f} | ind_max {row['ind_max']:.3f} | {row['seconds']:.0f}s", flush=True)
    seconds = time.time() - t0
    summary = {"steps": n_steps, "sequences": n_steps * cfg.batch_size, "seconds": seconds,
               "steps_per_second": n_steps / max(seconds, 1e-9), "final": last_row}
    if cfg.ckpt_path:
        _save(model, cfg, Path(cfg.ckpt_path), n_steps)
        save_json({k: v for k, v in summary.items() if k != "final"} | {"final": last_row}, Path(cfg.ckpt_path).with_suffix(".summary.json"))
    return {"model": model, "summary": summary, "evals": evals}


def _save(model: IHTransformer, cfg: TrainConfig, path: Path, step: int) -> None:
    ensure_dir(path.parent)
    torch.save({"model_state": model.state_dict(), "model_cfg": dataclasses.asdict(model.cfg),
                "train_cfg": dataclasses.asdict(cfg), "step": step}, path)


def load_model(path: str | os.PathLike) -> tuple[IHTransformer, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = IHTransformer(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck
