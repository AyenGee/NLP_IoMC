"""Training loop for the base and extended induction-head tasks.

Usage:
    python train.py --config ../configs/base_no_collapse.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import TaskConfig, make_splits, collate_batch, ListDataset
from model import InductionTransformer, ModelConfig
from metrics import compute_losses, perplexity, attention_entropy, induction_score, prev_token_score, induction_offset_scores, best_head_by_offset
from utils import set_seed, get_device, load_config, CSVLogger, save_json


@dataclasses.dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    n_epochs: int = 60
    warmup_steps: int = 100
    grad_clip: float = 1.0
    extended: bool = False
    aux_weight: float = 1.0
    eval_every: int = 1
    interp_every: int = 5
    patience: int | None = None
    seed: int = 0
    device: str = "auto"
    n_train: int = 20000
    n_val: int = 2000
    n_test: int = 2000
    data_seed: int = 0
    log_path: str = "results/logs/run.csv"
    ckpt_path: str | None = "results/checkpoints/run.pt"


def build_model(tcfg: TaskConfig, mcfg: ModelConfig, device: torch.device) -> InductionTransformer:
    model = InductionTransformer(tcfg, mcfg).to(device)
    return model


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, extended: bool, aux_weight: float, device) -> dict:
    model.eval()
    totals = {"loss": 0.0, "acc": 0.0, "n": 0}
    if extended:
        totals.update({"aux_label_loss": 0.0, "aux_label_acc": 0.0, "aux_symbol_loss": 0.0, "aux_symbol_acc": 0.0})
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["symbol_tokens"], batch["label_tokens"])
        loss_out = compute_losses(out, batch, extended, aux_weight)
        bs = batch["query_label"].shape[0]
        totals["loss"] += loss_out.base_loss.item() * bs
        totals["acc"] += loss_out.base_acc * bs
        totals["n"] += bs
        if extended:
            totals["aux_label_loss"] += loss_out.aux_label_loss.item() * bs
            totals["aux_label_acc"] += loss_out.aux_label_acc * bs
            totals["aux_symbol_loss"] += loss_out.aux_symbol_loss.item() * bs
            totals["aux_symbol_acc"] += loss_out.aux_symbol_acc * bs
    n = totals.pop("n")
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def interp_snapshot(model, loader, device) -> dict:
    """One batch's worth of attention entropy / induction / prev-token scores
    -- cheap enough to compute every few epochs during training."""
    model.eval()
    batch = next(iter(loader))
    batch = _move_batch(batch, device)
    out = model(batch["symbol_tokens"], batch["label_tokens"], return_attention=True)
    ent = attention_entropy(out.attn_maps)
    ind = induction_score(out.attn_maps, batch["symbol_tokens"])
    prev = prev_token_score(out.attn_maps)
    def best_head(scores: dict) -> float:
        # Per-head entries are keyed 'layer{l}_head{h}'; the other keys are aggregates.
        return max(v for k, v in scores.items() if "_head" in k)

    off = best_head_by_offset(induction_offset_scores(out.attn_maps, batch["symbol_tokens"]))
    return {
        "attn_entropy_mean": ent["overall_mean"],
        "induction_score_mean": ind["overall_mean"],
        "induction_score_baseline": ind["baseline_mean"],
        "prev_token_score_mean": prev["overall_mean"],
        # The mean over all heads is diluted by heads that play no part in the
        # circuit; the best single head is the better 'is there an induction
        # head' signal (added after the first 18 runs, so absent from those logs).
        "induction_score_max": best_head(ind),
        "prev_token_score_max": best_head(prev),
        # Best head at each offset from an earlier occurrence of the query symbol (see
        # metrics.induction_offset_scores): 0 = the earlier occurrence itself, 1 = the label
        # after it (== induction_score_max), 2 = the next symbol. Added for the corrected-
        # protocol runs after a model was found that solves the task with an offset-0 head.
        "induction_off0_max": off[0][1],
        "induction_off2_max": off[2][1],
    }


def train_model(
    tcfg: TaskConfig,
    mcfg: ModelConfig,
    train_cfg: TrainConfig,
    train_items: list[dict] | None = None,
    verbose: bool = True,
) -> dict:
    """Train one model. If `train_items` is given (list of already-tokenised
    sequence dicts, e.g. real+synthetic mix from collapse.py) it is used as
    the training pool instead of freshly generating one from `tcfg`; val/test
    are always freshly generated from `tcfg` (the ground-truth task
    distribution) so evaluation always measures true task performance."""
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)

    train_ds_fresh, val_ds, test_ds = make_splits(
        tcfg, train_cfg.n_train, train_cfg.n_val, train_cfg.n_test, base_seed=train_cfg.data_seed
    )
    if train_items is None:
        train_ds = train_ds_fresh
    else:
        train_ds = ListDataset(train_items, tcfg)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_model(tcfg, mcfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    total_steps = train_cfg.n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, train_cfg.warmup_steps, total_steps)
    )

    logger = CSVLogger(train_cfg.log_path)
    best_val_loss = float("inf")
    epochs_since_improve = 0
    step = 0
    t0 = time.time()

    for epoch in range(train_cfg.n_epochs):
        model.train()
        running = {"loss": 0.0, "acc": 0.0, "n": 0}
        for batch in train_loader:
            batch = _move_batch(batch, device)
            out = model(batch["symbol_tokens"], batch["label_tokens"])
            loss_out = compute_losses(out, batch, train_cfg.extended, train_cfg.aux_weight)
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            bs = batch["query_label"].shape[0]
            running["loss"] += loss_out.base_loss.item() * bs
            running["acc"] += loss_out.base_acc * bs
            running["n"] += bs

        train_loss = running["loss"] / running["n"]
        train_acc = running["acc"] / running["n"]

        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_ppl": perplexity(train_loss),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - t0,
        }

        if epoch % train_cfg.eval_every == 0 or epoch == train_cfg.n_epochs - 1:
            val_metrics = evaluate(model, val_loader, train_cfg.extended, train_cfg.aux_weight, device)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_ppl"] = perplexity(val_metrics["loss"])
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if epoch % train_cfg.interp_every == 0 or epoch == train_cfg.n_epochs - 1:
            row.update(interp_snapshot(model, val_loader, device))

        logger.log(row)
        if verbose:
            msg = f"epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.3f}"
            if "val_loss" in row:
                msg += f" | val_loss {row['val_loss']:.4f} acc {row['val_acc']:.3f}"
            if "induction_score_mean" in row:
                msg += f" | ind_score {row['induction_score_mean']:.3f} (base {row['induction_score_baseline']:.3f})"
            print(msg)

        if train_cfg.patience is not None and epochs_since_improve >= train_cfg.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no val improvement for {train_cfg.patience} evals).")
            break

    test_metrics = evaluate(model, test_loader, train_cfg.extended, train_cfg.aux_weight, device)
    summary = {
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_ppl": perplexity(test_metrics["loss"]),
        "best_val_loss": best_val_loss,
        "n_epochs_run": epoch + 1,
        "train_seconds": time.time() - t0,
    }
    if train_cfg.extended:
        summary["test_aux_label_acc"] = test_metrics["aux_label_acc"]
        summary["test_aux_symbol_acc"] = test_metrics["aux_symbol_acc"]

    if train_cfg.ckpt_path:
        ckpt_path = Path(train_cfg.ckpt_path)
        os.makedirs(ckpt_path.parent, exist_ok=True)  # race-safe under parallel Slurm array tasks
        torch.save(
            {
                "model_state": model.state_dict(),
                "tcfg": dataclasses.asdict(tcfg),
                "mcfg": dataclasses.asdict(mcfg),
                "train_cfg": dataclasses.asdict(train_cfg),
                "summary": summary,
            },
            ckpt_path,
        )
        save_json(summary, ckpt_path.with_suffix(".summary.json"))

    if verbose:
        print(f"Test: loss {summary['test_loss']:.4f} acc {summary['test_acc']:.3f} ppl {summary['test_ppl']:.2f}")

    return {"model": model, "device": device, "summary": summary, "val_loader": val_loader, "test_loader": test_loader}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tcfg = TaskConfig(**cfg["task"])
    mcfg = ModelConfig(**cfg["model"])
    train_cfg = TrainConfig(**cfg["train"])

    train_model(tcfg, mcfg, train_cfg)


if __name__ == "__main__":
    main()
