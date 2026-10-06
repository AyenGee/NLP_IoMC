"""Model collapse generation loop: Data0 -> model0 -> Data1 (sampled from
model0) -> model1 -> ... -> modelN (Shumailov et al.'s 'fit -> sample ->
refit' recursion).

Two sampling regimes, matching the base/extended task split:
  - base: the model only outputs a query label, so "sampling from the model"
    means taking a REAL context (freshly drawn from the true task
    distribution) and replacing only its query label with the model's own
    prediction. Context structure can never drift this way -- only the
    label assigned to it can -- so collapse is expected to be weak. This is
    implemented as the honest baseline the brief asks for, not skipped.
  - extended: the model can autoregressively roll out an entire synthetic
    sequence (context symbols, their labels, the query, and its label) via
    InductionTransformer.generate_batch, so its own errors on the auxiliary
    next-symbol/next-label predictions can compound and skew the symbol and
    label distributions of the data fed to the next generation.

At every generation n >= 1, the training pool mixes a `p_synthetic`
fraction sampled from generation (n-1)'s model with fresh real data (not
accumulated synthetic data from earlier generations), matching the classic
single-hop recursion in Shumailov et al.'s Figure 2. `p_synthetic` is a
config knob so partial-replacement can be studied, not just full
replacement.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

from data import TaskConfig, SymbolicICLDataset, dataset_to_items, mix_datasets, collate_batch, save_items
from model import ModelConfig, InductionTransformer
from metrics import distribution_stats, structure_stats, pool_distinct_fraction
from train import TrainConfig, train_model, interp_snapshot
from utils import get_device, CSVLogger, save_json, ensure_dir


@dataclasses.dataclass
class CollapseConfig:
    n_generations: int = 10
    p_synthetic: float = 1.0
    pool_size: int | None = None  # defaults to train_cfg.n_train
    variant: str = "extended"     # "extended" or "base"
    gen_batch_size: int = 256
    temperature: float = 1.0
    seed: int = 0
    # True (default, what the original 18 runs used): each part of a generation's
    # pool is a bootstrap resample, ~63% distinct sequences. False: distinct
    # sequences only (control for the duplication confound; see README).
    with_replacement: bool = True


@torch.no_grad()
def sample_base_synthetic(
    model: InductionTransformer, real_items: list[dict], device: torch.device, temperature: float = 1.0
) -> list[dict]:
    """Base-variant sampling: keep the real context, resample only the query
    label from the model's own predictive distribution."""
    model.eval()
    out_items = []
    bs = 256
    for i in range(0, len(real_items), bs):
        chunk = real_items[i : i + bs]
        batch = {k: v.to(device) for k, v in collate_batch(chunk).items()}
        out = model(batch["symbol_tokens"], batch["label_tokens"])
        probs = torch.softmax(out.label_logits[:, -1, :] / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1).cpu()
        for j, item in enumerate(chunk):
            new_item = dict(item)
            new_item["query_label"] = sampled[j]
            nlt = item["next_label_target"].clone()
            nlt[-1] = sampled[j]
            new_item["next_label_target"] = nlt
            out_items.append(new_item)
    return out_items


def sample_synthetic_pool(
    model: InductionTransformer,
    tcfg: TaskConfig,
    variant: str,
    pool_size: int,
    device: torch.device,
    gen_batch_size: int,
    temperature: float,
    real_seed: int,
) -> list[dict]:
    if variant == "extended":
        items = []
        model.eval()
        while len(items) < pool_size:
            bs = min(gen_batch_size, pool_size - len(items))
            items += model.generate_batch(bs, device, temperature=temperature)
        return items
    elif variant == "base":
        real_ctx_items = dataset_to_items(SymbolicICLDataset(tcfg, pool_size, seed=real_seed))
        return sample_base_synthetic(model, real_ctx_items, device, temperature=temperature)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_collapse(
    tcfg: TaskConfig,
    mcfg: ModelConfig,
    train_cfg: TrainConfig,
    gen_cfg: CollapseConfig,
    run_name: str,
    out_root: str = "results/collapse",
    verbose: bool = True,
) -> Path:
    device = get_device(train_cfg.device)
    out_dir = Path(out_root) / run_name
    ensure_dir(out_dir)  # race-safe: parallel Slurm array tasks create the shared parent concurrently
    gen_logger = CSVLogger(out_dir / "generations.csv")
    save_json(
        {"task": dataclasses.asdict(tcfg), "model": dataclasses.asdict(mcfg),
         "train": dataclasses.asdict(train_cfg), "collapse": dataclasses.asdict(gen_cfg)},
        out_dir / "run_config.json",
    )

    pool_size = gen_cfg.pool_size or train_cfg.n_train
    real_seed_base = train_cfg.data_seed

    gen0_items = dataset_to_items(SymbolicICLDataset(tcfg, pool_size, seed=real_seed_base))
    _, gen0_symbol_counts, gen0_label_counts = distribution_stats(gen0_items, tcfg.K, tcfg.L)

    model_prev = None
    for gen in range(gen_cfg.n_generations):
        if gen == 0:
            train_items = gen0_items
        else:
            p = gen_cfg.p_synthetic
            # Skip work whose result mix_datasets would never use (p=0: no synthetic
            # part; p=1: no real part). Neither touches the global RNG, so results
            # for the p values already run are unchanged.
            synthetic_items = [] if p <= 0.0 else sample_synthetic_pool(
                model_prev, tcfg, gen_cfg.variant, pool_size, device,
                gen_cfg.gen_batch_size, gen_cfg.temperature,
                real_seed=real_seed_base + gen * 7919 + 1,
            )
            real_pool_items = [] if p >= 1.0 else dataset_to_items(
                SymbolicICLDataset(tcfg, pool_size, seed=real_seed_base + gen * 7919 + 2)
            )
            train_items = mix_datasets(
                real_pool_items, synthetic_items, p, pool_size,
                seed=gen_cfg.seed * 1009 + gen,
                with_replacement=gen_cfg.with_replacement,
            )

        save_items(train_items, out_dir / f"gen{gen}_data.pt")

        gen_train_cfg = dataclasses.replace(
            train_cfg,
            # NOTE: model-init seed is deliberately held FIXED across
            # generations (not offset by `gen`). We found the base variant's
            # optimization landscape is seed-sensitive -- some
            # initializations permanently memorise (100% train acc) instead
            # of generalising via induction (~chance val/test acc), even on
            # clean, uncollapsed data (see README / abstract Limitations).
            # Varying the seed per generation would confound that
            # pre-existing seed-lottery effect with the collapse effect
            # we're trying to measure, so every generation reuses the same
            # verified-good initialization seed and only the training DATA
            # changes across generations.
            log_path=str(out_dir / f"gen{gen}_train.csv"),
            ckpt_path=str(out_dir / f"gen{gen}_model.pt"),
        )
        result = train_model(tcfg, mcfg, gen_train_cfg, train_items=train_items, verbose=False)
        model_prev = result["model"]

        dist_stats, _, _ = distribution_stats(
            train_items, tcfg.K, tcfg.L,
            ref_symbol_counts=gen0_symbol_counts, ref_label_counts=gen0_label_counts,
        )
        interp_stats = interp_snapshot(model_prev, result["val_loader"], device)

        # Pool-structure diagnostics: what the marginal KL above cannot see (label
        # consistency, repeat counts, ...), plus how many sequences are distinct.
        pool_stats = structure_stats(train_items, tcfg.n_classes, tcfg.burstiness)
        row = {
            "generation": gen, **result["summary"], **dist_stats, **interp_stats,
            "pool_distinct_frac": pool_distinct_fraction(train_items),
            **{f"pool_{k}": v for k, v in pool_stats.items()},
        }
        gen_logger.log(row)
        if verbose:
            print(
                f"[{run_name} gen {gen}] test_acc={result['summary']['test_acc']:.3f} "
                f"symbol_entropy={dist_stats['symbol_entropy']:.3f} "
                f"kl_vs_gen0={dist_stats.get('symbol_kl_vs_gen0', float('nan')):.4f} "
                f"induction_score={interp_stats['induction_score_mean']:.3f}"
            )

    return out_dir
