"""Model collapse on the exact task: fit -> sample -> refit (Shumailov et al.), 10 generations.

Design (plan section 6; the decisions that are not fixed by the paper or the brief are marked DECISION):

* Generation 0 trains on the real enumerated space (78,400 sequences base / 156,800 extended).
* Generation n >= 1 trains on a pool of N = 78,400 sequences (both variants; DECISION: one pool size for every later
  generation, so that conditions are size-matched; the p = 0 control shows the effect of the size change itself):
    - the SYNTHETIC share (round(p * N) rows) is drawn WITHOUT replacement from the synthetic pool of model n-1;
    - the REAL share (the rest) is drawn without replacement from the real space (fresh real data wherever the space is
      larger than N; for the base variant the real space is exactly N, so p = 0 retrains on the whole space).
  Base synthetic pool: every real context ``A la B lb Q`` appears exactly once, with the query label sampled from model
  n-1 (no duplicates by construction). Extended synthetic pool: every real context with lq, S2 and l_S2 sampled from
  model n-1 (S2 is fed back as exemplar 0 of the sampled class, as the model sees it at generation time).
* The SAME model initialisation (seed) in every generation of a run, which removes the initialisation lottery as a
  confound; the ORDER of training batches differs per generation (DECISION: data_seed = 1000 * seed + generation, so
  generations do not visit the same pool positions at the same steps).
* A fixed training budget per generation, no early stopping; validation and test sets are always the REAL ones.
* Every generation's pool and checkpoint are saved; one row per generation goes to generations.csv.

Tier 2 options (all off by default, so the Tier 1 chains are reproduced exactly):
* ``mode = "accumulate"`` (Gerstgrasser et al. 2024): generation g >= 1 trains on R + S_1 + ... + S_g, where R is a fixed pool of
  N real rows (drawn once per run) and S_k is the full synthetic pool (every context once) of model k - 1; ``replace`` (the default)
  uses only the newest model's pool. The per-generation budget is unchanged, batches are drawn uniformly from the whole union.
  gen<g>_pool.npz then holds S_g (the new synthetic pool), not the union.
* ``gen0_steps``: generation 0 (the generator of the weak-generator test) is trained for this many steps instead of the budget.
* Resumable: a generation counts as done when generations.csv has its row and gen<g>_model.pt and gen<g>_pool.npz exist; a
  re-run skips the done prefix, restores the sampling generator (saved as gen<g>_rng.pt, or replayed from the saved models for
  chains written before it was saved) and continues, so a timed-out chain is simply resubmitted and a 10-generation chain can be
  continued to 20 with the same result as a single 20-generation run. Resuming with different settings is refused.

Pool statistics logged for every generation (what the marginal statistics of Shumailov et al. look like here)
    base      share of the pool whose query label is correct (the in-context label of Q's class);
    extended  P(S2 = Q's class) (real value 0.5), P(S2 not in {A, B}), the S2 class histogram's entropy (real: ln 50), its KL
              to generation 0's histogram against a noise floor from two independent real pools, the mass on the k = 10
              least frequent classes of that pool ("tail"; uniform: about 0.2), and the correctness of lq and of l_S2.
    accumulate mode additionally logs the same statistics of the newest synthetic pool alone, prefixed ``pool_new_``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import metrics as M
from model import IHTransformer
from train import EvalSets, TrainConfig, load_model, train_model
from utils import CSVLogger, ensure_dir, save_json

POOL_SIZE = 78_400
TAIL_K = 10


@dataclasses.dataclass
class CollapseConfig:
    variant: str = "extended"
    p_synthetic: float = 1.0
    n_generations: int = 10
    temperature: float = 1.0
    greedy: bool = False
    seed: int = 0
    sequences: int | None = 1_000_000       # per-generation budget (or steps)
    steps: int | None = None
    lr: float = 1e-5
    loss_weights: tuple[float, float, float] = M.EXTENDED_LOSS_WEIGHTS   # extended variant only (see metrics.py)
    eval_every: int = 250
    gen_batch_size: int = 2048
    out_dir: str = "results/exact/collapse/run"
    pool_size: int = POOL_SIZE
    mode: str = "replace"                   # "replace" or "accumulate" (see the module docstring)
    gen0_steps: int | None = None           # train generation 0 for this many steps only (weak-generator test)


# ------------------------------------------------------------------------------------------------ sampling a pool
@torch.no_grad()
def synthetic_pool(model: IHTransformer, contexts: np.ndarray, features: torch.Tensor, variant: str, temperature: float,
                   greedy: bool, gen: torch.Generator, batch_size: int = 2048) -> np.ndarray:
    """Every real context once, with the model's own sampled targets written into the synthetic columns."""
    rows = np.array(contexts, copy=True)
    ex0 = features[D.TRAIN_CLASSES, 0]
    for i in range(0, len(rows), batch_size):
        chunk = torch.as_tensor(rows[i:i + batch_size])
        b = D.collate(chunk, features, False)
        ex_ctx, lab_ctx = b.ex[:, :3], b.labels[:, :2]
        if variant == "base":
            lq = model.sample_query_label(ex_ctx, lab_ctx, temperature, greedy, gen)
            rows[i:i + batch_size, D.LQ_SYN] = lq.numpy()
        else:
            lq, s2, ls2 = model.sample_extended(ex_ctx, lab_ctx, ex0, temperature, greedy, gen)
            rows[i:i + batch_size, D.LQ_SYN] = lq.numpy()
            rows[i:i + batch_size, D.S2C_SYN] = s2.numpy()
            rows[i:i + batch_size, D.LS2_SYN] = ls2.numpy()
            rows[i:i + batch_size, [D.EX_S2]] = 0
    return rows


def mix_pool(real_space: np.ndarray, synthetic: np.ndarray, p: float, n_total: int, seed: int) -> np.ndarray:
    """round(p * n_total) synthetic rows and the rest real, each drawn WITHOUT replacement from its source."""
    rng = np.random.default_rng(seed)
    n_syn = int(round(p * n_total))
    n_real = n_total - n_syn
    if n_real > len(real_space) or n_syn > len(synthetic):
        raise ValueError(f"cannot draw {n_real} real / {n_syn} synthetic rows without replacement from {len(real_space)} / {len(synthetic)}")
    parts = []
    if n_real:
        parts.append(real_space[rng.permutation(len(real_space))[:n_real]])
    if n_syn:
        parts.append(synthetic[rng.permutation(len(synthetic))[:n_syn]])
    pool = np.concatenate(parts)
    return pool[rng.permutation(len(pool))]


# ------------------------------------------------------------------------------------------------ pool statistics
def _hist(c: np.ndarray, n: int = 50) -> np.ndarray:
    return np.bincount(c, minlength=n).astype(np.float64)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p, q = np.clip(p / p.sum(), eps, None), np.clip(q / q.sum(), eps, None)
    return float((p * np.log(p / q)).sum())


def pool_stats(pool: np.ndarray, variant: str, ref_hist: np.ndarray | None = None) -> dict:
    """Statistics of a training pool (real or synthetic); see the module docstring."""
    r = torch.as_tensor(pool)
    out = {"pool_size": int(len(pool)), "frac_synthetic": float((pool[:, D.LQ_SYN] >= 0).mean())}
    syn = pool[:, D.LQ_SYN] >= 0
    # query-label correctness among synthetic rows (real rows are correct by construction)
    lq_ok = (D.query_label(r) == D.true_query_label(r)).float().numpy()
    out["lq_correct_all"] = float(lq_ok.mean())
    out["lq_correct_synthetic"] = float(lq_ok[syn].mean()) if syn.any() else float("nan")
    if variant == "extended":
        c2 = D.s2_class(r).numpy()
        cq = D.query_class(r).numpy()
        in_ctx = (c2 == pool[:, D.C_A]) | (c2 == pool[:, D.C_B])
        h = _hist(c2)
        p = h / h.sum()
        order = np.sort(h)
        ls2_ok = (D.s2_label(r) == D.true_s2_label(r)).numpy()
        out.update({
            "p_s2_equals_q_class": float((c2 == cq).mean()),
            "p_s2_outside_context": float((~in_ctx).mean()),
            "s2_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
            "s2_tail_mass_k10": float(order[:TAIL_K].sum() / h.sum()),
            "s2_n_classes_used": int((h > 0).sum()),
            "ls2_correct_given_in_context": float(ls2_ok[in_ctx].mean()) if in_ctx.any() else float("nan"),
        })
        if ref_hist is not None:
            out["s2_kl_vs_gen0"] = _kl(h, ref_hist)
    return out


def noise_floor(real_space: np.ndarray, n: int = POOL_SIZE, seed: int = 123) -> float:
    """KL between the S2 class histograms of two independent real pools drawn from the real extended space: what 'no drift'
    looks like, to read the KL of synthetic pools against."""
    rng = np.random.default_rng(seed)
    a = real_space[rng.permutation(len(real_space))[:n]]
    b = real_space[rng.permutation(len(real_space))[:n]]
    ha, hb = _hist(D.s2_class(torch.as_tensor(a)).numpy()), _hist(D.s2_class(torch.as_tensor(b)).numpy())
    return _kl(ha, hb)


# ------------------------------------------------------------------------------------------------ resuming
RESUME_IGNORED = {"n_generations", "out_dir"}


def _cfg_json(cfg: CollapseConfig) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


def check_resume_config(out: Path, cfg: CollapseConfig) -> list:
    """Refuse to continue a run whose saved settings differ from ``cfg`` (except the number of generations and the output
    directory); returns the list of earlier resumes recorded in run_config.json. Fields added after a run was written take their
    defaults."""
    f = out / "run_config.json"
    if not f.exists():
        return []
    saved = json.load(open(f))
    defaults, now = _cfg_json(CollapseConfig()), _cfg_json(cfg)
    prev = {**{k: v for k, v in defaults.items() if k not in saved["collapse"]}, **saved["collapse"]}
    diff = {k: (prev.get(k), now[k]) for k in now if k not in RESUME_IGNORED and prev.get(k) != now[k]}
    if diff:
        raise ValueError(f"{out} was written with different settings; refusing to resume (saved, requested): {diff}")
    return saved.get("resumes", [])


def completed_generations(out: Path) -> int:
    """Number of leading generations that are done: a row in generations.csv plus gen<g>_model.pt and gen<g>_pool.npz."""
    f = out / "generations.csv"
    if not f.exists():
        return 0
    done = 0
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["generation"]) != done or not all((out / f"gen{done}_{k}").exists() for k in ("model.pt", "pool.npz")):
                break
            done += 1
    return done


def _keep_first_rows(path: Path, n: int) -> None:
    """Rewrite generations.csv with only its first n rows (a partial generation left by a killed run is retrained)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    if len(rows) == n:
        return
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows[:n])
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ the pipeline
def run_collapse(cfg: CollapseConfig, features: torch.Tensor | None = None, verbose: bool = True) -> Path:
    if cfg.mode not in ("replace", "accumulate"):
        raise ValueError(f"mode must be 'replace' or 'accumulate', got {cfg.mode!r}")
    features = features if features is not None else D.OmniglotFeatures().tensor
    ext = cfg.variant == "extended"
    accumulate = cfg.mode == "accumulate"
    out = Path(cfg.out_dir)
    ensure_dir(out)
    resumes = check_resume_config(out, cfg)
    n_done = completed_generations(out)

    real_space = D.enumerate_train_space(extended=ext)
    contexts = D.enumerate_train_space(extended=False)
    evals = EvalSets(features, ext)
    gen_rng = torch.Generator().manual_seed(cfg.seed * 7919 + 17)
    ref_hist = _hist(D.s2_class(torch.as_tensor(real_space)).numpy()) if ext else None     # generation 0's S2 class histogram
    floor = noise_floor(D.enumerate_train_space(True)) if ext else float("nan")
    samples = accumulate or cfg.p_synthetic > 0      # does generation g >= 1 draw from the generator (and so consume gen_rng)?
    real_fixed = mix_pool(real_space, contexts, 0.0, cfg.pool_size, seed=cfg.seed * 1009) if accumulate else None   # R of accumulate mode
    syn_parts: list[np.ndarray] = []                 # S_1 ... S_g (accumulate mode)
    model_prev = None

    if n_done >= cfg.n_generations:
        if verbose:
            print(f"[{out.name}] all {n_done} requested generations are already done", flush=True)
        return out
    if n_done:
        _keep_first_rows(out / "generations.csv", n_done)
        model_prev = load_model(out / f"gen{n_done - 1}_model.pt")[0]
        rng_file = out / f"gen{n_done - 1}_rng.pt"
        if n_done >= 2 and rng_file.exists():
            gen_rng.set_state(torch.load(rng_file, weights_only=False))
        elif n_done >= 2 and samples:                # replay the draws of generations 1 .. n_done-1 (results discarded)
            for k in range(1, n_done):
                synthetic_pool(load_model(out / f"gen{k - 1}_model.pt")[0], contexts, features, cfg.variant, cfg.temperature, cfg.greedy,
                               gen_rng, cfg.gen_batch_size)
        if accumulate:
            syn_parts = [np.load(out / f"gen{k}_pool.npz")["pool"].astype(np.int64) for k in range(1, n_done)]
        resumes = resumes + [{"from_generation": n_done, "time": time.strftime("%Y-%m-%d %H:%M:%S")}]
        if verbose:
            print(f"[{out.name}] resuming: generations 0..{n_done - 1} are done, continuing to {cfg.n_generations - 1}", flush=True)
    save_json({"collapse": dataclasses.asdict(cfg), "resumes": resumes}, out / "run_config.json")
    logger = CSVLogger(out / "generations.csv", resume=n_done > 0)

    for g in range(n_done, cfg.n_generations):
        extra = {}
        if g == 0:
            pool = save_pool = real_space
        else:
            syn = synthetic_pool(model_prev, contexts, features, cfg.variant, cfg.temperature, cfg.greedy, gen_rng, cfg.gen_batch_size) if samples else contexts
            if samples:
                torch.save(gen_rng.get_state(), out / f"gen{g}_rng.pt")
            if accumulate:
                syn_parts.append(syn)
                pool, save_pool = np.concatenate([real_fixed] + syn_parts), syn
                extra = {f"pool_new_{k}": v for k, v in pool_stats(syn, cfg.variant, ref_hist).items()}
            else:
                pool = save_pool = mix_pool(real_space, syn, cfg.p_synthetic, cfg.pool_size, seed=cfg.seed * 1009 + g)
        np.savez_compressed(out / f"gen{g}_pool.npz", pool=save_pool.astype(np.int16))
        stats = pool_stats(pool, cfg.variant, ref_hist)
        if accumulate and g == 0:                        # the "newest data" of generation 0 is the real space itself (same columns in every row)
            extra = {f"pool_new_{k}": v for k, v in stats.items()}

        steps =cfg.gen0_steps if (g == 0 and cfg.gen0_steps) else cfg.steps
        tcfg = TrainConfig(variant=cfg.variant, steps=steps, sequences=None if steps is not None else cfg.sequences,
                           lr=cfg.lr, loss_weights=tuple(cfg.loss_weights), eval_every=cfg.eval_every, seed=cfg.seed, data_seed=1000 * cfg.seed + g,
                           log_path=str(out / f"gen{g}_train.csv"), ckpt_path=str(out / f"gen{g}_model.pt"), verbose=False)
        res = train_model(tcfg, features=features, pool=pool, evals=evals)
        model_prev = res["model"]
        last = res["summary"]["final"]
        row = {"generation": g, "steps": res["summary"]["steps"], "seconds": res["summary"]["seconds"], "noise_floor_s2_kl": floor,
               **{k: v for k, v in last.items() if "/" in k or k in ("ind_max", "ind_mean", "match_max", "prev_max")},
               **{f"ind_L2H{h}": last[f"ind_L2H{h}"] for h in range(model_prev.cfg.n_heads)},
               **{f"pool_{k}": v for k, v in stats.items()}, **extra}
        logger.log(row)
        if verbose:
            msg = f"[{out.name} gen {g}] val acc {row['val/acc']:.3f} copy {row['test_relabel/acc']:.3f} ind_max {row['ind_max']:.2f}"
            if ext:
                msg += f" | pool: P(S2=Q) {stats['p_s2_equals_q_class']:.3f} tail {stats['s2_tail_mass_k10']:.3f}"
            print(msg, flush=True)
    return out
