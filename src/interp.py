"""Interpretability tooling: attention-map visualisation, induction/prev-token
score and distribution-drift trends across generations, and symbol-embedding
geometry (PCA). Reuses the scoring functions in metrics.py so numbers shown
in figures are computed identically to what's logged during training.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from sklearn.decomposition import PCA

from data import TaskConfig, generate_sequence, sequence_to_tokens
from model import InductionTransformer, ModelConfig
from metrics import induction_score, prev_token_score


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tcfg = TaskConfig(**ckpt["tcfg"])
    mcfg = ModelConfig(**ckpt["mcfg"])
    model = InductionTransformer(tcfg, mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tcfg, mcfg, ckpt.get("summary", {})


def _position_labels(tcfg: TaskConfig) -> list[str]:
    m = tcfg.context_len
    labels = []
    for j in range(m):
        labels.append(f"S{j+1}")
        labels.append(f"L{j+1}")
    labels.append("Sq")
    return labels


@torch.no_grad()
def plot_attention_maps(
    model: InductionTransformer,
    tcfg: TaskConfig,
    seed: int = 0,
    title_prefix: str = "",
    save_path: str | Path | None = None,
):
    """One example sequence -> a grid of attention heatmaps, one per
    (layer, head), with the correct copy-source positions (earlier
    occurrences of the query symbol's label) marked with an outline.
    Mirrors the query-attends-to-matching-earlier-exemplar pattern in
    Figure 1 of the project brief / Olsson et al. (2022)."""
    rng = torch.Generator().manual_seed(seed)
    seq = generate_sequence(tcfg, rng)
    symbol_tokens, label_tokens, _, _ = sequence_to_tokens(seq, tcfg)
    out = model(symbol_tokens.unsqueeze(0), label_tokens.unsqueeze(0), return_attention=True)

    m = tcfg.context_len
    query_pos = 2 * m
    match_label_positions = [2 * j + 1 for j in range(m) if int(seq.symbols[j]) == int(seq.symbols[-1])]
    labels = _position_labels(tcfg)

    n_layers = len(out.attn_maps)
    n_heads = out.attn_maps[0].shape[1]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers), squeeze=False)
    for l in range(n_layers):
        for h in range(n_heads):
            attn = out.attn_maps[l][0, h].numpy()
            ax = axes[l][h]
            im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"layer {l} head {h}", fontsize=10)
            step = 1 if len(labels) <= 30 else max(1, len(labels) // 15)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels(labels[::step], rotation=90, fontsize=6)
            ax.set_yticks(range(0, len(labels), step))
            ax.set_yticklabels(labels[::step], fontsize=6)
            for pos in match_label_positions:
                ax.add_patch(
                    plt.Rectangle((pos - 0.5, query_pos - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2)
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{title_prefix}Attention maps (red boxes = correct copy-source label positions for the query, row {query_pos})"
    )
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _read_csv(path: str | Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_generation_trends(generations_csv: str | Path, save_path: str | Path | None = None, title: str = ""):
    """Plot test accuracy, induction score, and symbol-distribution KL drift
    across generations from a collapse.py run's generations.csv."""
    rows = _read_csv(generations_csv)
    gens = [int(r["generation"]) for r in rows]
    test_acc = [float(r["test_acc"]) for r in rows]
    ind_score = [float(r["induction_score_mean"]) for r in rows]
    ind_baseline = [float(r["induction_score_baseline"]) for r in rows]
    prev_score = [float(r["prev_token_score_mean"]) for r in rows]
    symbol_entropy = [float(r["symbol_entropy"]) for r in rows]
    kl = [float(r.get("symbol_kl_vs_gen0", 0.0) or 0.0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(gens, test_acc, marker="o", label="test accuracy")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Task accuracy across generations")
    axes[0].grid(alpha=0.3)

    axes[1].plot(gens, ind_score, marker="o", label="induction score")
    axes[1].plot(gens, ind_baseline, linestyle="--", color="gray", label="random baseline")
    axes[1].plot(gens, prev_score, marker="s", label="prev-token score")
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("score")
    axes[1].set_title("Induction circuit strength across generations")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    ax2 = axes[2]
    ax2.plot(gens, symbol_entropy, marker="o", color="tab:green", label="symbol entropy (nats)")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("symbol usage entropy (nats)", color="tab:green")
    ax2.set_title("Synthetic data distributional drift")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(gens, kl, marker="x", color="tab:red", label="KL(gen_n || gen_0)")
    ax3.set_ylabel("symbol KL vs. generation 0", color="tab:red")

    for ax_ in fig.axes:
        ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _aggregate_multiseed(csv_paths: list[str | Path], field: str) -> tuple[list[int], list[float], list[float]]:
    """Read `field` from several seeds' generations.csv files (one row per
    generation each) and return (generations, mean, std) per generation.
    Assumes every seed ran the same number of generations."""
    per_seed = [_read_csv(p) for p in csv_paths]
    n_gens = min(len(rows) for rows in per_seed)
    gens, means, stds = [], [], []
    for g in range(n_gens):
        vals = [float(rows[g].get(field, 0.0) or 0.0) for rows in per_seed]
        gens.append(int(per_seed[0][g]["generation"]))
        means.append(sum(vals) / len(vals))
        var = sum((v - means[-1]) ** 2 for v in vals) / len(vals)
        stds.append(var**0.5)
    return gens, means, stds


def plot_generation_trends_multiseed(
    seed_csvs: list[str | Path], save_path: str | Path | None = None, title: str = ""
):
    """Same three panels as plot_generation_trends, but aggregated (mean +-
    std shaded band) across several seeds' generations.csv files -- the
    multi-seed replication this project uses instead of trusting a single
    run, given base-variant training turned out to be seed-sensitive."""
    fields = {
        "test_acc": "test accuracy",
        "induction_score_mean": "induction score",
        "induction_score_baseline": "random baseline",
        "prev_token_score_mean": "prev-token score",
        "symbol_entropy": "symbol entropy (nats)",
        "symbol_kl_vs_gen0": "symbol KL vs. gen 0",
    }
    agg = {f: _aggregate_multiseed(seed_csvs, f) for f in fields}
    gens = agg["test_acc"][0]
    n_seeds = len(seed_csvs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    gens_, mean_, std_ = agg["test_acc"]
    axes[0].plot(gens_, mean_, marker="o", color="tab:blue")
    axes[0].fill_between(gens_, [m - s for m, s in zip(mean_, std_)], [m + s for m, s in zip(mean_, std_)], alpha=0.2)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Task accuracy across generations (n={n_seeds} seeds, mean±std)")
    axes[0].grid(alpha=0.3)

    for field, color, marker in [
        ("induction_score_mean", "tab:blue", "o"),
        ("prev_token_score_mean", "tab:orange", "s"),
    ]:
        gens_, mean_, std_ = agg[field]
        axes[1].plot(gens_, mean_, marker=marker, color=color, label=fields[field])
        axes[1].fill_between(
            gens_, [m - s for m, s in zip(mean_, std_)], [m + s for m, s in zip(mean_, std_)], alpha=0.2, color=color
        )
    gens_, mean_, _ = agg["induction_score_baseline"]
    axes[1].plot(gens_, mean_, linestyle="--", color="gray", label="random baseline")
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("score")
    axes[1].set_title("Induction circuit strength across generations")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    ax2 = axes[2]
    gens_, mean_, std_ = agg["symbol_entropy"]
    ax2.plot(gens_, mean_, marker="o", color="tab:green")
    ax2.fill_between(gens_, [m - s for m, s in zip(mean_, std_)], [m + s for m, s in zip(mean_, std_)], alpha=0.2, color="tab:green")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("symbol usage entropy (nats)", color="tab:green")
    ax2.set_title("Synthetic data distributional drift")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    gens_, mean_, std_ = agg["symbol_kl_vs_gen0"]
    ax3.plot(gens_, mean_, marker="x", color="tab:red")
    ax3.fill_between(gens_, [m - s for m, s in zip(mean_, std_)], [m + s for m, s in zip(mean_, std_)], alpha=0.2, color="tab:red")
    ax3.set_ylabel("symbol KL vs. generation 0", color="tab:red")

    for ax_ in fig.axes:
        ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ablation_comparison(
    runs: list[tuple[str, list[str | Path]]],
    save_path: str | Path | None = None,
    title: str = "Effect of synthetic-data mixing ratio p on collapse",
):
    """Overlay test accuracy, induction score, and symbol-KL-vs-gen0 across
    generations for several collapse runs (e.g. different p_synthetic), to
    compare partial- vs. full-replacement severity. `runs` is a list of
    (label, [generations_csv_path per seed]) -- each label's seeds are
    aggregated (mean, with a light std band) via _aggregate_multiseed."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, csv_paths in runs:
        for ax, field in zip(axes, ["test_acc", "induction_score_mean", "symbol_kl_vs_gen0"]):
            gens, mean_, std_ = _aggregate_multiseed(csv_paths, field)
            line = ax.plot(gens, mean_, marker="o", label=label)[0]
            if len(csv_paths) > 1:
                ax.fill_between(
                    gens, [m - s for m, s in zip(mean_, std_)], [m + s for m, s in zip(mean_, std_)],
                    alpha=0.15, color=line.get_color(),
                )
    axes[0].set_title("Test accuracy")
    axes[0].set_xlabel("generation")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("Induction score")
    axes[1].set_xlabel("generation")
    axes[2].set_title("Symbol KL(gen_n || gen_0)")
    axes[2].set_xlabel("generation")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax_ in fig.axes:
        ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def embedding_pca_figure(
    checkpoints: list[tuple[str, str]],
    embedding: str = "symbol",
    save_path: str | Path | None = None,
    title: str = "",
):
    """PCA of the symbol (or label) embedding matrix for one or more
    checkpoints, plotted side by side, to see whether embedding geometry
    degrades across generations. `checkpoints` is a list of (label, path)."""
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5 * len(checkpoints), 5), squeeze=False)
    axes = axes[0]
    for ax, (label, path) in zip(axes, checkpoints):
        model, tcfg, mcfg, _ = load_checkpoint(path)
        table = model.symbol_embed.weight.numpy() if embedding == "symbol" else model.label_embed.weight.numpy()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(table)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(table.shape[0]), cmap="viridis", s=25)
        ax.set_title(label)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    fig.colorbar(sc, ax=axes, label=f"{embedding} id", fraction=0.02)
    fig.suptitle(title or f"{embedding.capitalize()} embedding PCA across generations")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
