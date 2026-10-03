"""Qualitative figures (attention maps, embedding PCA) from saved checkpoints.

    python experiments/make_figures.py

Needs the model checkpoints (*.pt) in results/, which the Slurm jobs write on
the cluster. The QUANTITATIVE figures and every number in the report come
from experiments/analyze_results.py instead, which needs only the CSV/JSON
logs and draws one line per seed: outcomes here are bimodal (a model either
learns induction or does not), so the mean +- std plots this script used to
make described no real model and have been removed.

Writes PNGs to abstract/figures/. These figures illustrate a single
representative seed (0); they are not statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interp import load_checkpoint, plot_attention_maps, embedding_pca_figure

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = Path(__file__).resolve().parent.parent / "abstract" / "figures"
ILLUSTRATIVE_SEED = 0


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # --- Sanity-check attention maps (no collapse) ---
    for name in ["base_no_collapse", "extended_no_collapse"]:
        ckpt = RESULTS / "checkpoints" / f"{name}_seed{ILLUSTRATIVE_SEED}.pt"
        if ckpt.exists():
            model, tcfg, mcfg, summary = load_checkpoint(ckpt)
            plot_attention_maps(
                model, tcfg, seed=42, title_prefix=f"{name} (test_acc={summary.get('test_acc', float('nan')):.3f}): ",
                save_path=FIGDIR / f"attn_{name}.png",
            )
            print(f"wrote attn_{name}.png")

    # --- Attention maps and embedding PCA at gen0 / middle / final generation, extended collapse ---
    ext_dir = RESULTS / "collapse" / f"extended_collapse_seed{ILLUSTRATIVE_SEED}"
    gen0_ckpt = ext_dir / "gen0_model.pt"
    gens = sorted(ext_dir.glob("gen*_model.pt"), key=lambda p: int(p.stem.split("_")[0][3:])) if ext_dir.exists() else []
    if gen0_ckpt.exists() and gens:
        last_ckpt = gens[-1]
        last_idx = last_ckpt.stem.split("_")[0][3:]
        for label, ckpt in [("0", gen0_ckpt), (last_idx, last_ckpt)]:
            model, tcfg, _, summ = load_checkpoint(ckpt)
            plot_attention_maps(
                model, tcfg, seed=42,
                title_prefix=f"extended_collapse gen{label} (test_acc={summ.get('test_acc', float('nan')):.3f}): ",
                save_path=FIGDIR / f"attn_extended_collapse_gen{label}.png",
            )
        print("wrote attention maps for generation 0 and the final generation")

        mid = len(gens) // 2
        embedding_pca_figure(
            [("gen 0", str(gen0_ckpt)), (f"gen {mid}", str(gens[mid])), (f"gen {last_idx} (final)", str(last_ckpt))],
            embedding="symbol", save_path=FIGDIR / "embedding_pca_extended_collapse.png",
            title="Symbol embedding PCA across generations (extended, p=1.0)",
        )
        print("wrote embedding_pca_extended_collapse.png")
    else:
        print("no checkpoints found under", ext_dir, "- nothing to draw")


if __name__ == "__main__":
    main()
