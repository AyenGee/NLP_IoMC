"""Generate every figure used in the extended abstract from results/.

    python experiments/make_figures.py

Requires experiments/run_all.py (local, serial) or the Slurm array job
(slurm/run_experiments.slurm, on mscluster) to have completed first -- both
produce the same seed-suffixed paths this script reads. Writes PNGs to
abstract/figures/. Quantitative trend/ablation figures aggregate across all
seeds in manifest.SEEDS (mean +- std); qualitative figures (attention maps,
embedding PCA) use a single representative seed (0) since they're
illustrative, not a statistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from interp import (
    load_checkpoint,
    plot_attention_maps,
    plot_generation_trends_multiseed,
    plot_ablation_comparison,
    embedding_pca_figure,
)
from manifest import SEEDS

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = Path(__file__).resolve().parent.parent / "abstract" / "figures"
ILLUSTRATIVE_SEED = 0


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # --- Sanity-check attention maps (no collapse), illustrative seed ---
    for name in ["base_no_collapse", "extended_no_collapse"]:
        ckpt = RESULTS / "checkpoints" / f"{name}_seed{ILLUSTRATIVE_SEED}.pt"
        if ckpt.exists():
            model, tcfg, mcfg, summary = load_checkpoint(ckpt)
            plot_attention_maps(
                model, tcfg, seed=42, title_prefix=f"{name} (test_acc={summary.get('test_acc', float('nan')):.3f}): ",
                save_path=FIGDIR / f"attn_{name}.png",
            )
            print(f"wrote attn_{name}.png")

    # --- Generation trends (mean +- std across seeds) for both full collapse pipelines ---
    for name in ["base_collapse", "extended_collapse"]:
        seed_csvs = [RESULTS / "collapse" / f"{name}_seed{s}" / "generations.csv" for s in SEEDS]
        seed_csvs = [p for p in seed_csvs if p.exists()]
        if seed_csvs:
            plot_generation_trends_multiseed(
                seed_csvs, save_path=FIGDIR / f"trends_{name}.png", title=f"{name} (n={len(seed_csvs)} seeds)"
            )
            print(f"wrote trends_{name}.png ({len(seed_csvs)} seeds)")

    # --- Attention maps at gen0 vs. final generation, extended collapse, illustrative seed ---
    ext_dir = RESULTS / "collapse" / f"extended_collapse_seed{ILLUSTRATIVE_SEED}"
    gen0_ckpt = ext_dir / "gen0_model.pt"
    final_gens = sorted(ext_dir.glob("gen*_model.pt"), key=lambda p: int(p.stem.split("_")[0][3:])) if ext_dir.exists() else []
    if gen0_ckpt.exists() and final_gens:
        last_ckpt = final_gens[-1]
        last_gen_idx = last_ckpt.stem.split("_")[0][3:]
        model0, tcfg0, _, s0 = load_checkpoint(gen0_ckpt)
        plot_attention_maps(
            model0, tcfg0, seed=42, title_prefix=f"extended_collapse gen0 (test_acc={s0.get('test_acc', float('nan')):.3f}): ",
            save_path=FIGDIR / "attn_extended_collapse_gen0.png",
        )
        modelN, tcfgN, _, sN = load_checkpoint(last_ckpt)
        plot_attention_maps(
            modelN, tcfgN, seed=42,
            title_prefix=f"extended_collapse gen{last_gen_idx} (test_acc={sN.get('test_acc', float('nan')):.3f}): ",
            save_path=FIGDIR / f"attn_extended_collapse_gen{last_gen_idx}.png",
        )
        print("wrote attn_extended_collapse_gen0.png and final-generation counterpart")

        mid_idx = len(final_gens) // 2
        checkpoints = [
            ("gen 0", str(gen0_ckpt)),
            (f"gen {mid_idx}", str(final_gens[mid_idx])),
            (f"gen {last_gen_idx} (final)", str(last_ckpt)),
        ]
        embedding_pca_figure(
            checkpoints, embedding="symbol",
            save_path=FIGDIR / "embedding_pca_extended_collapse.png",
            title="Symbol embedding PCA across generations (extended, p=1.0)",
        )
        print("wrote embedding_pca_extended_collapse.png")

    # --- Mixing-ratio ablation comparison (mean +- std across seeds per p) ---
    ablation_runs = []
    for label, name in [("p=0.25", "ablation_p025"), ("p=0.5", "ablation_p05"), ("p=1.0", "extended_collapse")]:
        seed_csvs = [RESULTS / "collapse" / f"{name}_seed{s}" / "generations.csv" for s in SEEDS]
        seed_csvs = [p for p in seed_csvs if p.exists()]
        if seed_csvs:
            ablation_runs.append((label, seed_csvs))
    if ablation_runs:
        plot_ablation_comparison(ablation_runs, save_path=FIGDIR / "ablation_mixing_ratio.png")
        print("wrote ablation_mixing_ratio.png")

    print("Done.")


if __name__ == "__main__":
    main()
