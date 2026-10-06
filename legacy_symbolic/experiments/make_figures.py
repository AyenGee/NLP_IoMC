"""Qualitative figures (attention maps, head offset profiles, embedding PCA) from saved checkpoints.

    python experiments/make_figures.py                                  # v2_extended_p1_seed0 if it exists, else extended_collapse_seed0
    python experiments/make_figures.py --run v2_extended_p1_seed3
    python experiments/make_figures.py --run v2_extended_p1_seed0 --profile v2_extended_p1_seed0:0 v2_extended_p1_seed0:9

Needs the model checkpoints (*.pt) in results/, which the Slurm jobs write on
the cluster. The QUANTITATIVE figures and every number in the report come
from experiments/analyze_results.py instead, which needs only the CSV/JSON
logs and draws one line per seed: outcomes here are bimodal (a model either
learns induction or does not), so mean +- std plots would describe no real
model and are not made.

Writes PNGs to abstract/figures/, named after the run. These figures illustrate
single models; they are not statistics. `--profile run:generation ...` chooses
the models shown in the head-profile figure (default: generation 0, the middle
generation and the last generation of --run, plus the last generation of the
no-collapse sanity run if it exists).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interp import load_checkpoint, plot_attention_maps, plot_head_offset_profile, embedding_pca_figure

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = Path(__file__).resolve().parent.parent / "abstract" / "figures"


def _gens(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("gen*_model.pt"), key=lambda p: int(p.stem.split("_")[0][3:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run directory name under results/collapse (default: v2_extended_p1_seed0, else extended_collapse_seed0)")
    ap.add_argument("--profile", nargs="+", default=None, help="run:generation entries for the head-profile figure")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--figdir", default=str(FIGDIR))
    args = ap.parse_args()
    results, figdir = Path(args.results), Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    run = args.run or next((r for r in ["v2_extended_p1_seed0", "extended_collapse_seed0"] if (results / "collapse" / r).exists()), None)
    if run is None:
        print("no run found under", results / "collapse", "- nothing to draw")
        return
    run_dir = results / "collapse" / run
    ckpts = _gens(run_dir)
    if not ckpts:
        print("no checkpoints found under", run_dir, "- nothing to draw")
        return
    last_idx = ckpts[-1].stem.split("_")[0][3:]

    # Attention maps at generation 0 and the final generation
    for label, ckpt in [("0", ckpts[0]), (last_idx, ckpts[-1])]:
        model, tcfg, _, summ = load_checkpoint(ckpt)
        plot_attention_maps(model, tcfg, seed=42, title_prefix=f"{run} gen{label} (test_acc={summ.get('test_acc', float('nan')):.3f}): ",
                            save_path=figdir / f"attn_{run}_gen{label}.png")
    print(f"wrote attention maps for {run} generations 0 and {last_idx}")

    # Head profiles: which offset each head attends to
    if args.profile:
        wanted = [(r, int(g)) for r, g in (item.split(":") for item in args.profile)]
    else:
        wanted = [(run, 0), (run, len(ckpts) // 2), (run, int(last_idx))]
        sanity = results / "checkpoints" / "extended_no_collapse_seed0.pt"
        if sanity.exists() and not run.startswith("extended_no_collapse"):
            wanted.append(("__sanity__", 0))
    entries, tcfg_used = [], None
    for r, g in wanted:
        path = sanity if r == "__sanity__" else results / "collapse" / r / f"gen{g}_model.pt"
        if not path.exists():
            print("skip missing", path)
            continue
        model, tcfg_used, _, summ = load_checkpoint(path)
        name = "no-collapse extended model" if r == "__sanity__" else f"{r} generation {g}"
        entries.append((f"{name} (test acc {summ.get('test_acc', float('nan')):.3f})", model))
    if entries:
        plot_head_offset_profile(entries, tcfg_used, save_path=figdir / f"head_profile_{run}.png")
        print(f"wrote head_profile_{run}.png")

    # Embedding PCA across generations
    mid = len(ckpts) // 2
    embedding_pca_figure(
        [("gen 0", str(ckpts[0])), (f"gen {mid}", str(ckpts[mid])), (f"gen {last_idx} (final)", str(ckpts[-1]))],
        embedding="symbol", save_path=figdir / f"embedding_pca_{run}.png",
        title=f"Symbol embedding PCA across generations ({run})",
    )
    print(f"wrote embedding_pca_{run}.png")


if __name__ == "__main__":
    main()
