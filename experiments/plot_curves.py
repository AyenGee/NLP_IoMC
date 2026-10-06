"""Training-curve figure for one run on the exact task (the reproduction of Singh et al. 2024, Fig. 3).

    python experiments/plot_curves.py results/exact/smoke/logs/smoke_base.csv --out results/exact/smoke/smoke_base.png

Panels: (a) loss on the evaluation sets with the ln 2 line (the plateau at which the model answers with one of the
two labels it can see); (b) accuracy on the same sets; (c) induction strength of each layer-2 head; (d) previous-token
score of the layer-1 heads and match score of the layer-2 heads; for the extended model, (e) the accuracy of each of
its three targets against the Bayes-optimal value (S2 can reach only 0.5: it names one of two classes).
x axis: sequences seen (the paper's unit); 31,250 steps at batch 32 = 1e6 sequences.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_style import HEAD_COLORS, SET_COLOR, SET_LABEL, style  # noqa: E402


def read(path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in rows[0]}


def plot_run(csv_path, out_png, title: str = ""):
    d = read(csv_path)
    x = d["sequences"] / 1e3
    n_heads = sum(1 for k in d if k.startswith("ind_L2H"))
    extended = "train_eval/acc_s2" in d
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
    ax = axes[0, 0]
    for s in SET_COLOR:
        ax.plot(x, d[f"{s}/loss"], color=SET_COLOR[s], lw=0.9, label=SET_LABEL[s], ls="--" if s == "test_class_ex0" else "-")
    ax.axhline(math.log(2), color="gray", ls=":", lw=0.9); ax.text(x[-1], math.log(2) + 0.03, "ln 2", ha="right", fontsize=6, color="gray")
    ax.set_ylabel("loss on the query label"); ax.set_title("(a) loss"); ax.legend(frameon=False, loc="upper right")
    ax = axes[0, 1]
    for s in SET_COLOR:
        ax.plot(x, d[f"{s}/acc"], color=SET_COLOR[s], lw=0.9, ls="--" if s == "test_class_ex0" else "-")
    ax.axhline(0.5, color="gray", ls=":", lw=0.9); ax.set_ylim(0, 1.03); ax.set_ylabel("accuracy"); ax.set_title("(b) accuracy (dotted: 0.5)")
    ax = axes[0, 2]
    for h in range(n_heads):
        ax.plot(x, d[f"ind_L2H{h}"], color=HEAD_COLORS[h % 8], lw=0.9, label=f"L2H{h}")
    ax.set_ylabel("induction strength"); ax.set_title("(c) layer-2 heads"); ax.legend(frameon=False, ncol=2, loc="upper left")
    ax = axes[1, 0]
    for h in range(n_heads):
        ax.plot(x, d[f"prev_L1H{h}"], color=HEAD_COLORS[h % 8], lw=0.9)
    ax.set_ylabel("previous-token score"); ax.set_title("(d) layer-1 heads"); ax.set_ylim(0, 1.03)
    ax = axes[1, 1]
    for h in range(n_heads):
        ax.plot(x, d[f"match_L2H{h}"], color=HEAD_COLORS[h % 8], lw=0.9)
    ax.set_ylabel("match score"); ax.set_title("(e) layer-2 match score"); ax.set_ylim(0, 1.03)
    ax = axes[1, 2]
    if extended:
        for key, lab, col, bayes in [("acc", "lq (query label)", "#000000", 1.0), ("acc_s2", "S2 (next exemplar)", "#D55E00", 0.5), ("acc_ls2", "l_S2 (its label)", "#0072B2", 1.0)]:
            ax.plot(x, d[f"train_eval/{key}"], color=col, lw=0.9, label=lab)
            ax.axhline(bayes, color=col, ls=":", lw=0.6)
        ax.set_ylim(0, 1.03); ax.set_title("(f) extended targets (dotted: Bayes)"); ax.legend(frameon=False, loc="lower right")
    else:
        ax.plot(x, d["train_loss_running"], color="#000000", lw=0.9, label="train (running)")
        ax.plot(x, d["val/loss"], color="#0072B2", lw=0.9, label="val"); ax.set_yscale("log"); ax.legend(frameon=False)
        ax.set_title("(f) train vs. validation loss (log)")
    for a in axes.ravel():
        a.set_xlabel("sequences seen (thousands)")
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout(pad=0.5, w_pad=0.7, h_pad=0.8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    plot_run(args.csv, args.out, args.title)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
