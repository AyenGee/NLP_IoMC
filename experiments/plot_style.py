"""Shared figure style: Okabe-Ito colour-blind-safe palette, two-column-width fonts, consistent colour per evaluation set
and per condition. Figures are saved at 300 dpi."""

import matplotlib.pyplot as plt

# Okabe-Ito palette (colour-blind safe), one colour per evaluation set.
SET_COLOR = {"train_eval": "#000000", "val": "#0072B2", "test_exemplars": "#D55E00", "test_relabel": "#009E73", "test_class_ex0": "#CC79A7"}
SET_LABEL = {"train_eval": "train", "val": "val (held-out exemplars)", "test_exemplars": "test classes (match)",
             "test_relabel": "held-out label pairs (copy)", "test_class_ex0": "test classes, exemplar 0"}
# eight distinguishable colours for the eight heads of a layer
HEAD_COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def style():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 5.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.4})
