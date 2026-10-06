"""Analysis of the Tier 1 collapse runs on the exact task (plan sections 5.3 and 9).

    python experiments/analyze_collapse.py            # reads results/exact, writes analysis_summary_exact.json and figures/

Every number is computed from the logs by this script. Inputs: results/exact/collapse/<condition>_s<seed>/generations.csv
(always) and results/exact/analysis/{circuits,acdc}_<run>.csv (written on the cluster by analyze_collapse_circuits.py; optional).

Definitions (documented choices)
    degraded   a generation's model has train_eval accuracy < DEGRADED_ACC (0.9)  (the plateau, guessing between the two labels, is 0.5)
    failed     train_eval accuracy < FAILED_ACC (0.6): back at the plateau
    match / copy are read from test_class_ex0 (new classes, identical vector) and test_relabel (held-out label pairs): the paper's two halves
    of the circuit. "Degrades first": the first generation whose accuracy falls MORE than DROP (0.05) below that run's generation-0 accuracy.
Statistics: per-seed lines everywhere; exact permutation tests over seeds (the unit is the seed, i.e. one chain); Wilson intervals for
failure counts. Trainings within a chain are not independent, so the intervals on counts are descriptive.
"""

from __future__ import annotations

import csv
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_style import style  # noqa: E402

DEGRADED_ACC = 0.9
FAILED_ACC = 0.6
DROP = 0.05

CONDITIONS = ["exact_base_p0", "exact_base_p1", "exact_extended_p0", "exact_extended_p025", "exact_extended_p05", "exact_extended_p1"]
LABEL = {"exact_base_p0": "base, p=0 (control)", "exact_base_p1": "base, p=1", "exact_extended_p0": "extended, p=0 (control)",
         "exact_extended_p025": "extended, p=0.25", "exact_extended_p05": "extended, p=0.5", "exact_extended_p1": "extended, p=1"}
# one colour per condition, Okabe-Ito; controls grey / black, synthetic share increasing in saturation
COLOR = {"exact_base_p0": "#999999", "exact_base_p1": "#0072B2", "exact_extended_p0": "#555555",
         "exact_extended_p025": "#009E73", "exact_extended_p05": "#E69F00", "exact_extended_p1": "#D55E00"}
CONTROL_OF = {"exact_base_p1": "exact_base_p0", "exact_extended_p1": "exact_extended_p0", "exact_extended_p05": "exact_extended_p0", "exact_extended_p025": "exact_extended_p0"}


def read_csv(path: Path) -> list[dict]:
    return [{k: (float(v) if v not in ("", "nan") else float("nan")) if k not in ("run", "heads_L1", "heads_L2", "kcomp_edges", "edges", "corruption") else v
             for k, v in r.items()} for r in csv.DictReader(open(path))]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def perm_test(a, b):
    """Exact two-sided permutation p for a difference in means of two small samples (all splits)."""
    a, b = list(a), list(b)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled, n = a + b, len(a)
    obs = abs(np.mean(a) - np.mean(b))
    hits = total = 0
    for idx in combinations(range(len(pooled)), n):
        s = set(idx)
        ga = [pooled[i] for i in idx]
        gb = [pooled[i] for i in range(len(pooled)) if i not in s]
        total += 1
        hits += abs(np.mean(ga) - np.mean(gb)) >= obs - 1e-12
    return round(hits / total, 4)


def load_runs(results: Path) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for p in sorted((results / "collapse").glob("*_s[0-9]*")):
        if not (p / "generations.csv").exists():
            continue
        cond, seed = p.name.rsplit("_s", 1)
        out.setdefault(cond, {})[int(seed)] = read_csv(p / "generations.csv")
    return out


def load_circuits(results: Path) -> dict[tuple[str, int], list[dict]]:
    out = {}
    for p in sorted((results / "analysis").glob("circuits_*.csv")) if (results / "analysis").is_dir() else []:
        run = p.stem[len("circuits_"):]
        cond, seed = run.rsplit("_s", 1)
        out[(cond, int(seed))] = read_csv(p)
    return out


def load_acdc(results: Path) -> dict[tuple[str, int], list[dict]]:
    out = {}
    for p in sorted((results / "analysis").glob("acdc_*.csv")) if (results / "analysis").is_dir() else []:
        run = p.stem[len("acdc_"):]
        cond, seed = run.rsplit("_s", 1)
        out[(cond, int(seed))] = read_csv(p)
    return out


def series(runs: dict[int, list[dict]], key: str) -> np.ndarray:
    seeds = sorted(runs)
    return np.array([[r.get(key, float("nan")) for r in runs[s]] for s in seeds])


def first_drop(acc: np.ndarray) -> int | None:
    """First generation (>= 1) whose accuracy is more than DROP below generation 0's, else None."""
    base = acc[0]
    return next((g for g in range(1, len(acc)) if acc[g] < base - DROP), None)


def collapse_summary(results: Path) -> dict:
    runs = load_runs(results)
    circ = load_circuits(results)
    summary: dict = {"definitions": {"degraded_acc": DEGRADED_ACC, "failed_acc": FAILED_ACC, "drop": DROP}, "conditions": {}}
    for cond in CONDITIONS:
        if cond not in runs:
            continue
        r = runs[cond]
        seeds = sorted(r)
        entry = {"n_seeds": len(seeds), "seeds": seeds, "n_generations": len(r[seeds[0]])}
        train, val, copy = series(r, "train_eval/acc"), series(r, "val/acc"), series(r, "test_relabel/acc")
        match, ex = series(r, "test_class_ex0/acc"), series(r, "test_exemplars/acc")
        entry["acc_by_seed"] = {s: {"train": train[i].round(4).tolist(), "val": val[i].round(4).tolist(), "copy": copy[i].round(4).tolist(),
                                    "match": match[i].round(4).tolist()} for i, s in enumerate(seeds)}
        later = train[:, 1:]
        k_deg, k_fail, n = int((later < DEGRADED_ACC).sum()), int((later < FAILED_ACC).sum()), int(later.size)
        entry.update({"degraded_trainings": k_deg, "failed_trainings": k_fail, "trainings_gen_ge1": n,
                      "degraded_ci95": wilson(k_deg, n), "failed_ci95": wilson(k_fail, n),
                      "seeds_with_a_degraded_generation": [s for i, s in enumerate(seeds) if (later[i] < DEGRADED_ACC).any()],
                      "gen0_train_acc_by_seed": {s: round(float(train[i, 0]), 4) for i, s in enumerate(seeds)}})
        for name, m in [("train", train), ("val", val), ("copy_test_relabel", copy), ("match_test_class_ex0", match), ("match_test_exemplars", ex)]:
            entry[f"{name}_acc_median_by_gen"] = np.nanmedian(m, axis=0).round(4).tolist()
            entry[f"{name}_acc_final_by_seed"] = {s: round(float(m[i, -1]), 4) for i, s in enumerate(seeds)}
        # which half degrades first
        fm, fc = [first_drop(match[i]) for i in range(len(seeds))], [first_drop(copy[i]) for i in range(len(seeds))]
        entry["first_gen_match_drops"], entry["first_gen_copy_drops"] = fm, fc
        entry["match_before_copy"] = int(sum(a is not None and (b is None or a < b) for a, b in zip(fm, fc)))
        entry["copy_before_match"] = int(sum(b is not None and (a is None or b < a) for a, b in zip(fm, fc)))
        entry["same_generation"] = int(sum(a is not None and a == b for a, b in zip(fm, fc)))
        entry["neither_drops"] = int(sum(a is None and b is None for a, b in zip(fm, fc)))
        # heads and phase-change speed from the training logs
        ind = series(r, "ind_max")
        entry["ind_max_median_by_gen"] = np.nanmedian(ind, axis=0).round(3).tolist()
        entry["match_max_median_by_gen"] = np.nanmedian(series(r, "match_max"), axis=0).round(3).tolist()
        entry["prev_max_median_by_gen"] = np.nanmedian(series(r, "prev_max"), axis=0).round(3).tolist()
        # distribution drift of the training pools (extended)
        if "pool_p_s2_equals_q_class" in r[seeds[0]][0]:
            for key in ["pool_p_s2_equals_q_class", "pool_p_s2_outside_context", "pool_s2_entropy", "pool_s2_tail_mass_k10", "pool_s2_kl_vs_gen0",
                        "pool_lq_correct_synthetic", "pool_ls2_correct_given_in_context", "pool_s2_n_classes_used"]:
                entry[f"{key}_median_by_gen"] = np.nanmedian(series(r, key), axis=0).round(4).tolist()
            entry["s2_noise_floor_kl"] = float(r[seeds[0]][0]["noise_floor_s2_kl"])
        elif "pool_lq_correct_synthetic" in r[seeds[0]][0]:
            entry["pool_lq_correct_synthetic_median_by_gen"] = np.nanmedian(series(r, "pool_lq_correct_synthetic"), axis=0).round(4).tolist()
        # head redundancy from the circuit analysis, if it was run
        cs = {s: circ[(cond, s)] for s in seeds if (cond, s) in circ}
        if cs:
            nh = np.array([[row["n_induction_heads"] for row in cs[s]] for s in sorted(cs)])
            ko = np.array([[row["acc_keep_only_strongest"] for row in cs[s]] for s in sorted(cs)])
            sd = np.array([[row["max_single_head_drop"] for row in cs[s]] for s in sorted(cs)])
            entry["n_induction_heads_median_by_gen"] = np.nanmedian(nh, axis=0).tolist()
            entry["acc_keep_only_strongest_median_by_gen"] = np.nanmedian(ko, axis=0).round(4).tolist()
            entry["max_single_head_drop_median_by_gen"] = np.nanmedian(sd, axis=0).round(4).tolist()
        summary["conditions"][cond] = entry
    # contrasts with the real-data control of the same variant
    summary["contrasts_vs_control"] = {}
    for cond, ctl in CONTROL_OF.items():
        if cond in runs and ctl in runs:
            out = {}
            for name, key in [("final_train_acc", "train_eval/acc"), ("final_val_acc", "val/acc"), ("final_copy_acc", "test_relabel/acc"), ("final_match_acc", "test_class_ex0/acc")]:
                a, b = series(runs[cond], key)[:, -1], series(runs[ctl], key)[:, -1]
                out[name] = {"condition_mean": round(float(a.mean()), 4), "control_mean": round(float(b.mean()), 4), "permutation_p": perm_test(a, b)}
            summary["contrasts_vs_control"][cond] = out
    return summary


# ------------------------------------------------------------------------------------------------ figures
def _lines(ax, arr, color, label=None, median=True):
    for row in arr:
        ax.plot(range(len(row)), row, color=color, lw=0.7, alpha=0.55)
    if median:
        ax.plot(range(arr.shape[1]), np.nanmedian(arr, axis=0), color=color, lw=1.6, label=label)


def _grid(results: Path, name: str, key: str, ylabel: str, ylim=(0, 1.04), hline=None, runs=None):
    runs = runs or load_runs(results)
    present = [c for c in CONDITIONS if c in runs]
    if not present:
        return
    style()
    base_c = [c for c in present if "base" in c]
    ext_c = [c for c in present if "extended" in c]
    ncol = max(len(base_c), len(ext_c), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(7.0, 3.6), squeeze=False, sharey=True)
    for row_i, conds in enumerate([base_c, ext_c]):
        for j in range(ncol):
            ax = axes[row_i, j]
            if j >= len(conds):
                ax.axis("off")
                continue
            c = conds[j]
            _lines(ax, series(runs[c], key), COLOR[c])
            if hline is not None:
                ax.axhline(hline, color="gray", ls=":", lw=0.7)
            ax.set_title(f"{LABEL[c]} (n={len(runs[c])})", fontsize=6.5)
            ax.set_ylim(*ylim)
            ax.set_xlabel("generation")
            if j == 0:
                ax.set_ylabel(ylabel)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    fig.savefig(results / "figures" / name, dpi=300)
    plt.close(fig)


def fig_match_copy(results: Path, runs):
    present = [c for c in CONDITIONS if c in runs]
    if not present:
        return
    style()
    fig, axes = plt.subplots(1, len(present), figsize=(7.0, 1.9), squeeze=False, sharey=True)
    for ax, c in zip(axes[0], present):
        _lines(ax, series(runs[c], "test_class_ex0/acc"), "#D55E00", "match (new classes)")
        _lines(ax, series(runs[c], "test_relabel/acc"), "#009E73", "copy (held-out pairs)")
        ax.set_title(LABEL[c], fontsize=6)
        ax.set_xlabel("generation"); ax.set_ylim(0.4, 1.02)
    axes[0, 0].set_ylabel("accuracy"); axes[0, 0].legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.4, w_pad=0.4)
    fig.savefig(results / "figures" / "fig_match_vs_copy.png", dpi=300)
    plt.close(fig)


def fig_head_scores(results: Path, runs):
    present = [c for c in CONDITIONS if c in runs]
    if not present:
        return
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharex=True)
    for ax, key, title in zip(axes, ["ind_max", "prev_max", "match_max"], ["best induction strength (layer 2)", "best previous-token score (layer 1)", "best match score (layer 2)"]):
        for c in present:
            ax.plot(range(len(runs[c][sorted(runs[c])[0]])), np.nanmedian(series(runs[c], key), axis=0), color=COLOR[c], lw=1.2, label=LABEL[c])
        ax.set_title(title, fontsize=6.5); ax.set_xlabel("generation")
    axes[0].legend(frameon=False, fontsize=4.8)
    fig.tight_layout(pad=0.4, w_pad=0.5)
    fig.savefig(results / "figures" / "fig_head_scores.png", dpi=300)
    plt.close(fig)


def fig_ablation_heatmaps(results: Path, circ):
    keys = sorted(circ)
    if not keys:
        return
    style()
    chosen = [k for k in keys if k[1] == 0] or keys[:1]
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.2 * len(chosen) + 0.6, 3.6), squeeze=False)
    for j, k in enumerate(chosen):
        rows = circ[k]
        H = sum(1 for c in rows[0] if c.startswith("zero_L2H"))
        for i, (prefix, title, cmap, vmax) in enumerate([("zero_L2H{}_drop", "accuracy drop, zero one L2 head", "viridis", 0.3), ("keep_only_L2H{}_acc", "accuracy, keep only one L2 head", "viridis", 1.0)]):
            m = np.array([[row[prefix.format(h)] for row in rows] for h in range(H)])
            im = axes[i, j].imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
            axes[i, j].set_title(f"{LABEL.get(k[0], k[0])} s{k[1]}\n{title}", fontsize=5.5)
            axes[i, j].set_yticks(range(H)); axes[i, j].set_yticklabels([f"L2H{h}" for h in range(H)], fontsize=5); axes[i, j].set_xlabel("generation")
            axes[i, j].grid(False)
        fig.colorbar(im, ax=axes[:, j].tolist(), fraction=0.04, pad=0.02).ax.tick_params(labelsize=5)
    fig.savefig(results / "figures" / "fig_ablation_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_s2_drift(results: Path, runs):
    present = [c for c in CONDITIONS if c in runs and "extended" in c and "pool_s2_kl_vs_gen0" in runs[c][sorted(runs[c])[0]][0]]
    if not present:
        return
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    for c in present:
        for ax, key in zip(axes, ["pool_s2_kl_vs_gen0", "pool_p_s2_equals_q_class", "pool_s2_tail_mass_k10"]):
            arr = series(runs[c], key)
            ax.plot(range(arr.shape[1]), np.nanmedian(arr, axis=0), color=COLOR[c], lw=1.2, label=LABEL[c])
    floor = runs[present[0]][sorted(runs[present[0]])[0]][0]["noise_floor_s2_kl"]
    axes[0].axhline(floor, color="gray", ls="--", lw=0.8); axes[0].set_yscale("log"); axes[0].set_title("S2 class histogram KL to generation 0 (dashed: noise floor)", fontsize=5.8)
    axes[1].axhline(0.5, color="gray", ls=":", lw=0.8); axes[1].set_title("P(S2 = Q's class) (real: 0.5)", fontsize=6.5)
    axes[2].axhline(0.2, color="gray", ls=":", lw=0.8); axes[2].set_title("S2 mass on the 10 rarest classes (uniform: 0.2)", fontsize=6.2)
    for ax in axes:
        ax.set_xlabel("generation")
    axes[1].legend(frameon=False, fontsize=4.8)
    fig.tight_layout(pad=0.4, w_pad=0.5)
    fig.savefig(results / "figures" / "fig_s2_drift.png", dpi=300)
    plt.close(fig)


def fig_acdc(results: Path, acdc, tau: float = 0.01):
    if not acdc:
        return
    style()
    conds = [c for c in CONDITIONS if any(k[0] == c for k in acdc)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for c in conds:
        for ax, corr in zip(axes, ["resample", "labels"]):
            gens = None
            per = []
            for (cc, s), rows in acdc.items():
                if cc != c:
                    continue
                sel = [r for r in rows if r["corruption"] == corr and abs(r["tau"] - tau) < 1e-9]
                if sel:
                    per.append([r["n_edges"] for r in sorted(sel, key=lambda r: r["generation"])])
            if per:
                n = min(len(p) for p in per)
                arr = np.array([p[:n] for p in per])
                ax.plot(range(n), np.median(arr, axis=0), color=COLOR[c], lw=1.2, label=LABEL[c])
    axes[0].set_title(f"ACDC circuit size (edges of 257), whole-task corruption, tau={tau}", fontsize=6)
    axes[1].set_title(f"ACDC circuit size, label corruption, tau={tau}", fontsize=6)
    for ax in axes:
        ax.set_xlabel("generation")
    axes[0].legend(frameon=False, fontsize=4.8)
    fig.tight_layout(pad=0.4, w_pad=0.5)
    fig.savefig(results / "figures" / "fig_acdc.png", dpi=300)
    plt.close(fig)


def fig_embeddings(results: Path, circ):
    if not circ:
        return
    style()
    conds = [c for c in CONDITIONS if any(k[0] == c for k in circ)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    for c in conds:
        for ax, key in zip(axes, ["emb_cos_train", "emb_cos_test", "emb_effective_rank"]):
            arr = [[r[key] for r in rows] for (cc, s), rows in sorted(circ.items()) if cc == c]
            n = min(len(a) for a in arr)
            ax.plot(range(n), np.nanmedian(np.array([a[:n] for a in arr]), axis=0), color=COLOR[c], lw=1.2, label=LABEL[c])
    for ax, t in zip(axes, ["cosine to generation 0, train classes", "cosine to generation 0, test classes", "effective rank of the class embeddings"]):
        ax.set_title(t, fontsize=6.3); ax.set_xlabel("generation")
    axes[0].legend(frameon=False, fontsize=4.8)
    fig.tight_layout(pad=0.4, w_pad=0.5)
    fig.savefig(results / "figures" / "fig_embeddings.png", dpi=300)
    plt.close(fig)


def make_figures(results: Path):
    (results / "figures").mkdir(parents=True, exist_ok=True)
    runs, circ, acdc = load_runs(results), load_circuits(results), load_acdc(results)
    _grid(results, "fig_accuracy_by_generation.png", "train_eval/acc", "accuracy on train sequences", hline=0.5, runs=runs)
    fig_head_scores(results, runs)
    fig_match_copy(results, runs)
    fig_ablation_heatmaps(results, circ)
    fig_s2_drift(results, runs)
    fig_acdc(results, acdc)
    fig_embeddings(results, circ)
