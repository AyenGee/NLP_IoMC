"""Summarise the 6 conditions x 3 seeds of results and draw the report figures.

    python experiments/analyze_results.py

Reads results/ (written by the cluster jobs / run_all.py), writes
  results/analysis_summary.json     every number quoted in the report
  abstract/figures/fig_main.png     per-seed accuracy / induction score / train-vs-val dynamics
  abstract/figures/fig_failure_map.png   accuracy of every (condition, seed, generation) + KL vs. noise floor

Why per-seed views instead of mean +- std: outcomes are bimodal (a generation's
model either learns induction, ~100% accuracy, or fails to, ~25%), so a mean
describes no model that exists and a std band spills outside [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAIL_THRESHOLD = 0.5          # test accuracy below this = the generation's model failed to learn induction
# Conditions in plotting order; only those whose runs exist in results/ are analysed.
ALL_CONDS = ["extended_collapse", "ablation_p05", "ablation_p025", "base_collapse",
             "extended_collapse_nodup", "control_real_resampled", "control_real_fresh"]
LABEL = {
    "extended_collapse": "extended, p=1.0",
    "ablation_p05": "extended, p=0.5",
    "ablation_p025": "extended, p=0.25",
    "base_collapse": "base, p=1.0",
    "extended_collapse_nodup": "extended, p=1.0, no dups",
    "control_real_resampled": "control: real, resampled",
    "control_real_fresh": "control: real, fresh",
}
COLOR = {
    "extended_collapse": "#D55E00",
    "ablation_p05": "#E69F00",
    "ablation_p025": "#009E73",
    "base_collapse": "#0072B2",
    "extended_collapse_nodup": "#CC79A7",
    "control_real_resampled": "#56B4E9",
    "control_real_fresh": "#999999",
}
# Optional per-generation columns, written only by runs made after the diagnostics were added.
OPTIONAL_FIELDS = ["induction_score_max", "prev_token_score_max", "pool_distinct_frac", "pool_frac_fully_valid",
                   "pool_label_consistency_rate", "pool_frac_label_consistent", "pool_frac_repeats_ok",
                   "pool_frac_distinct_ok", "pool_frac_label_injective", "pool_frac_query_label_ok"]


def seeds_for(results: Path, cond: str) -> list[int]:
    return sorted(int(p.name.rsplit("_seed", 1)[1]) for p in (results / "collapse").glob(f"{cond}_seed*")
                  if (p / "generations.csv").exists())


def present_conds(results: Path) -> list[str]:
    return [c for c in ALL_CONDS if seeds_for(results, c)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    v = row.get(key, "")
    return float(v) if v != "" else float("nan")


def load_collapse(results: Path, cond: str) -> dict[int, list[dict]]:
    return {s: read_csv(results / "collapse" / f"{cond}_seed{s}" / "generations.csv") for s in seeds_for(results, cond)}


def load_epochs(results: Path, cond: str, seed: int, gen: int) -> list[dict]:
    return read_csv(results / "collapse" / f"{cond}_seed{seed}" / f"gen{gen}_train.csv")


def sanity_runs(results: Path) -> dict:
    out = {}
    for name in ["base_no_collapse", "extended_no_collapse"]:
        per_seed = {}
        found = sorted(int(p.name.split("_seed")[1].split(".")[0]) for p in (results / "checkpoints").glob(f"{name}_seed*.summary.json"))
        for s in found:
            summ = json.load(open(results / "checkpoints" / f"{name}_seed{s}.summary.json"))
            rows = read_csv(results / "logs" / f"{name}_seed{s}.csv")
            interp = [r for r in rows if r.get("induction_score_mean")]
            first90 = next((int(r["epoch"]) for r in rows if r.get("val_acc") and float(r["val_acc"]) > 0.9), None)
            per_seed[s] = {
                "test_acc": summ["test_acc"], "test_loss": summ["test_loss"], "epochs_run": summ["n_epochs_run"],
                "first_epoch_val_acc_gt_0.9": first90,
                "final_induction_score": fnum(interp[-1], "induction_score_mean"),
                "final_prev_token_score": fnum(interp[-1], "prev_token_score_mean"),
                "induction_baseline": fnum(interp[-1], "induction_score_baseline"),
                "test_aux_label_acc": summ.get("test_aux_label_acc"), "test_aux_symbol_acc": summ.get("test_aux_symbol_acc"),
            }
        if per_seed:
            out[name] = per_seed
    return out


def collapse_summary(results: Path) -> dict:
    out = {}
    for cond in present_conds(results):
        runs = load_collapse(results, cond)
        seeds = sorted(runs)
        acc = np.array([[fnum(r, "test_acc") for r in runs[s]] for s in seeds])          # (seed, gen)
        failed = acc < FAIL_THRESHOLD
        n_gens = acc.shape[1]
        entry = {
            "n_generations": n_gens,
            "seeds": seeds,
            "test_acc_by_seed": {s: acc[i].round(4).tolist() for i, s in enumerate(seeds)},
            "failed_trainings_gen_ge1": int(failed[:, 1:].sum()),
            "n_trainings_gen_ge1": int(failed[:, 1:].size),
            "seeds_with_any_failure": [s for i, s in enumerate(seeds) if failed[i, 1:].any()],
            "first_failure_gen_by_seed": {s: (int(np.argmax(failed[i])) if failed[i].any() else None) for i, s in enumerate(seeds)},
            "final_gen_acc_by_seed": {s: round(float(acc[i, -1]), 4) for i, s in enumerate(seeds)},
        }
        # recovery: a failed generation followed (later) by a non-failed one
        entry["recovered_after_failure_by_seed"] = {
            s: bool(any(failed[i, g] and (~failed[i, g + 1:]).any() for g in range(1, n_gens - 1))) for i, s in enumerate(seeds)
        }
        for field in ["test_loss", "induction_score_mean", "prev_token_score_mean", "attn_entropy_mean", "symbol_entropy",
                      "symbol_kl_vs_gen0", "label_kl_vs_gen0", "induction_score_baseline"]:
            mat = np.array([[fnum(r, field) for r in runs[s]] for s in seeds])
            entry[f"{field}_gen0_mean"] = round(float(mat[:, 0].mean()), 5)
            entry[f"{field}_final_mean"] = round(float(mat[:, -1].mean()), 5)
            entry[f"{field}_by_gen_mean"] = mat.mean(axis=0).round(5).tolist()
        # Columns only newer runs have: report them, and for the pool-structure ones also split by failed/healthy.
        for field in OPTIONAL_FIELDS:
            if all(field in runs[s][0] and runs[s][0][field] != "" for s in seeds):
                mat = np.array([[fnum(r, field) for r in runs[s]] for s in seeds])
                entry[f"{field}_by_gen_mean"] = mat.mean(axis=0).round(5).tolist()
                if field.startswith("pool_"):
                    entry[f"{field}_mean_at_failed_vs_healthy"] = {
                        "failed": round(float(mat[:, 1:][failed[:, 1:]].mean()), 5) if failed[:, 1:].any() else None,
                        "healthy": round(float(mat[:, 1:][~failed[:, 1:]].mean()), 5) if (~failed[:, 1:]).any() else None,
                    }
        # training dynamics of failed vs. healthy generation-trainings (gen >= 1)
        diag = {"failed": [], "ok": []}
        for s in seeds:
            for g in range(1, n_gens):
                ep = load_epochs(results, cond, s, g)
                val_acc = [fnum(r, "val_acc") for r in ep if r.get("val_acc", "") != ""]
                rec = {"train_acc_final": fnum(ep[-1], "train_acc"), "max_val_acc": max(val_acc), "epochs": len(ep),
                       "first_epoch_val_gt_0.9": next((int(r["epoch"]) for r in ep if r.get("val_acc") and float(r["val_acc"]) > 0.9), None)}
                diag["failed" if failed[seeds.index(s), g] else "ok"].append(rec)
        entry["diagnostics"] = {
            k: ({"n": len(v), "mean_train_acc_final": round(float(np.mean([r["train_acc_final"] for r in v])), 4),
                 "mean_max_val_acc": round(float(np.mean([r["max_val_acc"] for r in v])), 4),
                 "mean_epochs": round(float(np.mean([r["epochs"] for r in v])), 1)} if v else {"n": 0})
            for k, v in diag.items()
        }
        t90_ok = [r["first_epoch_val_gt_0.9"] for r in diag["ok"] if r["first_epoch_val_gt_0.9"] is not None]
        t90_g0 = []
        for s in seeds:
            ep0 = load_epochs(results, cond, s, 0)
            t = next((int(r["epoch"]) for r in ep0 if r.get("val_acc") and float(r["val_acc"]) > 0.9), None)
            if t is not None:
                t90_g0.append(t)
        entry["median_epoch_to_val90_gen0"] = float(np.median(t90_g0)) if t90_g0 else None
        entry["median_epoch_to_val90_healthy_gen_ge1"] = float(np.median(t90_ok)) if t90_ok else None
        out[cond] = entry
    return out


def reference_values(results: Path) -> dict:
    """Noise floor of the symbol KL and Bayes-optimal auxiliary accuracies, from the true generator."""
    import torch
    from data import TaskConfig, SymbolicICLDataset, dataset_to_items, generate_sequence
    from metrics import distribution_stats

    cfg = json.load(open(results / "collapse" / "extended_collapse_seed0" / "run_config.json"))
    tcfg = TaskConfig(**cfg["task"])
    n = cfg["collapse"]["pool_size"]
    seed0 = cfg["train"]["data_seed"]
    ref = dataset_to_items(SymbolicICLDataset(tcfg, n, seed=seed0))
    _, sc0, lc0 = distribution_stats(ref, tcfg.K, tcfg.L)
    kls, kll = [], []
    for g in range(1, 6):
        other = dataset_to_items(SymbolicICLDataset(tcfg, n, seed=seed0 + g * 7919 + 2))
        st, _, _ = distribution_stats(other, tcfg.K, tcfg.L, ref_symbol_counts=sc0, ref_label_counts=lc0)
        kls.append(st["symbol_kl_vs_gen0"]); kll.append(st["label_kl_vs_gen0"])
    # Bayes-optimal accuracy of the auxiliary heads under the true generating process
    rng = torch.Generator().manual_seed(123)
    lab, sym = [], []
    for _ in range(20000):
        syms = generate_sequence(tcfg, rng).symbols.tolist()
        m = len(syms) - 1
        seen, used, a = set(), 0, []
        for j in range(m):
            if syms[j] in seen:
                a.append(1.0)
            else:
                a.append(1.0 / (tcfg.L - used)); seen.add(syms[j]); used += 1
        lab.append(np.mean(a))
        counts, a = {}, []
        for t in range(m):
            counts[syms[t]] = counts.get(syms[t], 0) + 1
            rem_total = tcfg.n_classes * tcfg.burstiness - (t + 1)
            if t + 1 < m:
                rem = [tcfg.burstiness - c for c in counts.values() if tcfg.burstiness - c > 0]
                a.append((max(rem) / rem_total) if rem and rem_total > 0 else 0.0)
            else:
                a.append(1.0 / tcfg.n_classes)
        sym.append(np.mean(a))
    return {
        "symbol_kl_noise_floor_mean": float(np.mean(kls)), "symbol_kl_noise_floor_std": float(np.std(kls)),
        "label_kl_noise_floor_mean": float(np.mean(kll)),
        "max_symbol_entropy_ln_K": math.log(tcfg.K),
        "bayes_optimal_aux_label_acc": float(np.mean(lab)), "bayes_optimal_aux_symbol_acc": float(np.mean(sym)),
        "chance_query_label_1_over_L": 1.0 / tcfg.L, "guess_among_shown_labels_1_over_n_classes": 1.0 / tcfg.n_classes,
        "ln_L": math.log(tcfg.L),
        "pool_size": n,
    }


def style():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4})


def fig_main(results: Path, figdir: Path):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95))
    ext, base = load_collapse(results, "extended_collapse"), load_collapse(results, "base_collapse")

    ax = axes[0]
    for s in base:
        ax.plot(range(len(base[s])), [fnum(r, "test_acc") for r in base[s]], color=COLOR["base_collapse"], lw=1.0, marker="o", ms=2.2, alpha=0.85)
    for s in ext:
        ax.plot(range(len(ext[s])), [fnum(r, "test_acc") for r in ext[s]], color=COLOR["extended_collapse"], lw=1.0, marker="o", ms=2.2, alpha=0.85)
    ax.axhline(0.25, color="gray", ls=":", lw=0.8)
    ax.text(9.2, 0.27, "random pick among\nshown labels", ha="right", va="bottom", fontsize=5.5, color="gray")
    ax.plot([], [], color=COLOR["extended_collapse"], marker="o", ms=2.2, label="extended")
    ax.plot([], [], color=COLOR["base_collapse"], marker="o", ms=2.2, label="base")
    ax.legend(loc="center left", frameon=False)
    ax.set_xlabel("generation"); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.04)
    ax.set_title("(a) accuracy, p=1.0, one line per seed")

    ax = axes[1]
    for s in base:
        ax.plot(range(len(base[s])), [fnum(r, "induction_score_mean") for r in base[s]], color=COLOR["base_collapse"], lw=1.0, marker="o", ms=2.2, alpha=0.85)
    for s in ext:
        ax.plot(range(len(ext[s])), [fnum(r, "induction_score_mean") for r in ext[s]], color=COLOR["extended_collapse"], lw=1.0, marker="o", ms=2.2, alpha=0.85)
    baseline = fnum(next(iter(ext.values()))[0], "induction_score_baseline")
    ax.axhline(baseline, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0.08, 0.56)
    ax.text(0.0, baseline - 0.008, "random-attention baseline", ha="left", va="top", fontsize=5.5, color="gray")
    ax.set_xlabel("generation"); ax.set_ylabel("induction score")
    ax.set_title("(b) induction score (mean over heads)")

    ax = axes[2]
    seed0 = min(ext)
    accs = [fnum(r, "test_acc") for r in ext[seed0]]
    # Example failed generation for panel (c): generation 5 if it failed (what the report shows), else the first failure, else the last generation.
    g_fail = 5 if (len(accs) > 5 and accs[5] < FAIL_THRESHOLD) else next((g for g in range(1, len(accs)) if accs[g] < FAIL_THRESHOLD), len(accs) - 1)
    healthy = load_epochs(results, "extended_collapse", seed0, 0)
    failed = load_epochs(results, "extended_collapse", seed0, g_fail)
    for rows, col, name in [(healthy, "#009E73", "gen 0 (healthy)"), (failed, "#CC79A7", f"gen {g_fail} (failed)")]:
        ep = [int(r["epoch"]) for r in rows]
        ax.plot(ep, [fnum(r, "train_acc") for r in rows], color=col, lw=1.1, label=f"{name}: train")
        ax.plot(ep, [fnum(r, "val_acc") for r in rows], color=col, lw=1.1, ls="--", label=f"{name}: val")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.04)
    ax.legend(loc="center right", frameon=False)
    ax.set_title(f"(c) extended, seed {seed0}: train vs. val")

    fig.tight_layout(pad=0.5, w_pad=0.8)
    fig.savefig(figdir / "fig_main.png", dpi=220)
    plt.close(fig)


def fig_failure_map(results: Path, figdir: Path, floor: float):
    style()
    order = present_conds(results)
    mat, names, bounds = [], [], []
    for cond in order:
        runs = load_collapse(results, cond)
        for s in sorted(runs):
            mat.append([fnum(r, "test_acc") for r in runs[s]]); names.append(f"{LABEL[cond]}  s{s}")
        bounds.append(len(mat) - 0.5)
    mat = np.array(mat)
    height = min(4.8, max(2.1, 0.115 * len(mat) + 0.45))      # grows with the number of seeds/conditions
    fig, axes = plt.subplots(1, 2, figsize=(7.0, height), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=5.8)
    ax.set_xticks(range(mat.shape[1])); ax.set_xlabel("generation")
    for b in bounds[:-1]:
        ax.axhline(b, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("test accuracy", fontsize=6.5)
    ax.set_title("(a) test accuracy of every trained model")

    ax = axes[1]
    for cond in order:
        runs = load_collapse(results, cond)
        kl = np.array([[fnum(r, "symbol_kl_vs_gen0") for r in runs[s]] for s in sorted(runs)]).mean(axis=0)
        ax.plot(range(1, len(kl)), kl[1:], color=COLOR[cond], marker="o", ms=2.2, lw=1.0, label=LABEL[cond])
    ax.axhline(floor, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(4e-4, None)
    ax.text(9.0, floor * 0.9, "noise floor (two independent real pools)", fontsize=5.5, color="gray", va="top", ha="right")
    ax.set_xlabel("generation"); ax.set_ylabel("symbol KL(gen n || gen 0), mean of seeds")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) marginal symbol drift vs. noise floor")
    fig.tight_layout(pad=0.5, w_pad=1.0)
    fig.savefig(figdir / "fig_failure_map.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figdir", default=str(ROOT / "abstract" / "figures"))
    args = ap.parse_args()
    results, figdir = Path(args.results), Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    summary = {"sanity_no_collapse": sanity_runs(results), "collapse": collapse_summary(results), "reference": reference_values(results)}
    with open(results / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if {"extended_collapse", "base_collapse"} <= set(summary["collapse"]):
        fig_main(results, figdir)
    fig_failure_map(results, figdir, summary["reference"]["symbol_kl_noise_floor_mean"])
    for cond, e in summary["collapse"].items():
        print(f"{LABEL[cond]:24s} seeds={e['seeds']} failed {e['failed_trainings_gen_ge1']}/{e['n_trainings_gen_ge1']} | seeds failing: {e['seeds_with_any_failure']} | "
              f"first failure gen: {e['first_failure_gen_by_seed']} | final acc: {e['final_gen_acc_by_seed']} | recovered: {e['recovered_after_failure_by_seed']}")
        print(f"{'':18s} diag failed={e['diagnostics']['failed']}  ok={e['diagnostics']['ok']}")
        print(f"{'':18s} median epoch to val>0.9: gen0={e['median_epoch_to_val90_gen0']} healthy gen>=1={e['median_epoch_to_val90_healthy_gen_ge1']}")
    print("reference:", json.dumps(summary["reference"], indent=1))
    print("wrote", results / "analysis_summary.json", "and figures in", figdir)


if __name__ == "__main__":
    main()
