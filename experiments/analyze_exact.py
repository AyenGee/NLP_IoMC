"""Analysis of the exact-task runs. Phase 2: the generation-0 replication (the paper's Fig. 3 and Fig. 4).

    python experiments/analyze_exact.py --gen0                  # results/exact/gen0_summary.json + figures

(also reached as ``python experiments/analyze_results.py`` without --pilot). Reads logs/ and checkpoints/ under
--results (default results/exact) and writes the summary JSON and PNGs there; every number is computed from the logs and
checkpoints by this script.

For each run gen0_<variant>_s<seed> it reports: final loss / accuracy / perplexity on the four evaluation sets (plus the
repository's test_class_ex0), the per-head induction strength (layer 2), previous-token score (layer 1) and match score
(layer 2), the phase-change landmarks (documented thresholds in src/metrics.py), and the head ablations of the final
checkpoint (src/circuits.py): zeroing each layer-2 head, keeping only one, zeroing layer-1 heads, cutting layer 1's direct
path, and the additivity of the induction heads' effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import circuits as C  # noqa: E402
import metrics as M  # noqa: E402
from plot_style import HEAD_COLORS, SET_COLOR, SET_LABEL, style  # noqa: E402
from train import EvalSets, load_model  # noqa: E402
import data as D  # noqa: E402

SETS = ["train_eval", "val", "test_exemplars", "test_relabel", "test_class_ex0"]


def read_log(path: Path) -> list[dict]:
    return [{k: (float(v) if v not in ("", "nan") else float("nan")) for k, v in r.items()} for r in csv.DictReader(open(path))]


def find_runs(results: Path, prefix: str = "gen0") -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for p in sorted((results / "logs").glob(f"{prefix}_*_s*.csv")):
        _, variant, s = p.stem.split("_")[:3]
        out.setdefault(variant, {})[int(s[1:])] = p.stem
    return out


def run_summary(results: Path, name: str, variant: str, evals: EvalSets) -> dict:
    rows = read_log(results / "logs" / f"{name}.csv")
    last = rows[-1]
    n_heads = sum(1 for k in last if k.startswith("ind_L2H"))
    entry = {"steps": int(last["step"]), "sequences": int(last["sequences"]),
             "final": {s: {"loss": last[f"{s}/loss"], "acc": last[f"{s}/acc"], "ppl": last[f"{s}/ppl"]} for s in SETS},
             "induction_strength_L2": [last[f"ind_L2H{h}"] for h in range(n_heads)],
             "previous_token_L1": [last[f"prev_L1H{h}"] for h in range(n_heads)],
             "match_score_L2": [last[f"match_L2H{h}"] for h in range(n_heads)],
             "landmarks": M.phase_landmarks(rows)}
    if variant == "extended":
        for s in SETS:
            entry["final"][s].update({"loss_s2": last[f"{s}/loss_s2"], "acc_s2": last[f"{s}/acc_s2"],
                                      "loss_ls2": last[f"{s}/loss_ls2"], "acc_ls2": last[f"{s}/acc_ls2"]})
    ck = results / "checkpoints" / f"{name}.pt"
    if ck.exists():
        model, _ = load_model(ck)
        prof = C.ablation_profile(model, evals)
        entry["ablation"] = prof
        strengths = {h: last[f"ind_L2H{h}"] for h in range(n_heads)}
        entry["additivity"] = C.additivity(model, evals, strengths)
    return entry


def gen0_analysis(results: Path) -> dict:
    runs = find_runs(results, "gen0")
    summary: dict = {"thresholds": {"plateau_exit_loss": M.PLATEAU_EXIT_LOSS, "phase_acc": M.PHASE_ACC,
                                    "induction_head_strength": C.INDUCTION_HEAD_THRESHOLD},
                     "bayes_optimal_extended": M.BAYES_OPTIMAL, "variants": {}}
    feats = D.OmniglotFeatures().tensor
    for variant, seeds in runs.items():
        evals = EvalSets(feats, variant == "extended")
        per_seed = {s: run_summary(results, name, variant, evals) for s, name in sorted(seeds.items())}
        agg = {}
        for s_name in SETS:
            accs = [per_seed[s]["final"][s_name]["acc"] for s in per_seed]
            agg[s_name] = {"acc_mean": float(np.mean(accs)), "acc_min": float(np.min(accs)), "acc_max": float(np.max(accs))}
        summary["variants"][variant] = {"n_seeds": len(per_seed), "per_seed": per_seed, "aggregate_final_acc": agg}
    return summary


def fig_gen0_curves(results: Path, figdir: Path, variant: str, seeds: dict[int, str]):
    style()
    n = len(seeds)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n + 0.3, 4.0), squeeze=False, sharey="row")
    for j, (s, name) in enumerate(sorted(seeds.items())):
        rows = read_log(results / "logs" / f"{name}.csv")
        x = np.array([r["sequences"] for r in rows]) / 1e3
        ax = axes[0, j]
        for st in SETS:
            ax.plot(x, [r[f"{st}/loss"] for r in rows], color=SET_COLOR[st], lw=0.9, ls="--" if st == "test_class_ex0" else "-", label=SET_LABEL[st])
        ax.axhline(math.log(2), color="gray", ls=":", lw=0.8)
        ax.set_title(f"seed {s}"); ax.set_ylim(0, 2.1)
        if j == 0:
            ax.set_ylabel("loss on the query label"); ax.legend(frameon=False, loc="upper right")
        ax = axes[1, j]
        n_heads = sum(1 for k in rows[0] if k.startswith("ind_L2H"))
        for h in range(n_heads):
            ax.plot(x, [r[f"ind_L2H{h}"] for r in rows], color=HEAD_COLORS[h % 8], lw=0.9, label=f"L2H{h}")
        ax.set_xlabel("sequences seen (thousands)")
        if j == 0:
            ax.set_ylabel("induction strength (layer 2)"); ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle(f"{variant}: generation 0 (dotted: ln 2)", fontsize=8)
    fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.7)
    fig.savefig(figdir / f"fig_gen0_curves_{variant}.png", dpi=300)
    plt.close(fig)


def fig_gen0_ablation(summary: dict, figdir: Path, variant: str):
    v = summary["variants"][variant]["per_seed"]
    if not all("ablation" in e for e in v.values()):
        return
    style()
    seeds = sorted(v)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.4, 1.4, 1.0]})
    H = len(v[seeds[0]]["induction_strength_L2"])
    w = 0.8 / len(seeds)
    for ax, key, title in [(axes[0], "zero_L2H", "zero one layer-2 head: accuracy drop"), (axes[1], "keep_only_L2H", "keep only one layer-2 head: accuracy")]:
        for i, s in enumerate(seeds):
            ab = v[s]["ablation"]
            base = ab["train_eval/baseline"]
            vals = [(base - ab[f"train_eval/{key}{h}"]) if key == "zero_L2H" else ab[f"train_eval/{key}{h}"] for h in range(H)]
            ax.bar(np.arange(H) + (i - (len(seeds) - 1) / 2) * w, vals, width=w, label=f"seed {s}")
        ax.set_xticks(range(H)); ax.set_xticklabels([f"L2H{h}" for h in range(H)], fontsize=5.5, rotation=45)
        ax.set_title(title, fontsize=6.5)
    axes[0].legend(frameon=False)
    ax = axes[2]
    labels = ["zero all L2", "zero all L1", "cut L1 direct", "baseline"]
    keys = ["zero_all_L2", "zero_all_L1", "cut_L1_direct", "baseline"]
    for i, s in enumerate(seeds):
        ab = v[s]["ablation"]
        ax.bar(np.arange(4) + (i - (len(seeds) - 1) / 2) * w, [ab[f"train_eval/{k}"] for k in keys], width=w)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=5.5, rotation=45); ax.set_title("whole-layer interventions: accuracy", fontsize=6.5)
    fig.suptitle(f"{variant}: ablations of the final generation-0 model (accuracy on train sequences)", fontsize=7.5)
    fig.tight_layout(pad=0.5, w_pad=0.7)
    fig.savefig(figdir / f"fig_gen0_ablation_{variant}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "exact"))
    ap.add_argument("--gen0", action="store_true", help="generation-0 replication analysis")
    ap.add_argument("--collapse", action="store_true", help="Tier 1 collapse analysis (summary JSON + figures)")
    args = ap.parse_args()
    results = Path(args.results)
    figdir = results / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if args.gen0:
        summary = gen0_analysis(results)
        with open(results / "gen0_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        for variant, seeds in find_runs(results, "gen0").items():
            fig_gen0_curves(results, figdir, variant, seeds)
            fig_gen0_ablation(summary, figdir, variant)
        print("wrote", results / "gen0_summary.json", "and figures in", figdir)
    if args.collapse:
        import analyze_collapse as AC
        summary = AC.collapse_summary(results)
        with open(results / "analysis_summary_exact.json", "w") as f:
            json.dump(summary, f, indent=2)
        AC.make_figures(results)
        print("wrote", results / "analysis_summary_exact.json", "and collapse figures in", figdir)


if __name__ == "__main__":
    main()
