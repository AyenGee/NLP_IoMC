"""Learning-rate tuning on the exact task (plan section 7, Tier 1 item 2).

    python experiments/analyze_tuning.py            # reads results/exact, writes tuning_summary.json, tuning_table.md, figures

Grid: lr in {1e-5, 3e-5, 1e-4, 3e-4} x 3 seeds, for the base and the extended variant, each trained for the paper's budget
(1e6 sequences = 31,250 steps) with everything else fixed. Selection is by VALIDATION accuracy (held-out exemplars) and the
time to the phase change, as the plan says, but a learning rate is only eligible if it gives the same CIRCUIT as the paper's
lr: a larger lr can reach 100% training accuracy with a different, memorising solution (observed for the extended model at
lr 1e-4: induction strength ~0 and accuracy 0.29 on held-out label pairs), so validation accuracy alone is not enough.

Operational definition of "same circuit" (documented thresholds, CIRCUIT_*):
    copy generalises   mean accuracy on test_relabel (held-out label pairs) >= CIRCUIT_COPY_ACC
    induction heads    best layer-2 induction strength >= CIRCUIT_INDUCTION in every seed
(the base model at the paper's lr reaches 0.96-0.99 and 0.71-0.84).
The 1e-5 cells are the generation-0 runs.

Variants analysed
    base             the paper's task and loss
    extended         the extended model with the AUXILIARY losses weighted 0.1 (src/metrics.py EXTENDED_LOSS_WEIGHTS); this is the model
                     the collapse experiments use
    extended_equal   the extended model with the plan's EQUAL loss weights, kept as evidence: it does not form the paper's circuit at any
                     learning rate (lr 3e-5 was not run); not eligible for selection

Selection rule (the plan: "selection by validation accuracy and time to the phase change; a larger lr may be used if it gives the same final
validation accuracy and the same circuit"): among learning rates that (i) give the same circuit, (ii) show the phase change in every seed, keep
those whose mean validation accuracy is within VAL_TOL (0.02) of the best such lr, and select the one with the earliest median phase change.
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
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import circuits as C  # noqa: E402
import data as D  # noqa: E402
import metrics as M  # noqa: E402
from plot_style import style  # noqa: E402
from train import EvalSets, load_model  # noqa: E402

LRS = [1e-5, 3e-5, 1e-4, 3e-4]
LR_COLOR = {1e-5: "#000000", 3e-5: "#0072B2", 1e-4: "#D55E00", 3e-4: "#009E73"}
SEEDS = [0, 1, 2]
SETS = ["train_eval", "val", "test_exemplars", "test_relabel", "test_class_ex0"]
CIRCUIT_COPY_ACC = 0.9
CIRCUIT_INDUCTION = 0.3
VAL_TOL = 0.02
VARIANTS = ("base", "extended", "extended_equal")


def candidates(variant: str, lr: float, seed: int) -> list[str]:
    """Run names for a (variant, lr, seed) cell; the first that has a log is used."""
    t = {3e-5: "0.00003", 1e-4: "0.0001", 3e-4: "0.0003"}.get(lr)
    if variant == "base":
        return [f"gen0_base_s{seed}"] if lr == 1e-5 else [f"tune_base_lr{t}_s{seed}"]
    if variant == "extended":
        return [f"gen0w_extended_s{seed}"] if lr == 1e-5 else [f"tunew_extended_lr{t}_s{seed}"]
    if lr == 1e-5:                                   # extended_equal
        return [f"gen0_extended_s{seed}"]
    return [f"tune_extended_lr{t}_s{seed}"] + (["diag_extended_lr1e-4_s0"] if (lr == 1e-4 and seed == 0) else [])


def read_log(path: Path) -> list[dict]:
    return [{k: float(v) if v not in ("", "nan") else float("nan") for k, v in r.items()} for r in csv.DictReader(open(path))]


def find(results: Path, variant: str, lr: float, seed: int) -> str | None:
    for n in candidates(variant, lr, seed):
        if (results / "logs" / f"{n}.csv").exists():
            return n
    return None


def cell(results: Path, variant: str, lr: float, evals: EvalSets) -> dict | None:
    per = {}
    for seed in SEEDS:
        n = find(results, variant, lr, seed)
        if n is None:
            continue
        rows = read_log(results / "logs" / f"{n}.csv")
        last = rows[-1]
        H = sum(1 for k in last if k.startswith("ind_L2H"))
        e = {"run": n, "steps": int(last["step"]), "final": {s: last[f"{s}/acc"] for s in SETS},
             "final_loss": {s: last[f"{s}/loss"] for s in SETS}, "landmarks": M.phase_landmarks(rows),
             "ind_max": max(last[f"ind_L2H{h}"] for h in range(H)),
             "n_induction_heads": sum(last[f"ind_L2H{h}"] > C.INDUCTION_HEAD_THRESHOLD for h in range(H)),
             "match_max": max(last[f"match_L2H{h}"] for h in range(H))}
        if variant == "extended":
            e["acc_s2_train"], e["acc_ls2_train"] = last["train_eval/acc_s2"], last["train_eval/acc_ls2"]
        ck = results / "checkpoints" / f"{n}.pt"
        if ck.exists():
            model, _ = load_model(ck)
            ab = C.ablation_profile(model, evals, sets=("train_eval",))
            base = ab["train_eval/baseline"]
            e["ablation"] = {"baseline": base, "zero_all_L2": ab["train_eval/zero_all_L2"], "cut_L1_direct": ab["train_eval/cut_L1_direct"],
                             "max_single_head_drop": max(base - ab[f"train_eval/zero_L2H{h}"] for h in range(H)),
                             "best_keep_only": max(ab[f"train_eval/keep_only_L2H{h}"] for h in range(H))}
        per[seed] = e
    if not per:
        return None
    agg = {s: float(np.mean([p["final"][s] for p in per.values()])) for s in SETS}
    ph = [p["landmarks"]["train_acc_gt_0.9_step"] for p in per.values()]
    pe = [p["landmarks"]["plateau_exit_step"] for p in per.values()]
    out = {"n_seeds": len(per), "per_seed": per, "mean_final_acc": agg,
           "plateau_exit_steps": pe, "train_acc_gt_0.9_steps": ph,
           "n_seeds_phase_change": int(sum(x is not None for x in ph)),
           "median_train_acc_gt_0.9_step": float(np.median([x for x in ph if x is not None])) if any(x is not None for x in ph) else None,
           "min_ind_max": min(p["ind_max"] for p in per.values()), "mean_ind_max": float(np.mean([p["ind_max"] for p in per.values()]))}
    out["same_circuit"] = bool(agg["test_relabel"] >= CIRCUIT_COPY_ACC and out["min_ind_max"] >= CIRCUIT_INDUCTION)
    return out


def analysis(results: Path) -> dict:
    feats = D.OmniglotFeatures().tensor
    summary = {"lrs": LRS, "thresholds": {"copy_acc": CIRCUIT_COPY_ACC, "induction": CIRCUIT_INDUCTION}, "variants": {}}
    for variant in VARIANTS:
        evals = EvalSets(feats, variant.startswith("extended"))
        cells = {}
        for lr in LRS:
            c = cell(results, variant, lr, evals)
            if c:
                cells[f"{lr:g}"] = c
        eligible = {k: c for k, c in cells.items() if c["same_circuit"] and c["n_seeds_phase_change"] == c["n_seeds"]}
        best = None
        if eligible and variant != "extended_equal":
            top = max(c["mean_final_acc"]["val"] for c in eligible.values())
            near = {k: c for k, c in eligible.items() if c["mean_final_acc"]["val"] >= top - VAL_TOL}
            best = min(near, key=lambda k: near[k]["median_train_acc_gt_0.9_step"] or 1e9)
        summary["variants"][variant] = {"cells": cells, "eligible_lrs": sorted(eligible), "selected_lr": best}
    return summary


def table_md(summary: dict) -> str:
    lines = []
    for variant, v in summary["variants"].items():
        lines += [f"### {variant}", "", "| lr | seeds | phase change (train acc > 0.9), steps | train | val | test classes (ex 0) | held-out pairs (copy) | best induction strength | zero all L2 | cut L1 direct | max single-head drop | same circuit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in v["cells"].items():
            m = c["mean_final_acc"]
            steps = ", ".join("none" if x is None else str(int(x)) for x in c["train_acc_gt_0.9_steps"])
            ab = [p["ablation"] for p in c["per_seed"].values() if "ablation" in p]
            abm = (lambda key: f"{np.mean([a[key] for a in ab]):.3f}") if ab else (lambda key: "-")
            lines.append(f"| {k} | {c['n_seeds']} | {steps} | {m['train_eval']:.3f} | {m['val']:.3f} | {m['test_class_ex0']:.3f} | {m['test_relabel']:.3f} | {c['mean_ind_max']:.2f} (min {c['min_ind_max']:.2f}) | {abm('zero_all_L2')} | {abm('cut_L1_direct')} | {abm('max_single_head_drop')} | {'yes' if c['same_circuit'] else 'NO'} |")
        lines += ["", f"Eligible learning rates: {v['eligible_lrs']}; selected: {v['selected_lr']}", ""]
    return "\n".join(lines)


def fig_tuning(results: Path, figdir: Path):
    style()
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.2), sharex=True)
    for i, variant in enumerate(VARIANTS):
        for lr in LRS:
            first = True
            for seed in SEEDS:
                n = find(results, variant, lr, seed)
                if n is None:
                    continue
                rows = read_log(results / "logs" / f"{n}.csv")
                x = np.array([r["sequences"] for r in rows]) / 1e3
                kw = dict(color=LR_COLOR[lr], lw=0.8, alpha=0.8, label=f"lr {lr:g}" if first else None)
                first = False
                axes[i, 0].plot(x, [r["train_eval/loss"] for r in rows], **kw)
                axes[i, 1].plot(x, [r["val/acc"] for r in rows], **kw)
                axes[i, 2].plot(x, [r["test_relabel/acc"] for r in rows], **kw)
                axes[i, 3].plot(x, [r["ind_max"] for r in rows], **kw)
        for j, t in enumerate(["train loss (query label)", "val accuracy (held-out exemplars)", "held-out label pairs (copy)", "best induction strength"]):
            axes[i, j].set_title(f"{variant}: {t}" if j == 0 else t, fontsize=6.3)
        axes[i, 0].axhline(math.log(2), color="gray", ls=":", lw=0.7)
    axes[0, 0].legend(frameon=False, loc="upper right")
    for a in axes[1]:
        a.set_xlabel("sequences seen (thousands)")
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    fig.savefig(figdir / "fig_tuning.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "exact"))
    args = ap.parse_args()
    results = Path(args.results)
    figdir = results / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    summary = analysis(results)
    (results / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    (results / "tuning_table.md").write_text(table_md(summary))
    fig_tuning(results, figdir)
    print(table_md(summary))


if __name__ == "__main__":
    main()
