"""Summarise the 6 conditions x 3 seeds of results and draw the report figures.

    python experiments/analyze_results.py

Reads results/ (written by the cluster jobs / run_all.py), writes
  results/analysis_summary.json     every number quoted in the report
  abstract/figures/fig_main.png     per-seed accuracy / induction score / train-vs-val dynamics
  abstract/figures/fig_failure_map.png   accuracy of every (condition, seed, generation) + KL vs. noise floor
  abstract/figures/fig_training.png      generation-0 learning curves (loss, accuracy, induction score), per seed
  abstract/figures/fig_v2.png            corrected-protocol runs: accuracy per generation per seed, induction score
  abstract/figures/fig_main_v2.png       main-text figure: accuracy per seed, duplicates vs none vs real-data control vs base
  abstract/figures/fig_mech_v2.png       main-text figure: induction score, non-classic share, ACDC circuit size

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
CLASSIC_HEAD = 0.5            # best-head induction score at/above this = a classic induction head
ALT_HEAD = 0.3                # below this (while accurate) = no head shows the classic induction pattern
# Conditions in plotting order; only those whose runs exist in results/ are analysed.
ALL_CONDS = ["extended_collapse", "ablation_p05", "ablation_p025", "base_collapse",
             "extended_collapse_nodup", "control_real_resampled", "control_real_fresh",
             "v2_extended_p1", "v2_extended_p1_dups", "v2_extended_p05", "v2_extended_p025", "v2_base_p1", "v2_control_real"]
LABEL = {
    "v2_extended_p1": "v2 extended, p=1.0",
    "v2_extended_p1_dups": "v2 extended, p=1.0, dups",
    "v2_extended_p05": "v2 extended, p=0.5",
    "v2_extended_p025": "v2 extended, p=0.25",
    "v2_base_p1": "v2 base, p=1.0",
    "v2_control_real": "v2 control: real data only",
    "extended_collapse": "extended, p=1.0",
    "ablation_p05": "extended, p=0.5",
    "ablation_p025": "extended, p=0.25",
    "base_collapse": "base, p=1.0",
    "extended_collapse_nodup": "extended, p=1.0, no dups",
    "control_real_resampled": "control: real, resampled",
    "control_real_fresh": "control: real, fresh",
}
COLOR = {
    "v2_extended_p1": "#8C2D04", "v2_extended_p1_dups": "#D94801", "v2_extended_p05": "#B8860B",
    "v2_extended_p025": "#006D2C", "v2_base_p1": "#08519C", "v2_control_real": "#636363",
    "extended_collapse": "#D55E00",
    "ablation_p05": "#E69F00",
    "ablation_p025": "#009E73",
    "base_collapse": "#0072B2",
    "extended_collapse_nodup": "#CC79A7",
    "control_real_resampled": "#56B4E9",
    "control_real_fresh": "#999999",
}
# Optional per-generation columns, written only by runs made after the diagnostics were added.
OPTIONAL_FIELDS = ["induction_score_max", "prev_token_score_max", "induction_off0_max", "induction_off2_max", "pool_distinct_frac", "pool_frac_fully_valid",
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
        # Runs whose generation 0 (clean real data) already failed are not collapse; report the accounting without them too.
        ok_idx = [i for i in range(len(seeds)) if not failed[i, 0]]
        entry["seeds_with_healthy_gen0"] = [seeds[i] for i in ok_idx]
        entry["failed_trainings_gen_ge1_excl_failed_gen0"] = int(failed[ok_idx, 1:].sum())
        entry["n_trainings_gen_ge1_excl_failed_gen0"] = int(failed[ok_idx, 1:].size)
        entry["seeds_ever_failing_excl_failed_gen0"] = [seeds[i] for i in ok_idx if failed[i, 1:].any()]
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

    # Any run's config gives the task and pool size; prefer the original seed-0 run so the reference values stay what they were.
    cfg_path = next((results / "collapse" / r / "run_config.json" for r in ["extended_collapse_seed0", "v2_extended_p1_seed0"]
                     if (results / "collapse" / r / "run_config.json").exists()), None)
    if cfg_path is None:
        cfg_path = sorted((results / "collapse").glob("*/run_config.json"))[0]
    cfg = json.load(open(cfg_path))
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


def gen0_summary(results: Path) -> dict:
    """Generation 0 trains on clean real data, so a failure there is NOT collapse (a seed/budget failure)."""
    out = {}
    for cond in present_conds(results):
        runs = load_collapse(results, cond)
        out[cond] = {s: {"test_acc": round(fnum(runs[s][0], "test_acc"), 4), "epochs_run": int(fnum(runs[s][0], "n_epochs_run"))} for s in sorted(runs)}
    failed = sorted({(c, s) for c, d in out.items() for s, v in d.items() if v["test_acc"] < FAIL_THRESHOLD})
    return {"by_cond_seed": out, "gen0_failed_runs": failed}


def retrain_summary(results: Path, v2: bool = False) -> dict:
    """The first failing generation of each failing run, retrained with early stopping disabled (v2=False: the
    original-protocol runs) or for a longer budget than the 150 epochs they were trained for (v2=True)."""
    cases = []
    for p in sorted((results / "retrain").glob("summary_*.json")):
        j = json.load(open(p))
        if j["cond"].startswith("v2_") != v2:
            continue
        best = max(v["test_acc"] for v in j["retrained"].values())
        cases.append({
            "cond": j["cond"], "seed": j["seed"], "gen": j["gen"],
            "original_test_acc": j["original"]["test_acc"], "original_stopped_at_epoch": j["original"]["epochs_run"],
            "budgets": {k: {"test_acc": v["test_acc"], "epoch_first_val_gt_0.9": v["first_epoch_val_acc_gt_0.9"],
                            "max_val_acc": v["max_val_acc"], "epoch_of_max_val_acc": v["epoch_of_max_val_acc"]} for k, v in j["retrained"].items()},
            "best_test_acc_over_budgets": best,
            "outcome": "recovered" if best >= 0.9 else ("partial" if best >= FAIL_THRESHOLD else "persistent"),
        })
    return {"cases": cases, "n_cases": len(cases),
            "n_recovered": sum(c["outcome"] == "recovered" for c in cases),
            "n_partial": sum(c["outcome"] == "partial" for c in cases),
            "n_persistent": sum(c["outcome"] == "persistent" for c in cases)}


def circuit_rows(results: Path) -> list[dict]:
    rows = []
    for p in sorted((results / "analysis").glob("circuits_*.csv")):
        cond, seed = p.stem.replace("circuits_", "").rsplit("_seed", 1)
        for r in read_csv(p):
            d = {k: fnum(r, k) for k in r if k not in ("run", "induction_best_head", "prev_token_best_head", "ablate_max_single_head", "ind_off0_head", "ind_off1_head", "ind_off2_head")}
            d.update({"cond": cond, "seed": int(seed), "gen": int(r["generation"]), "best_head": r["induction_best_head"]})
            rows.append(d)
    return rows


def _mech(r: dict) -> str:
    return "classic" if r["induction_max"] >= CLASSIC_HEAD else ("alt" if r["induction_max"] < ALT_HEAD else "ambiguous")


def _family(r: dict) -> str | None:
    """Which kind of head the model's best heads look like, from the offset-resolved scores (None for
    runs analysed before those columns existed). classic: a head puts >= 0.5 of the query's attention on the
    labels after earlier occurrences (offset 1); shifted: on the next symbol after them (offset 2); match: on
    the earlier occurrences themselves (offset 0); none: no head reaches 0.3 at any offset; else mixed."""
    if "ind_off1_max" not in r or r["ind_off1_max"] != r["ind_off1_max"]:
        return None
    if r["ind_off1_max"] >= CLASSIC_HEAD:
        return "classic"
    if r["ind_off2_max"] >= CLASSIC_HEAD:
        return "shifted"
    if r["ind_off0_max"] >= CLASSIC_HEAD:
        return "match"
    return "none" if max(r["ind_off0_max"], r["ind_off1_max"], r["ind_off2_max"]) < ALT_HEAD else "mixed"


def circuit_summary(results: Path) -> dict:
    rows = circuit_rows(results)
    if not rows:
        return {}
    group = lambda r: "healthy" if r["eval_acc"] >= 0.9 else ("failed" if r["eval_acc"] < FAIL_THRESHOLD else "intermediate")
    out = {"n_models": len(rows), "thresholds": {"classic_head_ge": CLASSIC_HEAD, "alt_lt": ALT_HEAD}}
    cols = ["induction_max", "induction_mean", "prev_token_max", "ablate_max_single_delta_acc", "layer0_all_heads_delta_acc", "layer1_all_heads_delta_acc"]
    out["by_accuracy_group"] = {}
    for g in ["healthy", "intermediate", "failed"]:
        sub = [r for r in rows if group(r) == g]
        out["by_accuracy_group"][g] = {"n": len(sub), **{c: round(float(np.nanmean([r[c] for r in sub])), 4) for c in cols}} if sub else {"n": 0}
    healthy = [r for r in rows if group(r) == "healthy"]
    failed = [r for r in rows if group(r) == "failed"]
    out["failed_models_best_head_ge_0.30"] = int(sum(r["induction_max"] >= 0.30 for r in failed))
    out["failed_models_layer1_ablation_drop_gt_0.1"] = int(sum(r["layer1_all_heads_delta_acc"] > 0.1 for r in failed))
    out["healthy_models_best_head_in_layer1_share"] = round(float(np.mean([r["best_head"].startswith("layer1") for r in healthy])), 4) if healthy else None
    out["healthy_models_single_head_ablation_drop_gt_0.3_share"] = round(float(np.mean([r["ablate_max_single_delta_acc"] > 0.3 for r in healthy])), 4) if healthy else None
    out["healthy_models_both_layers_ablation_drop_gt_0.5_share"] = round(float(np.mean([r["layer0_all_heads_delta_acc"] > 0.5 and r["layer1_all_heads_delta_acc"] > 0.5 for r in healthy])), 4) if healthy else None
    out["mechanism_counts_accurate_models"] = {m: int(sum(_mech(r) == m for r in healthy)) for m in ["classic", "ambiguous", "alt"]}
    fam_rows = [r for r in healthy if _family(r) is not None]
    if fam_rows:
        fams = ["classic", "shifted", "match", "mixed", "none"]
        out["family_counts_accurate_models"] = {f: int(sum(_family(r) == f for r in fam_rows)) for f in fams}
        alt_rows = [r for r in fam_rows if _mech(r) == "alt"]
        out["family_of_models_the_offset1_score_calls_alt"] = {f: int(sum(_family(r) == f for r in alt_rows)) for f in fams}
        out["family_by_condition_accurate_gen_ge1"] = {
            name: {f: int(sum(_family(r) == f for r in fam_rows if r["cond"] in conds and r["gen"] >= 1)) for f in fams}
            for name, conds in {"real data (controls)": ["control_real_resampled", "control_real_fresh"],
                                "v2 extended p=1": ["v2_extended_p1"], "v2 extended p=1 dups": ["v2_extended_p1_dups"],
                                "v2 extended p=0.5": ["v2_extended_p05"], "v2 extended p=0.25": ["v2_extended_p025"],
                                "v2 base p=1": ["v2_base_p1"], "original extended p=1 no dups": ["extended_collapse_nodup"]}.items()
            if any(r["cond"] in conds for r in fam_rows)}
    classes = {"clean real data (controls)": ["control_real_resampled", "control_real_fresh"], "base p=1.0": ["base_collapse"],
               "extended p=0.25": ["ablation_p025"], "extended p=0.5": ["ablation_p05"],
               "extended p=1.0 (duplicates)": ["extended_collapse"], "extended p=1.0 (no duplicates)": ["extended_collapse_nodup"],
               "v2 extended p=1.0 (fixed budget, no dups)": ["v2_extended_p1"], "v2 extended p=1.0 (fixed budget, dups)": ["v2_extended_p1_dups"],
               "v2 extended p=0.5": ["v2_extended_p05"], "v2 extended p=0.25": ["v2_extended_p025"], "v2 base p=1.0": ["v2_base_p1"],
               "v2 control: real data only": ["v2_control_real"]}
    out["alt_mechanism_share_of_accurate_models"] = {}
    for name, conds in classes.items():
        s = [r for r in healthy if r["cond"] in conds]
        later = [r for r in s if r["gen"] >= 1]
        if s:
            out["alt_mechanism_share_of_accurate_models"][name] = {
                "n_accurate": len(s), "alt_share": round(float(np.mean([_mech(r) == "alt" for r in s])), 3),
                "n_accurate_gen_ge1": len(later), "alt_count_gen_ge1": int(sum(_mech(r) == "alt" for r in later)),
                "alt_share_gen_ge1": round(float(np.mean([_mech(r) == "alt" for r in later])), 3) if later else None}
    nf = [r for r in rows if r["eval_acc"] >= FAIL_THRESHOLD]
    out["accuracy_of_non_failed_models_by_mechanism"] = {
        m: {"n": int(sum(_mech(r) == m for r in nf)),
            "mean_acc": round(float(np.mean([r["eval_acc"] for r in nf if _mech(r) == m])), 4),
            "share_acc_lt_0.97": round(float(np.mean([r["eval_acc"] < 0.97 for r in nf if _mech(r) == m])), 3),
            "mean_layer0_ablation_drop": round(float(np.mean([r["layer0_all_heads_delta_acc"] for r in nf if _mech(r) == m])), 3),
            "mean_layer1_ablation_drop": round(float(np.mean([r["layer1_all_heads_delta_acc"] for r in nf if _mech(r) == m])), 3)}
        for m in ["classic", "ambiguous", "alt"] if any(_mech(r) == m for r in nf)}
    by_run: dict = {}
    for r in rows:
        by_run.setdefault((r["cond"], r["seed"]), []).append(r)
    c2a = a2c = pairs = 0
    for v in by_run.values():
        v = sorted(v, key=lambda r: r["gen"])
        for a, b in zip(v, v[1:]):
            if a["eval_acc"] >= 0.9 and b["eval_acc"] >= 0.9:
                pairs += 1; c2a += _mech(a) == "classic" and _mech(b) == "alt"; a2c += _mech(a) == "alt" and _mech(b) == "classic"
    out["switches_between_consecutive_accurate_generations"] = {"pairs": pairs, "classic_to_alt": int(c2a), "alt_to_classic": int(a2c)}
    out["nodup_runs"] = {s: [{"gen": r["gen"], "eval_acc": round(r["eval_acc"], 3), "induction_max": round(r["induction_max"], 3),
                              "layer1_ablation_drop": round(r["layer1_all_heads_delta_acc"], 3)} for r in sorted(by_run.get(("extended_collapse_nodup", s), []), key=lambda r: r["gen"])]
                         for s in sorted(k[1] for k in by_run if k[0] == "extended_collapse_nodup")}
    return out


def pool_structure_summary(results: Path) -> dict:
    """Structure statistics of the training pools (from the cluster snapshots), by generation, and the one-step test."""
    rows = circuit_rows(results)
    if not rows or "pool_frac_fully_valid" not in rows[0]:
        return {}
    gen0_ok = {(c, s) for c in present_conds(results) for s in seeds_for(results, c) if fnum(load_collapse(results, c)[s][0], "test_acc") >= FAIL_THRESHOLD}
    cols = ["pool_frac_distinct_ok", "pool_frac_repeats_ok", "pool_label_consistency_rate", "pool_frac_label_injective", "pool_frac_query_label_ok", "pool_frac_fully_valid"]
    out = {"by_generation_mean": {}}
    for cond in ["extended_collapse", "extended_collapse_nodup", "ablation_p05", "ablation_p025", "base_collapse"]:
        sub = [r for r in rows if r["cond"] == cond and (cond, r["seed"]) in gen0_ok]
        if sub:
            out["by_generation_mean"][cond] = {c: [round(float(np.nanmean([r[c] for r in sub if r["gen"] == g])), 4) for g in range(10)] for c in cols}
    # One-step test: every healthy generation-0 model generates the generation-1 pool; does that pool's quality predict a generation-1 failure?
    runs = []
    for cond in ["extended_collapse", "ablation_p05", "ablation_p025", "extended_collapse_nodup"]:
        for s in seeds_for(results, cond):
            g = load_collapse(results, cond)[s]
            c = [r for r in rows if r["cond"] == cond and r["seed"] == s]
            if fnum(g[0], "test_acc") >= 0.9 and len(c) > 1:
                c1 = sorted(c, key=lambda r: r["gen"])[1]
                runs.append({"failed": fnum(g[1], "test_acc") < FAIL_THRESHOLD, **{k: c1[k] for k in ["pool_label_consistency_rate", "pool_frac_fully_valid", "pool_frac_repeats_ok", "pool_frac_query_label_ok"]}})
    def auc(pos, neg):
        return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if pos and neg else None
    test = {"n_trainings": len(runs), "n_failed": int(sum(r["failed"] for r in runs))}
    for k in ["pool_label_consistency_rate", "pool_frac_fully_valid", "pool_frac_repeats_ok", "pool_frac_query_label_ok"]:
        f_, o_ = [r[k] for r in runs if r["failed"]], [r[k] for r in runs if not r["failed"]]
        test[k] = {"mean_failed": round(float(np.mean(f_)), 3) if f_ else None, "mean_ok": round(float(np.mean(o_)), 3) if o_ else None,
                   "auc_ok_greater_than_failed": round(auc(o_, f_), 2) if auc(o_, f_) is not None else None}
    out["one_step_test"] = test
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]] (a, c = failures; b, d = successes)."""
    n1, n2, k = a + b, c + d, a + c
    total = n1 + n2

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(total, k)

    p_obs = pmf(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return float(min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))))


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None

    def ranks(v):
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):             # average the ranks of ties
            m = v == val
            r[m] = r[m].mean()
        return r

    return round(float(np.corrcoef(ranks(x), ranks(y))[0, 1]), 3)


V2_CONDS = ["v2_extended_p1", "v2_extended_p1_dups", "v2_extended_p05", "v2_extended_p025", "v2_base_p1", "v2_control_real"]


def training_summary(results: Path) -> dict:
    """Reproduction check on generation 0 (real data only): the quantities the brief lists (train/val loss and
    accuracy over training, final test accuracy, perplexity), per variant, per seed and aggregated."""
    out = {}
    for variant, conds in [("base", ["v2_base_p1", "base_collapse"]), ("extended", ["v2_extended_p1", "extended_collapse"])]:
        cond = next((c for c in conds if seeds_for(results, c)), None)
        if cond is None:
            continue
        runs = load_collapse(results, cond)
        per_seed = {}
        for s in sorted(runs):
            g0, ep = runs[s][0], load_epochs(results, cond, s, 0)
            val_rows = [r for r in ep if r.get("val_acc", "") != ""]
            d = {"test_acc": fnum(g0, "test_acc"), "test_loss": fnum(g0, "test_loss"), "test_ppl": fnum(g0, "test_ppl"),
                 "final_train_acc": fnum(ep[-1], "train_acc"), "final_train_loss": fnum(ep[-1], "train_loss"),
                 "final_val_acc": fnum(val_rows[-1], "val_acc"), "final_val_loss": fnum(val_rows[-1], "val_loss"),
                 "epochs_run": len(ep),
                 "epoch_val_acc_gt_0.9": next((int(r["epoch"]) for r in val_rows if float(r["val_acc"]) > 0.9), None)}
            if "test_aux_label_acc" in g0:
                d["test_aux_label_acc"], d["test_aux_symbol_acc"] = fnum(g0, "test_aux_label_acc"), fnum(g0, "test_aux_symbol_acc")
            if "induction_score_max" in g0 and g0["induction_score_max"] != "":
                d["induction_score_max"] = fnum(g0, "induction_score_max")
            per_seed[s] = d
        ok = [d for d in per_seed.values() if d["test_acc"] >= FAIL_THRESHOLD]

        def agg(rows):
            keys = [k for k in rows[0] if k != "epoch_val_acc_gt_0.9"] if rows else []
            res = {k: {"mean": round(float(np.mean([r[k] for r in rows])), 5), "std": round(float(np.std([r[k] for r in rows])), 5)} for k in keys}
            t = [r["epoch_val_acc_gt_0.9"] for r in rows if r["epoch_val_acc_gt_0.9"] is not None]
            res["median_epoch_val_acc_gt_0.9"] = float(np.median(t)) if t else None
            return res

        out[variant] = {"cond": cond, "n_seeds": len(per_seed), "n_gen0_failed": len(per_seed) - len(ok),
                        "per_seed": per_seed, "mean_std_healthy_seeds": agg(ok), "mean_std_all_seeds": agg(list(per_seed.values()))}
    return out


def factorial_summary(results: Path) -> dict:
    """The 2x2 over the two training-protocol artifacts (early stopping on/off x duplicates on/off), extended p=1:
    failures among trainings of generations >= 1 of runs whose generation 0 is healthy."""
    cells = {"early_stopping_dups": "extended_collapse", "early_stopping_nodups": "extended_collapse_nodup",
             "fixed_budget_dups": "v2_extended_p1_dups", "fixed_budget_nodups": "v2_extended_p1"}
    out = {}
    for label, cond in cells.items():
        if not seeds_for(results, cond):
            continue
        runs = load_collapse(results, cond)
        for scope, seeds in [("seeds_0_2", [s for s in sorted(runs) if s <= 2]), ("all_seeds", sorted(runs))]:
            ok = [s for s in seeds if fnum(runs[s][0], "test_acc") >= FAIL_THRESHOLD]
            n = sum(len(runs[s]) - 1 for s in ok)
            k = sum(fnum(r, "test_acc") < FAIL_THRESHOLD for s in ok for r in runs[s][1:])
            out.setdefault(scope, {})[label] = {"cond": cond, "n_seeds": len(ok), "failed": int(k), "trainings": int(n),
                                                "share": round(k / n, 3) if n else None, "ci95": wilson(int(k), int(n))}
    for scope, d in out.items():
        def p(a, b):
            if a in d and b in d:
                x, y = d[a], d[b]
                return round(fisher_exact_p(x["failed"], x["trainings"] - x["failed"], y["failed"], y["trainings"] - y["failed"]), 5)
        d["fisher_p"] = {k: v for k, v in {
            "duplicates_effect_at_fixed_budget": p("fixed_budget_dups", "fixed_budget_nodups"),
            "early_stopping_effect_with_duplicates": p("early_stopping_dups", "fixed_budget_dups"),
            "early_stopping_effect_without_duplicates": p("early_stopping_nodups", "fixed_budget_nodups"),
        }.items() if v is not None}
        if d["fisher_p"]:
            d["fisher_p_note"] = "trainings within a run are not independent; read these p-values as descriptive only"
    return out


def v2_summary(results: Path, base: dict) -> dict:
    """Corrected-protocol runs: failures with intervals, accuracy by generation (median, since outcomes are bimodal),
    and the trend of accuracy with generation among all trainings."""
    out = {}
    for cond in V2_CONDS + ["extended_collapse_nodup"]:
        if cond not in base:
            continue
        runs = load_collapse(results, cond)
        seeds = sorted(runs)
        acc = np.array([[fnum(r, "test_acc") for r in runs[s]] for s in seeds])
        ok = [i for i in range(len(seeds)) if acc[i, 0] >= FAIL_THRESHOLD]
        later = acc[ok][:, 1:] if ok else np.zeros((0, acc.shape[1] - 1))
        k, n = int((later < FAIL_THRESHOLD).sum()), int(later.size)
        gens = np.tile(np.arange(1, acc.shape[1]), (len(ok), 1))
        out[cond] = {
            "seeds": seeds, "n_seeds_healthy_gen0": len(ok), "gen0_failed_seeds": [seeds[i] for i in range(len(seeds)) if i not in ok],
            "failed": k, "trainings": n, "failure_share": round(k / n, 3) if n else None, "failure_share_ci95": wilson(k, n),
            "seeds_with_any_failure": [seeds[i] for i in ok if (acc[i, 1:] < FAIL_THRESHOLD).any()],
            "trainings_below_0.9": int((later < 0.9).sum()) if n else 0,
            "median_acc_by_gen": np.median(acc[ok], axis=0).round(4).tolist() if ok else None,
            "min_acc_by_gen": acc[ok].min(axis=0).round(4).tolist() if ok else None,
            "mean_acc_gen0": round(float(acc[ok][:, 0].mean()), 4) if ok else None,
            "mean_acc_gen_ge1": round(float(later.mean()), 4) if n else None,
            "mean_acc_final_gen": round(float(acc[ok][:, -1].mean()), 4) if ok else None,
            "spearman_gen_vs_acc_all_trainings": spearman(gens.ravel(), later.ravel()) if n else None,
            "final_gen_acc_by_seed": {seeds[i]: round(float(acc[i, -1]), 4) for i in ok},
        }
    return out


def perm_test_mean_diff(a, b) -> float | None:
    """Exact two-sided permutation p-value for a difference in means of two small samples (all splits)."""
    from itertools import combinations
    a, b = list(a), list(b)
    if not a or not b:
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


def _slope(y) -> float:
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def v2_contrasts(results: Path) -> dict:
    """The comparisons the report's claims rest on, for the corrected-protocol runs. A seed whose generation 0 is
    itself weak (test accuracy < 0.9) starts the chain with a poor generator and is reported separately: with
    healthy starts, any decline is a property of the recursion, not of one under-trained first model."""
    if not seeds_for(results, "v2_extended_p1"):
        return {}
    conds = [c for c in ["v2_extended_p1", "v2_extended_p1_dups", "v2_extended_p05", "v2_extended_p025", "v2_control_real"] if seeds_for(results, c)]
    runs = {c: load_collapse(results, c) for c in conds}
    acc = {c: {s: np.array([fnum(r, "test_acc") for r in runs[c][s]]) for s in runs[c]} for c in conds}
    ref = runs["v2_extended_p1"]
    strong = [s for s in sorted(ref) if fnum(ref[s][0], "test_acc") >= 0.9]
    weak = [s for s in sorted(ref) if s not in strong]
    out = {"strong_start_seeds": strong, "weak_start_seeds": weak,
           "weak_start_gen0_acc": {s: round(fnum(ref[s][0], "test_acc"), 4) for s in weak}, "by_condition": {}}
    if not strong:
        out["note"] = "no seed has a healthy generation 0, so there is nothing to contrast"
        return out
    for c in conds:
        a = acc[c]
        sub = {s: a[s] for s in strong if s in a}
        later = np.array([v[1:] for v in sub.values()])
        final = {s: round(float(v[-1]), 4) for s, v in sub.items()}
        slopes = {s: round(_slope(v[1:]), 5) for s, v in sub.items()}
        entry = {"n_seeds": len(sub), "failed_trainings": int((later < FAIL_THRESHOLD).sum()), "trainings": int(later.size),
                 "trainings_below_0.9": int((later < 0.9).sum()), "mean_acc_gen_ge1": round(float(later.mean()), 4),
                 "final_gen_acc_by_seed": final, "mean_final_gen_acc": round(float(np.mean(list(final.values()))), 4),
                 "min_final_gen_acc": round(float(min(final.values())), 4),
                 "slope_per_generation_by_seed": slopes, "mean_slope_per_generation": round(float(np.mean(list(slopes.values()))), 5),
                 "seeds_with_final_acc_lt_0.99": int(sum(v < 0.99 for v in final.values())),
                 "median_acc_by_gen": np.median(np.array(list(sub.values())), axis=0).round(4).tolist()}
        wk = {s: [round(float(x), 3) for x in a[s]] for s in weak if s in a}
        if wk:
            entry["weak_start_seed_acc_by_gen"] = wk
        if sub and all(runs[c][x][0].get("induction_off0_max", "") != "" for x in sub):
            cl = np.array([[fnum(r, "induction_score_max") for r in runs[c][x]] for x in sub])
            anyo = np.array([[max(fnum(r, "induction_score_max"), fnum(r, "induction_off0_max"), fnum(r, "induction_off2_max")) for r in runs[c][x]] for x in sub])
            entry["median_classic_score_by_gen"] = np.median(cl, axis=0).round(3).tolist()
            entry["median_best_of_offsets_score_by_gen"] = np.median(anyo, axis=0).round(3).tolist()
        out["by_condition"][c] = entry
    ctrl = out["by_condition"].get("v2_control_real")
    if ctrl:
        out["permutation_tests_final_gen_acc_strong_seeds"] = {}
        for c in conds:
            if c != "v2_control_real":
                out["permutation_tests_final_gen_acc_strong_seeds"][f"{c}_vs_control"] = perm_test_mean_diff(
                    out["by_condition"][c]["final_gen_acc_by_seed"].values(), ctrl["final_gen_acc_by_seed"].values())
        if "v2_extended_p1" in conds and "v2_extended_p1_dups" in conds:
            out["permutation_tests_final_gen_acc_strong_seeds"]["dups_vs_nodups_at_p1"] = perm_test_mean_diff(
                out["by_condition"]["v2_extended_p1_dups"]["final_gen_acc_by_seed"].values(),
                out["by_condition"]["v2_extended_p1"]["final_gen_acc_by_seed"].values())
    # Mechanism: accurate models of generations >= 1 whose best head is not classic, against the control
    rows = [r for r in circuit_rows(results) if r["eval_acc"] >= 0.9 and r["gen"] >= 1]
    if rows:
        share = {}
        for c in conds:
            sub = [r for r in rows if r["cond"] == c and r["seed"] in strong]
            if sub:
                share[c] = {"n_accurate": len(sub), "non_classic": int(sum(_mech(r) == "alt" for r in sub)),
                            "share": round(float(np.mean([_mech(r) == "alt" for r in sub])), 3)}
        out["non_classic_share_accurate_gen_ge1_strong_seeds"] = share
        if "v2_extended_p1" in share and "v2_control_real" in share:
            a_, b_ = share["v2_extended_p1"], share["v2_control_real"]
            out["non_classic_share_fisher_p1_vs_control"] = round(fisher_exact_p(a_["non_classic"], a_["n_accurate"] - a_["non_classic"],
                                                                                  b_["non_classic"], b_["n_accurate"] - b_["non_classic"]), 5)
        v2rows = [r for r in rows if r["cond"] in V2_CONDS and _family(r) is not None]
        out["accuracy_by_family_v2_accurate_gen_ge1"] = {f: {"n": int(sum(_family(r) == f for r in v2rows)),
                                                              "mean_acc": round(float(np.mean([r["eval_acc"] for r in v2rows if _family(r) == f])), 4)}
                                                         for f in ["classic", "shifted", "match"] if any(_family(r) == f for r in v2rows)}
    return out


def base_failure_summary(results: Path) -> dict:
    """Base variant: is a failure preceded by a corrupted pool? For the first failing generation of each run, the
    share of its training pool whose query label is correct (1.0 = a clean pool), and what the failed model then
    does to the next pool (an absorbing failure)."""
    out = {}
    for cond in [c for c in ["v2_base_p1", "base_collapse"] if seeds_for(results, c)]:
        runs = load_collapse(results, cond)
        per = {}
        for s in sorted(runs):
            rows = runs[s]
            g = next((i for i, r in enumerate(rows) if fnum(r, "test_acc") < FAIL_THRESHOLD), None)
            if g is None or "pool_frac_query_label_ok" not in rows[0] or rows[0]["pool_frac_query_label_ok"] == "":
                per[s] = {"first_failure_gen": g}
                continue
            per[s] = {"first_failure_gen": g, "pool_label_ok_at_first_failure": round(fnum(rows[g], "pool_frac_query_label_ok"), 4),
                      "pool_label_ok_next_gen": round(fnum(rows[g + 1], "pool_frac_query_label_ok"), 4) if g + 1 < len(rows) else None,
                      "recovered_later": bool(any(fnum(r, "test_acc") >= 0.9 for r in rows[g + 1:]))}
        clean = [v for v in per.values() if v.get("pool_label_ok_at_first_failure") is not None]
        out[cond] = {"per_seed": per, "seeds_failing": len(clean),
                     "failures_whose_pool_had_every_label_correct": int(sum(v["pool_label_ok_at_first_failure"] >= 0.9999 for v in clean if v["first_failure_gen"] >= 1)),
                     "failures_at_generation_ge1": int(sum(v["first_failure_gen"] >= 1 for v in clean)),
                     "failed_at_generation_0": [s for s, v in per.items() if v.get("first_failure_gen") == 0]}
    return out


def tuning_summary(results: Path) -> dict:
    """Hyper-parameter study (experiments/tune_hparams.py), if its outputs are present."""
    if not (results / "tuning").is_dir() or not list((results / "tuning").glob("*.json")):
        return {}
    sys.path.insert(0, str(ROOT / "experiments"))
    from tune_hparams import summarize
    return summarize(results / "tuning")


ACDC_NUMERIC = ["acdc_tau", "acdc_n_edges", "acdc_n_edges_total", "acdc_kl", "acdc_acc", "acdc_full_acc", "acdc_n_heads",
                "acdc_n_heads_l0", "acdc_n_heads_l1", "acdc_mlp0", "acdc_mlp1", "acdc_kcomp", "acdc_qcomp", "acdc_vcomp"]


def acdc_rows(results: Path) -> list[dict]:
    rows = []
    for p in sorted((results / "analysis").glob("acdc_*.csv")):
        cond, seed = p.stem[len("acdc_"):].rsplit("_seed", 1)
        for r in read_csv(p):
            d = {k: fnum(r, k) for k in ACDC_NUMERIC}
            d.update({"cond": cond, "seed": int(seed), "gen": int(r["generation"]), "corruption": r.get("acdc_corruption", "resample"), "heads": r["acdc_heads"],
                      "kcomp_edges": r["acdc_kcomp_edges"], "edges": r["acdc_edges"]})
            rows.append(d)
    return rows


def acdc_summary(results: Path) -> dict:
    """Edge-level (ACDC) circuits of accurate models: size, faithfulness, composition, and how they differ for
    models with / without a classic induction head and across training conditions."""
    rows = acdc_rows(results)
    if not rows:
        return {}
    crow = circuit_rows(results)
    mech = {(r["cond"], r["seed"], r["gen"]): _mech(r) for r in crow}
    fam = {(r["cond"], r["seed"], r["gen"]): _family(r) for r in crow}
    out = {"n_rows": len(rows), "taus": sorted({r["acdc_tau"] for r in rows}), "corruptions": sorted({r["corruption"] for r in rows}), "by_corruption_tau": {}}
    classes = {"real data (controls)": ["control_real_resampled", "control_real_fresh"],
               "original extended p=1": ["extended_collapse"], "original extended p=1 no dups": ["extended_collapse_nodup"],
               "v2 extended p=1": ["v2_extended_p1"], "v2 extended p=1 dups": ["v2_extended_p1_dups"],
               "v2 extended p=0.5": ["v2_extended_p05"], "v2 extended p=0.25": ["v2_extended_p025"], "v2 base p=1": ["v2_base_p1"],
               "v2 control (real data)": ["v2_control_real"]}

    def stats(sub):
        if not sub:
            return {"n": 0}
        return {"n": len(sub), "mean_live_edges": round(float(np.mean([r["acdc_n_edges"] for r in sub])), 2),
                "mean_circuit_acc": round(float(np.mean([r["acdc_acc"] for r in sub])), 4),
                "mean_full_acc": round(float(np.mean([r["acdc_full_acc"] for r in sub])), 4),
                "share_layer0_to_layer1_key_edge": round(float(np.mean([r["acdc_kcomp"] for r in sub])), 3),
                "share_layer0_to_layer1_query_edge": round(float(np.mean([r["acdc_qcomp"] for r in sub])), 3),
                "share_layer0_to_layer1_value_edge": round(float(np.mean([r["acdc_vcomp"] for r in sub])), 3),
                "share_uses_mlp0": round(float(np.mean([r["acdc_mlp0"] for r in sub])), 3),
                "share_uses_mlp1": round(float(np.mean([r["acdc_mlp1"] for r in sub])), 3),
                "mean_heads_layer0": round(float(np.mean([r["acdc_n_heads_l0"] for r in sub])), 2),
                "mean_heads_layer1": round(float(np.mean([r["acdc_n_heads_l1"] for r in sub])), 2)}

    for corruption, tau in [(c, t) for c in out["corruptions"] for t in out["taus"]]:
        at = [r for r in rows if r["acdc_tau"] == tau and r["corruption"] == corruption]
        acc = [r for r in at if r["acdc_full_acc"] >= 0.9]
        entry = {"all_accurate_models": stats(acc), "by_condition": {n: stats([r for r in acc if r["cond"] in cs]) for n, cs in classes.items()},
                 "by_mechanism": {m: stats([r for r in acc if mech.get((r["cond"], r["seed"], r["gen"])) == m]) for m in ["classic", "ambiguous", "alt"]},
                 "by_family": {f: stats([r for r in acc if fam.get((r["cond"], r["seed"], r["gen"])) == f]) for f in ["classic", "shifted", "match", "mixed", "none"]},
                 "failed_models": stats([r for r in at if r["acdc_full_acc"] < FAIL_THRESHOLD])}
        entry["by_family"] = {k: v for k, v in entry["by_family"].items() if v["n"]}
        entry["by_condition"] = {k: v for k, v in entry["by_condition"].items() if v["n"]}
        entry["by_mechanism"] = {k: v for k, v in entry["by_mechanism"].items() if v["n"]}
        # most common heads / edge sets in classic-mechanism circuits vs alt ones (strings, for inspection)
        for m in ["classic", "alt"]:
            sub = [r for r in acc if mech.get((r["cond"], r["seed"], r["gen"])) == m]
            if sub:
                from collections import Counter
                entry.setdefault("head_sets", {})[m] = Counter(r["heads"] for r in sub).most_common(5)
                entry.setdefault("key_edges", {})[m] = Counter(r["kcomp_edges"] for r in sub).most_common(5)
        out["by_corruption_tau"][f"{corruption}@{tau}"] = entry
    return out


def fig_main(results: Path, figdir: Path):
    """(a) duplicates create the cliff, (b) early stopping cuts off delayed transitions, (c) mechanism switch without accuracy loss."""
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1, 1.05, 1]})
    boot, nod = load_collapse(results, "extended_collapse"), load_collapse(results, "extended_collapse_nodup")
    ax = axes[0]
    for s in sorted(set(boot) & set(nod))[:3]:
        ax.plot(range(len(boot[s])), [fnum(r, "test_acc") for r in boot[s]], color=COLOR["extended_collapse"], lw=1.0, marker="o", ms=2.0, alpha=0.85)
        ax.plot(range(len(nod[s])), [fnum(r, "test_acc") for r in nod[s]], color=COLOR["extended_collapse_nodup"], lw=1.0, marker="o", ms=2.0, alpha=0.85)
    ax.axhline(0.25, color="gray", ls=":", lw=0.8)
    ax.plot([], [], color=COLOR["extended_collapse"], marker="o", ms=2, label="with duplicates")
    ax.plot([], [], color=COLOR["extended_collapse_nodup"], marker="o", ms=2, label="no duplicates")
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(0.0, 0.42))
    ax.set_xlabel("generation"); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.04)
    ax.set_title("(a) p=1.0, seeds 0-2")

    ax = axes[1]
    rs = retrain_summary(results)["cases"]
    short = {"extended_collapse": "ext p1", "ablation_p05": "ext p.5", "ablation_p025": "ext p.25", "base_collapse": "base p1"}
    for i, c in enumerate(rs):
        b = c["budgets"]; keys = sorted(b, key=lambda k: int(k[2:]))
        ax.plot([i], [c["original_test_acc"]], marker="x", color="#D55E00", ms=4, ls="none")
        ax.plot([i - 0.12], [b[keys[0]]["test_acc"]], marker="o", color="#0072B2", ms=3.2, ls="none")
        ax.plot([i + 0.12], [b[keys[-1]]["test_acc"]], marker="s", color="#009E73", ms=3.2, ls="none")
        ax.plot([i, i], [c["original_test_acc"], max(b[keys[0]]["test_acc"], b[keys[-1]]["test_acc"])], color="gray", lw=0.5, zorder=0)
    ax.plot([], [], "x", color="#D55E00", label="original (early-stopped)")
    ax.plot([], [], "o", color="#0072B2", label="no early stop, same budget")
    ax.plot([], [], "s", color="#009E73", label="no early stop, 300 epochs")
    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.02, 0.5), fontsize=5.3)
    ax.set_xticks(range(len(rs))); ax.set_xticklabels([f"{short.get(c['cond'], c['cond'])}\ns{c['seed']} g{c['gen']}" for c in rs], fontsize=5.2)
    ax.set_ylim(0, 1.04); ax.set_ylabel("test accuracy")
    ax.set_title("(b) first failure of each failing run")

    ax = axes[2]
    for s, col in list(zip(sorted(nod)[:2], ["#CC79A7", "#7B3F6E"])):
        rows = sorted([r for r in circuit_rows(results) if r["cond"] == "extended_collapse_nodup" and r["seed"] == s], key=lambda r: r["gen"])
        ax.plot([r["gen"] for r in rows], [r["eval_acc"] for r in rows], color=col, lw=1.1, marker="o", ms=2.0)
        ax.plot([r["gen"] for r in rows], [r["induction_max"] for r in rows], color=col, lw=1.1, ls="--", marker="s", ms=2.0)
    ax.plot([], [], color="gray", marker="o", ms=2, label="accuracy"); ax.plot([], [], color="gray", ls="--", marker="s", ms=2, label="best-head induction score")
    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.62))
    ax.set_xlabel("generation"); ax.set_ylim(0, 1.04)
    ax.set_title("(c) no duplicates, seeds 0 and 1")
    fig.tight_layout(pad=0.5, w_pad=0.7)
    fig.savefig(figdir / "fig_main.png", dpi=220)
    plt.close(fig)


def fig_pools(results: Path, figdir: Path):
    """(a) share of failed models per generation, (b) how structurally valid the synthetic pools are, vs. the marginal KL."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.95))
    ax = axes[0]
    for cond in ["extended_collapse", "ablation_p05", "ablation_p025", "base_collapse", "extended_collapse_nodup"]:
        runs = load_collapse(results, cond)
        ok = [s for s in sorted(runs) if fnum(runs[s][0], "test_acc") >= FAIL_THRESHOLD]
        if not ok:
            continue
        frac = np.array([[fnum(r, "test_acc") < FAIL_THRESHOLD for r in runs[s]] for s in ok]).mean(axis=0)
        ax.plot(range(len(frac)), frac, color=COLOR[cond], marker="o", ms=2.0, lw=1.0, ls="--" if cond == "extended_collapse_nodup" else "-", label=f"{LABEL[cond]} (n={len(ok)})")
    ax.set_xlabel("generation"); ax.set_ylabel("share of runs whose model failed"); ax.set_ylim(-0.02, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=5.2, columnspacing=0.8, handlelength=1.6)
    ax.set_title("(a) failure share by generation")

    ax = axes[1]
    st = pool_structure_summary(results).get("by_generation_mean", {})
    for cond, ls in [("extended_collapse", "-"), ("extended_collapse_nodup", "--")]:
        if cond in st:
            ax.plot(range(10), st[cond]["pool_frac_fully_valid"], color=COLOR[cond], ls=ls, marker="o", ms=2.0, lw=1.0)
            ax.plot(range(10), st[cond]["pool_label_consistency_rate"], color=COLOR[cond], ls=ls, marker="s", ms=2.0, lw=1.0, alpha=0.55)
    ax.plot([], [], color="gray", marker="o", ms=2, label="sequences fully valid"); ax.plot([], [], color="gray", marker="s", ms=2, alpha=0.55, label="repeated-symbol label consistency")
    ax.plot([], [], color=COLOR["extended_collapse"], label="with duplicates"); ax.plot([], [], color=COLOR["extended_collapse_nodup"], ls="--", label="no duplicates")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=5.2, columnspacing=0.8, handlelength=1.6)
    ax.set_xlabel("generation"); ax.set_ylabel("share of the training pool"); ax.set_ylim(-0.02, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(b) synthetic-pool structure, p=1.0")
    fig.tight_layout(pad=0.5, w_pad=1.0)
    fig.savefig(figdir / "fig_pools.png", dpi=220)
    plt.close(fig)


def fig_training(results: Path, figdir: Path):
    """Learning curves of the generation-0 models (real data only), one line per seed: loss and accuracy on train
    (solid) and validation (dashed), and the induction score, showing when the induction circuit forms."""
    tr = training_summary(results)
    if not tr:
        return
    style()
    fig, axes = plt.subplots(len(tr), 3, figsize=(7.0, 1.85 * len(tr)), squeeze=False)
    for row, (variant, info) in enumerate(tr.items()):
        cond = info["cond"]
        col = {"base": "#0072B2", "extended": "#D55E00"}[variant]
        for s in sorted(info["per_seed"]):
            ep = load_epochs(results, cond, s, 0)
            val = [r for r in ep if r.get("val_acc", "") != ""]
            e_all, e_val = [int(r["epoch"]) for r in ep], [int(r["epoch"]) for r in val]
            axes[row, 0].plot(e_all, [fnum(r, "train_loss") for r in ep], color=col, lw=0.8, alpha=0.7)
            axes[row, 0].plot(e_val, [fnum(r, "val_loss") for r in val], color=col, lw=0.8, alpha=0.7, ls="--")
            axes[row, 1].plot(e_all, [fnum(r, "train_acc") for r in ep], color=col, lw=0.8, alpha=0.7)
            axes[row, 1].plot(e_val, [fnum(r, "val_acc") for r in val], color=col, lw=0.8, alpha=0.7, ls="--")
            key = "induction_score_max" if any(r.get("induction_score_max", "") != "" for r in ep) else "induction_score_mean"
            snap = [r for r in ep if r.get(key, "") != ""]
            axes[row, 2].plot([int(r["epoch"]) for r in snap], [fnum(r, key) for r in snap], color=col, lw=0.9, alpha=0.8)
            axes[row, 2].plot([int(r["epoch"]) for r in snap], [fnum(r, "prev_token_score_mean") for r in snap], color="gray", lw=0.6, alpha=0.5, ls=":")
        snap0 = [r for r in load_epochs(results, cond, sorted(info["per_seed"])[0], 0) if r.get("induction_score_baseline", "") != ""]
        if snap0:
            axes[row, 2].axhline(fnum(snap0[0], "induction_score_baseline"), color="gray", ls="--", lw=0.6)
        axes[row, 0].set_yscale("log"); axes[row, 0].set_ylabel(f"{variant}\nquery-label loss")
        axes[row, 1].axhline(0.25, color="gray", ls=":", lw=0.8); axes[row, 1].set_ylabel("query-label accuracy"); axes[row, 1].set_ylim(0, 1.04)
        axes[row, 2].set_ylabel("induction score"); axes[row, 2].set_ylim(0, 1.04)
        if row == 0:
            axes[row, 0].set_title("query-label loss (dashed: validation)")
            axes[row, 1].set_title("accuracy (dotted: chance among shown labels)")
            axes[row, 2].set_title("induction score (grey: previous-token)")
        if row == len(tr) - 1:
            for ax in axes[row]:
                ax.set_xlabel("epoch")
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.6)
    fig.savefig(figdir / "fig_training.png", dpi=220)
    plt.close(fig)


def fig_v2(results: Path, figdir: Path):
    """Corrected-protocol runs: test accuracy per generation, one line per seed, for each condition, plus the
    best-head induction score logged at every generation (median over seeds)."""
    conds = [c for c in V2_CONDS if seeds_for(results, c)]
    if not conds:
        return
    style()
    panels = [c for c in ["v2_extended_p1", "v2_control_real", "v2_extended_p1_dups", "v2_extended_p05", "v2_extended_p025", "v2_base_p1"] if c in conds]
    ncol = 3
    nrow = math.ceil((len(panels) + 1) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.9 * nrow), squeeze=False)
    flat = list(axes.ravel())
    for ax, cond in zip(flat, panels):
        runs = load_collapse(results, cond)
        for s in sorted(runs):
            acc = [fnum(r, "test_acc") for r in runs[s]]
            ax.plot(range(len(acc)), acc, color=COLOR[cond], lw=0.8, alpha=0.7, marker="o", ms=1.5)
        med = np.median([[fnum(r, "test_acc") for r in runs[s]] for s in sorted(runs)], axis=0)
        ax.plot(range(len(med)), med, color="black", lw=1.2, ls="--", label="median")
        ax.axhline(0.25, color="gray", ls=":", lw=0.8)
        ax.set_ylim(0, 1.04); ax.set_title(f"{LABEL[cond]} (n={len(runs)})", fontsize=6.5)
    for ax in flat[:len(panels)]:
        ax.set_xlabel("generation"); ax.set_ylabel("test accuracy")
    ax = flat[len(panels)]
    for cond in panels:
        runs = load_collapse(results, cond)
        if not all(runs[s][0].get("induction_score_max", "") != "" for s in runs):
            continue
        med = np.median([[fnum(r, "induction_score_max") for r in runs[s]] for s in sorted(runs)], axis=0)
        ax.plot(range(len(med)), med, color=COLOR[cond], lw=1.0, marker="o", ms=2, label=LABEL[cond].replace("v2 ", ""))
    ax.set_ylim(0, 1.04); ax.set_xlabel("generation"); ax.set_ylabel("best-head induction score (median)")
    ax.legend(frameon=False, fontsize=5, loc="lower left"); ax.set_title("induction score by generation", fontsize=6.5)
    for ax in flat[len(panels) + 1:]:
        ax.axis("off")
    fig.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.7)
    fig.savefig(figdir / "fig_v2.png", dpi=220)
    plt.close(fig)


def fig_main_v2(results: Path, figdir: Path):
    """Main-text figure: test accuracy by generation, one line per seed, for the factorial's two ends, the real-data
    control and the base variant. Seeds whose generation 0 is already weak are drawn dotted."""
    needed = ["v2_extended_p1", "v2_extended_p1_dups", "v2_control_real", "v2_base_p1"]
    if not all(seeds_for(results, c) for c in needed):
        return
    style()
    weak = [s for s in seeds_for(results, "v2_extended_p1") if fnum(load_collapse(results, "v2_extended_p1")[s][0], "test_acc") < 0.9]
    panels = [("v2_extended_p1_dups", "(a) extended, p=1, duplicates kept"), ("v2_extended_p1", "(b) extended, p=1, no duplicates"),
              ("v2_control_real", "(c) real data only (control)"), ("v2_base_p1", "(d) base, p=1, no duplicates")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75), sharey=True)
    for ax, (cond, title) in zip(axes, panels):
        runs = load_collapse(results, cond)
        for s in sorted(runs):
            acc = [fnum(r, "test_acc") for r in runs[s]]
            is_weak = (s in weak) and cond != "v2_base_p1"
            ax.plot(range(len(acc)), acc, color=COLOR[cond], lw=0.9, alpha=0.9 if is_weak else 0.75, ls=":" if is_weak else "-",
                    marker="o", ms=1.4)
        ax.axhline(0.25, color="gray", ls=":", lw=0.7)
        ax.set_title(title, fontsize=6.3)
        ax.set_xlabel("generation")
        ax.set_ylim(0, 1.04)
    axes[0].set_ylabel("test accuracy")
    if weak:
        axes[1].text(9, 0.12, f"dotted: seed {', '.join(map(str, weak))}\n(weak generation 0)", fontsize=5, ha="right", va="center", color="#444")
    fig.tight_layout(pad=0.4, w_pad=0.5)
    fig.savefig(figdir / "fig_main_v2.png", dpi=220)
    plt.close(fig)


def fig_mech_v2(results: Path, figdir: Path):
    """Main-text figure: the induction circuit across generations. (a) classic score (label after an earlier
    occurrence) vs the best score over three offsets; (b) share of accurate models without a classic head; (c) size of the
    ACDC circuit (label corruption) by head family."""
    if not seeds_for(results, "v2_extended_p1"):
        return
    style()
    contr = v2_contrasts(results)
    strong = contr.get("strong_start_seeds", [])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    for cond, label in [("v2_extended_p1", "p=1"), ("v2_control_real", "real data")]:
        if not seeds_for(results, cond):
            continue
        runs = load_collapse(results, cond)
        ss = [s for s in sorted(runs) if s in strong]
        if not ss or runs[ss[0]][0].get("induction_off0_max", "") == "":
            continue
        classic = np.median([[fnum(r, "induction_score_max") for r in runs[s]] for s in ss], axis=0)
        anyoff = np.median([[max(fnum(r, "induction_score_max"), fnum(r, "induction_off0_max"), fnum(r, "induction_off2_max")) for r in runs[s]] for s in ss], axis=0)
        ax.plot(range(len(classic)), classic, color=COLOR[cond], marker="o", ms=2, lw=1.0, label=f"{label}: classic (offset 1)")
        ax.plot(range(len(anyoff)), anyoff, color=COLOR[cond], marker="s", ms=2, lw=1.0, ls="--", label=f"{label}: best of offsets 0-2")
    ax.set_ylim(0, 1.04); ax.set_xlabel("generation"); ax.set_ylabel("best-head score (median)")
    ax.legend(frameon=False, fontsize=4.9, loc="center left", bbox_to_anchor=(0.0, 0.47)); ax.set_title("(a) induction score", fontsize=6.5)

    ax = axes[1]
    share = contr.get("non_classic_share_accurate_gen_ge1_strong_seeds", {})
    order = [c for c in ["v2_control_real", "v2_extended_p025", "v2_extended_p05", "v2_extended_p1", "v2_extended_p1_dups"] if c in share]
    names = {"v2_control_real": "real", "v2_extended_p025": "p=.25", "v2_extended_p05": "p=.5", "v2_extended_p1": "p=1", "v2_extended_p1_dups": "p=1\ndups"}
    if order:
        ax.bar(range(len(order)), [share[c]["share"] for c in order], color=[COLOR[c] for c in order], width=0.65)
        for i, c in enumerate(order):
            ax.text(i, share[c]["share"] + 0.02, f"{share[c]['non_classic']}/{share[c]['n_accurate']}", ha="center", fontsize=5)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([names[c] for c in order], fontsize=5.5)
    ax.set_ylim(0, 0.85); ax.set_ylabel("share without a classic head"); ax.set_title("(b) accurate models, gen. >= 1", fontsize=6.5)

    ax = axes[2]
    fam = acdc_summary(results).get("by_corruption_tau", {}).get("labels@0.01", {}).get("by_family", {})
    fams = [f for f in ["classic", "shifted", "match"] if f in fam]
    if fams:
        ax.bar(range(len(fams)), [fam[f]["mean_live_edges"] for f in fams], color=["#0072B2", "#E69F00", "#CC79A7"][:len(fams)], width=0.6)
        for i, f in enumerate(fams):
            ax.text(i, fam[f]["mean_live_edges"] + 1, f"n={fam[f]['n']}\nacc {fam[f]['mean_circuit_acc']:.3f}", ha="center", fontsize=5)
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, fontsize=6)
        ax.set_ylim(0, max(fam[f]["mean_live_edges"] for f in fams) * 1.3)
    ax.set_ylabel("edges in ACDC circuit (of 110)"); ax.set_title("(c) circuit size by head type", fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=0.7)
    fig.savefig(figdir / "fig_mech_v2.png", dpi=220)
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

    summary = {"sanity_no_collapse": sanity_runs(results), "collapse": collapse_summary(results), "reference": reference_values(results),
               "gen0": gen0_summary(results), "retrain": retrain_summary(results), "retrain_v2": retrain_summary(results, v2=True),
               "circuits": circuit_summary(results), "pool_structure": pool_structure_summary(results)}
    summary["training_gen0"] = training_summary(results)
    summary["factorial"] = factorial_summary(results)
    summary["v2"] = v2_summary(results, summary["collapse"])
    summary["tuning"] = tuning_summary(results)
    summary["v2_contrasts"] = v2_contrasts(results)
    summary["base_failures"] = base_failure_summary(results)
    summary["acdc"] = acdc_summary(results)
    # Marginal symbol KL of the pool a model was trained on, at that model's FIRST failure, in units of the noise floor.
    floor = summary["reference"]["symbol_kl_noise_floor_mean"]
    summary["kl_at_first_failure_over_floor"] = {}
    for cond in ["extended_collapse", "ablation_p05", "ablation_p025", "extended_collapse_nodup"]:
        if cond in summary["collapse"]:
            runs = load_collapse(results, cond)
            vals = {}
            for s in sorted(runs):
                accs = [fnum(r, "test_acc") for r in runs[s]]
                g = next((g for g in range(1, len(accs)) if accs[g] < FAIL_THRESHOLD and accs[0] >= FAIL_THRESHOLD), None)
                if g is not None:
                    vals[s] = round(fnum(runs[s][g], "symbol_kl_vs_gen0") / floor, 2)
            summary["kl_at_first_failure_over_floor"][cond] = {"by_seed": vals, "median": round(float(np.median(list(vals.values()))), 2) if vals else None,
                                                               "max": max(vals.values()) if vals else None}
    with open(results / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # The headline figures need the follow-up outputs (no-duplicates runs, retraining, circuit CSVs); skip them if absent.
    if {"extended_collapse", "extended_collapse_nodup"} <= set(summary["collapse"]) and summary["retrain"]["n_cases"] and summary["circuits"]:
        fig_main(results, figdir)
        fig_pools(results, figdir)
    fig_failure_map(results, figdir, summary["reference"]["symbol_kl_noise_floor_mean"])
    fig_training(results, figdir)
    fig_v2(results, figdir)
    fig_main_v2(results, figdir)
    fig_mech_v2(results, figdir)
    for cond, e in summary["collapse"].items():
        print(f"{LABEL[cond]:24s} seeds={e['seeds']} failed {e['failed_trainings_gen_ge1']}/{e['n_trainings_gen_ge1']} | seeds failing: {e['seeds_with_any_failure']} | "
              f"first failure gen: {e['first_failure_gen_by_seed']} | final acc: {e['final_gen_acc_by_seed']} | recovered: {e['recovered_after_failure_by_seed']}")
        print(f"{'':18s} diag failed={e['diagnostics']['failed']}  ok={e['diagnostics']['ok']}")
        print(f"{'':18s} median epoch to val>0.9: gen0={e['median_epoch_to_val90_gen0']} healthy gen>=1={e['median_epoch_to_val90_healthy_gen_ge1']}")
    print("reference:", json.dumps(summary["reference"], indent=1))
    print("wrote", results / "analysis_summary.json", "and figures in", figdir)


if __name__ == "__main__":
    main()
