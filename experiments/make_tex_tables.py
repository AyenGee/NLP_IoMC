"""Write the report's tables as LaTeX snippets, straight from results/analysis_summary.json, so no table number is typed
by hand.

    python experiments/analyze_results.py        # writes the summary JSON
    python experiments/make_tex_tables.py        # writes abstract/tables/*.tex

abstract/main.tex \\input's these files. Re-run both scripts whenever the results change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COND_NAME = {"v2_extended_p1_dups": "extended, $p{=}1$, duplicates kept", "v2_extended_p1": "extended, $p{=}1$, no duplicates",
             "v2_extended_p05": "extended, $p{=}0.5$", "v2_extended_p025": "extended, $p{=}0.25$",
             "v2_control_real": "real data only (control)"}


def f3(x, d=3):
    return f"{x:.{d}f}"


def write(out: Path, name: str, text: str):
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(text, encoding="utf-8")
    print("wrote", out / name)


def tab_main(S, out):
    V = S["v2_contrasts"]
    share = V["non_classic_share_accurate_gen_ge1_strong_seeds"]
    rows = []
    for c in ["v2_extended_p1_dups", "v2_extended_p1", "v2_extended_p05", "v2_extended_p025", "v2_control_real"]:
        e = V["by_condition"][c]
        sh = share[c]
        rows.append(f"{COND_NAME[c]} & {e['failed_trainings']}/{e['trainings']} & {f3(e['mean_final_gen_acc'])} ({f3(e['min_final_gen_acc'])}) & "
                    f"{sh['non_classic']}/{sh['n_accurate']} \\\\")
    body = "\n".join(rows)
    write(out, "tab_main.tex", f"""\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Training data (7 healthy-start seeds) & Failed & Final-gen.\\ acc. (min) & No classic head\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
""")


def tab_factorial(S, out):
    F = S["factorial"]
    names = {"early_stopping_dups": ("on", "kept"), "early_stopping_nodups": ("on", "removed"),
             "fixed_budget_dups": ("off", "kept"), "fixed_budget_nodups": ("off", "removed")}
    rows = []
    for key, (es, dup) in names.items():
        a, b = F["seeds_0_2"].get(key), F["all_seeds"].get(key)
        cell_a = f"{a['failed']}/{a['trainings']}" if a else "--"
        cell_b = f"{b['failed']}/{b['trainings']} ({b['ci95'][0]:.2f}--{b['ci95'][1]:.2f})" if b and b.get("ci95") else "--"
        rows.append(f"{es} & {dup} & {cell_a} & {cell_b} \\\\")
    write(out, "tab_factorial.tex", "\\begin{tabular}{@{}llcc@{}}\n\\toprule\nEarly stopping & Duplicates & Seeds 0--2 & All seeds (95\\% CI)\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _retrain_rows(cases, tag_orig=None):
    rows = []
    short = {"extended_collapse": "ext.\\ $p{=}1$", "ablation_p05": "ext.\\ $p{=}0.5$", "ablation_p025": "ext.\\ $p{=}0.25$", "base_collapse": "base $p{=}1$",
             "v2_extended_p1": "ext.\\ $p{=}1$", "v2_extended_p1_dups": "ext.\\ $p{=}1$, dups", "v2_extended_p05": "ext.\\ $p{=}0.5$",
             "v2_extended_p025": "ext.\\ $p{=}0.25$", "v2_base_p1": "base $p{=}1$"}
    for c in cases:
        b = c["budgets"]
        keys = sorted(b, key=lambda k: int(k[2:]))
        cells = []
        for k in keys:
            e = b[k]["epoch_first_val_gt_0.9"]
            cells.append(f"{b[k]['test_acc']:.3f} ({'--' if e is None else e})")
        rows.append(f"{short.get(c['cond'], c['cond'])}, s{c['seed']}, g{c['gen']} & {c['original_test_acc']:.3f} & " + " & ".join(cells) + " \\\\")
    return rows


def tab_retrain(S, out):
    rows = _retrain_rows(S["retrain"]["cases"])
    write(out, "tab_retrain_orig.tex", "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nRun & Original & No early stop, 80/100 epochs & No early stop, 300 epochs\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rows = _retrain_rows(S["retrain_v2"]["cases"])
    write(out, "tab_retrain_v2.tex", "\\begin{tabular}{@{}lcc@{}}\n\\toprule\nRun & Original (150 epochs) & Retrained, 450 epochs\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _setting_label(c, default):
    if c["is_default"]:
        return "lr $10^{-3}$, wd 0.1 (default)"
    if (c["d_model"], c["n_heads"], c["batch_size"]) == (default["d_model"], default["n_heads"], default["batch_size"]):
        return f"lr {c['lr']:g}, wd {c['weight_decay']:g}"
    if c["d_model"] != default["d_model"]:
        return f"$d_\\mathrm{{model}}{{=}}{c['d_model']}$"
    if c["n_heads"] != default["n_heads"]:
        return f"{c['n_heads']} heads"
    return f"batch {c['batch_size']}"


def tab_tuning(S, out):
    T = S["tuning"]["variants"]
    default = {"d_model": 64, "n_heads": 4, "batch_size": 128}
    by = {v: {} for v in T}
    for v in T:
        for c in T[v]["cells"]:
            by[v][_setting_label(c, default)] = c
    labels = [l for l in by["extended"]]
    labels.sort(key=lambda l: (not l.startswith("lr"), l))
    rows = []
    for l in labels:
        cells = []
        for v in ["base", "extended"]:
            c = by[v].get(l)
            e = "--" if c is None or c["median_epoch_to_val0.9"] is None else f"{c['median_epoch_to_val0.9']:g}"
            cells.append("-- & --" if c is None else f"{c['mean_val_acc']:.3f} ({c['n_seeds_val_ge_0.9']}/{c['n_seeds']}) & {e}")
        rows.append(f"{l} & " + " & ".join(cells) + " \\\\")
    write(out, "tab_tuning.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{base} & \\multicolumn{2}{c}{extended}\\\\\nSetting & val.\\ acc.\\ (seeds $\\geq$0.9) & epochs to 0.9 & val.\\ acc.\\ (seeds $\\geq$0.9) & epochs to 0.9\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_gen0(S, out):
    """Generation-0 models (real data only), over the seeds that learned the task (test accuracy >= 0.9); the
    others are counted, not averaged in (a failed model's perplexity is not informative)."""
    import numpy as np
    ref = S["reference"]
    rows = []
    for v, t in S["training_gen0"].items():
        ok = [d for d in t["per_seed"].values() if d["test_acc"] >= 0.9]
        n = len(t["per_seed"])
        g = lambda k, d=3: f"{np.mean([x[k] for x in ok]):.{d}f} $\\pm$ {np.std([x[k] for x in ok]):.{d}f}"
        aux = (f"{np.mean([x['test_aux_label_acc'] for x in ok]):.3f} / {np.mean([x['test_aux_symbol_acc'] for x in ok]):.3f}" if "test_aux_label_acc" in ok[0] else "--")
        rows.append(f"{v} & {len(ok)}/{n} & {g('final_train_acc')} & {g('final_val_acc')} & {g('test_acc')} & {g('test_ppl', 3)} & {aux} \\\\")
    head = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nVariant & Learned & Train acc. & Val.\\ acc. & Test acc. & Test ppl. & Aux.\\ label / symbol acc.\\\\\n\\midrule\n"
    bayes = f"\\midrule\nBayes optimum (aux.) & & & & & & {ref['bayes_optimal_aux_label_acc']:.3f} / {ref['bayes_optimal_aux_symbol_acc']:.3f} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    write(out, "tab_gen0.tex", head + "\n".join(rows) + "\n" + bayes)


def tab_base(S, out):
    B = S["base_failures"]["v2_base_p1"]["per_seed"]
    rows = []
    for seed in sorted(B, key=int):
        v = B[seed]
        if v["first_failure_gen"] is None:
            rows.append(f"{seed} & none & -- & -- & -- \\\\")
        else:
            rec = "yes" if v.get("recovered_later") else "no"
            nxt = "--" if v.get("pool_label_ok_next_gen") is None else f"{v['pool_label_ok_next_gen']:.3f}"
            rows.append(f"{seed} & {v['first_failure_gen']} & {v['pool_label_ok_at_first_failure']:.4f} & {nxt} & {rec} \\\\")
    head = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nSeed & First failing gen. & Labels correct in its pool & in the next pool & Recovers later\\\\\n\\midrule\n"
    write(out, "tab_base.tex", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "analysis_summary.json"))
    ap.add_argument("--out", default=str(ROOT / "abstract" / "tables"))
    args = ap.parse_args()
    S = json.load(open(args.summary))
    out = Path(args.out)
    for fn in (tab_main, tab_factorial, tab_retrain, tab_tuning, tab_gen0, tab_base):
        try:
            fn(S, out)
        except Exception as e:      # the results this table needs are not in the summary (yet); say so, do not stop
            print(f"SKIPPED {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
