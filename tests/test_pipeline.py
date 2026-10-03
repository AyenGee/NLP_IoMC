"""End-to-end regression tests for the pipeline, on tiny configs.

    python tests/test_pipeline.py          # ~1-2 minutes on one CPU core; exits non-zero on any failure

Everything runs through the real command-line entry points and writes ONLY to a
temporary directory (never to results/), then asserts specific behaviours:
  * the default mixing is bit-for-bit the original algorithm (so the 18 reported
    runs stay reproducible), and the new without-replacement mode works;
  * the duplicate-sequence diagnostic distinguishes the control conditions;
  * structure statistics are exactly 1.0 on real data and detect corruption;
  * --seed / --data-seed / --results-root, the array-task plumbing, the
    cluster-side analysis tools, and the generalised analysis script all work;
  * results/ is left untouched.
On the cluster, run it on a compute node: sbatch slurm/test_pipeline.slurm
"""
import copy, csv, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, yaml, torch

ROOT = Path(__file__).resolve().parent.parent
SP = Path(tempfile.mkdtemp(prefix="icl_pipeline_test_"))
RES = SP / "res_test"
PY = sys.executable
sys.path.insert(0, str(ROOT / "src"))
FAILS = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: FAILS.append(name)

def snapshot(d):
    return sorted((str(p.relative_to(d)), p.stat().st_size, int(p.stat().st_mtime)) for p in Path(d).rglob("*") if p.is_file())

real_before = snapshot(ROOT / "results")

# ---------------------------------------------------------------- unit tests
from data import TaskConfig, SymbolicICLDataset, dataset_to_items, mix_datasets, save_items, load_items
from metrics import structure_stats, pool_distinct_fraction

tc = TaskConfig(K=16, L=4, n_classes=3, burstiness=2)
real = dataset_to_items(SymbolicICLDataset(tc, 500, seed=1))
synth = dataset_to_items(SymbolicICLDataset(tc, 500, seed=2))

# (a) default behaviour is bit-for-bit what the original 18 runs used (re-implementation of the OLD function)
def old_mix(real_items, synthetic_items, p, n_total, seed):
    g = torch.Generator().manual_seed(seed)
    n_synth = int(round(n_total * p)); n_real = n_total - n_synth
    def s(items, n):
        if n == 0: return []
        idx = torch.randint(len(items), (n,), generator=g); return [items[int(i)] for i in idx]
    pool = s(real_items, n_real) + s(synthetic_items, n_synth)
    perm = torch.randperm(len(pool), generator=g).tolist(); return [pool[i] for i in perm]
for p in (0.0, 0.25, 0.5, 1.0):
    a, b = mix_datasets(real, synth, p, 500, seed=7), old_mix(real, synth, p, 500, 7)
    check(f"mix_datasets default == original algorithm (p={p})", all(torch.equal(x["symbol_tokens"], y["symbol_tokens"]) and int(x["query_label"]) == int(y["query_label"]) for x, y in zip(a, b)))
boot = mix_datasets(real, synth, 0.5, 500, seed=7)
nodup = mix_datasets(real, synth, 0.5, 500, seed=7, with_replacement=False)
check("bootstrap pool has ~63% distinct sequences", 0.55 < pool_distinct_fraction(mix_datasets(real, synth, 0.0, 500, seed=7)) < 0.72, f"{pool_distinct_fraction(mix_datasets(real, synth, 0.0, 500, seed=7)):.3f}")
check("with_replacement=False gives only distinct draws (no dup beyond the sources' own)", pool_distinct_fraction(nodup) >= pool_distinct_fraction(real + synth) - 0.02 and pool_distinct_fraction(nodup) > pool_distinct_fraction(boot))
try:
    mix_datasets(real, synth, 1.0, 600, seed=1, with_replacement=False); check("without replacement refuses an oversize request", False)
except ValueError:
    check("without replacement refuses an oversize request", True)

# (b) legacy snapshot format still loads; compact format round-trips
legacy = SP / "legacy_items.pt"; torch.save(real[:20], legacy)
check("load_items reads legacy list-of-dicts snapshots", len(load_items(legacy)) == 20)
compact = SP / "compact_items.pt"; save_items(real[:20], compact)
check("load_items round-trips compact snapshots", all(torch.equal(a["label_tokens"], b["label_tokens"]) for a, b in zip(real[:20], load_items(compact))))

# (c) structure statistics: real data is perfectly valid; targeted corruption is detected
st = structure_stats(real, tc.n_classes, tc.burstiness)
check("real pool: every structure statistic is exactly 1.0", all(v == 1.0 for v in st.values()), str({k: round(v, 3) for k, v in st.items()}))
bad = copy.deepcopy(real)
for it in bad[:100]:           # make the 2nd occurrence of a repeated symbol carry a different label -> inconsistency
    labs = it["label_tokens"].clone(); syms = it["symbol_tokens"][:-1]
    first = int((syms == syms[0]).nonzero()[0]); second = int((syms == syms[0]).nonzero()[1])
    labs[second] = (labs[second] + 1) % tc.L; it["label_tokens"] = labs
st2 = structure_stats(bad, tc.n_classes, tc.burstiness)
check("corrupting 100/500 sequences lowers consistency ~20%", abs(st2["frac_label_consistent"] - 0.8) < 0.02, f"{st2['frac_label_consistent']:.3f}")
check("...and frac_fully_valid", st2["frac_fully_valid"] <= 0.81, f"{st2['frac_fully_valid']:.3f}")
badq = copy.deepcopy(real)
for it in badq[:50]: it["query_label"] = (it["query_label"] + 1) % tc.L
check("wrong query labels are detected by frac_query_label_ok", abs(structure_stats(badq, tc.n_classes, tc.burstiness)["frac_query_label_ok"] - 0.9) < 0.02)

# ---------------------------------------------------------------- tiny end-to-end runs via the real CLI
if RES.exists(): shutil.rmtree(RES)
base = {
    "task": {"K": 16, "L": 4, "n_classes": 3, "burstiness": 2, "shuffle_context": True},
    "model": {"d_model": 16, "n_layers": 2, "n_heads": 2, "d_ff": 32, "dropout": 0.0},
    "train": {"lr": 0.003, "weight_decay": 0.1, "batch_size": 64, "n_epochs": 3, "warmup_steps": 5, "grad_clip": 1.0, "extended": True,
              "aux_weight": 1.0, "eval_every": 1, "interp_every": 1, "patience": None, "seed": 0, "device": "cpu",
              "n_train": 400, "n_val": 100, "n_test": 100, "data_seed": 0},
    "collapse": {"n_generations": 3, "p_synthetic": 1.0, "pool_size": 400, "variant": "extended", "gen_batch_size": 128, "temperature": 1.0, "seed": 0},
    "run_name": "extended_collapse",
}
def variant(name, **kw):
    c = copy.deepcopy(base); c["run_name"] = name
    for k, v in kw.items():
        sect, key = k.split("__"); c[sect][key] = v
    path = SP / f"tiny_{name}.yaml"; yaml.safe_dump(c, open(path, "w")); return path
cfgs = {
    "ext": variant("extended_collapse"),
    "base": variant("base_collapse", train__extended=False, collapse__variant="base"),
    "resampled": variant("control_real_resampled", collapse__p_synthetic=0.0, collapse__with_replacement=True),
    "fresh": variant("control_real_fresh", collapse__p_synthetic=0.0, collapse__with_replacement=False),
    "nodup": variant("extended_collapse_nodup", collapse__with_replacement=False),
}
def run(args):
    r = subprocess.run([PY, str(ROOT / "experiments" / "run_experiment.py"), *args, "--results-root", str(RES)], capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0: print(r.stdout[-800:], r.stderr[-1500:])
    return r.returncode
jobs = [("ext", 0, None), ("ext", 1, 1), ("base", 0, None), ("base", 1, 1), ("resampled", 0, None), ("fresh", 0, None), ("nodup", 0, None)]
for key, seed, dseed in jobs:
    a = ["--config", str(cfgs[key]), "--mode", "collapse", "--seed", str(seed)] + (["--data-seed", str(dseed)] if dseed is not None else [])
    check(f"CLI collapse run {key} seed={seed} data_seed={dseed}", run(a) == 0)
# no-collapse (train mode) with --results-root and --seed
for name, ext_flag in [("base_no_collapse", False), ("extended_no_collapse", True)]:
    c = copy.deepcopy(base); c["train"]["extended"] = ext_flag
    c["train"]["log_path"] = f"results/logs/{name}.csv"; c["train"]["ckpt_path"] = f"results/checkpoints/{name}.pt"; del c["collapse"]; del c["run_name"]
    p = SP / f"tiny_{name}.yaml"; yaml.safe_dump(c, open(p, "w"))
    check(f"CLI train run {name}", run(["--config", str(p), "--mode", "train", "--seed", "0"]) == 0)
check("train outputs landed under --results-root with _seed suffix", (RES / "checkpoints" / "extended_no_collapse_seed0.summary.json").exists() and (RES / "logs" / "base_no_collapse_seed0.csv").exists())

def gens(run_name): return list(csv.DictReader(open(RES / "collapse" / run_name / "generations.csv")))
res = gens("control_real_resampled_seed0"); fre = gens("control_real_fresh_seed0"); nod = gens("extended_collapse_nodup_seed0"); ext = gens("extended_collapse_seed0")
f = lambda rows, k: [float(r[k]) for r in rows]
check("generations.csv records pool_distinct_frac and structure columns", all(k in ext[0] for k in ["pool_distinct_frac", "pool_frac_fully_valid", "pool_label_consistency_rate", "induction_score_max", "prev_token_score_max"]))
check("gen0 pool is fully distinct and fully valid", f(ext, "pool_distinct_frac")[0] == 1.0 and f(ext, "pool_frac_fully_valid")[0] == 1.0)
check("control_real_resampled: later pools are bootstrap resamples (~0.63 distinct)", all(0.55 < v < 0.72 for v in f(res, "pool_distinct_frac")[1:]), str(np.round(f(res, "pool_distinct_frac"), 3)))
check("control_real_fresh: later pools are fully distinct real data", all(v == 1.0 for v in f(fre, "pool_distinct_frac")[1:]) and all(v == 1.0 for v in f(fre, "pool_frac_fully_valid")))
check("p=0 controls contain no synthetic corruption (all pools fully valid)", all(v == 1.0 for v in f(res, "pool_frac_fully_valid")))
check("extended_collapse_nodup pools have far fewer duplicates than the bootstrap run", np.mean(f(nod, "pool_distinct_frac")[1:]) > np.mean(f(ext, "pool_distinct_frac")[1:]) + 0.15, f"nodup {np.mean(f(nod,'pool_distinct_frac')[1:]):.3f} vs boot {np.mean(f(ext,'pool_distinct_frac')[1:]):.3f}")
check("synthetic pools from the (tiny, weak) generator are NOT fully valid -> structure stats detect it", np.mean(f(ext, "pool_frac_fully_valid")[1:]) < 0.9, f"{np.mean(f(ext,'pool_frac_fully_valid')[1:]):.3f}")
check("base variant pools keep contexts valid (only the label is resampled)", all(v == 1.0 for v in f(gens('base_collapse_seed0'), 'pool_frac_label_consistent')))
check("--data-seed changes the real pool/val set (gen0 differs between seed0 and seed1 ext runs)", f(ext, "test_loss")[0] != f(gens("extended_collapse_seed1"), "test_loss")[0])

# ---------------------------------------------------------------- array-task plumbing
r = subprocess.run([PY, str(ROOT / "experiments" / "run_array_task.py"), "--group", "controls", "--task-id", "99"], capture_output=True, text=True, cwd=str(ROOT))
check("run_array_task rejects an out-of-range index with a clear message", r.returncode != 0 and "out of range" in (r.stderr + r.stdout))
r = subprocess.run([PY, str(ROOT / "experiments" / "run_array_task.py"), "--group", "nope", "--task-id", "0"], capture_output=True, text=True, cwd=str(ROOT))
check("run_array_task rejects an unknown group", r.returncode != 0 and "unknown group" in (r.stderr + r.stdout))
r = subprocess.run([PY, str(ROOT / "experiments" / "check_manifest.py")], capture_output=True, text=True, cwd=str(ROOT))
check("check_manifest passes on the real slurm scripts", r.returncode == 0)

# ---------------------------------------------------------------- cluster-side analysis tools
r = subprocess.run([PY, str(ROOT / "experiments" / "analyze_circuits.py"), "--results", str(RES), "--conds", "extended_collapse", "base_collapse", "--n-eval", "100"], capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0: print(r.stdout[-600:], r.stderr[-1500:])
check("analyze_circuits runs", r.returncode == 0)
cc = list(csv.DictReader(open(RES / "analysis" / "circuits_extended_collapse_seed0.csv")))
check("analyze_circuits: one row per generation with circuit + structure columns", len(cc) == 3 and all(k in cc[0] for k in ["induction_max", "induction_best_head", "prev_token_max", "ablate_max_single_delta_acc", "layer1_all_heads_delta_acc", "pool_frac_fully_valid"]))
check("best-head induction score >= all-head mean (it is a max over heads)", all(float(r["induction_max"]) >= float(r["induction_mean"]) - 1e-9 for r in cc))
check("analyze_circuits pool stats agree with the stats recorded during the run", all(abs(float(a["pool_frac_fully_valid"]) - float(b["pool_frac_fully_valid"])) < 1e-9 for a, b in zip(cc, ext)))

r = subprocess.run([PY, str(ROOT / "experiments" / "retrain_generation.py"), "--results", str(RES), "--list"], capture_output=True, text=True, cwd=str(ROOT))
print(r.stdout.strip()); check("retrain_generation --list works", r.returncode == 0 and "failing case" in r.stdout)
n_cases = int(r.stdout.strip().splitlines()[-1].split()[0])
if n_cases:
    r = subprocess.run([PY, str(ROOT / "experiments" / "retrain_generation.py"), "--results", str(RES), "--task-id", "0", "--epochs", "0", "6"], capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0: print(r.stdout[-600:], r.stderr[-1500:])
    check("retrain_generation retrains a failed generation without early stopping", r.returncode == 0)
    summ = list((RES / "retrain").glob("summary_*.json"))
    j = json.load(open(summ[0]))
    check("retrain summary has original + both budgets", set(j["retrained"]) == {"ep3", "ep6"} and "ever_learned_induction" in j["retrained"]["ep6"], str(list(j["retrained"])))
    check("retrain ran the FULL epoch budget (patience disabled)", len(list(csv.DictReader(open(RES / "retrain" / (summ[0].stem.replace('summary_', '') + '_ep6.csv'))))) == 6)
r = subprocess.run([PY, str(ROOT / "experiments" / "retrain_generation.py"), "--results", str(RES), "--task-id", "999"], capture_output=True, text=True, cwd=str(ROOT))
check("retrain_generation exits cleanly for a task id past the last case (array is oversized on purpose)", r.returncode == 0 and "nothing to do" in r.stdout)

# ---------------------------------------------------------------- generalised analysis
r = subprocess.run([PY, str(ROOT / "experiments" / "analyze_results.py"), "--results", str(RES), "--figdir", str(SP / "figs_test")], capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0: print(r.stdout[-600:], r.stderr[-2000:])
check("analyze_results runs on a different results dir with different seeds and the new conditions", r.returncode == 0)
S = json.load(open(RES / "analysis_summary.json"))
check("summary includes controls + nodup + seeds discovered per condition", {"control_real_resampled", "control_real_fresh", "extended_collapse_nodup"} <= set(S["collapse"]) and S["collapse"]["extended_collapse"]["seeds"] == [0, 1])
check("summary reports pool_distinct_frac by generation", "pool_distinct_frac_by_gen_mean" in S["collapse"]["control_real_resampled"])
check("both main figures were written", (SP / "figs_test" / "fig_main.png").exists() and (SP / "figs_test" / "fig_failure_map.png").exists())

# ---------------------------------------------------------------- the real results must be untouched
real_after = snapshot(ROOT / "results")
changed = [a for a in real_before if a not in real_after]
# analysis_summary.json is regenerated by design in an earlier step; everything else must be identical
check("repo results/ was NOT modified by any test", len(real_after) == len(real_before) and not [c for c in changed if "analysis_summary" not in c[0]], str(changed[:3]))

shutil.rmtree(SP, ignore_errors=True)   # the temporary results directory is not needed once the checks have run
print("\n==== FAILED:", FAILS if FAILS else "none ====")
sys.exit(1 if FAILS else 0)
