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

# (d) offset-resolved induction score on hand-built attention maps
from metrics import induction_score, induction_offset_scores, best_head_by_offset
sy = torch.tensor([[3, 5, 3, 7, 3]])             # context symbols 3,5,3,7 (m=4); the query symbol is 3: earlier occurrences at context index 0 and 2
T_ = 2 * 4 + 1
def attn_on(positions):
    a = torch.zeros(1, 1, T_, T_); a[0, 0, T_ - 1, positions] = 1.0 / len(positions); return [a]
p0 = [0, 4]; p1 = [1, 5]; p2 = [2, 6]            # symbol positions 2j, the labels after them 2j+1, the next symbols 2j+2
s0, s1, s2 = (induction_offset_scores(attn_on(x), sy) for x in (p0, p1, p2))
check("offset scores: attention on the earlier occurrences is offset 0 only", abs(s0["layer0_head0_off0"] - 1) < 1e-6 and s0["layer0_head0_off1"] == 0 and s0["layer0_head0_off2"] == 0)
check("offset scores: attention on the labels after them is offset 1 (= induction_score)", abs(s1["layer0_head0_off1"] - 1) < 1e-6 and abs(induction_score(attn_on(p1), sy)["layer0_head0"] - 1) < 1e-6)
check("offset scores: attention on the next symbols is offset 2", abs(s2["layer0_head0_off2"] - 1) < 1e-6 and s2["layer0_head0_off0"] == 0)
check("best_head_by_offset picks the head and score", best_head_by_offset(s1)[1] == ("layer0_head0", s1["layer0_head0_off1"]))

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
    "v2p1": variant("v2_extended_p1", collapse__with_replacement=False),
    "v2dups": variant("v2_extended_p1_dups", collapse__with_replacement=True),
    "v2base": variant("v2_base_p1", train__extended=False, collapse__variant="base", collapse__with_replacement=False),
    "v2ctl": variant("v2_control_real", collapse__p_synthetic=0.0, collapse__with_replacement=False),
}
def run(args):
    r = subprocess.run([PY, str(ROOT / "experiments" / "run_experiment.py"), *args, "--results-root", str(RES)], capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0: print(r.stdout[-800:], r.stderr[-1500:])
    return r.returncode
jobs = [("ext", 0, None), ("ext", 1, 1), ("base", 0, None), ("base", 1, 1), ("resampled", 0, None), ("fresh", 0, None), ("nodup", 0, None),
        ("v2p1", 0, 0), ("v2p1", 1, 1), ("v2dups", 0, 0), ("v2base", 0, 0), ("v2ctl", 0, 0)]
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

# ---------------------------------------------------------------- shipped configs and the manifest
sys.path.insert(0, str(ROOT / "experiments"))
from utils import load_config
from model import ModelConfig
from train import TrainConfig
from collapse import CollapseConfig
from manifest import GROUPS, CONFIG_DIR
bad = []
for p_ in sorted((ROOT / "configs").glob("*.yaml")):
    c_ = load_config(p_)
    try:
        TaskConfig(**c_["task"]); ModelConfig(**c_["model"]); TrainConfig(**c_["train"])
        if "collapse" in c_: CollapseConfig(**c_["collapse"])
    except Exception as e_:
        bad.append((p_.name, str(e_)))
check("every config in configs/ loads into its dataclasses", not bad, str(bad))
check("every manifest job points at an existing config", all((CONFIG_DIR / j.config).exists() for g_ in GROUPS.values() for j in g_))
v2 = [load_config(CONFIG_DIR / j.config) for j in GROUPS["v2"]]
check("v2 group: fixed 150-epoch budget with early stopping OFF", all(c_["train"]["patience"] is None and c_["train"]["n_epochs"] == 150 for c_ in v2))
check("v2 group: duplicates only in the deliberate '_dups' condition", all((c_["collapse"]["with_replacement"] is True) == c_["run_name"].endswith("_dups") for c_ in v2))
check("v2 group: 48 jobs (6 conditions x 8 seeds), each seed with its own real-data pool", len(GROUPS["v2"]) == 48 and all(j.data_seed == j.seed for j in GROUPS["v2"]))
check("v2 group: the real-data control uses no synthetic data, no early stopping, no duplicates", any(c_["run_name"] == "v2_control_real" and c_["collapse"]["p_synthetic"] == 0.0 and c_["train"]["patience"] is None and c_["collapse"]["with_replacement"] is False for c_ in v2))
check("v2 group: the original 40 job indices are unchanged by adding the control at the end", [j.config for j in GROUPS["v2"][:40]] == [c for c in ["v2_extended_p1.yaml", "v2_extended_p05.yaml", "v2_extended_p025.yaml", "v2_base_p1.yaml", "v2_extended_p1_dups.yaml"] for _ in range(8)])
tmp_cfg = SP / "regen_configs"; tmp_cfg.mkdir()
r_ = subprocess.run([PY, str(ROOT / "experiments" / "make_v2_configs.py"), "--out", str(tmp_cfg)], capture_output=True, text=True, cwd=str(ROOT))
same = r_.returncode == 0 and all(yaml.safe_load(open(tmp_cfg / j.config)) == yaml.safe_load(open(CONFIG_DIR / j.config)) for j in GROUPS["v2"])
check("make_v2_configs regenerates exactly the committed v2 configs", same)
r_ = subprocess.run([PY, str(ROOT / "experiments" / "make_v2_configs.py"), "--out", str(tmp_cfg), "--set", "train.lr=0.003", "--set", "model.d_model=128"], capture_output=True, text=True, cwd=str(ROOT))
c_ = yaml.safe_load(open(tmp_cfg / "v2_extended_p1.yaml")); hdr = open(tmp_cfg / "v2_extended_p1.yaml").read()
check("make_v2_configs --set applies overrides to every config and records them in the header", r_.returncode == 0 and c_["train"]["lr"] == 0.003 and c_["model"]["d_model"] == 128 and "train.lr=0.003" in hdr)

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
r2 = subprocess.run([PY, str(ROOT / "experiments" / "analyze_circuits.py"), "--list-groups"], capture_output=True, text=True, cwd=str(ROOT))
r3 = subprocess.run([PY, str(ROOT / "experiments" / "analyze_circuits.py"), "--group-id", "99"], capture_output=True, text=True, cwd=str(ROOT))
r4 = subprocess.run([PY, str(ROOT / "experiments" / "analyze_circuits.py"), "--results", str(RES), "--group-id", "10", "--n-eval", "50"], capture_output=True, text=True, cwd=str(ROOT))   # v2_control_real exists in the tiny results
check("analyze_circuits: --list-groups, a bad group id is rejected, and a group id analyses exactly that condition", r2.returncode == 0 and "v2_extended_p1" in r2.stdout and r3.returncode != 0 and r4.returncode == 0 and (RES / "analysis" / "circuits_v2_control_real_seed0.csv").exists() and not (RES / "analysis" / "circuits_v2_base_p1_seed0.csv").exists(), r4.stderr[-300:])
cc = list(csv.DictReader(open(RES / "analysis" / "circuits_extended_collapse_seed0.csv")))
check("analyze_circuits: one row per generation with circuit + structure columns", len(cc) == 3 and all(k in cc[0] for k in ["induction_max", "induction_best_head", "prev_token_max", "ablate_max_single_delta_acc", "layer1_all_heads_delta_acc", "pool_frac_fully_valid"]))
check("analyze_circuits writes the offset-resolved columns, and offset 1 equals the classic score", all(k in cc[0] for k in ["ind_off0_max", "ind_off1_max", "ind_off2_max", "ind_off1_head"]) and all(abs(float(r["ind_off1_max"]) - float(r["induction_max"])) < 1e-6 for r in cc))
check("generations.csv logs the offset-resolved best-head scores during the run", all(k in ext[0] for k in ["induction_off0_max", "induction_off2_max"]))
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

# ---------------------------------------------------------------- qualitative figures from checkpoints
r = subprocess.run([PY, str(ROOT / "experiments" / "make_figures.py"), "--run", "extended_collapse_seed0", "--results", str(RES), "--figdir", str(SP / "figs_mf")], capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0: print(r.stdout[-600:], r.stderr[-1500:])
check("make_figures runs on a given run and writes attention maps, head profile and PCA", r.returncode == 0 and all((SP / "figs_mf" / f).exists() for f in ["attn_extended_collapse_seed0_gen0.png", "attn_extended_collapse_seed0_gen2.png", "head_profile_extended_collapse_seed0.png", "embedding_pca_extended_collapse_seed0.png"]), r.stderr[-300:])

# ---------------------------------------------------------------- ACDC (edge-level circuit discovery)
from acdc import run_graph, all_edges, acdc, describe, live_subgraph
from model import InductionTransformer
from data import collate_batch
from interp import load_checkpoint
torch.manual_seed(0)
rm = InductionTransformer(tc, ModelConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32)).eval()
for q_ in rm.parameters():                      # larger weights so the check is not trivially satisfied by a near-zero model
    torch.nn.init.normal_(q_, 0, 0.3) if q_.dim() > 1 else torch.nn.init.normal_(q_, 0, 0.1)
dsa, dsb = SymbolicICLDataset(tc, 48, seed=11), SymbolicICLDataset(tc, 48, seed=12)
ba = collate_batch([dsa[i] for i in range(48)]); bb = collate_batch([dsb[i] for i in range(48)])
lg_full, _ = run_graph(rm, ba["symbol_tokens"], ba["label_tokens"])
check("ACDC graph with every edge kept reproduces the model's own forward pass", (lg_full - rm(ba["symbol_tokens"], ba["label_tokens"]).label_logits[:, -1]).abs().max() < 1e-4)
_, corr = run_graph(rm, bb["symbol_tokens"], bb["label_tokens"])
edges_rm = all_edges(2, 2)
lg_cut, _ = run_graph(rm, ba["symbol_tokens"], ba["label_tokens"], {e: False for e in edges_rm}, corr)
check("ACDC graph with every edge cut reproduces the model on the corrupted input", (lg_cut - rm(bb["symbol_tokens"], bb["label_tokens"]).label_logits[:, -1]).abs().max() < 1e-4)
check("ACDC edge count for 2 layers x 4 heads is 110", len(all_edges(2, 4)) == 110, str(len(all_edges(2, 4))))
keep_all = acdc(rm, ba["symbol_tokens"], ba["label_tokens"], ba["query_label"], bb["symbol_tokens"], bb["label_tokens"], tau=-1.0)
check("ACDC with a negative threshold keeps everything (circuit == full model)", keep_all.kl < 1e-9 and keep_all.accuracy == keep_all.full_accuracy and all(keep_all.kept.values()))
cut_all = acdc(rm, ba["symbol_tokens"], ba["label_tokens"], ba["query_label"], bb["symbol_tokens"], bb["label_tokens"], tau=1e9)
check("ACDC with a huge threshold removes every edge and leaves an empty circuit", not any(cut_all.kept.values()) and cut_all.n_live_edges == 0)
mid = acdc(rm, ba["symbol_tokens"], ba["label_tokens"], ba["query_label"], bb["symbol_tokens"], bb["label_tokens"], tau=0.05)
check("ACDC circuit size shrinks as the threshold grows", cut_all.n_live_edges <= mid.n_live_edges <= keep_all.n_live_edges, f"{cut_all.n_live_edges}<={mid.n_live_edges}<={keep_all.n_live_edges}")
check("ACDC live edges are a subset of the kept edges", set(mid.live_edges) <= {e for e, v in mid.kept.items() if v})
d_ = describe(mid, 2, 2)
check("describe() reports the expected columns", all(k in d_ for k in ["acdc_n_edges", "acdc_kcomp", "acdc_heads", "acdc_mlp0", "acdc_acc"]))
ck_dir = RES / "collapse" / "extended_collapse_seed0"
m_, _, _, _ = load_checkpoint(ck_dir / "gen1_model.pt")
c_a = acdc(m_, ba["symbol_tokens"], ba["label_tokens"], ba["query_label"], bb["symbol_tokens"], bb["label_tokens"], tau=0.1)
check("ACDC runs on a real checkpoint from a collapse run", 0 <= c_a.accuracy <= 1 and 0 <= c_a.n_live_edges <= len(all_edges(2, 2)))

r = subprocess.run([PY, str(ROOT / "experiments" / "analyze_acdc.py"), "--results", str(RES), "--conds", "extended_collapse", "v2_extended_p1", "--taus", "0.05", "--n-eval", "64"], capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0: print(r.stdout[-600:], r.stderr[-1500:])
check("analyze_acdc runs", r.returncode == 0)
ac = list(csv.DictReader(open(RES / "analysis" / "acdc_extended_collapse_seed0.csv")))
check("analyze_acdc: one row per generation and corruption with circuit columns", len(ac) == 3 * 2 and {r["acdc_corruption"] for r in ac} == {"resample", "labels"} and all(k in ac[0] for k in ["acdc_n_edges", "acdc_kcomp", "acdc_heads", "acdc_full_acc", "acdc_edges"]))
from acdc import relabel_corruption
rl = relabel_corruption(ba["symbol_tokens"], ba["label_tokens"], tc.L, seed=3)
same_sym_same_label = all(len({int(rl[i, j]) for j in range(rl.shape[1]) if ba["symbol_tokens"][i, j] == s_}) == 1 for i in range(10) for s_ in ba["symbol_tokens"][i, :-1].unique())
check("relabel_corruption keeps the symbols consistent (one new label per symbol) and changes the labels", same_sym_same_label and not torch.equal(rl, ba["label_tokens"]))
r = subprocess.run([PY, str(ROOT / "experiments" / "analyze_acdc.py"), "--list-groups"], capture_output=True, text=True, cwd=str(ROOT))
check("analyze_acdc --list-groups lists the groups and a bad group id is rejected", r.returncode == 0 and "v2_extended_p1" in r.stdout and subprocess.run([PY, str(ROOT / "experiments" / "analyze_acdc.py"), "--group-id", "99"], capture_output=True, text=True, cwd=str(ROOT)).returncode != 0)

# ---------------------------------------------------------------- hyper-parameter study
from tune_hparams import CELLS, run_cell, summarize, TUNE_SEEDS
check("tuning grid: 15 cells per variant, 30 in all, with exactly one default per variant", len(CELLS) == 30 and sum(c.is_default for c in CELLS) == 2 and len({c.name for c in CELLS}) == 30)
check("tuning seeds are disjoint from every seed used by the main experiments", not (set(TUNE_SEEDS) & set(range(0, 50))))
TUNE = SP / "tune_res" / "tuning"
for c in CELLS:                                   # every cell must construct, train and log (1 tiny epoch each)
    run_cell(c, TUNE, seeds=[100], epochs=1, n_train=64, n_val=64, n_test=64)
check("every tuning cell trains for a tiny budget", len(list(TUNE.glob("*.json"))) == 30)
sm = summarize(TUNE)
check("tuning summary covers both variants and picks a winner for each", set(sm["variants"]) == {"base", "extended"} and all(v["winner"] and v["n_cells"] == 15 for v in sm["variants"].values()))
check("tuning selection uses validation accuracy only (no test field in the cell tables)", all("test" not in k for v in sm["variants"].values() for r in v["cells"] for k in r))
r = subprocess.run([PY, str(ROOT / "experiments" / "tune_hparams.py"), "--task-id", "99"], capture_output=True, text=True, cwd=str(ROOT))
check("tune_hparams rejects an out-of-range task id", r.returncode != 0 and "out of range" in (r.stderr + r.stdout))

# ---------------------------------------------------------------- statistics helpers and retrain filters
import importlib.util
spec = importlib.util.spec_from_file_location("analyze_results", ROOT / "experiments" / "analyze_results.py")
ar = importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
check("Fisher exact p for a perfectly separated 10 vs 10 table is 2/C(20,10)", abs(ar.fisher_exact_p(10, 0, 0, 10) - 2 / 184756) < 1e-12)
check("Fisher exact p is 1 for identical groups", abs(ar.fisher_exact_p(3, 7, 3, 7) - 1.0) < 1e-9)
w = ar.wilson(0, 10); check("Wilson interval for 0/10 is [0, ~0.28]", w[0] == 0.0 and abs(w[1] - 0.2775) < 0.002, str(w))
w = ar.wilson(5, 10); check("Wilson interval for 5/10 is ~[0.24, 0.76]", abs(w[0] - 0.237) < 0.003 and abs(w[1] - 0.763) < 0.003, str(w))
check("Spearman is 1 for a monotone relation and -1 for a reversed one", ar.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and ar.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0)
import retrain_generation as rg
only_v2, no_v2 = rg.failing_cases(RES, prefix="v2_"), rg.failing_cases(RES, exclude_prefix="v2_")
check("retrain cases can be filtered by prefix", all(c[0].startswith("v2_") for c in only_v2) and not any(c[0].startswith("v2_") for c in no_v2))

# ---------------------------------------------------------------- generalised analysis
r = subprocess.run([PY, str(ROOT / "experiments" / "analyze_results.py"), "--results", str(RES), "--figdir", str(SP / "figs_test")], capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0: print(r.stdout[-600:], r.stderr[-2000:])
check("analyze_results runs on a different results dir with different seeds and the new conditions", r.returncode == 0)
S = json.load(open(RES / "analysis_summary.json"))
check("summary includes controls + nodup + seeds discovered per condition", {"control_real_resampled", "control_real_fresh", "extended_collapse_nodup"} <= set(S["collapse"]) and S["collapse"]["extended_collapse"]["seeds"] == [0, 1])
check("summary reports pool_distinct_frac by generation", "pool_distinct_frac_by_gen_mean" in S["collapse"]["control_real_resampled"])
check("both main figures were written", (SP / "figs_test" / "fig_main.png").exists() and (SP / "figs_test" / "fig_failure_map.png").exists())
check("summary has the v2 conditions, the 2x2 factorial and the generation-0 training table", {"v2_extended_p1", "v2_extended_p1_dups", "v2_base_p1", "v2_control_real"} <= set(S["v2"]) and "all_seeds" in S["factorial"] and set(S["training_gen0"]) == {"base", "extended"})
check("generation-0 training table reports train/val/test accuracy and perplexity", all(k in S["training_gen0"]["extended"]["mean_std_all_seeds"] for k in ["final_train_acc", "final_val_acc", "test_acc", "test_ppl", "final_val_loss"]))
check("ACDC summary appears once ACDC CSVs exist", S["acdc"] and "resample@0.05" in S["acdc"]["by_corruption_tau"] and "labels@0.05" in S["acdc"]["by_corruption_tau"])
check("the v2 figure and the training-curve figure were written", (SP / "figs_test" / "fig_v2.png").exists() and (SP / "figs_test" / "fig_training.png").exists())

# ---------------------------------------------------------------- the real results must be untouched
real_after = snapshot(ROOT / "results")
changed = [a for a in real_before if a not in real_after]
# analysis_summary.json is regenerated by design in an earlier step; everything else must be identical
check("repo results/ was NOT modified by any test", len(real_after) == len(real_before) and not [c for c in changed if "analysis_summary" not in c[0]], str(changed[:3]))

shutil.rmtree(SP, ignore_errors=True)   # the temporary results directory is not needed once the checks have run
print("\n==== FAILED:", FAILS if FAILS else "none ====")
sys.exit(1 if FAILS else 0)
