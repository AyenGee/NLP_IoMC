"""Tests for the exact Singh et al. (2024) implementation (data, model, metrics, training).

    python tests/test_exact.py        # about a minute on one CPU core; the last line must be "FAILED: none"

Writes only to a temporary directory. The .h5 embeddings file must be in data/ (see README.txt).
"""
import csv, math, sys, tempfile, shutil
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
TMP = Path(tempfile.mkdtemp(prefix="exact_test_"))
FAILS = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: FAILS.append(name)

import data as D
import metrics as M
from model import IHTransformer, ModelConfig, rope_tables, apply_rope
from train import TrainConfig, train_model, EvalSets, evaluate

torch.set_num_threads(1)
feats = D.OmniglotFeatures().tensor

# ------------------------------------------------------------------ the embeddings file and the splits
check("features have the documented shape and dtype (1623, 5, 512) float32", tuple(feats.shape) == (1623, 5, 512) and feats.dtype == torch.float32)
check("train classes 0..49, test classes the last 100 (1523..1622), disjoint", list(D.TRAIN_CLASSES) == list(range(50)) and list(D.TEST_CLASSES) == list(range(1523, 1623)) and not set(D.TRAIN_CLASSES) & set(D.TEST_CLASSES))
check("validation exemplars 1..4 never include the training exemplar 0", set(D.VAL_EXEMPLARS.tolist()) == {1, 2, 3, 4} and 0 not in D.VAL_EXEMPLARS and list(D.TRAIN_EXEMPLARS) == [0])
trp, tep = D.label_pair_split()
as_set = lambda a: {tuple(sorted(p)) for p in a.tolist()}
check("label pairs: 8 train + 2 held out = the 10 unordered pairs over 0..4, disjoint", len(trp) == 8 and len(tep) == 2 and not as_set(trp) & as_set(tep) and len(as_set(trp) | as_set(tep)) == 10)
check("the label-pair split is deterministic", np.array_equal(D.label_pair_split()[0], trp))

# ------------------------------------------------------------------ the enumerated training space
base = D.enumerate_train_space(False)
ext = D.enumerate_train_space(True)
check("enumerated training space is exactly 78,400 sequences (base) and 156,800 (extended)", len(base) == 78400 and len(ext) == 156800, f"{len(base)}, {len(ext)}")
check("every enumerated row is distinct", len({tuple(r) for r in base.tolist()}) == 78400 and len({tuple(r) for r in ext.tolist()}) == 156800)
check("training rows use only train classes, exemplar 0 and the 8 train label pairs", base[:, [D.C_A, D.C_B]].max() < 50 and not base[:, [D.EX_A, D.EX_B, D.EX_Q]].any() and all(tuple(sorted((a, b))) in as_set(trp) for a, b in base[:, [D.L_A, D.L_B]].tolist()))
check("A and B are always different classes", bool((base[:, D.C_A] != base[:, D.C_B]).all()))
check("each ordered class pair occurs 8 pairs x 2 assignments x 2 query choices = 32 times", set(np.unique(base[:, D.C_A] * 50 + base[:, D.C_B], return_counts=True)[1].tolist()) == {32})
check("extended space = every base episode with each of the two S2 choices", int(ext[:, D.S2_IS_A].sum()) == 78400 and np.array_equal(ext[:78400, :8], ext[78400:, :8]))

# ------------------------------------------------------------------ evaluation sets: the train / validation / test split
ev = D.build_eval_sets()
check("five evaluation sets of 1,000 sequences each", set(ev) == set(D.EVAL_SET_NAMES) and all(v.shape == (1000, D.N_COLS) for v in ev.values()))
check("evaluation sets are deterministic (rebuilt identically from the fixed seed)", all(np.array_equal(ev[k], D.build_eval_sets()[k]) for k in ev))
p = TMP / "ev.npz"; D.save_eval_sets(p, ev); back = D.load_or_build_eval_sets(p)
check("evaluation sets round-trip through the file on disk", all(np.array_equal(ev[k], back[k]) for k in ev))
test_classes = set(D.TEST_CLASSES.tolist())
check("no test class appears in training: test_exemplars and test_class_ex0 use only test classes", all(set(ev[k][:, [D.C_A, D.C_B]].ravel().tolist()) <= test_classes for k in ("test_exemplars", "test_class_ex0")))
check("train_eval / val / test_relabel use only training classes", all(ev[k][:, [D.C_A, D.C_B]].max() < 50 for k in ("train_eval", "val", "test_relabel")))
check("no held-out label pair appears in the training space", not {tuple(sorted(p)) for p in base[:, [D.L_A, D.L_B]].tolist()} & as_set(tep))
check("test_relabel uses ONLY the held-out pairs; the other sets use only training pairs", all(tuple(sorted(p)) in as_set(tep) for p in ev["test_relabel"][:, [D.L_A, D.L_B]].tolist()) and all(tuple(sorted(p)) in as_set(trp) for k in ("train_eval", "val", "test_exemplars", "test_class_ex0") for p in ev[k][:, [D.L_A, D.L_B]].tolist()))
check("validation exemplars (1..4) are never training exemplars (0), for A, B, Q and S2", set(np.unique(ev["val"][:, [D.EX_A, D.EX_B, D.EX_Q, D.EX_S2]]).tolist()) <= {1, 2, 3, 4} and not ev["val"][:, [D.EX_A, D.EX_B, D.EX_Q]].min() == 0)
check("test_exemplars draws exemplars from all of 0..4 (and test_class_ex0 only from 0)", set(np.unique(ev["test_exemplars"][:, [D.EX_A, D.EX_B, D.EX_Q]]).tolist()) == {0, 1, 2, 3, 4} and not ev["test_class_ex0"][:, [D.EX_A, D.EX_B, D.EX_Q]].any())
train_keys = {tuple(r[:8]) for r in base.tolist()}
check("no validation / test sequence is a training sequence (rows compared on the first 8 columns)", not any(tuple(r[:8]) in train_keys for k in ("val", "test_exemplars", "test_relabel", "test_class_ex0") for r in ev[k].tolist()))
check("train_eval sequences are drawn from the training space", all(tuple(r[:8]) in train_keys for r in ev["train_eval"].tolist()))

# ------------------------------------------------------------------ sequence semantics
rows = torch.as_tensor(base[:2000])
b = D.collate(rows, feats, False)
qa = rows[:, D.Q_IS_A].bool()
check("base batch: shapes ex (B,3,512), labels (B,2)", tuple(b.ex.shape) == (2000, 3, 512) and tuple(b.labels.shape) == (2000, 2))
same_as_A = (b.ex[:, 2] == b.ex[:, 0]).all(-1); same_as_B = (b.ex[:, 2] == b.ex[:, 1]).all(-1)
check("with E=1 the query is the same exemplar as the one it matches (and not the other)", bool((torch.where(qa, same_as_A, same_as_B)).all()) and not bool(torch.where(qa, same_as_B, same_as_A).any()))
check("the target is the in-context label of the query's class", bool((b.query_label == torch.where(qa, b.labels[:, 0], b.labels[:, 1])).all()))
be = D.collate(torch.as_tensor(ext[::37]), feats, True)
c2 = D.s2_class(torch.as_tensor(ext[::37])); r_ = torch.as_tensor(ext[::37])
check("extended batch: shapes ex (B,4,512), labels (B,3); lq is teacher-forced as the third label", tuple(be.ex.shape)[1:] == (4, 512) and tuple(be.labels.shape)[1:] == (3,) and bool((be.labels[:, 2] == be.query_label).all()))
check("extended S2 is one of the episode's two classes and l_S2 is its in-context label", bool(((c2 == r_[:, D.C_A]) | (c2 == r_[:, D.C_B])).all()) and bool((be.s2_label == torch.where(r_[:, D.S2_IS_A] == 1, r_[:, D.L_A], r_[:, D.L_B])).all()))
check("Bayes-optimal accuracies: S2 can be named only half the time, lq and l_S2 always", M.BAYES_OPTIMAL == {"acc_q": 1.0, "acc_s2": 0.5, "acc_ls2": 1.0} and abs(float(ext[:, D.S2_IS_A].mean()) - 0.5) < 1e-9)
ev_ext = D.collate(torch.as_tensor(ev["test_exemplars"]), feats, True)
check("S2 targets outside the 50 training classes are marked -1 (ignored in the loss)", bool((ev_ext.s2_class == -1).all()))

# ------------------------------------------------------------------ model: architecture
torch.manual_seed(0)
mb, me = IHTransformer(ModelConfig()), IHTransformer(ModelConfig(extended=True))
check("attention-only parameter count: base 66,629 and extended 69,879", mb.n_parameters() == 66629 and me.n_parameters() == 69879, f"{mb.n_parameters()}, {me.n_parameters()}")
check("no MLP and no absolute positional embedding", not any("mlp" in n or "pos" in n for n, _ in mb.named_parameters()) and len(mb.blocks) == 2)
blk = mb.blocks[0]
check("qkv Linear(64,192) and proj Linear(64,64), both bias-free; 8 heads of dimension 8", blk.qkv.weight.shape == (192, 64) and blk.qkv.bias is None and blk.proj.weight.shape == (64, 64) and blk.proj.bias is None and blk.cfg.n_heads == 8 and blk.head_dim == 8)
check("input embedding is Linear(512,64) with bias; labels Embedding(5,64); heads Linear(64,5) / Linear(64,50) with bias", mb.example_embed.weight.shape == (64, 512) and mb.example_embed.bias is not None and mb.label_embed.weight.shape == (5, 64) and mb.label_head.weight.shape == (5, 64) and me.symbol_head.weight.shape == (50, 64) and me.symbol_head.bias is not None)
check("initialisation: Linear std ~ 1/sqrt(fan_in), biases 0, label embedding std ~ 0.02 x 0.88", abs(mb.example_embed.weight.std().item() * math.sqrt(512) - 1) < 0.05 and float(mb.example_embed.bias.abs().max()) == 0.0 and abs(mb.label_embed.weight.std().item() / 0.02 - 0.88) < 0.1)
mm = IHTransformer(ModelConfig(mlp_ratio=4.0))
check("the optional MLP (architecture-sensitivity experiment) adds parameters and is off by default", mm.n_parameters() > mb.n_parameters() and mb.cfg.mlp_ratio is None)

# RoPE: relative position property and agreement with an independent (complex-number) formulation
cos, sin = rope_tables(16, 8, 10000.0)
g = torch.Generator().manual_seed(1)
q0, k0 = torch.randn(8, generator=g), torch.randn(8, generator=g)
def score(m, n):
    qm = apply_rope(q0[None, None, :].expand(1, 16, 8), cos, sin)[0, m]
    kn = apply_rope(k0[None, None, :].expand(1, 16, 8), cos, sin)[0, n]
    return float(qm @ kn)
diffs = [abs(score(m + s, n + s) - score(m, n)) for m in range(0, 5) for n in range(0, 5) for s in range(0, 11)]
check("RoPE attention scores depend only on relative position (m - n)", max(diffs) < 1e-4, f"max diff {max(diffs):.2e}")
check("RoPE changes the score when the relative offset changes", abs(score(3, 1) - score(3, 2)) > 1e-3)
pos, j = 5, torch.arange(4)
z = (q0[0::2] + 1j * q0[1::2]) * torch.exp(1j * pos * 10000.0 ** (-2.0 * j / 8))
ref = torch.stack([z.real, z.imag], dim=-1).reshape(8)
check("RoPE equals rotating each interleaved pair by pos * 10000^(-2j/8)", float((apply_rope(q0[None, None, :], cos[pos:pos + 1], sin[pos:pos + 1])[0, 0] - ref).abs().max()) < 1e-5)

# causal mask
b5 = D.collate(torch.as_tensor(base[:16]), feats, False)
o1 = mb(b5.ex, b5.labels, return_details=True)
ex2 = b5.ex.clone(); ex2[:, 2] = torch.randn_like(ex2[:, 2])          # change the query exemplar (token 4)
o2 = mb(ex2, b5.labels, return_details=True)
check("causal mask: changing the query token leaves every earlier position unchanged", float((o1["hidden"][:, :4] - o2["hidden"][:, :4]).abs().max()) < 1e-6 and float((o1["hidden"][:, 4] - o2["hidden"][:, 4]).abs().max()) > 1e-6)
a = o1["attn"][0]
check("attention patterns are row-stochastic and exactly zero above the diagonal", bool((a.triu(1) == 0).all()) and float((a.sum(-1) - 1).abs().max()) < 1e-5)
b7 = D.collate(torch.as_tensor(ext[:16]), feats, True)
p1 = me(b7.ex, b7.labels, return_details=True)
ex7 = b7.ex.clone(); ex7[:, 3] = torch.randn_like(ex7[:, 3])           # change S2 (token 6)
p2 = me(ex7, b7.labels, return_details=True)
check("causal mask (extended): changing S2 leaves positions 0..5, hence lq and S2 predictions, unchanged", float((p1["hidden"][:, :6] - p2["hidden"][:, :6]).abs().max()) < 1e-6 and torch.allclose(p1["label_logits_q"], p2["label_logits_q"]) and torch.allclose(p1["symbol_logits"], p2["symbol_logits"]) and not torch.allclose(p1["label_logits_s2"], p2["label_logits_s2"]))
check("extended outputs: lq (B,5) at position 4, S2 (B,50) at position 5, l_S2 (B,5) at position 6", tuple(p1["label_logits_q"].shape) == (16, 5) and tuple(p1["symbol_logits"].shape) == (16, 50) and tuple(p1["label_logits_s2"].shape) == (16, 5) and tuple(p1["hidden"].shape) == (16, 7, 64))

# hooks
base_logits = mb(b5.ex, b5.labels)["label_logits_q"]
check("no ablation reproduces the plain forward pass", torch.allclose(mb(b5.ex, b5.labels, ablate={})["label_logits_q"], base_logits))
check("zeroing a head changes the output; zeroing all heads of a layer removes that layer's attention", not torch.allclose(mb(b5.ex, b5.labels, ablate={1: {3}})["label_logits_q"], base_logits) and float(mb(b5.ex, b5.labels, ablate={0: set(range(8))}, return_details=True)["head_out"][0].abs().max()) == 0.0)
own = {(1, h): o1["attn"][1][:, h] for h in range(8)}
check("overriding every layer-2 attention pattern with its own reproduces the output", torch.allclose(mb(b5.ex, b5.labels, attn_override=own)["label_logits_q"], base_logits, atol=1e-6))
det = mb(b5.ex, b5.labels, return_details=True, want_contrib=True)
contrib, resid = det["head_contrib"], det["resid"]
# the residual after block 0 is the input plus the sum of the heads' contributions (attention-only block)
check("per-head contributions sum to the block's output: resid_after = resid_before + sum over heads", tuple(contrib[0].shape) == (16, 8, 5, 64) and float((resid[0] + contrib[0].sum(1) - resid[1]).abs().max()) < 1e-5 and float((resid[1] + contrib[1].sum(1) - resid[2]).abs().max()) < 1e-5)

# generation
ex0_table = feats[D.TRAIN_CLASSES, 0]
gen = torch.Generator().manual_seed(3)
lq, s2, ls2 = me.sample_extended(b5.ex, b5.labels, ex0_table, temperature=1.0, gen=gen)
check("sample_extended returns lq, S2 class, l_S2 with the right shapes and ranges", lq.shape == s2.shape == ls2.shape == (16,) and int(lq.max()) < 5 and int(s2.max()) < 50 and int(ls2.max()) < 5 and int(s2.min()) >= 0)
g1 = me.sample_extended(b5.ex, b5.labels, ex0_table, greedy=True); g2 = me.sample_extended(b5.ex, b5.labels, ex0_table, greedy=True)
check("greedy generation is deterministic", all(torch.equal(x, y) for x, y in zip(g1, g2)))
check("base model samples a query label per context", mb.sample_query_label(b5.ex, b5.labels, gen=gen).shape == (16,))

# ------------------------------------------------------------------ metrics on hand-built attention
B_ = 4
q_is_A = torch.tensor([1, 0, 1, 0])
def attn_to(positions):                        # layer with every head attending (from position 4) to the given token per sequence
    a = torch.zeros(B_, 8, 5, 5)
    for i in range(B_):
        a[i, :, 4, positions[i]] = 1.0
    return a
l1 = torch.zeros(B_, 8, 5, 5); l1[:, :, 1, 0] = 1; l1[:, :, 3, 2] = 1       # previous-token heads in layer 1
correct = [1 if x else 3 for x in q_is_A.tolist()]; wrong = [3 if x else 1 for x in q_is_A.tolist()]; matchx = [0 if x else 2 for x in q_is_A.tolist()]
m_ok = M.attention_metrics([l1, attn_to(correct)], q_is_A)
m_bad = M.attention_metrics([l1, attn_to(wrong)], q_is_A)
m_match = M.attention_metrics([l1, attn_to(matchx)], q_is_A)
check("induction strength is +1 when a layer-2 head attends the correct label, -1 for the wrong one", abs(m_ok["ind_L2H0"] - 1) < 1e-6 and abs(m_bad["ind_L2H0"] + 1) < 1e-6)
check("previous-token score is 1 for heads attending label->preceding exemplar", abs(m_ok["prev_L1H3"] - 1) < 1e-6)
check("match score is 1 when a layer-2 head attends the matching exemplar", abs(m_match["match_L2H5"] - 1) < 1e-6 and m_ok["match_L2H5"] == 0.0)
check("attention entropy is 0 for one-hot rows", abs(m_ok["ent_L1H0"]) < 1e-6)

# ------------------------------------------------------------------ training: configuration and a tiny end-to-end run
c = TrainConfig(sequences=1_000_000, batch_size=32)
check("budget conversion: 1e6 sequences at batch 32 = 31,250 optimiser steps; steps= overrides", c.n_steps == 31250 and TrainConfig(steps=7).n_steps == 7)
check("defaults are the paper's: Adam lr 1e-5, batch 32, no early stopping option, one thread", c.lr == 1e-5 and c.batch_size == 32 and c.n_threads == 1 and not hasattr(c, "patience") and not hasattr(c, "weight_decay"))
small = {k: v[:60] for k, v in D.build_eval_sets().items()}
for variant in ("base", "extended"):
    cfg = TrainConfig(variant=variant, steps=40, eval_every=20, log_path=str(TMP / f"{variant}.csv"), ckpt_path=str(TMP / f"{variant}.pt"), ckpt_steps=(20,), verbose=False)
    res = train_model(cfg, features=feats, evals=EvalSets(feats, cfg.extended, small))
    rows_ = list(csv.DictReader(open(cfg.log_path)))
    need = ["step", "sequences", "train_loss_running", "val/loss", "val/acc", "val/ppl", "test_relabel/acc", "test_exemplars/acc", "ind_L2H0", "prev_L1H0", "match_L2H0", "ent_L1H0", "ind_max"] + (["train_eval/acc_s2", "train_eval/acc_ls2", "val/loss_s2"] if variant == "extended" else [])
    check(f"{variant}: a 40-step run logs every evaluation set and the interpretability metrics", all(k in rows_[0] for k in need) and rows_[-1]["step"] == "40" and rows_[-1]["sequences"] == "1280", str([k for k in need if k not in rows_[0]]))
    check(f"{variant}: checkpoints (final and the requested step) are written and reload to the same outputs", (TMP / f"{variant}_step20.pt").exists() and (TMP / f"{variant}.pt").exists() and torch.allclose(__import__("train").load_model(cfg.ckpt_path)[0](b5.ex, b5.labels)["label_logits_q"], res["model"].eval()(b5.ex, b5.labels)["label_logits_q"], atol=1e-6))
check("training moves the parameters (the optimiser is doing something)", float((res["model"].example_embed.weight - IHTransformer(ModelConfig(extended=True)).example_embed.weight).abs().max()) > 0)
r1 = train_model(TrainConfig(variant="base", steps=12, eval_every=12, log_path=str(TMP / "d1.csv"), verbose=False), features=feats, evals=EvalSets(feats, False, small))
r2 = train_model(TrainConfig(variant="base", steps=12, eval_every=12, log_path=str(TMP / "d2.csv"), verbose=False), features=feats, evals=EvalSets(feats, False, small))
check("training is deterministic given the seed", all(torch.equal(p, q) for p, q in zip(r1["model"].parameters(), r2["model"].parameters())))

# ------------------------------------------------------------------ ablations and phase landmarks (Phase 2)
import circuits as CI
ev_small = EvalSets(feats, False, small)
prof = CI.ablation_profile(mb, ev_small)
need = ["train_eval/baseline", "train_eval/zero_L2H0", "train_eval/keep_only_L2H7", "train_eval/zero_L1H3", "train_eval/zero_all_L1", "train_eval/zero_all_L2", "train_eval/cut_L1_direct", "test_relabel/zero_L2H2", "val/baseline"]
check("ablation profile reports every intervention on every evaluation set", all(k in prof for k in need) and sum(k.startswith("test_exemplars/") for k in prof) == 1 + 8 * 3 + 3)
bb = ev_small.batches["train_eval"]
check("keep_only_Lh equals zeroing the other seven layer-2 heads", abs(prof["train_eval/keep_only_L2H3"] - CI._acc(mb, bb, {1: set(range(8)) - {3}})) < 1e-12)
dd = mb(bb.ex[:, :3], bb.labels[:, :2], return_details=True, want_contrib=True)
check("the final residual is the embedding plus both layers' head contributions (so 'cut layer 1 direct' subtracts exactly layer 1)", float((dd["resid"][0] + dd["head_contrib"][0].sum(1) + dd["head_contrib"][1].sum(1) - dd["resid"][-1]).abs().max()) < 1e-5)
logits_cut = mb.label_head(mb.ln_f(dd["resid"][-1] - dd["head_contrib"][0].sum(1))[:, 4])
check("cut_L1_direct matches an independent recomputation of the output without layer 1's direct write", abs(prof["train_eval/cut_L1_direct"] - (logits_cut.argmax(-1) == bb.query_label).float().mean().item()) < 1e-12)
ad = CI.additivity(mb, ev_small, {0: 0.9, 1: 0.5, 2: 0.0, 3: -0.1, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0})
check("additivity picks the heads above the strength threshold, strongest first", ad["heads"] == [0, 1] and ad["n_induction_heads"] == 2)
rows_ph = [{"step": s_, "train_eval/loss": l_, "val/acc": a_, "train_eval/acc": a_} for s_, l_, a_ in [(1, 1.6, 0.2), (2, 0.69, 0.5), (3, 0.5, 0.7), (4, 0.1, 0.95)]]
lm = M.phase_landmarks(rows_ph)
check("phase landmarks: plateau exit below ln2-0.1, accuracy above 0.9", lm["plateau_exit_step"] == 3 and lm["val_acc_gt_0.9_step"] == 4 and M.phase_landmarks(rows_ph[:2])["val_acc_gt_0.9_step"] is None)

# ------------------------------------------------------------------ ACDC on the attention-only graph
import acdc as AC
torch.manual_seed(5)
ra = IHTransformer(ModelConfig())
for q_ in ra.parameters():                      # larger weights so the invariants are not satisfied by a near-zero model
    torch.nn.init.normal_(q_, 0, 0.3) if q_.dim() > 1 else torch.nn.init.normal_(q_, 0, 0.1)
ba_ = D.collate(torch.as_tensor(base[:64]), feats, False); bb_ = D.collate(torch.as_tensor(base[5000:5064]), feats, False)
exa, laa, exb, lab_ = ba_.ex[:, :3], ba_.labels[:, :2], bb_.ex[:, :3], bb_.labels[:, :2]
check("ACDC graph for 2 layers x 8 heads has 257 edges (24 + 216 + 17)", len(AC.all_edges(2, 8)) == 257, str(len(AC.all_edges(2, 8))))
lg, _ = AC.run_graph(ra, exa, laa)
check("ACDC graph with every edge kept reproduces the model's own forward pass", float((lg - ra(exa, laa)["label_logits_q"]).abs().max()) < 1e-4)
_, cor_ = AC.run_graph(ra, exb, lab_)
cut_, _ = AC.run_graph(ra, exa, laa, {e: False for e in AC.all_edges(2, 8)}, cor_)
check("ACDC graph with every edge cut reproduces the model on the corrupted input", float((cut_ - ra(exb, lab_)["label_logits_q"]).abs().max()) < 1e-4)
keep_all = AC.acdc(ra, exa, laa, ba_.query_label, exb, lab_, tau=-1.0)
cut_all = AC.acdc(ra, exa, laa, ba_.query_label, exb, lab_, tau=1e9)
mid_ = AC.acdc(ra, exa, laa, ba_.query_label, exb, lab_, tau=0.05)
check("ACDC: a negative threshold keeps every edge (circuit == model); a huge one leaves an empty circuit", keep_all.kl < 1e-9 and all(keep_all.kept.values()) and not any(cut_all.kept.values()) and cut_all.n_live_edges == 0)
check("ACDC circuit size shrinks as the threshold grows and live edges are kept edges", cut_all.n_live_edges <= mid_.n_live_edges <= keep_all.n_live_edges and set(mid_.live_edges) <= {e for e, v in mid_.kept.items() if v})
rl_ = AC.relabel_corruption(laa, 5, seed=2)
check("relabel_corruption redraws two DISTINCT labels per sequence from the 5", rl_.shape == laa.shape and bool((rl_[:, 0] != rl_[:, 1]).all()) and int(rl_.max()) < 5)
d_ = AC.describe(mid_)
check("describe() reports size, live heads and the layer-1 -> layer-2 composition edges", all(k in d_ for k in ["n_edges", "heads_L1", "heads_L2", "kcomp", "qcomp", "vcomp", "edges"]))

# ------------------------------------------------------------------ collapse pipeline
import collapse as CO
rs_ext = D.enumerate_train_space(True)
check("real rows carry -1 in the three synthetic-target columns and 13 columns in all", base.shape[1] == D.N_COLS == 13 and bool((base[:, 10:] == -1).all()))
st_real = CO.pool_stats(rs_ext, "extended")
check("real extended pool: P(S2 = Q's class) = 0.5, S2 always in context, uniform S2 histogram (entropy ln 50, tail mass 0.2), labels all correct",
      abs(st_real["p_s2_equals_q_class"] - 0.5) < 1e-9 and st_real["p_s2_outside_context"] == 0.0 and abs(st_real["s2_entropy"] - math.log(50)) < 1e-9 and abs(st_real["s2_tail_mass_k10"] - 0.2) < 1e-9 and st_real["lq_correct_all"] == 1.0 and st_real["ls2_correct_given_in_context"] == 1.0, str({k: round(v, 4) for k, v in st_real.items()}))
check("noise floor of the S2 histogram KL between two independent real pools is small but positive", 0 < CO.noise_floor(rs_ext) < 0.01, f"{CO.noise_floor(rs_ext):.2e}")
# synthetic rows: targets come from the override columns
srows = base[:6].copy(); srows[:, D.LQ_SYN] = (D.true_query_label(torch.as_tensor(srows)).numpy() + 1) % 5
bs = D.collate(torch.as_tensor(srows), feats, False)
check("a synthetic query label replaces the in-context label as the training target", bool((bs.query_label != D.true_query_label(torch.as_tensor(srows))).all()))
erows = D.enumerate_train_space(False)[:6].copy(); erows[:, D.LQ_SYN] = 2; erows[:, D.S2C_SYN] = [3, 7, 11, 13, 17, 49]; erows[:, D.LS2_SYN] = 4
be_ = D.collate(torch.as_tensor(erows), feats, True)
check("synthetic extended rows: S2 is exemplar 0 of the sampled class; lq is fed back as the third label; l_S2 and S2 targets are the sampled ones",
      bool((be_.ex[:, 3] == feats[torch.as_tensor(erows[:, D.S2C_SYN]), 0]).all()) and bool((be_.labels[:, 2] == 2).all()) and bool((be_.s2_label == 4).all()) and be_.s2_class.tolist() == [3, 7, 11, 13, 17, 49])
# mixing
pm = CO.mix_pool(rs_ext, D.enumerate_train_space(False), 0.0, 1000, seed=1)
check("mix_pool at p = 0: only real rows, drawn without replacement (all distinct)", len(pm) == 1000 and int((pm[:, D.LQ_SYN] >= 0).sum()) == 0 and len({tuple(r) for r in pm.tolist()}) == 1000)
syn_fake = D.enumerate_train_space(False).copy(); syn_fake[:, D.LQ_SYN] = 0
pm2 = CO.mix_pool(rs_ext, syn_fake, 0.25, 1000, seed=1)
check("mix_pool at p = 0.25 gives exactly 250 synthetic and 750 real rows", int((pm2[:, D.LQ_SYN] >= 0).sum()) == 250 and int((pm2[:, D.LQ_SYN] < 0).sum()) == 750)
try:
    CO.mix_pool(rs_ext, syn_fake, 1.0, 100_000, seed=1); check("mix_pool refuses to draw more rows than the source has (no replacement)", False)
except ValueError:
    check("mix_pool refuses to draw more rows than the source has (no replacement)", True)
# generated pools
gen_ = torch.Generator().manual_seed(9)
ctx = D.enumerate_train_space(False)[::50]
sb = CO.synthetic_pool(mb, ctx, feats, "base", 1.0, False, gen_)
check("base synthetic pool: every context once (no duplicate contexts), a sampled label in every row, contexts unchanged", len({tuple(r[:8]) for r in sb.tolist()}) == len(ctx) and bool((sb[:, D.LQ_SYN] >= 0).all()) and bool((sb[:, D.LQ_SYN] < 5).all()) and np.array_equal(sb[:, :8], ctx[:, :8]))
se = CO.synthetic_pool(me, ctx, feats, "extended", 1.0, False, gen_)
check("extended synthetic pool: lq, S2 class and l_S2 sampled for every context", bool((se[:, D.LQ_SYN] >= 0).all()) and bool((se[:, D.S2C_SYN] >= 0).all()) and bool((se[:, D.S2C_SYN] < 50).all()) and bool((se[:, D.LS2_SYN] >= 0).all()))
st_syn = CO.pool_stats(se, "extended", np.bincount(D.s2_class(torch.as_tensor(rs_ext)).numpy(), minlength=50).astype(float))
check("pool statistics for a synthetic extended pool are reported (S2 outside context, KL to generation 0, tail mass)", all(k in st_syn for k in ["p_s2_outside_context", "s2_kl_vs_gen0", "s2_tail_mass_k10", "lq_correct_synthetic"]) and st_syn["frac_synthetic"] == 1.0)
g1 = CO.synthetic_pool(mb, ctx, feats, "base", 1.0, True, torch.Generator().manual_seed(1)); g2 = CO.synthetic_pool(mb, ctx, feats, "base", 1.0, True, torch.Generator().manual_seed(2))
check("greedy sampling is deterministic and ignores the random generator", np.array_equal(g1, g2))
# end to end: two generations, both variants
for variant in ("base", "extended"):
    cfg_c = CO.CollapseConfig(variant=variant, p_synthetic=0.5, n_generations=2, seed=3, steps=15, eval_every=15, out_dir=str(TMP / f"collapse_{variant}"))
    CO.run_collapse(cfg_c, features=feats, verbose=False)
    od = TMP / f"collapse_{variant}"
    gens_ = list(csv.DictReader(open(od / "generations.csv")))
    check(f"collapse {variant}: 2 generations run end to end and write generations.csv, per-generation logs, models and pools",
          len(gens_) == 2 and all((od / f"gen{g}_{k}").exists() for g in (0, 1) for k in ("train.csv", "model.pt", "pool.npz")) and "val/acc" in gens_[0] and "pool_lq_correct_all" in gens_[0])
    p0, p1 = np.load(od / "gen0_pool.npz")["pool"], np.load(od / "gen1_pool.npz")["pool"]
    check(f"collapse {variant}: generation 0 trains on the whole real space, generation 1 on a 78,400-row pool of which half is synthetic",
          len(p0) == (156800 if variant == "extended" else 78400) and len(p1) == 78400 and abs(float((p1[:, D.LQ_SYN] >= 0).mean()) - 0.5) < 1e-6)
    ck0 = torch.load(od / "gen0_model.pt", weights_only=False); ck1 = torch.load(od / "gen1_model.pt", weights_only=False)
    check(f"collapse {variant}: the same initialisation seed in every generation, a different batch-order seed", ck0["train_cfg"]["seed"] == ck1["train_cfg"]["seed"] == 3 and ck0["train_cfg"]["data_seed"] != ck1["train_cfg"]["data_seed"])
    check(f"collapse {variant}: validation and test sets are the real ones in every generation (identical evaluation sets, so a model's logged accuracy is on real data)", all(f"{s}/acc" in gens_[1] for s in D.EVAL_SET_NAMES))

# ------------------------------------------------------------------ Tier 1 analysis on a small results tree
sys.path.insert(0, str(ROOT / "experiments"))
import analyze_collapse as ACL, analyze_collapse_circuits as ACC
tree = TMP / "tree"; (tree / "collapse").mkdir(parents=True)
for cond, src in [("exact_extended_p05", "collapse_extended"), ("exact_extended_p0", "collapse_extended"), ("exact_base_p1", "collapse_base"), ("exact_base_p0", "collapse_base")]:
    for sd in (0, 1):
        shutil.copytree(TMP / src, tree / "collapse" / f"{cond}_s{sd}")
ev_x = EvalSets(feats, True, small)
rws, acr = ACC.analyze_run(tree, "exact_extended_p05_s0", ev_x, feats, taus=[0.01], n_acdc=16)
write_ = ACC.write_csv; write_(rws, tree / "analysis" / "circuits_exact_extended_p05_s0.csv"); write_(acr, tree / "analysis" / "acdc_exact_extended_p05_s0.csv")
check("per-generation circuit analysis writes one row per generation with ablation, additivity and embedding-geometry columns",
      len(rws) == 2 and all(k in rws[0] for k in ["zero_L2H0_drop", "keep_only_L2H7_acc", "zero_L1H3_drop", "cut_L1_direct_acc", "n_induction_heads", "sum_single_drops", "joint_drop", "emb_cos_train", "emb_cos_test", "emb_effective_rank", "label_cos_gen0", "label_mean_pair_cos"]))
check("generation 0 is identical to itself in embedding space (cosine 1) and the effective rank is between 1 and 50", abs(rws[0]["emb_cos_train"] - 1) < 1e-5 and 1 <= rws[0]["emb_effective_rank"] <= 50)
check("ACDC rows: one per generation x corruption x tau with circuit size and composition columns", len(acr) == 2 * 2 * 1 and {r["corruption"] for r in acr} == {"resample", "labels"} and all(k in acr[0] for k in ["n_edges", "kl", "acc", "heads_L2", "kcomp"]))
sm_ = ACL.collapse_summary(tree)
check("collapse summary covers all four conditions with per-seed accuracy by generation, failure counts with intervals and the match-vs-copy tally",
      set(sm_["conditions"]) == {"exact_extended_p05", "exact_extended_p0", "exact_base_p1", "exact_base_p0"} and all(k in sm_["conditions"]["exact_base_p1"] for k in ["acc_by_seed", "degraded_trainings", "failed_ci95", "match_before_copy", "copy_before_match", "same_generation", "neither_drops"]))
check("contrasts with the real-data control are computed (exact permutation test over the seeds)", set(sm_["contrasts_vs_control"]) == {"exact_base_p1", "exact_extended_p05"} and sm_["contrasts_vs_control"]["exact_base_p1"]["final_copy_acc"]["permutation_p"] is not None)
check("extended summaries include the S2 distribution-drift statistics and the noise floor", "pool_s2_kl_vs_gen0_median_by_gen" in sm_["conditions"]["exact_extended_p05"] and "s2_noise_floor_kl" in sm_["conditions"]["exact_extended_p05"])
check("head-redundancy statistics appear once the circuit analysis exists", "n_induction_heads_median_by_gen" in sm_["conditions"]["exact_extended_p05"])
ACL.make_figures(tree)
check("the Tier 1 figures are written (accuracy, head scores, match vs copy, S2 drift, ACDC, ablation heatmaps, embeddings)", all((tree / "figures" / f).exists() for f in ["fig_accuracy_by_generation.png", "fig_head_scores.png", "fig_match_vs_copy.png", "fig_s2_drift.png", "fig_ablation_heatmaps.png", "fig_acdc.png", "fig_embeddings.png"]), str(sorted(p.name for p in (tree / "figures").glob("*"))))
check("permutation test: identical groups give p = 1 and fully separated groups the minimum", ACL.perm_test([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0 and abs(ACL.perm_test([1, 2, 3, 4], [10, 11, 12, 13]) - round(2 / 70, 4)) < 1e-9)
check("Wilson interval for 0 of 10 is [0, ~0.28]", ACL.wilson(0, 10)[0] == 0.0 and abs(ACL.wilson(0, 10)[1] - 0.278) < 0.002)

shutil.rmtree(TMP, ignore_errors=True)
print("\n==== FAILED:", FAILS if FAILS else "none ====")
sys.exit(1 if FAILS else 0)
