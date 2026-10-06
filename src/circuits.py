"""Head-ablation analyses of a trained model (Singh et al. 2024, Section 3), for the attention-only network.

Everything is measured as accuracy on each evaluation set, so the MATCH step (generalisation to unseen exemplars:
``test_exemplars``, ``val``, ``test_class_ex0``) and the COPY step (unseen label pairs: ``test_relabel``) are reported
separately, as the paper splits them.

Interventions (all use the model's own hooks; nothing is retrained):
    zero_L2H{h}        zero layer-2 head h               -> accuracy drop shows how much that head is needed
    keep_only_L2H{h}   zero every OTHER layer-2 head      -> accuracy with that head alone
    zero_L1H{h}        zero layer-1 head h
    zero_all_L1 / zero_all_L2
    cut_L1_direct      remove layer 1's DIRECT write to the output (the residual stream still carries layer 1 into
                       layer 2): the final residual is  x_final - sum_h(layer-1 head contributions).
The paper finds several layer-2 heads that each carry part of the induction function with roughly ADDITIVE effects
under ablation; ``additivity`` compares the sum of single-head drops with the drop from ablating the same heads together.

INDUCTION_HEAD_THRESHOLD: a layer-2 head counts as an "induction head" when its induction strength is above this.
The value is a documented choice (the strengths of untrained / non-induction heads stay within about +-0.05).
"""

from __future__ import annotations

import torch

from model import IHTransformer

INDUCTION_HEAD_THRESHOLD = 0.2


@torch.no_grad()
def _acc(model: IHTransformer, batch, ablate=None) -> float:
    out = model(batch.ex[:, :3], batch.labels[:, :2], ablate=ablate)
    return (out["label_logits_q"].argmax(-1) == batch.query_label).float().mean().item()


@torch.no_grad()
def _acc_cut_direct(model: IHTransformer, batch, layer: int = 0) -> float:
    """Accuracy when layer ``layer``'s heads (default layer 1 = index 0) are removed from the residual stream the output
    reads, while later layers still see them."""
    out = model(batch.ex[:, :3], batch.labels[:, :2], return_details=True, want_contrib=True)
    final = out["resid"][-1] - out["head_contrib"][layer].sum(1)
    logits = model.label_head(model.ln_f(final)[:, 4])
    return (logits.argmax(-1) == batch.query_label).float().mean().item()


def ablation_profile(model: IHTransformer, evals, sets: tuple[str, ...] = ("train_eval", "val", "test_exemplars", "test_relabel", "test_class_ex0")) -> dict:
    """Accuracy on each requested set under every intervention above. Keys: ``<set>/<intervention>``."""
    model.eval()
    H = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    last = n_layers - 1
    out: dict[str, float] = {}
    for s in sets:
        b = evals.batches[s]
        out[f"{s}/baseline"] = _acc(model, b)
        for h in range(H):
            out[f"{s}/zero_L2H{h}"] = _acc(model, b, {last: {h}})
            out[f"{s}/keep_only_L2H{h}"] = _acc(model, b, {last: set(range(H)) - {h}})
            out[f"{s}/zero_L1H{h}"] = _acc(model, b, {0: {h}})
        out[f"{s}/zero_all_L1"] = _acc(model, b, {0: set(range(H))})
        out[f"{s}/zero_all_L2"] = _acc(model, b, {last: set(range(H))})
        out[f"{s}/cut_L1_direct"] = _acc_cut_direct(model, b, 0)
    return out


def additivity(model: IHTransformer, evals, strengths: dict[int, float], threshold: float = INDUCTION_HEAD_THRESHOLD,
               set_name: str = "train_eval") -> dict:
    """For the layer-2 heads whose induction strength exceeds ``threshold``: how many there are, the sum of their
    single-head accuracy drops, the drop from ablating all of them together, and the accuracy with only the strongest left."""
    b = evals.batches[set_name]
    H, last = model.cfg.n_heads, model.cfg.n_layers - 1
    heads = sorted([h for h, v in strengths.items() if v > threshold], key=lambda h: -strengths[h])
    base = _acc(model, b)
    singles = [base - _acc(model, b, {last: {h}}) for h in heads]
    joint = base - _acc(model, b, {last: set(heads)}) if heads else 0.0
    best_only = _acc(model, b, {last: set(range(H)) - {heads[0]}}) if heads else float("nan")
    return {"n_induction_heads": len(heads), "heads": heads, "sum_single_drops": float(sum(singles)), "joint_drop": float(joint),
            "baseline_acc": base, "acc_keep_only_strongest": best_only}
