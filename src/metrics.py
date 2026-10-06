"""Losses, accuracies and the interpretability metrics defined by Singh et al. (2024).

Token positions of the first five tokens ``A la B lb Q`` are 0..4 in both the base and the extended
sequence, and attention is causal, so every attention metric below is read from those five positions
whatever the model variant.

Metrics (paper Section 3 and the plan's 5.1)
    induction strength  (layer-2 heads)  attention from the query Q (position 4) to the label token of the
                        CORRECT in-context exemplar minus attention to the label token of the other one:
                        attn(4 -> 1) - attn(4 -> 3) when Q matches A, the reverse when Q matches B.
    previous-token score (layer-1 heads) mean attention from each label position to the exemplar just before it:
                        (attn(1 -> 0) + attn(3 -> 2)) / 2.
    match score         (layer-2 heads)  attention from Q to the position of the MATCHING exemplar (offset 0):
                        attn(4 -> 0) when Q matches A, attn(4 -> 2) when it matches B.
    attention entropy   mean over sequences and positions of the entropy (nats) of each head's attention row.

Phase-change landmarks (documented thresholds; used by the analysis scripts)
    The loss on the query label starts near ln 5 ~ 1.61 (five labels), falls to a plateau near ln 2 ~ 0.69 (the
    model has learnt to answer with one of the two labels it can see, without telling which), and later falls
    sharply when the induction circuit forms.
    PLATEAU_EXIT_LOSS: the plateau is "left" the first time the loss on ``train_eval`` is below ln 2 - 0.1.
    PHASE_ACC: the phase change is dated by the first step at which ``val`` accuracy exceeds 0.9.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
PLATEAU_EXIT_LOSS = LN2 - 0.1
PHASE_ACC = 0.9

# Bayes-optimal accuracies of the extended targets under REAL data (see data.py): lq 1.0, S2 0.5, l_S2 1.0.
BAYES_OPTIMAL = {"acc_q": 1.0, "acc_s2": 0.5, "acc_ls2": 1.0}


def _acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    ok = target >= 0
    if not bool(ok.any()):
        return float("nan")
    return (logits[ok].argmax(-1) == target[ok]).float().mean().item()


def _ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ok = target >= 0                      # -1 marks "no target" (S2 outside the 50 training classes)
    if not bool(ok.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[ok], target[ok])


def loss_terms(out: dict, batch, extended: bool) -> dict:
    """Cross-entropy of each target. ``loss`` is the training objective: the query label only (base), or
    CE(lq) + CE(S2) + CE(l_S2), equally weighted (extended)."""
    terms = {"loss_q": _ce(out["label_logits_q"], batch.query_label)}
    total = terms["loss_q"]
    if extended:
        terms["loss_s2"] = _ce(out["symbol_logits"], batch.s2_class)
        terms["loss_ls2"] = _ce(out["label_logits_s2"], batch.s2_label)
        total = terms["loss_q"] + terms["loss_s2"] + terms["loss_ls2"]
    terms["loss"] = total
    return terms


def accuracies(out: dict, batch, extended: bool) -> dict:
    acc = {"acc_q": _acc(out["label_logits_q"], batch.query_label)}
    if extended:
        acc["acc_s2"] = _acc(out["symbol_logits"], batch.s2_class)
        acc["acc_ls2"] = _acc(out["label_logits_s2"], batch.s2_label)
    return acc


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def attention_metrics(attn: list[torch.Tensor], q_is_A: torch.Tensor) -> dict:
    """Per-head induction strength / match score (layer 2), previous-token score (layer 1), entropy (all heads).

    attn: per layer (B, H, T, T) attention patterns with T >= 5; q_is_A: (B,) 1 where Q matches A."""
    qa = q_is_A.bool()
    correct_label = torch.where(qa, 1, 3)           # token index of the correct in-context label
    wrong_label = torch.where(qa, 3, 1)
    match_ex = torch.where(qa, 0, 2)                # token index of the matching exemplar
    idx = torch.arange(len(qa))
    out = {}
    last = attn[-1]
    ind = last[idx, :, 4, correct_label] - last[idx, :, 4, wrong_label]          # (B, H)
    mat = last[idx, :, 4, match_ex]
    prev = 0.5 * (attn[0][:, :, 1, 0] + attn[0][:, :, 3, 2])                       # (B, H)
    for h in range(last.shape[1]):
        out[f"ind_L{len(attn)}H{h}"] = ind[:, h].mean().item()
        out[f"match_L{len(attn)}H{h}"] = mat[:, h].mean().item()
        out[f"prev_L1H{h}"] = prev[:, h].mean().item()
    for li, a in enumerate(attn):
        ent = -(a.clamp_min(1e-12).log() * a).sum(-1).mean(dim=(0, 2))             # (H,)
        for h in range(a.shape[1]):
            out[f"ent_L{li + 1}H{h}"] = ent[h].item()
    return out


def head_summary(m: dict, n_heads: int, n_layers: int = 2) -> dict:
    """Best-head and mean summaries of the per-head metrics (for compact logs)."""
    ind = [m[f"ind_L{n_layers}H{h}"] for h in range(n_heads)]
    return {"ind_max": max(ind), "ind_mean": sum(ind) / len(ind),
            "match_max": max(m[f"match_L{n_layers}H{h}"] for h in range(n_heads)),
            "prev_max": max(m[f"prev_L1H{h}"] for h in range(n_heads))}
