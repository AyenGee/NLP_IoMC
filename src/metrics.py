"""Loss / accuracy / perplexity / attention-entropy utilities.

Kept separate from model.py so the same functions can be reused by train.py,
collapse.py and interp.py without duplicating the base-vs-extended loss
bookkeeping.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from model import ModelOutput


@dataclasses.dataclass
class LossOutput:
    total_loss: torch.Tensor
    base_loss: torch.Tensor
    base_acc: float
    aux_label_loss: torch.Tensor | None = None
    aux_label_acc: float | None = None
    aux_symbol_loss: torch.Tensor | None = None
    aux_symbol_acc: float | None = None


def _acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def base_task_loss(output: ModelOutput, batch: dict) -> tuple[torch.Tensor, float]:
    """Base task: label prediction at the final (query) position only."""
    final_logits = output.label_logits[:, -1, :]
    target = batch["query_label"]
    loss = F.cross_entropy(final_logits, target)
    return loss, _acc(final_logits, target)


def aux_losses(output: ModelOutput, batch: dict) -> tuple[torch.Tensor, float, torch.Tensor, float]:
    """Extended task auxiliaries: 'corresponding label' at intermediate symbol
    positions (excludes the final/query position, which is the base loss),
    and 'next symbol' at every label position."""
    m = output.symbol_logits.shape[1]
    inter_label_logits = output.label_logits[:, :m, :]           # (B, m, L)
    inter_label_target = batch["next_label_target"][:, :m]       # (B, m)
    label_loss = F.cross_entropy(
        inter_label_logits.reshape(-1, inter_label_logits.shape[-1]),
        inter_label_target.reshape(-1),
    )
    label_acc = _acc(inter_label_logits, inter_label_target)

    symbol_logits = output.symbol_logits                          # (B, m, K)
    symbol_target = batch["next_symbol_target"]                   # (B, m)
    symbol_loss = F.cross_entropy(
        symbol_logits.reshape(-1, symbol_logits.shape[-1]), symbol_target.reshape(-1)
    )
    symbol_acc = _acc(symbol_logits, symbol_target)
    return label_loss, label_acc, symbol_loss, symbol_acc


def compute_losses(output: ModelOutput, batch: dict, extended: bool, aux_weight: float = 1.0) -> LossOutput:
    base_loss, base_acc = base_task_loss(output, batch)
    if not extended:
        return LossOutput(total_loss=base_loss, base_loss=base_loss, base_acc=base_acc)

    aux_label_loss, aux_label_acc, aux_symbol_loss, aux_symbol_acc = aux_losses(output, batch)
    total = base_loss + aux_weight * (aux_label_loss + aux_symbol_loss)
    return LossOutput(
        total_loss=total,
        base_loss=base_loss,
        base_acc=base_acc,
        aux_label_loss=aux_label_loss,
        aux_label_acc=aux_label_acc,
        aux_symbol_loss=aux_symbol_loss,
        aux_symbol_acc=aux_symbol_acc,
    )


def symbol_label_counts(items: list[dict], K: int, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw usage counts of symbol ids and label ids across a pool of
    (real or synthetic) sequence items -- the direct analogue of Shumailov et
    al.'s 'distribution contracting to the modes' for this task."""
    symbol_counts = torch.zeros(K)
    label_counts = torch.zeros(L)
    for it in items:
        for s in it["symbol_tokens"].tolist():
            symbol_counts[s] += 1
        for l in it["label_tokens"].tolist():
            label_counts[l] += 1
        label_counts[int(it["query_label"])] += 1
    return symbol_counts, label_counts


def entropy_from_counts(counts: torch.Tensor) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return -(p * p.log()).sum().item()


def kl_divergence(counts_p: torch.Tensor, counts_q: torch.Tensor, eps: float = 1e-8) -> float:
    """KL(p || q) between two usage-count histograms over the same support."""
    p = (counts_p / counts_p.sum()).clamp_min(eps)
    q = (counts_q / counts_q.sum()).clamp_min(eps)
    return (p * (p / q).log()).sum().item()


def distribution_stats(
    items: list[dict],
    K: int,
    L: int,
    ref_symbol_counts: torch.Tensor | None = None,
    ref_label_counts: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    symbol_counts, label_counts = symbol_label_counts(items, K, L)
    stats = {
        "symbol_entropy": entropy_from_counts(symbol_counts),
        "label_entropy": entropy_from_counts(label_counts),
        "symbol_entropy_normalized": entropy_from_counts(symbol_counts) / math.log(K),
        "label_entropy_normalized": entropy_from_counts(label_counts) / math.log(L),
    }
    if ref_symbol_counts is not None:
        stats["symbol_kl_vs_gen0"] = kl_divergence(symbol_counts, ref_symbol_counts)
    if ref_label_counts is not None:
        stats["label_kl_vs_gen0"] = kl_divergence(label_counts, ref_label_counts)
    return stats, symbol_counts, label_counts


def label_consistency_rate(items: list[dict]) -> float:
    """Fraction of repeated symbol occurrences (across a set of items) whose
    label matches that symbol's FIRST occurrence in the same context. 1.0
    for well-formed real data by construction; below 1.0 means a context
    gives the same symbol two different labels -- a structural corruption
    that a model solving the task by induction cannot fit."""
    total, consistent = 0, 0
    for item in items:
        first_label = {}
        for s, l in zip(item["symbol_tokens"][:-1].tolist(), item["label_tokens"].tolist()):
            if s not in first_label:
                first_label[s] = l
            else:
                total += 1
                consistent += int(l == first_label[s])
    return consistent / total if total > 0 else float("nan")


def structure_stats(items: list[dict], n_classes: int, burstiness: int) -> dict:
    """Sequence-STRUCTURE statistics of a training pool, the things marginal
    symbol/label usage entropy and KL cannot see. For real data every one of
    these is exactly 1.0 by construction, so any shortfall is corruption
    introduced by a generating model:

      frac_distinct_ok     context has exactly `n_classes` distinct symbols
      frac_repeats_ok      every distinct context symbol appears exactly `burstiness` times
      frac_label_consistent  repeated occurrences of a symbol all carry its first label
      frac_label_injective   distinct symbols carry distinct labels
      frac_query_in_context  the query symbol occurs in the context
      frac_query_label_ok    the query's target label equals the label its symbol
                             carries in the context (among sequences where it occurs)
      frac_fully_valid       all of the above hold simultaneously
    """
    n = len(items)
    c = {k: 0 for k in ["distinct", "repeats", "consistent", "injective", "qin", "qok", "valid"]}
    n_qin = 0
    for item in items:
        syms = item["symbol_tokens"][:-1].tolist()
        labs = item["label_tokens"].tolist()
        query = int(item["symbol_tokens"][-1])
        first, counts = {}, {}
        consistent = True
        for s, l in zip(syms, labs):
            counts[s] = counts.get(s, 0) + 1
            if s not in first:
                first[s] = l
            elif first[s] != l:
                consistent = False
        distinct_ok = len(counts) == n_classes
        repeats_ok = all(v == burstiness for v in counts.values())
        injective = len(set(first.values())) == len(first)
        q_in = query in first
        q_ok = q_in and int(item["query_label"]) == first[query]
        c["distinct"] += distinct_ok
        c["repeats"] += repeats_ok
        c["consistent"] += consistent
        c["injective"] += injective
        c["qin"] += q_in
        n_qin += q_in
        c["qok"] += q_ok
        c["valid"] += distinct_ok and repeats_ok and consistent and injective and q_in and q_ok
    return {
        "frac_distinct_ok": c["distinct"] / n,
        "frac_repeats_ok": c["repeats"] / n,
        "frac_label_consistent": c["consistent"] / n,
        "label_consistency_rate": label_consistency_rate(items),
        "frac_label_injective": c["injective"] / n,
        "frac_query_in_context": c["qin"] / n,
        "frac_query_label_ok": (c["qok"] / n_qin) if n_qin else float("nan"),
        "frac_fully_valid": c["valid"] / n,
    }


def pool_distinct_fraction(items: list[dict]) -> float:
    """Fraction of a pool's sequences that are distinct (1.0 = no duplicates).
    A with-replacement resample of a size-matched pool gives ~1 - 1/e = 0.63."""
    seen = {
        (tuple(it["symbol_tokens"].tolist()), tuple(it["label_tokens"].tolist()), int(it["query_label"]))
        for it in items
    }
    return len(seen) / len(items)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def prev_token_score(attn_maps: list[torch.Tensor], offset: int = 1) -> dict:
    """Mean attention mass each head places on the position `offset` steps
    back, i.e. how 'previous-token-head'-like it is (Olsson et al., 2022).
    Averaged over batch and all valid (t >= offset) query positions."""
    result = {}
    all_vals = []
    for layer_idx, attn in enumerate(attn_maps):
        B, H, T, _ = attn.shape
        if T <= offset:
            continue
        # attn[b, h, t, t-offset] for t in [offset, T)
        idx = torch.arange(offset, T, device=attn.device)
        gathered = attn[:, :, idx, idx - offset]  # (B, H, T-offset)
        for h in range(H):
            v = gathered[:, h, :].mean().item()
            result[f"layer{layer_idx}_head{h}"] = v
            all_vals.append(v)
    result["overall_mean"] = sum(all_vals) / len(all_vals) if all_vals else float("nan")
    return result


def induction_score(attn_maps: list[torch.Tensor], symbol_tokens: torch.Tensor) -> dict:
    """Task-specific induction score: how much attention the final query
    position places on the *correct copy sources* -- the label positions
    that immediately follow earlier in-context occurrences of the same
    exemplar symbol as the query. This is the task-grounded analogue of the
    generic repeated-random-sequence induction score in Olsson et al. (2022)
    / Singh et al. (2024): a well-formed previous-token-head -> induction-head
    circuit should route most of the query's attention onto exactly these
    positions, since copying the label found there solves the task.
    A random/untrained head's baseline is roughly (matches / seq_len)."""
    B, m1 = symbol_tokens.shape
    m = m1 - 1
    query_symbol = symbol_tokens[:, -1]           # (B,)
    context_symbols = symbol_tokens[:, :m]         # (B, m)
    match_mask = (context_symbols == query_symbol.unsqueeze(1)).float()  # (B, m)
    label_positions = torch.arange(m, device=symbol_tokens.device) * 2 + 1
    query_pos = 2 * m

    result = {}
    all_vals = []
    for layer_idx, attn in enumerate(attn_maps):
        H = attn.shape[1]
        att_to_labels = attn[:, :, query_pos, label_positions]  # (B, H, m)
        masked = att_to_labels * match_mask.unsqueeze(1)
        score_per_bh = masked.sum(dim=-1)  # (B, H)
        for h in range(H):
            v = score_per_bh[:, h].mean().item()
            result[f"layer{layer_idx}_head{h}"] = v
            all_vals.append(v)
    result["overall_mean"] = sum(all_vals) / len(all_vals) if all_vals else float("nan")
    result["baseline_mean"] = (match_mask.sum(dim=-1) / (2 * m + 1)).mean().item()
    return result


def attention_entropy(attn_maps: list[torch.Tensor]) -> dict:
    """Mean entropy (nats) of each head's attention distribution, per layer.
    attn_maps[l]: (B, n_heads, seq_len, seq_len), rows sum to 1 over the
    causal-allowed support. Returns {'layer0_head0': ..., 'overall_mean': ...}."""
    result = {}
    all_vals = []
    for layer_idx, attn in enumerate(attn_maps):
        B, H, T, _ = attn.shape
        p = attn.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)  # (B, H, T)
        for h in range(H):
            v = ent[:, h, :].mean().item()
            result[f"layer{layer_idx}_head{h}"] = v
            all_vals.append(v)
    result["overall_mean"] = sum(all_vals) / len(all_vals) if all_vals else float("nan")
    return result
