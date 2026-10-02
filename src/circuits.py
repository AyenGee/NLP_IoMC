"""Simplified ACDC-style circuit discovery via per-head zero-ablation.

Stretch goal, deliberately scoped down from full ACDC (Conmy et al., 2023).
Full ACDC performs iterative greedy pruning over a model's fine-grained
computational graph (attention-head-to-head, head-to-MLP, etc. edges),
using patching/resampling ablation and a threshold on a task metric to
decide which edges survive. That is substantial extra machinery for a
model this size, and is not implemented here. What IS implemented: each
attention head is zero-ablated independently (holding everything else
intact) and the resulting drop in query-label accuracy on a held-out batch
is measured. This identifies which individual heads are causally
necessary for the task -- a coarse "node-level" precursor to a true edge-
level circuit -- and is exactly the kind of thing that's honestly reported
as an approximation, not a faithful ACDC reimplementation, in the write-up.

nn.MultiheadAttention does not expose per-head outputs before its output
projection, so this module reimplements the same self-attention computation
manually (reusing the *trained* in_proj / out_proj weights unchanged) to
get access to each head's pre-projection output for ablation. This is
validated against the real module's output before being trusted (see
`_check_manual_attention_matches`).
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from model import InductionTransformer, ModelOutput


@torch.no_grad()
def manual_attention_forward(
    attn: torch.nn.MultiheadAttention,
    x: torch.Tensor,
    causal_mask: torch.Tensor,
    n_heads: int,
    ablate_heads: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplements nn.MultiheadAttention(x, x, x, attn_mask=causal_mask)
    for the self-attention, batch_first=True case, using the module's own
    trained weights, with the option to zero-ablate a set of head indices
    before the output projection. Returns (output, attn_weights) matching
    the shapes MultiheadAttention itself would return with
    average_attn_weights=False."""
    B, T, D = x.shape
    head_dim = D // n_heads
    ablate_heads = ablate_heads or set()

    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)  # (B, T, 3D)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, H, T, hd)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, H, T, T)
    scores = scores + causal_mask[None, None, :, :]
    attn_weights = torch.softmax(scores, dim=-1)
    head_out = attn_weights @ v  # (B, H, T, hd)

    if ablate_heads:
        head_out = head_out.clone()
        for h in ablate_heads:
            head_out[:, h, :, :] = 0.0

    concat = head_out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
    output = F.linear(concat, attn.out_proj.weight, attn.out_proj.bias)
    return output, attn_weights


@torch.no_grad()
def _check_manual_attention_matches(model: InductionTransformer, x: torch.Tensor, causal_mask: torch.Tensor, atol: float = 1e-4) -> float:
    """Sanity check: manual (no-ablation) attention output must match
    nn.MultiheadAttention's own output for every block. Returns the max
    absolute difference found (should be << atol)."""
    max_diff = 0.0
    h = x
    for block in model.blocks:
        ln = block.ln1(h)
        real_out, _ = block.attn(ln, ln, ln, attn_mask=causal_mask, need_weights=False)
        manual_out, _ = manual_attention_forward(block.attn, ln, causal_mask, model.mcfg.n_heads)
        max_diff = max(max_diff, (real_out - manual_out).abs().max().item())
        h = h + block.dropout(real_out)
        h = h + block.dropout(block.mlp(block.ln2(h)))
    return max_diff


@torch.no_grad()
def forward_with_ablation(
    model: InductionTransformer,
    symbol_tokens: torch.Tensor,
    label_tokens: torch.Tensor,
    ablate: dict[int, set[int]] | None = None,
) -> ModelOutput:
    """Forward pass identical to InductionTransformer.forward, except
    attention heads named in `ablate` ({layer_idx: {head_idx, ...}}) are
    zero-ablated. Used to measure each head's causal contribution to the
    task by comparing accuracy with vs. without it."""
    ablate = ablate or {}
    x = model._interleave(symbol_tokens, label_tokens)
    seq_len = x.shape[1]
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)

    for layer_idx, block in enumerate(model.blocks):
        ln = block.ln1(x)
        attn_out, _ = manual_attention_forward(
            block.attn, ln, causal_mask, model.mcfg.n_heads, ablate_heads=ablate.get(layer_idx)
        )
        x = x + block.dropout(attn_out)
        x = x + block.dropout(block.mlp(block.ln2(x)))
    x = model.ln_f(x)
    label_logits = model.label_head(x[:, 0::2, :])
    symbol_logits = model.symbol_head(x[:, 1::2, :])
    return ModelOutput(label_logits=label_logits, symbol_logits=symbol_logits, hidden=x, attn_maps=None)


@dataclasses.dataclass
class HeadImportance:
    layer: int
    head: int
    baseline_acc: float
    ablated_acc: float
    delta_acc: float  # baseline - ablated; large positive = causally important


@torch.no_grad()
def head_ablation_sweep(
    model: InductionTransformer, symbol_tokens: torch.Tensor, label_tokens: torch.Tensor, query_label: torch.Tensor
) -> list[HeadImportance]:
    """Zero-ablate each head one at a time and measure the drop in
    query-label accuracy on the given batch. This is the project's
    simplified stand-in for an ACDC sweep -- see module docstring."""
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.mcfg.n_heads

    baseline_out = forward_with_ablation(model, symbol_tokens, label_tokens, ablate={})
    baseline_acc = (baseline_out.label_logits[:, -1, :].argmax(-1) == query_label).float().mean().item()

    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            out = forward_with_ablation(model, symbol_tokens, label_tokens, ablate={layer: {head}})
            acc = (out.label_logits[:, -1, :].argmax(-1) == query_label).float().mean().item()
            results.append(
                HeadImportance(layer=layer, head=head, baseline_acc=baseline_acc, ablated_acc=acc, delta_acc=baseline_acc - acc)
            )
    return results


@torch.no_grad()
def layer_ablation_summary(
    model: InductionTransformer, symbol_tokens: torch.Tensor, label_tokens: torch.Tensor, query_label: torch.Tensor
) -> dict:
    """Jointly ablate every head in each layer (rather than one head at a
    time) and report the accuracy drop. Individual-head ablation can
    under-report a layer's importance if its function is redundantly
    distributed across heads (as we found for this task's layer-1
    induction heads: no single one matters much alone, but ablating all of
    them together collapses accuracy to near chance). Cheap enough (one
    forward pass per layer) to track across every collapse generation."""
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.mcfg.n_heads

    def acc(ablate):
        out = forward_with_ablation(model, symbol_tokens, label_tokens, ablate=ablate)
        return (out.label_logits[:, -1, :].argmax(-1) == query_label).float().mean().item()

    baseline = acc({})
    result = {"baseline_acc": baseline}
    for layer in range(n_layers):
        ablated = acc({layer: set(range(n_heads))})
        result[f"layer{layer}_all_heads_ablated_acc"] = ablated
        result[f"layer{layer}_all_heads_delta_acc"] = baseline - ablated
    return result
