"""ACDC (Conmy et al., 2023) edge-level circuit discovery for the exact attention-only model.

The model is a directed acyclic graph. SENDER nodes write into the residual stream: the embedding ``E`` (the interleaved
exemplar and label embeddings) and each attention head ``L{l}H{h}`` (its output after the output projection). RECEIVER
nodes read it: every head reads three times, as its query, key and value input (``L{l}H{h}.q|k|v``), and the unembedding
reads it once (``OUT``). An EDGE (receiver, sender) says that the receiver reads that sender's output. A receiver reads
the SUM of the senders before it, so an edge can be cut individually: its contribution is replaced by the sender's
activation on a CORRUPTED input (resample ablation), i.e. ACDC's patching of an edge.

For the 2-layer, 8-head attention-only model the graph has
    layer-1 heads read E                         8 x 3 x 1      =  24
    layer-2 heads read E and the layer-1 heads   8 x 3 x 9      = 216
    OUT reads E and every head                   1 + 8 + 8      =  17      -> 257 edges.
There are no MLP nodes and no biases in the attention blocks, so nothing else feeds the residual stream.

ACDC (Algorithm 1 of the paper) visits receivers from the output backwards and, for each incoming edge, removes it if doing
so raises KL(full model || pruned graph) by less than tau, otherwise keeps it. The metric is KL over the label distribution
at the query position, averaged over a batch of clean sequences. Two corruptions are supported:
    resample   every clean sequence is paired with a different random sequence (exemplars and labels both change);
    labels     the same exemplars with freshly drawn labels: only edges that carry LABEL information survive.

Two invariants are tests (tests/test_exact.py): with every edge kept the graph reproduces the model's own forward
pass; with every edge cut it reproduces the model on the corrupted input.

Differences from the paper's ACDC, stated plainly: corrupted inputs are other random task sequences (or the same
exemplars with redrawn labels) rather than hand-designed prompts; a head is patched at every position at once; one metric;
a small grid of thresholds. The graph and the greedy search are ACDC's.
"""

from __future__ import annotations

import dataclasses

import torch

from model import IHTransformer, apply_rope


def graph(n_layers: int, n_heads: int) -> list[tuple[str, list[str]]]:
    """Receivers in topological order, each with the senders it can read."""
    senders = ["E"]
    recv = []
    for l in range(1, n_layers + 1):
        for h in range(n_heads):
            for t in "qkv":
                recv.append((f"L{l}H{h}.{t}", list(senders)))
        senders = senders + [f"L{l}H{h}" for h in range(n_heads)]
    recv.append(("OUT", list(senders)))
    return recv


def all_edges(n_layers: int, n_heads: int) -> list[tuple[str, str]]:
    return [(r, s) for r, ss in graph(n_layers, n_heads) for s in ss]


def node_of(receiver: str) -> str:
    """'L2H3.k' -> 'L2H3'; 'OUT' -> 'OUT'."""
    return receiver.rsplit(".", 1)[0] if receiver.startswith("L") else receiver


@torch.no_grad()
def run_graph(model: IHTransformer, ex: torch.Tensor, labels: torch.Tensor, kept: dict | None = None,
              corrupt: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Forward pass through the edge graph (base-format input: 3 exemplars, 2 labels). ``kept[(receiver, sender)]`` False
    makes that edge read ``corrupt[sender]`` instead of the live activation. Returns (label logits at the query, {sender: activation})."""
    model.eval()
    cfg = model.cfg
    H, D = cfg.n_heads, cfg.d_model
    hd = D // H
    x0 = model.interleave(model.example_embed(ex), model.label_embed(labels))
    T = x0.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    out = {"E": x0}

    def read(receiver: str, names: list[str]) -> torch.Tensor:
        total = 0
        for s in names:
            live = kept is None or kept[(receiver, s)]
            total = total + (out[s] if live else corrupt[s])
        return total

    senders = ["E"]
    for l, block in enumerate(model.blocks, start=1):
        W = block.qkv.weight                                  # (3D, D): rows [q | k | v], each split into H heads of hd
        Wo = block.proj.weight.view(D, H, hd)                 # (D_out, H, hd)
        for h in range(H):
            proj = {}
            for ti, t in enumerate("qkv"):
                z = block.ln(read(f"L{l}H{h}.{t}", senders))
                rows = slice(ti * D + h * hd, ti * D + (h + 1) * hd)
                proj[t] = z @ W[rows].T                        # (B, T, hd)
            q = apply_rope(proj["q"][:, None], block.cos[:T], block.sin[:T])[:, 0]
            k = apply_rope(proj["k"][:, None], block.cos[:T], block.sin[:T])[:, 0]
            scores = (q @ k.transpose(-2, -1)) * hd ** -0.5
            attn = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            out[f"L{l}H{h}"] = (attn @ proj["v"]) @ Wo[:, h, :].T       # (B, T, D)
        senders = senders + [f"L{l}H{h}" for h in range(H)]
    final = model.ln_f(read("OUT", senders))
    return model.label_head(final[:, 4]), out


def kl_from_clean(clean_logp: torch.Tensor, logits: torch.Tensor) -> float:
    return (clean_logp.exp() * (clean_logp - torch.log_softmax(logits, dim=-1))).sum(-1).mean().item()


def relabel_corruption(labels: torch.Tensor, n_labels: int, seed: int = 0) -> torch.Tensor:
    """Fresh random distinct labels (la', lb') for each sequence: the same exemplars with the labels redrawn."""
    g = torch.Generator().manual_seed(seed)
    out = torch.empty_like(labels)
    for i in range(labels.shape[0]):
        out[i] = torch.randperm(n_labels, generator=g)[: labels.shape[1]]
    return out


@dataclasses.dataclass
class Circuit:
    tau: float
    kept: dict
    kl: float
    accuracy: float
    full_accuracy: float
    live_edges: list
    live_nodes: list

    @property
    def n_live_edges(self) -> int:
        return len(self.live_edges)


def live_subgraph(kept: dict, n_layers: int, n_heads: int) -> tuple[list, list]:
    """Edges and nodes that can still influence the output (walk back from OUT through kept edges)."""
    recvs = graph(n_layers, n_heads)
    live = {"OUT"}
    for recv, senders in reversed(recvs):
        if node_of(recv) in live:
            live.update(s for s in senders if kept[(recv, s)])
    edges = [(r, s) for r, ss in recvs if node_of(r) in live for s in ss if kept[(r, s)]]
    return edges, sorted(live - {"OUT"}, key=lambda n: (n != "E", n))


@torch.no_grad()
def acdc(model: IHTransformer, ex: torch.Tensor, labels: torch.Tensor, query_label: torch.Tensor,
         c_ex: torch.Tensor, c_labels: torch.Tensor, tau: float) -> Circuit:
    L, H = model.cfg.n_layers, model.cfg.n_heads
    clean_logits, _ = run_graph(model, ex, labels)
    clean_logp = torch.log_softmax(clean_logits, dim=-1)
    _, corrupt = run_graph(model, c_ex, c_labels)
    kept = {e: True for e in all_edges(L, H)}
    current = 0.0
    for recv, senders in reversed(graph(L, H)):
        for s in reversed(senders):
            kept[(recv, s)] = False
            new = kl_from_clean(clean_logp, run_graph(model, ex, labels, kept, corrupt)[0])
            if new - current < tau:
                current = new
            else:
                kept[(recv, s)] = True
    logits, _ = run_graph(model, ex, labels, kept, corrupt)
    edges, nodes = live_subgraph(kept, L, H)
    return Circuit(tau=tau, kept=kept, kl=kl_from_clean(clean_logp, logits),
                   accuracy=(logits.argmax(-1) == query_label).float().mean().item(),
                   full_accuracy=(clean_logits.argmax(-1) == query_label).float().mean().item(),
                   live_edges=edges, live_nodes=nodes)


def describe(c: Circuit, n_layers: int = 2) -> dict:
    """Flat summary: size, which heads are live, composition edges from layer-1 heads into layer-2 q / k / v inputs."""
    nodes = set(c.live_nodes)
    l1 = sorted(n for n in nodes if n.startswith("L1H"))
    l2 = sorted(n for n in nodes if n.startswith("L2H"))
    comp = {t: [(s, r) for r, s in c.live_edges if r.startswith("L2H") and r.endswith("." + t) and s.startswith("L1H")] for t in "qkv"}
    return {"tau": c.tau, "n_edges": c.n_live_edges, "kl": c.kl, "acc": c.accuracy, "full_acc": c.full_accuracy,
            "heads_L1": ";".join(l1), "heads_L2": ";".join(l2), "n_heads_L1": len(l1), "n_heads_L2": len(l2),
            "kcomp": int(bool(comp["k"])), "qcomp": int(bool(comp["q"])), "vcomp": int(bool(comp["v"])),
            "kcomp_edges": ";".join(f"{s}->{r}" for s, r in comp["k"]), "edges": ";".join(f"{s}->{r}" for r, s in c.live_edges)}
