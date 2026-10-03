"""Edge-level automated circuit discovery (ACDC; Conmy et al., 2023) for the
small pre-LayerNorm transformer in model.py.

The model is viewed as a directed acyclic graph. SENDER nodes write into the
residual stream: the embedding `E` (token + position), each attention head
`A<l>.h<h>` (its output after the output projection) and each MLP `M<l>`.
RECEIVER nodes read the stream: every head reads it three times, as its query,
key and value input (`A<l>.h<h>.q|k|v`); each MLP reads it once (`M<l>`), and
the unembedding reads it once (`OUT`). An EDGE (receiver, sender) says that the
receiver reads that sender's output. Because every receiver reads the *sum* of
the senders before it, an edge can be cut individually: its contribution is
replaced by the sender's activation on a CORRUPTED input (resample ablation), so
the receiver sees the clean contributions of the kept edges and the corrupted
contribution of the cut ones. This is exactly ACDC's patching of an edge.

For this 2-layer, 4-head model the graph has 110 edges:
    heads of layer 0 read E (12);     M0 reads E and layer-0 heads (5);
    heads of layer 1 read E, layer-0 heads and M0 (72);
    M1 reads everything before it (10);   OUT reads everything (11).

ACDC (Algorithm 1 of the paper) visits the receivers from the output backwards
and, for each incoming edge, removes it if doing so raises the divergence from
the full model's output by less than a threshold tau, otherwise keeps it. The
metric is KL(full model || pruned graph) over the label distribution at the
final (query) position, averaged over a batch of clean sequences; the corrupted
input for each clean sequence is a different random sequence from the same
task distribution. What survives is the circuit for that model at that tau.

Two invariants make the implementation checkable (both are tests in
tests/test_pipeline.py): with every edge kept the graph reproduces the model's
own forward pass, and with every edge cut it reproduces the model's output on the
corrupted input.

Differences from the paper, honestly stated: no dataset of corrupted prompts
beyond a shuffled copy of the clean batch, no per-token position patching (a head
is patched at every position at once), and a single metric. The graph and the
search are otherwise ACDC's.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from model import InductionTransformer


def graph_nodes(n_layers: int, n_heads: int) -> dict:
    """Ordered sender nodes and the (receiver, senders) lists, in topological order."""
    senders = ["E"]
    receivers = []   # (receiver name, list of sender names it can read)
    for l in range(n_layers):
        heads = [f"A{l}.h{h}" for h in range(n_heads)]
        for h in range(n_heads):
            for t in "qkv":
                receivers.append((f"A{l}.h{h}.{t}", list(senders)))
        senders = senders + heads
        receivers.append((f"M{l}", list(senders)))
        senders = senders + [f"M{l}"]
    receivers.append(("OUT", list(senders)))
    return {"senders": senders, "receivers": receivers}


def all_edges(n_layers: int, n_heads: int) -> list[tuple[str, str]]:
    return [(r, s) for r, ss in graph_nodes(n_layers, n_heads)["receivers"] for s in ss]


def node_of(receiver: str) -> str:
    """The node a receiver belongs to: 'A1.h2.k' -> 'A1.h2'; 'M0' -> 'M0'; 'OUT' -> 'OUT'."""
    return receiver.rsplit(".", 1)[0] if receiver.startswith("A") else receiver


@torch.no_grad()
def run_graph(
    model: InductionTransformer,
    symbol_tokens: torch.Tensor,
    label_tokens: torch.Tensor,
    kept: dict | None = None,
    corrupt: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Forward pass through the edge graph. `kept[(receiver, sender)]` False means
    that edge reads `corrupt[sender]` instead of the live activation. Returns
    (label logits at the final position, {sender: activation})."""
    model.eval()
    D, H = model.mcfg.d_model, model.mcfg.n_heads
    hd = D // H
    x0 = model._interleave(symbol_tokens, label_tokens)
    T = x0.shape[1]
    mask = torch.triu(torch.full((T, T), float("-inf"), device=x0.device), diagonal=1)
    out = {"E": x0}
    biases = [blk.attn.out_proj.bias for blk in model.blocks]

    def read(receiver: str, sender_names: list[str], const: torch.Tensor) -> torch.Tensor:
        total = const
        for s in sender_names:
            use_live = kept is None or kept[(receiver, s)]
            total = total + (out[s] if use_live else corrupt[s])
        return total

    senders = ["E"]
    const = torch.zeros(D, device=x0.device)
    for l, block in enumerate(model.blocks):
        W, b = block.attn.in_proj_weight, block.attn.in_proj_bias
        Wo = block.attn.out_proj.weight                       # (D, D)
        for h in range(H):
            proj = {}
            for ti, t in enumerate("qkv"):
                ln = block.ln1(read(f"A{l}.h{h}.{t}", senders, const))
                rows = slice(ti * D + h * hd, ti * D + (h + 1) * hd)
                proj[t] = F.linear(ln, W[rows], b[rows])      # (B, T, hd)
            scores = proj["q"] @ proj["k"].transpose(-2, -1) / math.sqrt(hd) + mask
            head = torch.softmax(scores, dim=-1) @ proj["v"]  # (B, T, hd)
            out[f"A{l}.h{h}"] = head @ Wo[:, h * hd:(h + 1) * hd].T
        senders = senders + [f"A{l}.h{h}" for h in range(H)]
        const = const + biases[l]
        out[f"M{l}"] = block.mlp(block.ln2(read(f"M{l}", senders, const)))
        senders = senders + [f"M{l}"]
    final = model.ln_f(read("OUT", senders, const))
    logits = model.label_head(final[:, 0::2, :])[:, -1, :]
    return logits, out


def relabel_corruption(symbol_tokens: torch.Tensor, label_tokens: torch.Tensor, n_labels: int, seed: int = 0) -> torch.Tensor:
    """Corrupted LABELS for a clean batch: the same symbols, but each sequence gets a fresh random
    injective symbol -> label map. Patching with this input leaves everything that only carries symbol
    identity (matching the query to earlier occurrences) unchanged, so the circuit ACDC finds is the one
    that carries LABEL information to the output. Compare with the default corruption (a different
    random sequence), which changes symbols and labels alike."""
    g = torch.Generator().manual_seed(seed)
    out = label_tokens.clone()
    for i in range(symbol_tokens.shape[0]):
        ctx = symbol_tokens[i, :-1]
        uniq = ctx.unique()
        new = torch.randperm(n_labels, generator=g)[: len(uniq)]
        table = {int(u): int(n) for u, n in zip(uniq, new)}
        out[i] = torch.tensor([table[int(x)] for x in ctx], dtype=label_tokens.dtype)
    return out


def kl_from_clean(clean_logp: torch.Tensor, logits: torch.Tensor) -> float:
    """Mean over the batch of KL(clean || pruned) in nats."""
    logq = torch.log_softmax(logits, dim=-1)
    return (clean_logp.exp() * (clean_logp - logq)).sum(-1).mean().item()


@dataclasses.dataclass
class Circuit:
    tau: float
    kept: dict                      # (receiver, sender) -> bool, after ACDC
    kl: float                       # KL(full || circuit) on the discovery batch
    accuracy: float                 # query-label accuracy of the circuit alone
    full_accuracy: float
    live_edges: list                # kept edges that can influence the output
    live_nodes: list

    @property
    def n_live_edges(self) -> int:
        return len(self.live_edges)


def live_subgraph(kept: dict, n_layers: int, n_heads: int) -> tuple[list, list]:
    """Edges and nodes that can still influence the output: walk back from OUT
    through kept edges. ACDC can leave kept edges into nodes whose own output is
    no longer read by anything; those do nothing and are not part of the circuit."""
    graph = graph_nodes(n_layers, n_heads)
    recvs = graph["receivers"]
    live_nodes = {"OUT"}
    for recv, senders in reversed(recvs):
        if node_of(recv) not in live_nodes:
            continue
        for s in senders:
            if kept[(recv, s)]:
                live_nodes.add(s)
    live_edges = [(r, s) for r, ss in recvs if node_of(r) in live_nodes for s in ss if kept[(r, s)]]
    order = ["E"] + [n for r, _ in recvs for n in [node_of(r)] if n != "OUT"]
    ordered = []
    for n in order:
        if n in live_nodes and n not in ordered:
            ordered.append(n)
    return live_edges, ordered


@torch.no_grad()
def acdc(
    model: InductionTransformer,
    symbol_tokens: torch.Tensor,
    label_tokens: torch.Tensor,
    query_label: torch.Tensor,
    corrupt_symbol_tokens: torch.Tensor,
    corrupt_label_tokens: torch.Tensor,
    tau: float,
) -> Circuit:
    L, H = model.mcfg.n_layers, model.mcfg.n_heads
    clean_logits, _ = run_graph(model, symbol_tokens, label_tokens)
    clean_logp = torch.log_softmax(clean_logits, dim=-1)
    _, corrupt = run_graph(model, corrupt_symbol_tokens, corrupt_label_tokens)

    kept = {e: True for e in all_edges(L, H)}
    current = 0.0
    for recv, senders in reversed(graph_nodes(L, H)["receivers"]):
        for s in reversed(senders):
            kept[(recv, s)] = False
            logits, _ = run_graph(model, symbol_tokens, label_tokens, kept, corrupt)
            new = kl_from_clean(clean_logp, logits)
            if new - current < tau:
                current = new            # removing the edge costs less than tau: drop it
            else:
                kept[(recv, s)] = True
    logits, _ = run_graph(model, symbol_tokens, label_tokens, kept, corrupt)
    live_edges, live_nodes = live_subgraph(kept, L, H)
    return Circuit(
        tau=tau, kept=kept, kl=kl_from_clean(clean_logp, logits),
        accuracy=(logits.argmax(-1) == query_label).float().mean().item(),
        full_accuracy=(clean_logits.argmax(-1) == query_label).float().mean().item(),
        live_edges=live_edges, live_nodes=live_nodes,
    )


def describe(circuit: Circuit, n_layers: int = 2, n_heads: int = 4) -> dict:
    """Flat, CSV-friendly summary of a discovered circuit."""
    edges, nodes = circuit.live_edges, set(circuit.live_nodes)
    heads_by_layer = {l: sorted(n for n in nodes if n.startswith(f"A{l}.")) for l in range(n_layers)}
    # composition: a head of an earlier layer feeding the q/k/v input of a later head
    comp = {t: [(s, r) for r, s in edges if r.startswith("A") and r.endswith("." + t) and s.startswith("A")] for t in "qkv"}
    return {
        "acdc_tau": circuit.tau,
        "acdc_n_edges": circuit.n_live_edges,
        "acdc_n_edges_total": len(all_edges(n_layers, n_heads)),
        "acdc_kl": circuit.kl,
        "acdc_acc": circuit.accuracy,
        "acdc_full_acc": circuit.full_accuracy,
        "acdc_heads": ";".join(n for l in range(n_layers) for n in heads_by_layer[l]),
        "acdc_n_heads": sum(len(v) for v in heads_by_layer.values()),
        "acdc_n_heads_l0": len(heads_by_layer[0]),
        "acdc_n_heads_l1": len(heads_by_layer[n_layers - 1]) if n_layers > 1 else 0,
        "acdc_mlp0": int("M0" in nodes), "acdc_mlp1": int(f"M{n_layers - 1}" in nodes) if n_layers > 1 else 0,
        "acdc_kcomp": int(len(comp["k"]) > 0), "acdc_qcomp": int(len(comp["q"]) > 0), "acdc_vcomp": int(len(comp["v"]) > 0),
        "acdc_kcomp_edges": ";".join(f"{s}->{r}" for s, r in comp["k"]),
        "acdc_edges": ";".join(f"{s}->{r}" for r, s in edges),
    }
