"""Minimal decoder-only transformer for the symbolic in-context-learning task.

Depth defaults to 2 layers to mirror the minimal circuit depth Olsson et al.
(2022) and Singh et al. (2024) identify for induction: a previous-token head
in an earlier layer feeding an induction head in a later layer. Width is kept
small deliberately -- big enough to solve the task, small enough that
attention patterns and embeddings stay interpretable by inspection.

Sequence layout (see data.py): s_1, l_1, s_2, l_2, ..., s_m, l_m, s_query.
Two output heads are applied at every position:
  - label_head, applied at symbol positions (0, 2, 4, ...), predicts the
    label associated with that symbol. At the final position (s_query) this
    *is* the base task. At earlier symbol positions it is only solvable
    once that symbol's label has appeared in-context, i.e. after its first
    occurrence -- the extended task's auxiliary "corresponding label" loss.
  - symbol_head, applied at label positions (1, 3, 5, ...), predicts the
    symbol that follows -- the extended task's auxiliary "next symbol" loss.
This keeps the base loss (label_head at the final position only) and the
auxiliary losses (both heads at all other positions) cleanly separable so
they can be ablated independently in train.py.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

from data import TaskConfig


@dataclasses.dataclass
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0


@dataclasses.dataclass
class ModelOutput:
    label_logits: torch.Tensor    # (B, m+1, L) at symbol positions
    symbol_logits: torch.Tensor   # (B, m, K)   at label positions
    hidden: torch.Tensor          # (B, seq_len, d_model) post final LayerNorm
    attn_maps: list[torch.Tensor] | None  # per layer: (B, n_heads, seq_len, seq_len)


class TransformerBlock(nn.Module):
    def __init__(self, mcfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(mcfg.d_model)
        self.attn = nn.MultiheadAttention(
            mcfg.d_model, mcfg.n_heads, dropout=mcfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(mcfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(mcfg.d_model, mcfg.d_ff),
            nn.GELU(),
            nn.Linear(mcfg.d_ff, mcfg.d_model),
        )
        self.dropout = nn.Dropout(mcfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, need_weights: bool):
        h = self.ln1(x)
        attn_out, attn_w = self.attn(
            h, h, h, attn_mask=attn_mask, need_weights=need_weights, average_attn_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x, attn_w


class InductionTransformer(nn.Module):
    def __init__(self, tcfg: TaskConfig, mcfg: ModelConfig):
        super().__init__()
        self.tcfg = tcfg
        self.mcfg = mcfg
        self.symbol_embed = nn.Embedding(tcfg.K, mcfg.d_model)
        self.label_embed = nn.Embedding(tcfg.L, mcfg.d_model)
        self.pos_embed = nn.Embedding(tcfg.seq_len, mcfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(mcfg) for _ in range(mcfg.n_layers)])
        self.ln_f = nn.LayerNorm(mcfg.d_model)
        self.label_head = nn.Linear(mcfg.d_model, tcfg.L)
        self.symbol_head = nn.Linear(mcfg.d_model, tcfg.K)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _interleave(self, symbol_tokens: torch.Tensor, label_tokens: torch.Tensor) -> torch.Tensor:
        B, m1 = symbol_tokens.shape
        m = label_tokens.shape[1]
        sym_emb = self.symbol_embed(symbol_tokens)  # (B, m+1, d)
        lab_emb = self.label_embed(label_tokens)    # (B, m, d)
        if m == 0:
            paired = sym_emb.new_zeros(B, 0, self.mcfg.d_model)
        else:
            paired = torch.stack([sym_emb[:, :m, :], lab_emb], dim=2).reshape(B, 2 * m, -1)
        x = torch.cat([paired, sym_emb[:, m:, :]], dim=1)  # (B, 2m+1, d)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)[None, :, :]

    def forward(
        self, symbol_tokens: torch.Tensor, label_tokens: torch.Tensor, return_attention: bool = False
    ) -> ModelOutput:
        x = self._interleave(symbol_tokens, label_tokens)
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        attn_maps = [] if return_attention else None
        for block in self.blocks:
            x, attn_w = block(x, causal_mask, need_weights=return_attention)
            if return_attention:
                attn_maps.append(attn_w)
        x = self.ln_f(x)
        label_logits = self.label_head(x[:, 0::2, :])   # (B, m+1, L)
        symbol_logits = self.symbol_head(x[:, 1::2, :])  # (B, m, K)
        return ModelOutput(
            label_logits=label_logits, symbol_logits=symbol_logits, hidden=x, attn_maps=attn_maps
        )

    @torch.no_grad()
    def generate_batch(self, batch_size: int, device: torch.device, temperature: float = 1.0) -> list[dict]:
        """Autoregressively sample `batch_size` full synthetic sequences from
        the model itself (used by collapse.py for the extended task): at each
        label slot, sample the next symbol from symbol_head; at each symbol
        slot, sample the label for it from label_head. The very first symbol
        has no preceding context to condition on, so it is drawn uniformly --
        everything after that is the model's own (possibly wrong / skewed)
        autoregressive rollout, which is exactly the channel through which
        collapse can occur. Batched (rather than one sequence at a time) so
        generating a whole generation's synthetic pool is tractable.

        Returns a list of dicts with the same keys as
        SymbolicICLDataset.__getitem__, so the output can be fed straight
        into ListDataset / mix_datasets.
        """
        cfg = self.tcfg
        m = cfg.context_len
        symbols = torch.randint(cfg.K, (batch_size, 1), device=device)
        labels = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        for _ in range(m):
            out = self.forward(symbols, labels)
            next_label = torch.multinomial(
                torch.softmax(out.label_logits[:, -1, :] / temperature, dim=-1), 1
            )
            labels = torch.cat([labels, next_label], dim=1)
            out = self.forward(symbols, labels)
            next_symbol = torch.multinomial(
                torch.softmax(out.symbol_logits[:, -1, :] / temperature, dim=-1), 1
            )
            symbols = torch.cat([symbols, next_symbol], dim=1)
        out = self.forward(symbols, labels)
        query_label = torch.multinomial(
            torch.softmax(out.label_logits[:, -1, :] / temperature, dim=-1), 1
        )
        items = []
        for b in range(batch_size):
            items.append(
                {
                    "symbol_tokens": symbols[b].cpu(),
                    "label_tokens": labels[b].cpu(),
                    "next_symbol_target": symbols[b, 1:].cpu().clone(),
                    "next_label_target": torch.cat([labels[b], query_label[b]]).cpu(),
                    "query_label": query_label[b, 0].cpu(),
                }
            )
        return items
