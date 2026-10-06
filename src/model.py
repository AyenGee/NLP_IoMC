"""The exact architecture of Singh et al. (2024), Appendix A, written from the paper and from the
reference repository's model description (``models.py`` / ``main_utils.py``, read for specifications
only; no code is copied). PyTorch instead of the paper's JAX.

    inputs   exemplars -> Linear(512, 64, bias=True)            (``example_embed``)
             labels    -> Embedding(5, 64)                       (``label_embed``; the repository uses a
                                                                  bias-free Linear on a one-hot, which is
                                                                  the same lookup)
             the two streams are interleaved; NO absolute positional embedding (RoPE inside attention)
    blocks   2 layers, ATTENTION-ONLY (no MLP; ``mlp_ratio=None`` in the repository)
             pre-LayerNorm -> qkv = Linear(64, 192, bias=False) -> 8 heads of dim 8 -> RoPE on q and k
             (theta = 10000, interleaved pairs) -> causal softmax(QK^T / sqrt(8)) -> Linear(64, 64, bias=False)
             -> residual add
    output   final LayerNorm -> label_head Linear(64, 5, bias=True) read at the query position (base) or at
             positions 4 and 6 (extended); symbol_head Linear(64, 50, bias=True) at position 5 (extended)
    init     Linear weights ~ truncated normal with std 1/sqrt(fan_in) (JAX ``lecun_normal``), biases 0;
             label embedding ~ 0.02 x truncated normal, as in the repository.

Optional ``mlp_ratio`` (default None = the paper) adds a pre-LN GELU MLP to each block; it exists only for
the architecture-sensitivity experiment.

The forward pass can return everything the analyses need: attention patterns, each head's output (before the
output projection mixes them), residual-stream activations; and it accepts ablation / pattern-override hooks.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import FEATURE_DIM, N_LABELS, TRAIN_CLASSES

_TRUNC_STD_CORRECTION = 0.87962566   # std of a unit normal truncated to [-2, 2]


@dataclasses.dataclass
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 8
    d_in: int = FEATURE_DIM
    n_labels: int = N_LABELS
    n_classes: int = len(TRAIN_CLASSES)
    extended: bool = False                 # adds symbol_head and reads two more positions
    mlp_ratio: float | None = None          # None = attention-only (the paper)
    rope_theta: float = 10000.0
    max_len: int = 16


def _trunc_normal_(w: torch.Tensor, std: float) -> None:
    """Truncated normal on [-2, 2] x std_param whose RESULTING std equals ``std`` (JAX variance scaling)."""
    s = std / _TRUNC_STD_CORRECTION
    nn.init.trunc_normal_(w, mean=0.0, std=s, a=-2 * s, b=2 * s)


def rope_tables(max_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin tables (max_len, head_dim) with each frequency duplicated for its (even, odd) pair."""
    assert head_dim % 2 == 0
    freq = theta ** (-2.0 * torch.arange(head_dim // 2, dtype=torch.float32) / head_dim)     # (hd/2,)
    angle = torch.arange(max_len, dtype=torch.float32)[:, None] * freq[None, :]                 # (T, hd/2)
    return angle.cos().repeat_interleave(2, dim=-1), angle.sin().repeat_interleave(2, dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs (x_2j, x_2j+1) by the position's angle. x: (..., T, hd); cos/sin: (T, hd)."""
    rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(x.shape)
    return x * cos + rotated * sin


class AttentionOnlyBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        self.head_dim = d // h
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        if cfg.mlp_ratio is not None:
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, int(d * cfg.mlp_ratio)), nn.GELU(), nn.Linear(int(d * cfg.mlp_ratio), d))
        cos, sin = rope_tables(cfg.max_len, self.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        ablate_heads: set[int] | None = None,
        attn_override: dict[int, torch.Tensor] | None = None,
        want_contrib: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """x: (B, T, D). ``ablate_heads`` zeroes those heads' outputs; ``attn_override`` maps a head index to
        an attention pattern ((T, T) or (B, T, T)) that replaces its own. Returns (new residual, details)."""
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.head_dim
        qkv = self.qkv(self.ln(x)).view(B, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))                   # (B, H, T, hd)
        q = apply_rope(q, self.cos[:T], self.sin[:T])
        k = apply_rope(k, self.cos[:T], self.sin[:T])
        scores = (q @ k.transpose(-2, -1)) * hd ** -0.5
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        attn = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)      # (B, H, T, T)
        if attn_override:
            attn = attn.clone()
            for h, pattern in attn_override.items():
                attn[:, h] = pattern
        head_out = attn @ v                                                           # (B, H, T, hd)
        if ablate_heads:
            head_out = head_out.clone()
            for h in ablate_heads:
                head_out[:, h] = 0.0
        out = self.proj(head_out.transpose(1, 2).reshape(B, T, D))
        details = {"attn": attn, "head_out": head_out}
        if want_contrib:      # each head's contribution to the residual stream: (B, H, T, D)
            w = self.proj.weight.view(D, H, hd)
            details["head_contrib"] = torch.einsum("bhtd,ohd->bhto", head_out, w)
        x = x + out
        if self.cfg.mlp_ratio is not None:
            x = x + self.mlp(self.ln2(x))
        return x, details


class IHTransformer(nn.Module):
    """Interleaves exemplar features with label embeddings and reads out label (and symbol) predictions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.example_embed = nn.Linear(cfg.d_in, cfg.d_model, bias=True)
        self.label_embed = nn.Embedding(cfg.n_labels, cfg.d_model)
        self.blocks = nn.ModuleList([AttentionOnlyBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.label_head = nn.Linear(cfg.d_model, cfg.n_labels, bias=True)
        self.symbol_head = nn.Linear(cfg.d_model, cfg.n_classes, bias=True) if cfg.extended else None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, 1.0 / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.label_embed.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def interleave(ex: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """e0 l0 e1 l1 ... : ex (B, n, D) and labels (B, n-1 or n, D) -> (B, n + n_labels, D)."""
        n_e, n_l = ex.shape[1], labels.shape[1]
        assert n_l in (n_e - 1, n_e), (n_e, n_l)
        B, D = ex.shape[0], ex.shape[2]
        out = ex.new_zeros(B, n_e + n_l, D)
        out[:, 0::2] = ex
        out[:, 1::2] = labels
        return out

    def forward(
        self,
        ex: torch.Tensor,
        labels: torch.Tensor,
        ablate: dict[int, set[int]] | None = None,
        attn_override: dict[tuple[int, int], torch.Tensor] | None = None,
        return_details: bool = False,
        want_contrib: bool = False,
    ) -> dict:
        """ex: (B, n_ex, 512) features; labels: (B, n_ex-1 or n_ex) integer ids.

        Returns a dict with ``label_logits_q`` (B, 5) at position 4 and, when the input has 7 tokens and the model
        is extended, ``symbol_logits`` (B, 50) at position 5 and ``label_logits_s2`` (B, 5) at position 6. With
        ``return_details`` also ``attn`` (list per layer of (B, H, T, T)), ``head_out`` and ``resid`` (the
        residual stream before block 0 and after each block)."""
        x = self.interleave(self.example_embed(ex), self.label_embed(labels))
        resid = [x]
        attn_l, head_l, contrib_l = [], [], []
        for li, block in enumerate(self.blocks):
            override = {h: p for (l, h), p in (attn_override or {}).items() if l == li} or None
            x, det = block(x, ablate_heads=(ablate or {}).get(li), attn_override=override, want_contrib=want_contrib)
            resid.append(x)
            attn_l.append(det["attn"]); head_l.append(det["head_out"])
            if want_contrib:
                contrib_l.append(det["head_contrib"])
        h = self.ln_f(x)
        T = h.shape[1]
        out = {"label_logits_q": self.label_head(h[:, 4])}
        if self.symbol_head is not None and T >= 6:
            out["symbol_logits"] = self.symbol_head(h[:, 5])
        if self.symbol_head is not None and T >= 7:
            out["label_logits_s2"] = self.label_head(h[:, 6])
        if return_details:
            out.update(attn=attn_l, head_out=head_l, resid=resid, hidden=h)
            if want_contrib:
                out["head_contrib"] = contrib_l
        return out

    # ---------------------------------------------------------------- generation (for the collapse pipeline)
    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, greedy: bool, gen: torch.Generator | None) -> torch.Tensor:
        if greedy or temperature <= 0:
            return logits.argmax(-1)
        return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=gen).squeeze(-1)

    @torch.no_grad()
    def sample_query_label(self, ex_ctx: torch.Tensor, labels_ctx: torch.Tensor, temperature: float = 1.0,
                           greedy: bool = False, gen: torch.Generator | None = None) -> torch.Tensor:
        """Base model: the label it assigns to Q given a real context. ex_ctx (B, 3, 512) = [A, B, Q]; labels_ctx (B, 2)."""
        self.eval()
        return self._sample(self(ex_ctx, labels_ctx)["label_logits_q"], temperature, greedy, gen)

    @torch.no_grad()
    def sample_extended(self, ex_ctx: torch.Tensor, labels_ctx: torch.Tensor, ex0_table: torch.Tensor,
                        temperature: float = 1.0, greedy: bool = False, gen: torch.Generator | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extended model: given a real context ``A la B lb Q`` sample lq, then S2's class, feed back exemplar 0 of
        that class, then sample l_S2. ex_ctx (B, 3, 512), labels_ctx (B, 2), ex0_table (50, 512) = exemplar 0 of each
        of the 50 training classes. Returns (lq (B,), s2_class (B,), l_s2 (B,))."""
        assert self.symbol_head is not None, "sample_extended needs an extended model"
        self.eval()
        lq = self._sample(self(ex_ctx, labels_ctx)["label_logits_q"], temperature, greedy, gen)
        labels3 = torch.cat([labels_ctx, lq[:, None]], dim=1)
        s2 = self._sample(self(ex_ctx, labels3)["symbol_logits"], temperature, greedy, gen)
        ex4 = torch.cat([ex_ctx, ex0_table[s2][:, None]], dim=1)
        ls2 = self._sample(self(ex4, labels3)["label_logits_s2"], temperature, greedy, gen)
        return lq, s2, ls2
