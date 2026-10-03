"""Synthetic few-shot in-context-learning symbol task (Singh et al., 2024 style).

A sequence is built from `n_classes` distinct "exemplar" symbols, each shown
`burstiness` times paired with a label that is assigned *fresh, uniformly at
random, per sequence* (the `fewshot_relabel` regime). The context pairs are
then shuffled and followed by a query exemplar drawn from the same
`n_classes` symbols; the task is to output the label that sequence assigned
to that exemplar. Because labels are re-randomised every sequence, the task
cannot be solved by memorising a global exemplar->label table -- it can only
be solved by finding the matching earlier occurrence in-context and copying
its label (induction).

Simplification vs. Singh et al. (2024): exemplars are plain integer symbol
IDs (drawn from a vocabulary of size K) rather than Omniglot character
images. This is a deliberate scope decision -- the image encoder is an
implementation detail for realism, not something the induction-head
mechanism depends on, and a symbolic vocabulary is much easier to control
and interpret. This should be noted explicitly in the write-up.

Sequence layout (token stream fed to the model):
    s_1, l_1, s_2, l_2, ..., s_m, l_m, s_query
where m = n_classes * burstiness. The base task supervises only the label
that follows s_query (predicted at the final position). The extended task
additionally supervises, at every position, the label that follows a symbol
token and the symbol that follows a label token -- see model.py.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclasses.dataclass
class TaskConfig:
    K: int = 64          # symbol (exemplar) vocabulary size
    L: int = 8           # label vocabulary size
    n_classes: int = 4   # number of distinct exemplars shown per sequence ("N-way")
    burstiness: int = 3  # number of times each exemplar/label pair appears in context
    shuffle_context: bool = True  # shuffle order of context pairs (bursty, non-blocked)

    def __post_init__(self):
        if self.n_classes > self.L:
            raise ValueError(
                f"n_classes ({self.n_classes}) must be <= L ({self.L}) "
                "so each shown exemplar can get a distinct label."
            )
        if self.n_classes > self.K:
            raise ValueError(f"n_classes ({self.n_classes}) must be <= K ({self.K}).")

    @property
    def context_len(self) -> int:
        return self.n_classes * self.burstiness

    @property
    def seq_len(self) -> int:
        # context pairs (symbol, label) * 2, plus the final query symbol
        return 2 * self.context_len + 1


@dataclasses.dataclass
class Sequence:
    symbols: torch.Tensor   # (context_len + 1,) symbol ids, last entry is the query symbol
    labels: torch.Tensor    # (context_len,) label ids for the context pairs, in sequence order
    query_label: int        # ground-truth label for the query symbol


def generate_sequence(cfg: TaskConfig, rng: torch.Generator) -> Sequence:
    """Sample a single sequence under `cfg` using `rng` for reproducibility."""
    k_idx = torch.randperm(cfg.K, generator=rng)[: cfg.n_classes]
    l_idx = torch.randperm(cfg.L, generator=rng)[: cfg.n_classes]
    # exemplar_to_label[i] is the label assigned (this sequence only) to k_idx[i]
    exemplar_to_label = {int(k_idx[i]): int(l_idx[i]) for i in range(cfg.n_classes)}

    context_symbols = k_idx.repeat_interleave(cfg.burstiness)
    if cfg.shuffle_context:
        perm = torch.randperm(context_symbols.shape[0], generator=rng)
        context_symbols = context_symbols[perm]
    context_labels = torch.tensor(
        [exemplar_to_label[int(s)] for s in context_symbols], dtype=torch.long
    )

    query_pos = int(torch.randint(cfg.n_classes, (1,), generator=rng))
    query_symbol = k_idx[query_pos]
    query_label = exemplar_to_label[int(query_symbol)]

    symbols = torch.cat([context_symbols, query_symbol.unsqueeze(0)])
    return Sequence(symbols=symbols, labels=context_labels, query_label=query_label)


def sequence_to_tokens(seq: Sequence, cfg: TaskConfig):
    """Interleave a Sequence into the (symbols, labels)-alternating token stream
    used by the model. Returns:
      symbol_tokens: (context_len + 1,) symbol id at each symbol slot
      label_tokens:  (context_len,) label id at each label slot (context only)
      next_symbol_target: (context_len,) symbol that follows each label slot
      next_label_target:  (context_len + 1,) label that follows each symbol slot
                           (last entry is the query's ground-truth label)
    """
    symbol_tokens = seq.symbols  # (context_len + 1,)
    label_tokens = seq.labels    # (context_len,)
    next_symbol_target = symbol_tokens[1:].clone()  # symbol after each label slot
    next_label_target = torch.cat([seq.labels, torch.tensor([seq.query_label])])
    return symbol_tokens, label_tokens, next_symbol_target, next_label_target


class SymbolicICLDataset(Dataset):
    """A materialised (fixed, reproducible) pool of sequences for one split."""

    def __init__(self, cfg: TaskConfig, n_sequences: int, seed: int):
        self.cfg = cfg
        self.n_sequences = n_sequences
        self.seed = seed
        rng = torch.Generator().manual_seed(seed)
        self.sequences = [generate_sequence(cfg, rng) for _ in range(n_sequences)]

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        symbol_tokens, label_tokens, next_symbol_target, next_label_target = sequence_to_tokens(
            seq, self.cfg
        )
        return {
            "symbol_tokens": symbol_tokens,
            "label_tokens": label_tokens,
            "next_symbol_target": next_symbol_target,
            "next_label_target": next_label_target,
            "query_label": torch.tensor(seq.query_label, dtype=torch.long),
        }


def make_splits(
    cfg: TaskConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    base_seed: int = 0,
) -> tuple[SymbolicICLDataset, SymbolicICLDataset, SymbolicICLDataset]:
    """Sequence-level train/val/test split via disjoint RNG seed streams.

    Sequences are generated on the fly from (n_classes, K, L) combinatorics
    that are astronomically larger than n_train+n_val+n_test for the configs
    used in this project, so disjoint seeds give split independence without
    needing explicit deduplication.
    """
    train = SymbolicICLDataset(cfg, n_train, seed=base_seed)
    val = SymbolicICLDataset(cfg, n_val, seed=base_seed + 1_000_003)
    test = SymbolicICLDataset(cfg, n_test, seed=base_seed + 2_000_003)
    return train, val, test


def collate_batch(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


class ListDataset(Dataset):
    """Wrap an in-memory list of already-tokenised sequence dicts (used to hold
    synthetic data sampled from a model during the collapse pipeline, and for
    real+synthetic mixes)."""

    def __init__(self, items: list[dict], cfg: TaskConfig):
        self.items = items
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def dataset_to_items(ds: SymbolicICLDataset) -> list[dict]:
    return [ds[i] for i in range(len(ds))]


def save_items(items: list[dict], path) -> None:
    """Save a pool of tokenised sequence dicts compactly: one stacked int32
    tensor per field (~4 MB for 20k sequences) instead of torch.save-ing a
    list of ~100k tiny tensors (~32 MB). Symbol/label ids are small
    non-negative ints, so int32 is lossless. Restore with load_items."""
    packed = {k: torch.stack([it[k] for it in items]).to(torch.int32) for k in items[0]}
    torch.save(packed, path)


def load_items(path) -> list[dict]:
    """Inverse of save_items: returns the original list-of-dicts format
    (int64 tensors) expected by ListDataset / collate_batch / attribution."""
    packed = torch.load(path, weights_only=True)
    if isinstance(packed, list):  # legacy snapshot: already a list of dicts (written before save_items existed)
        return packed
    n = next(iter(packed.values())).shape[0]
    return [{k: v[i].long() for k, v in packed.items()} for i in range(n)]


def mix_datasets(
    real_items: list[dict],
    synthetic_items: list[dict],
    p_synthetic: float,
    n_total: int,
    seed: int,
    with_replacement: bool = True,
) -> list[dict]:
    """Build a generation's training pool by sampling a `p_synthetic` fraction
    from `synthetic_items` and the rest from `real_items`.

    with_replacement=True (default; what every result so far used) draws
    each part as a bootstrap resample, so when a part is as large as its
    source only ~63% of its sequences are distinct. with_replacement=False
    draws distinct sequences (each source must hold at least as many items as
    requested), which removes that duplication as a confound when comparing
    generation >= 1 pools against generation 0's fully distinct pool."""
    g = torch.Generator().manual_seed(seed)
    n_synth = int(round(n_total * p_synthetic))
    n_real = n_total - n_synth

    def sample(items, n):
        if n == 0:
            return []
        if with_replacement:
            idx = torch.randint(len(items), (n,), generator=g)
        else:
            if n > len(items):
                raise ValueError(
                    f"with_replacement=False needs >= {n} source items, got {len(items)}"
                )
            idx = torch.randperm(len(items), generator=g)[:n]
        return [items[int(i)] for i in idx]

    pool = sample(real_items, n_real) + sample(synthetic_items, n_synth)
    perm = torch.randperm(len(pool), generator=g).tolist()
    return [pool[i] for i in perm]
