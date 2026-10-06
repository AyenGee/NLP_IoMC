"""The exact few-shot in-context-learning task of Singh et al. (2024), "What needs to go right
for an induction head?" (ICML), as set up in their repository aadityasingh/icl-dynamics.

Everything here is specified by the paper (Section 2.1, Appendix A and C) and the reference
repository's flags in ``ih_paper_runs.sh`` (read for specifications only; no code is copied).

Inputs
------
Exemplars are the precomputed Omniglot ResNet18 features shipped with the reference repository,
``omniglot_resnet18_randomized_order_s0.h5``, dataset ``resnet18/224/feat``, shape
(1623 classes, 5 exemplars, 512 features), float32. They are used exactly as given: no image
and no ResNet is touched here. Download instructions are in the README.

Splits (repo flags ``--class_split 50 1473 100 --exemplar_split 1 4 0``)
    train classes     indices 0..49          (the class order in the file is already randomised)
    unused classes    indices 50..1522
    test classes      indices 1523..1622     (the last 100)
    train exemplar    index 0                (E = 1)
    val exemplars     indices 1..4           (held-out exemplars of the TRAIN classes)

Labels (repo flags ``--fs_relabel 5 --fs_relabel_split 8 2 0``)
    L = 5 labels; the 10 unordered label pairs {i<j} are shuffled with a fixed seed
    (LABEL_PAIR_SEED); 8 go to training and 2 are held out ("relabel" test). The repository
    draws its split from a JAX RNG with seed 20, which cannot be reproduced bit for bit in
    PyTorch, so the SPECIFIC held-out pairs differ from the paper's (the split sizes do not).

Sequence (2-way, 1-shot):  ``A, la, B, lb, Q``
    A and B are exemplars of two different classes; {la, lb} is a label pair assigned to
    (A, B) in one of two ways; Q is a new instance of A's or B's class; the target is the
    label of Q's class. Labels are redrawn per sequence, so only in-context learning
    (find the matching exemplar, copy its label) solves the task.

    Full training space (Appendix C), enumerated exactly:
        2450 ordered class pairs x 8 label pairs x 2 label assignments x 2 query choices
        = 78,400 sequences.
    (Equivalent to 1225 unordered pairs x 8 x 8 with 2 context orders; ordered pairs carry the
    context order.) Each sequence is stored as integers and embeddings are looked up at collate
    time.

Extended sequence (the brief's "also predict the next symbol and its corresponding value")
    ``A, la, B, lb, Q, lq, S2`` with three teacher-forced predictions:
        position 4 (Q)  -> lq   the label of Q's class   (as in the base task)
        position 5 (lq) -> S2   the class of the next exemplar
        position 6 (S2) -> l_S2 the in-context label of S2's class
    Design decision (not fixed by the paper or the brief): in REAL data S2 is another query of
    the same episode, i.e. uniform over {A's class, B's class}, with exemplar 0 in training.
    This is the natural continuation of a few-shot episode, and it keeps ``l_S2`` solvable by
    induction (S2's class is always one whose label is in the context). Bayes-optimal
    accuracies under real data: lq 1.0; S2 over the 50 classes 0.5 (it can name only one of
    the two); l_S2 given S2 1.0. The 0.5 for S2 is therefore NOT a failure.
    Enumerated extended training space: 78,400 x 2 = 156,800.

Evaluation sets (1,000 sequences each, fixed seed, saved to disk so every model sees the same):
    train_eval      train classes, exemplar 0, train label pairs             fit
    val             train classes, exemplars drawn from 1..4, train pairs    model selection
                    (the paper's ``fsl_train_valex``)
    test_exemplars  the 100 test classes, exemplars drawn from 0..4,         the MATCH step on
                    train pairs                                              unseen symbols
    test_relabel    train classes, exemplar 0, the 2 held-out pairs          the COPY step on
                    (the paper's ``fsl_val_rl``)                             unseen label pairs
    test_class_ex0  the 100 test classes, exemplar 0, train pairs            the repository's
                    (the paper's ``fsl_test_class``; logged for comparability)
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import torch

H5_NAME = "omniglot_resnet18_randomized_order_s0.h5"
H5_DATASET = "resnet18/224/feat"
N_CLASSES_TOTAL, N_EXEMPLARS_TOTAL, FEATURE_DIM = 1623, 5, 512

TRAIN_CLASSES = np.arange(0, 50)
TEST_CLASSES = np.arange(1523, 1623)
TRAIN_EXEMPLARS = np.array([0])
VAL_EXEMPLARS = np.array([1, 2, 3, 4])
N_LABELS = 5
LABEL_PAIR_SEED = 20          # the repository's seed, but PyTorch/NumPy cannot reproduce its JAX draw
N_TRAIN_PAIRS = 8
EVAL_SEED = 12345
N_EVAL = 1000

# Integer row layout of a sequence ("episode"). Base uses the first 8 columns; extended all 10.
COLS = ("cA", "cB", "la", "lb", "q_is_A", "exA", "exB", "exQ", "s2_is_A", "exS2")
C_A, C_B, L_A, L_B, Q_IS_A, EX_A, EX_B, EX_Q, S2_IS_A, EX_S2 = range(10)

EVAL_SET_NAMES = ("train_eval", "val", "test_exemplars", "test_relabel", "test_class_ex0")


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


class OmniglotFeatures:
    """The (1623, 5, 512) float32 feature tensor, loaded once from the .h5 file."""

    def __init__(self, path: str | os.PathLike | None = None):
        import h5py

        path = Path(path) if path else default_data_dir() / H5_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download it from https://github.com/aadityasingh/icl-dynamics "
                f"(file {H5_NAME}, about 16.6 MB) into data/; see README.txt."
            )
        with h5py.File(path, "r") as f:
            feats = f[H5_DATASET][:]
        if feats.shape != (N_CLASSES_TOTAL, N_EXEMPLARS_TOTAL, FEATURE_DIM) or feats.dtype != np.float32:
            raise ValueError(f"unexpected feature array {feats.shape} {feats.dtype}")
        self.tensor = torch.from_numpy(feats)


def label_pair_split(seed: int = LABEL_PAIR_SEED) -> tuple[np.ndarray, np.ndarray]:
    """(train_pairs (8, 2), test_pairs (2, 2)): the 10 unordered pairs over 0..4, shuffled with a fixed seed."""
    pairs = np.array([(i, j) for i in range(N_LABELS) for j in range(i + 1, N_LABELS)])
    order = np.random.default_rng(seed).permutation(len(pairs))
    pairs = pairs[order]
    return pairs[:N_TRAIN_PAIRS], pairs[N_TRAIN_PAIRS:]


def enumerate_train_space(extended: bool = False) -> np.ndarray:
    """Every training sequence once, as integer rows (N, 10), exemplar indices all 0.

    Base: 78,400 rows (``s2_is_A`` and ``exS2`` are 0 and unused). Extended: 156,800 rows, each
    base episode with both choices of S2."""
    train_pairs, _ = label_pair_split()
    rows = []
    for cA in TRAIN_CLASSES:
        for cB in TRAIN_CLASSES:
            if cA == cB:
                continue
            for pair in train_pairs:
                for la, lb in ((pair[0], pair[1]), (pair[1], pair[0])):
                    for q_is_A in (1, 0):
                        rows.append((cA, cB, la, lb, q_is_A, 0, 0, 0, 0, 0))
    rows = np.array(rows, dtype=np.int64)
    if extended:
        rows = np.concatenate([rows, rows], axis=0)
        rows[len(rows) // 2:, S2_IS_A] = 1
    return rows


def _sample_rows(rng: np.random.Generator, n: int, classes: np.ndarray, exemplars: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    cA = rng.choice(classes, size=n)
    cB = rng.choice(classes, size=n)
    clash = cA == cB
    while clash.any():                                # two DIFFERENT classes
        cB[clash] = rng.choice(classes, size=int(clash.sum()))
        clash = cA == cB
    pair = pairs[rng.integers(0, len(pairs), size=n)]
    swap = rng.integers(0, 2, size=n).astype(bool)
    la = np.where(swap, pair[:, 1], pair[:, 0])
    lb = np.where(swap, pair[:, 0], pair[:, 1])
    rows = np.zeros((n, 10), dtype=np.int64)
    rows[:, C_A], rows[:, C_B], rows[:, L_A], rows[:, L_B] = cA, cB, la, lb
    rows[:, Q_IS_A] = rng.integers(0, 2, size=n)
    rows[:, S2_IS_A] = rng.integers(0, 2, size=n)
    for col in (EX_A, EX_B, EX_Q, EX_S2):
        rows[:, col] = rng.choice(exemplars, size=n)
    return rows


def build_eval_sets(seed: int = EVAL_SEED, n: int = N_EVAL) -> dict[str, np.ndarray]:
    """The five evaluation sets as integer rows (n, 10) each, in a fixed order from one seed."""
    train_pairs, test_pairs = label_pair_split()
    rng = np.random.default_rng(seed)
    all_ex = np.arange(N_EXEMPLARS_TOTAL)
    return {
        "train_eval": _sample_rows(rng, n, TRAIN_CLASSES, TRAIN_EXEMPLARS, train_pairs),
        "val": _sample_rows(rng, n, TRAIN_CLASSES, VAL_EXEMPLARS, train_pairs),
        "test_exemplars": _sample_rows(rng, n, TEST_CLASSES, all_ex, train_pairs),
        "test_relabel": _sample_rows(rng, n, TRAIN_CLASSES, TRAIN_EXEMPLARS, test_pairs),
        "test_class_ex0": _sample_rows(rng, n, TEST_CLASSES, TRAIN_EXEMPLARS, train_pairs),
    }


def save_eval_sets(path: str | os.PathLike, sets: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **sets)


def load_or_build_eval_sets(path: str | os.PathLike | None = None) -> dict[str, np.ndarray]:
    """Read the saved evaluation sets, creating the file the first time (deterministic from EVAL_SEED)."""
    path = Path(path) if path else default_data_dir() / "eval_sets_exact.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in EVAL_SET_NAMES}
    sets = build_eval_sets()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_eval_sets(path, sets)
    return sets


def query_class(rows: torch.Tensor) -> torch.Tensor:
    return torch.where(rows[:, Q_IS_A] == 1, rows[:, C_A], rows[:, C_B])


def query_label(rows: torch.Tensor) -> torch.Tensor:
    return torch.where(rows[:, Q_IS_A] == 1, rows[:, L_A], rows[:, L_B])


def s2_class(rows: torch.Tensor) -> torch.Tensor:
    return torch.where(rows[:, S2_IS_A] == 1, rows[:, C_A], rows[:, C_B])


def s2_label(rows: torch.Tensor) -> torch.Tensor:
    return torch.where(rows[:, S2_IS_A] == 1, rows[:, L_A], rows[:, L_B])


@dataclasses.dataclass
class Batch:
    """A collated batch. ``ex`` holds the exemplar features [A, B, Q] (+ [S2] if extended), ``labels`` the
    context labels [la, lb] (+ [lq] if extended), in the interleaved order of the token stream. Targets
    follow the model's output positions (see data.py docstring)."""
    ex: torch.Tensor              # (B, 3 or 4, 512)
    labels: torch.Tensor          # (B, 2 or 3)
    query_label: torch.Tensor     # (B,)  target at position 4
    q_is_A: torch.Tensor          # (B,)  which context exemplar Q matches (for the interpretability metrics)
    s2_class: torch.Tensor | None = None   # (B,)  target at position 5 (class index; -1 if outside the 50 train classes)
    s2_label: torch.Tensor | None = None   # (B,)  target at position 6


def collate(rows: torch.Tensor, features: torch.Tensor, extended: bool) -> Batch:
    """Look up the embeddings for integer rows (N, 10)."""
    cq = query_class(rows)
    exemplars = [features[rows[:, C_A], rows[:, EX_A]], features[rows[:, C_B], rows[:, EX_B]], features[cq, rows[:, EX_Q]]]
    labels = [rows[:, L_A], rows[:, L_B]]
    s2c = s2l = None
    if extended:
        c2 = s2_class(rows)
        exemplars.append(features[c2, rows[:, EX_S2]])
        labels.append(query_label(rows))
        s2c = torch.where(c2 < len(TRAIN_CLASSES), c2, torch.full_like(c2, -1))
        s2l = s2_label(rows)
    return Batch(ex=torch.stack(exemplars, dim=1), labels=torch.stack(labels, dim=1), query_label=query_label(rows),
                 q_is_A=rows[:, Q_IS_A], s2_class=s2c, s2_label=s2l)


class TrainSampler:
    """Draws training batches uniformly at random (with replacement across steps) from a pool of integer rows;
    the reference repository samples online from the finite space in the same way."""

    def __init__(self, pool: np.ndarray | torch.Tensor, features: torch.Tensor, extended: bool, seed: int):
        self.pool = torch.as_tensor(pool, dtype=torch.long)
        self.features, self.extended = features, extended
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.pool)

    def next_batch(self, batch_size: int) -> Batch:
        idx = torch.randint(len(self.pool), (batch_size,), generator=self.gen)
        return collate(self.pool[idx], self.features, self.extended)
