"""Canonical list of experiments and seeds, shared by run_all.py (serial,
local) and run_array_task.py (one Slurm array task per combination)."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

EXPERIMENTS = [
    ("base_no_collapse.yaml", "train"),
    ("extended_no_collapse.yaml", "train"),
    ("base_collapse.yaml", "collapse"),
    ("extended_collapse.yaml", "collapse"),
    ("ablation_p025.yaml", "collapse"),
    ("ablation_p05.yaml", "collapse"),
]

# 3 seeds per condition for statistical robustness -- see README / abstract
# Limitations for why this matters (base-variant training turned out to be
# seed-sensitive; single-seed curves would be a weak basis for claims about
# collapse vs. seed luck).
SEEDS = [0, 1, 2]


def all_combinations() -> list[tuple[str, str, int]]:
    """Every (config filename, mode, seed) combination to run -- 18 with the
    defaults above, i.e. 18 independent Slurm array tasks."""
    return [(config_name, mode, seed) for config_name, mode in EXPERIMENTS for seed in SEEDS]
