"""Canonical list of experiment jobs, shared by run_all.py (serial, local) and
run_array_task.py (one Slurm array task per job).

Jobs are organised in named GROUPS so each Slurm script can use indices
relative to its own group (`--array=0-8` with `--group controls`) instead of
fragile global indices:

  main          the original 18 runs (6 conditions x seeds 0-2). Order and
                indices are unchanged, so slurm/run_fast|medium|slow.slurm and
                their existing logs remain valid.
  controls      3 control conditions x seeds 0-2: tests whether the
                duplicate-sequence resampling in generations >= 1 (rather than
                synthetic data) explains the faster learning / the collapse.
  extra_medium  3 extended conditions x 5 new seeds, each with its OWN
                real-data pool (data_seed = seed), unlike seeds 0-2 which share
                one pool: more seeds AND independent data draws.
  extra_slow    the base collapse condition x the same 5 new seeds.
  v2            the CORRECTED-PROTOCOL experiment: 6 conditions x 8 seeds (0-7),
                every generation trained for a fixed budget with early stopping
                OFF, on pools without duplicate sequences (plus one condition that
                keeps duplicates, to separate the two artifacts; one is a real-data control
                under the same protocol). Each seed has its own real-data pool
                (data_seed = seed). See configs/v2_*.yaml; regenerate with
                experiments/make_v2_configs.py.

Run `python experiments/check_manifest.py` to confirm every Slurm script's
--array range matches its group's size.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclasses.dataclass(frozen=True)
class Job:
    config: str                 # filename under configs/
    mode: str                   # "train" or "collapse"
    seed: int                   # model init + sampling randomness
    data_seed: int | None = None  # None keeps the config's data_seed (0): seeds 0-2 share one real-data pool


EXPERIMENTS = [
    ("base_no_collapse.yaml", "train"),
    ("extended_no_collapse.yaml", "train"),
    ("base_collapse.yaml", "collapse"),
    ("extended_collapse.yaml", "collapse"),
    ("ablation_p025.yaml", "collapse"),
    ("ablation_p05.yaml", "collapse"),
]

SEEDS = [0, 1, 2]
EXTRA_SEEDS = [3, 4, 5, 6, 7]

CONTROL_CONFIGS = ["control_real_resampled.yaml", "control_real_fresh.yaml", "extended_collapse_nodup.yaml"]
EXTRA_MEDIUM_CONFIGS = ["extended_collapse.yaml", "ablation_p05.yaml", "ablation_p025.yaml"]
V2_CONFIGS = ["v2_extended_p1.yaml", "v2_extended_p05.yaml", "v2_extended_p025.yaml", "v2_base_p1.yaml", "v2_extended_p1_dups.yaml",
              "v2_control_real.yaml"]
V2_SEEDS = list(range(8))

GROUPS: dict[str, list[Job]] = {
    "main": [Job(cfg, mode, s) for cfg, mode in EXPERIMENTS for s in SEEDS],
    "controls": [Job(cfg, "collapse", s) for cfg in CONTROL_CONFIGS for s in SEEDS],
    "extra_medium": [Job(cfg, "collapse", s, data_seed=s) for cfg in EXTRA_MEDIUM_CONFIGS for s in EXTRA_SEEDS],
    "extra_slow": [Job("base_collapse.yaml", "collapse", s, data_seed=s) for s in EXTRA_SEEDS],
    "v2": [Job(cfg, "collapse", s, data_seed=s) for cfg in V2_CONFIGS for s in V2_SEEDS],
}


# Condition groups for slurm/run_acdc.slurm (one array task each); used by experiments/analyze_acdc.py.
ACDC_GROUPS = [
    ["v2_extended_p1"], ["v2_extended_p1_dups"], ["v2_extended_p05"], ["v2_extended_p025"], ["v2_base_p1"],
    ["v2_control_real"], ["control_real_resampled", "control_real_fresh"], ["extended_collapse_nodup"],
]


# Condition groups for slurm/run_circuit_analysis.slurm (one array task each); used by experiments/analyze_circuits.py.
CIRCUIT_GROUPS = [
    ["extended_collapse"], ["ablation_p05"], ["ablation_p025"], ["base_collapse"],
    ["control_real_resampled", "control_real_fresh", "extended_collapse_nodup"],
    ["v2_extended_p1"], ["v2_extended_p1_dups"], ["v2_extended_p05"], ["v2_extended_p025"], ["v2_base_p1"], ["v2_control_real"],
]


def group_jobs(group: str) -> list[Job]:
    if group not in GROUPS:
        raise SystemExit(f"unknown group {group!r}; choose from {sorted(GROUPS)}")
    return GROUPS[group]
