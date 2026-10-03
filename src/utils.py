"""Small shared helpers: seeding, device selection, config loading, CSV logging."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred is not None and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class CSVLogger:
    """Append-only CSV logger; writes the header from the first row's keys."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        os.makedirs(self.path.parent, exist_ok=True)
        self._fieldnames: list[str] | None = None
        if self.path.exists():
            self.path.unlink()

    def log(self, row: dict) -> None:
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        write_header = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)
