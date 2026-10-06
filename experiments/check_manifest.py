"""Check, before submitting anything, that every Slurm array script's --array range matches the manifest group it runs and
that every job's config exists. A wrong range silently runs the wrong jobs or none.

    python3 experiments/check_manifest.py          # needs only the standard library (no virtualenv)

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import CONFIG_DIR, GROUPS  # noqa: E402

SLURM_DIR = HERE.parent / "slurm"


def parse_array(text: str):
    m = re.search(r"^#SBATCH\s+--array=(\d+)-(\d+)", text, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main() -> int:
    print("manifest groups:", {g: len(j) for g, j in GROUPS.items()})
    bad = 0
    missing = sorted({j.config for jobs in GROUPS.values() for j in jobs if not (CONFIG_DIR / j.config).exists()})
    if missing:
        print("  BAD missing configs (run experiments/make_configs.py):", missing)
        bad += 1
    for path in sorted(SLURM_DIR.glob("*.slurm")):
        text = path.read_text()
        m = re.search(r"run_array_task\.py[^\n]*--group\s+(\w+)", text)
        if not m:
            continue
        group, rng = m.group(1), parse_array(text)
        n = len(GROUPS[group]) if group in GROUPS else None
        ok = rng is not None and n is not None and rng == (0, n - 1)
        print(f"  {'OK ' if ok else 'BAD'} {path.name:24s} group={group:10s} array={rng} group size={n}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
