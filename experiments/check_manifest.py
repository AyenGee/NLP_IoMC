"""Check that every Slurm array script's index range matches the manifest
group it runs, before you submit anything.

    python experiments/check_manifest.py

A wrong --array silently runs the wrong experiments or none at all (an index
past the end of a group makes that task exit with an error), so this is worth
the second it takes. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import GROUPS, ACDC_GROUPS, CIRCUIT_GROUPS

SLURM_DIR = Path(__file__).resolve().parent.parent / "slurm"


def parse_array(text: str) -> tuple[int, int] | None:
    m = re.search(r"^#SBATCH\s+--array=(\d+)-(\d+)", text, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main() -> int:
    print("manifest groups:", {g: len(j) for g, j in GROUPS.items()})
    bad = 0
    for path in sorted(SLURM_DIR.glob("*.slurm")):
        text = path.read_text()
        if "tune_hparams.py" in text:
            from tune_hparams import CELLS
            rng = parse_array(text)
            ok = rng == (0, len(CELLS) - 1)
            print(f"  {'OK ' if ok else 'BAD'} {path.name:24s} tuning cells={len(CELLS)}   array={rng}")
            bad += not ok
            continue
        if "analyze_circuits.py" in text:
            rng = parse_array(text)
            ok = rng == (0, len(CIRCUIT_GROUPS) - 1)
            print(f"  {'OK ' if ok else 'BAD'} {path.name:24s} circuit groups={len(CIRCUIT_GROUPS)} array={rng}")
            bad += not ok
            continue
        if "analyze_acdc.py" in text:
            rng = parse_array(text)
            ok = rng == (0, len(ACDC_GROUPS) - 1)
            print(f"  {'OK ' if ok else 'BAD'} {path.name:24s} acdc groups={len(ACDC_GROUPS)}   array={rng}")
            bad += not ok
            continue
        if "run_array_task.py" not in text:
            continue
        rng = parse_array(text)
        gm = re.search(r"run_array_task\.py[^\n]*--group\s+(\w+)", text)
        group = gm.group(1) if gm else "main"
        n = len(GROUPS[group]) if group in GROUPS else None
        ok = rng is not None and n is not None and 0 <= rng[0] <= rng[1] < n
        if ok and group != "main":
            ok = rng == (0, n - 1)          # new group scripts must cover the whole group
        print(f"  {'OK ' if ok else 'BAD'} {path.name:24s} group={group:12s} array={rng} group size={n}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
