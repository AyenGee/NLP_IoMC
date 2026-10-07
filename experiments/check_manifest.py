"""Check, before submitting anything, that every Slurm array script's --array range and --time match the manifest group it runs,
that every group has its script, that every job's config exists and that no two jobs of a group write to the same output.
A wrong range silently runs the wrong jobs or none; a short time limit kills the last generations of a chain.

    python3 experiments/check_manifest.py          # needs only the standard library plus PyYAML-free imports (no virtualenv)

Exits non-zero on any mismatch. (Tier 1's script keeps its original 8 h limit: Tier 1 is done and is not time-checked.)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import CONFIG_DIR, GROUP_INFO, GROUPS, time_limit  # noqa: E402

SLURM_DIR = HERE.parent / "slurm"


def parse_array(text: str):
    m = re.search(r"^#SBATCH\s+--array=(\d+)-(\d+)", text, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_time(text: str):
    m = re.search(r"^#SBATCH\s+--time=(\S+)", text, re.M)
    return m.group(1) if m else None


def main() -> int:
    print("manifest groups:", {g: len(j) for g, j in GROUPS.items()})
    bad = 0
    missing = sorted({j.config for jobs in GROUPS.values() for j in jobs if j.kind == "collapse" and not (CONFIG_DIR / j.config).exists()})
    if missing:
        print("  BAD missing configs (run experiments/make_configs.py and experiments/make_configs.py --tier2):", missing)
        bad += 1
    for g, jobs in GROUPS.items():
        names = [j.run_name for j in jobs]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            print(f"  BAD group {g}: duplicate run names {dup[:5]}")
            bad += 1
    seen = {}
    for g, jobs in GROUPS.items():                       # a continuation intentionally reuses the Tier 1 run names (it resumes them)
        for j in jobs:
            if g in seen.get(j.run_name, ()) :
                continue
            seen.setdefault(j.run_name, []).append(g)
    clash = {n: gs for n, gs in seen.items() if len(gs) > 1 and set(gs) != {"tier1", "continue20"}}
    if clash:
        print("  BAD run names shared between groups:", dict(list(clash.items())[:5]))
        bad += 1
    for g, info in GROUP_INFO.items():
        path = SLURM_DIR / info["slurm"]
        if not path.exists():
            print(f"  BAD {g}: script {info['slurm']} is missing")
            bad += 1
            continue
        text = path.read_text()
        m = re.search(r"run_array_task\.py[^\n]*--group\s+(\w+)", text)
        rng, n = parse_array(text), len(GROUPS[g])
        ok = m is not None and m.group(1) == g and rng == (0, n - 1)
        t_ok = g == "tier1" or parse_time(text) == time_limit(g)
        print(f"  {'OK ' if ok and t_ok else 'BAD'} {path.name:22s} group={g:11s} array={rng} (group size {n})  time={parse_time(text)}"
              + ("" if g == "tier1" else f" (manifest: {time_limit(g)})"))
        bad += not (ok and t_ok)
    for path in sorted(SLURM_DIR.glob("*.slurm")):       # any other script that runs a group must agree too
        if path.name in {i["slurm"] for i in GROUP_INFO.values()}:
            continue
        m = re.search(r"run_array_task\.py[^\n]*--group\s+(\w+)", path.read_text())
        if m:
            print(f"  BAD {path.name} runs group {m.group(1)} but is not listed in manifest.GROUP_INFO")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
