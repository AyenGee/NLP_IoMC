"""Progress of the experiment groups, from the files on disk (no Slurm needed), and the array ids still to run.

    python experiments/progress.py                       # the seven Tier 2 groups
    python experiments/progress.py --groups weakgen lr1e5 --verbose
    python experiments/progress.py --out-root results/exact

For a collapse job: generations completed / requested (generations.csv rows). For a "train" job: its checkpoint exists. The last line
of each group gives the array ids that are not finished, ready for   sbatch --array=<ids> slurm/t2_<group>.slurm   (a resumed task
continues from its last completed generation, so partially done chains lose nothing).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from manifest import GROUPS  # noqa: E402
from run_array_task import resolve  # noqa: E402

TIER2 = ["weakgen", "continue20", "temperature", "accumulate", "lr1e5", "sweep", "sweep1e4"]


def state(job, out_root: Path) -> tuple[int, int]:
    """(done, total) units of work of a job."""
    r = resolve(job, out_root, {})
    if r["kind"] == "train":
        return (1 if Path(r["kw"]["ckpt_path"]).exists() else 0), 1
    total = int(r["kw"]["n_generations"])
    f = Path(r["kw"]["out_dir"]) / "generations.csv"
    if not f.exists():
        return 0, total
    with open(f, newline="") as fh:
        return min(sum(1 for _ in csv.DictReader(fh)), total), total


def ranges(ids: list[int]) -> str:
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=TIER2)
    ap.add_argument("--out-root", default="results/exact")
    ap.add_argument("--verbose", action="store_true", help="one line per unfinished job")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    for g in args.groups:
        jobs = GROUPS[g]
        st = [state(j, out_root) for j in jobs]
        done = [i for i, (d, t) in enumerate(st) if d >= t]
        started = [i for i, (d, t) in enumerate(st) if 0 < d < t]
        todo = [i for i in range(len(jobs)) if i not in done]
        print(f"{g:12s} finished {len(done):3d}/{len(jobs)}   partly done {len(started):3d}   not started {len(todo) - len(started):3d}")
        if args.verbose:
            for i in todo:
                print(f"    task {i:3d} {jobs[i].run_name}: {st[i][0]}/{st[i][1]}")
        if todo:
            print(f"    unfinished array ids: {ranges(todo)}")


if __name__ == "__main__":
    main()
