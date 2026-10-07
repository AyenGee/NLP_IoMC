"""Resolve one Slurm array task index to a job from manifest.py and run it (one collapse chain, or one training, per task).

    python experiments/run_array_task.py --group tier1 --task-id $SLURM_ARRAY_TASK_ID
    python experiments/run_array_task.py --group weakgen --task-id 3 --dry-run          # print the resolved job, run nothing
    python experiments/run_array_task.py --group lr1e5 --task-id 0 --out-root results/smoke_tier2 \\
        --override steps=40 --override sequences=null --override eval_every=20 --override n_generations=2     # a smoke run

The index is relative to the group. Output (under --out-root, default results/exact): collapse/<config stem>_s<seed>/ for a
collapse chain (re-running the same task resumes it from its last completed generation, see src/collapse.py), and
logs/<name>.csv + checkpoints/<name>.pt for a "train" job (skipped if its checkpoint already exists).
--override KEY=VALUE (repeatable, VALUE read as YAML) replaces a config field after the config and the job's own overrides.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from manifest import CONFIG_DIR, group_jobs  # noqa: E402


def parse_overrides(items: list[str]) -> dict:
    import yaml
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--override expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def resolve(job, out_root: Path, cli_overrides: dict) -> dict:
    """The fully resolved settings of a job: kind, name, config kwargs (a dict of CollapseConfig / TrainConfig fields)."""
    from utils import load_config
    if job.kind == "collapse":
        kw = dict(load_config(CONFIG_DIR / job.config)["collapse"])
    else:
        kw = {}
    kw.update(dict(job.overrides))
    kw.update(cli_overrides)
    if ("steps" in cli_overrides or "steps" in dict(job.overrides)) and "sequences" not in cli_overrides and "sequences" not in dict(job.overrides):
        kw["sequences"] = None                      # an explicit step budget replaces the configured sequence budget
    kw["seed"] = job.seed
    name = job.run_name
    if job.kind == "collapse":
        kw["out_dir"] = str(out_root / "collapse" / name)
    else:
        kw["log_path"] = str(out_root / "logs" / f"{name}.csv")
        kw["ckpt_path"] = str(out_root / "checkpoints" / f"{name}.pt")
    return {"kind": job.kind, "name": name, "kw": kw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--group", default="tier1")
    ap.add_argument("--out-root", default="results/exact")
    ap.add_argument("--override", action="append", default=[], help="KEY=VALUE, repeatable (YAML values)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="sleep task_id * this many seconds (plus a random 0-2 s) first, so that array tasks do not all create the shared "
                         "output directories and read the same files in the same instant")
    args = ap.parse_args()
    jobs = group_jobs(args.group)
    if not (0 <= args.task_id < len(jobs)):
        raise SystemExit(f"task-id {args.task_id} out of range [0, {len(jobs)}) for group {args.group!r} "
                         "-- check the script's --array against experiments/manifest.py (python3 experiments/check_manifest.py)")
    job = jobs[args.task_id]
    r = resolve(job, Path(args.out_root), parse_overrides(args.override))
    if args.dry_run:
        print(f"[{args.group} task {args.task_id}/{len(jobs) - 1}] {r['kind']} {r['name']}: {r['kw']}")
        return
    if args.stagger:
        time.sleep(args.task_id * args.stagger + random.uniform(0, 2))

    if r["kind"] == "collapse":
        from collapse import CollapseConfig, run_collapse
        cfg = CollapseConfig(**r["kw"])
        print(f"[{args.group} task {args.task_id}/{len(jobs) - 1}] {r['name']}: {cfg.variant} p={cfg.p_synthetic} mode={cfg.mode} T={'greedy' if cfg.greedy else cfg.temperature} "
              f"lr={cfg.lr} {cfg.steps if cfg.steps else str(cfg.sequences) + ' sequences'} x {cfg.n_generations} generations"
              + (f" (generation 0: {cfg.gen0_steps} steps)" if cfg.gen0_steps else ""), flush=True)
        t = time.time()
        run_collapse(cfg)
    else:
        from train import TrainConfig, train_model
        kw = dict(r["kw"])
        if "loss_weights" in kw:
            kw["loss_weights"] = tuple(kw["loss_weights"])
        cfg = TrainConfig(**kw)
        if Path(cfg.ckpt_path).exists():
            print(f"[{args.group} task {args.task_id}] {r['name']}: checkpoint exists, skipped", flush=True)
            return
        print(f"[{args.group} task {args.task_id}/{len(jobs) - 1}] {r['name']}: {cfg.variant} lr={cfg.lr} {cfg.n_steps} steps, loss weights {cfg.loss_weights}", flush=True)
        t = time.time()
        train_model(cfg)
    print(f"[{args.group} task {args.task_id}] done in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
