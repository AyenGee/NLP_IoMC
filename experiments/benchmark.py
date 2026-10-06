"""Training-step throughput on ONE core, for planning the Slurm budget.

    python experiments/benchmark.py                 # base and extended, 400 timed steps each
    sbatch slurm/benchmark.slurm                    # the same on a cluster compute node (what GATE 1 asks for)

Times the full training step (batch assembly, forward, backward, Adam update) and, separately, one evaluation pass
(five sets of 1,000 sequences plus the attention metrics), at batch size 32. Prints a line per variant plus the compute
implied for the paper's budget (1e6 sequences = 31,250 steps) and for the plan's literal budget (1,000,000 steps).
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import data as D  # noqa: E402
import metrics as M  # noqa: E402
from model import IHTransformer, ModelConfig  # noqa: E402
from train import EvalSets, evaluate  # noqa: E402


def bench(variant: str, features, n_warm: int = 50, n: int = 400) -> dict:
    ext = variant == "extended"
    torch.manual_seed(0)
    model = IHTransformer(ModelConfig(extended=ext))
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    sampler = D.TrainSampler(D.enumerate_train_space(ext), features, ext, seed=1)

    def step():
        b = sampler.next_batch(32)
        loss = M.loss_terms(model(b.ex, b.labels), b, ext)["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    for _ in range(n_warm):
        step()
    t = time.time()
    for _ in range(n):
        step()
    sps = n / (time.time() - t)
    evals = EvalSets(features, ext)
    t = time.time(); evaluate(model, evals); eval_s = time.time() - t
    return {"variant": variant, "steps_per_s": sps, "eval_s": eval_s, "params": model.n_parameters()}


def main():
    torch.set_num_threads(1)
    features = D.OmniglotFeatures().tensor
    print(f"host {platform.node()} | {platform.processor()} | torch {torch.__version__} | threads {torch.get_num_threads()}")
    for variant in ("base", "extended"):
        r = bench(variant, features)
        paper = 31250 / r["steps_per_s"]
        print(f"{variant:9s} {r['params']:6d} params | {r['steps_per_s']:6.1f} steps/s | one evaluation {r['eval_s']:.2f}s | "
              f"31,250 steps = {paper / 60:.1f} min | 1,000,000 steps = {1e6 / r['steps_per_s'] / 3600:.1f} h")


if __name__ == "__main__":
    main()
