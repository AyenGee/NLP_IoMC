#!/bin/bash
# One-time environment setup. Run this directly on the mscluster LOGIN node
# (not via sbatch) -- creating a venv and pip-installing is preparation
# work, not a computation, so it's fine there per the HPC guide.
#
#   ssh <your-username>@<mscluster-login-host>   # TODO: fill in from onboarding
#   cd <path-to-this-repo-on-the-cluster>
#   bash slurm/setup_env.sh
#
# Re-run safely any time; it just recreates .venv.

set -euo pipefail

echo "=== Available python modules (pick one below) ==="
module avail python 2>&1 | head -30
echo "==================================================="

module purge
module load python/3.11   # TODO: replace with the exact name/version shown above

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU build by default. If you're targeting a GPU partition, load the
# matching CUDA module above FIRST (check `module avail cuda`), then install
# the matching CUDA build of torch from https://pytorch.org/get-started/locally/
# instead of the line below -- configs already use device: auto, so the code
# will pick up the GPU automatically once a CUDA-enabled torch is installed.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo ""
echo "Environment ready in .venv/. Activate it in future sessions with:"
echo "  source .venv/bin/activate"
