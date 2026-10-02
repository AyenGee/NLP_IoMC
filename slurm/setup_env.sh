#!/bin/bash
# One-time environment setup. Run this directly on the mscluster LOGIN node
# (not via sbatch) -- creating a venv and pip-installing is preparation
# work, not a computation, so it's fine there per the HPC guide.
#
#   ssh eazubuike@<mscluster-login-host>
#   mkdir -p ~/projects && cd ~/projects
#   git clone https://github.com/AyenGee/NLP_IoMC.git
#   cd NLP_IoMC
#   bash slurm/setup_env.sh
#
# Re-run safely any time; it just recreates .venv.

set -euo pipefail

# Python: mscluster's system python3 (3.14.x at time of writing) - no
# `module load` needed. It is the same interpreter on the login and compute
# nodes, so the .venv built here works inside every slurm/*.slurm job.
# torch==2.13.0 publishes cp314 wheels, so the pins below need no loosening.
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"python3 is {sys.version.split()[0]}; this project needs 3.10+")
print(f"Using system python3 {sys.version.split()[0]}")
PY

python3 -m venv .venv
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
