#!/bin/bash
# Build the virtualenv on mscluster ONCE (run on the login node; the installs are light):
#   bash slurm/setup_env.sh
# Uses the cluster's system python3 (no `module load` needed) and a CPU build of torch. After this, every
# slurm/*.slurm script does `source .venv/bin/activate`. Also download the embeddings (see README.txt, section 1):
#   mkdir -p data && curl -L -o data/omniglot_resnet18_randomized_order_s0.h5 \
#     https://github.com/aadityasingh/icl-dynamics/raw/main/omniglot_resnet18_randomized_order_s0.h5
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import torch, h5py; print('torch', torch.__version__, '| h5py', h5py.__version__)"
