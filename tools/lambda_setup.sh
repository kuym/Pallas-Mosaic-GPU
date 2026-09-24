#!/usr/bin/env bash
# Bootstrap a Lambda Cloud B200 instance and validate the kernel.
#
#   scp -r . ubuntu@<ip>:splash && ssh ubuntu@<ip> 'bash splash/tools/lambda_setup.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -q --upgrade pip
pip install -q "jax[cuda13]" pytest pytest-xdist absl-py
python -c "import jax; print(jax.__version__, jax.devices())"
# Native Blackwell tests (plus the CPU-interpreter and lowering tiers).
PYTHONPATH=. python -m pytest tests -q -p no:cacheprovider
PYTHONPATH=. python tools/bench.py
