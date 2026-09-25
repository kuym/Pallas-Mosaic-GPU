#!/usr/bin/env bash
# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
# Bootstrap a Lambda Cloud B200 instance and validate the kernel.
#
#   scp -r . ubuntu@<ip>:splash && ssh ubuntu@<ip> 'bash splash/tools/lambda_setup.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
# Pick the JAX CUDA wheel supported by the installed driver.
cuda_major=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
if [ "${cuda_major:-12}" -ge 13 ]; then extra="cuda13"; else extra="cuda12"; fi
echo "driver CUDA ${cuda_major}: installing jax[${extra}]"
if [ ! -d .venv-gpu ]; then
  python3 -m venv .venv-gpu
fi
. .venv-gpu/bin/activate
pip install -q --upgrade pip
pip install -q "jax[${extra}]" pytest pytest-xdist absl-py
python -c "import jax; print(jax.__version__, jax.devices(), jax.devices()[0].compute_capability)"
if [ "${1:-}" != "--bench-only" ]; then
  # Native Blackwell tests (plus the CPU-interpreter and lowering tiers).
  PYTHONPATH=. python -m pytest tests -q -p no:cacheprovider -x -rs
fi
PYTHONPATH=. python tools/bench.py --backward
