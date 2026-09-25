#!/usr/bin/env bash
# Bring up an 8-GPU host (e.g. a Together 8xB200 node) and keep every GPU busy.
#
#   rsync -a --exclude .venv --exclude farm . HOST:splash/
#   ssh HOST 'cd splash && bash tools/multi_gpu_launch.sh'
#
# 1. Installs JAX with the CUDA wheel matching the driver (CPU work only).
# 2. Starts tools/gpu_farm.py (nohup, survives ssh disconnects) on all GPUs,
#    with the correctness fuzzer as pre-emptible filler.
# 3. Queues, in priority order:
#      p0   the native test suite, sharded one shard per GPU
#      p20  the tile-size tuning sweep (tools/sweep.py)
#    while the CPU-only test tiers (interpreter + sm_100 lowering) run on the
#    host CPUs in parallel, outside the farm.
# Check progress with:  python tools/gpu_farm.py status --root farm
#                       python tools/summarize.py farm/results.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
ngpu=$(nvidia-smi --list-gpus | wc -l)
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
cuda_major=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
if [ "${cuda_major:-12}" -ge 13 ]; then extra="cuda13"; else extra="cuda12"; fi

if [ ! -x .venv-gpu/bin/python ]; then
  python3 -m venv .venv-gpu || { pip3 install --user -q uv && ~/.local/bin/uv venv .venv-gpu; }
fi
. .venv-gpu/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q "jax[${extra}]" pytest pytest-xdist absl-py
python -c "import jax; print(jax.__version__, jax.devices())"
export PYTHONPATH=$PWD

if pgrep -f "gpu_farm.py serve" >/dev/null; then
  echo "farm already running"
else
  nohup python tools/gpu_farm.py serve --root farm --gpus "0-$((ngpu - 1))" \
      --filler "python tools/fuzz.py --minutes 10" > farm.log 2>&1 &
  echo "farm started (pid $!)"
fi

for i in $(seq 0 $((ngpu - 1))); do
  python tools/gpu_farm.py submit --root farm --priority 0 --name "tests$i/$ngpu" \
      -- env FARM_SHARD="$i/$ngpu" timeout 1800 python -m pytest tests -q -p no:cacheprovider \
         -rs -k "gpu_" >/dev/null
done
python tools/sweep.py --root farm --priority 20

# CPU-only tiers on the host CPUs, concurrently with the GPU work.
nohup env JAX_PLATFORMS=cpu python -m pytest tests -q -p no:cacheprovider \
    -n "$(( $(nproc) / 4 > 1 ? $(nproc) / 4 : 1 ))" -k "not gpu_" \
    > cpu_tests.log 2>&1 &
echo "CPU test tiers running (cpu_tests.log)"
python tools/gpu_farm.py status --root farm
