#!/usr/bin/env bash
# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
# Recreate the development virtualenv (.venv) from requirements.txt.
#
#   bash tools/setup_venv.sh            # create .venv (fails if it exists)
#   bash tools/setup_venv.sh --force    # delete and recreate .venv
#
# JAX 0.11 needs Python >= 3.11.  Uses the first suitable python3.x on PATH,
# otherwise creates a Python 3.12 environment with conda or uv.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.venv

if [ -e "$VENV" ]; then
  if [ "${1:-}" = "--force" ]; then
    rm -rf "$VENV"
  else
    echo "$VENV already exists (use --force to recreate)" >&2
    exit 1
  fi
fi

python_ok() {  # $1: interpreter; true if Python >= 3.11
  "$1" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null
}

PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null && python_ok "$cand"; then
    PY=$(command -v "$cand")
    break
  fi
done

if [ -n "$PY" ]; then
  echo "creating $VENV with $PY"
  "$PY" -m venv "$VENV"
elif command -v conda >/dev/null; then
  echo "creating $VENV with conda (python 3.12)"
  conda create -q -y -p "$VENV" python=3.12 >/dev/null
elif command -v uv >/dev/null; then
  echo "creating $VENV with uv (python 3.12)"
  uv venv --python 3.12 "$VENV"
else
  echo "need Python >= 3.11, conda, or uv" >&2
  exit 1
fi

"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r requirements.txt
"$VENV/bin/python" -c 'import jax; print("jax", jax.__version__, jax.devices())'
echo "done: PYTHONPATH=. $VENV/bin/python -m pytest tests -n 8"
