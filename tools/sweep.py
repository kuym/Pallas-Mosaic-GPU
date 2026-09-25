# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Submit a tile-size tuning sweep to gpu_farm.

  python tools/sweep.py --root farm [--quick] [--priority 20]

Forward parameters (block_kv, num_stages) and backward parameters
(block_kv_dq, block_q_dkv, num_stages_bwd) are swept separately, for every
(mask, seq, head_dim) problem.  Configurations are shuffled so that every job
gets a similar mix of heavy and light problems, then chunked into jobs of
--chunk configs (one process per chunk amortizes JAX startup).  Configurations
that exceed the SMEM/TMEM budgets fail fast and are recorded as such.
"""

import argparse
import itertools
import json
import os
import random
import shlex
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gpu_farm  # noqa: E402

MASKS = ["full", "causal", "local1k", "chunked2k"]


def configs(quick=False, parts=("fwd", "bwd"), block_qs=(128,)):
  seqs = [4096, 16384] if not quick else [8192]
  dims = [64, 128]
  heads = [(16, 16), (32, 8)] if not quick else [(16, 16)]
  out = []
  for mask, seq, d, (h, kvh) in itertools.product(MASKS, seqs, dims, heads):
    base = dict(mask=mask, seq=seq, head_dim=d, heads=h, kv_heads=kvh,
                check=seq <= 8192)
    if "fwd" in parts:
      for bq, bkv, st in itertools.product(block_qs, [64, 128], [2, 3, 4]):
        out.append(dict(base, block_q=bq, block_kv=bkv, num_stages=st))
    if "bwd" in parts and d <= 128:
      dkv_sizes = [64, 128] if d <= 64 else [64]  # 128 exceeds TMEM at d=128
      for dq, dkv, st in itertools.product([64, 128], dkv_sizes, [1, 2, 3]):
        out.append(dict(base, backward=True, block_kv_dq=dq, block_q_dkv=dkv,
                        num_stages_bwd=st))
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--root", default="farm")
  p.add_argument("--priority", type=int, default=20)
  p.add_argument("--chunk", type=int, default=12)
  p.add_argument("--quick", action="store_true")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--parts", default="fwd,bwd")
  p.add_argument("--block-q", default="128", help="comma list, e.g. 128,256")
  p.add_argument("--tag", default="sweep")
  args = p.parse_args()
  cfgs = configs(args.quick, args.parts.split(","),
                 tuple(int(x) for x in args.block_q.split(",")))
  random.Random(0).shuffle(cfgs)
  chunks = [cfgs[i:i + args.chunk] for i in range(0, len(cfgs), args.chunk)]
  print(f"{len(cfgs)} configs in {len(chunks)} jobs")
  if args.dry_run:
    return
  os.makedirs(os.path.join(args.root, "sweeps"), exist_ok=True)
  for n, chunk in enumerate(chunks):
    path = os.path.abspath(
        os.path.join(args.root, "sweeps", f"{args.tag}{n:04d}.json"))
    with open(path, "w") as f:
      json.dump(chunk, f)
    # A hard timeout: a hung kernel must never hold a GPU hostage.
    cmd = f"timeout 1200 python tools/bench.py --configs @{shlex.quote(path)}"
    gpu_farm.submit(args.root, cmd, args.priority, f"{args.tag}{n:04d}")


if __name__ == "__main__":
  main()
