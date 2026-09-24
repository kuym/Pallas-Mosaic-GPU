# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Validate and benchmark splash attention on a Blackwell GPU.

  python tools/bench.py                 # default sweep
  python tools/bench.py --seq 8192 --heads 16 --head-dim 128

For every configuration the kernel output is checked against the dense
reference (on a slice of heads, to bound reference memory), then timed.
TFLOP/s counts only the visible (unmasked) blocks' work, i.e. the work the
kernel actually does; "dense-equiv" counts the full seq x seq attention.
"""

import argparse
import os
import statistics
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import splash_attention_mgpu as sa  # noqa: E402


def _time(f, *args, iters=20):
  from jax.experimental.mosaic.gpu import profiler  # CUDA only.
  _, runtimes_ms = profiler.measure(f, iterations=iters)(*args)
  return statistics.median(runtimes_ms)


def run(mask_name, seq, heads, kv_heads, head_dim, block_sizes, check=True):
  masks = {
      "full": sa.FullMask((seq, seq)),
      "causal": sa.CausalMask((seq, seq)),
      "local1k": sa.LocalMask((seq, seq), (1024, 0), 0),
      "chunked2k": sa.ChunkedCausalMask((seq, seq), chunk_size=2048),
  }
  mask = masks[mask_name]
  kernel = sa.make_splash_mha(mask, block_sizes=block_sizes)
  ks = jax.random.split(jax.random.key(0), 3)
  scale = head_dim ** -0.5
  q = (jax.random.normal(ks[0], (heads, seq, head_dim), jnp.float32) * scale
       ).astype(jnp.bfloat16)
  k = jax.random.normal(ks[1], (kv_heads, seq, head_dim), jnp.bfloat16)
  v = jax.random.normal(ks[2], (kv_heads, seq, head_dim), jnp.bfloat16)

  out = jax.block_until_ready(kernel(q, k, v))
  if check:
    # Check the first KV head group only, to bound the reference's memory.
    rep = heads // kv_heads
    dense = jnp.asarray(np.asarray(mask[:, :]))[None]
    ref = sa.attention_reference(dense, q[:rep], k[:1], v[:1])
    err = float(jnp.abs(out[:rep].astype(jnp.float32) - ref).max())
    assert err < 3e-2, f"max abs error {err}"
  else:
    err = float("nan")

  ms = _time(kernel, q, k, v)
  info = kernel.info
  visible_blocks = float(info.num_steps.sum()) * (
      heads if info.num_steps.shape[0] == 1 else 1)
  flops = 4 * visible_blocks * info.block_q * info.block_kv * head_dim
  dense_flops = 4 * heads * seq * seq * head_dim
  print(
      f"{mask_name:>9} S={seq:<6} H={heads:<3} KVH={kv_heads:<3} D={head_dim:<4}"
      f" {block_sizes}: {ms * 1e3:8.1f} us  "
      f"{flops / ms / 1e9:7.1f} TFLOP/s  "
      f"(dense-equiv {dense_flops / ms / 1e9:7.1f})  "
      f"density={info.density:.3f} err={err:.4f}"
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--seq", type=int, nargs="*", default=[2048, 8192, 16384])
  p.add_argument("--heads", type=int, default=16)
  p.add_argument("--kv-heads", type=int, default=None)
  p.add_argument("--head-dim", type=int, nargs="*", default=[64, 128])
  p.add_argument("--mask", nargs="*",
                 default=["full", "causal", "local1k", "chunked2k"])
  p.add_argument("--block-kv", type=int, nargs="*", default=[128])
  p.add_argument("--stages", type=int, nargs="*", default=[2])
  p.add_argument("--no-check", action="store_true")
  args = p.parse_args()
  print(jax.devices())
  for seq in args.seq:
    for d in args.head_dim:
      for mask in args.mask:
        for bkv in args.block_kv:
          for stages in args.stages:
            try:
              bs = sa.BlockSizes(block_kv=bkv, num_stages=stages)
              run(mask, seq, args.heads, args.kv_heads or args.heads, d, bs,
                  check=not args.no_check and seq <= 8192)
            except ValueError as e:
              print(f"skip {mask} S={seq} D={d} bkv={bkv} stages={stages}: {e}")


if __name__ == "__main__":
  main()
