# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Validate and benchmark splash attention on a Blackwell GPU.

  python tools/bench.py                          # default human-readable sweep
  python tools/bench.py --seq 8192 --head-dim 128 --backward
  python tools/bench.py --configs '[{"mask": "causal", "seq": 8192}]' \\
      --jsonl results.jsonl                      # machine-readable (gpu_farm)

Every configuration is first checked against the dense reference (on one KV
head group, to bound the reference's memory), then timed with CUDA events.
TFLOP/s counts only the visible blocks' work (what the kernel actually does);
"dense-equiv" counts the full seq x seq attention.  Failures (compile errors,
wrong results, budget errors) are recorded as results instead of aborting the
batch, so one bad tile configuration never wastes the rest of a GPU's queue.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import splash_attention_mgpu as sa  # noqa: E402

BLOCK_SIZE_KEYS = ("block_kv", "num_stages", "block_kv_dq", "block_q_dkv",
                   "num_stages_bwd")
DEFAULTS = dict(mask="causal", seq=8192, heads=16, kv_heads=None, head_dim=128,
                batch=1, backward=False, check=True, iters=20)


def _time_ms(f, *args, iters):
  from jax.experimental.mosaic.gpu import profiler  # CUDA only.
  _, runtimes_ms = profiler.measure(f, iterations=iters)(*args)
  return statistics.median(runtimes_ms)


def _mask(name, seq):
  return {
      "full": lambda: sa.FullMask((seq, seq)),
      "causal": lambda: sa.CausalMask((seq, seq)),
      "local1k": lambda: sa.LocalMask((seq, seq), (1024, 0), 0),
      "local4k": lambda: sa.LocalMask((seq, seq), (4096, 0), 0),
      "chunked2k": lambda: sa.ChunkedCausalMask((seq, seq), chunk_size=2048),
  }[name]()


def measure(config: dict) -> dict:
  """Checks and times one configuration; returns a JSON-serializable record."""
  cfg = {**DEFAULTS, **config}
  bs = sa.BlockSizes(**{k: cfg[k] for k in BLOCK_SIZE_KEYS if k in cfg})
  seq, heads, d = cfg["seq"], cfg["heads"], cfg["head_dim"]
  kv_heads = cfg["kv_heads"] or heads
  mask = _mask(cfg["mask"], seq)
  kernel = sa.make_splash_mha(mask, block_sizes=bs)
  ks = jax.random.split(jax.random.key(0), 3)
  lead = (cfg["batch"],) if cfg["batch"] > 1 else ()
  q = (jax.random.normal(ks[0], lead + (heads, seq, d), jnp.float32)
       * d ** -0.5).astype(jnp.bfloat16)
  k = jax.random.normal(ks[1], lead + (kv_heads, seq, d), jnp.bfloat16)
  v = jax.random.normal(ks[2], lead + (kv_heads, seq, d), jnp.bfloat16)

  record = dict(cfg, block_sizes=dict(
      (key, getattr(bs, key)) for key in BLOCK_SIZE_KEYS))
  t0 = time.time()
  out = jax.block_until_ready(kernel(q, k, v))
  record["compile_s"] = round(time.time() - t0, 2)
  if cfg["check"]:
    rep = heads // kv_heads
    dense = jnp.asarray(np.asarray(mask[:, :]))[None]
    q0, k0, v0, out0 = (x[0] if lead else x for x in (q, k, v, out))
    ref = sa.attention_reference(dense, q0[:rep], k0[:1], v0[:1])
    record["max_err"] = float(jnp.abs(out0[:rep].astype(jnp.float32)
                                      - ref).max())
    if not record["max_err"] < 3e-2:
      raise AssertionError(f"forward max abs error {record['max_err']}")

  info = kernel.info
  visible = float(info.num_steps.sum()) * (
      heads if info.num_steps.shape[0] == 1 else 1) * cfg["batch"]
  flops = 4 * visible * info.block_q * info.block_kv * d
  dense_flops = 4 * cfg["batch"] * heads * seq * seq * d
  ms = _time_ms(kernel, q, k, v, iters=cfg["iters"])
  record.update(fwd_us=ms * 1e3, fwd_tflops=flops / ms / 1e9,
                fwd_dense_equiv_tflops=dense_flops / ms / 1e9,
                density=info.density)
  if cfg["backward"]:
    grad = jax.jit(jax.grad(
        lambda q, k, v: kernel(q, k, v).astype(jnp.float32).sum(),
        argnums=(0, 1, 2)))
    jax.block_until_ready(grad(q, k, v))
    ms = _time_ms(grad, q, k, v, iters=cfg["iters"])
    # Forward + backward = 2 + 5 matmuls of the visible blocks.
    record.update(fwd_bwd_us=ms * 1e3, fwd_bwd_tflops=flops * 3.5 / ms / 1e9)
  return record


def _describe(r):
  s = (f"{r['mask']:>9} S={r['seq']:<6} H={r['heads']:<3} D={r['head_dim']:<4}"
       f" {r.get('block_sizes', {})}")
  if "error" in r:
    return f"FAIL {s}: {r['error']}"
  s = f"{s}: fwd {r['fwd_us']:8.1f} us {r['fwd_tflops']:7.1f} TFLOP/s"
  if "fwd_bwd_us" in r:
    s += f" | f+b {r['fwd_bwd_us']:8.1f} us {r['fwd_bwd_tflops']:7.1f} TFLOP/s"
  return s


def run_all(configs, jsonl=None):
  for config in configs:
    try:
      record = measure(config)
    except Exception as e:  # noqa: BLE001 -- record and keep going
      record = dict(config, error=f"{type(e).__name__}: {e}"[:2000],
                    traceback=traceback.format_exc()[-4000:])
    record.update(gpu=os.environ.get("FARM_GPU"),
                  job=os.environ.get("FARM_JOB_ID"), time=time.time())
    print(_describe(record), flush=True)
    if jsonl:
      with open(jsonl, "a") as f:  # one short line per write: append-safe
        f.write(json.dumps(record) + "\n")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--configs", help="JSON list of config dicts (or @file)")
  p.add_argument("--jsonl", default=os.environ.get("FARM_RESULTS"))
  p.add_argument("--seq", type=int, nargs="*", default=[2048, 8192, 16384])
  p.add_argument("--heads", type=int, default=16)
  p.add_argument("--kv-heads", type=int, default=None)
  p.add_argument("--head-dim", type=int, nargs="*", default=[64, 128])
  p.add_argument("--mask", nargs="*",
                 default=["full", "causal", "local1k", "chunked2k"])
  p.add_argument("--block-kv", type=int, nargs="*", default=[128])
  p.add_argument("--stages", type=int, nargs="*", default=[2])
  p.add_argument("--no-check", action="store_true")
  p.add_argument("--backward", action="store_true",
                 help="also time forward + backward")
  args = p.parse_args()
  print(jax.devices(), flush=True)
  if args.configs:
    text = args.configs
    if text.startswith("@"):
      text = open(text[1:]).read()
    configs = json.loads(text)
  else:
    configs = [
        dict(mask=m, seq=s, head_dim=d, heads=args.heads,
             kv_heads=args.kv_heads, block_kv=b, num_stages=st,
             backward=args.backward, check=not args.no_check and s <= 8192)
        for s, d, m, b, st in itertools.product(
            args.seq, args.head_dim, args.mask, args.block_kv, args.stages)
    ]
  run_all(configs, args.jsonl)


if __name__ == "__main__":
  main()
