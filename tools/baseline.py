# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Measure cuDNN flash attention on the same problems, as a practical ceiling.

  python tools/baseline.py            # results -> $FARM_RESULTS (kind=baseline)

Uses jax.nn.dot_product_attention(implementation="cudnn"), NVIDIA's tuned
Blackwell attention, for full and causal masks.  TFLOP/s use the same
visible-work accounting as tools/bench.py (causal counts ~half of S^2).
"""

import json
import os
import statistics
import sys
import time
import traceback

import jax
import jax.numpy as jnp


def measure(seq, d, heads, kv_heads, causal, backward, iters=20):
  from jax.experimental.mosaic.gpu import profiler
  ks = jax.random.split(jax.random.key(0), 3)
  # cuDNN layout: [batch, seq, heads, head_dim].
  q = jax.random.normal(ks[0], (1, seq, heads, d), jnp.bfloat16)
  k = jax.random.normal(ks[1], (1, seq, kv_heads, d), jnp.bfloat16)
  v = jax.random.normal(ks[2], (1, seq, kv_heads, d), jnp.bfloat16)
  attn = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(
      q, k, v, is_causal=causal, implementation="cudnn"))
  f = attn
  if backward:
    f = jax.jit(jax.grad(
        lambda q, k, v: attn(q, k, v).astype(jnp.float32).sum(),
        argnums=(0, 1, 2)))
  jax.block_until_ready(f(q, k, v))
  _, ms = profiler.measure(f, iterations=iters)(q, k, v)
  ms = statistics.median(ms)
  # Visible work: causal visits the lower triangle (incl. diagonal blocks).
  flops = 4 * heads * seq * seq * d * (0.5 if causal else 1.0)
  flops *= 3.5 if backward else 1.0
  return ms * 1e3, flops / ms / 1e9


BACKWARD = (False, True)


def main():
  global BACKWARD
  if "--backward-only" in sys.argv:
    BACKWARD = (True,)
  out = os.environ.get("FARM_RESULTS")
  for seq in (4096, 16384):
    for d in (64, 128):
      for heads, kvh in ((16, 16), (32, 8)):
        for causal in (False, True):
          for backward in BACKWARD:
            rec = dict(kind="baseline", impl="cudnn", seq=seq, head_dim=d,
                       heads=heads, kv_heads=kvh,
                       mask="causal" if causal else "full", backward=backward)
            try:
              us, tf = measure(seq, d, heads, kvh, causal, backward)
              rec.update(us=us, tflops=tf)
            except Exception as e:  # noqa: BLE001
              rec.update(error=f"{type(e).__name__}: {e}"[:500],
                         traceback=traceback.format_exc()[-2000:])
            rec.update(gpu=os.environ.get("FARM_GPU"), time=time.time())
            print(json.dumps(rec)[:300], flush=True)
            if out:
              with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
  sys.exit(main())
