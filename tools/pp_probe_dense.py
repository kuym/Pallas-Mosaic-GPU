"""Localize ping-pong dense-mask mismatches on hardware (debug tool)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
import jax.numpy as jnp
import numpy as np

import splash_attention_mgpu as sa

S, D, H = 2048, 64, 2
rng = np.random.default_rng(0)
dense = np.tril(np.ones((S, S), bool)) & (rng.random((S, S)) > 0.3)
dense |= np.eye(S, dtype=bool)
q, k, v = (jax.random.normal(jax.random.key(i), (H, S, D), jnp.bfloat16)
           for i in range(3))
ref = np.asarray(sa.attention_reference(jnp.asarray(dense)[None], q, k, v))

for bq, bkv in [(128, 64), (128, 128), (256, 64), (256, 128)]:
  for trial in range(3):
    kern = sa.make_splash_mha(sa.NumpyMask(dense),
                              block_sizes=sa.BlockSizes(block_q=bq, block_kv=bkv))
    out = np.asarray(kern(q, k, v), np.float32)
    bad = np.abs(out - ref) > 0.05
    rows = np.unique(np.nonzero(bad.any(-1))[1])
    print(f"bq={bq} bkv={bkv} trial={trial}: bad elems {int(bad.sum())}, "
          f"bad rows {len(rows)} {rows[:12].tolist()} "
          f"(row%256: {sorted(set((rows % 256).tolist()))[:12]}; "
          f"heads {np.unique(np.nonzero(bad.any(-1))[0]).tolist()})", flush=True)
