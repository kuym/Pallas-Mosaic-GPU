# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Run one small ping-pong forward case and report (used to debug hangs)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp, numpy as np
import splash_attention_mgpu as sa
mask = {"full": sa.FullMask, "causal": sa.CausalMask}[sys.argv[1]]((1024, 1024))
kern = sa.make_splash_mha(mask, block_sizes=sa.BlockSizes(block_q=256))
q, k, v = (jax.random.normal(jax.random.key(i), (2, 1024, 128), jnp.bfloat16) for i in range(3))
t = time.time()
out = jax.block_until_ready(kern(q, k, v))
ref = sa.attention_reference(jnp.asarray(np.asarray(mask[:, :]))[None], q, k, v)
print("PROBE", sys.argv[1], os.environ.get("SPLASH_PP_REGS"), "ok in %.1fs" % (time.time() - t),
      "max err", float(jnp.abs(out.astype(jnp.float32) - ref).max()), flush=True)
