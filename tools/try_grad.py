import sys, time, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "tools")
from lower_check import pretend_arch
import splash_attention_mgpu as sa
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mi

def run(which, H=2, KVH=None, S=256, D=64, seg=False, cap=None, mqa=False, bs=sa.BlockSizes(), races=True):
  rng = np.random.default_rng(0)
  masks = dict(causal=sa.CausalMask((S, S)), full=sa.FullMask((S, S)), local=sa.LocalMask((S, S), (100, 20), 0),
    dense=sa.NumpyMask(np.tril(np.ones((S, S), bool)) & (rng.random((S, S)) > 0.3) | np.eye(S, dtype=bool)))
  mask = masks[which]; dense = jnp.asarray(np.asarray(mask[:, :])[None])
  kern = (sa.make_splash_mqa if mqa else sa.make_splash_mha)(mask, block_sizes=bs, attn_logits_soft_cap=cap, interpret=plgpu.InterpretGPUParams(detect_races=races))
  ks = jax.random.split(jax.random.key(0), 4)
  q = jax.random.normal(ks[0], (H, S, D), jnp.bfloat16)
  kvs = (S, D) if mqa else (KVH or H, S, D)
  k = jax.random.normal(ks[1], kvs, jnp.bfloat16); v = jax.random.normal(ks[2], kvs, jnp.bfloat16)
  w = jax.random.normal(ks[3], (H, S, D), jnp.float32)
  segs = None
  if seg:
    ids = jnp.asarray(np.repeat(np.arange(4), S // 4), jnp.int32); segs = sa.SegmentIds(ids, ids)
  loss = lambda f: (lambda q, k, v: jnp.sum(f(q, k, v).astype(jnp.float32) * w))
  mi.gpu_callbacks.reset_gpu_interpret_mode_state()
  t = time.time()
  with pretend_arch():
    g = jax.grad(loss(lambda q, k, v: kern(q, k, v, segs)), argnums=(0, 1, 2))(q, k, v)
  races_found = mi.get_races().races_found
  gr = jax.grad(loss(lambda q, k, v: sa.attention_reference(dense, q, k, v, segs, is_mqa=mqa, attn_logits_soft_cap=cap)), argnums=(0, 1, 2))(q, k, v)
  errs = [float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max() / (np.abs(np.asarray(b, np.float32)).max())) for a, b in zip(g, gr)]
  print(f"{which} H={H} KVH={KVH} S={S} D={D} seg={seg} cap={cap} mqa={mqa}: rel err dq={errs[0]:.4f} dk={errs[1]:.4f} dv={errs[2]:.4f} races={races_found} ({time.time()-t:.1f}s)")

if __name__ == "__main__":
  kw = {}
  for a in sys.argv[2:]:
    k_, v_ = a.split("=", 1); kw[k_] = eval(v_)
  run(sys.argv[1], **kw)
