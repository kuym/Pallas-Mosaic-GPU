import sys, time, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "tools")
from lower_check import pretend_arch
import splash_attention_mgpu as sa
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mi

def run(which, KVH=None, B=None, H=2, S=256, D=64, seg=False, cap=None, mqa=False, bs=sa.BlockSizes(), races=True):
  rng = np.random.default_rng(0)
  masks = dict(
    causal=sa.CausalMask((S, S)), full=sa.FullMask((S, S)),
    local=sa.LocalMask((S, S), (100, 20), 0),
    dense=sa.NumpyMask(np.tril(np.ones((S, S), bool)) & (rng.random((S, S)) > 0.3) | np.eye(S, dtype=bool)),
  )
  mask = masks[which]
  dense = np.asarray(mask[:, :])[None]
  kw = dict(block_sizes=bs, attn_logits_soft_cap=cap, interpret=plgpu.InterpretGPUParams(detect_races=races))
  kern = (sa.make_splash_mqa if mqa else sa.make_splash_mha)(mask, **kw)
  ks = jax.random.split(jax.random.key(0), 3)
  q = jax.random.normal(ks[0], (H, S, D), jnp.bfloat16)
  kvshape = (S, D) if mqa else (KVH or H, S, D)
  if B:
    q = jax.random.normal(ks[0], (B, H, S, D), jnp.bfloat16); kvshape = (B,) + kvshape
  k = jax.random.normal(ks[1], kvshape, jnp.bfloat16); v = jax.random.normal(ks[2], kvshape, jnp.bfloat16)
  segs = None
  if seg:
    ids = jnp.asarray(np.repeat(np.arange(4), S // 4)[rng.permutation(S) % 1 == 0], jnp.int32)
    segs = sa.SegmentIds(ids, ids)
  mi.gpu_callbacks.reset_gpu_interpret_mode_state()
  t = time.time()
  with pretend_arch():
    out, (lse,) = kern(q, k, v, segs, save_residuals=True)
  out = np.asarray(out, np.float32)
  ref, (rlse,) = sa.attention_reference(jnp.asarray(dense), q, k, v, segs, is_mqa=mqa, attn_logits_soft_cap=cap, save_residuals=True)
  err = np.abs(out - np.asarray(ref)).max(); lerr = np.abs(np.asarray(lse) - np.asarray(rlse)).max()
  print(f"{which} B={B} KVH={KVH} H={H} S={S} D={D} seg={seg} cap={cap} mqa={mqa} {bs}: out_err={err:.4f} lse_err={lerr:.5f} races={mi.get_races().races_found} ({time.time()-t:.1f}s)")
  return err, lerr

if __name__ == "__main__":
  kw = {}
  for a in sys.argv[2:]:
    k_, v_ = a.split("=", 1); kw[k_] = eval(v_)
  run(sys.argv[1], **kw)
