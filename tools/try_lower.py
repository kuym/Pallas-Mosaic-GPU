import sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "tools")
from lower_check import lower_for_blackwell
import splash_attention_mgpu as sa

def run(mask, H=2, S=512, D=128, seg=False, residuals=False, bs=sa.BlockSizes(), cap=None, mqa=False):
  kern = sa.make_splash_mqa(mask, block_sizes=bs, attn_logits_soft_cap=cap) if mqa else sa.make_splash_mha(mask, block_sizes=bs, attn_logits_soft_cap=cap)
  q = jax.ShapeDtypeStruct((H, S, D), jnp.bfloat16)
  kv = jax.ShapeDtypeStruct((S, D) if mqa else (H, S, D), jnp.bfloat16)
  args = [q, kv, kv]
  if seg:
    args.append(sa.SegmentIds(jax.ShapeDtypeStruct((S,), jnp.int32), jax.ShapeDtypeStruct((S,), jnp.int32)))
  txt = lower_for_blackwell(lambda *a: kern(*a, save_residuals=residuals), *args)
  assert "mosaic_gpu" in txt
  return "ok"

if __name__ == "__main__":
  which = sys.argv[1]
  S = 512
  masks = dict(
    causal=sa.CausalMask((S, S)),
    full=sa.FullMask((S, S)),
    local=sa.LocalMask((S, S), (128, 0), 0),
    dense=sa.NumpyMask(np.tril(np.ones((S, S), bool)) & (np.random.default_rng(0).random((S, S)) > 0.3)),
  )
  kw = {}
  for a in sys.argv[2:]:
    k_, v_ = a.split("=", 1); kw[k_] = eval(v_)
  print(which, kw, run(masks[which], S=S, **kw))
