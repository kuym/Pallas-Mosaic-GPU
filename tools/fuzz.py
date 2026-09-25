# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Randomized correctness fuzzing of splash attention on one GPU.

  python tools/fuzz.py --minutes 10            # random seeds until time is up
  python tools/fuzz.py --seed 12345            # replay one case

Each case draws a mask (causal / local / chunked / full / random dense /
per-head mixes), shapes (batch, GQA/MQA, head dims, sequence lengths),
features (segment ids, soft-cap, f16/bf16) and block sizes, then checks the
forward output, logsumexp and dQ/dK/dV against the dense f32 reference.
Results (pass or fail, with the seed) go to $FARM_RESULTS as JSON lines.
Used as gpu_farm's `--filler`, so otherwise idle GPUs keep hunting for
hardware-only bugs.
"""

import argparse
import json
import os
import random
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import splash_attention_mgpu as sa  # noqa: E402


def draw(seed):
  r = random.Random(seed)
  seq = r.choice([256, 512, 1024, 2048, 3072])
  head_dim = r.choice([64, 128, 128])
  grad = r.random() < 0.6
  block_kv = r.choice([64, 128]) if not grad else 128
  heads = r.choice([1, 2, 3, 4, 8])
  kv_heads = r.choice([h for h in (1, 2, heads) if heads % h == 0])
  case = dict(
      seed=seed, seq=seq, head_dim=head_dim, heads=heads, kv_heads=kv_heads,
      batch=r.choice([None, None, 2]), mqa=kv_heads == 1 and r.random() < 0.5,
      dtype=r.choice(["bfloat16", "bfloat16", "float16"]),
      segments=r.random() < 0.3, cap=r.choice([None, None, 5.0, 30.0]),
      grad=grad, per_head_masks=r.random() < 0.2,
      mask=r.choice(["full", "causal", "local", "chunked", "dense"]),
      block_sizes=dict(
          block_kv=block_kv, num_stages=r.choice([2, 2, 3]),
          block_kv_dq=r.choice([64, 128]),
          block_q_dkv=64 if head_dim == 128 else r.choice([64, 128]),
          num_stages_bwd=r.choice([1, 2, 3])),
  )
  if case["block_sizes"]["block_kv_dq"] > block_kv:
    case["block_sizes"]["block_kv_dq"] = block_kv
  # Half the cases exercise the two-tile ping-pong forward kernel.
  if seq % 256 == 0 and r.random() < 0.5:
    case["block_sizes"]["block_q"] = 256
    if case["mask"] == "dense" or case["per_head_masks"]:
      case["block_sizes"]["block_kv"] = 64  # SMEM for 256-row mask blocks
      case["block_sizes"]["block_kv_dq"] = 64
  return case


def _mask(kind, seq, rng):
  if kind == "full":
    return sa.FullMask((seq, seq))
  if kind == "causal":
    return sa.CausalMask((seq, seq))
  if kind == "local":
    return sa.LocalMask((seq, seq), (int(rng.integers(0, seq)),
                                     int(rng.integers(0, seq // 2))), 0)
  if kind == "chunked":
    return sa.ChunkedCausalMask((seq, seq),
                                chunk_size=int(rng.integers(1, seq // 64 + 1)) * 64)
  dense = rng.random((seq, seq)) < rng.uniform(0.05, 0.9)
  dense |= np.eye(seq, dtype=bool)  # no fully masked rows
  if rng.random() < 0.5:
    dense = np.tril(dense)
  return sa.NumpyMask(dense)


def run_case(case, interpret=None):
  rng = np.random.default_rng(case["seed"])
  seq, h = case["seq"], case["heads"]
  if case["per_head_masks"]:
    kinds = ["full", "causal", "local", "chunked", "dense"]
    masks = [_mask(rng.choice(kinds), seq, rng) for _ in range(h)]
    mask = sa.MultiHeadMask(masks)
  else:
    masks = [_mask(case["mask"], seq, rng)]
    mask = masks[0]
  dense = jnp.asarray(np.stack([np.asarray(m[:, :]) for m in masks]))
  bs = sa.BlockSizes(**case["block_sizes"])
  mqa = case["mqa"]
  make = sa.make_splash_mqa if mqa else sa.make_splash_mha
  kernel = make(mask, block_sizes=bs, attn_logits_soft_cap=case["cap"],
                interpret=interpret)

  dtype = jnp.dtype(case["dtype"])
  lead = (case["batch"],) if case["batch"] else ()
  kv_lead = lead + (() if mqa else (case["kv_heads"],))
  keys = jax.random.split(jax.random.key(case["seed"]), 4)
  d = case["head_dim"]
  q = (jax.random.normal(keys[0], lead + (h, seq, d), jnp.float32)
       * rng.uniform(0.05, 0.5)).astype(dtype)
  k = jax.random.normal(keys[1], kv_lead + (seq, d), dtype)
  v = jax.random.normal(keys[2], kv_lead + (seq, d), dtype)
  seg = None
  if case["segments"]:
    cuts = np.sort(rng.choice(np.arange(1, seq), size=3, replace=False))
    ids = jnp.asarray(np.searchsorted(cuts, np.arange(seq), side="right"),
                      jnp.int32)
    seg = sa.SegmentIds(ids, ids)  # the diagonal stays visible

  out, (lse,) = kernel(q, k, v, seg, save_residuals=True)
  ref, (ref_lse,) = sa.attention_reference(
      dense, q, k, v, seg, is_mqa=mqa, attn_logits_soft_cap=case["cap"],
      save_residuals=True)
  # Rows with no visible key (possible with local masks + segments) are
  # implementation-defined; compare only rows that see at least one key.
  vis = np.asarray(dense).any(-1)  # [mask_heads, q]
  errs = {}
  out32, ref32 = np.asarray(out, np.float32), np.asarray(ref, np.float32)
  row_ok = np.broadcast_to(vis[..., None], out32.shape[-3:])
  errs["out"] = float(np.abs(out32 - ref32)[..., row_ok].max()) if row_ok.any() else 0.0
  lse_ok = np.broadcast_to(vis, np.shape(lse)[-2:])
  errs["lse"] = float(np.abs(np.asarray(lse) - np.asarray(ref_lse))[..., lse_ok].max())
  tol = 3e-2 if dtype == jnp.bfloat16 else 8e-3
  ok = errs["out"] < tol and errs["lse"] < 2e-3

  if case["grad"] and ok:
    w = jax.random.normal(keys[3], q.shape, jnp.float32)
    loss = lambda f: (lambda q, k, v: jnp.sum(f(q, k, v).astype(jnp.float32) * w))
    got = jax.grad(loss(lambda q, k, v: kernel(q, k, v, seg)),
                   argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss(lambda q, k, v: sa.attention_reference(
        dense, q, k, v, seg, is_mqa=mqa, attn_logits_soft_cap=case["cap"])),
        argnums=(0, 1, 2))(q, k, v)
    for name, g, r in zip("qkv", got, want):
      g, r = np.asarray(g, np.float32), np.asarray(r, np.float32)
      errs["d" + name] = float(np.abs(g - r).max() / max(np.abs(r).max(), 1e-6))
    ok = all(errs["d" + n] < 3e-2 for n in "qkv")
  return ok, errs


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--minutes", type=float, default=10.0)
  p.add_argument("--seed", type=int)
  p.add_argument("--jsonl", default=os.environ.get("FARM_RESULTS"))
  p.add_argument("--interpret", action="store_true",
                 help="run in the CPU interpreter (to test the fuzzer itself)")
  args = p.parse_args()
  interpret = None
  if args.interpret:
    from jax.experimental.pallas import mosaic_gpu as plgpu
    sys.path.insert(0, os.path.dirname(__file__))
    from lower_check import pretend_arch
    pretend_arch().__enter__()
    interpret = plgpu.InterpretGPUParams()
  deadline = time.time() + args.minutes * 60
  seeds = ([args.seed] if args.seed is not None
           else iter(lambda: random.SystemRandom().randrange(2**31), None))
  failures = 0
  for seed in seeds:
    case = draw(seed)
    t0 = time.time()
    try:
      ok, errs = run_case(case, interpret)
      rec = dict(kind="fuzz", ok=ok, errs=errs, case=case)
    except ValueError as e:  # budget / validation errors are not bugs
      rec = dict(kind="fuzz", ok=True, skipped=str(e)[:300], case=case)
    except Exception as e:  # noqa: BLE001
      rec = dict(kind="fuzz", ok=False, error=f"{type(e).__name__}: {e}"[:2000],
                 traceback=traceback.format_exc()[-4000:], case=case)
    rec.update(seconds=round(time.time() - t0, 1),
               gpu=os.environ.get("FARM_GPU"), time=time.time())
    failures += not rec["ok"]
    print(("PASS " if rec["ok"] else "FAIL ") + json.dumps(rec)[:500],
          flush=True)
    if args.jsonl:
      with open(args.jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if time.time() > deadline:
      break
    jax.clear_caches()  # bound compilation-cache memory across cases
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
