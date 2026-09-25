# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Tests for the Mosaic GPU splash attention kernel.

Three tiers, picked automatically:

* On a Blackwell GPU (compute capability 10.x) the kernel is compiled and run
  natively and compared against a dense f32 reference.
* Anywhere, the kernel runs in the Mosaic GPU interpreter on CPU (numerics +
  data race detection for the warp-specialized pipeline).
* Anywhere, the kernel is lowered to Mosaic GPU MLIR for sm_100, which runs
  layout inference and the Mosaic GPU dialect lowering.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from lower_check import lower_for_blackwell, pretend_arch  # noqa: E402

import splash_attention_mgpu as sa  # noqa: E402
from splash_attention_mgpu import mask_info as mask_info_lib  # noqa: E402


def _on_blackwell() -> bool:
  try:
    dev = jax.devices("gpu")[0]
  except RuntimeError:
    return False
  return getattr(dev, "compute_capability", "").startswith("10.")


def _reset_interpreter():
  # Interpreter state is set up while tracing, so drop cached executables too.
  jax.clear_caches()
  interpret_pallas_call.gpu_callbacks.reset_gpu_interpret_mode_state()


ON_BLACKWELL = _on_blackwell()
needs_blackwell = pytest.mark.skipif(
    not ON_BLACKWELL, reason="needs a Blackwell (sm_100) GPU"
)


def _masks(s, seed=0):
  rng = np.random.default_rng(seed)
  dense = np.tril(np.ones((s, s), bool)) & (rng.random((s, s)) > 0.3)
  dense |= np.eye(s, dtype=bool)  # every row attends to something
  return {
      "full": sa.FullMask((s, s)),
      "causal": sa.CausalMask((s, s)),
      "local": sa.LocalMask((s, s), (100, 20), 0),
      "chunked": sa.ChunkedCausalMask((s, s), chunk_size=192),
      "dense": sa.NumpyMask(dense),
  }


def _inputs(b, h, kvh, s, d, dv, dtype, mqa, seed=0):
  ks = jax.random.split(jax.random.key(seed), 3)
  lead = (b,) if b else ()
  q = jax.random.normal(ks[0], lead + (h, s, d), dtype)
  kv_lead = lead + (() if mqa else (kvh,))
  k = jax.random.normal(ks[1], kv_lead + (s, d), dtype)
  v = jax.random.normal(ks[2], kv_lead + (s, dv), dtype)
  return q, k, v


def _check(
    mask_name, *, s=512, b=None, h=2, kvh=None, d=64, dv=None, dtype=jnp.bfloat16,
    mqa=False, segments=False, cap=None, block_sizes=sa.BlockSizes(),
    interpret=None, multi_head_mask=False,
):
  dv = dv or d
  kvh = kvh or h
  if multi_head_mask:
    names = list(_masks(s))
    heads = [_masks(s)[names[(names.index(mask_name) + i) % len(names)]]
             for i in range(h)]
    mask = sa.MultiHeadMask(heads)
    dense = np.stack([np.asarray(m[:, :]) for m in heads])
  else:
    mask = _masks(s)[mask_name]
    dense = np.asarray(mask[:, :])[None]
  make = sa.make_splash_mqa if mqa else sa.make_splash_mha
  kernel = make(
      mask, block_sizes=block_sizes, attn_logits_soft_cap=cap,
      interpret=interpret,
  )
  device = jax.devices("cpu")[0] if interpret is not None else None
  with jax.default_device(device):
    _check_on_device(kernel, dense, s=s, b=b, h=h, kvh=kvh, d=d, dv=dv,
                     dtype=dtype, mqa=mqa, segments=segments, cap=cap,
                     interpret=interpret)


def _check_on_device(kernel, dense, *, s, b, h, kvh, d, dv, dtype, mqa,
                     segments, cap, interpret):
  q, k, v = _inputs(b, h, kvh, s, d, dv, dtype, mqa)
  seg = None
  if segments:
    ids = jnp.asarray(np.repeat(np.arange(4), s // 4), jnp.int32)
    seg = sa.SegmentIds(ids, ids)
  if interpret is not None:
    _reset_interpreter()
    with pretend_arch():
      out, (lse,) = kernel(q, k, v, seg, save_residuals=True)
    assert not interpret_pallas_call.get_races().races_found
  else:
    out, (lse,) = kernel(q, k, v, seg, save_residuals=True)
  ref, (ref_lse,) = sa.attention_reference(
      jnp.asarray(dense), q, k, v, seg, is_mqa=mqa,
      attn_logits_soft_cap=cap, save_residuals=True,
  )
  tol = 2e-2 if dtype == jnp.bfloat16 else 5e-3
  np.testing.assert_allclose(
      np.asarray(out, np.float32), np.asarray(ref), atol=tol, rtol=tol
  )
  np.testing.assert_allclose(np.asarray(lse), np.asarray(ref_lse), atol=1e-3,
                             rtol=1e-3)


# ---------------------------------------------------------------------------
# Mask preprocessing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["full", "causal", "local", "chunked", "dense"])
@pytest.mark.parametrize("bkv", [64, 128])
def test_mask_info_reconstructs_mask(name, bkv):
  s = 512
  mask = _masks(s)[name]
  info = mask_info_lib.process_mask(mask, (128, bkv))
  dense = np.asarray(mask[:, :])
  rebuilt = np.zeros_like(dense)
  q_pos = np.arange(128)[:, None]
  for i in range(s // 128):
    for step in range(info.num_steps[0, i]):
      j = info.kv_block[0, i, step]
      rows, cols = slice(i * 128, (i + 1) * 128), slice(j * bkv, (j + 1) * bkv)
      kind = info.block_kind[0, i, step]
      if kind == mask_info_lib.FULL:
        rebuilt[rows, cols] = True
      elif info.partial_mask_blocks is not None:
        rebuilt[rows, cols] = info.partial_mask_blocks[
            info.mask_block[0, i, step]].astype(bool)
      else:
        kv_pos = np.arange(bkv)[None, :]
        rebuilt[rows, cols] = np.asarray(info.mask_function(
            i * 128 + q_pos, j * bkv + kv_pos))
  np.testing.assert_array_equal(rebuilt, dense)
  # Launch orders are permutations, heaviest rows / columns first.
  for order, steps in ((info.q_block_order, info.num_steps),
                       (info.dkv_kv_block_order, info.dkv_num_steps)):
    np.testing.assert_array_equal(np.sort(order[0]), np.arange(len(order[0])))
    assert np.all(np.diff(steps[0][order[0]]) <= 0)


def test_mask_info_is_sparse():
  info = mask_info_lib.process_mask(sa.LocalMask((4096, 4096), (256, 0), 0),
                                    (128, 128))
  assert info.max_steps == 3
  assert info.density < 0.1


# ---------------------------------------------------------------------------
# CPU interpreter: numerics and race detection
# ---------------------------------------------------------------------------

INTERPRET = plgpu.InterpretGPUParams(detect_races=True)


@pytest.mark.parametrize("name", ["full", "causal", "local", "chunked", "dense"])
def test_interpret_masks(name):
  _check(name, interpret=INTERPRET)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(segments=True),
        dict(segments=True, mask_name="dense"),
        dict(cap=5.0),
        dict(mqa=True),
        dict(h=4, kvh=2),
        dict(b=2, segments=True),
        dict(d=128),
        dict(d=128, dv=64),
        dict(d=256, block_sizes=sa.BlockSizes(block_kv=64)),
        dict(block_sizes=sa.BlockSizes(block_kv=64), s=1024),
        dict(block_sizes=sa.BlockSizes(num_stages=3), mask_name="dense"),
        dict(multi_head_mask=True, h=3),
        dict(dtype=jnp.float16),
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()),
)
def test_interpret_features(kwargs):
  kwargs = dict(kwargs)
  _check(kwargs.pop("mask_name", "causal"), interpret=INTERPRET, **kwargs)


def test_interpret_empty_rows_are_zero():
  # The first q block sees nothing: its output is zero and lse = mask_value.
  s = 256
  dense = np.zeros((s, s), bool)
  dense[128:, :] = True
  kernel = sa.make_splash_mha(sa.NumpyMask(dense), interpret=INTERPRET)
  q, k, v = _inputs(None, 1, 1, s, 64, 64, jnp.bfloat16, False)
  _reset_interpreter()
  with pretend_arch(), jax.default_device(jax.devices("cpu")[0]):
    out, (lse,) = kernel(q, k, v, save_residuals=True)
  assert not interpret_pallas_call.get_races().races_found
  np.testing.assert_array_equal(np.asarray(out[:, :128], np.float32), 0.0)
  assert np.all(np.asarray(lse[:, :128]) == np.float32(sa.DEFAULT_MASK_VALUE))
  ref = sa.attention_reference(jnp.asarray(dense)[None], q, k, v)
  np.testing.assert_allclose(np.asarray(out[:, 128:], np.float32),
                             np.asarray(ref[:, 128:]), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Lowering for sm_100 (runs without a GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(mask_name="dense", segments=True),
        dict(mask_name="local", cap=30.0),
        dict(block_sizes=sa.BlockSizes(num_stages=3)),
        dict(d=128, block_sizes=sa.BlockSizes(block_kv=64, num_stages=4)),
        dict(d=256, block_sizes=sa.BlockSizes(block_kv=64)),
        dict(mqa=True, b=2),
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "default",
)
def test_lowers_for_sm100(kwargs):
  kwargs = dict(kwargs)
  mask = _masks(512)[kwargs.pop("mask_name", "causal")]
  make = sa.make_splash_mqa if kwargs.get("mqa") else sa.make_splash_mha
  kernel = make(mask, block_sizes=kwargs.get("block_sizes", sa.BlockSizes()),
                attn_logits_soft_cap=kwargs.get("cap"))
  d = kwargs.get("d", 64)
  b = kwargs.get("b")
  q, k, v = jax.eval_shape(
      lambda: _inputs(b, 4, 4, 512, d, d, jnp.bfloat16, kwargs.get("mqa")))
  seg = None
  if kwargs.get("segments"):
    ids = jax.ShapeDtypeStruct((512,), jnp.int32)
    seg = sa.SegmentIds(ids, ids)
  text = lower_for_blackwell(
      lambda *a: kernel(*a, save_residuals=True), q, k, v, seg)
  assert "mosaic_gpu" in text


def test_rejects_invalid_block_sizes():
  with pytest.raises(ValueError, match="num_stages"):
    sa.BlockSizes(num_stages=1)
  with pytest.raises(ValueError, match="block_kv"):
    sa.BlockSizes(block_kv=256)
  with pytest.raises(ValueError, match="block_q_dkv"):
    sa.BlockSizes(block_q_dkv=32)  # tcgen05 contraction dim must be >= 64


def test_rejects_oversized_configs():
  kernel = sa.make_splash_mha(sa.CausalMask((512, 512)))
  q, k, v = jax.eval_shape(
      lambda: _inputs(None, 1, 1, 512, 256, 256, jnp.bfloat16, False))
  with pytest.raises(ValueError, match="TMEM budget"):
    lower_for_blackwell(kernel, q, k, v)
  kernel = sa.make_splash_mha(sa.CausalMask((512, 512)),
                              block_sizes=sa.BlockSizes(num_stages=3))
  q, k, v = jax.eval_shape(
      lambda: _inputs(None, 1, 1, 512, 128, 128, jnp.bfloat16, False))
  with pytest.raises(ValueError, match="Shared memory budget"):
    lower_for_blackwell(kernel, q, k, v)


# ---------------------------------------------------------------------------
# Native execution on B200
# ---------------------------------------------------------------------------


@needs_blackwell
@pytest.mark.parametrize("name", ["full", "causal", "local", "chunked", "dense"])
@pytest.mark.parametrize("d", [64, 128])
def test_gpu_masks(name, d):
  _check(name, s=2048, h=4, d=d)


@needs_blackwell
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(segments=True),
        dict(segments=True, mask_name="dense"),
        dict(cap=5.0),
        dict(mqa=True, b=2),
        dict(h=8, kvh=2),
        dict(d=256, block_sizes=sa.BlockSizes(block_kv=64)),
        dict(block_sizes=sa.BlockSizes(num_stages=3)),
        dict(d=128, block_sizes=sa.BlockSizes(block_kv=64, num_stages=4)),
        dict(multi_head_mask=True, h=5),
        dict(dtype=jnp.float16),
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()),
)
def test_gpu_features(kwargs):
  kwargs = dict(kwargs)
  _check(kwargs.pop("mask_name", "causal"), s=2048, **kwargs)


@pytest.mark.parametrize("growth", [0.0, 0.05, 40.0])
def test_interpret_lazy_rescaling(growth):
  # Logit maxima that grow along the KV axis force O to be rescaled on later
  # blocks (growth=40); tiny growth stays under the threshold and exercises
  # the stale-max path; zero growth never rescales after the first block.
  s, d = 1024, 64
  kernel = sa.make_splash_mha(sa.CausalMask((s, s)), interpret=INTERPRET)
  cpu = jax.devices("cpu")[0]
  with jax.default_device(cpu):
    q, k, v = _inputs(None, 2, 2, s, d, d, jnp.bfloat16, False)
    pos = jnp.arange(s, dtype=jnp.float32)[None, :, None] / s
    k = (k.astype(jnp.float32) * 0.3 + growth * pos).astype(jnp.bfloat16)
    q = (jnp.abs(q.astype(jnp.float32)) * 0.3).astype(jnp.bfloat16)
    _reset_interpreter()
    with pretend_arch():
      out, (lse,) = kernel(q, k, v, save_residuals=True)
    assert not interpret_pallas_call.get_races().races_found
    dense = jnp.asarray(np.tril(np.ones((s, s), bool)))[None]
    ref, (ref_lse,) = sa.attention_reference(dense, q, k, v,
                                             save_residuals=True)
  np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(ref),
                             atol=2e-2, rtol=2e-2)
  np.testing.assert_allclose(np.asarray(lse), np.asarray(ref_lse), rtol=1e-3,
                             atol=1e-3)


# ---------------------------------------------------------------------------
# Backward pass (dQ and dK/dV kernels)
# ---------------------------------------------------------------------------


def _check_grads(
    mask_name, *, s=512, b=None, h=2, kvh=None, d=64, mqa=False,
    segments=False, cap=None, block_sizes=sa.BlockSizes(), interpret=None,
    multi_head_mask=False, dtype=jnp.bfloat16,
):
  kvh = kvh or h
  if multi_head_mask:
    names = list(_masks(s))
    heads = [_masks(s)[names[(names.index(mask_name) + i) % len(names)]]
             for i in range(h)]
    mask = sa.MultiHeadMask(heads)
    dense = np.stack([np.asarray(m[:, :]) for m in heads])
  else:
    mask = _masks(s)[mask_name]
    dense = np.asarray(mask[:, :])[None]
  make = sa.make_splash_mqa if mqa else sa.make_splash_mha
  kernel = make(mask, block_sizes=block_sizes, attn_logits_soft_cap=cap,
                interpret=interpret)
  device = jax.devices("cpu")[0] if interpret is not None else None
  with jax.default_device(device):
    q, k, v = _inputs(b, h, kvh, s, d, d, dtype, mqa)
    w = jax.random.normal(jax.random.key(7), q.shape, jnp.float32)
    seg = None
    if segments:
      ids = jnp.asarray(np.repeat(np.arange(4), s // 4), jnp.int32)
      seg = sa.SegmentIds(ids, ids)

    def loss(f):
      return lambda q, k, v: jnp.sum(f(q, k, v).astype(jnp.float32) * w)

    grad = lambda f: jax.grad(loss(f), argnums=(0, 1, 2))
    if interpret is not None:
      _reset_interpreter()
      with pretend_arch():
        got = grad(lambda q, k, v: kernel(q, k, v, seg))(q, k, v)
      assert not interpret_pallas_call.get_races().races_found
    else:
      got = grad(lambda q, k, v: kernel(q, k, v, seg))(q, k, v)
    want = grad(lambda q, k, v: sa.attention_reference(
        jnp.asarray(dense), q, k, v, seg, is_mqa=mqa,
        attn_logits_soft_cap=cap))(q, k, v)
  for name, g, r in zip("qkv", got, want):
    g, r = np.asarray(g, np.float32), np.asarray(r, np.float32)
    rel = np.abs(g - r).max() / np.abs(r).max()
    assert rel < 2e-2, f"d{name}: relative error {rel}"


GRAD_CASES = [
    dict(mask_name="full"),
    dict(mask_name="causal"),
    dict(mask_name="local", segments=True),
    dict(mask_name="chunked"),
    dict(mask_name="dense"),
    dict(mask_name="dense", segments=True),
    dict(mask_name="causal", cap=5.0),
    dict(mask_name="causal", h=4, kvh=2),
    dict(mask_name="causal", mqa=True, h=3),
    dict(mask_name="causal", b=2),
    dict(mask_name="causal", d=128),
    dict(mask_name="causal", dtype=jnp.float16),
    dict(mask_name="causal", multi_head_mask=True, h=3),
    dict(mask_name="causal",
         block_sizes=sa.BlockSizes(block_kv_dq=128, block_q_dkv=64,
                                   num_stages_bwd=1)),
    dict(mask_name="dense",
         block_sizes=sa.BlockSizes(block_kv_dq=64, block_q_dkv=128,
                                   num_stages_bwd=3)),
]
_case_id = lambda kw: ",".join(f"{k}={v}" for k, v in kw.items())


@pytest.mark.parametrize("kwargs", GRAD_CASES, ids=_case_id)
def test_interpret_grads(kwargs):
  kwargs = dict(kwargs)
  _check_grads(kwargs.pop("mask_name"), interpret=INTERPRET, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(segments=True, mask_name="dense", d=128),
     dict(mask_name="local", cap=30.0, mqa=True)],
    ids=lambda kw: _case_id(kw) or "default",
)
def test_grads_lower_for_sm100(kwargs):
  kwargs = dict(kwargs)
  mask = _masks(512)[kwargs.pop("mask_name", "causal")]
  mqa = kwargs.get("mqa", False)
  kernel = (sa.make_splash_mqa if mqa else sa.make_splash_mha)(
      mask, attn_logits_soft_cap=kwargs.get("cap"))
  d = kwargs.get("d", 64)
  q, k, v = jax.eval_shape(
      lambda: _inputs(None, 4, 2, 512, d, d, jnp.bfloat16, mqa))
  seg = None
  if kwargs.get("segments"):
    ids = jax.ShapeDtypeStruct((512,), jnp.int32)
    seg = sa.SegmentIds(ids, ids)
  f = jax.grad(
      lambda q, k, v, seg: kernel(q, k, v, seg).astype(jnp.float32).sum(),
      argnums=(0, 1, 2))
  text = lower_for_blackwell(f, q, k, v, seg)
  assert text.count("mosaic_gpu") >= 3  # forward, dQ and dK/dV kernels


@needs_blackwell
@pytest.mark.parametrize("kwargs", GRAD_CASES, ids=_case_id)
def test_gpu_grads(kwargs):
  kwargs = dict(kwargs)
  _check_grads(kwargs.pop("mask_name"), s=2048, **kwargs)


# ---------------------------------------------------------------------------
# Two-tile ping-pong forward kernel (block_q=256)
# ---------------------------------------------------------------------------

PINGPONG = sa.BlockSizes(block_q=256)
PINGPONG_CASES = [
    dict(mask_name="full"),
    dict(mask_name="causal"),
    dict(mask_name="local"),
    dict(mask_name="chunked"),
    dict(mask_name="dense", block_sizes=sa.BlockSizes(block_q=256, block_kv=64)),
    dict(mask_name="causal", segments=True),
    dict(mask_name="dense", segments=True,
         block_sizes=sa.BlockSizes(block_q=256, block_kv=64)),
    dict(mask_name="causal", cap=5.0),
    dict(mask_name="causal", h=4, kvh=2),
    dict(mask_name="full", mqa=True, b=2),
    dict(mask_name="causal", d=128),
    dict(mask_name="local", d=128, s=1024),
    dict(mask_name="causal", multi_head_mask=True, h=3),
    dict(mask_name="causal", dtype=jnp.float16),
    dict(mask_name="causal", block_sizes=sa.BlockSizes(block_q=256, num_stages=3)),
]


@pytest.mark.parametrize("kwargs", PINGPONG_CASES, ids=_case_id)
def test_interpret_pingpong(kwargs):
  kwargs = dict(kwargs)
  kwargs.setdefault("block_sizes", PINGPONG)
  _check(kwargs.pop("mask_name"), interpret=INTERPRET, **kwargs)


@pytest.mark.parametrize("kwargs", PINGPONG_CASES[:4], ids=_case_id)
def test_interpret_pingpong_grads(kwargs):
  kwargs = dict(kwargs)
  kwargs.setdefault("block_sizes", PINGPONG)
  _check_grads(kwargs.pop("mask_name"), interpret=INTERPRET, **kwargs)


@needs_blackwell
@pytest.mark.parametrize("kwargs", PINGPONG_CASES, ids=_case_id)
def test_gpu_pingpong(kwargs):
  kwargs = dict(kwargs)
  kwargs.setdefault("block_sizes", PINGPONG)
  kwargs.setdefault("s", 2048)
  _check(kwargs.pop("mask_name"), **kwargs)


@needs_blackwell
@pytest.mark.parametrize("kwargs", PINGPONG_CASES[:4], ids=_case_id)
def test_gpu_pingpong_grads(kwargs):
  kwargs = dict(kwargs)
  kwargs.setdefault("block_sizes", PINGPONG)
  _check_grads(kwargs.pop("mask_name"), s=2048, **kwargs)
