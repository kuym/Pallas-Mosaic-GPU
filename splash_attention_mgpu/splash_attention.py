# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Public API for Mosaic GPU splash attention, mirroring the TPU module.

  kernel = make_splash_mha(mask, block_sizes=BlockSizes())
  out = kernel(q, k, v)                     # q: [heads, q_len, head_dim]
  out = kernel(q, k, v, segment_ids)        # or with a leading batch dim

`q` is expected to be pre-scaled (the TPU kernel applies no softmax scale
either).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from . import backward as backward_lib
from . import forward_pingpong
from . import kernel as kernel_lib
from . import mask_info as mask_info_lib
from .kernel import DEFAULT_MASK_VALUE, BlockSizes, SegmentIds


# Automatic block_q: the two-tile kernel is used for head_dim 128 when its
# 256-row schedule visits at most this much more KV-block work than the
# 128-row one (measured on B200: +16-28% for full/causal/chunked masks,
# slower for narrow local windows and at head_dim 64).
AUTO_PINGPONG_MAX_EXTRA_WORK = 1.10


class SplashAttentionKernel:
  """Callable holding the processed mask.

  Differentiable with respect to q, k and v (dQ and dK/dV Mosaic GPU kernels,
  as in the TPU module).  With `save_residuals=True` the logsumexp is also
  returned, and that variant is forward-only.
  """

  def __init__(
      self,
      info: mask_info_lib.GpuMaskInfo,
      *,
      fwd_infos: dict[int, mask_info_lib.GpuMaskInfo],
      is_mqa: bool,
      block_sizes: BlockSizes,
      mask_value: float,
      attn_logits_soft_cap: float | None,
      interpret: Any,
  ):
    # `info` has 128x128 granularity (backward kernels, FLOP accounting);
    # `fwd_infos[block_q]` is the forward schedule for each candidate block_q.
    self.info = info
    self.fwd_infos = fwd_infos
    self.is_mqa = is_mqa
    self.requested_block_sizes = block_sizes
    to_dev = lambda x: None if x is None else jnp.asarray(x)
    bwd = dict(
        num_steps=to_dev(info.num_steps),
        q_block_order=to_dev(info.q_block_order),
        kv_block=to_dev(info.kv_block),
        block_kind=to_dev(info.block_kind),
        mask_block=to_dev(info.mask_block),
        partial_mask_blocks=to_dev(info.partial_mask_blocks),
        dkv_num_steps=to_dev(info.dkv_num_steps),
        dkv_kv_block_order=to_dev(info.dkv_kv_block_order),
        dkv_q_block=to_dev(info.dkv_q_block),
        dkv_block_kind=to_dev(info.dkv_block_kind),
        dkv_mask_block=to_dev(info.dkv_mask_block),
        partial_mask_blocks_t=to_dev(info.partial_mask_blocks_t),
    )
    self._variants = {}
    for bq, fwd in fwd_infos.items():
      static = _Static(
          mask_function=info.mask_function,
          block_sizes=dataclasses.replace(block_sizes, block_q=bq),
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          interpret=interpret,
      )
      schedule = _Schedule(
          fwd_num_steps=to_dev(fwd.num_steps),
          fwd_q_block_order=to_dev(fwd.q_block_order),
          fwd_kv_block=to_dev(fwd.kv_block),
          fwd_block_kind=to_dev(fwd.block_kind),
          fwd_mask_block=to_dev(fwd.mask_block),
          fwd_partial_mask_blocks=to_dev(fwd.partial_mask_blocks),
          **bwd,
      )
      self._variants[bq] = (static, schedule)
    # Extra KV-block work of the 256-row schedule relative to the 128-row one.
    self.pingpong_extra_work = None
    if 256 in fwd_infos and 128 in fwd_infos:
      self.pingpong_extra_work = (
          float(fwd_infos[256].num_steps.sum()) * 256
          / max(1.0, float(fwd_infos[128].num_steps.sum()) * 128))

  def choose_block_q(self, head_dim: int, head_dim_v: int, *,
                     itemsize: int = 2, has_segments: bool = False,
                     save_residuals: bool = True) -> int:
    requested = self.requested_block_sizes.block_q
    if requested is not None:
      return requested
    bs = self.requested_block_sizes
    if (256 in self._variants and head_dim == head_dim_v == 128
        and self.pingpong_extra_work is not None
        and self.pingpong_extra_work <= AUTO_PINGPONG_MAX_EXTRA_WORK
        and forward_pingpong.pingpong_fits(
            head_dim=head_dim, block_kv=bs.block_kv,
            num_stages=bs.num_stages, itemsize=itemsize,
            has_dense_mask=self.fwd_infos[256].partial_mask_blocks is not None,
            has_segments=has_segments, save_residuals=save_residuals)):
      return 256
    return 128

  @property
  def block_sizes(self) -> BlockSizes:
    return self.requested_block_sizes

  def __call__(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      segment_ids: SegmentIds | None = None,
      *,
      save_residuals: bool = False,
  ):
    static, schedule = self._variants[self.choose_block_q(
        q.shape[-1], v.shape[-1], itemsize=jnp.dtype(q.dtype).itemsize,
        has_segments=segment_ids is not None)]
    return _splash_attention(
        q, k, v, segment_ids, schedule,
        is_mqa=self.is_mqa, static=static, save_residuals=save_residuals,
    )


@dataclasses.dataclass(frozen=True)
class _Static:
  mask_function: Any
  block_sizes: BlockSizes
  mask_value: float
  attn_logits_soft_cap: float | None
  interpret: Any


class _Schedule(NamedTuple):
  fwd_num_steps: jax.Array
  fwd_q_block_order: jax.Array
  fwd_kv_block: jax.Array
  fwd_block_kind: jax.Array
  fwd_mask_block: jax.Array | None
  fwd_partial_mask_blocks: jax.Array | None
  num_steps: jax.Array
  q_block_order: jax.Array
  kv_block: jax.Array
  block_kind: jax.Array
  mask_block: jax.Array | None
  partial_mask_blocks: jax.Array | None
  dkv_num_steps: jax.Array
  dkv_kv_block_order: jax.Array
  dkv_q_block: jax.Array
  dkv_block_kind: jax.Array
  dkv_mask_block: jax.Array | None
  partial_mask_blocks_t: jax.Array | None


@functools.partial(
    jax.jit, static_argnames=("is_mqa", "static", "save_residuals")
)
def _splash_attention(
    q, k, v, segment_ids, schedule, *, is_mqa, static, save_residuals
):
  batched = q.ndim == 4
  if not batched:
    q = q[None]
  if is_mqa:
    # [(batch,) kv_len, head_dim] -> [batch, 1, kv_len, head_dim]
    k = k[..., None, :, :]
    v = v[..., None, :, :]
  if k.ndim == 3:
    k, v = k[None], v[None]
  if segment_ids is not None:
    seg_q, seg_kv = segment_ids
    if seg_q.ndim == 1:
      seg_q = jnp.broadcast_to(seg_q, (q.shape[0], seg_q.shape[0]))
    if seg_kv.ndim == 1:
      seg_kv = jnp.broadcast_to(seg_kv, (q.shape[0], seg_kv.shape[0]))
    segment_ids = SegmentIds(seg_q.astype(jnp.int32), seg_kv.astype(jnp.int32))
  if save_residuals:
    out, lse = _forward(static, q, k, v, segment_ids, schedule,
                        save_residuals=True)
  else:
    out, lse = _attention(static, q, k, v, segment_ids, schedule), None
  if not batched:
    out = out[0]
    lse = None if lse is None else lse[0]
  return (out, (lse,)) if save_residuals else out


def _forward(static, q, k, v, segment_ids, schedule, *, save_residuals):
  bs = static.block_sizes
  sched = (schedule.fwd_num_steps, schedule.fwd_q_block_order,
           schedule.fwd_kv_block, schedule.fwd_block_kind,
           schedule.fwd_mask_block, schedule.fwd_partial_mask_blocks)
  if bs.block_q == forward_pingpong.CTA_ROWS:
    return forward_pingpong.splash_attention_forward_pingpong(
        q, k, v, segment_ids, *sched,
        mask_function=static.mask_function,
        block_kv=bs.block_kv,
        num_stages=bs.num_stages,
        mask_value=static.mask_value,
        attn_logits_soft_cap=static.attn_logits_soft_cap,
        save_residuals=save_residuals,
        interpret=static.interpret,
    )
  return kernel_lib._splash_attention_forward(
      q, k, v, segment_ids, *sched,
      mask_function=static.mask_function,
      block_sizes=static.block_sizes,
      mask_value=static.mask_value,
      attn_logits_soft_cap=static.attn_logits_soft_cap,
      save_residuals=save_residuals,
      interpret=static.interpret,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _attention(static, q, k, v, segment_ids, schedule):
  return _forward(static, q, k, v, segment_ids, schedule,
                  save_residuals=False)


def _attention_fwd(static, q, k, v, segment_ids, schedule):
  out, lse = _forward(static, q, k, v, segment_ids, schedule,
                      save_residuals=True)
  return out, (q, k, v, segment_ids, schedule, out, lse)


def _attention_bwd(static, residuals, do):
  q, k, v, segment_ids, schedule, out, lse = residuals
  bs = static.block_sizes
  delta = jnp.sum(do.astype(jnp.float32) * out.astype(jnp.float32), axis=-1)
  do = do.astype(q.dtype)
  common = dict(
      mask_function=static.mask_function,
      block_kv=128,  # the backward schedule is always 128x128
      num_stages=bs.num_stages_bwd,
      mask_value=static.mask_value,
      attn_logits_soft_cap=static.attn_logits_soft_cap,
      interpret=static.interpret,
  )
  dq = backward_lib.splash_attention_bwd_dq(
      q, k, v, do, lse, delta, segment_ids,
      schedule.num_steps, schedule.q_block_order, schedule.kv_block,
      schedule.block_kind, schedule.mask_block, schedule.partial_mask_blocks,
      block_kv_compute=bs.block_kv_dq, **common,
  )
  dk, dv = backward_lib.splash_attention_bwd_dkv(
      q, k, v, do, lse, delta, segment_ids,
      schedule.dkv_num_steps, schedule.dkv_kv_block_order,
      schedule.dkv_q_block, schedule.dkv_block_kind,
      schedule.dkv_mask_block, schedule.partial_mask_blocks_t,
      block_q_compute=bs.block_q_dkv, **common,
  )
  return dq, dk, dv, None, None


_attention.defvjp(_attention_fwd, _attention_bwd)


def _make_splash_attention(
    mask: np.ndarray | mask_lib.MultiHeadMask | mask_lib.Mask,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    interpret: Any = None,
) -> SplashAttentionKernel:
  if block_sizes is None:
    block_sizes = BlockSizes()
  if isinstance(mask, np.ndarray):
    if mask.ndim == 2:
      mask = mask[None]
    mask = mask_lib.MultiHeadMask(
        [mask_lib.NumpyMask(m) for m in mask]
    )
  # The backward kernels (and FLOP accounting) always use a 128x128 schedule;
  # the forward kernel uses (block_q, block_kv) for each candidate block_q.
  info = mask_info_lib.process_mask(mask, (128, 128))
  q_len = mask.shape[1]
  if block_sizes.block_q is not None:
    candidates = [block_sizes.block_q]
  else:
    candidates = [128] + ([256] if q_len % 256 == 0 else [])
  fwd_infos = {}
  for bq in candidates:
    if (bq, block_sizes.block_kv) == (128, 128):
      fwd_infos[bq] = info
    else:
      fwd_infos[bq] = mask_info_lib.process_mask(mask, (bq, block_sizes.block_kv))
  return SplashAttentionKernel(
      info,
      fwd_infos=fwd_infos,
      is_mqa=is_mqa,
      block_sizes=block_sizes,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
      interpret=interpret,
  )


make_splash_mha = functools.partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = functools.partial(_make_splash_attention, is_mqa=True)


def attention_reference(
    mask: jax.Array,  # bool[heads or 1, q_len, kv_len]
    q: jax.Array,  # [(batch,) heads, q_len, head_dim]
    k: jax.Array,  # [(batch,) kv_heads, kv_len, head_dim] (MQA: no head dim)
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    *,
    is_mqa: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    save_residuals: bool = False,
):
  """Dense f32 reference with the TPU kernel's masking semantics."""
  batched = q.ndim == 4
  if not batched:
    q = q[None]
    k = k[None]
    v = v[None]
  if is_mqa:
    k, v = k[:, None], v[:, None]
  q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
  rep = q.shape[1] // k.shape[1]
  k = jnp.repeat(k, rep, axis=1)
  v = jnp.repeat(v, rep, axis=1)
  logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)
  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap
  full_mask = jnp.broadcast_to(mask[None], logits.shape)
  if segment_ids is not None:
    seg_q, seg_kv = segment_ids
    seg_q = jnp.broadcast_to(seg_q, (q.shape[0], q.shape[2]))
    seg_kv = jnp.broadcast_to(seg_kv, (q.shape[0], k.shape[2]))
    full_mask = full_mask & (seg_q[:, None, :, None] == seg_kv[:, None, None, :])
  logits = jnp.where(full_mask, logits, mask_value)
  m = logits.max(axis=-1, keepdims=True)
  p = jnp.exp(logits - m)
  l = p.sum(axis=-1, keepdims=True)
  out = jnp.einsum("bhqk,bhkd->bhqd", p / l, v)
  lse = (jnp.log(l) + m)[..., 0]
  if not batched:
    out, lse = out[0], lse[0]
  return (out, (lse,)) if save_residuals else out
