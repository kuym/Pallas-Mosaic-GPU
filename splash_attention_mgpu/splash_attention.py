# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Public API for Mosaic GPU splash attention, mirroring the TPU module.

  kernel = make_splash_mha(mask, block_sizes=BlockSizes())
  out = kernel(q, k, v)                     # q: [heads, q_len, head_dim]
  out = kernel(q, k, v, segment_ids)        # or with a leading batch dim

`q` is expected to be pre-scaled (the TPU kernel applies no softmax scale
either).
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from . import kernel as kernel_lib
from . import mask_info as mask_info_lib
from .kernel import DEFAULT_MASK_VALUE, BlockSizes, SegmentIds


class SplashAttentionKernel:
  """Callable holding the processed mask. Forward pass only."""

  def __init__(
      self,
      info: mask_info_lib.GpuMaskInfo,
      *,
      is_mqa: bool,
      block_sizes: BlockSizes,
      mask_value: float,
      attn_logits_soft_cap: float | None,
      interpret: Any,
  ):
    self.info = info
    self.is_mqa = is_mqa
    self.block_sizes = block_sizes
    self.mask_value = mask_value
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.interpret = interpret
    to_dev = lambda x: None if x is None else jnp.asarray(x)
    self._num_steps = to_dev(info.num_steps)
    self._kv_block = to_dev(info.kv_block)
    self._block_kind = to_dev(info.block_kind)
    self._mask_block = to_dev(info.mask_block)
    self._partial_mask_blocks = to_dev(info.partial_mask_blocks)

  def __call__(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      segment_ids: SegmentIds | None = None,
      *,
      save_residuals: bool = False,
  ):
    return _splash_attention(
        q, k, v, segment_ids,
        self._num_steps, self._kv_block, self._block_kind, self._mask_block,
        self._partial_mask_blocks,
        is_mqa=self.is_mqa,
        mask_function=self.info.mask_function,
        block_sizes=self.block_sizes,
        mask_value=self.mask_value,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        save_residuals=save_residuals,
        interpret=self.interpret,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "is_mqa", "mask_function", "block_sizes", "mask_value",
        "attn_logits_soft_cap", "save_residuals", "interpret",
    ),
)
def _splash_attention(
    q, k, v, segment_ids, num_steps, kv_block, block_kind, mask_block,
    partial_mask_blocks, *, is_mqa, mask_function, block_sizes, mask_value,
    attn_logits_soft_cap, save_residuals, interpret,
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
  result = kernel_lib._splash_attention_forward(
      q, k, v, segment_ids, num_steps, kv_block, block_kind, mask_block,
      partial_mask_blocks,
      mask_function=mask_function,
      block_sizes=block_sizes,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
      save_residuals=save_residuals,
      interpret=interpret,
  )
  if save_residuals:
    out, lse = result
  else:
    out, lse = result, None
  if not batched:
    out = out[0]
    lse = None if lse is None else lse[0]
  return (out, (lse,)) if save_residuals else out


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
  info = mask_info_lib.process_mask(
      mask, (block_sizes.block_q, block_sizes.block_kv)
  )
  return SplashAttentionKernel(
      info,
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
