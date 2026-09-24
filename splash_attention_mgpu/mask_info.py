# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Sparse block schedules for the Mosaic GPU splash attention kernel.

The TPU kernel walks a dense (head, q_block, kv_block) grid and uses the
scalar-prefetched `data_next`/`block_mask` arrays to skip empty blocks.  On GPU
each CTA owns one (head, q_block) row and loops over KV blocks itself, so we
compact every row of the block mask into an explicit list of the KV blocks that
must be visited:

  num_steps[h, i]       number of non-empty KV blocks in row i
  kv_block[h, i, s]     index of the s-th non-empty KV block
  block_kind[h, i, s]   PARTIAL (needs masking) or FULL (every entry visible)
  mask_block[h, i, s]   index into `partial_mask_blocks` (dense masks only)

We reuse the upstream TPU mask library (`splash_attention_mask`) and its block
classification (`process_mask`) so that masks behave identically on both
backends.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as tpu_mask_info,
)

EMPTY = 0
PARTIAL = 1
FULL = 2

MaskFunction = Callable[[jax.Array, jax.Array], jax.Array]


@dataclasses.dataclass(frozen=True)
class GpuMaskInfo:
  """Per-row sparse KV schedule.

  The leading dimension of the schedule arrays is either 1 (all heads share
  one mask) or `num_heads`.
  """

  num_steps: np.ndarray  # i32[mask_heads, q_blocks]
  kv_block: np.ndarray  # i32[mask_heads, q_blocks, max_steps]
  block_kind: np.ndarray  # i32[mask_heads, q_blocks, max_steps]
  mask_block: np.ndarray | None  # i32[mask_heads, q_blocks, max_steps]
  # int8[num_partial_blocks, block_q, block_kv], 1 = attend.  None when the
  # mask is computed inside the kernel with `mask_function`.
  partial_mask_blocks: np.ndarray | None
  mask_function: MaskFunction | None
  block_q: int
  block_kv: int
  num_kv_blocks: int

  @property
  def max_steps(self) -> int:
    return self.kv_block.shape[-1]

  @property
  def density(self) -> float:
    """Fraction of KV blocks visited compared to dense attention."""
    return float(self.num_steps.mean()) / self.num_kv_blocks


def process_mask(
    mask: mask_lib.MultiHeadMask | mask_lib.Mask,
    block_shape: tuple[int, int],
    num_heads: int | None = None,
) -> GpuMaskInfo:
  """Builds the sparse schedule for a (multi-head) mask."""
  if not isinstance(mask, mask_lib.MultiHeadMask):
    if num_heads is None:
      num_heads = 1
    mask = mask_lib.MultiHeadMask([mask] * num_heads)
  block_q, block_kv = block_shape
  info, mask_function = tpu_mask_info.process_mask(
      mask, block_shape, shrink_grid=False, downcast_smem_data=False
  )
  block_mask = np.asarray(info.block_mask, dtype=np.int32)
  mask_next = (
      None if info.mask_next is None else np.asarray(info.mask_next, np.int32)
  )
  mask_heads, q_blocks, kv_blocks = block_mask.shape

  num_steps = (block_mask != EMPTY).sum(axis=-1).astype(np.int32)
  # Keep at least one (unused) column so every array has a non-empty shape.
  max_steps = max(1, int(num_steps.max()))
  kv_block = np.zeros((mask_heads, q_blocks, max_steps), np.int32)
  block_kind = np.zeros((mask_heads, q_blocks, max_steps), np.int32)
  mask_block = (
      None
      if mask_next is None
      else np.zeros((mask_heads, q_blocks, max_steps), np.int32)
  )
  for h in range(mask_heads):
    for i in range(q_blocks):
      (cols,) = np.nonzero(block_mask[h, i])
      n = len(cols)
      kv_block[h, i, :n] = cols
      block_kind[h, i, :n] = block_mask[h, i, cols]
      if mask_block is not None:
        mask_block[h, i, :n] = mask_next[h, i, cols]
      if n:
        # Pad by repeating the last block; padding is never visited.
        kv_block[h, i, n:] = cols[-1]

  partial_mask_blocks = None
  if mask_function is None and info.partial_mask_blocks is not None:
    partial_mask_blocks = np.asarray(info.partial_mask_blocks).astype(np.int8)
  if mask_function is None and partial_mask_blocks is None:
    # No partial blocks at all: nothing ever needs masking.
    assert not (block_kind == PARTIAL).any()

  return GpuMaskInfo(
      num_steps=num_steps,
      kv_block=kv_block,
      block_kind=block_kind,
      mask_block=mask_block if partial_mask_blocks is not None else None,
      partial_mask_blocks=partial_mask_blocks,
      mask_function=mask_function,
      block_q=block_q,
      block_kv=block_kv,
      num_kv_blocks=kv_blocks,
  )
