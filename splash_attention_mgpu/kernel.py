# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Splash (block-sparse flash) attention for Blackwell in Pallas Mosaic GPU.

This is a port of the TPU splash attention forward kernel
(jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
to the Mosaic GPU dialect of Pallas, targeting sm_100 (B200).

Structure (one CTA per (q_block, head, batch)):

  warpgroup 1, warp 0  TMA producer: Q once, then K/V (+ mask block, KV
                       segment ids) for every non-empty KV block of the row,
                       through a `num_stages`-deep SMEM ring.
  warpgroup 1, warp 1  tcgen05 issuer: S = Q K^T into one of two TMEM S
                       buffers (so QK of step s+1 overlaps softmax of step s),
                       then O += P V with P read directly from TMEM.
  warpgroup 0          softmax: loads S from TMEM, applies soft-cap, masks
                       (partial mask blocks, in-kernel mask functions, segment
                       ids), does the online softmax in the log2 domain,
                       rescales O in TMEM, writes bf16/f16 P to TMEM, and
                       finally normalizes O and stores out (and logsumexp).

Sparsity follows the TPU kernel: `mask_info.process_mask` classifies every
(block_q, block_kv) block of the mask as empty / partial / full.  Empty blocks
are never loaded or computed, full blocks skip the masking arithmetic, and
partial blocks are masked with either the stored dense block or the mask
function (causal, local, chunked-causal, ...).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from . import mask_info as mask_info_lib

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
# Rows only rescale their running max (and O) when it grows by more than this
# many powers of two; see the softmax loop.
RESCALE_THRESHOLD = 8.0

# The softmax warpgroup handles 128 query rows, one TMEM lane per row.
BLOCK_Q = 128
TMEM_COLS = 512
# sm_100 allows 227 KiB of dynamic shared memory per block; leave room for
# barriers and the 1 KiB alignment of swizzled buffers.
SMEM_BYTES = 227 * 1024 - 4096


class SegmentIds(NamedTuple):
  """Segment ids, as in the TPU kernel. Tokens only attend within a segment.

  q: i32[q_seq_len] or i32[batch, q_seq_len]
  kv: i32[kv_seq_len] or i32[batch, kv_seq_len]
  """

  q: jax.Array
  kv: jax.Array


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tiling of the forward and backward kernels.

  block_q / block_kv: granularity of the sparse schedule (and of the forward
    kernel's tiles).
  num_stages: depth of the forward kernel's K/V SMEM ring.
  block_kv_dq: KV rows per step of the dQ kernel (64 or 128; None picks the
    largest that fits the SMEM/TMEM budgets).
  block_q_dkv: q rows per step of the dK/dV kernel (64 or 128; None likewise).
  num_stages_bwd: depth of the backward kernels' SMEM rings.
  """

  block_q: int | None = None  # None: choose 128 or 256 per call (see below)
  block_kv: int = 128
  num_stages: int = 2
  block_kv_dq: int | None = None
  block_q_dkv: int | None = None
  num_stages_bwd: int = 2

  def __post_init__(self):
    if self.block_q not in (None, BLOCK_Q, 2 * BLOCK_Q):
      # 256 selects the two-tile ping-pong forward kernel; None picks it
      # automatically where it is faster (head_dim 128, little extra work).
      raise ValueError(f"block_q must be None, 128 or 256, got {self.block_q}")
    if self.block_kv not in (64, 128):
      # Two f32 S buffers of 256 columns would fill all of TMEM.
      raise ValueError(f"block_kv must be 64 or 128, got {self.block_kv}")
    if self.num_stages < 2:
      # QK of step s+1 is issued before PV of step s, so the K/V ring must
      # hold two steps or the pipeline deadlocks.
      raise ValueError(f"num_stages must be >= 2, got {self.num_stages}")
    # The backward kernels contract over these sub-block sizes, and tcgen05
    # needs 16-bit contraction dims in multiples of 64.
    if self.block_kv_dq not in (None, 64, 128):
      raise ValueError(f"block_kv_dq={self.block_kv_dq} must be 64 or 128")
    if self.block_q_dkv not in (None, 64, 128):
      raise ValueError(f"block_q_dkv={self.block_q_dkv} must be 64 or 128")
    if self.num_stages_bwd < 1:
      raise ValueError("num_stages_bwd must be >= 1")


def _where(pred, x, y):
  """`jnp.where` with explicit broadcasting.

  Warpgroup lowering of `select_n` needs all operands to have the same shape.
  """
  shape = jnp.broadcast_shapes(jnp.shape(pred), jnp.shape(x), jnp.shape(y))
  dtype = x.dtype if hasattr(x, "dtype") else y.dtype
  bcast = lambda a, dt: jnp.broadcast_to(jnp.asarray(a, dt), shape)
  return lax.select(bcast(pred, jnp.bool_), bcast(x, dtype), bcast(y, dtype))


def _swizzle_transforms(minor_dim: int, dtype) -> tuple[Any, ...]:
  swizzle = plgpu.find_swizzle(minor_dim * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  return (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )


def _splash_attention_forward(
    q: jax.Array,  # [batch, num_q_heads, q_seq_len, head_dim]
    k: jax.Array,  # [batch, num_kv_heads, kv_seq_len, head_dim]
    v: jax.Array,  # [batch, num_kv_heads, kv_seq_len, head_dim_v]
    segment_ids: SegmentIds | None,  # i32[batch, seq] each
    num_steps: jax.Array,
    q_block_order: jax.Array,
    kv_block: jax.Array,
    block_kind: jax.Array,
    mask_block: jax.Array | None,
    partial_mask_blocks: jax.Array | None,
    *,
    mask_function,
    block_sizes: BlockSizes,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    save_residuals: bool,
    interpret: Any = None,
):
  batch, num_q_heads, q_seq_len, head_dim = q.shape
  _, num_kv_heads, kv_seq_len, head_dim_v = v.shape
  dtype = q.dtype
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  num_stages = block_sizes.num_stages
  if k.shape[:3] != v.shape[:3] or k.shape[0] != batch:
    raise ValueError(f"Incompatible k {k.shape} / v {v.shape} for q {q.shape}")
  if k.shape[-1] != head_dim:
    raise ValueError(f"q and k head dims differ: {head_dim} vs {k.shape[-1]}")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} not a multiple of {num_kv_heads=}")
  if k.dtype != dtype or v.dtype != dtype:
    raise ValueError(f"q, k, v dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")
  if jnp.dtype(dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
    raise NotImplementedError(f"Only bf16/f16 are supported, got {dtype}")
  if head_dim % 64 or head_dim_v % 64 or head_dim > 256 or head_dim_v > 256:
    raise NotImplementedError(
        f"head dims must be multiples of 64 and <= 256: {head_dim}, {head_dim_v}"
    )
  if q_seq_len % bq or kv_seq_len % bkv:
    raise ValueError(
        f"Sequence lengths ({q_seq_len}, {kv_seq_len}) must be multiples of"
        f" the block sizes ({bq}, {bkv})."
    )
  tmem_cols = head_dim_v + 2 * bkv + bkv // 2
  if tmem_cols > TMEM_COLS:
    raise ValueError(
        f"TMEM budget exceeded ({tmem_cols} > {TMEM_COLS} columns); use a"
        " smaller block_kv."
    )
  itemsize = jnp.dtype(dtype).itemsize
  smem_bytes = (
      bq * head_dim * itemsize  # Q (reused for the output tile)
      + (0 if head_dim == head_dim_v else bq * head_dim_v * itemsize)
      + num_stages * bkv * (head_dim + head_dim_v) * itemsize  # K, V ring
      + (num_stages * bq * bkv if partial_mask_blocks is not None else 0)
      + (bq + num_stages * bkv) * 4 * (segment_ids is not None)
      + bq * 4 * save_residuals
  )
  if smem_bytes > SMEM_BYTES:
    raise ValueError(
        f"Shared memory budget exceeded ({smem_bytes} > {SMEM_BYTES} bytes);"
        " reduce num_stages or block_kv."
    )
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  num_q_blocks = q_seq_len // bq
  mask_heads = num_steps.shape[0]
  if mask_heads not in (1, num_q_heads):
    raise ValueError(f"Mask has {mask_heads} heads, q has {num_q_heads}")
  if num_steps.shape[1] != num_q_blocks:
    raise ValueError("Mask info does not match the q sequence length")
  has_dense_mask = partial_mask_blocks is not None
  has_segments = segment_ids is not None
  # Mask blocks and KV segment ids are read by the softmax warpgroup, which
  # releases their SMEM slots through `aux_consumed`.
  has_aux = has_dense_mask or has_segments

  def kernel(*refs):
    refs = list(refs)
    q_gmem, k_gmem, v_gmem = refs[:3]
    del refs[:3]
    if has_segments:
      q_seg_gmem, kv_seg_gmem = refs[:2]
      del refs[:2]
    num_steps_gmem, q_order_gmem, kv_block_gmem, block_kind_gmem = refs[:4]
    del refs[:4]
    if has_dense_mask:
      mask_block_gmem, mask_blocks_gmem = refs[:2]
      del refs[:2]
    out_gmem = refs.pop(0)
    lse_gmem = refs.pop(0) if save_residuals else None
    (
        q_smem, k_smem, v_smem, o_smem, lse_smem,
        q_seg_smem, kv_seg_smem, mask_smem,
        s_tmem, p_tmem, o_tmem,
        q_barrier, k_barriers, v_barriers, seg_barriers, mask_barriers,
        kv_consumed, aux_consumed, s_ready, p_ready, pv_done,
    ) = refs

    if reuse_q_smem:
      o_smem = q_smem
    h = lax.axis_index("h")
    b = lax.axis_index("b")
    wg = lax.axis_index("wg")
    mh = h if mask_heads > 1 else 0
    qi = q_order_gmem[mh, lax.axis_index("q")]  # heaviest rows first
    kv_head = lax.div(h, q_heads_per_kv_head)
    n = num_steps_gmem[mh, qi]
    q_slice = pl.ds(qi * bq, bq)

    @pl.when(wg == 1)
    def _producer_wg():
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id == 0)
        def _tma_warp():
          plgpu.copy_gmem_to_smem(q_gmem.at[b, h, q_slice], q_smem, q_barrier)
          if has_segments:
            plgpu.copy_gmem_to_smem(
                q_seg_gmem.at[b, q_slice], q_seg_smem, q_barrier
            )

          @pl.loop(0, n)
          def _kv_loop(s):
            slot = lax.rem(s, num_stages)

            @pl.when(s >= num_stages)
            def _():
              # Wait until the PV MMA of step s - num_stages finished reading
              # this slot's K/V and the softmax is done with its mask / segment
              # ids.
              plgpu.barrier_wait(kv_consumed.at[slot])
              if has_aux:
                plgpu.barrier_wait(aux_consumed.at[slot])

            kv_slice = pl.ds(kv_block_gmem[mh, qi, s] * bkv, bkv)
            plgpu.copy_gmem_to_smem(
                k_gmem.at[b, kv_head, kv_slice], k_smem.at[slot],
                k_barriers.at[slot],
            )
            plgpu.copy_gmem_to_smem(
                v_gmem.at[b, kv_head, kv_slice], v_smem.at[slot],
                v_barriers.at[slot],
            )
            if has_segments:
              plgpu.copy_gmem_to_smem(
                  kv_seg_gmem.at[b, kv_slice], kv_seg_smem.at[slot],
                  seg_barriers.at[slot],
              )
            if has_dense_mask:
              @pl.when(block_kind_gmem[mh, qi, s] == mask_info_lib.PARTIAL)
              def _():
                plgpu.copy_gmem_to_smem(
                    mask_blocks_gmem.at[mask_block_gmem[mh, qi, s]],
                    mask_smem.at[slot],
                    mask_barriers.at[slot],
                )

          # Drain: observe the completions of the last `num_stages` steps so
          # that every barrier phase is consumed before exit.
          @pl.loop(jnp.maximum(n - num_stages, 0), n)
          def _drain(s):
            plgpu.barrier_wait(kv_consumed.at[lax.rem(s, num_stages)])
            if has_aux:
              plgpu.barrier_wait(aux_consumed.at[lax.rem(s, num_stages)])

        @pl.when(warp_id == 1)
        def _mma_warp():
          def issue_qk(s):
            slot = lax.rem(s, num_stages)
            s_buf = lax.rem(s, 2)
            plgpu.barrier_wait(k_barriers.at[slot])
            plgpu.tcgen05_mma(
                s_tmem.at[:, pl.ds(s_buf * bkv, bkv)],
                q_smem,
                k_smem.at[slot].T,
                s_ready.at[s_buf],
                accumulate=False,
            )

          plgpu.barrier_wait(q_barrier)

          @pl.when(n > 0)
          def _():
            issue_qk(0)

          @pl.loop(0, n)
          def _mma_loop(s):
            # Overlap: S_{s+1} = Q K_{s+1}^T runs while softmax works on S_s.
            # S buffer (s+1)%2 was last read by softmax step s-1, which
            # signalled p_ready before PV_{s-1} was issued below.
            @pl.when(s + 1 < n)
            def _():
              issue_qk(s + 1)

            slot = lax.rem(s, num_stages)
            plgpu.barrier_wait(v_barriers.at[slot])
            plgpu.barrier_wait(p_ready)
            plgpu.tcgen05_mma(
                o_tmem, p_tmem, v_smem.at[slot], kv_consumed.at[slot],
                accumulate=s > 0,
            )
            plgpu.tcgen05_commit_arrive(pv_done)

    @pl.when(wg == 0)
    def _softmax_wg():
      plgpu.barrier_wait(q_barrier)
      if has_segments:
        q_ids = plgpu.load(
            q_seg_smem, layout=plgpu.Layout.TCGEN05.reduce(1)
        )

      def apply_masks(qk, s, slot, kv_blk):
        kind = block_kind_gmem[mh, qi, s]
        is_partial = kind == mask_info_lib.PARTIAL
        if has_dense_mask:
          def load_mask_block():
            plgpu.barrier_wait(mask_barriers.at[slot])
            mask = plgpu.load(
                mask_smem.at[slot], layout=plgpu.Layout.TCGEN05
            )
            return _where(mask != 0, qk, mask_value)
          qk = lax.cond(is_partial, load_mask_block, lambda: qk)
        elif mask_function is not None:
          def compute_mask():
            q_pos = qi * bq + plgpu.broadcasted_iota(
                jnp.int32, (bq, bkv), 0, layout=plgpu.Layout.TCGEN05
            )
            kv_pos = kv_blk * bkv + plgpu.broadcasted_iota(
                jnp.int32, (bq, bkv), 1, layout=plgpu.Layout.TCGEN05
            )
            return _where(mask_function(q_pos, kv_pos), qk, mask_value)
          qk = lax.cond(is_partial, compute_mask, lambda: qk)
        if has_segments:
          plgpu.barrier_wait(seg_barriers.at[slot])
          kv_ids = plgpu.load(
              kv_seg_smem.at[slot], layout=plgpu.Layout.TCGEN05.reduce(0)
          )
          same_segment = (
              lax.broadcast_in_dim(q_ids, (bq, bkv), [0])
              == lax.broadcast_in_dim(kv_ids, (bq, bkv), [1])
          )
          qk = _where(same_segment, qk, mask_value)
        return qk

      def body(s, carry):
        m_prev, l_prev = carry
        slot = lax.rem(s, num_stages)
        s_buf = lax.rem(s, 2)
        kv_blk = kv_block_gmem[mh, qi, s]

        plgpu.barrier_wait(s_ready.at[s_buf])
        qk = plgpu.async_load_tmem(s_tmem.at[:, pl.ds(s_buf * bkv, bkv)])
        if attn_logits_soft_cap is not None:
          qk = jnp.tanh(qk / attn_logits_soft_cap) * attn_logits_soft_cap
        # Work in the log2 domain so that exp becomes a single exp2.  Masking
        # happens after the scaling so that `mask_value` is not rescaled
        # (it would overflow to -inf).
        qk = qk * LOG2E
        qk = apply_masks(qk, s, slot, kv_blk)
        if has_aux:
          # The mask / segment ids were read through the generic proxy; fence
          # them against the async-proxy (TMA) overwrite of the slot.
          plgpu.commit_smem()
          plgpu.barrier_arrive(aux_consumed.at[slot])

        m_curr = jnp.maximum(m_prev, qk.max(axis=1))
        # Lazy rescaling (as in FlashAttention-4): a row keeps its stale
        # running max unless the new one exceeds it by more than
        # RESCALE_THRESHOLD (log2 units).  P is then bounded by
        # 2**RESCALE_THRESHOLD, and O only needs rescaling in TMEM when some
        # row's max actually moved.
        needs_rescale = m_curr - m_prev > RESCALE_THRESHOLD
        m_next = _where(needs_rescale, m_curr, m_prev)
        alpha = jnp.exp2(m_prev - m_next)
        p = jnp.exp2(qk - lax.broadcast_in_dim(m_next, qk.shape, [0]))
        l_next = l_prev * alpha + p.sum(axis=1)
        any_rescale = jnp.max(needs_rescale.astype(jnp.int32)) > 0

        # PV_{s-1} must be complete before we overwrite P and rescale O.
        @pl.when(s > 0)
        def _():
          plgpu.barrier_wait(pv_done)

        plgpu.async_store_tmem(p_tmem, p.astype(dtype))

        @pl.when(jnp.logical_and(s > 0, any_rescale))
        def _rescale_o():
          o = plgpu.async_load_tmem(o_tmem)
          plgpu.wait_load_tmem()  # The load must finish before we overwrite.
          plgpu.async_store_tmem(
              o_tmem, o * lax.broadcast_in_dim(alpha, o.shape, [0])
          )

        plgpu.wait_load_tmem()  # S and O loads are done: TMEM may be reused.
        plgpu.commit_tmem()  # P and O stores are visible to the tensor core.
        plgpu.barrier_arrive(p_ready)
        return m_next, l_next

      m_init = jnp.full((bq,), mask_value, jnp.float32)
      l_init = jnp.zeros((bq,), jnp.float32)
      m_i, l_i = lax.fori_loop(0, n, body, (m_init, l_init))

      def normalized_output():
        plgpu.barrier_wait(pv_done)  # Wait for the last PV MMA.
        o = plgpu.async_load_tmem(o_tmem)
        plgpu.wait_load_tmem()
        o = o * lax.broadcast_in_dim(1.0 / l_i, o.shape, [0])
        # Rows whose logits are all masked keep m == mask_value; report
        # lse == mask_value for them, as the TPU kernel does (in f32,
        # mask_value + log(count) == mask_value).
        lse = _where(m_i == mask_value, mask_value,
                     (m_i + jnp.log2(l_i)) * LN2)
        return o.astype(dtype), lse

      def empty_output():
        # No visible KV block: the output is zero and the TMEM accumulator was
        # never written, so it must not be read.
        return (
            jnp.zeros((bq, head_dim_v), dtype),
            jnp.full((bq,), mask_value, jnp.float32),
        )

      o, lse = lax.cond(n > 0, normalized_output, empty_output)
      # If o_smem aliases q_smem this is safe: all MMAs reading q_smem have
      # completed (pv_done), or none were issued and we waited for the Q TMA.
      o_smem[...] = o
      if save_residuals:
        lse_smem[...] = lse
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, out_gmem.at[b, h, q_slice])
      if save_residuals:
        plgpu.copy_smem_to_gmem(lse_smem, lse_gmem.at[b, h, q_slice])
      plgpu.wait_smem_to_gmem(0)

  q_transforms = _swizzle_transforms(head_dim, dtype)
  v_transforms = _swizzle_transforms(head_dim_v, dtype)
  # When the head dims match the output tile reuses the Q buffer.
  reuse_q_smem = head_dim == head_dim_v
  o_smem_type = (
      None
      if reuse_q_smem
      else plgpu.SMEM((bq, head_dim_v), dtype, transforms=v_transforms)
  )
  scratch_types = [
      plgpu.SMEM((bq, head_dim), dtype, transforms=q_transforms),
      plgpu.SMEM((num_stages, bkv, head_dim), dtype, transforms=q_transforms),
      plgpu.SMEM((num_stages, bkv, head_dim_v), dtype, transforms=v_transforms),
      o_smem_type,
      plgpu.SMEM((bq,), jnp.float32) if save_residuals else None,
      plgpu.SMEM((bq,), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bkv), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bq, bkv), jnp.int8) if has_dense_mask else None,
      plgpu.TMEM((bq, 2 * bkv), jnp.float32),
      plgpu.TMEM((bq, bkv), dtype, packed=True),
      plgpu.TMEM((bq, head_dim_v), jnp.float32),
      plgpu.Barrier(num_arrivals=1 + has_segments),
      plgpu.Barrier(num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages) if has_segments else None,
      plgpu.Barrier(num_barriers=num_stages) if has_dense_mask else None,
      plgpu.Barrier(num_barriers=num_stages, orders_tensor_core=True),
      plgpu.Barrier(num_barriers=num_stages) if has_aux else None,
      plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
  ]

  def entry(*refs):
    # `None` placeholders keep the scratch signature of `kernel` fixed.
    present = [t for t in scratch_types if t is not None]

    def unflatten(*scratch):
      it = iter(scratch)
      kernel(*refs, *[None if t is None else next(it) for t in scratch_types])

    pl.run_scoped(unflatten, *present, collective_axes="wg")

  out_type = [jax.ShapeDtypeStruct((batch, num_q_heads, q_seq_len, head_dim_v), dtype)]
  if save_residuals:
    out_type.append(
        jax.ShapeDtypeStruct((batch, num_q_heads, q_seq_len), jnp.float32)
    )
  inputs = [q, k, v]
  if has_segments:
    inputs += [segment_ids.q, segment_ids.kv]
  inputs += [num_steps, q_block_order, kv_block, block_kind]
  if has_dense_mask:
    inputs += [mask_block, partial_mask_blocks]

  f = plgpu.kernel(
      entry,
      out_type=tuple(out_type),
      grid=(num_q_blocks, num_q_heads, batch),
      grid_names=("q", "h", "b"),
      num_threads=2,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
          approx_math=True,
      ),
      interpret=interpret,
  )
  outs = f(*inputs)
  if save_residuals:
    return outs[0], outs[1]
  return outs[0]
