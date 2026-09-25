# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Two-tile ("ping-pong") forward kernel, FlashAttention-4 style.

Each CTA owns 256 query rows as two 128-row tiles that share every K/V block:

  warpgroup 2, warp 0  TMA: Q0, Q1 once; K, V (+ mask block, KV segment ids)
                       per step through a num_stages-deep ring.
  warpgroup 2, warp 1  tcgen05, interleaving the tiles so that the tensor core
                       works on one tile while the other tile's softmax runs:
                         S0(0) S1(0) | PV0(j) S0(j+1) | PV1(j) S1(j+1) | ...
  warpgroups 0 and 1   softmax of tile 0 / tile 1.

TMEM per CTA: for each tile an S buffer (block_kv f32 columns) with P (bf16)
aliased onto it, plus the O accumulator (head_dim columns).  At head_dim 128
and block_kv 128 that is exactly the 512 TMEM columns.

Ordering relies on tcgen05 semantics: MMAs issued by one thread execute in
order, and a commit tracks all previously issued MMAs.  So when softmax t sees
S_t(j) ready, PV_t(j-1) has finished reading P_t(j-1) (which S_t(j) overwrote)
and has finished updating O_t: softmax t may rescale O_t and write P_t(j)
without any further wait.

The sparse schedule is built at 256-row granularity.  A KV block that is
visible to only one of the tiles is computed for both and fully masked for the
other; rows whose first visited block is fully masked are corrected by the
(lazy) online-softmax rescaling once a visible key arrives.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from . import mask_info as mask_info_lib
from .kernel import (
    LN2,
    LOG2E,
    RESCALE_THRESHOLD,
    SMEM_BYTES,
    TMEM_COLS,
    SegmentIds,
    _swizzle_transforms,
    _where,
)

TILE = 128
NUM_TILES = 2
CTA_ROWS = TILE * NUM_TILES
_ROWS = plgpu.Layout.TCGEN05.reduce(1)
_COLS = plgpu.Layout.TCGEN05.reduce(0)


def splash_attention_forward_pingpong(
    q, k, v,
    segment_ids: SegmentIds | None,
    num_steps, q_block_order, kv_block, block_kind, mask_block,
    partial_mask_blocks,
    *,
    mask_function,
    block_kv: int,
    num_stages: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    save_residuals: bool,
    softmax_registers: int | None = None,
    producer_registers: int | None = None,
    interpret: Any = None,
):
  batch, num_q_heads, q_seq_len, head_dim = q.shape
  _, num_kv_heads, kv_seq_len, head_dim_v = v.shape
  dtype = q.dtype
  bkv = block_kv
  if head_dim != head_dim_v:
    raise NotImplementedError("ping-pong kernel needs head_dim == head_dim_v")
  if head_dim % 64 or head_dim > 128:
    raise NotImplementedError(f"ping-pong kernel supports head_dim 64/128, got {head_dim}")
  if q_seq_len % CTA_ROWS or kv_seq_len % bkv:
    raise ValueError(f"seq lens must be multiples of ({CTA_ROWS}, {bkv})")
  if num_stages < 2:
    raise ValueError("num_stages must be >= 2")
  itemsize = jnp.dtype(dtype).itemsize
  has_dense_mask = partial_mask_blocks is not None
  has_segments = segment_ids is not None
  has_aux = has_dense_mask or has_segments
  tmem_cols = NUM_TILES * (bkv + head_dim)
  if tmem_cols > TMEM_COLS:
    raise ValueError(f"TMEM budget exceeded ({tmem_cols} > {TMEM_COLS}); "
                     "use a smaller block_kv")
  smem_bytes = (
      CTA_ROWS * head_dim * itemsize
      + num_stages * bkv * 2 * head_dim * itemsize
      + (num_stages * CTA_ROWS * bkv if has_dense_mask else 0)
      + (CTA_ROWS + num_stages * bkv) * 4 * has_segments
      + CTA_ROWS * 4 * save_residuals
  )
  if smem_bytes > SMEM_BYTES:
    raise ValueError(f"Shared memory budget exceeded ({smem_bytes} > "
                     f"{SMEM_BYTES} bytes); reduce num_stages or block_kv")
  # setmaxnreg: 2 x 128 x 232 + 128 x 40 = 64512 of the SM's 65536 registers
  # (the same split as JAX's Hopper attention kernel).  SPLASH_PP_REGS="a,b"
  # overrides it for experiments; "0,0" disables register reallocation.
  env_regs = os.environ.get("SPLASH_PP_REGS")
  if env_regs:
    softmax_registers, producer_registers = map(int, env_regs.split(","))
  softmax_registers = 232 if softmax_registers is None else softmax_registers
  producer_registers = 40 if producer_registers is None else producer_registers
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  mask_heads = num_steps.shape[0]
  serialize_pv = interpret is not None
  if num_steps.shape[1] != q_seq_len // CTA_ROWS:
    raise ValueError("mask info must be built with block_q=256")

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
        q_smem, k_smem, v_smem, lse_smem, q_seg_smem, kv_seg_smem, mask_smem,
        sp0, sp1, o_tmem,
        q_barrier, k_barriers, v_barriers, seg_barriers, mask_barriers,
        kv_consumed, aux_consumed, s_ready, p_ready, o_done, pv_order,
    ) = refs
    s_tmems = (sp0[0], sp1[0])
    p_tmems = (sp0[1], sp1[1])

    h = lax.axis_index("h")
    b = lax.axis_index("b")
    wg = lax.axis_index("wg")
    mh = h if mask_heads > 1 else 0
    qi = q_order_gmem[mh, lax.axis_index("q")]  # heaviest rows first
    kv_head = lax.div(h, q_heads_per_kv_head)
    n = num_steps_gmem[mh, qi]
    cta_rows = pl.ds(qi * CTA_ROWS, CTA_ROWS)

    @pl.when(wg == NUM_TILES)
    def _producer_wg():
      if producer_registers:
        plgpu.set_max_registers(producer_registers, action="decrease")

      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id == 0)
        def _tma_warp():
          plgpu.copy_gmem_to_smem(q_gmem.at[b, h, cta_rows], q_smem, q_barrier)
          if has_segments:
            plgpu.copy_gmem_to_smem(q_seg_gmem.at[b, cta_rows], q_seg_smem,
                                    q_barrier)

          @pl.loop(0, n)
          def _kv_loop(s):
            slot = lax.rem(s, num_stages)

            @pl.when(s >= num_stages)
            def _():
              plgpu.barrier_wait(kv_consumed.at[slot])
              if has_aux:
                plgpu.barrier_wait(aux_consumed.at[slot])

            kv_slice = pl.ds(kv_block_gmem[mh, qi, s] * bkv, bkv)
            plgpu.copy_gmem_to_smem(k_gmem.at[b, kv_head, kv_slice],
                                    k_smem.at[slot], k_barriers.at[slot])
            plgpu.copy_gmem_to_smem(v_gmem.at[b, kv_head, kv_slice],
                                    v_smem.at[slot], v_barriers.at[slot])
            if has_segments:
              plgpu.copy_gmem_to_smem(kv_seg_gmem.at[b, kv_slice],
                                      kv_seg_smem.at[slot],
                                      seg_barriers.at[slot])
            if has_dense_mask:
              @pl.when(block_kind_gmem[mh, qi, s] == mask_info_lib.PARTIAL)
              def _():
                plgpu.copy_gmem_to_smem(
                    mask_blocks_gmem.at[mask_block_gmem[mh, qi, s]],
                    mask_smem.at[slot], mask_barriers.at[slot])

          @pl.loop(jnp.maximum(n - num_stages, 0), n)
          def _drain(s):
            plgpu.barrier_wait(kv_consumed.at[lax.rem(s, num_stages)])
            if has_aux:
              plgpu.barrier_wait(aux_consumed.at[lax.rem(s, num_stages)])

        @pl.when(warp_id == 1)
        def _mma_warp():
          def issue_s(t, s):
            slot = lax.rem(s, num_stages)
            if serialize_pv:
              # Interpreter only: it does not model the in-order execution of
              # tcgen05 MMAs, so make "PV_t(s-1) done before S_t(s) overwrites
              # P_t" explicit.  On hardware the MMAs simply pipeline.
              @pl.when(s > 0)
              def _():
                plgpu.barrier_wait(pv_order.at[t])
            plgpu.tcgen05_mma(s_tmems[t], q_smem.at[pl.ds(t * TILE, TILE)],
                              k_smem.at[slot].T, accumulate=False)
            # An explicit commit tracks every earlier MMA of this thread,
            # including PV_t(s-1), which the softmax relies on.
            plgpu.tcgen05_commit_arrive(s_ready.at[t])

          def issue_pv(t, s):
            slot = lax.rem(s, num_stages)
            plgpu.barrier_wait(p_ready.at[t])
            plgpu.tcgen05_mma(o_tmem.at[t], p_tmems[t], v_smem.at[slot],
                              accumulate=s > 0)
            if serialize_pv:
              plgpu.tcgen05_commit_arrive(pv_order.at[t])

          plgpu.barrier_wait(q_barrier)

          @pl.when(n > 0)
          def _():
            plgpu.barrier_wait(k_barriers.at[0])
            issue_s(0, 0)
            issue_s(1, 0)

          @pl.loop(0, n)
          def _mma_loop(s):
            slot = lax.rem(s, num_stages)
            has_next = s + 1 < n
            plgpu.barrier_wait(v_barriers.at[slot])
            issue_pv(0, s)

            @pl.when(has_next)
            def _():
              plgpu.barrier_wait(k_barriers.at[lax.rem(s + 1, num_stages)])
              issue_s(0, s + 1)

            issue_pv(1, s)
            # Both tiles are done with this slot's K and V.
            plgpu.tcgen05_commit_arrive(kv_consumed.at[slot])

            @pl.when(has_next)
            def _():
              issue_s(1, s + 1)

          @pl.when(n > 0)
          def _():
            plgpu.tcgen05_commit_arrive(o_done)
            if serialize_pv:  # observe the last PV of each tile
              for t in range(NUM_TILES):
                plgpu.barrier_wait(pv_order.at[t])

    def softmax_wg(t):  # t: static tile index
      if softmax_registers:
        plgpu.set_max_registers(softmax_registers, action="increase")
      rows = pl.ds(t * TILE, TILE)
      plgpu.barrier_wait(q_barrier)
      if has_segments:
        q_ids = plgpu.load(q_seg_smem.at[rows], layout=_ROWS)

      def apply_masks(x, s, slot, kv_blk):
        is_partial = block_kind_gmem[mh, qi, s] == mask_info_lib.PARTIAL
        if has_dense_mask:
          def load_mask():
            plgpu.barrier_wait(mask_barriers.at[slot])
            m = plgpu.load(mask_smem.at[slot, rows],
                           layout=plgpu.Layout.TCGEN05)
            return _where(m != 0, x, mask_value)
          x = lax.cond(is_partial, load_mask, lambda: x)
        elif mask_function is not None:
          def compute_mask():
            q_pos = qi * CTA_ROWS + t * TILE + plgpu.broadcasted_iota(
                jnp.int32, (TILE, bkv), 0, layout=plgpu.Layout.TCGEN05)
            kv_pos = kv_blk * bkv + plgpu.broadcasted_iota(
                jnp.int32, (TILE, bkv), 1, layout=plgpu.Layout.TCGEN05)
            return _where(mask_function(q_pos, kv_pos), x, mask_value)
          x = lax.cond(is_partial, compute_mask, lambda: x)
        if has_segments:
          plgpu.barrier_wait(seg_barriers.at[slot])
          kv_ids = plgpu.load(kv_seg_smem.at[slot], layout=_COLS)
          same = (lax.broadcast_in_dim(q_ids, (TILE, bkv), [0])
                  == lax.broadcast_in_dim(kv_ids, (TILE, bkv), [1]))
          x = _where(same, x, mask_value)
        return x

      def body(s, carry):
        m_prev, l_prev = carry
        slot = lax.rem(s, num_stages)
        kv_blk = kv_block_gmem[mh, qi, s]
        plgpu.barrier_wait(s_ready.at[t])
        # PV_t(s-1) has completed (see module docstring): S_t may be
        # overwritten with P_t and O_t may be rescaled.
        qk = plgpu.async_load_tmem(s_tmems[t])
        plgpu.wait_load_tmem()
        if attn_logits_soft_cap is not None:
          qk = jnp.tanh(qk / attn_logits_soft_cap) * attn_logits_soft_cap
        qk = apply_masks(qk * LOG2E, s, slot, kv_blk)
        if has_aux:
          plgpu.commit_smem()  # Fence generic reads before TMA overwrites.
          plgpu.barrier_arrive(aux_consumed.at[slot])
        m_curr = jnp.maximum(m_prev, qk.max(axis=1))
        needs_rescale = m_curr - m_prev > RESCALE_THRESHOLD
        m_next = _where(needs_rescale, m_curr, m_prev)
        alpha = jnp.exp2(m_prev - m_next)
        p = jnp.exp2(qk - lax.broadcast_in_dim(m_next, qk.shape, [0]))
        l_next = l_prev * alpha + p.sum(axis=1)
        plgpu.async_store_tmem(p_tmems[t], p.astype(dtype))

        # Unlike the single-tile kernel, O is rescaled on every step (alpha is
        # exactly 1 for rows whose max did not move).  Skipping the rescale
        # needs a warpgroup-wide "any row moved" scalar, i.e. a cross-warp
        # reduction, and Mosaic GPU places every cross-warp reduction scratch
        # at the same SMEM offset: the two softmax warpgroups would clobber
        # each other (observed on B200: wrong results and deadlocks).
        @pl.when(s > 0)
        def _rescale_o():
          o = plgpu.async_load_tmem(o_tmem.at[t])
          plgpu.wait_load_tmem()
          plgpu.async_store_tmem(
              o_tmem.at[t], o * lax.broadcast_in_dim(alpha, o.shape, [0]))

        plgpu.commit_tmem()
        plgpu.barrier_arrive(p_ready.at[t])
        return m_next, l_next

      m_i, l_i = lax.fori_loop(
          0, n, body,
          (jnp.full((TILE,), mask_value, jnp.float32),
           jnp.zeros((TILE,), jnp.float32)))

      def normalized_output():
        plgpu.barrier_wait(o_done)
        o = plgpu.async_load_tmem(o_tmem.at[t])
        plgpu.wait_load_tmem()
        o = o * lax.broadcast_in_dim(1.0 / l_i, o.shape, [0])
        lse = _where(m_i == mask_value, mask_value,
                     (m_i + jnp.log2(l_i)) * LN2)
        return o.astype(dtype), lse

      def empty_output():
        return (jnp.zeros((TILE, head_dim), dtype),
                jnp.full((TILE,), mask_value, jnp.float32))

      o, lse = lax.cond(n > 0, normalized_output, empty_output)
      # Every MMA reading q_smem has completed (o_done), or none was issued.
      q_smem[rows] = o
      if save_residuals:
        lse_smem[rows] = lse
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem.at[rows],
                              out_gmem.at[b, h, pl.ds(qi * CTA_ROWS + t * TILE, TILE)])
      if save_residuals:
        plgpu.copy_smem_to_gmem(
            lse_smem.at[rows],
            lse_gmem.at[b, h, pl.ds(qi * CTA_ROWS + t * TILE, TILE)])
      plgpu.wait_smem_to_gmem(0)

    for t in range(NUM_TILES):
      pl.when(wg == t)(functools.partial(softmax_wg, t))

  qk_t = _swizzle_transforms(head_dim, dtype)
  sp_union = lambda: plgpu.RefUnion(
      plgpu.TMEM((TILE, bkv), jnp.float32),
      plgpu.TMEM((TILE, bkv), dtype, packed=True))
  scratch_types = [
      plgpu.SMEM((CTA_ROWS, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bkv, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bkv, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((CTA_ROWS,), jnp.float32) if save_residuals else None,
      plgpu.SMEM((CTA_ROWS,), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bkv), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, CTA_ROWS, bkv), jnp.int8) if has_dense_mask
      else None,
      sp_union(),
      sp_union(),
      plgpu.TMEM((NUM_TILES, TILE, head_dim), jnp.float32),
      plgpu.Barrier(num_arrivals=1 + has_segments),
      plgpu.Barrier(num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages) if has_segments else None,
      plgpu.Barrier(num_barriers=num_stages) if has_dense_mask else None,
      plgpu.Barrier(num_barriers=num_stages, orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=NUM_TILES, num_barriers=num_stages)
      if has_aux else None,
      plgpu.Barrier(num_barriers=NUM_TILES, orders_tensor_core=True),
      plgpu.Barrier(num_barriers=NUM_TILES, orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(num_barriers=NUM_TILES, orders_tensor_core=True)
      if serialize_pv else None,
  ]

  def entry(*refs):
    present = [t for t in scratch_types if t is not None]

    def unflatten(*scratch):
      it = iter(scratch)
      kernel(*refs, *[None if t is None else next(it) for t in scratch_types])

    pl.run_scoped(unflatten, *present, collective_axes="wg")

  out_type = [jax.ShapeDtypeStruct(q.shape, dtype)]
  if save_residuals:
    out_type.append(jax.ShapeDtypeStruct(q.shape[:3], jnp.float32))
  inputs = [q, k, v]
  if has_segments:
    inputs += [segment_ids.q, segment_ids.kv]
  inputs += [num_steps, q_block_order, kv_block, block_kind]
  if has_dense_mask:
    inputs += [mask_block, partial_mask_blocks]
  outs = plgpu.kernel(
      entry,
      out_type=tuple(out_type),
      grid=(q_seq_len // CTA_ROWS, num_q_heads, batch),
      grid_names=("q", "h", "b"),
      num_threads=NUM_TILES + 1,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
          approx_math=True,
      ),
      interpret=interpret,
  )(*inputs)
  return (outs[0], outs[1]) if save_residuals else outs[0]
