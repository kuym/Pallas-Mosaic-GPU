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
# O columns rescaled per TMEM round trip by the correction warpgroup (bounds
# its register use).
_CORRECTION_CHUNK = 32
_ROWS = plgpu.Layout.TCGEN05.reduce(1)
_COLS = plgpu.Layout.TCGEN05.reduce(0)


# Cubic minimax fit of 2**f on [-0.5, 0.5]; max relative error 7.6e-5, far
# below bf16 resolution (P is stored as bf16).
_EXP2_COEFFS = (0.9999275516718514, 0.6932516151409869, 0.24261537603713018,
                0.05522259924620865)


def _tree(op, xs):
  """Combines xs pairwise (a balanced tree, not a serial chain)."""
  while len(xs) > 1:
    xs = [op(xs[i], xs[i + 1]) if i + 1 < len(xs) else xs[i]
          for i in range(0, len(xs), 2)]
  return xs[0]


SHIFTER = 12582912.0  # 1.5 * 2**23


def exp2_emulated(x, shifter):
  """2**x for x <= 0 on the FMA/ALU pipes, without float<->int conversions
  (those run on the same low-throughput unit as MUFU.EX2).

  Adding 1.5 * 2**23 rounds x to an integer r held in the low mantissa bits
  of t; the low 9 bits of the constant's bit pattern are zero, so
  bits(t) << 23 == r << 23 exactly, which is added to the exponent of the
  polynomial for 2**(x - r).  x is clamped at -125 so the exponent never
  underflows.

  `shifter` must equal SHIFTER but be opaque to the compiler (a runtime
  value): otherwise algebraic simplification folds (x + c) - c into x (XLA
  does), which silently turns f into 0."""
  x = jnp.maximum(x, -125.0)
  t = x + shifter
  f = x - (t - shifter)  # in [-0.5, 0.5]
  c0, c1, c2, c3 = _EXP2_COEFFS
  p = c0 + f * (c1 + f * (c2 + f * c3))
  exponent = lax.shift_left(lax.bitcast_convert_type(t, jnp.int32),
                            jnp.int32(23))
  return lax.bitcast_convert_type(
      lax.bitcast_convert_type(p, jnp.int32) + exponent, jnp.float32)


def _profile_params():
  """SPLASH_PROFILE_DIR=dir records a per-warp trace of the named scopes."""
  profile_dir = os.environ.get("SPLASH_PROFILE_DIR")
  if not profile_dir:
    return {}
  # The profiler buffers events in SMEM, so the space is small; bounds
  # checking truncates the trace instead of corrupting memory.
  return dict(profile_space=int(os.environ.get("SPLASH_PROFILE_SPACE", 256)),
              profile_dir=profile_dir,
              profile_trace_scope=plgpu.TraceScope.WARP,
              profile_bounds_check=True)


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
    exp_emulation_cols: int | None = None,
    softmax_parts: int | None = None,
    correction: bool | None = None,
    schedule: bool | None = None,
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
  if correction is None:
    correction = os.environ.get("SPLASH_CORRECTION", "0") == "1"
  if schedule is None:
    schedule = os.environ.get("SPLASH_SCHEDULE", "0") == "1"
  correction_registers = 64
  env_regs = os.environ.get("SPLASH_PP_REGS")
  if env_regs:
    regs = list(map(int, env_regs.split(",")))
    softmax_registers, producer_registers = regs[:2]
    if len(regs) > 2:
      correction_registers = regs[2]
  # 4 warpgroups: 2 x 200 + 64 + 40 = 504 x 128 registers (slack for
  # setmaxnreg); 3 warpgroups: 2 x 232 + 40 = 504 x 128.
  default_softmax = 200 if correction else 232
  softmax_registers = (default_softmax if softmax_registers is None
                       else softmax_registers)
  producer_registers = 40 if producer_registers is None else producer_registers
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  mask_heads = num_steps.shape[0]
  serialize_pv = interpret is not None
  # Debug-only ablations that remove work to bound its cost (wrong results):
  # SPLASH_ABLATE in {norescale, noexp, nosoftmax}.
  ablate = os.environ.get("SPLASH_ABLATE", "")
  if exp_emulation_cols is None:
    exp_emulation_cols = int(os.environ.get("SPLASH_EXP_EMU_COLS", 0))
  if exp_emulation_cols % 16 or not 0 <= exp_emulation_cols < bkv:
    raise ValueError(f"exp_emulation_cols={exp_emulation_cols} must be a "
                     f"multiple of 16 in [0, {bkv})")
  # The S tile is processed as independent column slices: the per-row max and
  # sum of each slice are separate dependency chains, which the compiler
  # otherwise emits as one long serial chain per row (latency-bound with only
  # two softmax warps per SM sub-partition).
  if softmax_parts is None:
    softmax_parts = int(os.environ.get("SPLASH_SOFTMAX_PARTS", 1))
  if bkv % softmax_parts or (bkv // softmax_parts) % 16:
    raise ValueError(f"softmax_parts={softmax_parts} must split block_kv="
                     f"{bkv} into multiples of 16 columns")
  if exp_emulation_cols:
    parts = [(0, exp_emulation_cols),
             (exp_emulation_cols, bkv - exp_emulation_cols)]
  else:
    w = bkv // softmax_parts
    parts = [(i * w, w) for i in range(softmax_parts)]
  fast_full_blocks = (len(parts) == 1 and not exp_emulation_cols
                      and not ablate and not has_segments
                      and os.environ.get("SPLASH_FAST_FULL", "1") == "1")
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
        alpha_smem, alpha_ready, alpha_consumed, o_free, o_corrected, turn,
    ) = refs
    CORRECTION_WG = NUM_TILES
    PRODUCER_WG = NUM_TILES + int(correction)
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

    @pl.when(wg == PRODUCER_WG)
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
              with jax.named_scope("tma_wait_slot"):
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
            if correction:
              # O_t is free for the correction warpgroup (PV_t(s-1) is done).
              @pl.when(s > 0)
              def _():
                plgpu.tcgen05_commit_arrive(o_free.at[t])

          def issue_pv(t, s):
            slot = lax.rem(s, num_stages)
            with jax.named_scope(f"mma_wait_p{t}"):
              plgpu.barrier_wait(p_ready.at[t])
              if correction:
                @pl.when(s > 0)
                def _():
                  plgpu.barrier_wait(o_corrected.at[t])
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
            with jax.named_scope("mma_wait_v"):
              plgpu.barrier_wait(v_barriers.at[slot])
            issue_pv(0, s)

            @pl.when(has_next)
            def _():
              with jax.named_scope("mma_wait_k"):
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

      def apply_masks(x, s, slot, kv_blk, c0, nc, first_part):
        """Masks columns [c0, c0 + nc) of the tile (x: [TILE, nc])."""
        is_partial = block_kind_gmem[mh, qi, s] == mask_info_lib.PARTIAL
        cols = pl.ds(c0, nc)
        if has_dense_mask:
          def load_mask():
            if first_part:
              plgpu.barrier_wait(mask_barriers.at[slot])
            m = plgpu.load(mask_smem.at[slot, rows, cols],
                           layout=plgpu.Layout.TCGEN05)
            return _where(m != 0, x, mask_value)
          x = lax.cond(is_partial, load_mask, lambda: x)
        elif mask_function is not None:
          def compute_mask():
            q_pos = qi * CTA_ROWS + t * TILE + plgpu.broadcasted_iota(
                jnp.int32, (TILE, nc), 0, layout=plgpu.Layout.TCGEN05)
            kv_pos = kv_blk * bkv + c0 + plgpu.broadcasted_iota(
                jnp.int32, (TILE, nc), 1, layout=plgpu.Layout.TCGEN05)
            return _where(mask_function(q_pos, kv_pos), x, mask_value)
          x = lax.cond(is_partial, compute_mask, lambda: x)
        if has_segments:
          if first_part:
            plgpu.barrier_wait(seg_barriers.at[slot])
          kv_ids = plgpu.load(kv_seg_smem.at[slot, cols], layout=_COLS)
          same = (lax.broadcast_in_dim(q_ids, (TILE, nc), [0])
                  == lax.broadcast_in_dim(kv_ids, (TILE, nc), [1]))
          x = _where(same, x, mask_value)
        return x

      def body(s, carry):
        m_prev, l_prev = carry
        # SHIFTER as a runtime value (see exp2_emulated): 0 * n is not
        # foldable for floats.
        shifter = jnp.float32(SHIFTER) + 0.0 * n.astype(jnp.float32)
        slot = lax.rem(s, num_stages)
        kv_blk = kv_block_gmem[mh, qi, s]
        if schedule:
          # Ping-pong scheduling: only one tile runs its exp-heavy phase at a
          # time, so the other tile's MMAs overlap it (FA3/FA4-style).
          with jax.named_scope("sm_wait_turn"):
            if t == 0:
              @pl.when(s > 0)
              def _():
                plgpu.barrier_wait(turn.at[0])
            else:
              plgpu.barrier_wait(turn.at[1])
        with jax.named_scope("sm_wait_s"):
          plgpu.barrier_wait(s_ready.at[t])
        # PV_t(s-1) has completed (see module docstring): S_t may be
        # overwritten with P_t and O_t may be rescaled.
        with jax.named_scope("sm_load_s"):
          qks = [plgpu.async_load_tmem(s_tmems[t].at[:, pl.ds(c0, nc)])
                 for c0, nc in parts]
          plgpu.wait_load_tmem()
        if attn_logits_soft_cap is not None:
          qks = [jnp.tanh(x / attn_logits_soft_cap) * attn_logits_soft_cap
                 for x in qks]

        def online_max(row_maxes):
          m_curr = _tree(jnp.maximum, [m_prev] + row_maxes)
          needs_rescale = m_curr - m_prev > RESCALE_THRESHOLD
          m_next = _where(needs_rescale, m_curr, m_prev)
          return m_next, jnp.exp2(m_prev - m_next)

        if fast_full_blocks:
          # Blocks that need no masking (FULL blocks without segment ids, the
          # vast majority) take a short path: the log2e scaling is folded
          # into exp2's argument (one FMA per element), and the row max is
          # taken on the raw logits.  Partial blocks keep the masked path,
          # whose semantics (mask_value in log2 units) are unchanged.
          [qk] = qks
          is_partial = block_kind_gmem[mh, qi, s] == mask_info_lib.PARTIAL

          def full_block():
            m_next, alpha = online_max([qk.max(axis=1) * LOG2E])
            p = jnp.exp2(qk * LOG2E - lax.broadcast_in_dim(m_next, qk.shape, [0]))
            return p, m_next, alpha

          def partial_block():
            x = apply_masks(qk * LOG2E, s, slot, kv_blk, 0, bkv, True)
            m_next, alpha = online_max([x.max(axis=1)])
            p = jnp.exp2(x - lax.broadcast_in_dim(m_next, x.shape, [0]))
            return p, m_next, alpha

          with jax.named_scope("sm_exp"):
            if has_dense_mask or mask_function is not None:
              p, m_next, alpha = lax.cond(is_partial, partial_block, full_block)
            else:
              p, m_next, alpha = full_block()
          if has_aux:
            plgpu.commit_smem()  # Fence generic reads before TMA overwrites.
            plgpu.barrier_arrive(aux_consumed.at[slot])
        else:
          with jax.named_scope("sm_mask"):
            qks = [apply_masks(x * LOG2E, s, slot, kv_blk, c0, nc, i == 0)
                   for i, (x, (c0, nc)) in enumerate(zip(qks, parts))]
            if has_aux:
              plgpu.commit_smem()  # Fence generic reads before TMA overwrites.
              plgpu.barrier_arrive(aux_consumed.at[slot])
          with jax.named_scope("sm_exp"):
            m_next, alpha = online_max([x.max(axis=1) for x in qks])
        with jax.named_scope("sm_exp"):
          if correction:
            @pl.when(s > 0)
            def _publish_alpha():
              @pl.when(s > 1)
              def _():  # the correction warpgroup has read alpha(s - 1)
                plgpu.barrier_wait(alpha_consumed.at[t])
              alpha_smem[t] = alpha
              plgpu.barrier_arrive(alpha_ready.at[t])
          ps = [p] if fast_full_blocks else []
          for i, x in enumerate([] if fast_full_blocks else qks):
            if ablate == "nosoftmax":  # perf experiment only: wrong results
              ps.append(x)
              continue
            x = x - lax.broadcast_in_dim(m_next, x.shape, [0])
            if ablate == "noexp":  # perf experiment only: wrong results
              ps.append(x)
              continue
            # The first `exp_emulation_cols` columns use a polynomial on the
            # FMA pipe, relieving the special-function unit (FA4's trick).
            p = (exp2_emulated(x, shifter) if (i == 0 and exp_emulation_cols)
                 else jnp.exp2(x))
            ps.append(p)
          l_next = _tree(jnp.add, [l_prev * alpha] + [p.sum(axis=1) for p in ps])
        if schedule:
          plgpu.barrier_arrive(turn.at[1 - t])  # hand the turn to the other tile
        with jax.named_scope("sm_store_p"):
          for p, (c0, nc) in zip(ps, parts):
            plgpu.async_store_tmem(p_tmems[t].at[:, pl.ds(c0, nc)],
                                   p.astype(dtype))

        # Unlike the single-tile kernel, O is rescaled on every step (alpha is
        # exactly 1 for rows whose max did not move).  Skipping the rescale
        # needs a warpgroup-wide "any row moved" scalar, i.e. a cross-warp
        # reduction, and Mosaic GPU places every cross-warp reduction scratch
        # at the same SMEM offset: the two softmax warpgroups would clobber
        # each other (observed on B200: wrong results and deadlocks).
        @pl.when(jnp.logical_and(
            s > 0, not correction and ablate not in ("norescale", "nosoftmax")))
        def _rescale_o():
          with jax.named_scope("sm_rescale"):
            o = plgpu.async_load_tmem(o_tmem.at[t])
            plgpu.wait_load_tmem()
            plgpu.async_store_tmem(
                o_tmem.at[t], o * lax.broadcast_in_dim(alpha, o.shape, [0]))

        with jax.named_scope("sm_commit"):
          plgpu.commit_tmem()
          plgpu.barrier_arrive(p_ready.at[t])
        return m_next, l_next

      m_i, l_i = lax.fori_loop(
          0, n, body,
          (jnp.full((TILE,), mask_value, jnp.float32),
           jnp.zeros((TILE,), jnp.float32)))
      if correction:
        @pl.when(n > 1)
        def _():  # observe the consumption of the last published alpha
          plgpu.barrier_wait(alpha_consumed.at[t])
      if schedule and t == 0:
        @pl.when(n > 0)
        def _():  # observe tile 1's final hand-off
          plgpu.barrier_wait(turn.at[0])

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

    if correction:
      @pl.when(wg == CORRECTION_WG)
      def _correction_wg():
        plgpu.set_max_registers(correction_registers, action="decrease")

        @pl.loop(1, jnp.maximum(n, 1))
        def _(s):
          for t in range(NUM_TILES):
            with jax.named_scope(f"corr_wait{t}"):
              plgpu.barrier_wait(o_free.at[t])
              plgpu.barrier_wait(alpha_ready.at[t])
            with jax.named_scope(f"corr_rescale{t}"):
              alpha = plgpu.load(alpha_smem.at[t], layout=_ROWS)
              plgpu.barrier_arrive(alpha_consumed.at[t])
              for c0 in range(0, head_dim, _CORRECTION_CHUNK):
                chunk = o_tmem.at[t, :, pl.ds(c0, _CORRECTION_CHUNK)]
                o = plgpu.async_load_tmem(chunk)
                plgpu.wait_load_tmem()
                plgpu.async_store_tmem(
                    chunk, o * lax.broadcast_in_dim(alpha, o.shape, [0]))
              plgpu.commit_tmem()
              plgpu.barrier_arrive(o_corrected.at[t])

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
      plgpu.SMEM((NUM_TILES, TILE), jnp.float32) if correction else None,
      plgpu.Barrier(num_barriers=NUM_TILES) if correction else None,
      plgpu.Barrier(num_barriers=NUM_TILES) if correction else None,
      plgpu.Barrier(num_barriers=NUM_TILES, orders_tensor_core=True)
      if correction else None,
      plgpu.Barrier(num_barriers=NUM_TILES, orders_tensor_core=True)
      if correction else None,
      plgpu.Barrier(num_barriers=NUM_TILES) if schedule else None,
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
      num_threads=NUM_TILES + 1 + int(correction),
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
          approx_math=True,
          **_profile_params(),
      ),
      interpret=interpret,
  )(*inputs)
  return (outs[0], outs[1]) if save_residuals else outs[0]
