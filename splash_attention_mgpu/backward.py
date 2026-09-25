# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Backward pass of Mosaic GPU splash attention (Blackwell, sm_100).

Like the TPU module, the backward pass is split in two kernels that recompute
the attention probabilities from the saved logsumexp:

dQ kernel   one CTA per (q block, head, batch), walking the forward schedule
            (non-empty KV blocks of the row).  Per KV sub-block:
              S = Q K^T, dP = dO V^T         (tcgen05, TMEM)
              P = exp(S - lse), dS = P (dP - delta)   (softmax warpgroup)
              dQ += dS K                     (tcgen05, dS read from TMEM)

dKV kernel  one CTA per (KV block, KV head, batch), walking the transposed
            schedule (the q blocks that see this KV block) for every q head
            of the KV head's group.  Per q sub-block:
              S^T = K Q^T, dP^T = V dO^T
              P^T = exp(S^T - lse), dS^T = P^T (dP^T - delta)
              dV += P^T dO,  dK += dS^T Q    (P^T, dS^T read from TMEM)

Both kernels use the same warp specialization as the forward kernel: warp 0
of warpgroup 1 issues TMA, warp 1 issues tcgen05 MMAs and warpgroup 0 does
the elementwise work.  The sub-block sizes keep all accumulators in the 512
TMEM columns for head dims up to 128.
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

from . import forward_pingpong
from . import mask_info as mask_info_lib
from .kernel import (
    LOG2E,
    SMEM_BYTES,
    TMEM_COLS,
    SegmentIds,
    _swizzle_transforms,
    _where,
)

_ROWS = plgpu.Layout.TCGEN05.reduce(1)  # one value per TMEM lane (row)
_COLS = plgpu.Layout.TCGEN05.reduce(0)  # one value per column


def _check_budgets(name, tmem_cols, smem_bytes):
  if tmem_cols > TMEM_COLS:
    raise ValueError(
        f"{name}: TMEM budget exceeded ({tmem_cols} > {TMEM_COLS} columns)."
        " The backward pass supports head dims up to 128."
    )
  if smem_bytes > SMEM_BYTES:
    raise ValueError(
        f"{name}: shared memory budget exceeded ({smem_bytes} > {SMEM_BYTES}"
        " bytes); reduce num_stages or the sub-block sizes."
    )


def _probs_and_dlogits(
    logits, dp, lse, delta, *, masks, soft_cap, mask_value, lse_dims
):
  """Recomputes P and returns (P, dS) for one tile.

  `lse` and `delta` are per-query vectors broadcast along `lse_dims`.  Masked
  logits are set to `mask_value` in natural units before subtracting the
  logsumexp, so fully masked rows stay finite exactly as in the forward pass.
  """
  if soft_cap is not None:
    t = jnp.tanh(logits / soft_cap)
    logits = t * soft_cap
  logits = masks(logits)
  bcast = lambda x: lax.broadcast_in_dim(x, logits.shape, lse_dims)
  p = jnp.exp2((logits - bcast(lse)) * LOG2E)
  ds = p * (dp - bcast(delta))
  if soft_cap is not None:
    ds = ds * (1.0 - t * t)
  return p, ds


def splash_attention_bwd_dq(
    q, k, v, do, lse, delta,
    segment_ids: SegmentIds | None,
    num_steps, q_block_order, kv_block, block_kind, mask_block,
    partial_mask_blocks,
    *,
    mask_function,
    block_kv: int,
    block_kv_compute: int,
    num_stages: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    interpret: Any = None,
):
  """dQ. Shapes as in the forward kernel; lse/delta: f32[batch, heads, q]."""
  batch, num_q_heads, q_seq_len, head_dim = q.shape
  _, num_kv_heads, _, head_dim_v = v.shape
  dtype = q.dtype
  bq = 128
  bkv = block_kv  # schedule granularity
  itemsize = jnp.dtype(dtype).itemsize
  has_dense_mask = partial_mask_blocks is not None
  has_segments = segment_ids is not None
  has_aux = has_dense_mask or has_segments

  def budgets(bc):
    return dict(
        tmem_cols=2 * bc + bc // 2 + head_dim,
        smem_bytes=(
            bq * (head_dim + head_dim_v) * itemsize
            + num_stages * bc * (head_dim + head_dim_v) * itemsize
            + 2 * bq * 4
            + (num_stages * bq * bc if has_dense_mask else 0)
            + (bq + num_stages * bc) * 4 * has_segments
        ))

  # None: the largest sub-block that fits (128-wide steps amortize the
  # per-step MMA <-> elementwise handoff latency; +7-15% on B200).
  bc = block_kv_compute or _largest_fitting(bkv, budgets)
  if bkv % bc:
    raise ValueError(f"{block_kv_compute=} must divide {block_kv=}")
  sub = bkv // bc
  _check_budgets("dQ kernel", **budgets(bc))
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  mask_heads = num_steps.shape[0]
  # Elementwise warpgroups, each owning bc / ne columns (dS has its own TMEM
  # buffer, so the warpgroups never touch each other's columns).
  ne = _num_elementwise_wgs(bc)

  def kernel(*refs):
    refs = list(refs)
    q_gmem, k_gmem, v_gmem, do_gmem, lse_gmem, delta_gmem = refs[:6]
    del refs[:6]
    if has_segments:
      q_seg_gmem, kv_seg_gmem = refs[:2]
      del refs[:2]
    num_steps_gmem, q_order_gmem, kv_block_gmem, block_kind_gmem = refs[:4]
    del refs[:4]
    if has_dense_mask:
      mask_block_gmem, mask_blocks_gmem = refs[:2]
      del refs[:2]
    dq_gmem = refs.pop(0)
    (
        q_smem, do_smem, k_smem, v_smem, lse_smem, delta_smem,
        q_seg_smem, kv_seg_smem, mask_smem,
        s_tmem, dp_tmem, ds_tmem, dq_tmem,
        q_barrier, k_barriers, v_barriers, seg_barriers, mask_barriers,
        kv_consumed, aux_consumed, s_ready, ds_ready, dq_done,
    ) = refs

    h = lax.axis_index("h")
    b = lax.axis_index("b")
    wg = lax.axis_index("wg")
    mh = h if mask_heads > 1 else 0
    qi = q_order_gmem[mh, lax.axis_index("q")]  # heaviest rows first
    kv_head = lax.div(h, q_heads_per_kv_head)
    n = num_steps_gmem[mh, qi] * sub
    q_slice = pl.ds(qi * bq, bq)

    def step_block(t):
      """(kv block index, first kv row, sub-block index) of step t."""
      blk = kv_block_gmem[mh, qi, lax.div(t, sub)]
      part = lax.rem(t, sub)
      return blk, blk * bkv + part * bc, part

    @pl.when(wg == ne)
    def _producer_wg():
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id == 0)
        def _tma_warp():
          plgpu.copy_gmem_to_smem(q_gmem.at[b, h, q_slice], q_smem, q_barrier)
          plgpu.copy_gmem_to_smem(do_gmem.at[b, h, q_slice], do_smem,
                                  q_barrier)
          plgpu.copy_gmem_to_smem(lse_gmem.at[b, h, q_slice], lse_smem,
                                  q_barrier)
          plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, q_slice], delta_smem,
                                  q_barrier)
          if has_segments:
            plgpu.copy_gmem_to_smem(q_seg_gmem.at[b, q_slice], q_seg_smem,
                                    q_barrier)

          @pl.loop(0, n)
          def _kv_loop(t):
            slot = lax.rem(t, num_stages)

            @pl.when(t >= num_stages)
            def _():
              plgpu.barrier_wait(kv_consumed.at[slot])
              if has_aux:
                plgpu.barrier_wait(aux_consumed.at[slot])

            _, row0, part = step_block(t)
            kv_slice = pl.ds(row0, bc)
            plgpu.copy_gmem_to_smem(k_gmem.at[b, kv_head, kv_slice],
                                    k_smem.at[slot], k_barriers.at[slot])
            plgpu.copy_gmem_to_smem(v_gmem.at[b, kv_head, kv_slice],
                                    v_smem.at[slot], v_barriers.at[slot])
            if has_segments:
              plgpu.copy_gmem_to_smem(kv_seg_gmem.at[b, kv_slice],
                                      kv_seg_smem.at[slot],
                                      seg_barriers.at[slot])
            if has_dense_mask:
              s_idx = lax.div(t, sub)

              @pl.when(block_kind_gmem[mh, qi, s_idx] == mask_info_lib.PARTIAL)
              def _():
                plgpu.copy_gmem_to_smem(
                    mask_blocks_gmem.at[mask_block_gmem[mh, qi, s_idx], :,
                                        pl.ds(part * bc, bc)],
                    mask_smem.at[slot],
                    mask_barriers.at[slot],
                )

          @pl.loop(jnp.maximum(n - num_stages, 0), n)
          def _drain(t):
            plgpu.barrier_wait(kv_consumed.at[lax.rem(t, num_stages)])
            if has_aux:
              plgpu.barrier_wait(aux_consumed.at[lax.rem(t, num_stages)])

        @pl.when(warp_id == 1)
        def _mma_warp():
          plgpu.barrier_wait(q_barrier)

          @pl.loop(0, n)
          def _mma_loop(t):
            slot = lax.rem(t, num_stages)
            with jax.named_scope("mma_wait_kv"):
              plgpu.barrier_wait(k_barriers.at[slot])
              plgpu.barrier_wait(v_barriers.at[slot])
            # S/dP of step t-1 were consumed before ds_ready(t-1), and the
            # dQ MMA of step t-1 is ordered before these by the tensor core.
            plgpu.tcgen05_mma(s_tmem, q_smem, k_smem.at[slot].T,
                              accumulate=False)
            plgpu.tcgen05_mma(dp_tmem, do_smem, v_smem.at[slot].T,
                              accumulate=False)
            plgpu.tcgen05_commit_arrive(s_ready)
            with jax.named_scope("mma_wait_ds"):
              plgpu.barrier_wait(ds_ready)
            plgpu.tcgen05_mma(dq_tmem, ds_tmem, k_smem.at[slot],
                              kv_consumed.at[slot], accumulate=t > 0)

          @pl.when(n > 0)
          def _():
            plgpu.tcgen05_commit_arrive(dq_done)

    def elementwise_wg(e):  # e: static index; owns columns [e*w, (e+1)*w)
      w = bc // ne
      cols = pl.ds(e * w, w)
      plgpu.barrier_wait(q_barrier)
      lse_rows = plgpu.load(lse_smem, layout=_ROWS)
      delta_rows = plgpu.load(delta_smem, layout=_ROWS)
      if has_segments:
        q_ids = plgpu.load(q_seg_smem, layout=_ROWS)

      @pl.loop(0, n)
      def _softmax_loop(t):
        slot = lax.rem(t, num_stages)
        _, row0, _ = step_block(t)
        row0 = row0 + e * w
        is_partial = (
            block_kind_gmem[mh, qi, lax.div(t, sub)] == mask_info_lib.PARTIAL
        )

        def masks(x):
          if has_dense_mask:
            def load_mask():
              plgpu.barrier_wait(mask_barriers.at[slot])
              m = plgpu.load(mask_smem.at[slot, :, cols],
                             layout=plgpu.Layout.TCGEN05)
              return _where(m != 0, x, mask_value)
            x = lax.cond(is_partial, load_mask, lambda: x)
          elif mask_function is not None:
            def compute_mask():
              q_pos = qi * bq + plgpu.broadcasted_iota(
                  jnp.int32, (bq, w), 0, layout=plgpu.Layout.TCGEN05)
              kv_pos = row0 + plgpu.broadcasted_iota(
                  jnp.int32, (bq, w), 1, layout=plgpu.Layout.TCGEN05)
              return _where(mask_function(q_pos, kv_pos), x, mask_value)
            x = lax.cond(is_partial, compute_mask, lambda: x)
          if has_segments:
            plgpu.barrier_wait(seg_barriers.at[slot])
            kv_ids = plgpu.load(kv_seg_smem.at[slot, cols], layout=_COLS)
            same = (lax.broadcast_in_dim(q_ids, (bq, w), [0])
                    == lax.broadcast_in_dim(kv_ids, (bq, w), [1]))
            x = _where(same, x, mask_value)
          return x

        with jax.named_scope("ew_wait_s"):
          plgpu.barrier_wait(s_ready)
        with jax.named_scope("ew_load"):
          logits = plgpu.async_load_tmem(s_tmem.at[:, cols])
          dp = plgpu.async_load_tmem(dp_tmem.at[:, cols])
          plgpu.wait_load_tmem()
        with jax.named_scope("ew_math"):
          _, ds = _probs_and_dlogits(
              logits, dp, lse_rows, delta_rows, masks=masks,
              soft_cap=attn_logits_soft_cap, mask_value=mask_value,
              lse_dims=[0],
          )
        with jax.named_scope("ew_store"):
          plgpu.async_store_tmem(ds_tmem.at[:, cols], ds.astype(dtype))
        plgpu.commit_tmem()
        if has_aux:
          plgpu.commit_smem()  # Fence generic reads before TMA overwrites.
          plgpu.barrier_arrive(aux_consumed.at[slot])
        plgpu.barrier_arrive(ds_ready)

      if e != 0:
        return

      def read_dq():
        plgpu.barrier_wait(dq_done)
        dq = plgpu.async_load_tmem(dq_tmem)
        plgpu.wait_load_tmem()
        return dq.astype(dtype)

      dq = lax.cond(n > 0, read_dq,
                    lambda: jnp.zeros((bq, head_dim), dtype))
      q_smem[...] = dq  # All MMAs reading q_smem are done.
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_gmem.at[b, h, q_slice])
      plgpu.wait_smem_to_gmem(0)

    for e in range(ne):
      pl.when(wg == e)(functools.partial(elementwise_wg, e))

  qk_t = _swizzle_transforms(head_dim, dtype)
  v_t = _swizzle_transforms(head_dim_v, dtype)
  scratch_types = [
      plgpu.SMEM((bq, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((bq, head_dim_v), dtype, transforms=v_t),
      plgpu.SMEM((num_stages, bc, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bc, head_dim_v), dtype, transforms=v_t),
      plgpu.SMEM((bq,), jnp.float32),
      plgpu.SMEM((bq,), jnp.float32),
      plgpu.SMEM((bq,), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bc), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bq, bc), jnp.int8) if has_dense_mask else None,
      plgpu.TMEM((bq, bc), jnp.float32),
      plgpu.TMEM((bq, bc), jnp.float32),
      plgpu.TMEM((bq, bc), dtype, packed=True),
      plgpu.TMEM((bq, head_dim), jnp.float32),
      plgpu.Barrier(num_arrivals=4 + has_segments),
      plgpu.Barrier(num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages) if has_segments else None,
      plgpu.Barrier(num_barriers=num_stages) if has_dense_mask else None,
      plgpu.Barrier(num_barriers=num_stages, orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=ne, num_barriers=num_stages)
      if has_aux else None,
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=ne, orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
  ]
  inputs = [q, k, v, do, lse, delta]
  if has_segments:
    inputs += [segment_ids.q, segment_ids.kv]
  inputs += [num_steps, q_block_order, kv_block, block_kind]
  if has_dense_mask:
    inputs += [mask_block, partial_mask_blocks]
  return _launch(
      kernel, scratch_types, inputs,
      out_type=jax.ShapeDtypeStruct(q.shape, dtype),
      grid=(q_seq_len // bq, num_q_heads, batch),
      num_threads=ne + 1,
      interpret=interpret,
  )


def splash_attention_bwd_dkv(
    q, k, v, do, lse, delta,
    segment_ids: SegmentIds | None,
    num_steps, kv_block_order, q_block, block_kind, mask_block,
    partial_mask_blocks_t,
    *,
    mask_function,
    block_kv: int,
    block_q_compute: int,
    num_stages: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    interpret: Any = None,
):
  """dK and dV, accumulated over all q heads sharing a KV head.

  `num_steps`, `q_block`, ... are the transposed schedule (`dkv_*` fields of
  `GpuMaskInfo`) and `partial_mask_blocks_t` the transposed mask blocks.
  """
  batch, num_q_heads, q_seq_len, head_dim = q.shape
  _, num_kv_heads, kv_seq_len, head_dim_v = v.shape
  dtype = q.dtype
  bq = 128  # schedule granularity along q
  bkv = block_kv
  if bkv != 128:
    raise NotImplementedError("The dKV kernel needs block_kv == 128.")
  itemsize = jnp.dtype(dtype).itemsize
  has_dense_mask = partial_mask_blocks_t is not None
  has_segments = segment_ids is not None
  double_buffer = os.environ.get("SPLASH_BWD_DOUBLE_BUFFER", "0") == "1"

  def num_buffers(bc):
    # Double-buffer S^T/dP^T (with P^T/dS^T aliased onto them) so the MMAs
    # of step t+1 overlap the elementwise work of step t.  Needs a second
    # SMEM stage (the prefetched step's Q/dO) and 4 * bc + dk + dv columns.
    return 2 if (double_buffer and num_stages >= 2
                 and 4 * bc + head_dim + head_dim_v <= TMEM_COLS) else 1

  def budgets(bc):
    return dict(
        tmem_cols=2 * num_buffers(bc) * bc + head_dim + head_dim_v,
        smem_bytes=(
            bkv * (head_dim + head_dim_v) * itemsize
            + num_stages * bc * ((head_dim + head_dim_v) * itemsize + 8)
            + (num_stages * bkv * bc if has_dense_mask else 0)
            + (bkv + num_stages * bc) * 4 * has_segments
        ))

  bc = block_q_compute or _largest_fitting(bq, budgets)  # q rows per step
  if bq % bc:
    raise ValueError(f"{block_q_compute=} must divide {bq}")
  sub = bq // bc
  nbuf = num_buffers(bc)
  serialize_mma = interpret is not None  # P^T aliases S^T in both modes
  # Elementwise warpgroups, each owning bc / ne columns of the score tile
  # (the dK/dV elementwise work needs no row reductions, so no cross-
  # warpgroup communication): more warps per SM sub-partition hide latency.
  ne = _num_elementwise_wgs(bc)
  _check_budgets("dKV kernel", **budgets(bc))
  group = num_q_heads // num_kv_heads
  mask_heads = num_steps.shape[0]

  def kernel(*refs):
    refs = list(refs)
    q_gmem, k_gmem, v_gmem, do_gmem, lse_gmem, delta_gmem = refs[:6]
    del refs[:6]
    if has_segments:
      q_seg_gmem, kv_seg_gmem = refs[:2]
      del refs[:2]
    num_steps_gmem, kv_order_gmem, q_block_gmem, block_kind_gmem = refs[:4]
    del refs[:4]
    if has_dense_mask:
      mask_block_gmem, mask_blocks_gmem = refs[:2]
      del refs[:2]
    dk_gmem, dv_gmem = refs[:2]
    del refs[:2]
    (
        k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem,
        kv_seg_smem, q_seg_smem, mask_smem,
        sp_unions, dd_unions, dk_tmem, dv_tmem,
        kv_barrier, q_barriers, mask_barriers,
        consumed, aux_consumed, s_ready, p_ready, done, mma_order, loaded,
    ) = refs

    def bufs(i):
      """TMEM refs of buffer i (static): (S^T, dP^T, P^T, dS^T), where P^T
      aliases S^T and dS^T aliases dP^T.  Each buffer is its own RefUnion."""
      (st, pt), (dpt, dst) = sp_unions[i], dd_unions[i]
      return st, dpt, pt, dst

    def for_buffer(t, f):
      """Runs f(i) for the static buffer index i == t % nbuf."""
      if nbuf == 1:
        f(0)
        return
      for i in range(nbuf):
        pl.when(lax.rem(t, nbuf) == i)(functools.partial(f, i))

    hk = lax.axis_index("h")
    # Heaviest KV blocks first.  With per-head masks the order of the group's
    # first q head is used.
    kj = kv_order_gmem[hk * group if mask_heads > 1 else 0,
                       lax.axis_index("kv")]
    b = lax.axis_index("b")
    wg = lax.axis_index("wg")
    kv_slice = pl.ds(kj * bkv, bkv)

    def head_steps(g):
      h = hk * group + g
      mh = h if mask_heads > 1 else 0
      return h, mh, num_steps_gmem[mh, kj] * sub

    def total_steps():
      return lax.fori_loop(0, group, lambda g, acc: acc + head_steps(g)[2],
                           jnp.int32(0))

    def for_each_step(body):
      """Runs body(t, h, mh, u) over all (q head, q sub-block) steps.

      t is the global step index, u the step within the head.
      """
      def per_head(g, base):
        h, mh, n_h = head_steps(g)

        @pl.loop(0, n_h)
        def _(u):
          body(base + u, h, mh, u)

        return base + n_h

      lax.fori_loop(0, group, per_head, jnp.int32(0))

    def step_rows(mh, u):
      """(q block index, first q row, sub-block index) of step u."""
      blk = q_block_gmem[mh, kj, lax.div(u, sub)]
      part = lax.rem(u, sub)
      return blk, blk * bq + part * bc, part

    n = total_steps()

    @pl.when(wg == ne)
    def _producer_wg():
      @plgpu.warp_map
      def _per_warp(warp_id):
        @pl.when(warp_id == 0)
        def _tma_warp():
          plgpu.copy_gmem_to_smem(k_gmem.at[b, hk, kv_slice], k_smem,
                                  kv_barrier)
          plgpu.copy_gmem_to_smem(v_gmem.at[b, hk, kv_slice], v_smem,
                                  kv_barrier)
          if has_segments:
            plgpu.copy_gmem_to_smem(kv_seg_gmem.at[b, kv_slice], kv_seg_smem,
                                    kv_barrier)

          def load(t, h, mh, u):
            slot = lax.rem(t, num_stages)

            @pl.when(t >= num_stages)
            def _():
              plgpu.barrier_wait(consumed.at[slot])
              plgpu.barrier_wait(aux_consumed.at[slot])

            _, row0, part = step_rows(mh, u)
            q_slice = pl.ds(row0, bc)
            bar = q_barriers.at[slot]
            plgpu.copy_gmem_to_smem(q_gmem.at[b, h, q_slice], q_smem.at[slot],
                                    bar)
            plgpu.copy_gmem_to_smem(do_gmem.at[b, h, q_slice],
                                    do_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(lse_gmem.at[b, h, q_slice],
                                    lse_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, q_slice],
                                    delta_smem.at[slot], bar)
            if has_segments:
              plgpu.copy_gmem_to_smem(q_seg_gmem.at[b, q_slice],
                                      q_seg_smem.at[slot], bar)
            if has_dense_mask:
              s_idx = lax.div(u, sub)

              @pl.when(block_kind_gmem[mh, kj, s_idx] == mask_info_lib.PARTIAL)
              def _():
                plgpu.copy_gmem_to_smem(
                    mask_blocks_gmem.at[mask_block_gmem[mh, kj, s_idx], :,
                                        pl.ds(part * bc, bc)],
                    mask_smem.at[slot],
                    mask_barriers.at[slot],
                )

          for_each_step(load)

          @pl.loop(jnp.maximum(n - num_stages, 0), n)
          def _drain(t):
            plgpu.barrier_wait(consumed.at[lax.rem(t, num_stages)])
            plgpu.barrier_wait(aux_consumed.at[lax.rem(t, num_stages)])

        @pl.when(warp_id == 1)
        def _mma_warp():
          plgpu.barrier_wait(kv_barrier)

          def issue_scores(t):
            for_buffer(t, functools.partial(issue_scores_into, t))

          def issue_scores_into(t, i):
            slot = lax.rem(t, num_stages)
            st, dpt, _, _ = bufs(i)
            if serialize_mma:
              # Interpreter only (it does not model in-order tcgen05
              # execution): the dV/dK MMAs of step t-2 read this buffer.
              @pl.when(t >= nbuf)
              def _():
                plgpu.barrier_wait(mma_order.at[lax.rem(t, nbuf)])
            with jax.named_scope("mma_wait_q"):
              plgpu.barrier_wait(q_barriers.at[slot])
            plgpu.tcgen05_mma(st, k_smem, q_smem.at[slot].T, accumulate=False)
            plgpu.tcgen05_mma(dpt, v_smem, do_smem.at[slot].T,
                              accumulate=False)
            plgpu.tcgen05_commit_arrive(s_ready.at[lax.rem(t, nbuf)])

          def mma(t, h, mh, u):
            del h, mh, u
            slot = lax.rem(t, num_stages)
            if nbuf == 1:
              issue_scores(t)
            else:
              @pl.when(t + 1 < n)
              def _():  # scores of the next step overlap this step's softmax
                issue_scores(t + 1)
            with jax.named_scope("mma_wait_p"):
              plgpu.barrier_wait(p_ready.at[lax.rem(t, nbuf)])

            def grads(i):
              _, _, pt, dst = bufs(i)
              plgpu.tcgen05_mma(dv_tmem, pt, do_smem.at[slot],
                                accumulate=t > 0)
              plgpu.tcgen05_mma(dk_tmem, dst, q_smem.at[slot],
                                accumulate=t > 0)
            for_buffer(t, grads)
            # Releases the slot once both the dV and dK MMAs have read it.
            plgpu.tcgen05_commit_arrive(consumed.at[slot])
            if serialize_mma:
              plgpu.tcgen05_commit_arrive(mma_order.at[lax.rem(t, nbuf)])

          if nbuf == 2:
            @pl.when(n > 0)
            def _():
              issue_scores(jnp.int32(0))
          for_each_step(mma)
          if serialize_mma:
            # Observe the last dV/dK completion of each buffer.
            @pl.loop(jnp.maximum(n - nbuf, 0), n)
            def _(t):
              plgpu.barrier_wait(mma_order.at[lax.rem(t, nbuf)])

          @pl.when(n > 0)
          def _():
            plgpu.tcgen05_commit_arrive(done)

    def elementwise_wg(e):  # e: static index; owns columns [e*w, (e+1)*w)
      w = bc // ne
      cols = pl.ds(e * w, w)
      plgpu.barrier_wait(kv_barrier)
      if has_segments:
        kv_ids = plgpu.load(kv_seg_smem, layout=_ROWS)

      def softmax(t, h, mh, u):
        del h
        slot = lax.rem(t, num_stages)
        _, row0, _ = step_rows(mh, u)
        row0 = row0 + e * w
        is_partial = (
            block_kind_gmem[mh, kj, lax.div(u, sub)] == mask_info_lib.PARTIAL
        )

        def masks(x):  # x: [kv rows, q cols]
          if has_dense_mask:
            def load_mask():
              plgpu.barrier_wait(mask_barriers.at[slot])
              m = plgpu.load(mask_smem.at[slot, :, cols],
                             layout=plgpu.Layout.TCGEN05)
              return _where(m != 0, x, mask_value)
            x = lax.cond(is_partial, load_mask, lambda: x)
          elif mask_function is not None:
            def compute_mask():
              kv_pos = kj * bkv + plgpu.broadcasted_iota(
                  jnp.int32, (bkv, w), 0, layout=plgpu.Layout.TCGEN05)
              q_pos = row0 + plgpu.broadcasted_iota(
                  jnp.int32, (bkv, w), 1, layout=plgpu.Layout.TCGEN05)
              return _where(mask_function(q_pos, kv_pos), x, mask_value)
            x = lax.cond(is_partial, compute_mask, lambda: x)
          if has_segments:
            q_ids = plgpu.load(q_seg_smem.at[slot, cols], layout=_COLS)
            same = (lax.broadcast_in_dim(kv_ids, (bkv, w), [0])
                    == lax.broadcast_in_dim(q_ids, (bkv, w), [1]))
            x = _where(same, x, mask_value)
          return x

        with jax.named_scope("ew_wait_q"):
          plgpu.barrier_wait(q_barriers.at[slot])  # lse / delta / q seg ids
        lse = plgpu.load(lse_smem.at[slot, cols], layout=_COLS)
        delta = plgpu.load(delta_smem.at[slot, cols], layout=_COLS)
        with jax.named_scope("ew_wait_s"):
          plgpu.barrier_wait(s_ready.at[lax.rem(t, nbuf)])
        for_buffer(t, functools.partial(elementwise, t, slot, masks, lse, delta))
        plgpu.commit_smem()  # Fence generic reads before TMA overwrites.
        plgpu.barrier_arrive(aux_consumed.at[slot])
        plgpu.barrier_arrive(p_ready.at[lax.rem(t, nbuf)])

      def elementwise(t, slot, masks, lse, delta, i):
        del t, slot
        st, dpt, pt, dst = (r.at[:, cols] for r in bufs(i))
        with jax.named_scope("ew_load"):
          logits_t = plgpu.async_load_tmem(st)
          dp_t = plgpu.async_load_tmem(dpt)
          plgpu.wait_load_tmem()  # P^T / dS^T overwrite these columns below
        if ne > 1:
          # Packed bf16 P^T/dS^T use half the columns of S^T/dP^T, so a
          # warpgroup's P^T lands in columns another warpgroup reads S^T
          # from: every warpgroup must have loaded before any stores.
          plgpu.barrier_arrive(loaded.at[i])
          plgpu.barrier_wait(loaded.at[i])
        with jax.named_scope("ew_math"):
          p_t, ds_t = _probs_and_dlogits(
              logits_t, dp_t, lse, delta, masks=masks,
              soft_cap=attn_logits_soft_cap, mask_value=mask_value,
              lse_dims=[1],
          )
        with jax.named_scope("ew_store"):
          plgpu.async_store_tmem(pt, p_t.astype(dtype))
          plgpu.async_store_tmem(dst, ds_t.astype(dtype))
          plgpu.commit_tmem()

      for_each_step(softmax)

      def read(ref, d):
        def f():
          x = plgpu.async_load_tmem(ref)
          plgpu.wait_load_tmem()
          return x.astype(dtype)
        return lax.cond(n > 0, f, lambda: jnp.zeros((bkv, d), dtype))

      @pl.when(n > 0)
      def _():
        plgpu.barrier_wait(done)

      # All MMAs reading k_smem / v_smem are done: reuse them for the output.
      # With two elementwise warpgroups one writes dK and the other dV.
      if e == 0:
        k_smem[...] = read(dk_tmem, head_dim)
      if e == ne - 1:
        v_smem[...] = read(dv_tmem, head_dim_v)
      plgpu.commit_smem()
      if e == 0:
        plgpu.copy_smem_to_gmem(k_smem, dk_gmem.at[b, hk, kv_slice])
      if e == ne - 1:
        plgpu.copy_smem_to_gmem(v_smem, dv_gmem.at[b, hk, kv_slice])
      plgpu.wait_smem_to_gmem(0)

    for e in range(ne):
      pl.when(wg == e)(functools.partial(elementwise_wg, e))

  qk_t = _swizzle_transforms(head_dim, dtype)
  v_t = _swizzle_transforms(head_dim_v, dtype)
  scratch_types = [
      plgpu.SMEM((bkv, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((bkv, head_dim_v), dtype, transforms=v_t),
      plgpu.SMEM((num_stages, bc, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bc, head_dim_v), dtype, transforms=v_t),
      plgpu.SMEM((num_stages, bc), jnp.float32),
      plgpu.SMEM((num_stages, bc), jnp.float32),
      plgpu.SMEM((bkv,), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bc), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bkv, bc), jnp.int8) if has_dense_mask else None,
      [plgpu.RefUnion(plgpu.TMEM((bkv, bc), jnp.float32),
                      plgpu.TMEM((bkv, bc), dtype, packed=True))
       for _ in range(nbuf)],
      [plgpu.RefUnion(plgpu.TMEM((bkv, bc), jnp.float32),
                      plgpu.TMEM((bkv, bc), dtype, packed=True))
       for _ in range(nbuf)],
      plgpu.TMEM((bkv, head_dim), jnp.float32),
      plgpu.TMEM((bkv, head_dim_v), jnp.float32),
      plgpu.Barrier(num_arrivals=2 + has_segments),
      plgpu.Barrier(num_arrivals=4 + has_segments, num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages) if has_dense_mask else None,
      plgpu.Barrier(num_barriers=num_stages, orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=ne, num_barriers=num_stages),
      plgpu.Barrier(num_barriers=nbuf, orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=ne, num_barriers=nbuf,
                    orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(num_barriers=nbuf, orders_tensor_core=True)
      if serialize_mma else None,
      plgpu.Barrier(num_arrivals=ne, num_barriers=nbuf, orders_tensor_core=True)
      if ne > 1 else None,
  ]
  inputs = [q, k, v, do, lse, delta]
  if has_segments:
    inputs += [segment_ids.q, segment_ids.kv]
  inputs += [num_steps, kv_block_order, q_block, block_kind]
  if has_dense_mask:
    inputs += [mask_block, partial_mask_blocks_t]
  return _launch(
      kernel, scratch_types, inputs,
      out_type=(jax.ShapeDtypeStruct(k.shape, dtype),
                jax.ShapeDtypeStruct(v.shape, dtype)),
      grid=(kv_seq_len // bkv, num_kv_heads, batch),
      grid_names=("kv", "h", "b"),
      num_threads=ne + 1,
      interpret=interpret,
  )


def _largest_fitting(limit, budgets):
  for bc in (128, 64):
    b = budgets(bc)
    if bc <= limit and b["tmem_cols"] <= TMEM_COLS and b["smem_bytes"] <= SMEM_BYTES:
      return bc
  return 64  # _check_budgets reports the problem


def _num_elementwise_wgs(bc):
  ne = int(os.environ.get("SPLASH_BWD_WGS", 2))
  while ne > 1 and (bc // ne) % 16:
    ne //= 2  # each warpgroup needs a multiple of 16 columns
  return ne


def _launch(kernel, scratch_types, inputs, *, out_type, grid,
            grid_names=("q", "h", "b"), num_threads=2, interpret):
  def entry(*refs):
    present = [t for t in scratch_types if t is not None]

    def unflatten(*scratch):
      it = iter(scratch)
      kernel(*refs, *[None if t is None else next(it) for t in scratch_types])

    pl.run_scoped(unflatten, *present, collective_axes="wg")

  return plgpu.kernel(
      entry,
      out_type=out_type,
      grid=grid,
      grid_names=grid_names,
      num_threads=num_threads,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
          approx_math=True,
          **forward_pingpong._profile_params(),
      ),
      interpret=interpret,
  )(*inputs)
