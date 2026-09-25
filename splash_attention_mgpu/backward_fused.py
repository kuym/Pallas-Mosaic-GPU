# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Fused single-kernel backward pass (FA3/FA4 style) for head_dim 128.

The split backward (backward.py) recomputes P in both the dQ and the dK/dV
kernels: 7 MMAs and two exp passes per block pair.  This kernel owns a KV
block and walks the transposed schedule over every q head of the KV head's
group, computing P and dS once per (q sub-block, KV block):

  MMA warp   S^T = K Q^T, dP^T = V dO^T
             dV += P^T dO, dK += dS^T Q, dQ^T = K^T dS^T          (5 MMAs)
  WG 0..ne-1 P^T = exp(S^T - lse), dS^T = P^T (dP^T - delta); P^T and dS^T
             into TMEM (aliasing S^T/dP^T), dS^T also into SMEM (B operand
             of the dQ^T MMA); each warpgroup owns a column slice.
  WG ne      dQ writer: dQ^T (TMEM) -> SMEM -> TMA reduce-add into an f32
             accumulator laid out [batch, heads, head_dim, q] (so no
             transpose is needed in-kernel); XLA transposes and casts once.

TMEM: S^T/P^T (bc) + dP^T/dS^T (bc) + dK + dV + dQ(^T) columns = 448 at
head_dim 128 (bc = 64, dQ^T) and at head_dim 64 (bc = 128, dQ = dS K).
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
from .backward import (
    _COLS,
    _ROWS,
    _check_budgets,
    _launch,
    _num_elementwise_wgs,
    _probs_and_dlogits,
)
from .kernel import SMEM_BYTES, SegmentIds, _swizzle_transforms, _where


def fused_supported(head_dim, head_dim_v, block_kv=128):
  return head_dim == head_dim_v and head_dim in (64, 128) and block_kv == 128


def fused_smem_bytes(*, head_dim, num_stages, itemsize, has_dense_mask,
                     has_segments, block_kv=128):
  bc = 64 if head_dim == 128 else 128
  return (block_kv * 2 * head_dim * itemsize
          + num_stages * bc * (2 * head_dim * itemsize + 8)
          + (num_stages * block_kv * bc if has_dense_mask else 0)
          + (block_kv + num_stages * bc) * 4 * has_segments
          + 2 * block_kv * bc * itemsize  # dS^T for the dQ MMA (2 buffers)
          + head_dim * bc * 4)  # dQ(^T) staging


def fused_fits(**kwargs) -> bool:
  return fused_smem_bytes(**kwargs) <= SMEM_BYTES


def splash_attention_bwd_fused(
    q, k, v, do, lse, delta,
    segment_ids: SegmentIds | None,
    num_steps, kv_block_order, q_block, block_kind, mask_block,
    partial_mask_blocks_t,
    *,
    mask_function,
    block_kv: int,
    num_stages: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    interpret: Any = None,
):
  """Returns (dq, dk, dv).  Schedule arguments as in splash_attention_bwd_dkv."""
  batch, num_q_heads, q_seq_len, head_dim = q.shape
  _, num_kv_heads, kv_seq_len, head_dim_v = v.shape
  dtype = q.dtype
  bq = 128
  bkv = block_kv
  if not fused_supported(head_dim, head_dim_v, bkv):
    raise NotImplementedError(
        "fused backward needs head_dim == head_dim_v in (64, 128), block_kv 128")
  # dQ formulation.  head_dim 128: 64-row q steps and dQ^T = K^T dS^T (M =
  # head_dim; accumulator [B, H, D, Sq]).  head_dim 64: M = 64 MMAs are not
  # usable here, so 128-row q steps and dQ = dS K with A = dS^T in SMEM read
  # transposed (M = 128; accumulator [B, H, Sq, D]).
  dq_transposed = head_dim == 128
  bc = 64 if dq_transposed else 128  # q rows per step
  sub = bq // bc
  itemsize = jnp.dtype(dtype).itemsize
  has_dense_mask = partial_mask_blocks_t is not None
  has_segments = segment_ids is not None
  serialize_mma = interpret is not None
  ne = _num_elementwise_wgs(bc)
  w = bc // ne
  # Issue the next step's S^T/dP^T before this step's dQ MMA; then the dQ MMA
  # of step t may still read dS^T while the elementwise warpgroups write the
  # next one, so dS^T is double-buffered in SMEM.
  early_scores = os.environ.get("SPLASH_BWD_EARLY_S", "1") == "1"
  ds_bufs = 2 if early_scores else 1
  _check_budgets(
      "fused backward kernel",
      tmem_cols=2 * bc + head_dim + head_dim_v + (bc if dq_transposed
                                                    else head_dim),
      smem_bytes=fused_smem_bytes(
          head_dim=head_dim, num_stages=num_stages, itemsize=itemsize,
          has_dense_mask=has_dense_mask, has_segments=has_segments,
          block_kv=bkv),
  )
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
    dq_acc = refs.pop(0)  # f32 [batch, q heads, head_dim, q] accumulator Ref
    dk_gmem, dv_gmem = refs[:2]
    del refs[:2]
    (
        k_smem, v_smem, q_smem, do_smem, lse_smem, delta_smem,
        kv_seg_smem, q_seg_smem, mask_smem, ds_smem, dq_stage, dq_stage2,
        sp_union, dd_union, dk_tmem, dv_tmem, dqt_tmem,
        kv_barrier, q_barriers, mask_barriers,
        consumed, aux_consumed, s_ready, p_ready, done, mma_order, loaded,
        dq_ready, dq_free, dq_load_barrier,
    ) = refs
    st_tmem, pt_tmem = sp_union
    dpt_tmem, dst_tmem = dd_union

    hk = lax.axis_index("h")
    b = lax.axis_index("b")
    wg = lax.axis_index("wg")
    kj = kv_order_gmem[hk * group if mask_heads > 1 else 0,
                       lax.axis_index("kv")]
    kv_slice = pl.ds(kj * bkv, bkv)

    def head_steps(g):
      h = hk * group + g
      mh = h if mask_heads > 1 else 0
      return h, mh, num_steps_gmem[mh, kj] * sub

    n = lax.fori_loop(0, group, lambda g, acc: acc + head_steps(g)[2],
                      jnp.int32(0))

    def for_each_step(body):
      def per_head(g, base):
        h, mh, n_h = head_steps(g)

        @pl.loop(0, n_h)
        def _(u):
          body(base + u, h, mh, u)

        return base + n_h

      lax.fori_loop(0, group, per_head, jnp.int32(0))

    def step_rows(mh, u):
      blk = q_block_gmem[mh, kj, lax.div(u, sub)]
      part = lax.rem(u, sub)
      return blk, blk * bq + part * bc, part

    @pl.when(wg == ne + 1)
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
            rows = pl.ds(row0, bc)
            bar = q_barriers.at[slot]
            plgpu.copy_gmem_to_smem(q_gmem.at[b, h, rows], q_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(do_gmem.at[b, h, rows], do_smem.at[slot],
                                    bar)
            plgpu.copy_gmem_to_smem(lse_gmem.at[b, h, rows], lse_smem.at[slot],
                                    bar)
            plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, rows],
                                    delta_smem.at[slot], bar)
            if has_segments:
              plgpu.copy_gmem_to_smem(q_seg_gmem.at[b, rows],
                                      q_seg_smem.at[slot], bar)
            if has_dense_mask:
              s_idx = lax.div(u, sub)

              @pl.when(block_kind_gmem[mh, kj, s_idx] == mask_info_lib.PARTIAL)
              def _():
                plgpu.copy_gmem_to_smem(
                    mask_blocks_gmem.at[mask_block_gmem[mh, kj, s_idx], :,
                                        pl.ds(part * bc, bc)],
                    mask_smem.at[slot], mask_barriers.at[slot])

          for_each_step(load)

          @pl.loop(jnp.maximum(n - num_stages, 0), n)
          def _drain(t):
            plgpu.barrier_wait(consumed.at[lax.rem(t, num_stages)])
            plgpu.barrier_wait(aux_consumed.at[lax.rem(t, num_stages)])

        @pl.when(warp_id == 1)
        def _mma_warp():
          plgpu.barrier_wait(kv_barrier)

          def issue_scores(t):  # S^T(t), dP^T(t)
            slot = lax.rem(t, num_stages)
            if serialize_mma:
              # Interpreter only (no model of in-order tcgen05 execution):
              # the dV/dK MMAs of step t-1 read the P^T/dS^T that this
              # step's S^T/dP^T overwrite.
              @pl.when(t > 0)
              def _():
                plgpu.barrier_wait(mma_order)
            with jax.named_scope("mma_wait_q"):
              plgpu.barrier_wait(q_barriers.at[slot])
            plgpu.tcgen05_mma(st_tmem, k_smem, q_smem.at[slot].T,
                              accumulate=False)
            plgpu.tcgen05_mma(dpt_tmem, v_smem, do_smem.at[slot].T,
                              accumulate=False)
            plgpu.tcgen05_commit_arrive(s_ready)

          if early_scores:
            @pl.when(n > 0)
            def _():
              issue_scores(jnp.int32(0))

          def mma(t, h, mh, u):
            del h, mh, u
            slot = lax.rem(t, num_stages)
            if not early_scores:
              issue_scores(t)
            with jax.named_scope("mma_wait_p"):
              plgpu.barrier_wait(p_ready)
            plgpu.tcgen05_mma(dv_tmem, pt_tmem, do_smem.at[slot],
                              accumulate=t > 0)
            plgpu.tcgen05_mma(dk_tmem, dst_tmem, q_smem.at[slot],
                              accumulate=t > 0)
            if serialize_mma:
              plgpu.tcgen05_commit_arrive(mma_order)
            if early_scores:
              # The next step's scores go ahead of this step's dQ MMA (which
              # reads dS^T from SMEM, not the TMEM the scores overwrite), so
              # the elementwise warpgroups get S sooner.
              @pl.when(t + 1 < n)
              def _():
                issue_scores(t + 1)

            @pl.when(t > 0)
            def _():  # the dQ writer has read dQ^T(t-1)
              with jax.named_scope("mma_wait_dq_free"):
                plgpu.barrier_wait(dq_free)
            ds_buf = ds_smem.at[lax.rem(t, ds_bufs)]
            if dq_transposed:
              for e in range(ne):  # dQ^T[:, e*w:(e+1)*w] = K^T dS^T_e
                plgpu.tcgen05_mma(dqt_tmem.at[:, pl.ds(e * w, w)], k_smem.T,
                                  ds_buf.at[e], accumulate=False)
            else:  # dQ = dS K, dS read as the transpose of dS^T
              plgpu.tcgen05_mma(dqt_tmem, ds_buf.T, k_smem, accumulate=False)
            plgpu.tcgen05_commit_arrive(dq_ready)
            # Releases the slot once every MMA reading Q / dO has completed.
            plgpu.tcgen05_commit_arrive(consumed.at[slot])

          for_each_step(mma)

          @pl.when(n > 0)
          def _():
            plgpu.tcgen05_commit_arrive(done)
            plgpu.barrier_wait(dq_free)  # observe the last dQ^T read-out
            if serialize_mma:
              plgpu.barrier_wait(mma_order)

    def elementwise_wg(e):
      cols = pl.ds(e * w, w)
      plgpu.barrier_wait(kv_barrier)
      if has_segments:
        kv_ids = plgpu.load(kv_seg_smem, layout=_ROWS)

      def step(t, h, mh, u):
        del h
        slot = lax.rem(t, num_stages)  # t is also used for the dS^T buffer
        _, row0, _ = step_rows(mh, u)
        row0 = row0 + e * w
        is_partial = (
            block_kind_gmem[mh, kj, lax.div(u, sub)] == mask_info_lib.PARTIAL)

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
          plgpu.barrier_wait(q_barriers.at[slot])
        lse_c = plgpu.load(lse_smem.at[slot, cols], layout=_COLS)
        delta_c = plgpu.load(delta_smem.at[slot, cols], layout=_COLS)
        with jax.named_scope("ew_wait_s"):
          plgpu.barrier_wait(s_ready)
        with jax.named_scope("ew_load"):
          logits_t = plgpu.async_load_tmem(st_tmem.at[:, cols])
          dp_t = plgpu.async_load_tmem(dpt_tmem.at[:, cols])
          plgpu.wait_load_tmem()
        if ne > 1:
          # Packed P^T/dS^T use half the columns: another warpgroup's stores
          # land where this one reads, so all loads precede any store.
          plgpu.barrier_arrive(loaded)
          plgpu.barrier_wait(loaded)
        with jax.named_scope("ew_math"):
          p_t, ds_t = _probs_and_dlogits(
              logits_t, dp_t, lse_c, delta_c, masks=masks,
              soft_cap=attn_logits_soft_cap, mask_value=mask_value,
              lse_dims=[1])
        with jax.named_scope("ew_store"):
          ds16 = ds_t.astype(dtype)
          plgpu.async_store_tmem(pt_tmem.at[:, cols], p_t.astype(dtype))
          plgpu.async_store_tmem(dst_tmem.at[:, cols], ds16)
          ds_buf = ds_smem.at[lax.rem(t, ds_bufs)]
          if dq_transposed:
            ds_buf[e] = ds16  # B operand of the dQ^T MMA
          else:
            ds_buf[:, cols] = ds16  # A operand (transposed) of the dQ MMA
          plgpu.commit_tmem()
          plgpu.commit_smem()  # dS^T for the async proxy; fences our reads
        plgpu.barrier_arrive(aux_consumed.at[slot])
        plgpu.barrier_arrive(p_ready)

      for_each_step(step)

      def read(ref, d):
        def f():
          x = plgpu.async_load_tmem(ref)
          plgpu.wait_load_tmem()
          return x.astype(dtype)
        return lax.cond(n > 0, f, lambda: jnp.zeros((bkv, d), dtype))

      @pl.when(n > 0)
      def _():
        plgpu.barrier_wait(done)

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

    @pl.when(wg == ne)
    def _dq_writer_wg():
      def write(t, h, mh, u):
        del t
        _, row0, _ = step_rows(mh, u)
        acc = dq_acc.at[lax.axis_index("kv")] if serialize_mma else dq_acc
        if dq_transposed:
          dst = acc.at[b, h, :, pl.ds(row0, bc)]
        else:
          dst = acc.at[b, h, pl.ds(row0, bc), :]
        with jax.named_scope("dq_wait"):
          plgpu.barrier_wait(dq_ready)
        dqt = plgpu.async_load_tmem(dqt_tmem)
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(dq_free)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # dq_stage reusable
        if serialize_mma:
          # Interpreter only (it has no TMA reductions): each CTA writes its
          # partial dQ^T to a private [kv block] slice (a CTA's steps cover
          # distinct q rows), summed by XLA afterwards.
          dq_stage[...] = dqt
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(dq_stage, dst)
        else:
          dq_stage[...] = dqt
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(dq_stage, dst, reduction_op="add")

      for_each_step(write)
      plgpu.wait_smem_to_gmem(0)

  qk_t = _swizzle_transforms(head_dim, dtype)
  ds_t = _swizzle_transforms(w if dq_transposed else bc, dtype)
  scratch_types = [
      plgpu.SMEM((bkv, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((bkv, head_dim_v), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bc, head_dim), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bc, head_dim_v), dtype, transforms=qk_t),
      plgpu.SMEM((num_stages, bc), jnp.float32),
      plgpu.SMEM((num_stages, bc), jnp.float32),
      plgpu.SMEM((bkv,), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bc), jnp.int32) if has_segments else None,
      plgpu.SMEM((num_stages, bkv, bc), jnp.int8) if has_dense_mask else None,
      plgpu.SMEM((ds_bufs, ne, bkv, w) if dq_transposed
                 else (ds_bufs, bkv, bc), dtype, transforms=ds_t),
      plgpu.SMEM((head_dim, bc) if dq_transposed else (bc, head_dim),
                 jnp.float32),
      None,
      plgpu.RefUnion(plgpu.TMEM((bkv, bc), jnp.float32),
                     plgpu.TMEM((bkv, bc), dtype, packed=True)),
      plgpu.RefUnion(plgpu.TMEM((bkv, bc), jnp.float32),
                     plgpu.TMEM((bkv, bc), dtype, packed=True)),
      plgpu.TMEM((bkv, head_dim), jnp.float32),
      plgpu.TMEM((bkv, head_dim_v), jnp.float32),
      plgpu.TMEM((head_dim, bc) if dq_transposed else (bc, head_dim),
                 jnp.float32),
      plgpu.Barrier(num_arrivals=2 + has_segments),
      plgpu.Barrier(num_arrivals=4 + has_segments, num_barriers=num_stages),
      plgpu.Barrier(num_barriers=num_stages) if has_dense_mask else None,
      plgpu.Barrier(num_barriers=num_stages, orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=ne, num_barriers=num_stages),
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(num_arrivals=ne, orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True) if serialize_mma else None,
      plgpu.Barrier(num_arrivals=ne, orders_tensor_core=True)
      if ne > 1 else None,
      plgpu.Barrier(orders_tensor_core=True),
      plgpu.Barrier(orders_tensor_core=True),
      None,
  ]
  inputs = [q, k, v, do, lse, delta]
  if has_segments:
    inputs += [segment_ids.q, segment_ids.kv]
  inputs += [num_steps, kv_block_order, q_block, block_kind]
  if has_dense_mask:
    inputs += [mask_block, partial_mask_blocks_t]
  acc_shape = ((batch, num_q_heads, head_dim, q_seq_len) if dq_transposed
               else (batch, num_q_heads, q_seq_len, head_dim))
  if serialize_mma:
    acc_shape = (kv_seq_len // bkv,) + acc_shape  # interpret: per-CTA partials
  dq_acc = jax.new_ref(jnp.zeros(acc_shape, jnp.float32))
  dk, dv = _launch(
      kernel, scratch_types, inputs + [dq_acc],
      out_type=(jax.ShapeDtypeStruct(k.shape, dtype),
                jax.ShapeDtypeStruct(v.shape, dtype)),
      grid=(kv_seq_len // bkv, num_kv_heads, batch),
      grid_names=("kv", "h", "b"),
      num_threads=ne + 2,
      interpret=interpret,
  )
  dq = dq_acc[...]
  if serialize_mma:
    dq = dq.sum(axis=0)
  if dq_transposed:
    dq = jnp.swapaxes(dq, 2, 3)
  dq = dq.astype(dtype)
  return dq, dk, dv
