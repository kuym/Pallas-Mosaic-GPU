# Splash attention for Blackwell (Pallas Mosaic GPU)

A port of JAX's TPU **splash attention** (block-sparse flash attention,
[`jax/experimental/pallas/ops/tpu/splash_attention`](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py))
to the **Mosaic GPU** dialect of Pallas, targeting NVIDIA **B200 (sm_100)**:
TMA, `tcgen05` MMAs and tensor memory (TMEM), with warp specialization.

```python
import splash_attention_mgpu as sa

kernel = sa.make_splash_mha(sa.CausalMask((seq, seq)))   # any TPU splash mask
out = kernel(q, k, v)                        # q: [(batch,) heads, seq, d]
out = kernel(q, k, v, sa.SegmentIds(q_ids, kv_ids))
dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)  # differentiable
out, (lse,) = kernel(q, k, v, save_residuals=True)
```

As in the TPU kernel, `q` is expected to be pre-scaled.

## Features

| | |
|---|---|
| Masks | Every `splash_attention_mask` type (`CausalMask`, `LocalMask`, `ChunkedCausalMask`, `FullMask`, `NumpyMask`, `MultiHeadMask`), reusing the upstream mask library and block classification |
| Sparsity | Empty blocks are never loaded or computed; full blocks skip masking; partial blocks are masked with the in-kernel mask function or with the stored dense block |
| Attention variants | MHA, GQA, MQA, segment ids, logit soft-capping, optional batch dim |
| Outputs | Output and optional logsumexp residual (`save_residuals=True`) |
| Gradients | `custom_vjp` with dQ and dK/dV Mosaic GPU kernels (split like the TPU module) |
| Dtypes | bf16, f16 (f32 accumulation) |
| Head dims | Forward: multiples of 64 up to 256 (`block_kv=64` for 256). Backward: up to 128 |

## Design

**Schedule.** `mask_info.process_mask` runs the upstream TPU block
classification (`shrink_grid=False`), then compacts every row of the block mask
into an explicit list of non-empty KV blocks: `num_steps[h, i]`,
`kv_block[h, i, s]`, `block_kind[h, i, s]` (partial or full) and, for dense
masks, `mask_block[h, i, s]`. The TPU kernel walks a dense grid and skips empty
blocks through scalar-prefetched `data_next`. Here each CTA owns one
(q block, head, batch) row and loops over its own list. The dK/dV kernel uses
the transposed (column-wise) schedule.

**Forward kernel** (`kernel.py`). Per CTA: 128 query rows and 2 warpgroups.

```
WG1 warp 0  TMA ─ Q once; K, V (+ mask block, KV segment ids) per step,
                  num_stages-deep SMEM ring
WG1 warp 1  MMA ─ S_{s+1} = Q K_{s+1}^T  → TMEM S buffer (s+1)%2
                  O += P_s V_s           (P read from TMEM)
WG0 softmax     ─ S_s from TMEM → soft-cap, masks → online softmax (exp2)
                  → lazily rescale O in TMEM → P_s (bf16) into TMEM
```

The two S buffers let the QK MMA of step s+1 run under the softmax of step s.
O is rescaled in TMEM only when some row's running max grows by more than
2^8 (as in FlashAttention-4). TMEM holds O (`head_dim` columns), two S
buffers and P, which is at most 512 columns.

**Backward kernels** (`backward.py`). The same warp specialization is used.
The dQ kernel recomputes S and dP = dO Vᵀ for each KV sub-block, forms
dS = P ⊙ (dP − Δ) and accumulates dQ += dS K. The dK/dV kernel works on
Sᵀ = K Qᵀ and dPᵀ = V dOᵀ for q sub-blocks of 64 rows, accumulates
dV += Pᵀ dO and dK += dSᵀ Q in TMEM, and loops over every q head of the KV
head's group, so GQA/MQA gradients are reduced in-kernel.

**Synchronization.** Each ring slot has a TMA barrier (producer to consumer)
and a release barrier (tcgen05 commit, or a softmax arrive for data the
softmax reads). Generic-proxy SMEM reads are fenced (`commit_smem`) before the
slot is released for a TMA overwrite. Every waiter observes every barrier
phase, and no barrier can run more than one phase ahead of a waiter.

## Verification

```bash
python -m venv .venv && .venv/bin/pip install jax pytest pytest-xdist absl-py
PYTHONPATH=. .venv/bin/python -m pytest tests -n 8
```

The test suite runs at three levels:

1. **Mosaic GPU interpreter on CPU** with data race detection. This checks
   numerics against a dense f32 reference, and gradients against `jax.grad` of
   the reference, for all mask types and features. The race detector checks
   the barrier protocol of the warp-specialized pipelines, including TMA vs.
   generic proxy ordering and TMEM load/store ordering.
2. **Lowering for sm_100** (`tools/lower_check.py`). This runs on any machine,
   including one without a GPU. It covers Pallas → Mosaic GPU MLIR, layout
   inference, dialect lowering, and the SMEM and TMEM allocation checks for
   every configuration.
3. **Native on Blackwell.** These tests are skipped unless a compute
   capability 10.x GPU is present.

On a B200 (for example a Lambda Cloud `gpu_1x_b200_sxm6`):

```bash
scp -r . ubuntu@<ip>:splash
ssh ubuntu@<ip> 'bash splash/tools/lambda_setup.sh'   # tests + tools/bench.py
```

Tiers 1 and 2 pass. Tier 3 (native execution and benchmarks) has not been run
yet, because no B200 was available while this was written.

## Tuning

`BlockSizes(block_kv=128, num_stages=2, block_kv_dq=64, block_q_dkv=64,
num_stages_bwd=2)`. The kernel validates its SMEM (227 KiB) and TMEM
(512 columns) budgets up front and raises a `ValueError` with the numbers.
`tools/bench.py` sweeps masks, sequence lengths and head dims, and reports
TFLOP/s over the visible blocks.

Possible next steps once hardware numbers exist: two ping-ponged Q tiles per
CTA (FlashAttention-4 style), partial exp2 emulation on the FMA units,
double-buffered S in the backward kernels, and a persistent tile scheduler.
