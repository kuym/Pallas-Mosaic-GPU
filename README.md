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

**Status (2026-09-25).** The forward and backward kernels run correctly on
real B200s. They were tested on a Together AI 8×B200 node: the native test
suite passes, and 1,020 randomized fuzz cases (415 of them with gradients)
produced no failures (a later run over the ping-pong kernel found one API bug,
which is now fixed). The best forward throughput is 1,105 TFLOP/s at
head_dim 128, which is 70% of cuDNN's flash attention on the same node; see
[Performance](#performance-on-b200) and the
[optimization log](#optimization-log-b200).

## Features

| | |
|---|---|
| Masks | Every `splash_attention_mask` type (`CausalMask`, `LocalMask`, `ChunkedCausalMask`, `FullMask`, `NumpyMask`, `MultiHeadMask`), reusing the upstream mask library and block classification |
| Sparsity | Empty blocks are never loaded or computed; full blocks skip masking; partial blocks are masked with the in-kernel mask function or with the stored dense block |
| Attention variants | MHA, GQA, MQA, segment ids, logit soft-capping, optional batch dim |
| Outputs | Output and optional logsumexp residual (`save_residuals=True`) |
| Gradients | `custom_vjp`. For head_dim 128, a fused single-kernel backward (dQ via TMA reduce-add). Otherwise, dQ and dK/dV kernels split like the TPU module |
| Dtypes | bf16, f16 (f32 accumulation) |
| Head dims | Forward: multiples of 64 up to 256 (`block_kv=64` for 256). Backward: up to 128 |
| Kernels | `block_q=128` (default): one softmax warpgroup per CTA. `block_q=256`: experimental two-tile "ping-pong" kernel (FlashAttention-4 style), head_dim 64/128 |

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

**Two-tile ping-pong kernel** (`forward_pingpong.py`, `BlockSizes(block_q=256)`).
Each CTA holds 256 query rows as two 128-row tiles over the same K/V blocks.
Each tile has its own softmax warpgroup, plus one producer warpgroup. The MMA
warp interleaves the tiles: `PV0(j) S0(j+1) | PV1(j) S1(j+1)`. The tensor core
therefore works on one tile while the other tile's softmax runs. P is stored
in TMEM over S (`RefUnion`), so two tiles × (S/P + O) fit in the 512 TMEM
columns at head_dim 128. The sparse schedule is built at 256-row granularity.

**Fused backward** (`backward_fused.py`, head_dim 128, the default there).
One CTA owns a KV block and loops over the q sub-blocks of every q head in the
group. It computes Sᵀ and dPᵀ once per step. The MMA warp issues
Sᵀ = K·Qᵀ, dPᵀ = V·dOᵀ, dV += Pᵀ·dO, dK += dSᵀ·Q and dQᵀ = Kᵀ·dSᵀ: 5 MMAs,
where the split backward needs 7. Two elementwise warpgroups compute Pᵀ and dSᵀ,
each owning a column slice. A dQ-writer warpgroup moves dQᵀ from TMEM to SMEM
and TMA reduce-adds it into an f32 `[B, H, D, Sq]` accumulator, a
`jax.new_ref` operand that XLA transposes once at the end.

**Split backward kernels** (`backward.py`). The same warp specialization is used.
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

All three tiers pass. Tier 3 ran on a Together AI 8×B200 node.

### Running on a multi-GPU host

Expensive multi-GPU machines are driven by `tools/gpu_farm.py`, a priority
job queue that keeps every GPU busy:

- One job runs per GPU (`CUDA_VISIBLE_DEVICES`), and a GPU starts its next
  job as soon as it finishes one. Jobs can be submitted while the farm runs.
- Idle GPUs run the correctness fuzzer, `tools/fuzz.py`, which checks random
  masks, shapes, GQA/MQA, segments, soft-cap and dtypes, forward and
  gradients, against the reference.
- Fuzzer jobs are pre-empted as soon as real work is queued.
- Test suites are sharded one shard per GPU (`FARM_SHARD=i/n`).
- `tools/sweep.py` submits tile-size sweeps, and `tools/summarize.py` reports
  the best configuration per problem.
- Every job runs under a hard timeout, so a hung kernel cannot hold a GPU.

`tools/multi_gpu_launch.sh` brings up a host. For a Together Kubernetes
cluster, `tools/k8s/run.sh {kubeconfig,up,sync,status,submit,fetch,down}`
drives the same thing through one pod that holds all 8 GPUs.

## Performance on B200

These are bf16 TFLOP/s with the default (automatic) configuration, measured
with CUDA events on a Together AI 8×B200 node (driver 610, CUDA 13.3, JAX
0.11). Splash TFLOP/s count only the visible (computed) 128×128 blocks, and
fwd+bwd counts 3.5× the forward FLOPs. The cuDNN baseline, `tools/baseline.py`,
is `jax.nn.dot_product_attention(implementation="cudnn")`; the percentages in
parentheses are Splash as a share of cuDNN. Every configuration with S ≤ 4K is
also checked against the dense reference.

| Mask | S | D | heads (q/kv) | Splash fwd | cuDNN fwd | Splash fwd+bwd | cuDNN fwd+bwd |
|---|---|---|---|---|---|---|---|
| causal | 16384 | 128 | 16/16 | 960 | 1398 (69%) | 875 | 1317 (66%) |
| causal | 16384 | 128 | 32/8 | 976 | 1461 (67%) | 900 | 1291 (70%) |
| causal | 16384 | 64 | 16/16 | 545 | 946 (58%) | 425 | 890 (48%) |
| causal | 16384 | 64 | 32/8 | 552 | 964 (57%) | 432 | 887 (49%) |
| causal | 4096 | 128 | 16/16 | 773 | 922 (84%) | 735 | 975 (75%) |
| causal | 4096 | 128 | 32/8 | 807 | 1130 (71%) | 759 | 1016 (75%) |
| causal | 4096 | 64 | 16/16 | 455 | 696 (65%) | 360 | 679 (53%) |
| causal | 4096 | 64 | 32/8 | 481 | 734 (65%) | 375 | 702 (53%) |
| chunked2k | 16384 | 128 | 16/16 | 643 | — | 525 | — |
| chunked2k | 16384 | 128 | 32/8 | 672 | — | 643 | — |
| chunked2k | 16384 | 64 | 16/16 | 264 | — | 260 | — |
| chunked2k | 16384 | 64 | 32/8 | 266 | — | 274 | — |
| chunked2k | 4096 | 128 | 16/16 | 589 | — | 560 | — |
| chunked2k | 4096 | 128 | 32/8 | 616 | — | 632 | — |
| chunked2k | 4096 | 64 | 16/16 | 255 | — | 243 | — |
| chunked2k | 4096 | 64 | 32/8 | 266 | — | 269 | — |
| full | 16384 | 128 | 16/16 | 1080 | 1571 (69%) | 891 | 1317 (68%) |
| full | 16384 | 128 | 32/8 | 1084 | 1574 (69%) | 911 | 1299 (70%) |
| full | 16384 | 64 | 16/16 | 620 | 999 (62%) | 487 | 919 (53%) |
| full | 16384 | 64 | 32/8 | 624 | 1008 (62%) | 489 | 903 (54%) |
| full | 4096 | 128 | 16/16 | 823 | 1319 (62%) | 736 | 1141 (64%) |
| full | 4096 | 128 | 32/8 | 879 | 1396 (63%) | 756 | 1239 (61%) |
| full | 4096 | 64 | 16/16 | 495 | 753 (66%) | 399 | 730 (55%) |
| full | 4096 | 64 | 32/8 | 539 | 799 (68%) | 430 | 800 (54%) |
| local1k | 16384 | 128 | 16/16 | 606 | — | 617 | — |
| local1k | 16384 | 128 | 32/8 | 618 | — | 667 | — |
| local1k | 16384 | 64 | 16/16 | 413 | — | 261 | — |
| local1k | 16384 | 64 | 32/8 | 436 | — | 278 | — |
| local1k | 4096 | 128 | 16/16 | 540 | — | 535 | — |
| local1k | 4096 | 128 | 32/8 | 582 | — | 561 | — |
| local1k | 4096 | 64 | 16/16 | 350 | — | 238 | — |
| local1k | 4096 | 64 | 32/8 | 389 | — | 244 | — |

For causal masks, the 128×128 accounting counts whole diagonal blocks, which
is 0.8% more FLOPs than cuDNN's S²/2 at 16K.

**Roofline.** Per score element the tensor cores do `4·d` FLOPs, at about
8,192 FLOP/clk/SM, and the special-function unit does one `exp2`, at 16/clk/SM.
At d=128 both take 0.0625 clk, so the kernel can reach about 2.2 PF only if
the two are perfectly overlapped. At d=64 the exponentials are the bottleneck,
at about 1.1 PF. cuDNN reaches about 70% of peak (1.57 PF) at d=128, which is
the practical target.

**Where the gap comes from.** The `block_q=128` kernel has a single softmax
warpgroup, and its per-block softmax takes about as long as the block's MMAs,
so the tensor core idles for part of each step. Sparse masks with short rows
(local, chunked) also pay a per-row prologue and epilogue cost.

### Two-tile ping-pong kernel (`block_q=256`) vs. `block_q=128`

This is the best forward throughput per problem (TFLOP/s) over `block_kv` ×
`num_stages`, with H=16 MHA unless noted:

| Problem | `block_q=128` | `block_q=256` | Ratio |
|---|---|---|---|
| full, S=16K, D=128 | 866 | **1,074–1,105** | 1.24–1.28× |
| causal, S=16K, D=128 | 840 | 975 | 1.16× |
| chunked-causal, S=16K, D=128 | 550 | 650 | 1.18× |
| local 1K, S=16K, D=128 | 609 | 618 | 1.01× |
| full, S=16K, D=64 | 628 | 598–627 | ~1.0× |
| local 1K, S=4K, D=64 | 348 | 262 | 0.75× |

`block_q=256` wins for head_dim 128 with long rows. `block_q=128` is better
at head_dim 64 and for narrow local windows, where the 256-row schedule
visits about 25% more KV blocks.

### Backward on B200 (fwd+bwd TFLOP/s, head_dim 128, H=16)

| Problem | two-kernel, 64-wide tiles (first version) | two-kernel, 128-wide tiles, 2 WGs | **fused** | cuDNN |
|---|---|---|---|---|
| full, S=16K | 678 | 783 | **832** | 1,317 |
| causal, S=16K | 639 | 709 | **796** | 1,317 |
| local 1K, S=16K | 477 | 447 | **568** | — |
| chunked 2K, S=16K | 476 | 437 | **557** | — |
| causal, S=4K | 555 | 586 | **668** | 975 |

## Optimization log (B200)

Each item was A/B-tested on the 8×B200 farm, on full S=16K D=128 unless
noted:

| Change | Result |
|---|---|
| Two Q tiles per CTA with ping-ponged MMAs (`block_q=256`) | **+16–28%** at D=128 (kept, `block_q=256`) |
| Register split 240/32 → 232/40 | fixed a `setmaxnreg` deadlock |
| No scalar cross-warp reduction in the ping-pong kernel | fixed wrong results and deadlocks (finding 3) |
| exp2 polynomial with `round` + `cvt` | −13% or worse: conversions use the same slow unit as `exp2` |
| exp2 polynomial without conversions (FA4 magic-number rounding) | −2 to +3.5% at best (full D=64). Column splits hurt masked blocks. Off by default |
| Softmax over 2–8 independent column slices (shorter max/sum chains) | −2% to −39%. ptxas already hides the chains |
| Correction warpgroup that rescales O off the softmax path | +3% (full D=128), −3 to −16% elsewhere. Off by default |
| Unmasked-block fast path (log2e folded into the exp2 FMA) | −1% to +5% (D=64). On by default |
| Scheduling token so the two tiles' softmax phases alternate | +2% (full), −3 to −20% elsewhere. Off by default |
| Double-buffered Sᵀ/dPᵀ in the dK/dV kernel (overlap MMAs with elementwise work) | −8% (full D=128) to +3% (short/masked). Off by default (`SPLASH_BWD_DOUBLE_BUFFER=1`) |
| Automatic `block_q` (ping-pong for D=128 when the 256-row schedule adds ≤10% work) | the best kernel per problem by default |
| Automatic forward tiles (D=64 with `block_q=128`: `block_kv=64`, 3 stages) | +25–28% forward at D=64 (full, causal, local). Chunked-causal D=64 −18%, to revisit |
| Backward: two elementwise warpgroups (column split; `loaded` barrier for packed-P aliasing) | +0–2% alone |
| Backward: 128-wide sub-blocks (fits TMEM once Pᵀ/dSᵀ alias Sᵀ/dPᵀ) | with 2 WGs: +7–15% fwd+bwd. Profile: per-step MMA↔elementwise handoff latency dominated at 64-wide steps |
| **Fused backward** (one kernel, 5 MMAs, dQ via TMA reduce-add) | **+6% (full) to +30% (local/chunked)** fwd+bwd at D=128 |
| Fused backward at D=64 (128-row steps, dQ = dS·K via transposed-SMEM A) | correct, but −10 to −30% vs the split kernels. Opt-in (`SPLASH_BWD_FUSED_D64=1`) |

Ablations, with deliberately wrong results, show where the time goes. At full
S=16K D=128 the kernel does 1,087 TFLOP/s. Without the O rescale it does 1,217.
Without `exp2` it does 1,288. With no softmax math at all it does 1,484, which
is 94% of cuDNN. So the pipeline itself can keep up, and the remaining gap is
the cost of the softmax arithmetic. At D=128 the `exp2` work on the
special-function unit (about 2,048 clk per step per SM sub-partition) equals
the MMA time. Matching cuDNN therefore requires the softmax warps to keep
MUFU.EX2 fully busy while hiding all other FP work, which in FA4 and cuDNN is
SASS-level tuning. Tools: `SPLASH_PROFILE_DIR` (per-warp trace with named
scopes, summarized by `tools/profile_run.py`) and the `SPLASH_ABLATE`,
`SPLASH_EXP_EMU_COLS`, `SPLASH_SOFTMAX_PARTS`, `SPLASH_CORRECTION`,
`SPLASH_SCHEDULE` and `SPLASH_FAST_FULL` switches.

## Hardware findings

These were found on B200. None of them shows up in the CPU interpreter:

1. **tcgen05 needs 16-bit contraction dims in multiples of 64.** Backward
   sub-blocks of 32 fail to lower (`K must be a multiple of 64`), so
   `block_kv_dq` and `block_q_dkv` must be 64 or 128.
2. **`setmaxnreg` must leave slack.** Splitting 2×240 + 32 registers across
   three warpgroups uses exactly the SM's 65,536 registers, and
   `setmaxnreg.inc` then waits forever: a deadlock at full "utilization" and
   about 245 W. 232/40 (64,512 registers) works.
3. **Cross-warp reduction scratch is shared by all warpgroups.** Mosaic GPU
   places every cross-warp reduction's SMEM scratch at the same offset. Two
   softmax warpgroups each reducing a per-row flag to a scalar clobber each
   other. This gave wrong results with dense masks and deadlocks with local
   masks, where warps disagree on a branch that contains warpgroup barriers.
   The ping-pong kernel therefore avoids scalar reductions. Per-row reductions
   in the TCGEN05 layout stay within a thread and are safe.
4. The CPU interpreter scopes an MMA's `barrier=` to that MMA alone and does
   not model the in-order execution of tcgen05 MMAs. The kernels therefore
   use explicit `tcgen05_commit_arrive` calls. The ping-pong kernel adds
   interpret-only waits that stand in for the in-order guarantee.

## Tuning

The defaults are `BlockSizes(block_q=128, block_kv=128, num_stages=2,
block_kv_dq=64, block_q_dkv=64, num_stages_bwd=2)`. The kernels validate
their SMEM (227 KiB) and TMEM (512 columns) budgets up front and raise a
`ValueError` with the numbers. Set `block_q=256` to use the ping-pong forward
kernel.

## Optimization plan

Each step is A/B-benchmarked on the farm over the full problem grid and fuzzed
before it is kept:

1. ~~Two ping-ponged Q tiles per CTA~~: done (`block_q=256`), +16–28% at D=128.
2. ~~exp2 emulation~~ and 3. ~~correction warpgroup~~: implemented. Neither
   gains in this code-generation setting; see the optimization log.
4. A persistent kernel with a heaviest-first tile scheduler and overlapped
   epilogue and prologue, for sparse masks with short rows. After that, 2-CTA
   (`M=256`) MMAs with multicast K/V.
5. Backward: double-buffered Sᵀ/dPᵀ in the dK/dV kernel, and a fused
   single-pass backward with a TMA reduce-add for dQ.
