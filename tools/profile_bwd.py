# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Profile the backward kernels (dQ and dK/dV) of one config; summarize.

  python tools/profile_bwd.py MASK SEQ HEAD_DIM [SPACE]

Warp roles (TraceScope.WARP): with ne elementwise warpgroups, warps
0..4*ne-1 are elementwise, then TMA (4*ne) and MMA (4*ne+1).
"""
import collections, glob, json, os, sys, tempfile, time

mask_name, seq, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out_dir = tempfile.mkdtemp(prefix="splash-prof-bwd-")
os.environ["SPLASH_PROFILE_DIR"] = out_dir
os.environ.setdefault("SPLASH_PROFILE_SPACE", sys.argv[4] if len(sys.argv) > 4 else "256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp
import splash_attention_mgpu as sa

ne = int(os.environ.get("SPLASH_BWD_WGS", 2))
mask = {"full": sa.FullMask, "causal": sa.CausalMask}[mask_name]((seq, seq))
kern = sa.make_splash_mha(mask, block_sizes=sa.BlockSizes(block_q=128))
q, k, v = (jax.random.normal(jax.random.key(i), (2, seq, d), jnp.bfloat16) for i in range(3))
g = jax.grad(lambda q, k, v: kern(q, k, v).astype(jnp.float32).sum(), argnums=(0, 1, 2))
jax.block_until_ready(g(q, k, v))
time.sleep(3)
for f in sorted(glob.glob(os.path.join(out_dir, "*trace.json"))):
  trace = json.load(open(f))["traceEvents"]
  names = {e["name"] for e in trace}
  if "dq_wait" in names:
    kind = "fused"
    warps = 4 * (ne + 2)
    roles = {**{w: f"ew{w // 4}" for w in range(4 * ne)},
             **{w: "dq_writer" for w in range(4 * ne, 4 * ne + 4)},
             4 * ne + 4: "tma", 4 * ne + 5: "mma"}
  else:
    kind = "dkv" if "mma_wait_q" in names else ("dq" if "mma_wait_ds" in names else "fwd")
    warps = 4 * (ne + 1)
    roles = {**{w: f"ew{w // 4}" for w in range(4 * ne)}, 4 * ne: "tma", 4 * ne + 1: "mma"}
  tot = collections.defaultdict(float); cnt = collections.Counter(); open_ = {}
  span = collections.defaultdict(lambda: [float("inf"), float("-inf")])
  for e in trace:
    warp = (e["tid"] - 1) % warps
    role = roles.get(warp, f"w{warp}")
    key = (e["tid"], e["name"])
    sp = span[e["tid"]]; sp[0] = min(sp[0], e["ts"]); sp[1] = max(sp[1], e["ts"])
    if e["ph"] == "B":
      open_[key] = e["ts"]
    elif key in open_:
      tot[(role, e["name"])] += e["ts"] - open_.pop(key); cnt[(role, e["name"])] += 1
  role_span = collections.defaultdict(list)
  for tid, (a, b) in span.items():
    role_span[roles.get((tid - 1) % warps, "?")].append(b - a)
  print(f"== {kind} kernel ({os.path.basename(f)}, {len(trace)} events)")
  for role in sorted(role_span):
    n_warps = max(1, len(role_span[role]))
    wall = sum(role_span[role]) / n_warps
    print(f"  {role}: traced wall {wall:.2f} us per warp")
    for (r, name), t in sorted(tot.items(), key=lambda x: -x[1]):
      if r == role:
        print(f"    {name:14s} {t / n_warps:8.2f} us ({100 * t / n_warps / max(wall, 1e-9):5.1f}%) x{cnt[(r, name)] // n_warps}")
