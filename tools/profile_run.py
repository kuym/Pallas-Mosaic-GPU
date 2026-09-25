# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Profile one ping-pong forward config and summarize time per phase.

  SPLASH_PROFILE_DIR is set automatically.  Usage:
  python tools/profile_run.py MASK SEQ HEAD_DIM BLOCK_KV [NUM_STAGES] [SPACE]

Warp roles (TraceScope.WARP, 12 warps per CTA): 0-3 softmax tile 0,
4-7 softmax tile 1, 8 TMA, 9 MMA, 10-11 idle.
"""
import collections, glob, json, os, sys, tempfile, time

mask_name, seq, d, bkv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
stages = int(sys.argv[5]) if len(sys.argv) > 5 else 2
out_dir = tempfile.mkdtemp(prefix="splash-prof-")
os.environ["SPLASH_PROFILE_DIR"] = out_dir
if len(sys.argv) > 6:
  os.environ["SPLASH_PROFILE_SPACE"] = sys.argv[6]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp
import splash_attention_mgpu as sa

mask = {"full": sa.FullMask, "causal": sa.CausalMask}[mask_name]((seq, seq))
kern = sa.make_splash_mha(mask, block_sizes=sa.BlockSizes(block_q=256, block_kv=bkv, num_stages=stages))
q, k, v = (jax.random.normal(jax.random.key(i), (2, seq, d), jnp.bfloat16) for i in range(3))
jax.block_until_ready(kern(q, k, v))
time.sleep(2)
files = sorted(glob.glob(os.path.join(out_dir, "*trace.json")))
trace = json.load(open(files[-1]))["traceEvents"]
roles = {**{w: "softmax0" for w in range(4)}, **{w: "softmax1" for w in range(4, 8)}, 8: "tma", 9: "mma"}
# Pair B/E per (tid, name) and sum durations per (role, name).
tot = collections.defaultdict(float); cnt = collections.Counter(); open_ = {}
span = collections.defaultdict(lambda: [float("inf"), float("-inf")])
for e in trace:
  warp = (e["tid"] - 1) % 12
  role = roles.get(warp, f"w{warp}")
  key = (e["tid"], e["name"])
  sp = span[e["tid"]]; sp[0] = min(sp[0], e["ts"]); sp[1] = max(sp[1], e["ts"])
  if e["ph"] == "B":
    open_[key] = e["ts"]
  elif key in open_:
    tot[(role, e["name"])] += e["ts"] - open_.pop(key); cnt[(role, e["name"])] += 1
# Per-warp traced wall time, averaged over warps in the role.
role_span = collections.defaultdict(list)
for tid, (a, b) in span.items():
  role_span[roles.get((tid - 1) % 12, "?")].append(b - a)
print(f"profile {mask_name} S={seq} D={d} bkv={bkv} stages={stages} ({len(trace)} events)")
for role in ("softmax0", "softmax1", "mma", "tma"):
  n_warps = max(1, len(role_span[role]))
  wall = sum(role_span[role]) / n_warps
  print(f"  {role}: traced wall {wall:.2f} us per warp")
  for (r, name), t in sorted(tot.items(), key=lambda x: -x[1]):
    if r == role:
      print(f"    {name:14s} {t / n_warps:8.2f} us  ({100 * t / n_warps / max(wall, 1e-9):5.1f}%)  x{cnt[(r, name)] // n_warps}")
