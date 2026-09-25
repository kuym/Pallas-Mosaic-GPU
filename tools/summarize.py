# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Summarize gpu_farm results: best tile sizes per problem, and failures.

  python tools/summarize.py farm/results.jsonl
"""

import collections
import json
import sys


def main(path):
  recs = [json.loads(line) for line in open(path) if line.strip()]
  bench = [r for r in recs if r.get("kind") != "fuzz"]
  fuzz = [r for r in recs if r.get("kind") == "fuzz"]

  best = collections.defaultdict(dict)
  errors = collections.Counter()
  for r in bench:
    if "error" in r:
      errors[r["error"].split(":")[0] + ": " + r["error"].split(":", 1)[-1][:80]] += 1
      continue
    prob = (r["mask"], r["seq"], r["head_dim"], r["heads"], r.get("kv_heads"))
    for metric in ("fwd_tflops", "fwd_bwd_tflops"):
      if metric in r and (metric not in best[prob]
                          or r[metric] > best[prob][metric][0]):
        best[prob][metric] = (r[metric], r["block_sizes"],
                              r.get("fwd_us") if metric == "fwd_tflops"
                              else r.get("fwd_bwd_us"))
  print(f"{len(bench)} benchmark records, {len(fuzz)} fuzz cases\n")
  print("best configurations (TFLOP/s over visible blocks):")
  for prob in sorted(best):
    for metric, (tf, bs, us) in sorted(best[prob].items()):
      keys = (("block_q", "block_kv", "num_stages") if metric == "fwd_tflops" else
              ("block_kv_dq", "block_q_dkv", "num_stages_bwd"))
      print(f"  {str(prob):42s} {metric:15s} {tf:7.1f} ({us:9.1f} us) "
            + " ".join(f"{k}={bs.get(k, 128)}" for k in keys))
  if errors:
    print("\nbenchmark errors:")
    for e, n in errors.most_common(15):
      print(f"  {n:4d} x {e}")
  fails = [r for r in fuzz if not r["ok"]]
  print(f"\nfuzz: {len(fuzz) - len(fails)} passed, {len(fails)} failed")
  for r in fails[:10]:
    print(f"  seed={r['case']['seed']} {r.get('error', r.get('errs'))}"[:300])


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "farm/results.jsonl")
