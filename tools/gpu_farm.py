#!/usr/bin/env python3
# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Keep every GPU of a multi-GPU host busy with a priority queue of jobs.

  gpu_farm.py serve  --root farm --gpus 0-7 [--filler "python tools/fuzz.py --minutes 10"]
  gpu_farm.py submit --root farm [--priority 0] [--name tests] -- CMD ARGS...
  gpu_farm.py status --root farm

`serve` runs one job per GPU (CUDA_VISIBLE_DEVICES=<gpu>) and starts the next
queued job the moment a GPU frees up: lowest priority number first, then
submission order.  Jobs are files in <root>/queue, so work can be submitted
from anywhere (e.g. over ssh from a laptop) while the farm runs.  When the
queue is empty, idle GPUs run the optional `--filler` command (a bounded,
lowest-priority job such as a correctness fuzzer), so no GPU sits idle while
new work is being prepared.  Fillers are pre-emptible: as soon as real work is
queued and no GPU is free, a filler is stopped (SIGTERM to its process group)
to make room, so fillers never delay real work.

Each job gets FARM_GPU, FARM_JOB_ID and FARM_RESULTS (a shared JSON-lines file)
in its environment; its output goes to <root>/logs/<job>.log and its exit code
and timing to <root>/done/<job>.json.  Stdlib only.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time


def _dirs(root):
  d = {name: os.path.join(root, name)
       for name in ("queue", "running", "done", "logs")}
  for path in d.values():
    os.makedirs(path, exist_ok=True)
  return d


def _gpus(spec):
  out = []
  for part in spec.split(","):
    if "-" in part:
      lo, hi = map(int, part.split("-"))
      out.extend(range(lo, hi + 1))
    else:
      out.append(int(part))
  return out


def submit(root, cmd, priority=10, name=None, cwd=None):
  d = _dirs(root)
  seq = time.time_ns()
  job_id = f"p{priority:03d}-{seq}-{(name or 'job').replace('/', '_')[:40]}"
  job = dict(id=job_id, cmd=cmd, priority=priority, name=name or "job",
             cwd=cwd or os.getcwd(), submitted=time.time())
  tmp = os.path.join(d["queue"], f".{job_id}.tmp")
  with open(tmp, "w") as f:
    json.dump(job, f)
  os.rename(tmp, os.path.join(d["queue"], f"{job_id}.json"))  # atomic
  return job_id


def _kill(proc, sig=signal.SIGTERM):
  try:
    os.killpg(proc.pid, sig)  # the job's whole process group
  except ProcessLookupError:
    pass


def _queued(d):
  return any(n.endswith(".json") for n in os.listdir(d["queue"]))


def _next_job(d):
  names = sorted(n for n in os.listdir(d["queue"]) if n.endswith(".json"))
  for n in names:  # ids sort by (priority, submission time)
    src = os.path.join(d["queue"], n)
    dst = os.path.join(d["running"], n)
    try:
      os.rename(src, dst)  # claim it
    except FileNotFoundError:
      continue
    with open(dst) as f:
      return json.load(f), dst
  return None, None


def serve(root, gpus, filler=None, poll=1.0, results=None):
  d = _dirs(root)
  results = results or os.path.join(root, "results.jsonl")
  running = {}  # gpu -> (Popen, job, path, start, log file)
  stopping = False

  def stop(*_):
    nonlocal stopping
    stopping = True

  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)
  # Jobs left in running/ by a previous (killed) farm go back to the queue.
  for n in os.listdir(d["running"]):
    os.rename(os.path.join(d["running"], n), os.path.join(d["queue"], n))
  print(f"farm: serving GPUs {gpus} from {root}", flush=True)

  while True:
    # Reap finished jobs.
    for gpu, (proc, job, path, start, log) in list(running.items()):
      rc = proc.poll()
      if rc is None:
        continue
      log.close()
      job.update(gpu=gpu, rc=rc, start=start, end=time.time(),
                 seconds=round(time.time() - start, 1))
      with open(os.path.join(d["done"], f"{job['id']}.json"), "w") as f:
        json.dump(job, f)
      if path:
        os.remove(path)
      print(f"farm: gpu{gpu} finished {job['id']} rc={rc} "
            f"({job['seconds']}s)", flush=True)
      del running[gpu]
    if stopping:
      for proc, *_ in running.values():
        _kill(proc)
      break
    # Pre-empt one filler if real work is waiting and every GPU is busy.
    if len(running) == len(gpus) and _queued(d):
      fillers = [(s, g) for g, (_, j, _, s, _) in running.items()
                 if j.get("priority", 0) >= 999 and not j.get("preempted")]
      if fillers:
        _, gpu = max(fillers)  # the youngest filler loses the least work
        proc, job = running[gpu][0], running[gpu][1]
        job["preempted"] = True
        _kill(proc)
        print(f"farm: gpu{gpu} pre-empting {job['id']}", flush=True)
    # Start new jobs on idle GPUs; real work first, filler only if none.
    for gpu in gpus:
      if gpu in running:
        continue
      job, path = _next_job(d)
      if job is None:
        if not filler:
          continue
        job = dict(id=f"filler-{time.time_ns()}-gpu{gpu}", cmd=filler,
                   priority=999, name="filler", cwd=os.getcwd())
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), FARM_GPU=str(gpu),
                 FARM_JOB_ID=job["id"], FARM_RESULTS=os.path.abspath(results))
      log = open(os.path.join(d["logs"], f"{job['id']}.log"), "w")
      cmd = job["cmd"]
      proc = subprocess.Popen(
          cmd if isinstance(cmd, list) else ["bash", "-lc", cmd],
          cwd=job.get("cwd"), env=env, stdout=log, stderr=subprocess.STDOUT,
          start_new_session=True)
      running[gpu] = (proc, job, path, time.time(), log)
      print(f"farm: gpu{gpu} started {job['id']}", flush=True)
    _write_status(root, running)
    time.sleep(poll)
  _write_status(root, running)


def _write_status(root, running):
  now = time.time()
  status = dict(time=now, running={
      str(g): dict(id=j["id"], seconds=round(now - s))
      for g, (_, j, _, s, _) in running.items()})
  tmp = os.path.join(root, ".status.tmp")
  with open(tmp, "w") as f:
    json.dump(status, f)
  os.replace(tmp, os.path.join(root, "status.json"))


def status(root):
  d = _dirs(root)
  try:
    with open(os.path.join(root, "status.json")) as f:
      st = json.load(f)
    print(f"status age {time.time() - st['time']:.0f}s")
    for g, r in sorted(st["running"].items(), key=lambda x: int(x[0])):
      print(f"  gpu{g}: {r['id']} ({r['seconds']}s)")
  except FileNotFoundError:
    print("no status yet")
  queued = sorted(n for n in os.listdir(d["queue"]) if n.endswith(".json"))
  done = [json.load(open(os.path.join(d["done"], n)))
          for n in os.listdir(d["done"])]
  real = [j for j in done if j.get("priority", 0) < 999]
  failed = [j for j in done if j.get("rc") and not j.get("preempted")]
  print(f"queued {len(queued)} | done {len(real)} jobs "
        f"(+{len(done) - len(real)} filler) | failed {len(failed)}")
  for j in sorted(failed, key=lambda j: j.get("end", 0))[-10:]:
    print(f"  FAILED rc={j['rc']} {j['id']} (logs/{j['id']}.log)")
  for n in queued[:10]:
    print(f"  queued {n[:-5]}")


def wait(root, pattern, timeout=3600, poll=5.0):
  """Blocks until no queued/running job name contains `pattern` and at least
  one such job is done; prints the matching jobs' exit codes."""
  import fnmatch
  d = _dirs(root)
  deadline = time.time() + timeout
  match = lambda n: fnmatch.fnmatch(n, f"*{pattern}*")
  while time.time() < deadline:
    pending = [n for sub in ("queue", "running") for n in os.listdir(d[sub])
               if n.endswith(".json") and match(n)]
    done = [n for n in os.listdir(d["done"]) if match(n)]
    if not pending and done:
      for n in sorted(done):
        j = json.load(open(os.path.join(d["done"], n)))
        print(f"{j['name']}: rc={j['rc']} ({j['seconds']}s)")
      return 0
    time.sleep(poll)
  print(f"timeout waiting for {pattern}")
  return 1


def main():
  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("serve")
  s.add_argument("--root", default="farm")
  s.add_argument("--gpus", default="0-7")
  s.add_argument("--filler")
  s.add_argument("--results")
  s = sub.add_parser("submit")
  s.add_argument("--root", default="farm")
  s.add_argument("--priority", type=int, default=10)
  s.add_argument("--name")
  s.add_argument("command", nargs=argparse.REMAINDER)
  s = sub.add_parser("status")
  s.add_argument("--root", default="farm")
  s = sub.add_parser("wait")
  s.add_argument("--root", default="farm")
  s.add_argument("--timeout", type=float, default=3600)
  s.add_argument("pattern")
  args = p.parse_args()
  if args.cmd == "serve":
    serve(args.root, _gpus(args.gpus), args.filler, results=args.results)
  elif args.cmd == "submit":
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
      sys.exit("submit: missing command")
    print(submit(args.root, shlex.join(cmd), args.priority, args.name))
  elif args.cmd == "wait":
    sys.exit(wait(args.root, args.pattern, args.timeout))
  else:
    status(args.root)


if __name__ == "__main__":
  main()
