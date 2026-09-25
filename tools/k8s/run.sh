#!/usr/bin/env bash
# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
# Drive tools/multi_gpu_launch.sh on a Together Kubernetes GPU cluster.
#
#   tools/k8s/run.sh kubeconfig CLUSTER_NAME   # write ./kubeconfig from the API
#   tools/k8s/run.sh up                        # start pod, upload code, launch
#   tools/k8s/run.sh sync                      # re-upload code (keeps the farm)
#   tools/k8s/run.sh status                    # farm status + best configs
#   tools/k8s/run.sh fetch                     # copy farm results to ./farm-remote
#   tools/k8s/run.sh submit PRIORITY NAME CMD… # queue a job on the farm
#   tools/k8s/run.sh down                      # delete the pod (NOT the cluster)
#
# Needs TOGETHER_API_KEY (or ~/.together/api_key) for `kubeconfig`.
set -euo pipefail
cd "$(dirname "$0")/../.."
export KUBECONFIG="${KUBECONFIG:-$PWD/kubeconfig}"
POD=splash-farm
K=(kubectl)
in_pod() { "${K[@]}" exec "$POD" -- bash -lc "cd /splash && $*"; }

upload() {
  git archive --format=tar HEAD | "${K[@]}" exec -i "$POD" -- \
      bash -c 'mkdir -p /splash && tar -x -C /splash'
  # Uncommitted edits too, so experiments need no commit.
  git diff HEAD --binary | "${K[@]}" exec -i "$POD" -- \
      bash -c 'cd /splash && (git apply --allow-empty - 2>/dev/null || patch -p1 -s)' || true
}

case "${1:-}" in
  kubeconfig)
    # $2 = cluster name.  The list endpoint omits the kubeconfig; the
    # per-cluster endpoint returns it base64-encoded.
    key="${TOGETHER_API_KEY:-$(cat ~/.together/api_key)}"
    api=https://api.together.ai/v1/compute/clusters
    id=$(curl -fsS -m 60 -H "Authorization: Bearer $key" "$api" |
      python3 -c 'import sys, json
print([c for c in json.load(sys.stdin)["clusters"] if c["cluster_name"] == sys.argv[1]][0]["cluster_id"])' "$2")
    (umask 077; curl -fsS -m 60 -H "Authorization: Bearer $key" "$api/$id" |
      python3 -c 'import sys, json, base64
c = json.load(sys.stdin); c = c.get("cluster", c)
assert c.get("kube_config"), "no kubeconfig yet (status %s)" % c["status"]
sys.stdout.write(base64.b64decode(c["kube_config"]).decode())' > kubeconfig)
    "${K[@]}" get nodes
    ;;
  up)
    "${K[@]}" apply -f tools/k8s/splash-farm.yaml
    "${K[@]}" wait --for=condition=Ready "pod/$POD" --timeout=20m
    "${K[@]}" exec "$POD" -- bash -c \
        'apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip procps patch >/dev/null && nvidia-smi -L'
    upload
    in_pod "bash tools/multi_gpu_launch.sh"
    ;;
  sync) upload ;;
  status)
    in_pod ". .venv-gpu/bin/activate && python tools/gpu_farm.py status --root farm && \
            (test -f farm/results.jsonl && python tools/summarize.py farm/results.jsonl | tail -40 || true); \
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader"
    ;;
  fetch)
    mkdir -p farm-remote
    "${K[@]}" exec "$POD" -- tar -C /splash -c farm cpu_tests.log farm.log | tar -x -C farm-remote
    echo "results in farm-remote/"
    ;;
  submit)
    shift; prio=$1; name=$2; shift 2
    in_pod ". .venv-gpu/bin/activate && python tools/gpu_farm.py submit --root farm --priority $prio --name $name -- $*"
    ;;
  down) "${K[@]}" delete pod "$POD" --wait=true ;;
  *) sed -n '3,13p' "$0"; exit 2 ;;
esac
