# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Test sharding for multi-GPU hosts.

With FARM_SHARD="i/n" only every n-th collected test (offset i) runs, so a
suite can be split into n jobs, one per GPU (see tools/gpu_farm.py).
"""

import os


def pytest_collection_modifyitems(config, items):
  shard = os.environ.get("FARM_SHARD")
  if not shard:
    return
  index, count = map(int, shard.split("/"))
  keep = [item for i, item in enumerate(items) if i % count == index]
  deselected = [item for i, item in enumerate(items) if i % count != index]
  items[:] = keep
  config.hook.pytest_deselected(items=deselected)
