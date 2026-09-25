# Copyright 2026, Kuy Mainwaring (github.com/kuym). Licensed under the Apache 2.0 License.
"""Lower a Pallas Mosaic GPU function for sm_100a on a machine with no GPU.

Runs the Python half of the Mosaic GPU pipeline (Pallas -> Mosaic GPU MLIR,
layout inference, dialect lowering) so that most kernel-construction errors
surface without access to Blackwell hardware. PTX generation and ptxas still
happen only on a real GPU.
"""
import contextlib

import jax
from jax.experimental.mosaic.gpu import utils as mgpu_utils


@contextlib.contextmanager
def pretend_arch(major: int = 10, minor: int = 0):
  orig = mgpu_utils._infer_arch
  mgpu_utils._infer_arch = lambda: (major, minor)
  try:
    yield
  finally:
    mgpu_utils._infer_arch = orig


def lower_for_blackwell(f, *args, **kwargs):
  """Returns the StableHLO text containing the Mosaic GPU custom call."""
  with pretend_arch():
    lowered = jax.jit(f).trace(*args, **kwargs).lower(
        lowering_platforms=("cuda",))
  return lowered.as_text()
