"""Drone graphs that exist so the training leaves of the device rollout never leave the GPU.

The numpy residency of ``hover_neat_env.py`` performs one operation on the host that the compiled
graphs of ``hover_kernels.py`` do not cover: it reshapes the ``(rows,)`` episode returns into the
policy-major ``(pop_size, episodes)`` matrix the trainer scores.  That reshape is a graph here, so
a device-resident generation hands its returns back in the shape ``env_base.validate_evaluation``
expects with no host array in between.

Nothing here imports ``genesis`` or Torch, and nothing here simulates; the module speaks only
``numpy`` and ``evoxc``.

Graphs owned here (compiled once and reused for every generation)
----------------------------------------------------------------
``hover_device_episode_returns``  ``(rows,)`` returns -> the policy-major ``(pop_size, episodes)`` matrix

``DeviceArray.reshape`` and every numpy function called on a device array return a HOST copy
(``iree.runtime`` dispatches them through numpy), so a device buffer's shape is only changed by a
graph -- that is what ``hover_device_episode_returns`` exists for, and no other reshaping of a
device value is allowed in the rollout.

Go2 counterparts that have no drone analogue
-------------------------------------------
``go2_device_kernels`` owns three graphs because a device-resident Go2 step has three host
operations to replace; a device-resident drone step has one, and the other two are already device
routes.  This is a finding, not a gap:

* ``go2_device_control`` has no analogue because ``hover_controls`` already *is* that graph.  It
  clips the raw action and turns it into rotor RPMs, and it returns two device tensors when the
  rollout calls ``evoxc.run`` on it directly, which is exactly the form the drone step writes to
  the drone and observes.  Go2's control graph additionally folds the ``last_actions[reset_idx] =
  0.0`` scatter of the rows that just ended; the drone evaluator has no such scatter to fold,
  because it runs one episode per row and never resets inside a generation, so the rollout's
  ``last_actions`` is the previous ``hover_controls`` output unchanged.
* ``go2_device_state`` has no analogue because the drone's graph-facing state is already device
  data.  The observation slot carries the raw absolute base quaternion (``hover_advance`` builds
  the relative rotation itself from a constant), and the base position, quaternion and both
  world-frame velocities are exactly the operands ``hover_advance`` takes, in an order it already
  matches.  In a Torch-free scene ``drone.get_pos`` / ``get_quat`` / ``get_vel`` / ``get_ang``
  return ``DeviceView`` s of the live solver buffers -- ``get_vel`` carrying the
  ``cd_vel + cd_ang x (pos - root_COM)`` transport term the Torch rollout had to compose by hand --
  and a view whose link slice is strided is compacted with ``DeviceView.contiguous()``, on the
  device.  The solver-error mask needs no conversion graph either: ``hover_advance`` consumes it as
  a bool operand while Torch-free ``scene.get_error_envs_mask()`` hands out the int32 view of its
  own buffer, and the rollout reads that latch to the host anyway (the ``clear_errno`` decision and
  the ``end_reasons`` accounting), so a device-side int32 -> bool graph would replace one
  ``(rows,)`` upload and save no readback.
* the rotor-RPM hand-off is not a graph concern: it is ``drone.set_propellers_rpm``, the single
  actuator write Genesis requires, and ``hover_controls`` already hands it a device tensor;
  whether Genesis takes that tensor without a host round trip is a Genesis-side question, not a
  graph one.

Static operands
---------------
``n_policies`` and ``episodes_per_policy`` are the static specializations of the reshape, and
``evoxc`` validates them on every call, so ``build_episode_returns`` stashes them on the executable
(``.n_policies``, ``.episodes_per_policy``, ``.rows``) and ``run_episode_returns`` re-passes them.
"""

from __future__ import annotations

import numpy as np

import evoxc

import hover_kernels  # noqa: F401  (imported first: it owns the shared-layer ``sys.path`` shim)
from evox_runtime import (
    _attach,
    _device_name,
    _outputs,
    _positive_int,
    _rows_of,
    _to_device_array,
    resolve_backend_device,
)
from seeding import EPISODES_PER_POLICY


# --- Episode returns (the one shape change of a device buffer) -----------------------


@evoxc.defn
def hover_device_episode_returns(returns, n_policies, episodes_per_policy):
    """``(rows,)`` returns -> the policy-major ``(pop_size, episodes)`` matrix, on the device."""
    return evoxc.reshape(returns, (n_policies, episodes_per_policy))


def build_episode_returns(n_policies, *, episodes_per_policy=EPISODES_PER_POLICY, backend="numpy", device=None):
    """Compile the episode-returns reshape for ``n_policies`` policies.

    This is the only way to change a device buffer's shape without a host roundtrip:
    ``iree.runtime.array_interop.DeviceArray.reshape`` goes through numpy dispatch and silently
    returns a host ``numpy.ndarray`` (its ``_buffer_view`` is ``None``), so wrapping it as a device
    payload makes the next readback fail.  ``evoxc.reshape`` on a symbolic tensor is a real device op.
    The rows are policy-major (rows ``3p, 3p+1, 3p+2`` are the episodes of policy ``p``), so the
    reshape is the whole mapping: no reordering is involved.

    Args:
        n_policies (int): Population size (``HoverNeatEnv.pop_size``).
        episodes_per_policy (int): Episodes per policy (``EPISODES_PER_POLICY``).
        backend (str): ``evoxc`` backend name (``resolve_backend_device`` rules).
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_episode_returns`` understands its metadata.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    episodes_per_policy = _positive_int(episodes_per_policy, "episodes_per_policy")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    executable = evoxc.build(
        hover_device_episode_returns,
        evoxc.TensorSpec((n_policies * episodes_per_policy,), evoxc.float32, name="returns"),
        n_policies,
        episodes_per_policy,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="episode_returns",
        rows=n_policies * episodes_per_policy,
        n_policies=n_policies,
        episodes_per_policy=episodes_per_policy,
        device_name=_device_name(resolved_device),
    )


def run_episode_returns(executable, returns):
    """Reshape the device-resident ``(rows,)`` returns into ``(n_policies, episodes)``.

    Args:
        executable (evoxc.Executable): Built by ``build_episode_returns``.
        returns: ``(rows,)`` float32 tensor on the executable's device (a graph output passes
            through with no staging; a NumPy array is staged for a host-side check).

    Returns:
        The ``(n_policies, episodes)`` float32 tensor on the executable's device.
    """
    rows = _rows_of(executable)
    shape = tuple(int(dim) for dim in returns.shape)
    if shape != (rows,):
        raise ValueError(f"returns must have shape {(rows,)}, got {shape}")
    if isinstance(returns, np.ndarray):
        returns = _to_device_array(executable, returns)
    outputs = evoxc.run(
        executable,
        returns,
        executable.n_policies,
        executable.episodes_per_policy,
    )
    return _outputs(outputs)[0]
