"""Go2 graphs that exist so the per-step values of the device rollout never leave the GPU.

The numpy residency of ``go2_neat_env.py`` performs two operations on the host that the
compiled graphs of ``go2_neat_kernels.py`` do not cover: it turns a raw policy action into
the PD targets the twelve actuators are driven with (clip, scale, add the default pose and
permute the action order into the dof order), and it composes the live solver state into the
graph-facing quaternion and joint blocks (relative base quaternion, joint order, error flag).
Both are graphs here, so a device-resident step reads the solver, runs five graphs and writes
the targets back with no host array in between -- only the ``done`` byte mask and, on an
ending step, the base quaternion and the error flags are read back.

Nothing here imports ``genesis`` or Torch, and nothing here simulates; the module speaks only
``numpy`` and ``evoxc``.

Graphs owned here (each compiled once and reused for every step and every generation)
---------------------------------------------------------------------------------------
``go2_device_control``       raw action + that step's ``done`` -> (next step's PD targets, last_actions)
``go2_device_state``         live solver buffers -> (base_quat, dof_pos, dof_vel, error_flag)
``go2_device_episode_returns`` ``(rows,)`` returns -> the policy-major ``(pop_size, episodes)`` matrix

``DeviceArray.reshape`` and every numpy function called on a device array return a HOST copy
(``iree.runtime`` dispatches them through numpy), so a device buffer's shape is only changed by
a graph -- that is what ``go2_device_episode_returns`` exists for, and no other reshaping of a
device value is allowed in the rollout.

``go2_device_control`` replaces three host operations of the numpy path in one pass: the
``np.clip`` of the raw action, the ``last_actions[reset_idx] = 0.0`` scatter of the rows that
just ended (the graph receives that step's ``done`` mask and zeroes them itself) and the
``targets[:, actions_dof_idx]`` permutation into the dof order.  Its targets and its
``last_actions`` are the very values ``go2_env`` feeds the PD controllers and the ``action_rate``
reward term respectively -- one step later, because a step executes the previous step's action::

    clipped      = min(max(raw, -clip_actions), clip_actions)
    last_actions = clipped * (1 - done)
    targets      = last_actions @ W * action_scale + default_targets

``W`` is the ``(12, 12)`` permutation with ``W[actions_dof_idx[k], k] = 1``, i.e. the
``targets[:, actions_dof_idx]`` write of ``go2_neat_env._write_targets``, and
``default_targets`` is the ``(rows, 12)`` tile of the default pose in the same dof order.

``go2_device_state`` transposes the batch-last joint blocks, gathers them from the dof order
of ``motors_dof_idx`` into the action order with the ``(12, 12)`` permutation
``P[qpos_dof_slot[k], k] = 1``, and composes the base quaternion relative to the generation's
init pose::

    base_quat = normalize(quat_world * inv_init_quat)

with the w-first Hamilton product ``go2_neat_env._quat_mul`` computes on the host (the
``genesis.utils.geom._tc_quat_mul`` coefficient form), then normalizes.  The error flag is
``error_mask != 0``: genesis reports a latched solver error per row, and the advance graph
consumes it as a bool.

Static operands
---------------
Every graph keeps the static half of its math as operands that ``run_*`` re-passes on every
call (``evoxc`` validates them): the ``(12, 12)`` permutations and the default-pose tile are
runtime operands because a static ndarray cannot enter ``evoxc.matmul``, so the caller stages
them on the device once per generation and hands the same tensors back for every step.
``build_*`` stashes them on the executable (``.permutation``, ``.gather``, ``.rows``,
``.action_scale``, ``.clip_actions``, ``.n_policies``, ``.episodes_per_policy``), so a runner
never has to rebuild them.
"""
from __future__ import annotations

import numpy as np
import evoxc

import go2_neat_kernels  # noqa: F401  (imported first: it owns the shared-layer ``sys.path`` shim)
from evox_runtime import _attach, _device_name, _outputs, _positive_int, _rows_of
from go2_evox_kernels import _to_device_array, resolve_backend_device
from go2_neat_kernels import EPISODES_PER_POLICY, NUM_ACTIONS, _int32, _quat_columns

#: Columns of the joint slice the twelve actuators live in (``qpos``/``dofs`` dof order).
NUM_MOTORS = NUM_ACTIONS


# --- Shared helpers -----------------------------------------------------------------


def permutation_matrix(order, width: int) -> np.ndarray:
    """The ``(width, width)`` float32 matrix with ``P[order[k], k] = 1``.

    ``matmul(x, P)`` then equals ``x[:, order]``: it maps a ``(rows, width)`` block whose
    columns are indexed by ``k`` into one whose columns are indexed by ``order[k]``.  The
    graph needs a matrix rather than a gather because only ``matmul`` by a runtime operand is
    a proven compile path for both the IREE and the numpy backends.

    Args:
        order (array-like): ``(width,)`` integer permutation (``actions_dof_idx``,
            ``qpos_dof_slot`` or their inverse).
        width (int): Number of columns.

    Returns:
        np.ndarray: ``(width, width)`` float32.
    """
    order_np = np.asarray(order, dtype=np.int64).reshape(width)
    scatter = np.zeros((width, width), dtype=np.float32)
    scatter[order_np, np.arange(width)] = 1.0
    return scatter


def default_targets_tile(rows, default_dof_pos, actions_dof_idx) -> np.ndarray:
    """The ``(rows, 12)`` tile of the default pose, in the dof order of the actuator slice.

    Args:
        rows (int): Number of environment rows.
        default_dof_pos (array-like): ``(12,)`` default joint angles in action order.
        actions_dof_idx (array-like): ``(12,)`` int index array (``np.argsort(motors_dof_idx)``).

    Returns:
        np.ndarray: ``(rows, 12)`` float32, contiguous.
    """
    posed = np.asarray(default_dof_pos, dtype=np.float32).reshape(NUM_MOTORS)
    return np.ascontiguousarray(np.tile(posed[np.asarray(actions_dof_idx, dtype=np.int64)], (_positive_int(rows, "rows"), 1)))


def _quat_product_rows(u: evoxc.SymbolicTensor, v: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Row-wise w-first Hamilton product ``u * v``, normalized, as ``go2_neat_env._quat_mul``.

    ``u`` is the live world quaternion and ``v`` the inverse of the generation's init
    quaternion, so the result is the orientation of the base relative to its reset pose.
    """
    uw, ux, uy, uz = _quat_columns(u)
    vw, vx, vy, vz = _quat_columns(v)
    qw = uw * vw - ux * vx - uy * vy - uz * vz
    qx = uw * vx + ux * vw + uy * vz - uz * vy
    qy = uw * vy - ux * vz + uy * vw + uz * vx
    qz = uw * vz + ux * vy - uy * vx + uz * vw
    norm = evoxc.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return evoxc.stack([qw / norm, qx / norm, qy / norm, qz / norm], axis=-1)


def _operand(executable, value):
    """Hand one operand to a graph: a NumPy array is staged, anything else passes through.

    The device rollout always passes ``evoxc`` tensors (a graph output, or the
    ``IreeDevicePayload`` wrapper around a live solver buffer), which travel with no copy at
    all; a NumPy array only appears when this module is driven from a host-side check.
    """
    if isinstance(value, np.ndarray):
        return _to_device_array(executable, value)
    return value


# --- Control (raw action -> PD targets) ---------------------------------------------


@evoxc.defn
def go2_device_control(raw_actions, done, permutation, default_targets, rows, action_scale, clip_actions):
    """One control step: clip the raw action, zero the ended rows, scale and permute."""
    clipped = evoxc.maximum(evoxc.minimum(raw_actions, clip_actions), -clip_actions)
    keep = 1.0 - evoxc.cast(evoxc.reshape(done, (rows, 1)), evoxc.float32)
    last_actions = clipped * keep
    targets = evoxc.matmul(last_actions, permutation) * action_scale + default_targets
    return [targets, last_actions]


def build_control(rows, *, action_scale, actions_dof_idx, clip_actions, backend="numpy", device=None):
    """Compile the control graph for ``rows`` environments.

    Args:
        rows (int): Number of environment rows.
        action_scale (float): Radians of target offset per unit of action.
        actions_dof_idx (array-like): ``(12,)`` int index array mapping action order to the
            dof order of the actuator slice (``np.argsort(motors_dof_idx)``).
        clip_actions (float): Symmetric clip of the raw action, ``go2_neat_kernels.CLIP_ACTIONS``.
        backend (str): ``evoxc`` backend name (``resolve_backend_device`` rules).
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_control`` understands its metadata.
    """
    rows = _positive_int(rows, "rows")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    permutation = permutation_matrix(actions_dof_idx, NUM_MOTORS)
    executable = evoxc.build(
        go2_device_control,
        evoxc.TensorSpec((rows, NUM_MOTORS), evoxc.float32, name="raw_actions"),
        evoxc.TensorSpec((rows,), evoxc.bool_, name="done"),
        evoxc.TensorSpec((NUM_MOTORS, NUM_MOTORS), evoxc.float32, name="permutation"),
        evoxc.TensorSpec((rows, NUM_MOTORS), evoxc.float32, name="default_targets"),
        rows,
        float(action_scale),
        float(clip_actions),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="control",
        rows=rows,
        action_scale=float(action_scale),
        clip_actions=float(clip_actions),
        actions_dof_idx=np.asarray(actions_dof_idx, dtype=np.int64).reshape(NUM_MOTORS),
        permutation=permutation,
        device_name=_device_name(resolved_device),
    )


def run_control(executable, *, raw_actions, done, permutation=None, default_targets, clip_actions=None):
    """Run the control graph; returns ``(targets, last_actions)``, both ``(rows, 12)`` float32.

    ``targets`` is in the dof order of the actuator slice (write it with
    ``control_dofs_position(targets, slice(dof_low, dof_high))``), ``last_actions`` is in
    action order (the ``actions`` slot of the observation and the ``action_rate`` reference).
    ``done`` is the end mask of the step these raw actions belong to: those rows get zero actions,
    which is what the numpy path does with ``last_actions[reset_idx] = 0.0`` after a reset.
    Both outputs are the values of the NEXT step (``go2_env`` executes the previous step's action,
    ``simulate_action_latency``), so the rollout writes ``targets`` one step later and feeds
    ``last_actions`` as the ``action_rate`` reference of the following ``run_advance``.
    """
    rows = _rows_of(executable)
    clip_actions = executable.clip_actions if clip_actions is None else float(clip_actions)
    if float(clip_actions) != executable.clip_actions:
        raise ValueError(f"clip_actions={clip_actions!r} does not match the compiled {executable.clip_actions!r}")
    outputs = evoxc.run(
        executable,
        _operand(executable, raw_actions),
        _operand(executable, done),
        _operand(executable, executable.permutation if permutation is None else permutation),
        _operand(executable, default_targets),
        rows,
        executable.action_scale,
        clip_actions,
    )
    targets, last_actions = _outputs(outputs)[:2]
    return targets, last_actions


# --- State (live solver buffers -> graph operands) ----------------------------------


@evoxc.defn
def go2_device_state(base_quat_world, dof_pos_block, dof_vel_block, inv_init_quat, error_mask, gather, mask_is_bool):
    """One state step: relative base quaternion, action-order joint blocks, error flag."""
    dof_pos = evoxc.matmul(evoxc.transpose(dof_pos_block, (1, 0)), gather)
    dof_vel = evoxc.matmul(evoxc.transpose(dof_vel_block, (1, 0)), gather)
    base_quat = _quat_product_rows(base_quat_world, inv_init_quat)
    # ``Scene.get_error_envs_mask`` is a boolean tensor when Torch is present and the zero-copy
    # view of an int32 buffer when it is not, and a compiled graph has one input dtype, so the
    # caller builds the graph for the mask its scene hands out (``mask_is_bool`` is static).
    error_flag = error_mask if mask_is_bool else evoxc.not_equal(error_mask, _int32(0))
    return [base_quat, dof_pos, dof_vel, error_flag]


def build_state(rows, *, num_dof, qpos_dof_slot, mask_is_bool=False, backend="numpy", device=None):
    """Compile the state graph for ``rows`` environments.

    Args:
        rows (int): Number of environment rows.
        num_dof (int): Width of the actuator slice (``12`` for the Go2).
        qpos_dof_slot (array-like): ``(num_dof,)`` int index array mapping action order to
            the slot inside the actuator slice (``motors_dof_idx - dof_low``).
        mask_is_bool (bool): Whether ``scene.get_error_envs_mask()`` hands out a boolean tensor
            (Torch present) instead of the int32 zero-copy buffer view (Torch-free), which fixes
            the dtype of that graph input.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_state`` understands its metadata.
    """
    rows = _positive_int(rows, "rows")
    num_dof = _positive_int(num_dof, "num_dof")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    gather = permutation_matrix(qpos_dof_slot, num_dof)
    executable = evoxc.build(
        go2_device_state,
        evoxc.TensorSpec((rows, 4), evoxc.float32, name="base_quat_world"),
        evoxc.TensorSpec((num_dof, rows), evoxc.float32, name="dof_pos_block"),
        evoxc.TensorSpec((num_dof, rows), evoxc.float32, name="dof_vel_block"),
        evoxc.TensorSpec((rows, 4), evoxc.float32, name="inv_init_quat"),
        evoxc.TensorSpec((rows,), evoxc.bool_ if mask_is_bool else evoxc.int32, name="error_mask"),
        evoxc.TensorSpec((num_dof, num_dof), evoxc.float32, name="gather"),
        bool(mask_is_bool),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="state",
        rows=rows,
        num_dof=num_dof,
        qpos_dof_slot=np.asarray(qpos_dof_slot, dtype=np.int64).reshape(num_dof),
        mask_is_bool=bool(mask_is_bool),
        gather=gather,
        device_name=_device_name(resolved_device),
    )


def run_state(
    executable,
    *,
    base_quat_world,
    dof_pos_block,
    dof_vel_block,
    inv_init_quat,
    error_mask,
    gather=None,
):
    """Run the state graph.

    Args:
        executable (evoxc.Executable): Built by ``build_state``.
        base_quat_world: ``(rows, 4)`` float32 world base quaternion (``robot.get_quat()``).
        dof_pos_block: ``(num_dof, rows)`` float32 **batch-last** joint positions of the
            actuator slice (a contiguous slice of ``solver.dyn_state.dofs.pos``).
        dof_vel_block: ``(num_dof, rows)`` float32, same layout, joint velocities.
        inv_init_quat: ``(rows, 4)`` float32 inverse of the generation's init quaternion.
        error_mask: ``(rows,)`` solver error mask (``scene.get_error_envs_mask()``): int32, or
            bool when the graph was built with ``mask_is_bool=True``.
        gather: ``(num_dof, num_dof)`` float32 permutation; default = the compiled one.

    Returns:
        tuple: ``(base_quat, dof_pos, dof_vel, error_flag)`` -- ``(rows, 4)`` float32,
        ``(rows, 12)`` float32, ``(rows, 12)`` float32 and ``(rows,)`` bool, all device
        tensors of the executable's device.
    """
    outputs = evoxc.run(
        executable,
        _operand(executable, base_quat_world),
        _operand(executable, dof_pos_block),
        _operand(executable, dof_vel_block),
        _operand(executable, inv_init_quat),
        _operand(executable, error_mask),
        _operand(executable, executable.gather if gather is None else gather),
        executable.mask_is_bool,
    )
    base_quat, dof_pos, dof_vel, error_flag = _outputs(outputs)[:4]
    return base_quat, dof_pos, dof_vel, error_flag


# --- Episode returns (the one shape change of a device buffer) -----------------------


@evoxc.defn
def go2_device_episode_returns(returns, n_policies, episodes_per_policy):
    """``(rows,)`` returns -> the policy-major ``(pop_size, episodes)`` matrix, on the device."""
    return evoxc.reshape(returns, (n_policies, episodes_per_policy))


def build_episode_returns(n_policies, *, episodes_per_policy=EPISODES_PER_POLICY, backend="numpy", device=None):
    """Compile the episode-returns reshape for ``n_policies`` policies.

    This is the only way to change a device buffer's shape without a host roundtrip:
    ``iree.runtime.array_interop.DeviceArray.reshape`` goes through numpy dispatch and silently
    returns a host ``numpy.ndarray`` (its ``_buffer_view`` is ``None``), so wrapping it as a device
    payload makes the next readback fail.  ``evoxc.reshape`` on a symbolic tensor is a real device op.

    Args:
        n_policies (int): Population size (``Go2NeatEnv.pop_size``).
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
        go2_device_episode_returns,
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
    if not isinstance(returns, np.ndarray) and tuple(int(dim) for dim in returns.shape) != (rows,):
        raise ValueError(f"returns must have shape {(rows,)}, got {tuple(int(dim) for dim in returns.shape)}")
    outputs = evoxc.run(
        executable,
        _operand(executable, returns),
        executable.n_policies,
        executable.episodes_per_policy,
    )
    return _outputs(outputs)[0]
