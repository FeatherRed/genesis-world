"""Go2 Walking graphs for the TensorNEAT trainer, as EvoX Compiler (``evoxc``) graphs.

This module owns every compiled graph the TensorNEAT Go2-walking trainer asks for,
so the environment and trainer modules stay thin: they read the simulator, hand
the state to a graph, and write the policy targets back.  Nothing here imports
``genesis`` or Torch, and nothing here simulates; the module speaks only
``numpy``, ``evoxc`` and ``TensorNEAT``.

Graphs (each compiled once, then reused for every step and every generation)
--------------------------------------------------------------------------
``go2_neat_obs``          state -> the 45-dim policy observation row of ``go2_env``.
``go2_neat_policy``       genome payload + observation -> raw actions.
``go2_neat_advance``      one environment step: clip -> obs -> reward -> bookkeeping.
``go2_neat_fitness``      per-episode returns -> per-policy fitness (episode mean).
``go2_neat_tell``         TensorNEAT ``tell``: one full generation transition.

The observation row, in this exact order, mirrors ``go2_env._update_observation``::

    [base_ang_vel_body * 0.25 (3), projected_gravity (3), commands * [2, 2, 0.25] (3),
     (dof_pos - default) * 1.0 (12), dof_vel * 0.05 (12), actions (12)]

``base_ang_vel_body`` is ``transform_by_quat(world_ang_vel, inv(base_quat))`` and
``projected_gravity`` is ``transform_by_quat([0, 0, -1], inv(base_quat))`` with
``inv_quat(q) = [w, -x, -y, -z]``; the quaternion is deliberately never normalized
(the rotation divides by ``|q|^2`` instead), exactly like ``genesis.utils.geom``.

The reward is ``go2_env``'s six-term Go2 Walking reward, ``dt``-scaled::

    dt * (1.0 * tracking_lin_vel + 0.2 * tracking_ang_vel - 1.0 * lin_vel_z
          - 50.0 * base_height - 0.005 * action_rate - 0.1 * similar_to_default)

Static specializations
----------------------
Every ``build_*`` bakes its static arrays (the standing pose, the command vector,
``dt``, the thresholds, ``max_episode_length``) into the graph, and every ``run_*``
must still pass the very same values: ``evoxc`` validates static operands with
``array_equal`` on every call.  The builders stash them on the executable (``.rows``,
``.standing_flat``, ``.commands``, ...), so a runner never has to guess them; only
the arguments that appear in a ``run_*`` signature have to come from the caller.

Frames and conventions the caller must honour
---------------------------------------------
* ``run_obs`` / ``run_advance`` take ``base_ang_vel_world`` / ``base_lin_vel_world``
  in the **world** frame (``robot.get_ang()`` / ``robot.get_vel()``); the graph
  rotates them into the body frame by ``inv(base_quat)``.
* ``run_advance`` takes the base quaternion **already relative to the initial
  pose** (``go2_env``'s ``base_init_quat`` is the identity, so the composed
  quaternion equals ``base_quat``); no composition happens inside the graph.
* ``run_advance`` clips raw actions to ``+-CLIP_ACTIONS`` inside the graph, uses the
  clipped value both for ``action_rate`` (against ``last_actions``) and for the
  observation's ``actions`` slot, and reports the pre-step ``returns``/``alive``
  after the update.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

import evoxc

from tensorneat.algorithm.neat.species import SpeciesState
from tensorneat.genome.utils import GenomeTensors

from go2_evox_kernels import (
    DT,
    STANDING_FLAT,
    _constant,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)

logger = logging.getLogger(__name__)

# --- Task constants (``go2_train.get_cfgs()`` / ``go2_env``) ------------------------

NUM_OBS = 45
NUM_ACTIONS = 12
#: Episodes one generation of policies is scored over; rows are policy-major.
EPISODES_PER_POLICY = 3
CLIP_ACTIONS = 100.0
TERMINATION_LIMIT_DEG = 10.0
#: ``obs_cfg["obs_scales"]``; ``lin_vel`` also scales the x/y velocity commands.
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
#: ``go2_train.get_cfgs()`` draws from singleton ranges (``[0.5, 0.5]``, ``[0, 0]``,
#: ``[0, 0]``), so ``go2_env``'s per-episode resampling is a no-op and this constant
#: vector is exactly faithful to the original task.
DEFAULT_COMMANDS = np.array([0.5, 0.0, 0.0], dtype=np.float32)
#: ``reward_cfg["reward_scales"]``; ``go2_env`` multiplies each term by ``dt``.
REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "base_height": -50.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
}
TRACKING_SIGMA = 0.25
BASE_HEIGHT_TARGET = 0.3

#: Per-axis scale applied to the command vector in the observation slot.
_COMMAND_OBS_SCALE = np.array(
    [OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]], dtype=np.float32
)


# --- Reset perturbation -------------------------------------------------------------


@dataclass(frozen=True)
class Perturbation:
    """Reset-time perturbation amplitudes for the env module to sample from.

    Each field is the **maximum absolute offset**; the env module owns the RNG and
    draws, e.g., ``uniform(-base_pos, +base_pos)`` per axis.  ``base_pos`` is in
    metres, ``base_rpy`` in radians, ``joint_pos`` in radians.  The all-zero default
    reproduces the original fixed-reset Go2 task, which is why ``is_zero`` exists.

    ``perturbed_init_qpos`` deliberately does not live here: RNG semantics belong to
    the environment, not to a compiled-graph module.
    """

    base_pos: float = 0.0
    base_rpy: float = 0.0
    joint_pos: float = 0.0

    @property
    def is_zero(self) -> bool:
        """Whether every amplitude is zero (the unperturbed reset)."""
        return self.base_pos == 0.0 and self.base_rpy == 0.0 and self.joint_pos == 0.0


# --- Small graph helpers ------------------------------------------------------------


def _int32(value: int) -> evoxc.SymbolicTensor:
    """Embed a Python int as a 0-d int32 graph constant.

    Plain-``int`` arithmetic promotes int32 to int64, so counters must be added to
    and compared against an int32 constant to keep the episode length int32.
    """
    return _constant(np.array(value, dtype=np.int32))


def _quat_columns(quat: evoxc.SymbolicTensor):
    """Split ``(rows, 4)`` quaternions into four ``(rows,)`` component columns."""
    return quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]


def _inv_quat_rows(quat: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Row-wise ``[w, -x, -y, -z]`` (a conjugate, NOT normalized)."""
    qw, qx, qy, qz = _quat_columns(quat)
    return evoxc.stack([qw, -qx, -qy, -qz], axis=-1)


def _rotate_rows(v: evoxc.SymbolicTensor, quat: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Rotate ``(rows, 3)`` vectors by a quaternion, as ``genesis.utils.geom`` does.

    The rotation matrix is built from the raw (possibly non-unit) quaternion and the
    result is divided by ``|q|^2``; the quaternion is never normalized.  Every
    intermediate is ``(rows,)``, so no rank-2 broadcasting is involved.
    """
    qw, qx, qy, qz = _quat_columns(quat)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    ww, xx, xy, xz = qw * qw, qx * qx, qx * qy, qx * qz
    yy, yz, zz = qy * qy, qy * qz, qz * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    norm_sqr = ww + xx + yy + zz
    ox = vx * (xx + ww - yy - zz) + vy * (2.0 * xy - 2.0 * wz) + vz * (2.0 * xz + 2.0 * wy)
    oy = vx * (2.0 * wz + 2.0 * xy) + vy * (ww - xx + yy - zz) + vz * (2.0 * yz - 2.0 * wx)
    oz = vx * (2.0 * xz - 2.0 * wy) + vy * (2.0 * wx + 2.0 * yz) + vz * (ww - xx - yy + zz)
    return evoxc.stack([ox / norm_sqr, oy / norm_sqr, oz / norm_sqr], axis=-1)


def _projected_gravity_rows(quat: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """``transform_by_quat([0, 0, -1], inv(quat))`` written out for a batch of rows."""
    qw, qx, qy, qz = _quat_columns(quat)
    norm_sqr = qw * qw + qx * qx + qy * qy + qz * qz
    ox = (2.0 * qw * qy - 2.0 * qx * qz) / norm_sqr
    oy = (-2.0 * qw * qx - 2.0 * qy * qz) / norm_sqr
    oz = (-(qw * qw) + qx * qx + qy * qy - qz * qz) / norm_sqr
    return evoxc.stack([ox, oy, oz], axis=-1)


def _tilt_exceeded_rows(quat: evoxc.SymbolicTensor, tan_limit: float):
    """Roll/pitch termination predicate WITHOUT ``arctan2``.

    Equivalent to ``|quat_to_xyz(quat, rpy=True, degrees=True)[0:2]| > limit`` (the
    ``go2_env`` rule) for ``|roll| <= 180deg`` and ``|pitch| <= 90deg``::

        pitch = atan2(sinp, cosp)          -> |sinp| > tan(limit) * cosp
        roll  = atan2(sinrcosp, cosrcosp)  -> |sinrcosp| > tan(limit) * cosrcosp * cosp
                                              (with the cosrcosp*cosp <= 0 quadrant guard)

    The sine/cosine products are the ones ``genesis.utils.geom._np_quat_to_xyz``
    builds (``s = 2 / |q|^2``), so the comparison is the same float32 decision.

    Returns:
        tuple: ``(roll_exceeded, pitch_exceeded)``, both ``(rows,)`` bool.
    """
    qw, qx, qy, qz = _quat_columns(quat)
    norm_sqr = qw * qw + qx * qx + qy * qy + qz * qz
    scale = 2.0 / norm_sqr
    sinp = (qw * qy - qx * qz) * scale
    sinrcosp = (qw * qx + qy * qz) * scale
    cosrcosp = 1.0 - (qx * qx + qy * qy) * scale
    cosycosp = 1.0 - (qy * qy + qz * qz) * scale
    sinycosp = (qw * qz + qx * qy) * scale
    cosp = evoxc.sqrt(cosycosp * cosycosp + sinycosp * sinycosp)
    cosr = cosrcosp * cosp
    pitch_exceeded = evoxc.greater(evoxc.abs(sinp), tan_limit * cosp)
    roll_exceeded = evoxc.logical_or(
        evoxc.less_equal(cosr, 0.0),
        evoxc.greater(evoxc.abs(sinrcosp), tan_limit * cosr),
    )
    return roll_exceeded, pitch_exceeded


def _obs_rows(
    base_ang_vel_world,
    base_quat,
    dof_pos,
    dof_vel,
    actions,
    standing_tiled,
    commands_scaled_tiled,
):
    """The 45-dim observation rows of ``go2_env._update_observation``."""
    inv_quat = _inv_quat_rows(base_quat)
    body_ang_vel = _rotate_rows(base_ang_vel_world, inv_quat)
    projected_gravity = _projected_gravity_rows(base_quat)
    return evoxc.concatenate(
        [
            body_ang_vel * OBS_SCALES["ang_vel"],
            projected_gravity,
            _constant(commands_scaled_tiled),
            (dof_pos - _constant(standing_tiled)) * OBS_SCALES["dof_pos"],
            dof_vel * OBS_SCALES["dof_vel"],
            actions,
        ],
        axis=-1,
    )


# --- Static specialization arrays ---------------------------------------------------


def _command_vector(commands) -> np.ndarray:
    """Validate a command vector as a contiguous float32 ``(3,)`` array."""
    arr = np.ascontiguousarray(np.asarray(commands, dtype=np.float32).reshape(3))
    return arr


def _command_columns(commands_tiled):
    """Split the ``(rows, 3)`` static command tile into three graph columns."""
    commands = _constant(commands_tiled)
    return commands[:, 0], commands[:, 1], commands[:, 2]


def _tile_standing(standing_flat, rows: int) -> np.ndarray:
    """``(rows, 12)`` float32 tile of the standing pose (the ``dof_pos`` reference)."""
    arr = np.ascontiguousarray(np.asarray(standing_flat, dtype=np.float32).reshape(NUM_ACTIONS))
    return np.ascontiguousarray(np.tile(arr, (rows, 1)))


def _tile_commands(commands, rows: int) -> np.ndarray:
    """``(rows, 3)`` float32 tile of the command vector (reward reference)."""
    return np.ascontiguousarray(np.tile(_command_vector(commands), (rows, 1)))


def _tile_scaled_commands(commands, rows: int) -> np.ndarray:
    """``(rows, 3)`` float32 tile of the command vector scaled for the observation."""
    scaled = _command_vector(commands) * _COMMAND_OBS_SCALE
    return np.ascontiguousarray(np.tile(scaled, (rows, 1)))


def _device_name(device) -> str:
    """``"kind:index"`` of a resolved device, for cache keys and log lines."""
    return f"{device.kind}:{device.index}"


def _attach(executable, **metadata):
    """Stash build-time metadata on an ``evoxc`` executable (it has a ``__dict__``).

    ``Executable.backend`` / ``.device`` / ``.functions`` are read-only properties, so
    the metadata names deliberately avoid them.
    """
    for name, value in metadata.items():
        setattr(executable, name, value)
    return executable


def _positive_int(value, name: str) -> int:
    """Validate a strict positive integer build argument."""
    ivalue = int(value)
    if ivalue != value or ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


def _rows_of(executable) -> int:
    """The row count a graph was compiled for."""
    rows = getattr(executable, "rows", None)
    if rows is None:
        raise TypeError(f"{executable!r} was not built by this module (no '.rows' metadata)")
    return int(rows)


def _staged(executable, name: str, value, width=None, *, dtype=np.float32) -> np.ndarray:
    """Stage one runtime argument, checking its shape against the compiled graph."""
    rows = _rows_of(executable)
    expected = (rows,) if width is None else (rows, width)
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected} to match the compiled graph, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _host(value) -> np.ndarray:
    """Materialize one payload leaf as a host numpy array.

    ``state.pop`` / ``evoxc.tree_leaves(state.species)`` hand back ``evoxc.Tensor``
    leaves, and ``np.asarray`` on those yields a 0-d object array, so tensors are
    routed through ``_to_numpy`` instead.
    """
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "numpy"):
        arr = _to_numpy(value)
    else:
        arr = np.asarray(value)
    if arr.dtype == object:
        raise TypeError(
            "payload leaf is a 0-d object array; materialize an evoxc tensor with `.numpy()` "
            "(see go2_evox_kernels._to_numpy / this module's _host) before passing it in"
        )
    return arr


def _outputs(result) -> tuple:
    """Normalize a ``evoxc.run`` result: a single-leaf graph returns the tensor bare."""
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def _spec_for(array, name: str) -> evoxc.TensorSpec:
    """A ``TensorSpec`` matching one payload leaf (shape + dtype + name)."""
    arr = _host(array)
    return evoxc.TensorSpec(tuple(int(dim) for dim in arr.shape), evoxc.dtype(arr.dtype), name=name)


# --- Observations -------------------------------------------------------------------


@evoxc.defn
def go2_neat_obs(
    base_ang_vel_world,
    base_quat,
    dof_pos,
    dof_vel,
    actions,
    standing_tiled,
    commands_scaled_tiled,
):
    """Observation graph: ``(rows,)``-batched state -> ``(rows, 45)`` float32.

    ``actions`` is the **clipped** action vector the policy applied last (the
    ``go2_env`` observation slot), ``standing_tiled`` is the ``(rows, 12)`` standing
    pose and ``commands_scaled_tiled`` the ``(rows, 3)`` observation-scaled commands.
    """
    return _obs_rows(
        base_ang_vel_world,
        base_quat,
        dof_pos,
        dof_vel,
        actions,
        standing_tiled,
        commands_scaled_tiled,
    )


def build_obs(rows, *, standing_flat=STANDING_FLAT, commands=DEFAULT_COMMANDS, backend="numpy", device=None):
    """Compile the observation graph for ``rows`` parallel environments.

    Args:
        rows (int): Number of environment rows (also the policy count times
            ``EPISODES_PER_POLICY`` when used inside the trainer).
        standing_flat (array-like): ``(12,)`` default joint pose, ``STANDING_FLAT``.
        commands (array-like): ``(3,)`` command vector, ``[vx, vy, yaw_rate]``.
        backend (str): ``evoxc`` backend name (``resolve_backend_device`` rules).
        device (str | evoxc.Device | None): ``evoxc`` device, ``None`` -> env/default.

    Returns:
        evoxc.Executable: bound executable; ``run_obs`` understands its metadata.
    """
    rows = _positive_int(rows, "rows")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    standing_tiled = _tile_standing(standing_flat, rows)
    commands_scaled_tiled = _tile_scaled_commands(commands, rows)
    executable = evoxc.build(
        go2_neat_obs,
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_ang_vel_world"),
        evoxc.TensorSpec((rows, 4), evoxc.float32, name="base_quat"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="dof_pos"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="dof_vel"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="actions"),
        standing_tiled,
        commands_scaled_tiled,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="obs",
        rows=rows,
        standing_flat=np.ascontiguousarray(np.asarray(standing_flat, dtype=np.float32).reshape(NUM_ACTIONS)),
        commands=_command_vector(commands),
        device_name=_device_name(resolved_device),
    )


def run_obs(executable, *, base_ang_vel_world, base_quat, dof_pos, dof_vel, actions, commands):
    """Run the observation graph; returns ``(rows, 45)`` float32.

    ``commands`` is a **static** operand of the graph, so it must equal the vector the
    executable was built with (``evoxc`` re-validates it with ``array_equal``).
    """
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "base_ang_vel_world", base_ang_vel_world, 3)),
        _to_device_array(executable, _staged(executable, "base_quat", base_quat, 4)),
        _to_device_array(executable, _staged(executable, "dof_pos", dof_pos, NUM_ACTIONS)),
        _to_device_array(executable, _staged(executable, "dof_vel", dof_vel, NUM_ACTIONS)),
        _to_device_array(executable, _staged(executable, "actions", actions, NUM_ACTIONS)),
        _tile_standing(executable.standing_flat, _rows_of(executable)),
        _tile_scaled_commands(commands, _rows_of(executable)),
    )
    return _to_numpy(_outputs(outputs)[0])


# --- Policy -------------------------------------------------------------------------


def _pop_specs(pop_arrays) -> list:
    """``TensorSpec`` list for the four ``GenomeTensors`` payload arrays."""
    arrays = tuple(pop_arrays)
    if len(arrays) != 4:
        raise ValueError(f"pop_arrays must hold the 4 GenomeTensors payloads, got {len(arrays)}")
    names = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
    return [_spec_for(array, name) for array, name in zip(arrays, names)]


def build_policy(algo, config, pop_arrays, rows, *, backend="numpy", device=None):
    """Compile the genome-payload -> action graph for ``rows`` policy-episode rows.

    ``rows`` must be ``algo.pop_size * EPISODES_PER_POLICY`` and the rows are
    **policy-major**: rows ``3p, 3p+1, 3p+2`` are the ``EPISODES_PER_POLICY``
    episodes of policy ``p`` (that is the tiling ``prepare_forward(samples=...)``
    declares).  One trace only: ``transform``/``prepare_forward`` are trace-time
    operations and cannot be called outside a build.

    Args:
        algo: a TensorNEAT ``NEAT`` algorithm instance.
        config: the tensor-free ``pipe._static_config(state)`` captured at build time.
        pop_arrays: the four genome payload arrays (``state.pop`` as numpy).
        rows (int): ``NP * EPISODES_PER_POLICY``.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_policy`` stages its operands.
    """
    n_policies = int(algo.pop_size)
    rows = _positive_int(rows, "rows")
    if rows != n_policies * EPISODES_PER_POLICY:
        raise ValueError(
            f"rows must be pop_size * EPISODES_PER_POLICY = {n_policies * EPISODES_PER_POLICY}, got {rows}"
        )
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    specs = [
        *_pop_specs(pop_arrays),
        evoxc.TensorSpec((rows, NUM_OBS), evoxc.float32, name="obs"),
    ]

    def kernel(node_keys, node_attrs, conn_keys, conn_attrs, obs):
        payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        transformed = algo.transform(config, payload)
        prepared = algo.prepare_forward(config, transformed, payload, samples=EPISODES_PER_POLICY)
        return algo.forward_prepared(config, transformed, prepared, obs)

    executable = evoxc.build(kernel, *specs, backend=resolved_backend, device=resolved_device)
    return _attach(
        executable,
        graph="policy",
        rows=rows,
        n_policies=n_policies,
        samples=EPISODES_PER_POLICY,
        pop_shapes=tuple(_host(array).shape for array in pop_arrays),
        device_name=_device_name(resolved_device),
    )


def run_policy(executable, pop_arrays, obs):
    """Run the policy graph; returns raw (unclipped) actions ``(rows, 12)`` float32."""
    arrays = tuple(_host(array) for array in pop_arrays)
    if len(arrays) != 4:
        raise ValueError(f"pop_arrays must hold the 4 GenomeTensors payloads, got {len(arrays)}")
    rows = _rows_of(executable)
    for array, shape in zip(arrays, executable.pop_shapes):
        if array.shape != shape:
            raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {array.shape}")
    outputs = evoxc.run(
        executable,
        *[_to_device_array(executable, np.ascontiguousarray(array)) for array in arrays],
        _to_device_array(executable, _staged(executable, "obs", obs, NUM_OBS)),
    )
    return _to_numpy(_outputs(outputs)[0])


# --- Advance (one environment step) -------------------------------------------------


@evoxc.defn
def go2_neat_advance(
    base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    dof_pos,
    dof_vel,
    actions,
    last_actions,
    returns,
    alive,
    episode_length,
    error_mask,
    standing_tiled,
    commands_tiled,
    commands_scaled_tiled,
    max_episode_length,
    dt,
    tan_termination,
    clip_actions,
):
    """One environment step: clip -> observation -> reward -> episode bookkeeping.

    Returns:
        list: ``[obs (rows,45), reward (rows,), returns (rows,), alive (rows,) bool,
        done (rows,) bool, episode_length (rows,) int32]``.
    """
    clipped = evoxc.maximum(evoxc.minimum(actions, clip_actions), -clip_actions)
    obs = _obs_rows(
        base_ang_vel_world,
        base_quat,
        dof_pos,
        dof_vel,
        clipped,
        standing_tiled,
        commands_scaled_tiled,
    )

    inv_quat = _inv_quat_rows(base_quat)
    body_lin_vel = _rotate_rows(base_lin_vel_world, inv_quat)
    body_ang_vel = _rotate_rows(base_ang_vel_world, inv_quat)
    cmd_x, cmd_y, cmd_z = _command_columns(commands_tiled)

    lin_error = cmd_x - body_lin_vel[:, 0]
    lat_error = cmd_y - body_lin_vel[:, 1]
    tracking_lin_vel = evoxc.exp(-(lin_error * lin_error + lat_error * lat_error) / TRACKING_SIGMA)
    yaw_error = cmd_z - body_ang_vel[:, 2]
    tracking_ang_vel = evoxc.exp(-(yaw_error * yaw_error) / TRACKING_SIGMA)
    lin_vel_z = body_lin_vel[:, 2] * body_lin_vel[:, 2]
    height_error = base_pos[:, 2] - BASE_HEIGHT_TARGET
    base_height = height_error * height_error
    action_delta = clipped - last_actions
    action_rate = evoxc.reduce_sum(action_delta * action_delta, axes=-1)
    pose_error = evoxc.reduce_sum(evoxc.abs(dof_pos - _constant(standing_tiled)), axes=-1)
    reward = dt * (
        REWARD_SCALES["tracking_lin_vel"] * tracking_lin_vel
        + REWARD_SCALES["tracking_ang_vel"] * tracking_ang_vel
        + REWARD_SCALES["lin_vel_z"] * lin_vel_z
        + REWARD_SCALES["base_height"] * base_height
        + REWARD_SCALES["action_rate"] * action_rate
        + REWARD_SCALES["similar_to_default"] * pose_error
    )

    roll_exceeded, pitch_exceeded = _tilt_exceeded_rows(base_quat, tan_termination)
    episode_length_next = evoxc.cast(episode_length + _int32(1), evoxc.int32)
    died = evoxc.logical_or(
        evoxc.logical_or(evoxc.greater(episode_length_next, _int32(max_episode_length)), error_mask),
        evoxc.logical_or(roll_exceeded, pitch_exceeded),
    )
    alive_before = alive
    done = evoxc.logical_and(alive_before, died)
    alive_new = evoxc.logical_and(alive_before, evoxc.logical_not(died))
    returns_new = returns + reward * evoxc.cast(alive_before, evoxc.float32)
    # ``episode_length - episode_length`` is an int32 zeros vector of the right shape,
    # so the ``select`` branches stay int32 (a Python ``0`` would promote to int64).
    zero_int32 = episode_length - episode_length
    episode_length_new = evoxc.select(done, zero_int32, episode_length_next)
    return [obs, reward, returns_new, alive_new, done, episode_length_new]


def build_advance(
    rows,
    *,
    max_episode_length,
    standing_flat=STANDING_FLAT,
    commands=DEFAULT_COMMANDS,
    dt=DT,
    termination_limit_deg=TERMINATION_LIMIT_DEG,
    backend="numpy",
    device=None,
):
    """Compile the one-step advance graph for ``rows`` environments.

    Args:
        rows (int): Number of environment rows.
        max_episode_length (int): Episode cap in control steps
            (``ceil(episode_length_s / dt)``); the timeout fires once the counter
            exceeds it, as in ``go2_env``.
        standing_flat (array-like): ``(12,)`` default joint pose.
        commands (array-like): ``(3,)`` command vector (a static operand).
        dt (float): Control timestep the reward is scaled by.
        termination_limit_deg (float): Roll/pitch limit in degrees.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_advance`` stages its operands.
    """
    rows = _positive_int(rows, "rows")
    max_episode_length = _positive_int(max_episode_length, "max_episode_length")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    standing_tiled = _tile_standing(standing_flat, rows)
    commands_tiled = _tile_commands(commands, rows)
    commands_scaled_tiled = _tile_scaled_commands(commands, rows)
    tan_termination = math.tan(math.radians(float(termination_limit_deg)))
    specs = [
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_pos"),
        evoxc.TensorSpec((rows, 4), evoxc.float32, name="base_quat"),
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_lin_vel_world"),
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_ang_vel_world"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="dof_pos"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="dof_vel"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="actions"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="last_actions"),
        evoxc.TensorSpec((rows,), evoxc.float32, name="returns"),
        evoxc.TensorSpec((rows,), evoxc.bool_, name="alive"),
        evoxc.TensorSpec((rows,), evoxc.int32, name="episode_length"),
        evoxc.TensorSpec((rows,), evoxc.bool_, name="error_mask"),
        standing_tiled,
        commands_tiled,
        commands_scaled_tiled,
        max_episode_length,
        float(dt),
        tan_termination,
        CLIP_ACTIONS,
    ]
    executable = evoxc.build(go2_neat_advance, *specs, backend=resolved_backend, device=resolved_device)
    return _attach(
        executable,
        graph="advance",
        rows=rows,
        standing_flat=np.ascontiguousarray(np.asarray(standing_flat, dtype=np.float32).reshape(NUM_ACTIONS)),
        commands=_command_vector(commands),
        max_episode_length=max_episode_length,
        dt=float(dt),
        tan_termination=tan_termination,
        termination_limit_deg=float(termination_limit_deg),
        device_name=_device_name(resolved_device),
    )


def run_advance(
    executable,
    *,
    base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    dof_pos,
    dof_vel,
    actions,
    last_actions,
    returns,
    alive,
    episode_length,
    error_mask,
    commands,
):
    """Run one advance step; returns the observation plus the updated bookkeeping.

    ``actions`` are raw (unclipped) policy outputs; the graph clips them to
    ``+-CLIP_ACTIONS``.  ``base_quat`` is expected to be relative to the initial pose
    (identity for ``go2_env``), and the velocities are world-frame.  ``commands`` is a
    static operand, so it must match what ``build_advance`` was given.

    Args:
        executable: the bound graph from ``build_advance``.
        base_pos: ``(rows, 3)`` world base position (post-step).
        base_quat: ``(rows, 4)`` base quaternion relative to the initial pose.
        base_lin_vel_world: ``(rows, 3)`` ``robot.get_vel()``.
        base_ang_vel_world: ``(rows, 3)`` ``robot.get_ang()``.
        dof_pos: ``(rows, 12)`` joint positions.
        dof_vel: ``(rows, 12)`` joint velocities.
        actions: ``(rows, 12)`` raw policy output.
        last_actions: ``(rows, 12)`` previously applied (clipped) actions.
        returns: ``(rows,)`` accumulated episode returns (pre-step).
        alive: ``(rows,)`` bool, whether the episode is still running (pre-step).
        episode_length: ``(rows,)`` int32 step counter (pre-step).
        error_mask: ``(rows,)`` bool, externally signalled failures.
        commands: ``(3,)`` static command vector.

    Returns:
        dict: ``obs`` (rows, 45), ``reward`` (rows,), ``returns`` (rows,),
        ``alive`` (rows,) bool, ``done`` (rows,) bool, ``episode_length`` (rows,) int32.
    """
    rows = _rows_of(executable)
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "base_pos", base_pos, 3)),
        _to_device_array(executable, _staged(executable, "base_quat", base_quat, 4)),
        _to_device_array(executable, _staged(executable, "base_lin_vel_world", base_lin_vel_world, 3)),
        _to_device_array(executable, _staged(executable, "base_ang_vel_world", base_ang_vel_world, 3)),
        _to_device_array(executable, _staged(executable, "dof_pos", dof_pos, NUM_ACTIONS)),
        _to_device_array(executable, _staged(executable, "dof_vel", dof_vel, NUM_ACTIONS)),
        _to_device_array(executable, _staged(executable, "actions", actions, NUM_ACTIONS)),
        _to_device_array(executable, _staged(executable, "last_actions", last_actions, NUM_ACTIONS)),
        _to_device_array(executable, _staged(executable, "returns", returns)),
        _to_device_array(executable, _staged(executable, "alive", alive, dtype=bool)),
        _to_device_array(executable, _staged(executable, "episode_length", episode_length, dtype=np.int32)),
        _to_device_array(executable, _staged(executable, "error_mask", error_mask, dtype=bool)),
        _tile_standing(executable.standing_flat, rows),
        _tile_commands(commands, rows),
        _tile_scaled_commands(commands, rows),
        executable.max_episode_length,
        executable.dt,
        executable.tan_termination,
        CLIP_ACTIONS,
    )
    obs, reward, returns_new, alive_new, done, episode_length_new = _outputs(outputs)
    return {
        "obs": _to_numpy(obs),
        "reward": _to_numpy(reward),
        "returns": _to_numpy(returns_new),
        "alive": np.asarray(_to_numpy(alive_new), dtype=bool),
        "done": np.asarray(_to_numpy(done), dtype=bool),
        "episode_length": np.asarray(_to_numpy(episode_length_new), dtype=np.int32),
    }


# --- Fitness ------------------------------------------------------------------------


@evoxc.defn
def go2_neat_fitness(returns, n_policies, episodes_per_policy):
    """Episode returns ``(NP * episodes_per_policy,)`` -> fitness ``(NP,)`` float32."""
    return evoxc.reduce_mean(
        evoxc.reshape(returns, (n_policies, episodes_per_policy)),
        axes=1,
    )


def build_fitness(n_policies, *, episodes_per_policy=EPISODES_PER_POLICY, backend="numpy", device=None):
    """Compile the fitness aggregation graph (mean of one policy's episode returns).

    Args:
        n_policies (int): Population size (``algo.pop_size``).
        episodes_per_policy (int): Episodes per policy, ``EPISODES_PER_POLICY``.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_fitness`` stages its operand.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    episodes_per_policy = _positive_int(episodes_per_policy, "episodes_per_policy")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    rows = n_policies * episodes_per_policy
    executable = evoxc.build(
        go2_neat_fitness,
        evoxc.TensorSpec((rows,), evoxc.float32, name="returns"),
        n_policies,
        episodes_per_policy,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="fitness",
        rows=rows,
        n_policies=n_policies,
        episodes_per_policy=episodes_per_policy,
        device_name=_device_name(resolved_device),
    )


def run_fitness(executable, returns):
    """Run the fitness graph; ``returns`` is ``(NP * episodes, )``, result ``(NP,)``."""
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "returns", returns)),
        executable.n_policies,
        executable.episodes_per_policy,
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)


# --- Tell (one generation transition) -----------------------------------------------


def _species_specs(species_leaves) -> list:
    """``TensorSpec`` list for the ``SpeciesState`` leaves, in leaf order."""
    n_fields = len(SpeciesState.__dataclass_fields__)
    leaves = tuple(species_leaves)
    if len(leaves) != n_fields:
        raise ValueError(f"species_leaves must hold the {n_fields} SpeciesState leaves, got {len(leaves)}")
    return [_spec_for(leaf, f"species_{index}") for index, leaf in enumerate(leaves)]


def build_tell(algo, config, pop_arrays, species_leaves, n_policies, *, backend="numpy", device=None):
    """Compile the TensorNEAT ``tell`` graph: payload + fitness -> next generation.

    The graph returns the flat list ``[randkey, generation, *pop, *species]``; the
    ``SpeciesState`` pytree layout is captured during the trace and stored on the
    executable, so ``run_tell`` can rebuild it without a ``state`` argument.

    Args:
        algo: a TensorNEAT ``NEAT`` algorithm instance.
        config: the tensor-free ``pipe._static_config(state)`` (must not already
            register ``pop``/``generation``/``species``/``randkey``).
        pop_arrays: the four genome payload arrays (``state.pop`` as numpy).
        species_leaves: ``evoxc.tree_leaves(state.species)`` as numpy.
        n_policies (int): Population size, the fitness vector length.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_tell`` stages its operands.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    if n_policies != int(algo.pop_size):
        raise ValueError(f"n_policies must be algo.pop_size = {int(algo.pop_size)}, got {n_policies}")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    n_leaves = len(SpeciesState.__dataclass_fields__)
    specs = [
        evoxc.TensorSpec((), evoxc.int64, name="randkey"),
        *_pop_specs(pop_arrays),
        evoxc.TensorSpec((), evoxc.float32, name="generation"),
        evoxc.TensorSpec((n_policies,), evoxc.float32, name="fitness"),
        *_species_specs(species_leaves),
    ]
    captured = []

    def kernel(randkey, node_keys, node_attrs, conn_keys, conn_attrs, generation, fitness, *species_values):
        species_tree = evoxc.tree_structure(SpeciesState(*species_values))
        if not captured:
            captured.append(species_tree)
        payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        species = evoxc.tree_unflatten(list(species_values), species_tree)
        current = config.register(pop=payload, generation=generation, species=species, randkey=randkey)
        nxt = algo.tell(current, fitness)
        next_species = evoxc.tree_leaves(nxt.species)
        if len(next_species) != n_leaves:
            raise ValueError(
                f"algorithm.tell changed the species pytree: {n_leaves} leaves in, {len(next_species)} out"
            )
        return [nxt.randkey, nxt.generation, *nxt.pop, *next_species]

    executable = evoxc.build(kernel, *specs, backend=resolved_backend, device=resolved_device)
    return _attach(
        executable,
        graph="tell",
        rows=n_policies,
        n_policies=n_policies,
        species_tree=captured[-1] if captured else None,
        species_leaf_count=n_leaves,
        pop_shapes=tuple(_host(array).shape for array in pop_arrays),
        device_name=_device_name(resolved_device),
    )


def run_tell(executable, *, randkey, pop_arrays, generation, fitness, species_leaves):
    """Run the tell graph; returns the next ``randkey``/``generation``/``pop``/``species``.

    Args:
        executable: the bound graph from ``build_tell``.
        randkey (array-like): 0-d int64 RNG key.
        pop_arrays: the four genome payload arrays (same shapes as at build time).
        generation (array-like): 0-d float32 generation counter.
        fitness (array-like): ``(NP,)`` float32, maximized.
        species_leaves: the ``SpeciesState`` leaves (same layout as at build time).

    Returns:
        dict: ``randkey`` (0-d int64), ``generation`` (0-d float32), ``pop`` (tuple of
        the four payload arrays) and ``species`` (rebuilt ``SpeciesState``).
    """
    arrays = tuple(_host(array) for array in pop_arrays)
    if len(arrays) != 4:
        raise ValueError(f"pop_arrays must hold the 4 GenomeTensors payloads, got {len(arrays)}")
    leaves = tuple(_host(leaf) for leaf in species_leaves)
    if len(leaves) != executable.species_leaf_count:
        raise ValueError(
            f"species_leaves must hold {executable.species_leaf_count} leaves, got {len(leaves)}"
        )
    for array, shape in zip(arrays, executable.pop_shapes):
        if array.shape != shape:
            raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {array.shape}")
    n_policies = int(executable.n_policies)
    key = _host(randkey).astype(np.int64, copy=False)
    if key.shape != ():
        raise ValueError(f"randkey must be a 0-d int64 key, got shape {key.shape}")
    gen = _host(generation).astype(np.float32, copy=False)
    if gen.shape != ():
        raise ValueError(f"generation must be a 0-d float32 counter, got shape {gen.shape}")
    fit = _host(fitness).astype(np.float32, copy=False)
    if fit.shape != (n_policies,):
        raise ValueError(f"fitness must have shape ({n_policies},), got {fit.shape}")
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, key),
        *[_to_device_array(executable, np.ascontiguousarray(array)) for array in arrays],
        _to_device_array(executable, gen),
        _to_device_array(executable, np.ascontiguousarray(fit)),
        *[_to_device_array(executable, np.ascontiguousarray(leaf)) for leaf in leaves],
    )
    n_leaves = int(executable.species_leaf_count)
    flat = _outputs(outputs)
    next_pop = tuple(_to_numpy(flat[2 + index]) for index in range(4))
    next_species = [_to_numpy(flat[6 + index]) for index in range(n_leaves)]
    return {
        "randkey": _to_numpy(flat[0]),
        "generation": _to_numpy(flat[1]),
        "pop": next_pop,
        "species": evoxc.tree_unflatten(next_species, executable.species_tree),
    }


# --- Executable cache ---------------------------------------------------------------


class ExecutableCache:
    """Shape-keyed cache of bound executables, so a generation only builds once.

    Keys are hashable tuples of the static dimensions that specialize a graph --
    graph kind, row count, ``max_episode_length``, resolved backend, resolved device
    name, thresholds -- e.g. ``("advance", rows, max_episode_length, backend, device_name)``.
    Same key -> same executable; the next generation of the same shape hits the cache.
    A hit/miss debug line is logged per call.
    """

    def __init__(self):
        self._executables = {}
        self.hits = 0
        self.misses = 0

    def get(self, key, builder):
        """Return the executable for ``key``, building it with ``builder()`` on a miss.

        Args:
            key: hashable tuple of static dimensions.
            builder: zero-argument callable returning a bound executable.

        Returns:
            evoxc.Executable: the cached (or freshly built) executable.
        """
        key = tuple(key)
        cached = self._executables.get(key)
        if cached is not None:
            self.hits += 1
            logger.debug("executable cache hit: %s", key)
            return cached
        self.misses += 1
        logger.debug("executable cache miss: %s", key)
        executable = builder()
        self._executables[key] = executable
        return executable

    def clear(self):
        """Drop every cached executable and reset the hit/miss counters."""
        self._executables.clear()
        self.hits = 0
        self.misses = 0
