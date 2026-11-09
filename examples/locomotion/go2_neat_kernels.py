"""Go2 walking graphs for the TensorNEAT trainer, as EvoX Compiler (``evoxc``) graphs.

This module owns the Go2 task graphs of the walking trainer -- the observation assembly and the
one-step advance (reward, termination, episode bookkeeping) -- so the environment module stays
thin: it reads the simulator, hands the state to a graph, and writes the policy targets back.
Nothing here imports ``genesis`` or Torch, and nothing here simulates; the module speaks only
``numpy`` and ``evoxc``.

The task-agnostic half of the same stack lives in ``examples/neuroevolution``: the width-generic
policy graph (``policy_graphs``), the fitness and ``tell`` graphs plus the shape-keyed
``ExecutableCache`` (``evolution_graphs``), the seeding rules (``seeding``) and the plain
``evoxc`` plumbing (``evox_runtime``).  This module imports them by bare module name;
``_ensure_shared_layer_path()`` runs as the first import-time action and puts
``<repo>/examples/neuroevolution`` on ``sys.path``, the bare-name convention the rest of
``examples/`` uses.  The shim lives here and only here: every other module of this stack imports
``go2_neat_kernels`` first, which is what makes the shared layer importable in them.

Graphs owned here (each compiled once, then reused for every step and every generation)
---------------------------------------------------------------------------------------
``go2_neat_obs``          state -> the 45-dim policy observation row of ``go2_env``.
``go2_neat_advance``      one environment step: clip -> obs -> reward -> bookkeeping.

The policy, fitness and tell graphs are the shared ones; ``build_policy`` here only binds the Go2
widths (45 observations, 12 actions) around the shared builder, while ``run_policy``,
``build_fitness``, ``run_fitness``, ``build_tell``, ``run_tell`` and ``ExecutableCache`` are the
shared objects re-exported unchanged.

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

import math
import sys
from pathlib import Path

#: Shared, task-agnostic neuroevolution layer: ``<repo>/examples/neuroevolution``.
_SHARED_LAYER_DIR = Path(__file__).resolve().parents[1] / "neuroevolution"


def _ensure_shared_layer_path():
    """Put ``examples/neuroevolution`` on ``sys.path`` so its modules import by bare name.

    The example tree is a set of plain scripts rather than a package, so a shared module is
    imported by its file name; this shim is the single place that makes those names resolvable
    for the Go2 stack (every other module here imports ``go2_neat_kernels`` first).

    Returns:
        str: The absolute path that is now on ``sys.path``.
    """
    shared = str(_SHARED_LAYER_DIR)
    if shared not in sys.path:
        sys.path.insert(0, shared)
    return shared


_ensure_shared_layer_path()

import numpy as np

import evoxc

from evolution_graphs import (
    ExecutableCache,
    build_fitness,
    build_tell,
    run_fitness,
    run_tell,
)
from evox_runtime import (
    _attach,
    _device_name,
    _outputs,
    _positive_int,
    _rows_of,
    _staged,
)
from go2_evox_kernels import (
    DT,
    STANDING_FLAT,
    _constant,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)
from policy_graphs import build_policy as _shared_build_policy, run_policy
from seeding import EPISODES_PER_POLICY, Perturbation


# --- Task constants (``go2_train.get_cfgs()`` / ``go2_env``) ------------------------

NUM_OBS = 45
NUM_ACTIONS = 12
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
        roll  = atan2(sinrcosp, cosrcosp)  -> |sinrcosp| > tan(limit) * cosrcosp
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
        evoxc.greater(evoxc.abs(sinrcosp), tan_limit * cosrcosp),
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
    # ``axis=1``, never ``-1``: evoxc emits a negative axis verbatim into StableHLO
    # (``dimension = -1``) and ``iree-compile`` rejects it; the operands are rank-2.
    return evoxc.concatenate(
        [
            body_ang_vel * OBS_SCALES["ang_vel"],
            projected_gravity,
            _constant(commands_scaled_tiled),
            (dof_pos - _constant(standing_tiled)) * OBS_SCALES["dof_pos"],
            dof_vel * OBS_SCALES["dof_vel"],
            actions,
        ],
        axis=1,
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


# --- Policy (the Go2 widths around the shared graph) --------------------------------


def build_policy(algo, config, pop_arrays, rows, *, backend="numpy", device=None):
    """Compile the genome-payload -> action graph for the Go2 task, 45 observations -> 12 actions.

    The graph is the shared one of ``examples/neuroevolution/policy_graphs.py``; this adapter
    exists because the widths are a property of the task and the callers of this module (the env
    and the graph tests) build it without naming ``num_inputs`` / ``num_outputs``.

    Args:
        algo: a TensorNEAT ``NEAT`` algorithm instance.
        config: the tensor-free ``pipe._static_config(state)`` captured at build time.
        pop_arrays: the four genome payload arrays (``state.pop`` as numpy).
        rows (int): ``NP * EPISODES_PER_POLICY``; the rows are policy-major.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_policy`` stages its operands.
    """
    return _shared_build_policy(
        algo,
        config,
        pop_arrays,
        rows,
        num_inputs=NUM_OBS,
        num_outputs=NUM_ACTIONS,
        episodes=EPISODES_PER_POLICY,
        backend=backend,
        device=device,
    )


# --- Fitness, tell and cache (shared layer) -----------------------------------------
# ``build_fitness`` / ``run_fitness`` / ``build_tell`` / ``run_tell`` / ``ExecutableCache`` are
# imported from ``examples/neuroevolution`` at the top of this module and re-exported unchanged:
# this module stays the single import surface the env and the trainer build against.
