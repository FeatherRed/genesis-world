"""Drone hover graphs for the EvoX Compiler (``evoxc``) evaluation environment.

This module owns every compiled graph the drone-hover *eval* environment asks for, so
the environment module stays thin: it reads the simulator, hands the measured state to
a graph, and writes the policy targets (rotor RPMs) back.  Nothing here imports
``genesis`` or Torch, nothing here simulates, and nothing here trains; the module
speaks ``numpy`` and ``evoxc``.  The generic build/run helpers
(``resolve_backend_device``, ``_constant``, ``_staged``, ``_to_device_array``,
``_to_numpy``, ...) are imported from the shared neuroevolution layer instead of being
copied a third time.

Graphs (each compiled once, then reused every step and every generation)
-----------------------------------------------------------------------
``hover_controls``   raw actions -> ``(clipped actions, rotor RPMs)``.
``hover_obs``        measured base state -> the 17-dim policy observation row.
``hover_reset_obs``  commands + init pose -> the analytic post-``reset()`` observation row.
``hover_advance``    one environment step: geometry -> masks -> reward -> observation.

Seeding, fitness and the policy graph are **not** task graphs: they live in the shared
layer (``examples/neuroevolution``) as ``seeding.generation_seed_table`` /
``seeding.evolution_seed``, ``evolution_graphs.build_fitness`` / ``run_fitness`` and
``policy_graphs.build_policy`` / ``run_policy``, and the environment module imports them
from there.

The baseline being migrated is ``examples/drone/hover_env.py`` (torch + genesis +
tensordict), with its config in ``hover_train.get_cfgs()``; the parity oracle the graphs
are checked against is ``hover_legacy_reference`` in this directory.

Contracts
---------
Every ``build_*`` compiles for a fixed row count and stashes its static values on the
executable (``.rows``, ``.max_episode_length``, ...), so a ``run_*`` never guesses them.
``evoxc`` re-validates static operands with ``array_equal`` on every call, so a runner must
pass back exactly what its builder was given.  Rows are **policy-major**: rows ``3p, 3p+1,
3p+2`` are the ``EPISODES_PER_POLICY`` episodes of policy ``p``.  ``hover_advance`` also latches
each row's first ending into an ``END_REASONS`` code and hands back the ``base_pos`` it consumed,
so a device-resident caller can carry both across steps without a host round-trip.

The baseline's step order, reproduced in ``hover_advance``
-----------------------------------------------------------
``hover_env.HoverEnv.step`` (``hover_env.py:139-197``) does, in order: (1) clip the raw
action to ``+-1``; (2) ``rpm = (1 + 0.8 * actions) * 14468.429183500699`` into
``drone.set_propellers_rpm``; (3) ``scene.step()``; (4) ``episode_length += 1``, save
the old ``base_pos`` and read the new one; (5) ``rel_pos`` / ``last_rel_pos`` against
the commands; (6) read ``base_quat`` and build ``base_euler`` from
``quat_to_xyz(transform_quat_by_quat(inv(base_init_quat), base_quat), rpy=True,
degrees=True)``; (7) ``v_body`` / ``w_body`` via ``transform_by_quat(world, inv(base_quat))``;
(8) resample the commands of the rows with ``norm(rel_pos) < 0.1``; (9) crash predicate
(roll/pitch, ``rel_pos`` x/y/z, ``base_pos.z``); (10) timeout at
``episode_length > ceil(15 / 0.01) = 1500``; (11) the reward terms, computed *after*
the resample; (12) the observation, from the post-resample ``rel_pos`` and the previous
actions; (13) ``last_actions = actions``.

Steps 1-2 must run **before** ``scene.step()`` (the RPMs drive the step) while steps 4-13
are computed **after** it; that is why ``hover_controls`` exists separately from the advance
graph, which consumes the clipped action and reproduces steps 4-12 (step 13 is the caller's
``last_actions = actions``).

Divergences from the baseline
-----------------------------
* **One episode per row, no in-episode reset.**  Every row is a single episode, so rows
  ``3p..3p+2`` are three independent episodes of policy ``p`` and there is no
  mid-episode ``reset_idx``.
* **The baseline's reset-step reward pass is not reproduced.**  ``hover_env.step`` calls
  ``reset_idx`` *before* computing the reward, which zeroes ``episode_sums`` and
  ``last_actions`` for the reset rows and then adds the post-reset-step reward on top of the
  new episode.  Consequences: the reset-step reward and observation are replaced by the
  analytic reset observation at generation start (``hover_reset_obs``), and a truncated PPO
  episode return includes one extra post-reset step that this eval model does not.
  ``last_actions`` is thus never zeroed inside an episode, so ``smooth`` compares against the
  genuinely previous action.
* **Roll/pitch 180-degree crash checks dropped.**  Euler roll lives in ``[-180, 180]``
  and pitch in ``[-90, 90]``, so ``|roll| > 180`` and ``|pitch| > 180`` can never fire;
  the graph omits them instead of computing dead comparisons.  The ``yaw > 180`` unwrap
  in ``_reward_yaw`` is dropped for the same reason.
* **Commands are resampled for every row, every step.**  The baseline draws only for the
  rows that reached their target (and again inside ``reset_idx`` for reset rows).  Here
  the host draws three uniforms per row per step (x ~ U(-1, 1), y ~ U(-1, 1), z pinned
  to 1.0) from per-row ``np.random.Generator`` s, so the stream is row-order independent
  and reproducible; the distribution is unchanged from the baseline's iid draws.
  ``hover_advance`` selects the draw for the rows that reached their target this step.

Seeding rules (shared with the Go2 pilot, ``examples/neuroevolution``)
---------------------------------------------------------------------
``generation_seed_table`` gives every policy in a generation the *same ordered triple* of
episode seeds, a fresh triple per generation, and ``evolution_seed`` an independent evolution
stream; both live in ``seeding.py`` of the shared layer, not here.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

import evoxc

# The example stack resolves bare module names by putting sibling directories on ``sys.path``
# (``examples/locomotion`` already does it for the Go2 helpers); this is the shared
# neuroevolution layer, not a third copy of its seeding/graph modules.
_SHARED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neuroevolution")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from evox_runtime import (  # noqa: E402  (path set up above, as a script would)
    POP_OPERANDS,
    _attach,
    _constant,
    _device_name,
    _outputs,
    _positive_int,
    _staged,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)
from seeding import EPISODES_PER_POLICY  # noqa: E402  (path set up above)

# --- Task constants (``hover_train.get_cfgs()`` / ``hover_env.py``) ------------------

NUM_OBS = 17
NUM_ACTIONS = 4
NUM_COMMANDS = 3
DT = 0.01
#: ``ceil(15 / 0.01)`` -- the timeout fires *above* this many steps.
MAX_EPISODE_LENGTH = 1500
CLIP_ACTIONS = 1.0
HOVER_RPM = 14468.429183500699
ACTION_GAIN = 0.8
AT_TARGET_THRESHOLD = 0.1
#: ``obs_cfg["obs_scales"]``.
OBS_SCALES = {"rel_pos": 1.0 / 3.0, "lin_vel": 1.0 / 3.0, "ang_vel": 1.0 / 3.14159}
#: ``reward_cfg["reward_scales"]`` **before** the per-step ``dt`` scaling.
REWARD_SCALES = {"target": 10.0, "smooth": -1e-4, "yaw": 0.01, "angular": -2e-4, "crash": -10.0}
YAW_LAMBDA = -10.0
#: ``env_cfg`` termination thresholds (degrees for the two Euler checks).
TERMINATION = {"roll_deg": 180.0, "pitch_deg": 180.0, "ground_z": 0.1, "x_m": 3.0, "y_m": 3.0, "z_m": 2.0}
BASE_INIT_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
BASE_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
#: ``gs.EPS`` at the default (float32) precision.
QUAT_EPS = 1.1920928955078125e-07
#: Why an episode row stopped; the env module reports these to the trainer.
END_REASONS = {"unfinished": 0, "timeout": 1, "crash": 2, "error": 3}

_DEG_PER_RAD = 180.0 / math.pi
#: Odd minimax polynomial of degree 15 of ``arctan`` on ``[0, 1]`` (``qd_atan2_fast``).
_ATAN2_TAIL = (-0.0040545664, 0.0218629568, -0.0559123287, 0.0964219782, -0.1390862994, 0.1994656579, -0.3332986081)
_ATAN2_CONST = 0.9999993356
#: The largest float32 angle within ``[-pi, pi]``: ``pi`` rounded down.
_ATAN2_ANGLE_NEG_X = float(np.nextafter(np.float32(math.pi), np.float32(0.0)))

REWARD_TERM_NAMES = ("target", "smooth", "yaw", "angular", "crash")
#: ``inv_quat(BASE_INIT_QUAT)`` as a ``(1, 4)`` row: the left factor of the relative rotation.
_BASE_INIT_QUAT_INV = np.ascontiguousarray((BASE_INIT_QUAT * np.array([1.0, -1.0, -1.0, -1.0], np.float32))[None, :])


# --- Small graph helpers ------------------------------------------------------------


def _int32(value: int) -> evoxc.SymbolicTensor:
    """Embed a Python int as a 0-d int32 constant (plain ``int`` arithmetic promotes to int64)."""
    return _constant(np.array(value, dtype=np.int32))


def _quat_columns(quat: evoxc.SymbolicTensor):
    """Split ``(rows, 4)`` quaternions into four ``(rows,)`` component columns."""
    return quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]


def _inv_quat_rows(quat: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Row-wise ``[w, -x, -y, -z]`` (a conjugate, NOT normalized)."""
    qw, qx, qy, qz = _quat_columns(quat)
    return evoxc.stack([qw, -qx, -qy, -qz], axis=-1)


def _rotate_rows(v: evoxc.SymbolicTensor, quat: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Rotate ``(rows, 3)`` vectors by the raw quaternion, dividing by ``|q|^2`` (never normalized)."""
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


def _quat_mul_rows(u: evoxc.SymbolicTensor, v: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Normalized Hamilton product ``u * v`` (``u`` may be a single ``(1, 4)`` row).

    Transcribes ``genesis.utils.geom._np_quat_mul`` and the normalization of
    ``transform_quat_by_quat``.
    """
    w1, x1, y1, z1 = _quat_columns(u)
    w2, x2, y2, z2 = _quat_columns(v)
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    quat = evoxc.stack(
        [
            qq - ww + (z1 - y1) * (y2 - z2),
            qq - xx + (x1 + w1) * (x2 + w2),
            qq - yy + (w1 - x1) * (y2 + z2),
            qq - zz + (z1 + y1) * (w2 - x2),
        ],
        axis=-1,
    )
    return quat / evoxc.norm(quat, axis=1, keepdims=True)


def _atan2_rows(y: evoxc.SymbolicTensor, x: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Row-wise four-quadrant ``atan2`` from the ``qd_atan2_fast`` polynomial.

    ``evoxc`` has no ``atan2``, so this transcribes the Quadrants helper: a degree-15 odd
    minimax polynomial of ``arctan(min/max)`` on ``[0, 1]`` mapped back through the octant
    symmetries (``abs``/``min``/``max``/``select`` only).  The diagonals get exactly
    ``pi/4``; the negative ``x`` axis with ``y <= 0`` gets the largest float32 angle within
    ``[-pi, pi]`` (``pi`` rounded down).
    """
    abs_x, abs_y = evoxc.abs(x), evoxc.abs(y)
    ratio = evoxc.minimum(abs_x, abs_y) / evoxc.maximum(evoxc.maximum(abs_x, abs_y), 1e-30)
    ratio_sqr = ratio * ratio
    angle = _constant(np.float32(_ATAN2_TAIL[0]))
    for coef in _ATAN2_TAIL[1:]:
        angle = angle * ratio_sqr + _constant(np.float32(coef))
    angle = ratio * (angle * ratio_sqr + _constant(np.float32(_ATAN2_CONST)))
    angle = evoxc.select(
        evoxc.greater_equal(ratio, 1.0),
        _constant(np.float32(0.25 * math.pi)),
        angle,
    )
    angle = evoxc.select(evoxc.greater(abs_y, abs_x), 0.5 * math.pi - angle, angle)
    angle = evoxc.select(
        evoxc.less(x, 0.0),
        evoxc.select(
            evoxc.greater(angle, 0.0),
            math.pi - angle,
            _constant(np.float32(_ATAN2_ANGLE_NEG_X)),
        ),
        angle,
    )
    return evoxc.select(evoxc.less(y, 0.0), -angle, angle)


def _quat_to_xyz_rows(quat: evoxc.SymbolicTensor):
    """``quat_to_xyz(quat, rpy=True, degrees=True)`` factored as ``genesis.utils.geom`` does.

    With ``s = 2 / |q|^2``: ``sinp = q_wy - q_xz``, ``sinrcosp = q_wx + q_yz``,
    ``sinycosp = q_wz + q_xy``, ``cosr_cosp = 1 - (q_xx + q_yy)``,
    ``cosycosp = 1 - (q_yy + q_zz)``, ``cosp = sqrt(cosycosp^2 + sinycosp^2)``; at
    ``cosp < QUAT_EPS`` (gimbal lock) roll is zero and yaw becomes
    ``atan2(q_wz - q_xy, 1 - (q_xx + q_zz))``.  Returns ``(roll_deg, pitch_deg, yaw_deg)``
    as ``(rows,)`` columns -- only yaw is consumed downstream.
    """
    qw, qx, qy, qz = _quat_columns(quat)
    # Sequential order ``((a + b) + c) + d`` matches ``np.sum``'s small-array reduction.
    s = 2.0 / (((qw * qw + qx * qx) + qy * qy) + qz * qz)
    q_ws, q_xs, q_ys, q_zs = qw * s, qx * s, qy * s, qz * s
    q_wx, q_wy, q_wz = qw * q_xs, qw * q_ys, qw * q_zs
    q_xx, q_xy, q_xz = qx * q_xs, qx * q_ys, qx * q_zs
    q_yy, q_yz, q_zz = qy * q_ys, qy * q_zs, qz * q_zs

    sinp = q_wy - q_xz
    sinrcosp = q_wx + q_yz
    sinycosp = q_wz + q_xy
    cosr_cosp = 1.0 - (q_xx + q_yy)
    cosycosp = 1.0 - (q_yy + q_zz)
    cosp = evoxc.sqrt(cosycosp * cosycosp + sinycosp * sinycosp)

    gimbal = evoxc.less(cosp, QUAT_EPS)
    roll = evoxc.select(gimbal, _constant(np.float32(0.0)), _atan2_rows(sinrcosp, cosr_cosp))
    pitch = _atan2_rows(sinp, cosp)
    yaw = evoxc.select(
        gimbal,
        _atan2_rows(q_wz - q_xy, 1.0 - (q_xx + q_zz)),
        _atan2_rows(sinycosp, cosycosp),
    )
    return roll * _DEG_PER_RAD, pitch * _DEG_PER_RAD, yaw * _DEG_PER_RAD


def _obs_rows(rel_pos, base_quat, lin_vel_body, ang_vel_body, last_actions):
    """The 17-dim hover observation rows, written once for ``obs`` and ``advance``.

    ``[clip(rel_pos / 3), base_quat, clip(v_body / 3), clip(w_body / 3.14159), last_actions]``
    with the raw, absolute ``drone.get_quat()`` in the quaternion slot.
    """
    return evoxc.concatenate(
        [
            evoxc.clamp(rel_pos * OBS_SCALES["rel_pos"], -1.0, 1.0),
            base_quat,
            evoxc.clamp(lin_vel_body * OBS_SCALES["lin_vel"], -1.0, 1.0),
            evoxc.clamp(ang_vel_body * OBS_SCALES["ang_vel"], -1.0, 1.0),
            last_actions,
        ],
        axis=1,
    )


# --- Static specialization arrays ---------------------------------------------------

#: ``(name, width, dtype)`` of every runtime operand of ``hover_advance``, in spec order.
_ADVANCE_OPERANDS = (
    ("commands", NUM_COMMANDS, np.float32),
    ("base_pos", 3, np.float32),
    ("last_base_pos", 3, np.float32),
    ("base_quat", 4, np.float32),
    ("base_lin_vel_world", 3, np.float32),
    ("base_ang_vel_world", 3, np.float32),
    ("actions", NUM_ACTIONS, np.float32),
    ("last_actions", NUM_ACTIONS, np.float32),
    ("episode_length", None, np.int32),
    ("resampled_commands", NUM_COMMANDS, np.float32),
    ("error_mask", None, bool),
    ("alive", None, bool),
    ("returns", None, np.float32),
    # Appended last so the spec order of the older operands never moves: the latched
    # ``END_REASONS`` code of the row (0 = unfinished) carried by a device-resident caller.
    ("end_code", None, np.int8),
)

# --- Controls -----------------------------------------------------------------------


@evoxc.defn
def hover_controls(raw_actions):
    """Clip the raw action and turn it into rotor RPMs.

    Split out of ``hover_advance`` because the RPMs must be written to the drone
    *before* ``scene.step()`` while every other quantity is measured *after* it.

    Returns:
        tuple: ``actions`` (the clipped actions, ``(rows, 4)``) and ``rpm`` (``(rows, 4)``).
    """
    actions = evoxc.clamp(raw_actions, -CLIP_ACTIONS, CLIP_ACTIONS)
    rpm = (1.0 + ACTION_GAIN * actions) * HOVER_RPM
    return actions, rpm


def build_controls(rows, *, backend="numpy", device=None):
    """Compile the action -> RPM graph for ``rows`` parallel environments."""
    rows = _positive_int(rows, "rows")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    executable = evoxc.build(
        hover_controls,
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="raw_actions"),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(executable, graph="controls", rows=rows, device_name=_device_name(resolved_device))


def run_controls(executable, actions):
    """Run the controls graph on ``(rows, 4)`` raw actions; returns clipped actions and RPMs."""
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "actions", actions, NUM_ACTIONS)),
    )
    clipped, rpm = _outputs(outputs)
    return {
        "actions": np.asarray(_to_numpy(clipped), dtype=np.float32),
        "rpm": np.asarray(_to_numpy(rpm), dtype=np.float32),
    }


# --- Observation --------------------------------------------------------------------


@evoxc.defn
def hover_obs(commands, base_pos, base_quat, base_lin_vel_world, base_ang_vel_world, last_actions):
    """The 17-dim observation of the current state (``hover_env._update_observation``)."""
    rel_pos = commands - base_pos
    inv_quat = _inv_quat_rows(base_quat)
    lin_vel_body = _rotate_rows(base_lin_vel_world, inv_quat)
    ang_vel_body = _rotate_rows(base_ang_vel_world, inv_quat)
    return _obs_rows(rel_pos, base_quat, lin_vel_body, ang_vel_body, last_actions)


def build_obs(rows, *, backend="numpy", device=None):
    """Compile the observation graph for ``rows`` parallel environments."""
    rows = _positive_int(rows, "rows")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    executable = evoxc.build(
        hover_obs,
        evoxc.TensorSpec((rows, NUM_COMMANDS), evoxc.float32, name="commands"),
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_pos"),
        evoxc.TensorSpec((rows, 4), evoxc.float32, name="base_quat"),
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_lin_vel_world"),
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_ang_vel_world"),
        evoxc.TensorSpec((rows, NUM_ACTIONS), evoxc.float32, name="last_actions"),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(executable, graph="obs", rows=rows, device_name=_device_name(resolved_device))


def run_obs(executable, *, commands, base_pos, base_quat, base_lin_vel_world, base_ang_vel_world, last_actions):
    """Run the observation graph; returns ``{"obs": (rows, 17)}``.

    The velocities are **world**-frame (``drone.get_vel()`` / ``drone.get_ang()``); the
    graph rotates them into the body frame, and ``base_quat`` is the raw absolute
    ``drone.get_quat()`` (never normalized).
    """
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "commands", commands, NUM_COMMANDS)),
        _to_device_array(executable, _staged(executable, "base_pos", base_pos, 3)),
        _to_device_array(executable, _staged(executable, "base_quat", base_quat, 4)),
        _to_device_array(executable, _staged(executable, "base_lin_vel_world", base_lin_vel_world, 3)),
        _to_device_array(executable, _staged(executable, "base_ang_vel_world", base_ang_vel_world, 3)),
        _to_device_array(executable, _staged(executable, "last_actions", last_actions, NUM_ACTIONS)),
    )
    return {"obs": np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)}


# --- Reset observation --------------------------------------------------------------


@evoxc.defn
def hover_reset_obs(commands, base_init_pos, base_init_quat):
    """The analytic post-``reset()`` observation, with no ``scene.step()``.

    ``reset_idx`` pins the base to the init pose and zeroes the velocities,
    ``last_actions`` and the counter, so the observation reduces to
    ``[clip((commands - base_init_pos) / 3), base_init_quat, zeros(3), zeros(3), zeros(4)]``
    -- the eval environment can start a generation without stepping the simulator.

    The pose is a **runtime operand** (one row per environment), not a baked constant, so a
    perturbed reset pose is supported by the same graph.
    """
    rows = commands.shape[0]
    rel_pos = commands - base_init_pos
    zeros3 = evoxc.numpy.zeros((rows, 3), evoxc.float32)
    zeros4 = evoxc.numpy.zeros((rows, 4), evoxc.float32)
    return evoxc.concatenate(
        [
            evoxc.clamp(rel_pos * OBS_SCALES["rel_pos"], -1.0, 1.0),
            base_init_quat,
            zeros3,
            zeros3,
            zeros4,
        ],
        axis=1,
    )


def build_reset_obs(rows, *, backend="numpy", device=None):
    """Compile the analytic reset-observation graph for ``rows`` environments."""
    rows = _positive_int(rows, "rows")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    executable = evoxc.build(
        hover_reset_obs,
        evoxc.TensorSpec((rows, NUM_COMMANDS), evoxc.float32, name="commands"),
        evoxc.TensorSpec((rows, 3), evoxc.float32, name="base_init_pos"),
        evoxc.TensorSpec((rows, 4), evoxc.float32, name="base_init_quat"),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(executable, graph="reset_obs", rows=rows, device_name=_device_name(resolved_device))


def run_reset_obs(executable, *, commands, base_init_pos, base_init_quat):
    """Run the reset-observation graph; returns ``{"obs": (rows, 17)}``.

    ``base_init_pos`` / ``base_init_quat`` are the per-row pose the rows were reset to (the
    authored ``BASE_INIT_POS`` / ``BASE_INIT_QUAT``, or a perturbation of it).
    """
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "commands", commands, NUM_COMMANDS)),
        _to_device_array(executable, _staged(executable, "base_init_pos", base_init_pos, 3)),
        _to_device_array(executable, _staged(executable, "base_init_quat", base_init_quat, 4)),
    )
    return {"obs": np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)}


# --- Advance ------------------------------------------------------------------------


@evoxc.defn
def hover_advance(
    commands,
    base_pos,
    last_base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    actions,
    last_actions,
    episode_length,
    resampled_commands,
    error_mask,
    alive,
    returns,
    end_code,
    max_episode_length,
):
    """One hover step on the state measured after ``scene.step()``.

    ``actions`` are the **clipped** actions from ``hover_controls``; ``resampled_commands``
    is this step's host draw, selected only for the rows whose ``norm(rel_pos)`` fell below
    ``AT_TARGET_THRESHOLD``; ``error_mask`` is the external solver-failure signal; ``end_code``
    is the row's already-latched ``END_REASONS`` code (0 = unfinished).  Rows that are already
    ``done`` still produce a consistent step; the caller stops crediting them.
    """
    rows = commands.shape[0]
    episode_length_next = evoxc.cast(episode_length + _int32(1), evoxc.int32)

    # Pre-resample geometry, in the baseline's order.
    rel_pos = commands - base_pos
    last_rel_pos = commands - last_base_pos
    inv_quat = _inv_quat_rows(base_quat)
    lin_vel_body = _rotate_rows(base_lin_vel_world, inv_quat)
    ang_vel_body = _rotate_rows(base_ang_vel_world, inv_quat)
    # ``transform_quat_by_quat(inv(base_init_quat), base_quat)``: a constant left factor.
    relative_quat = _quat_mul_rows(_constant(_BASE_INIT_QUAT_INV), base_quat)
    yaw_deg = _quat_to_xyz_rows(relative_quat)[2]

    # Command resampling: only the rows that reached their target take the host draw.
    at_target = evoxc.less(evoxc.norm(rel_pos, axis=1), AT_TARGET_THRESHOLD)
    commands_next = evoxc.select(
        evoxc.reshape(at_target, (rows, 1)),
        resampled_commands,
        commands,
    )
    rel_pos_next = commands_next - base_pos
    last_rel_pos_next = commands_next - last_base_pos

    # Termination.  The roll/pitch 180-degree checks are omitted: Euler roll is bounded
    # by 180 and pitch by 90, so neither comparison can ever be true.
    crash = evoxc.logical_or(
        evoxc.logical_or(
            evoxc.logical_or(
                evoxc.greater(evoxc.abs(rel_pos_next[:, 0]), TERMINATION["x_m"]),
                evoxc.greater(evoxc.abs(rel_pos_next[:, 1]), TERMINATION["y_m"]),
            ),
            evoxc.greater(evoxc.abs(rel_pos_next[:, 2]), TERMINATION["z_m"]),
        ),
        evoxc.logical_or(
            evoxc.less(base_pos[:, 2], TERMINATION["ground_z"]),
            error_mask,
        ),
    )
    timeout = evoxc.greater(episode_length_next, _int32(max_episode_length))
    done = evoxc.logical_or(crash, timeout)

    # Reward terms, on the post-resample geometry, each scaled by ``dt`` once.
    target = evoxc.reduce_sum(last_rel_pos_next * last_rel_pos_next, axes=1) - evoxc.reduce_sum(
        rel_pos_next * rel_pos_next, axes=1
    )
    smooth = evoxc.reduce_sum((actions - last_actions) * (actions - last_actions), axes=1)
    yaw = evoxc.exp(YAW_LAMBDA * evoxc.abs(yaw_deg / 180.0 * 3.14159))
    angular = evoxc.norm(ang_vel_body / 3.14159, axis=1)
    crash_term = evoxc.cast(crash, evoxc.float32)

    target_rew = DT * REWARD_SCALES["target"] * target
    smooth_rew = DT * REWARD_SCALES["smooth"] * smooth
    yaw_rew = DT * REWARD_SCALES["yaw"] * yaw
    angular_rew = DT * REWARD_SCALES["angular"] * angular
    crash_rew = DT * REWARD_SCALES["crash"] * crash_term
    total = target_rew + smooth_rew + yaw_rew + angular_rew + crash_rew

    obs = _obs_rows(rel_pos_next, base_quat, lin_vel_body, ang_vel_body, last_actions)

    # ``returns`` is credited on the pre-step ``alive`` state, and error rows are never
    # credited (their measured state is meaningless).
    credit = evoxc.logical_and(alive, evoxc.logical_not(error_mask))
    zero = evoxc.numpy.zeros((rows,), evoxc.float32)
    returns_next = returns + evoxc.select(credit, total, zero)
    alive_next = evoxc.logical_and(alive, evoxc.logical_not(done))

    # The first ending is latched, exactly like the host rule this graph replaced:
    # error wins over timeout wins over crash, and an already-tagged row is never re-armed.
    code_this = evoxc.select(
        error_mask,
        _int32(END_REASONS["error"]),
        evoxc.select(timeout, _int32(END_REASONS["timeout"]), _int32(END_REASONS["crash"])),
    )
    first = evoxc.logical_and(evoxc.equal(end_code, _constant(np.array(0, dtype=np.int8))), done)
    end_code_next = evoxc.cast(evoxc.select(first, code_this, evoxc.cast(end_code, evoxc.int32)), evoxc.int8)
    # A column stack, not the operand itself: the caller may keep this tensor while the next
    # step overwrites the position it was read from.
    base_pos_snapshot = evoxc.stack([base_pos[:, 0], base_pos[:, 1], base_pos[:, 2]], axis=-1)

    return [
        obs,
        target_rew, smooth_rew, yaw_rew, angular_rew, crash_rew, total,
        crash, timeout, done, at_target, commands_next,
        episode_length_next, returns_next, alive_next, yaw_deg,
        end_code_next, base_pos_snapshot,
    ]


def build_advance(rows, max_episode_length=MAX_EPISODE_LENGTH, *, backend="numpy", device=None):
    """Compile the one-step advance graph for ``rows`` environments.

    ``max_episode_length`` is the episode cap in control steps (``ceil(15 / 0.01)``);
    the timeout fires once the counter exceeds it, as in ``hover_env``.
    """
    rows = _positive_int(rows, "rows")
    max_episode_length = _positive_int(max_episode_length, "max_episode_length")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    specs = [
        evoxc.TensorSpec((rows,) if width is None else (rows, width), evoxc.dtype(dtype), name=name)
        for name, width, dtype in _ADVANCE_OPERANDS
    ]
    executable = evoxc.build(
        hover_advance, *specs, max_episode_length, backend=resolved_backend, device=resolved_device
    )
    return _attach(
        executable,
        graph="advance",
        rows=rows,
        max_episode_length=max_episode_length,
        device_name=_device_name(resolved_device),
    )


def run_advance(
    executable,
    *,
    commands,
    base_pos,
    last_base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    actions,
    last_actions,
    episode_length,
    resampled_commands,
    error_mask,
    alive,
    returns,
    end_code=None,
):
    """Run one advance step: observation, six reward terms and bookkeeping.

    ``base_pos`` / ``last_base_pos`` are world positions measured after the step;
    ``base_quat`` is the raw absolute ``drone.get_quat()``; the velocities are world frame
    (``drone.get_vel()`` / ``drone.get_ang()``); ``actions`` are the *clipped* actions from
    ``hover_controls``; ``episode_length`` is the pre-step int32 counter;
    ``resampled_commands`` is this step's host draw; ``error_mask`` flags external
    failures; ``alive`` / ``returns`` are the pre-step bookkeeping; ``end_code`` is the
    row's latched ``END_REASONS`` code and defaults to all-unfinished, which reproduces the
    pre-latch behaviour for a caller that re-derives the endings on the host.

    Returns ``obs`` (rows, 17), ``reward_terms`` (five ``dt``-scaled terms plus ``total``),
    the ``crash``/``timeout``/``done``/``at_target``/``alive`` bool masks, the next
    ``commands`` (rows, 3), ``episode_length`` (int32), ``returns``, ``yaw_deg``, the next
    latched ``end_code`` (int8) and the ``base_pos`` the step consumed (rows, 3, a fresh
    buffer, so a device-resident caller can carry the previous position without a host
    round-trip).
    """
    values = (
        commands,
        base_pos,
        last_base_pos,
        base_quat,
        base_lin_vel_world,
        base_ang_vel_world,
        actions,
        last_actions,
        episode_length,
        resampled_commands,
        error_mask,
        alive,
        returns,
        np.zeros(np.shape(commands)[0], dtype=np.int8) if end_code is None else end_code,
    )
    outputs = evoxc.run(
        executable,
        *[
            _to_device_array(executable, _staged(executable, name, value, width, dtype=dtype))
            for (name, width, dtype), value in zip(_ADVANCE_OPERANDS, values, strict=True)
        ],
        executable.max_episode_length,
    )
    flat = [_to_numpy(leaf) for leaf in _outputs(outputs)]
    # Leaves 1..6 are the reward terms in ``REWARD_TERM_NAMES`` order, then ``total``.
    reward_terms = {
        name: flat[index].astype(np.float32) for index, name in enumerate((*REWARD_TERM_NAMES, "total"), start=1)
    }
    return {
        "obs": flat[0].astype(np.float32),
        "reward_terms": reward_terms,
        "crash": flat[7].astype(bool),
        "timeout": flat[8].astype(bool),
        "done": flat[9].astype(bool),
        "at_target": flat[10].astype(bool),
        "commands": flat[11].astype(np.float32),
        "episode_length": flat[12].astype(np.int32),
        "returns": flat[13].astype(np.float32),
        "alive": flat[14].astype(bool),
        "yaw_deg": flat[15].astype(np.float32),
        "end_code": flat[16].astype(np.int8),
        "base_pos": flat[17].astype(np.float32),
    }


# --- Self-check ---------------------------------------------------------------------


def _rpy_quat(roll, pitch, yaw):
    """Unit quaternions ``(n, 4)`` from intrinsic x-y-z Euler angles."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    ).astype(np.float32)


def _random_unit_quats(rng, n):
    """``(n, 4)`` float32 unit quaternions drawn uniformly on the 3-sphere."""
    quat = rng.normal(size=(n, 4))
    return (quat / np.linalg.norm(quat, axis=-1, keepdims=True)).astype(np.float32)


def _f32(*rows):
    """A float32 ``(len(rows), len(rows[0]))`` array from row sequences."""
    return np.array(rows, dtype=np.float32)


def _random_commands(rng, n):
    """``(n, 3)`` commands from the ``hover_env`` ranges (``z`` pinned to 1.0)."""
    return np.stack([rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), np.ones(n)], axis=-1).astype(np.float32)


def _neutral_state(rows):
    """An all-zero hover state; a check overrides only the fields it exercises."""
    zero3 = np.zeros((rows, 3), dtype=np.float32)
    zero4 = np.zeros((rows, NUM_ACTIONS), dtype=np.float32)
    return dict(
        commands=zero3, base_pos=zero3, last_base_pos=zero3, resampled_commands=zero3,
        base_quat=np.tile(BASE_INIT_QUAT, (rows, 1)), base_lin_vel_world=zero3, base_ang_vel_world=zero3,
        actions=zero4, last_actions=zero4, episode_length=np.zeros((rows,), dtype=np.int32),
        error_mask=np.zeros((rows,), dtype=bool), end_code=np.zeros((rows,), dtype=np.int8),
    )


def _latched_end_codes(end_code, done, timeout, error_mask):
    """Numpy replication of the graph's first-ending latch (error > timeout > crash, once)."""
    code_this = np.where(
        error_mask,
        END_REASONS["error"],
        np.where(timeout, END_REASONS["timeout"], END_REASONS["crash"]),
    )
    first = (np.asarray(end_code) == END_REASONS["unfinished"]) & done
    return np.where(first, code_this, end_code).astype(np.int8)


def _max_term_error(ref, stepped, reference):
    """Largest absolute difference over the six ``dt``-scaled reward terms."""
    expected = {name: ref.DT * ref.REWARD_SCALES[name] * reference["reward_terms"][name] for name in REWARD_TERM_NAMES}
    expected["total"] = reference["reward_terms"]["total"]
    return max(float(np.abs(stepped["reward_terms"][name] - expected[name]).max()) for name in expected)


def _self_check():
    """Parity check of every graph against ``hover_legacy_reference`` (fast, no genesis).

    The fitness/seeding assertions at the end use the shared layer's builders (imported here,
    not copied) so the drone task keeps validating the exact graphs it runs with.
    """
    import hover_legacy_reference as ref

    from evolution_graphs import build_fitness, run_fitness
    from seeding import evolution_seed, generation_seed_table

    rng = np.random.default_rng(20241109)

    # (a) The yaw transcription against ``math.atan2`` on 20k random quaternions plus 200
    # near-gimbal-lock ones (pitch within 1e-3 rad of +-90 degrees).
    n_random, n_gimbal = 20000, 200
    gimbal_quats = _rpy_quat(
        rng.uniform(-3.1, 3.1, n_gimbal),
        np.pi / 2.0 - 1e-3 * rng.random(n_gimbal),
        rng.uniform(-3.1, 3.1, n_gimbal),
    )
    quats = np.concatenate([_random_unit_quats(rng, n_random), gimbal_quats], axis=0)
    rows = quats.shape[0]
    check = run_advance(
        build_advance(rows),
        **{**_neutral_state(rows), "base_quat": quats},
        alive=np.ones(rows, dtype=bool),
        returns=np.zeros(rows, dtype=np.float32),
    )
    # The reference yaw is the atan2 of the same factorisation of the same normalized
    # relative quaternion the graph builds.  The polynomial is within ~4e-8 rad of
    # ``math.atan2``, but both angles are then rounded to float32 degrees, and one ulp at
    # +-180 deg is 1.526e-5 deg -- the tightest achievable bound on the stored values.
    relative = ref.quat_mul(ref.inv_quat(ref.BASE_INIT_QUAT)[None, :], quats)
    expected_yaw = ref.quat_to_xyz(relative, rpy=True, degrees=True)[:, 2]
    yaw_err = np.abs(check["yaw_deg"] - expected_yaw)
    assert np.all(yaw_err <= float(np.spacing(np.float32(180.0))) * 1.01), f"yaw parity failed: {yaw_err.max():.3e}"

    # (b) Controls: the zero action is exactly the hover RPM and the clip saturates.
    raw = np.array([[0.0, 0.0, 0.0, 0.0], [5.0, -5.0, 2.0, -2.0], [0.5, -0.5, 0.25, -0.25]], dtype=np.float32)
    ctrl = run_controls(build_controls(3), raw)
    assert np.allclose(ctrl["rpm"][0], ref.HOVER_RPM, rtol=1e-7), ctrl["rpm"][0]
    assert np.allclose(ctrl["actions"][1], [1.0, -1.0, 1.0, -1.0]), ctrl["actions"][1]
    assert np.allclose(ctrl["rpm"], ref.propeller_rpm(raw), rtol=1e-7), ctrl["rpm"]

    # (c) Fixed-state parity for the step, observation and reset graphs.  Row 2 crashes
    # on ``|rel_pos.x| > 3``, row 3 is at its target (so the resampled draw wins), and
    # rows 1/2 also exercise the error mask and the timeout; row 3 is already tagged as
    # crashed, so its code must stay latched instead of being re-armed.
    fixed_rows = 4
    commands = _f32([0.2, -0.1, 1.0], [-0.4, 0.3, 1.0], [3.5, 0.0, 1.0], [0.05, 0.05, 1.0])
    base_pos = commands - _f32([0.05, 0.02, 0.3], [-0.5, 0.4, -0.6], [0.0, 0.0, 0.0], [0.02, -0.03, 0.04])
    state = {
        "commands": commands,
        "base_pos": base_pos,
        "last_base_pos": base_pos + _f32([0.01, 0.0, 0.02], [0.0, -0.01, 0.0], [0.0, 0.0, 0.0], [-0.01, 0.0, 0.0]),
        "base_quat": _rpy_quat(*_f32([0.05, -0.03, 0.2, 0], [-0.04, 0.06, -0.1, 0.01], [0.3, -0.2, 0.05, 0])),
        "base_lin_vel_world": _f32([0.1, -0.2, 0.05], [1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [-0.05, 0.03, 0.02]),
        "base_ang_vel_world": _f32([0.02, -0.01, 0.03], [1.0, -1.0, 2.0], [0.0, 0.0, 0.0], [0.01, 0.02, -0.03]),
        "actions": _f32([0.1, -0.2, 0.3, 0.0], [1.0, -1.0, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0], [-0.4, 0.4, 0.0, 0.1]),
        "last_actions": _f32([0.0, 0.0, 0.0, 0.0], [0.9, -0.8, 0.4, -0.4], [0.1, 0.0, -0.1, 0.0],
                             [-0.3, 0.2, 0.0, 0.0]),
        "episode_length": np.array([0, 1499, 1500, 700], dtype=np.int32),
        "resampled_commands": _f32([0.5, -0.5, 1.0], [0.1, 0.2, 1.0], [-0.7, 0.8, 1.0], [0.9, 0.9, 1.0]),
        "error_mask": np.array([False, False, True, False]),
    }
    end_code = np.array([0, 0, 0, END_REASONS["crash"]], dtype=np.int8)
    reference = ref.step_maths(**state)
    stepped = run_advance(
        build_advance(fixed_rows, max_episode_length=ref.MAX_EPISODE_LENGTH),
        **state,
        alive=np.array([True, True, True, False]),
        returns=np.array([0.5, -0.25, 1.0, 2.0], dtype=np.float32),
        end_code=end_code,
    )
    obs_err = np.abs(stepped["obs"] - reference["obs"]).max()
    term_err = _max_term_error(ref, stepped, reference)
    assert obs_err <= 1e-6, f"observation parity failed: {obs_err:.3e}"
    assert term_err <= 1e-5, f"reward parity failed: {term_err:.3e}"
    for name in ("crash", "timeout", "done", "at_target"):
        assert np.array_equal(stepped[name], reference[name]), f"{name} masks disagree"
    assert np.array_equal(stepped["commands"], reference["commands"])
    assert np.allclose(stepped["yaw_deg"], reference["yaw_deg"], atol=1e-4)
    assert np.array_equal(stepped["episode_length"], state["episode_length"] + 1)
    # The endings are latched here and never re-armed; the returned position is the consumed one.
    expected_codes = _latched_end_codes(end_code, reference["done"], reference["timeout"], state["error_mask"])
    assert np.array_equal(stepped["end_code"], expected_codes), (
        f"end_code latch failed: {stepped['end_code'].tolist()} vs {expected_codes.tolist()}"
    )
    assert np.array_equal(stepped["base_pos"], base_pos), "base_pos passthrough is not the consumed position"

    carried = {key: state[key] for key in ("base_quat", "base_lin_vel_world", "base_ang_vel_world", "last_actions")}
    obs_only = run_obs(build_obs(fixed_rows), commands=commands, base_pos=base_pos, **carried)
    lin_world, ang_world, quat = (carried[k] for k in ("base_lin_vel_world", "base_ang_vel_world", "base_quat"))
    v_body, w_body = ref.body_frame_velocities(lin_world, ang_world, quat)
    obs_ref = ref.observation(commands - base_pos, quat, v_body, w_body, carried["last_actions"])
    obs_only_err = np.abs(obs_only["obs"] - obs_ref).max()
    assert obs_only_err <= 1e-6, f"hover_obs parity failed: {obs_only_err:.3e}"

    # The reset pose is a runtime operand: the authored pose reproduces the oracle, and a
    # perturbed one lands directly in the observation.
    init_pos = np.ascontiguousarray(np.tile(BASE_INIT_POS, (fixed_rows, 1)))
    init_quat = np.ascontiguousarray(np.tile(BASE_INIT_QUAT, (fixed_rows, 1)))
    reset_executable = build_reset_obs(fixed_rows)
    reset = run_reset_obs(reset_executable, commands=commands, base_init_pos=init_pos, base_init_quat=init_quat)
    reset_err = np.abs(reset["obs"] - ref.reset_observation(commands)).max()
    assert reset_err <= 1e-6, f"hover_reset_obs parity failed: {reset_err:.3e}"
    moved_quat = _rpy_quat(np.full(fixed_rows, 0.05), np.full(fixed_rows, -0.02), np.full(fixed_rows, 0.4))
    moved_pos = base_pos + np.float32(0.25)
    moved = run_reset_obs(
        reset_executable, commands=commands, base_init_pos=moved_pos, base_init_quat=moved_quat
    )["obs"]
    expected_reset = np.concatenate(
        [
            np.clip((commands - moved_pos) * OBS_SCALES["rel_pos"], -1.0, 1.0),
            moved_quat,
            np.zeros((fixed_rows, 10), np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert np.array_equal(moved, expected_reset), "the reset observation ignored the runtime init pose"

    # (d) Mask agreement on ~200 randomized plausible states.
    n_cases = 200
    rand_commands = _random_commands(rng, n_cases)
    offsets = np.stack([rng.uniform(-0.4, 0.4, n_cases), rng.uniform(-0.4, 0.4, n_cases),
                        rng.uniform(-0.5, 0.5, n_cases)], axis=-1).astype(np.float32)
    rand_base_pos = rand_commands - offsets
    near_target = rng.random(n_cases) < 0.2
    rand_base_pos[near_target] = rand_commands[near_target] + rng.uniform(
        -0.05, 0.05, (int(near_target.sum()), 3)
    ).astype(np.float32)
    rand_base_pos[rng.random(n_cases) < 0.1] += np.array([4.0, 0.0, 0.0], dtype=np.float32)
    rand_base_pos[rng.random(n_cases) < 0.1, 2] = 0.02
    rand_state = {
        "commands": rand_commands,
        "base_pos": rand_base_pos,
        "last_base_pos": (rand_base_pos + rng.uniform(-0.1, 0.1, (n_cases, 3))).astype(np.float32),
        "base_quat": _rpy_quat(
            rng.uniform(-1.5, 1.5, n_cases), rng.uniform(-1.4, 1.4, n_cases), rng.uniform(-3.1, 3.1, n_cases)
        ),
        "base_lin_vel_world": rng.uniform(-1.0, 1.0, (n_cases, 3)).astype(np.float32),
        "base_ang_vel_world": rng.uniform(-2.0, 2.0, (n_cases, 3)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, (n_cases, NUM_ACTIONS)).astype(np.float32),
        "last_actions": rng.uniform(-1.0, 1.0, (n_cases, NUM_ACTIONS)).astype(np.float32),
        "episode_length": rng.integers(0, 1600, n_cases).astype(np.int32),
        "resampled_commands": _random_commands(rng, n_cases),
        "error_mask": rng.random(n_cases) < 0.05,
    }
    rand_reference = ref.step_maths(**rand_state)
    # A third of the rows already carry an ending; only the unfinished ones may latch a new one.
    rand_end_code = np.where(
        rng.random(n_cases) < 1.0 / 3.0,
        rng.choice(np.array(list(END_REASONS.values()), dtype=np.int8), n_cases),
        np.int8(END_REASONS["unfinished"]),
    ).astype(np.int8)
    rand_out = run_advance(
        build_advance(n_cases),
        **rand_state,
        alive=rng.random(n_cases) < 0.9,
        returns=rng.uniform(-5.0, 5.0, n_cases).astype(np.float32),
        end_code=rand_end_code,
    )
    for name in ("crash", "timeout", "done", "at_target", "commands"):
        assert np.array_equal(rand_out[name], rand_reference[name]), f"{name} disagrees"
    rand_obs_err = np.abs(rand_out["obs"] - rand_reference["obs"]).max()
    rand_term_err = _max_term_error(ref, rand_out, rand_reference)
    assert rand_obs_err <= 1e-6, f"randomized observation parity failed: {rand_obs_err:.3e}"
    assert rand_term_err <= 1e-5, f"randomized reward parity failed: {rand_term_err:.3e}"
    rand_codes = _latched_end_codes(
        rand_end_code, rand_reference["done"], rand_reference["timeout"], rand_state["error_mask"]
    )
    assert np.array_equal(rand_out["end_code"], rand_codes), "randomized end_code latch disagrees"
    assert np.count_nonzero(rand_end_code) > 0, "the randomized states never exercise a latched ending"
    assert np.array_equal(rand_out["base_pos"], rand_state["base_pos"]), "randomized base_pos passthrough disagrees"

    # Fitness: the mean of each policy's three episode returns, and the shared seeds.
    fitness = run_fitness(build_fitness(2), np.array([1.0, 2.0, 3.0, -1.0, 0.0, 4.0], dtype=np.float32))
    assert np.allclose(fitness, [2.0, 1.0]), fitness
    table = generation_seed_table(7, 4)
    assert table.shape == (4, EPISODES_PER_POLICY) and np.array_equal(table[:2], generation_seed_table(7, 2))
    assert len(set(table[0].tolist())) == EPISODES_PER_POLICY and evolution_seed(7) != evolution_seed(8)

    print(f"yaw max error: {yaw_err.max():.3e} deg (random {yaw_err[:n_random].max():.3e}, "
          f"gimbal {yaw_err[n_random:].max():.3e})")
    print(f"fixed-state: obs {obs_err:.3e}, reward {term_err:.3e}, "
          f"obs-only {obs_only_err:.3e}, reset {reset_err:.3e}")
    print(f"randomized: obs {rand_obs_err:.3e}, reward {rand_term_err:.3e}, masks agree on {n_cases} states")
    print("ok")


if __name__ == "__main__":
    _self_check()
