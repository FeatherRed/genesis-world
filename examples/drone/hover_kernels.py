"""Drone hover graphs for the EvoX Compiler (``evoxc``) evaluation environment.

This module owns every compiled graph the drone-hover *eval* environment asks for, so
the environment module stays thin: it reads the simulator, hands the measured state to
a graph, and writes the policy targets (rotor RPMs) back.  Nothing here imports
``genesis`` or Torch, nothing here simulates, and nothing here trains; the module
speaks ``numpy`` and ``evoxc`` (plus TensorNEAT, imported lazily by the policy builder
that traces the genome).  The generic build/run helpers
(``resolve_backend_device``, ``_constant``, ``_to_device_array``, ``_to_numpy``) are
imported from the sibling Go2 example instead of being copied a third time.

Graphs (each compiled once, then reused every step and every generation)
-----------------------------------------------------------------------
``hover_controls``   raw actions -> ``(clipped actions, rotor RPMs)``.
``hover_obs``        measured base state -> the 17-dim policy observation row.
``hover_reset_obs``  commands -> the analytic post-``reset()`` observation row.
``hover_advance``    one environment step: geometry -> masks -> reward -> observation.
``hover_fitness``    per-episode returns -> per-policy fitness (episode mean).
``hover_policy``     genome payload + observations -> the raw actions of a generation.

The baseline being migrated is ``examples/drone/hover_env.py`` (torch + genesis +
tensordict), with its config in ``hover_train.get_cfgs()``; the parity oracle the graphs
are checked against is ``hover_legacy_reference`` in this directory.

Contracts
---------
Every ``build_*`` compiles for a fixed row count and stashes its static values on the
executable (``.rows``, ``.max_episode_length``, ...), so a ``run_*`` never guesses them.
``evoxc`` re-validates static operands with ``array_equal`` on every call, so a runner must
pass back exactly what its builder was given.  Rows are **policy-major**: rows ``3p, 3p+1,
3p+2`` are the ``EPISODES_PER_POLICY`` episodes of policy ``p``.

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

Seeding rules (shared with the Go2 pilot)
-----------------------------------------
``generation_seed_table`` gives every policy in a generation the *same ordered triple* of
episode seeds, a fresh triple per generation, and ``evolution_seed`` an independent evolution
stream.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

import evoxc

# The generic evoxc build/run helpers live in the sibling locomotion examples directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "locomotion"))

from go2_evox_kernels import (  # noqa: E402  (path set up above, as a script would)
    _constant,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)

# --- Task constants (``hover_train.get_cfgs()`` / ``hover_env.py``) ------------------

NUM_OBS = 17
NUM_ACTIONS = 4
NUM_COMMANDS = 3
DT = 0.01
#: Episodes one generation of policies is scored over; rows are policy-major.
EPISODES_PER_POLICY = 3
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


# --- Seeding (shared rules with the Go2 pilot) --------------------------------------


def generation_seed_table(master_seed, num_generations, episodes=EPISODES_PER_POLICY):
    """Return the ``(num_generations, episodes)`` int64 table of episode seeds.

    Every policy in one generation is scored over the *same ordered triple* of episode
    seeds (row ``g``), a fresh triple is drawn for each generation, and the draws come
    from the env stream of ``SeedSequence(master_seed).spawn(2)`` (the evolution stream
    is :func:`evolution_seed`, so it is independent).  One row is drawn at a time with
    ``rng.choice(2**31, size=episodes, replace=False)``: a row's seeds stay distinct and
    the table stays **prefix-stable** -- ``table(seed, N)[i]`` is identical for every
    ``N > i``.

    Returns:
        np.ndarray: ``(num_generations, episodes)`` int64, every row strictly distinct.
    """
    num_generations = int(num_generations)
    episodes = int(episodes)
    if num_generations < 0 or episodes < 1:
        raise ValueError(f"num_generations must be >= 0 and episodes >= 1, got {num_generations} and {episodes}")
    _, env_seq = np.random.SeedSequence(master_seed).spawn(2)
    rng = np.random.default_rng(env_seq)
    table = np.empty((num_generations, episodes), dtype=np.int64)
    for generation in range(num_generations):
        table[generation] = rng.choice(2**31, size=episodes, replace=False)
    return table


def evolution_seed(master_seed):
    """Return the unsigned 32-bit seed of the evolution stream of ``master_seed``.

    The sibling of the env stream :func:`generation_seed_table` draws from, so the
    trainer's evolution RNG is independent of -- and unaffected by -- the episode seeds.
    """
    evolution_seq, _ = np.random.SeedSequence(master_seed).spawn(2)
    return int(evolution_seq.generate_state(1, dtype=np.uint32)[0])


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


def _positive_int(value, name: str) -> int:
    """Validate a strict positive integer build argument."""
    ivalue = int(value)
    if ivalue != value or ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


def _device_name(device) -> str:
    """``"kind:index"`` of a resolved device, for cache keys and log lines."""
    return f"{device.kind}:{device.index}"


def _attach(executable, **metadata):
    """Stash build-time metadata on an ``evoxc`` executable (it has a ``__dict__``)."""
    for name, value in metadata.items():
        setattr(executable, name, value)
    return executable


def _rows_of(executable) -> int:
    """The row count a graph was compiled for."""
    rows = getattr(executable, "rows", None)
    if rows is None:
        raise TypeError(f"{executable!r} was not built by this module (no '.rows' metadata)")
    return int(rows)


def _stage(executable, name: str, value, width=None, *, dtype=np.float32) -> np.ndarray:
    """Stage one runtime argument, checking its shape against the compiled graph."""
    rows = _rows_of(executable)
    expected = (rows,) if width is None else (rows, width)
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected} to match the compiled graph, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _outputs(result) -> tuple:
    """Normalize an ``evoxc.run`` result: a single-leaf graph returns the tensor bare."""
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


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
        _to_device_array(executable, _stage(executable, "actions", actions, NUM_ACTIONS)),
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
        _to_device_array(executable, _stage(executable, "commands", commands, NUM_COMMANDS)),
        _to_device_array(executable, _stage(executable, "base_pos", base_pos, 3)),
        _to_device_array(executable, _stage(executable, "base_quat", base_quat, 4)),
        _to_device_array(executable, _stage(executable, "base_lin_vel_world", base_lin_vel_world, 3)),
        _to_device_array(executable, _stage(executable, "base_ang_vel_world", base_ang_vel_world, 3)),
        _to_device_array(executable, _stage(executable, "last_actions", last_actions, NUM_ACTIONS)),
    )
    return {"obs": np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)}


# --- Reset observation --------------------------------------------------------------


@evoxc.defn
def hover_reset_obs(commands, base_init_pos_tiled, base_init_quat_tiled):
    """The analytic post-``reset()`` observation, with no ``scene.step()``.

    ``reset_idx`` pins the base to ``BASE_INIT_POS`` / ``BASE_INIT_QUAT`` and zeroes the
    velocities, ``last_actions`` and the counter, so the observation reduces to
    ``[clip((commands - BASE_INIT_POS) / 3), init_quat, zeros(3), zeros(3), zeros(4)]`` --
    the eval environment can start a generation without stepping the simulator.
    """
    rows = commands.shape[0]
    rel_pos = commands - _constant(base_init_pos_tiled)
    zeros3 = evoxc.numpy.zeros((rows, 3), evoxc.float32)
    zeros4 = evoxc.numpy.zeros((rows, 4), evoxc.float32)
    return evoxc.concatenate(
        [
            evoxc.clamp(rel_pos * OBS_SCALES["rel_pos"], -1.0, 1.0),
            _constant(base_init_quat_tiled),
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
    base_init_pos_tiled = np.ascontiguousarray(np.tile(BASE_INIT_POS, (rows, 1)))
    base_init_quat_tiled = np.ascontiguousarray(np.tile(BASE_INIT_QUAT, (rows, 1)))
    executable = evoxc.build(
        hover_reset_obs,
        evoxc.TensorSpec((rows, NUM_COMMANDS), evoxc.float32, name="commands"),
        base_init_pos_tiled,
        base_init_quat_tiled,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="reset_obs",
        rows=rows,
        base_init_pos_tiled=base_init_pos_tiled,
        base_init_quat_tiled=base_init_quat_tiled,
        device_name=_device_name(resolved_device),
    )


def run_reset_obs(executable, commands):
    """Run the reset-observation graph; returns ``{"obs": (rows, 17)}``."""
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _stage(executable, "commands", commands, NUM_COMMANDS)),
        executable.base_init_pos_tiled,
        executable.base_init_quat_tiled,
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
    max_episode_length,
):
    """One hover step on the state measured after ``scene.step()``.

    ``actions`` are the **clipped** actions from ``hover_controls``; ``resampled_commands``
    is this step's host draw, selected only for the rows whose ``norm(rel_pos)`` fell below
    ``AT_TARGET_THRESHOLD``; ``error_mask`` is the external solver-failure signal.  Rows
    that are already ``done`` still produce a consistent step; the caller stops crediting them.
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

    return [
        obs,
        target_rew, smooth_rew, yaw_rew, angular_rew, crash_rew, total,
        crash, timeout, done, at_target, commands_next,
        episode_length_next, returns_next, alive_next, yaw_deg,
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
):
    """Run one advance step: observation, six reward terms and bookkeeping.

    ``base_pos`` / ``last_base_pos`` are world positions measured after the step;
    ``base_quat`` is the raw absolute ``drone.get_quat()``; the velocities are world frame
    (``drone.get_vel()`` / ``drone.get_ang()``); ``actions`` are the *clipped* actions from
    ``hover_controls``; ``episode_length`` is the pre-step int32 counter;
    ``resampled_commands`` is this step's host draw; ``error_mask`` flags external
    failures; ``alive`` / ``returns`` are the pre-step bookkeeping.  Returns ``obs``
    (rows, 17), ``reward_terms`` (five ``dt``-scaled terms plus ``total``), the
    ``crash``/``timeout``/``done``/``at_target``/``alive`` bool masks, the next
    ``commands`` (rows, 3), ``episode_length`` (int32), ``returns`` and ``yaw_deg``.
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
    )
    outputs = evoxc.run(
        executable,
        *[
            _to_device_array(executable, _stage(executable, name, value, width, dtype=dtype))
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
    }


# --- Fitness ------------------------------------------------------------------------


@evoxc.defn
def hover_fitness(returns, pop_size, episodes_per_policy):
    """Episode returns ``(pop_size * episodes_per_policy,)`` -> fitness ``(pop_size,)``."""
    return evoxc.reduce_mean(
        evoxc.reshape(returns, (pop_size, episodes_per_policy)),
        axes=1,
    )


def build_fitness(pop_size, episodes_per_policy=EPISODES_PER_POLICY, *, backend="numpy", device=None):
    """Compile the fitness graph (mean of one policy's episode returns), mirroring Go2.

    Rows are policy-major, so the reshape groups the ``episodes_per_policy`` episodes of
    every policy together.
    """
    pop_size = _positive_int(pop_size, "pop_size")
    episodes_per_policy = _positive_int(episodes_per_policy, "episodes_per_policy")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    rows = pop_size * episodes_per_policy
    executable = evoxc.build(
        hover_fitness,
        evoxc.TensorSpec((rows,), evoxc.float32, name="returns"),
        pop_size,
        episodes_per_policy,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="fitness",
        rows=rows,
        pop_size=pop_size,
        episodes_per_policy=episodes_per_policy,
        device_name=_device_name(resolved_device),
    )


def run_fitness(executable, returns):
    """Run the fitness graph; ``returns`` is ``(pop_size * episodes,)``, result ``(pop_size,)``."""
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _stage(executable, "returns", returns)),
        executable.pop_size,
        executable.episodes_per_policy,
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)


# --- Policy -------------------------------------------------------------------------

#: ``(name, ...)`` of the four ``tensorneat.genome.utils.GenomeTensors`` payload leaves.
POP_OPERANDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def _host(array) -> np.ndarray:
    """Materialize one genome-payload leaf (host array or ``evoxc`` tensor) as numpy."""
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(_to_numpy(array))


def _pop_specs(pop_arrays):
    """``TensorSpec`` list for the four ``GenomeTensors`` payload arrays."""
    arrays = tuple(pop_arrays)
    if len(arrays) != len(POP_OPERANDS):
        raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
    specs = []
    for array, name in zip(arrays, POP_OPERANDS, strict=True):
        arr = _host(array)
        specs.append(evoxc.TensorSpec(tuple(int(dim) for dim in arr.shape), evoxc.dtype(arr.dtype), name=name))
    return specs


def build_policy(algo, config, pop_arrays, rows, *, backend="numpy", device=None):
    """Compile the genome-payload + observation -> raw-action graph for ``rows`` rows.

    ``rows`` must be ``algo.pop_size * EPISODES_PER_POLICY`` and the rows are
    **policy-major**: rows ``3p, 3p+1, 3p+2`` are the three episodes of policy ``p``.
    ``prepare_forward(samples=EPISODES_PER_POLICY)`` tiles the *genome statics* of
    individual ``p`` to its three rows, so each row carries its own observation: the
    three episodes of a policy are independent and need not share a state.

    The observation and action widths come from the traced genome
    (``algo.genome.num_inputs`` / ``num_outputs``), not from a task constant -- the
    sibling ``go2_neat_kernels.build_policy`` bakes the Go2 widths (45 obs, 12 actions)
    into its spec and the staging check of its runner, so it cannot trace the drone's
    17 -> 4 genome.

    Args:
        algo: a TensorNEAT ``NEAT`` algorithm instance.
        config: the tensor-free ``pipeline._static_config(state)`` captured at build time.
        pop_arrays: the four genome payload arrays (``state.pop`` as numpy).
        rows (int): ``pop_size * EPISODES_PER_POLICY``.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_policy`` stages its operands.
    """
    from tensorneat.genome.utils import GenomeTensors

    n_policies = int(algo.pop_size)
    rows = _positive_int(rows, "rows")
    if rows != n_policies * EPISODES_PER_POLICY:
        raise ValueError(
            f"rows must be pop_size * EPISODES_PER_POLICY = {n_policies * EPISODES_PER_POLICY}, got {rows}"
        )
    num_obs = int(algo.genome.num_inputs)
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    specs = [*_pop_specs(pop_arrays), evoxc.TensorSpec((rows, num_obs), evoxc.float32, name="obs")]

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
        num_obs=num_obs,
        pop_shapes=tuple(_host(array).shape for array in pop_arrays),
        device_name=_device_name(resolved_device),
    )


def run_policy(executable, pop_arrays, obs):
    """Run the policy graph; returns raw (unclipped) actions ``(rows, num_actions)`` float32."""
    arrays = tuple(_host(array) for array in pop_arrays)
    if len(arrays) != len(POP_OPERANDS):
        raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
    for array, shape in zip(arrays, executable.pop_shapes, strict=True):
        if array.shape != shape:
            raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {array.shape}")
    outputs = evoxc.run(
        executable,
        *[_to_device_array(executable, np.ascontiguousarray(array)) for array in arrays],
        _to_device_array(executable, _stage(executable, "obs", obs, executable.num_obs)),
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)


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
        error_mask=np.zeros((rows,), dtype=bool),
    )


def _max_term_error(ref, stepped, reference):
    """Largest absolute difference over the six ``dt``-scaled reward terms."""
    expected = {name: ref.DT * ref.REWARD_SCALES[name] * reference["reward_terms"][name] for name in REWARD_TERM_NAMES}
    expected["total"] = reference["reward_terms"]["total"]
    return max(float(np.abs(stepped["reward_terms"][name] - expected[name]).max()) for name in expected)


def _self_check():
    """Parity check of every graph against ``hover_legacy_reference`` (fast, no genesis)."""
    import hover_legacy_reference as ref

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
    # rows 1/2 also exercise the error mask and the timeout.
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
    reference = ref.step_maths(**state)
    stepped = run_advance(
        build_advance(fixed_rows, max_episode_length=ref.MAX_EPISODE_LENGTH),
        **state,
        alive=np.array([True, True, True, False]),
        returns=np.array([0.5, -0.25, 1.0, 2.0], dtype=np.float32),
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

    carried = {key: state[key] for key in ("base_quat", "base_lin_vel_world", "base_ang_vel_world", "last_actions")}
    obs_only = run_obs(build_obs(fixed_rows), commands=commands, base_pos=base_pos, **carried)
    lin_world, ang_world, quat = (carried[k] for k in ("base_lin_vel_world", "base_ang_vel_world", "base_quat"))
    v_body, w_body = ref.body_frame_velocities(lin_world, ang_world, quat)
    obs_ref = ref.observation(commands - base_pos, quat, v_body, w_body, carried["last_actions"])
    obs_only_err = np.abs(obs_only["obs"] - obs_ref).max()
    assert obs_only_err <= 1e-6, f"hover_obs parity failed: {obs_only_err:.3e}"
    reset = run_reset_obs(build_reset_obs(fixed_rows), commands)
    reset_err = np.abs(reset["obs"] - ref.reset_observation(commands)).max()
    assert reset_err <= 1e-6, f"hover_reset_obs parity failed: {reset_err:.3e}"

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
    rand_out = run_advance(
        build_advance(n_cases),
        **rand_state,
        alive=rng.random(n_cases) < 0.9,
        returns=rng.uniform(-5.0, 5.0, n_cases).astype(np.float32),
    )
    for name in ("crash", "timeout", "done", "at_target", "commands"):
        assert np.array_equal(rand_out[name], rand_reference[name]), f"{name} disagrees"
    rand_obs_err = np.abs(rand_out["obs"] - rand_reference["obs"]).max()
    rand_term_err = _max_term_error(ref, rand_out, rand_reference)
    assert rand_obs_err <= 1e-6, f"randomized observation parity failed: {rand_obs_err:.3e}"
    assert rand_term_err <= 1e-5, f"randomized reward parity failed: {rand_term_err:.3e}"

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
