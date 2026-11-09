"""Parity oracle for the drone hover task: a numpy transcription of the pre-migration maths.

This module is the **oracle** the migrated EvoX Compiler graphs in
``examples/drone/hover_kernels.py`` are checked against.  Every function is a
line-by-line transcription of ``examples/drone/hover_env.py`` (plus the config in
``hover_train.get_cfgs()``), warts included: nothing here is a "fix", and where the
baseline drops, double-draws or never-reaches a branch, this transcription does the
same.

The module is deliberately dependency-free apart from ``numpy``: no Torch, no
``evoxc``, no ``genesis`` and no ``tensordict``, so it stays importable on machines
where the RL stack is not installed.  The dtype split of the baseline is kept --
``gs.tc_float`` is float32, so every function takes or produces float32 arrays --
and quaternions are never normalized inside ``transform_by_quat`` (the rotation
divides by ``|q|^2`` instead, exactly like ``genesis.utils.geom``).

Step semantics reproduced here
------------------------------
``step_maths`` is the **pre-reset, non-terminal** single-step semantics of
``HoverEnv.step`` (``hover_env.py:139-197``), in the baseline's exact order:

1. ``actions = clip(raw, +-1)`` and ``rpm = (1 + 0.8 * actions) * 14468.429183500699``;
2. ``scene.step()`` happens outside this module -- the caller hands in the measured
   post-step state;
3. ``episode_length += 1`` and ``rel_pos = commands - base_pos`` (pre-resample);
4. the base Euler angles come from ``quat_to_xyz(transform_quat_by_quat(inv(base_init_quat),
   base_quat), rpy=True, degrees=True)`` and the body velocities from
   ``transform_by_quat(world_vel, inv(base_quat))``;
5. commands are resampled for rows with ``norm(rel_pos) < 0.1`` (three uniforms per
   resampled row, in x, y, z order, ``z`` pinned to 1.0);
6. ``crash`` is evaluated on the **pre-resample** ``rel_pos`` and ``timeout`` on the
   incremented counter;
7. the reward terms are evaluated **after** the resample, so the ``target`` term
   compares the resampled command against both ``rel_pos`` and ``last_rel_pos``;
8. the observation uses the post-resample ``rel_pos`` and the *previous* step's
   ``actions`` (the baseline updates ``last_actions`` after the observation).

The baseline's extra post-reset reward pass is **not** reproduced: it is the one
semantic the batched eval model deliberately drops.  ``hover_env.py`` calls
``reset_idx`` before computing the reward, which zeroes ``last_actions`` and the
episode sums for reset rows and then adds one post-reset step to the new episode.
The eval model instead treats every row as a single episode (rows ``3p..3p+2`` are
the three episodes of policy ``p``) and replaces the reset step with the analytic
reset observation; see ``hover_kernels`` for the consequences.
"""

from __future__ import annotations

import numpy as np

# --- Constants (``hover_train.get_cfgs()`` / ``hover_env.py``) ----------------------

NUM_OBS = 17
NUM_ACTIONS = 4
NUM_COMMANDS = 3
DT = 0.01
EPISODE_LENGTH_S = 15.0
#: ``ceil(episode_length_s / dt)`` -- the timeout fires *above* this many steps.
MAX_EPISODE_LENGTH = 1500
CLIP_ACTIONS = 1.0
AT_TARGET_THRESHOLD = 0.1
#: ``env_cfg["base_init_pos"]``.
BASE_INIT_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
#: ``env_cfg["base_init_quat"]`` in ``(w, x, y, z)`` order.
BASE_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
#: ``obs_cfg["obs_scales"]``.
OBS_SCALES = {"rel_pos": 1.0 / 3.0, "lin_vel": 1.0 / 3.0, "ang_vel": 1.0 / 3.14159}
#: ``reward_cfg["reward_scales"]`` **before** ``hover_env`` multiplies each by ``dt``.
REWARD_SCALES = {"target": 10.0, "smooth": -1e-4, "yaw": 0.01, "angular": -2e-4, "crash": -10.0}
YAW_LAMBDA = -10.0
#: ``env_cfg`` termination thresholds (degrees for the two Euler checks).
TERMINATION = {
    "roll_deg": 180.0,
    "pitch_deg": 180.0,
    "ground_z": 0.1,
    "x_m": 3.0,
    "y_m": 3.0,
    "z_m": 2.0,
}
#: ``command_cfg``: ``pos_x_range``, ``pos_y_range``, ``pos_z_range``.
COMMAND_RANGES = {"pos_x": (-1.0, 1.0), "pos_y": (-1.0, 1.0), "pos_z": (1.0, 1.0)}
#: The base hover RPM ``hover_env`` hardcodes in ``step``.
HOVER_RPM = 14468.429183500699
#: The action -> RPM gain ``hover_env`` hardcodes in ``step``.
ACTION_GAIN = 0.8
#: ``gs.EPS`` at the default (float32) precision, i.e. ``float(np.finfo(np.float32).eps)``.
QUAT_EPS = 1.1920928955078125e-07


# --- Quaternion helpers (``genesis.utils.geom``) ------------------------------------


def inv_quat(quat):
    """``[w, -x, -y, -z]`` -- a conjugate, NOT normalized."""
    quat = np.asarray(quat, dtype=np.float32)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(u, v):
    """Hamilton product ``u * v``, normalized, as ``genesis.utils.geom`` does it.

    ``transform_quat_by_quat(v, u)`` is this function with the arguments swapped.
    The ``_np_quat_mul`` shear form is transcribed verbatim; the result is divided by
    its own Euclidean norm (the baseline's ``transform_quat_by_quat`` normalizes).
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    u2 = np.atleast_2d(u)
    v2 = np.atleast_2d(v)
    w1, x1, y1, z1 = u2[..., 0], u2[..., 1], u2[..., 2], u2[..., 3]
    w2, x2, y2, z2 = v2[..., 0], v2[..., 1], v2[..., 2], v2[..., 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out = np.empty(np.broadcast_shapes(u2.shape, v2.shape), dtype=np.float32)
    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)
    out /= np.sqrt(np.sum(np.square(out), axis=-1, keepdims=True))
    return out[0] if (u.ndim == 1 and v.ndim == 1) else out


def transform_quat_by_quat(v, u):
    """``quatmul(u, v)`` -- equivalent to ``R_u @ R_v``."""
    return quat_mul(u, v)


def transform_by_quat(v, quat):
    """Rotate ``v`` by the raw quaternion ``quat`` and divide by ``|q|^2``.

    Accepts a single ``(3,)`` vector with a ``(4,)`` quaternion or a batch of
    ``(rows, 3)`` vectors with ``(rows, 4)`` quaternions.  The quaternion is never
    normalized -- that is the point of the transcription.
    """
    v = np.asarray(v, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    single = v.ndim == 1 and quat.ndim == 1
    v2 = np.atleast_2d(v)
    q2 = np.atleast_2d(quat)

    vx, vy, vz = v2[..., 0], v2[..., 1], v2[..., 2]
    qw, qx, qy, qz = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    q_ww, q_wx, q_wy, q_wz = qw * qw, qw * qx, qw * qy, qw * qz
    q_xx, q_xy, q_xz = qx * qx, qx * qy, qx * qz
    q_yy, q_yz = qy * qy, qy * qz
    q_zz = qz * qz
    norm_sqr = q_ww + q_xx + q_yy + q_zz

    ox = vx * (q_xx + q_ww - q_yy - q_zz) + vy * (2.0 * q_xy - 2.0 * q_wz) + vz * (2.0 * q_xz + 2.0 * q_wy)
    oy = vx * (2.0 * q_wz + 2.0 * q_xy) + vy * (q_ww - q_xx + q_yy - q_zz) + vz * (2.0 * q_yz - 2.0 * q_wx)
    oz = vx * (2.0 * q_xz - 2.0 * q_wy) + vy * (2.0 * q_wx + 2.0 * q_yz) + vz * (q_ww - q_xx - q_yy + q_zz)
    out = np.stack([ox / norm_sqr, oy / norm_sqr, oz / norm_sqr], axis=-1)
    return out[0] if single else out


def quat_to_xyz(quat, rpy=True, degrees=True):
    """Quaternion -> Euler angles, transcribed from ``genesis.utils.geom._np_quat_to_xyz``.

    With ``rpy=True`` the angle order is ``[roll, pitch, yaw]`` about ``x, y, z``;
    the regular branch and the ``cosp < QUAT_EPS`` gimbal-lock branch (``roll = 0``
    and the singular yaw) are both reproduced.
    """
    quat = np.asarray(quat, dtype=np.float32)
    single = quat.ndim == 1
    quat_2d = np.atleast_2d(quat)
    out_2d = np.empty((quat_2d.shape[0], 3), dtype=np.float32)

    s = 2.0 / np.sum(np.square(quat_2d), axis=-1)
    q_w, q_x, q_y, q_z = quat_2d[:, 0], quat_2d[:, 1], quat_2d[:, 2], quat_2d[:, 3]
    q_ws, q_xs, q_ys, q_zs = q_w * s, q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

    if rpy:
        sinp = q_wy - q_xz
        sinrcosp = q_wx + q_yz
        sinycosp = q_wz + q_xy
    else:
        sinp = q_xz + q_wy
        sinrcosp = q_wx - q_yz
        sinycosp = q_wz - q_xy
    cosr_cosp = 1.0 - (q_xx + q_yy)
    cosycosp = 1.0 - (q_yy + q_zz)
    cosp = np.sqrt(cosycosp**2 + sinycosp**2)

    out_2d[:, 0] = np.arctan2(sinrcosp, cosr_cosp)
    out_2d[:, 1] = np.arctan2(sinp, cosp)
    out_2d[:, 2] = np.arctan2(sinycosp, cosycosp)

    cosp_mask = cosp < QUAT_EPS
    if rpy:
        singular_yaw_num = q_wz[cosp_mask] - q_xy[cosp_mask]
    else:
        singular_yaw_num = q_wz[cosp_mask] + q_xy[cosp_mask]
    singular_yaw_den = 1.0 - (q_xx[cosp_mask] + q_zz[cosp_mask])
    out_2d[cosp_mask, 0] = 0.0
    out_2d[cosp_mask, 2] = np.arctan2(singular_yaw_num, singular_yaw_den)

    if degrees:
        # ``hover_env`` runs this on a torch tensor, and ``torch.rad2deg`` multiplies by
        # the float32 constant -- unlike ``np.rad2deg``, which multiplies in double.
        out_2d = out_2d * np.float32(180.0 / np.pi)
    return out_2d[0] if single else out_2d


# --- Task maths ---------------------------------------------------------------------


def propeller_rpm(actions):
    """``(1 + 0.8 * actions) * HOVER_RPM`` on the clipped actions (``hover_env.step``)."""
    actions = np.asarray(actions, dtype=np.float32)
    exec_actions = np.clip(actions, -CLIP_ACTIONS, CLIP_ACTIONS)
    return (1.0 + exec_actions * ACTION_GAIN) * HOVER_RPM


def body_frame_velocities(lin_vel_world, ang_vel_world, base_quat):
    """``transform_by_quat(world_vel, inv(base_quat))`` for both linear and angular velocity."""
    inv_base_quat = inv_quat(base_quat)
    return transform_by_quat(lin_vel_world, inv_base_quat), transform_by_quat(ang_vel_world, inv_base_quat)


def observation(rel_pos, base_quat, lin_vel_body, ang_vel_body, last_actions):
    """The 17-dim observation row of ``hover_env._update_observation``.

    ``[clip(rel_pos / 3), base_quat, clip(v_body / 3), clip(w_body / 3.14159), last_actions]``
    -- the base quaternion is the raw, absolute ``drone.get_quat()`` value and the
    last slot holds the *previous* step's actions.
    """
    rel_pos = np.asarray(rel_pos, dtype=np.float32)
    lin_vel_body = np.asarray(lin_vel_body, dtype=np.float32)
    ang_vel_body = np.asarray(ang_vel_body, dtype=np.float32)
    return np.concatenate(
        [
            np.clip(rel_pos * OBS_SCALES["rel_pos"], -1.0, 1.0),
            np.asarray(base_quat, dtype=np.float32),
            np.clip(lin_vel_body * OBS_SCALES["lin_vel"], -1.0, 1.0),
            np.clip(ang_vel_body * OBS_SCALES["ang_vel"], -1.0, 1.0),
            np.asarray(last_actions, dtype=np.float32),
        ],
        axis=-1,
    )


def reset_observation(commands):
    """The analytic observation of ``hover_env.reset()`` for ``(rows, 3)`` commands.

    ``reset_idx`` pins the base to ``BASE_INIT_POS`` / ``BASE_INIT_QUAT`` and zeroes
    the velocities, ``last_actions`` and the episode counter, so the observation
    reduces to ``[clip((commands - BASE_INIT_POS) / 3), init_quat, 0, 0, 0]``.
    """
    commands = np.asarray(commands, dtype=np.float32)
    rel_pos = commands - BASE_INIT_POS
    rows = commands.shape[0]
    return np.concatenate(
        [
            np.clip(rel_pos * OBS_SCALES["rel_pos"], -1.0, 1.0),
            np.tile(BASE_INIT_QUAT, (rows, 1)),
            np.zeros((rows, 3), dtype=np.float32),
            np.zeros((rows, 3), dtype=np.float32),
            np.zeros((rows, 4), dtype=np.float32),
        ],
        axis=-1,
    )


def reward_terms(last_rel_pos, rel_pos, actions, last_actions, base_euler_deg, ang_vel_body, crash):
    """The five hover reward terms (PRE-scale) plus the ``dt``-scaled ``total``.

    ``total`` is ``dt * sum(scale * term)`` -- ``hover_env`` multiplies every entry of
    ``reward_scales`` by ``dt`` once at construction and then sums the scaled terms.
    Pass the **post-resample** ``rel_pos`` / ``last_rel_pos``: the baseline computes
    the reward after ``reset_idx`` has rewritten them.
    """
    last_rel_pos = np.asarray(last_rel_pos, dtype=np.float32)
    rel_pos = np.asarray(rel_pos, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    last_actions = np.asarray(last_actions, dtype=np.float32)
    ang_vel_body = np.asarray(ang_vel_body, dtype=np.float32)

    target = np.sum(np.square(last_rel_pos), axis=-1) - np.sum(np.square(rel_pos), axis=-1)
    smooth = np.sum(np.square(actions - last_actions), axis=-1)

    yaw_deg = np.asarray(base_euler_deg, dtype=np.float32)[..., 2]
    # Transcribed verbatim from ``_reward_yaw``: the ``yaw > 180`` wrap can never fire
    # because Euler yaw lives in [-180, 180], but the baseline keeps it.
    yaw = np.where(yaw_deg > 180.0, yaw_deg - 360.0, yaw_deg) / 180.0 * 3.14159
    yaw_term = np.exp(YAW_LAMBDA * np.abs(yaw))

    angular = np.linalg.norm(ang_vel_body / 3.14159, axis=-1)
    crash_term = np.asarray(crash, dtype=bool).astype(np.float32)

    total = DT * (
        REWARD_SCALES["target"] * target
        + REWARD_SCALES["smooth"] * smooth
        + REWARD_SCALES["yaw"] * yaw_term
        + REWARD_SCALES["angular"] * angular
        + REWARD_SCALES["crash"] * crash_term
    )
    return {
        "target": target,
        "smooth": smooth,
        "yaw": yaw_term,
        "angular": angular,
        "crash": crash_term,
        "total": np.asarray(total, dtype=np.float32),
    }


def crash_mask(base_euler_deg, rel_pos, base_pos):
    """``hover_env``'s crash predicate, evaluated on the pre-resample ``rel_pos``.

    The roll and pitch 180-degree terms are kept for fidelity but can never fire:
    Euler roll lives in ``[-180, 180]`` and pitch in ``[-90, 90]``, so
    ``|roll| > 180`` and ``|pitch| > 180`` are both unreachable.
    """
    euler = np.asarray(base_euler_deg, dtype=np.float32)
    rel = np.asarray(rel_pos, dtype=np.float32)
    pos = np.asarray(base_pos, dtype=np.float32)
    return (
        (np.abs(euler[..., 1]) > TERMINATION["pitch_deg"])
        | (np.abs(euler[..., 0]) > TERMINATION["roll_deg"])
        | (np.abs(rel[..., 0]) > TERMINATION["x_m"])
        | (np.abs(rel[..., 1]) > TERMINATION["y_m"])
        | (np.abs(rel[..., 2]) > TERMINATION["z_m"])
        | (pos[..., 2] < TERMINATION["ground_z"])
    )


def timeout_mask(episode_length):
    """``episode_length > max_episode_length`` -- the counter is incremented first."""
    return np.asarray(episode_length) > MAX_EPISODE_LENGTH


def resample_commands(rng, n):
    """Draw ``(n, 3)`` commands, three uniforms per row in x, y, z order.

    ``z`` is pinned to its singleton range ``[1, 1]`` -- the draw is still burned so
    the stream matches the baseline's three iid ``torch.rand`` calls.
    """
    n = int(n)
    out = np.empty((n, NUM_COMMANDS), dtype=np.float32)
    out[:, 0] = rng.uniform(*COMMAND_RANGES["pos_x"], size=n)
    out[:, 1] = rng.uniform(*COMMAND_RANGES["pos_y"], size=n)
    out[:, 2] = rng.uniform(*COMMAND_RANGES["pos_z"], size=n)
    return out


def step_maths(
    *,
    base_pos,
    last_base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    commands,
    actions,
    last_actions,
    episode_length,
    resampled_commands,
    error_mask=None,
):
    """One pre-reset, non-terminal hover step, in the baseline's exact order.

    All arguments are the **post-``scene.step()``** measured state plus the buffers the
    baseline carries.  ``actions`` are already clipped (``hover_env.step`` clips before
    writing the RPMs); ``resampled_commands`` is the host draw for the rows that hit
    the target this step; ``error_mask`` is the external solver-failure signal the
    baseline does not have (the eval env ORs it into ``crash``; ``None`` means "none").

    Returns:
        dict: ``obs`` (rows, 17), ``reward_terms`` (the five PRE-scale terms plus the
        ``dt``-scaled ``total``), ``crash`` / ``timeout`` / ``done`` / ``at_target``
        (rows,) bool, ``commands`` (rows, 3) the *next* commands and ``yaw_deg``.
    """
    base_pos = np.asarray(base_pos, dtype=np.float32)
    last_base_pos = np.asarray(last_base_pos, dtype=np.float32)
    base_quat = np.asarray(base_quat, dtype=np.float32)
    commands = np.asarray(commands, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    last_actions = np.asarray(last_actions, dtype=np.float32)

    # 3-5: pre-resample geometry, exactly as ``hover_env.step`` builds it.
    rel_pos = commands - base_pos
    last_rel_pos = commands - last_base_pos
    base_euler = quat_to_xyz(transform_quat_by_quat(inv_quat(BASE_INIT_QUAT), base_quat), rpy=True, degrees=True)
    lin_vel_body, ang_vel_body = body_frame_velocities(base_lin_vel_world, base_ang_vel_world, base_quat)

    # 5: resample the commands of the rows that reached their target.
    at_target = np.linalg.norm(rel_pos, axis=-1) < AT_TARGET_THRESHOLD
    commands_next = np.where(
        at_target[..., None],
        np.asarray(resampled_commands, dtype=np.float32),
        commands,
    )
    rel_pos_next = commands_next - base_pos
    last_rel_pos_next = commands_next - last_base_pos

    # 6: termination.  ``crash`` is the baseline's pre-resample predicate; the
    # post-resample one is equivalent, because ``at_target`` implies every component
    # of ``rel_pos`` is below 0.1 and therefore far below the 3/3/2 m thresholds.
    crash = crash_mask(base_euler, rel_pos, base_pos)
    if error_mask is not None:
        crash = crash | np.asarray(error_mask, dtype=bool)
    timeout = timeout_mask(np.asarray(episode_length) + 1)
    done = crash | timeout

    # 7-8: reward on the post-resample geometry, observation with the previous actions.
    return {
        "obs": observation(rel_pos_next, base_quat, lin_vel_body, ang_vel_body, last_actions),
        "reward_terms": reward_terms(
            last_rel_pos_next,
            rel_pos_next,
            actions,
            last_actions,
            base_euler,
            ang_vel_body,
            crash,
        ),
        "crash": crash,
        "timeout": timeout,
        "done": done,
        "at_target": at_target,
        "commands": commands_next,
        "yaw_deg": np.asarray(base_euler[..., 2], dtype=np.float32),
    }


# --- Self-check ---------------------------------------------------------------------


if __name__ == "__main__":
    _identity = BASE_INIT_QUAT
    assert np.allclose(quat_to_xyz(_identity), 0.0, atol=1e-6)

    _yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float32)
    assert np.allclose(quat_to_xyz(_yaw90), [0.0, 0.0, 90.0], atol=1e-4), quat_to_xyz(_yaw90)

    # The rotation is scale invariant: the raw-matrix rotation divides by |q|^2.
    _yaw90_x2 = 2.0 * _yaw90
    assert np.allclose(quat_to_xyz(_yaw90_x2), quat_to_xyz(_yaw90), atol=1e-4), quat_to_xyz(_yaw90_x2)

    _rpm = propeller_rpm(np.zeros((3, NUM_ACTIONS), dtype=np.float32))
    assert np.allclose(_rpm, HOVER_RPM, rtol=1e-7), _rpm

    _rest_obs = observation(
        np.zeros((2, 3), dtype=np.float32),
        np.tile(BASE_INIT_QUAT, (2, 1)),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, NUM_ACTIONS), dtype=np.float32),
    )
    assert _rest_obs.shape == (2, NUM_OBS)
    assert np.allclose(_rest_obs[:, :3], 0.0) and np.allclose(_rest_obs[:, 7:], 0.0)
    assert np.allclose(_rest_obs[:, 3:7], np.tile(BASE_INIT_QUAT, (2, 1)))

    print("ok")
