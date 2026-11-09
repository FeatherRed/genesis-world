"""Legacy (pre-migration) Go2 scripted-trot math, transcribed as pure functions.

This module is the **parity oracle** for the EvoX Compiler port in
``go2_evox_kernels.py``. It is a faithful, line-by-line transcription of the
math in ``examples/locomotion/try1.py`` and the helpers it leans on
(``genesis.utils.geom.inv_quat`` / ``transform_by_quat``), copied rather than
imported so that it stays import-cheap and independent of ``genesis``:

* the gait is plain NumPy (float64), exactly like ``try1.py``;
* the state-dependent reward terms are PyTorch (float32), exactly like the
  tensors ``genesis`` hands back, while ``action_rate`` is computed through
  NumPy float64 and materialized with ``.item()`` (again, as in ``try1.py``).

Do not "fix" the dtype split or normalize the quaternions: the whole point of
this module is to reproduce the legacy numerics, warts included.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# --- Constants shared with the demo -------------------------------------------------

DT = 0.02  # SimOptions(dt=0.02)
ACTION_SCALE = 0.25  # Go2 Walking normalized-action scale
PHASES = np.array([0.0, math.pi, math.pi, 0.0])  # diagonal trot pairs
# joints are ordered FR/FL/RR/RL x hip/thigh/calf
STANDING = np.array(
    [[0.0, 0.8, -1.5], [0.0, 0.8, -1.5], [0.0, 1.0, -1.5], [0.0, 1.0, -1.5]],
)
STANDING_FLAT = STANDING.ravel()

# Torch view of the standing pose, float32 like ``gs.tc_float``.
_STANDING_TORCH = torch.as_tensor(STANDING_FLAT, dtype=torch.float32)
_QUAT_CONJUGATE_SIGNS = torch.tensor((1.0, -1.0, -1.0, -1.0), dtype=torch.float32)


# --- Genesis-compatible quaternion helpers (copied from genesis/utils/geom.py) ------


def inv_quat(quat: np.ndarray) -> np.ndarray:
    """Conjugate of ``quat`` as ``[w, -x, -y, -z]`` (NOT normalized)."""
    out = np.array(quat, copy=True)
    out[..., 1:] *= -1
    return out


def transform_by_quat_np(v: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by ``quat`` exactly like ``genesis.utils.geom._np_transform_by_quat``.

    The rotation matrix is built from the raw (possibly non-unit) quaternion and
    the result is divided by ``|quat|^2``, mirroring the legacy implementation.
    """
    v = np.asarray(v, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q
    ww, xx, xy, xz = w * w, x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z
    vx, vy, vz = v
    ox = vx * (xx + ww - yy - zz) + vy * (2.0 * xy - 2.0 * wz) + vz * (2.0 * xz + 2.0 * wy)
    oy = vx * (2.0 * wz + 2.0 * xy) + vy * (ww - xx + yy - zz) + vz * (2.0 * yz - 2.0 * wx)
    oz = vx * (2.0 * xz - 2.0 * wy) + vy * (2.0 * wx + 2.0 * yz) + vz * (ww - xx - yy + zz)
    norm_sqr = ww + xx + yy + zz
    return np.array((ox / norm_sqr, oy / norm_sqr, oz / norm_sqr), dtype=np.float64)


def _tc_inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """Torch counterpart of :func:`inv_quat` (``quat * [1, -1, -1, -1]``)."""
    signs = _QUAT_CONJUGATE_SIGNS.to(dtype=quat.dtype, device=quat.device)
    return quat * signs


def _tc_transform_by_quat(v: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Torch counterpart of :func:`transform_by_quat_np` (1-D inputs)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    ww, xx, xy, xz = w * w, x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z
    vx, vy, vz = v[0], v[1], v[2]
    ox = vx * (xx + ww - yy - zz) + vy * (2.0 * xy - 2.0 * wz) + vz * (2.0 * xz + 2.0 * wy)
    oy = vx * (2.0 * wz + 2.0 * xy) + vy * (ww - xx + yy - zz) + vz * (2.0 * yz - 2.0 * wx)
    oz = vx * (2.0 * xz - 2.0 * wy) + vy * (2.0 * wx + 2.0 * yz) + vz * (ww - xx - yy + zz)
    norm_sqr = ww + xx + yy + zz
    return torch.stack((ox / norm_sqr, oy / norm_sqr, oz / norm_sqr))


# --- Gait ---------------------------------------------------------------------------


def gait_step(
    step: int,
    previous_action: np.ndarray,
    dt: float = DT,
    standing: np.ndarray = STANDING,
    phases: np.ndarray = PHASES,
    action_scale: float = ACTION_SCALE,
) -> dict[str, np.ndarray]:
    """One step of the legacy scripted trot.

    Parameters
    ----------
    step:
        Zero-based step index.
    previous_action:
        Previous normalized action, ``(12,)`` or ``(4, 3)`` (numpy float64).
    dt, standing, phases, action_scale:
        Legacy defaults.

    Returns
    -------
    dict with ``targets``, ``action`` and ``action_change`` — each a
    ``(12,)`` numpy float64 array (the demo works in ``(4, 3)`` and ravels).
    """
    standing = np.asarray(standing, dtype=np.float64)
    prev = np.asarray(previous_action, dtype=np.float64).reshape(4, 3)

    elapsed = max(0.0, step * dt - 0.5)
    phase = 2.0 * math.pi * 2.0 * elapsed + np.asarray(phases, dtype=np.float64)
    ramp = min(1.0, 2.0 * elapsed)
    lift = np.maximum(0.0, -np.cos(phase))

    target = standing.copy()
    target[:, 1] += ramp * (-0.20 * np.sin(phase) + 0.20 * lift)
    target[:, 2] -= ramp * 0.45 * lift

    action = (target - standing) / action_scale
    action_change = action - prev
    return {
        "targets": target.ravel(),
        "action": action.ravel(),
        "action_change": action_change.ravel(),
    }


# --- Reward -------------------------------------------------------------------------


def _as_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def walking_reward_terms(
    base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    dof_pos,
    action_change,
    dt: float = DT,
    standing: np.ndarray = STANDING_FLAT,
) -> dict[str, object]:
    """The six Go2 Walking reward terms plus the ``dt``-scaled total.

    State inputs are consumed as torch float32 tensors (converted if a numpy
    array is passed), exactly matching ``try1.py``'s ``walking_reward``.
    ``action_change`` is NumPy (float64) and feeds ``action_rate`` via
    ``np.square(...).sum().item()``, preserving the legacy dtype split.

    Returns
    -------
    dict with ``tracking_lin_vel``, ``tracking_ang_vel``, ``lin_vel_z``,
    ``base_height``, ``similar_to_default`` and ``total`` as torch float32
    scalars, and ``action_rate`` as a Python ``float`` (from numpy float64).
    """
    base_pos = _as_torch(base_pos)
    base_quat = _as_torch(base_quat)
    base_lin_vel_world = _as_torch(base_lin_vel_world)
    base_ang_vel_world = _as_torch(base_ang_vel_world)
    dof_pos = _as_torch(dof_pos)
    standing_t = torch.as_tensor(np.asarray(standing, dtype=np.float32), dtype=torch.float32, device=dof_pos.device)

    base_inv_quat = _tc_inv_quat(base_quat)
    velocity = _tc_transform_by_quat(base_lin_vel_world, base_inv_quat)
    yaw_rate = _tc_transform_by_quat(base_ang_vel_world, base_inv_quat)[2]

    velocity_error = (velocity[0] - 0.5).square() + velocity[1].square()
    height_error = base_pos[2] - 0.3
    pose_error = (dof_pos - standing_t).abs().sum()
    action_rate = float(np.square(np.asarray(action_change, dtype=np.float64)).sum().item())

    tracking_lin_vel = (-velocity_error / 0.25).exp()
    tracking_ang_vel = (-yaw_rate.square() / 0.25).exp()
    lin_vel_z = velocity[2].square()
    base_height = height_error.square()

    total = dt * (
        tracking_lin_vel
        + 0.2 * tracking_ang_vel
        - lin_vel_z
        - 50.0 * base_height
        - 0.005 * action_rate
        - 0.1 * pose_error
    )
    return {
        "tracking_lin_vel": tracking_lin_vel,
        "tracking_ang_vel": tracking_ang_vel,
        "lin_vel_z": lin_vel_z,
        "base_height": base_height,
        "action_rate": action_rate,
        "similar_to_default": pose_error,
        "total": total,
    }
