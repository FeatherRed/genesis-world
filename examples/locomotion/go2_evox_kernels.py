"""Go2 scripted-trot control and Go2 Walking reward as EvoX Compiler graphs.

This is the migration target for the math in ``examples/locomotion/try1.py``.
It exposes two pure-EvoX-Compiler (``evoxc``) tensor graphs — no Torch, no
Genesis — plus thin build/run helpers that compile a graph once and reuse it
for every step.

Graph contracts
---------------
``GAIT``
    Runtime inputs: ``step`` (0-d int32), ``previous_action`` (float32, 12).
    Static specialization constants (in the spec tuple, never runtime-varying):
    ``standing_flat`` (float32, 12), ``phases`` (float32, 4), ``dt`` (float),
    ``inv_action_scale`` (float).
    Outputs: ``targets`` (12), ``action`` (12), ``action_change`` (12), all
    float32.

``REWARD``
    Runtime inputs: ``base_pos`` (3), ``base_quat`` (4),
    ``base_lin_vel_world`` (3), ``base_ang_vel_world`` (3), ``dof_pos`` (12),
    ``action_change`` (12), all float32.
    Static specialization constants: ``standing_flat`` (float32, 12), ``dt``.
    Outputs: a dict of six named scalar terms plus ``total`` (all float32 0-d).

The quaternion rotation is transcribed from ``genesis.utils.geom``: build the
rotation matrix from the raw quaternion (``inv_quat`` = ``[w, -x, -y, -z]``)
and divide by ``|q|^2`` — the quaternion is deliberately NOT normalized.

Device handling
---------------
``resolve_backend_device()`` honors explicit ``backend``/``device`` arguments,
then the ``EVOXC_BACKEND`` / ``EVOXC_DEVICE`` environment variables, and
defaults to the ``numpy`` backend on ``cpu:0``. Selecting a CUDA or ROCm
target is therefore pure configuration (e.g. ``device="cuda:0"``); a
non-CPU device also needs a compiler backend such as ``iree``.
"""

from __future__ import annotations

import os

import numpy as np

import evoxc

# --- Constants (static specialization values) ---------------------------------------

DT = 0.02
ACTION_SCALE = 0.25
INV_ACTION_SCALE = 1.0 / ACTION_SCALE
PHASES = np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float32)
# joints are ordered FR/FL/RR/RL x hip/thigh/calf
STANDING = np.array(
    [[0.0, 0.8, -1.5], [0.0, 0.8, -1.5], [0.0, 1.0, -1.5], [0.0, 1.0, -1.5]],
    dtype=np.float32,
)
STANDING_FLAT = STANDING.ravel()

REWARD_TERM_NAMES = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "base_height",
    "action_rate",
    "similar_to_default",
    "total",
)


# --- Backend / device resolution ----------------------------------------------------


def _parse_device(spec) -> evoxc.Device:
    """Parse ``"kind[:index]"`` (or pass a :class:`evoxc.Device` through)."""
    if isinstance(spec, evoxc.Device):
        return spec
    if isinstance(spec, str):
        kind, _, index = spec.partition(":")
        kind = kind.strip()
        if not kind:
            raise ValueError(f"malformed device spec {spec!r}")
        return evoxc.Device(kind, int(index) if index.strip() else 0)
    raise TypeError(f"device must be a str or evoxc.Device, got {type(spec)!r}")


def resolve_backend_device(backend=None, device=None):
    """Resolve ``(backend, device)``: explicit args > env vars > cpu/numpy default."""
    resolved_backend = backend or os.environ.get("EVOXC_BACKEND") or "numpy"

    if device is None:
        env_device = os.environ.get("EVOXC_DEVICE", "").strip()
        resolved_device = _parse_device(env_device) if env_device else evoxc.Device("cpu", 0)
    else:
        resolved_device = _parse_device(device)
    return resolved_backend, resolved_device


# --- Graphs -------------------------------------------------------------------------


def _constant(array: np.ndarray) -> evoxc.SymbolicTensor:
    """Embed a host numpy array as a graph constant."""
    return evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))


def _transform_by_quat(v, quat):
    """Rotate 1-D ``v`` by ``quat`` (raw, not normalized), matching Genesis."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    ww, xx, xy, xz = w * w, x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z
    vx, vy, vz = v[0], v[1], v[2]
    ox = vx * (xx + ww - yy - zz) + vy * (2.0 * xy - 2.0 * wz) + vz * (2.0 * xz + 2.0 * wy)
    oy = vx * (2.0 * wz + 2.0 * xy) + vy * (ww - xx + yy - zz) + vz * (2.0 * yz - 2.0 * wx)
    oz = vx * (2.0 * xz - 2.0 * wy) + vy * (2.0 * wx + 2.0 * yz) + vz * (ww - xx - yy + zz)
    norm_sqr = ww + xx + yy + zz
    return evoxc.stack([ox / norm_sqr, oy / norm_sqr, oz / norm_sqr], axis=0)


@evoxc.defn
def go2_gait(step, previous_action, standing_flat, phases, dt, inv_action_scale):
    """Scripted-trot graph: ``(step, previous_action) -> (targets, action, change)``."""
    step_f = evoxc.cast(step, evoxc.float32)
    elapsed = evoxc.maximum(0.0, step_f * dt - 0.5)
    phase = 2.0 * np.pi * 2.0 * elapsed + _constant(phases)
    ramp = evoxc.minimum(1.0, 2.0 * elapsed)
    lift = evoxc.maximum(0.0, -evoxc.cos(phase))

    # Per-leg deltas for thigh and calf; the hip column stays at zero.
    thigh = ramp * (-0.20 * evoxc.sin(phase) + 0.20 * lift)
    calf = -ramp * 0.45 * lift
    hip = evoxc.numpy.zeros((4,), evoxc.float32)

    delta = evoxc.reshape(evoxc.stack([hip, thigh, calf], axis=1), (12,))
    standing = _constant(standing_flat)
    targets = standing + delta
    action = (targets - standing) * inv_action_scale
    action_change = action - previous_action
    return targets, action, action_change


@evoxc.defn
def go2_walking_reward(
    base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    dof_pos,
    action_change,
    standing_flat,
    dt,
):
    """Go2 Walking reward graph; returns the six terms plus the ``dt``-scaled total."""
    base_inv_quat = evoxc.stack(
        [base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]],
        axis=0,
    )
    velocity = _transform_by_quat(base_lin_vel_world, base_inv_quat)
    yaw_rate = _transform_by_quat(base_ang_vel_world, base_inv_quat)[2]

    velocity_error = (velocity[0] - 0.5) * (velocity[0] - 0.5) + velocity[1] * velocity[1]
    height_error = base_pos[2] - 0.3

    tracking_lin_vel = evoxc.exp(-(velocity_error / 0.25))
    tracking_ang_vel = evoxc.exp(-(yaw_rate * yaw_rate / 0.25))
    lin_vel_z = velocity[2] * velocity[2]
    base_height = height_error * height_error
    action_rate = evoxc.reduce_sum(action_change * action_change)
    similar_to_default = evoxc.reduce_sum(evoxc.abs(dof_pos - _constant(standing_flat)))

    total = dt * (
        tracking_lin_vel
        + 0.2 * tracking_ang_vel
        - lin_vel_z
        - 50.0 * base_height
        - 0.005 * action_rate
        - 0.1 * similar_to_default
    )
    return {
        "tracking_lin_vel": tracking_lin_vel,
        "tracking_ang_vel": tracking_ang_vel,
        "lin_vel_z": lin_vel_z,
        "base_height": base_height,
        "action_rate": action_rate,
        "similar_to_default": similar_to_default,
        "total": total,
    }


# --- Spec tuples --------------------------------------------------------------------


def gait_specs():
    """Full positional spec tuple for :func:`go2_gait` (tensors + static values)."""
    return (
        evoxc.TensorSpec((), evoxc.int32, name="step"),
        evoxc.TensorSpec((12,), evoxc.float32, name="previous_action"),
        STANDING_FLAT,
        PHASES,
        DT,
        INV_ACTION_SCALE,
    )


def reward_specs():
    """Full positional spec tuple for :func:`go2_walking_reward`."""
    return (
        evoxc.TensorSpec((3,), evoxc.float32, name="base_pos"),
        evoxc.TensorSpec((4,), evoxc.float32, name="base_quat"),
        evoxc.TensorSpec((3,), evoxc.float32, name="base_lin_vel_world"),
        evoxc.TensorSpec((3,), evoxc.float32, name="base_ang_vel_world"),
        evoxc.TensorSpec((12,), evoxc.float32, name="dof_pos"),
        evoxc.TensorSpec((12,), evoxc.float32, name="action_change"),
        STANDING_FLAT,
        DT,
    )


# --- Build / run helpers ------------------------------------------------------------


def build_gait(backend=None, device=None):
    """Compile :func:`go2_gait` once; reuse the returned executable every step."""
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    return evoxc.build(go2_gait, *gait_specs(), backend=resolved_backend, device=resolved_device)


def build_reward(backend=None, device=None):
    """Compile :func:`go2_walking_reward` once; reuse the returned executable."""
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    return evoxc.build(go2_walking_reward, *reward_specs(), backend=resolved_backend, device=resolved_device)


def _to_device_array(executable, array: np.ndarray):
    """Stage a host array onto the executable's device (no-op for cpu).

    Re-selects the executable's own backend first, so staging stays correct when several
    backends serve the same device kind in one process (mixed-backend staging).
    """
    arr = np.asarray(array)
    if arr.ndim > 0:
        arr = np.ascontiguousarray(arr)
    device = getattr(executable, "device", None)
    if device is not None and device.kind != "cpu":
        backend = getattr(executable, "backend", None)
        if backend:
            # IREE and XLA can both serve cuda/rocm in one process; Tensor.to
            # resolves the transfer provider of the last looked-up backend, so
            # re-select this executable's backend before staging its inputs.
            evoxc.backends.get(backend)
        return evoxc.from_numpy(arr).to(device)
    return arr


def _to_numpy(tensor) -> np.ndarray:
    """Materialize an output tensor as a host numpy array."""
    device = getattr(tensor, "device", None)
    if device is not None and device.kind != "cpu":
        tensor = tensor.to(evoxc.Device("cpu", 0))
    return tensor.numpy()


def run_gait(executable, step, previous_action) -> dict[str, np.ndarray]:
    """Run one gait step; returns ``targets``/``action``/``action_change`` (12,) arrays."""
    prev = np.asarray(previous_action, dtype=np.float32).reshape(12)
    step_arr = np.asarray(step, dtype=np.int32).reshape(())

    outputs = evoxc.run(
        executable,
        _to_device_array(executable, step_arr),
        _to_device_array(executable, prev),
        STANDING_FLAT,
        PHASES,
        DT,
        INV_ACTION_SCALE,
    )
    return {
        "targets": _to_numpy(outputs[0]),
        "action": _to_numpy(outputs[1]),
        "action_change": _to_numpy(outputs[2]),
    }


def run_reward(
    executable,
    base_pos,
    base_quat,
    base_lin_vel_world,
    base_ang_vel_world,
    dof_pos,
    action_change,
) -> dict[str, np.ndarray]:
    """Run the reward graph; returns the six terms plus ``total`` as numpy arrays."""
    arrays = [
        np.asarray(base_pos, dtype=np.float32).reshape(3),
        np.asarray(base_quat, dtype=np.float32).reshape(4),
        np.asarray(base_lin_vel_world, dtype=np.float32).reshape(3),
        np.asarray(base_ang_vel_world, dtype=np.float32).reshape(3),
        np.asarray(dof_pos, dtype=np.float32).reshape(12),
        np.asarray(action_change, dtype=np.float32).reshape(12),
    ]
    outputs = evoxc.run(
        executable,
        *[_to_device_array(executable, a) for a in arrays],
        STANDING_FLAT,
        DT,
    )
    return {name: _to_numpy(outputs[name]) for name in REWARD_TERM_NAMES}
