"""Independent NumPy/Torch oracles for the two pilot tasks' per-step maths.

The Stage-3 harness needs a second, independent implementation of every quantity the migrated
EvoX graphs compute, so a replay can report a real deviation instead of "the graph agrees with
itself". Two kinds of oracle are used here:

* the **legacy references** already in the tree -- ``examples/locomotion/go2_legacy_reference.py``
  (``walking_reward_terms``) and ``examples/drone/hover_legacy_reference.py`` (``step_maths``,
  ``reset_observation``), which transcribe the pre-migration PyTorch environments line by line;
* one **transcribed observation** for Go2, which the legacy reference module does not cover:
  :func:`go2_observations` re-implements ``go2_env._update_observation`` as NumPy, copied from the
  independent reference in ``tests/test_go2_neat_kernels.py`` (``_transform_by_inverse_quat`` and
  ``_reference_observations``, the documented NumPy reimplementation used by the graph tests).

Every sibling import happens inside a function: this module stays importable (and its docstrings
readable) without ``genesis``, ``torch`` or ``evoxc``, and the Go2 reward oracle is the only part
that pulls torch in, transitively, through the legacy reference.

Convention: all functions take the operands the migrated graphs take, in the same order and frame
(``base_quat`` relative to the init pose for Go2, absolute for hover; velocities world-frame).
"""

from __future__ import annotations

import numpy as np

from common import setup_paths

#: The six Go2 reward terms, in ``go2_env``'s order; ``total`` is their ``dt``-scaled sum.
GO2_REWARD_TERMS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "base_height",
    "action_rate",
    "similar_to_default",
)

#: The five hover reward terms (pre-scale values); ``total`` is their ``dt``-scaled sum.
HOVER_REWARD_TERMS = ("target", "smooth", "yaw", "angular", "crash")


def _setup() -> None:
    """Make the sibling example modules importable (idempotent)."""
    setup_paths()


def _scalar(value) -> float:
    """Materialise a torch scalar, numpy scalar or Python number to ``float``."""
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(np.asarray(value).reshape(-1)[0])


def _transform_by_inverse_quat(vectors, quats) -> np.ndarray:
    """``transform_by_quat(v, inv(q))`` with an explicit w-first float64 rotation matrix.

    Transcribed from ``tests/test_go2_neat_kernels.py::_transform_by_inverse_quat``: the conjugate
    ``[w, -x, -y, -z]`` is used raw (never normalized) and the rotated vector is divided by
    ``|q|^2``, exactly like ``genesis.utils.geom.transform_by_quat``.
    """
    q = np.asarray(quats, dtype=np.float64).copy()
    q[:, 1:] *= -1.0
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    norm_sqr = w * w + x * x + y * y + z * z
    matrix = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = w * w + x * x - y * y - z * z
    matrix[:, 0, 1] = 2.0 * (x * y - w * z)
    matrix[:, 0, 2] = 2.0 * (x * z + w * y)
    matrix[:, 1, 0] = 2.0 * (w * z + x * y)
    matrix[:, 1, 1] = w * w - x * x + y * y - z * z
    matrix[:, 1, 2] = 2.0 * (y * z - w * x)
    matrix[:, 2, 0] = 2.0 * (x * z - w * y)
    matrix[:, 2, 1] = 2.0 * (w * x + y * z)
    matrix[:, 2, 2] = w * w - x * x - y * y + z * z
    rotated = np.einsum("rij,rj->ri", matrix, np.asarray(vectors, dtype=np.float64))
    return rotated / norm_sqr[:, None]


def go2_observations(base_ang_vel_world, base_quat, dof_pos, dof_vel, actions) -> np.ndarray:
    """Recompute the 45-dim Go2 observation from one captured step's operands.

    Layout, exactly ``go2_env._update_observation``: ``[body_ang_vel * 0.25 (3), projected_gravity
    (3), commands * [2, 2, 0.25] (3), (dof_pos - standing) * 1.0 (12), dof_vel * 0.05 (12),
    clipped_actions (12)]``.

    Args:
        base_ang_vel_world: ``(rows, 3)`` world-frame base angular velocity.
        base_quat: ``(rows, 4)`` base quaternion **relative to the init pose** (w-first).
        dof_pos: ``(rows, 12)`` joint positions in control order.
        dof_vel: ``(rows, 12)`` joint velocities in control order.
        actions: ``(rows, 12)`` policy output; clipped to ``+-CLIP_ACTIONS`` here, because the
            advance graph feeds the *clipped* action into the observation.

    Returns:
        np.ndarray: ``(rows, 45)`` float32, the reference observation of that step.
    """
    _setup()
    from go2_neat_kernels import CLIP_ACTIONS, DEFAULT_COMMANDS, NUM_OBS, STANDING_FLAT

    base_ang_vel_world = np.asarray(base_ang_vel_world, dtype=np.float64)
    base_quat = np.asarray(base_quat, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    dof_vel = np.asarray(dof_vel, dtype=np.float64)
    clipped = np.clip(np.asarray(actions, dtype=np.float32), -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float64)

    rows = base_quat.shape[0]
    body_ang_vel = _transform_by_inverse_quat(base_ang_vel_world, base_quat)
    gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (rows, 1))
    projected_gravity = _transform_by_inverse_quat(gravity, base_quat)
    scaled_commands = np.asarray(DEFAULT_COMMANDS, dtype=np.float64) * np.array([2.0, 2.0, 0.25], dtype=np.float64)
    obs = np.concatenate(
        (
            body_ang_vel * 0.25,
            projected_gravity,
            np.tile(scaled_commands, (rows, 1)),
            (dof_pos - np.asarray(STANDING_FLAT, dtype=np.float64)) * 1.0,
            dof_vel * 0.05,
            clipped,
        ),
        axis=1,
    )
    if obs.shape[1] != NUM_OBS:
        raise ValueError(f"the reference observation has {obs.shape[1]} columns, expected {NUM_OBS}")
    return obs.astype(np.float32)


def go2_reward(base_pos, base_quat, base_lin_vel_world, base_ang_vel_world, dof_pos, action_change):
    """Recompute the six Go2 reward terms plus their ``dt``-scaled total for one row (or a batch).

    The maths is ``go2_legacy_reference.walking_reward_terms`` (a transcription of
    ``examples/locomotion/try1.py``'s ``walking_reward``, the same six terms and scales as
    ``go2_env``); this wrapper only loops rows, materialises the torch scalars and applies the
    kernel's definition of the rate term, ``action_change = clipped_actions - last_actions``.

    Args:
        base_pos: ``(3,)`` or ``(rows, 3)`` world base position **after** the step.
        base_quat: ``(rows, 4)`` base quaternion relative to the init pose.
        base_lin_vel_world: ``(rows, 3)`` world linear velocity.
        base_ang_vel_world: ``(rows, 3)`` world angular velocity.
        dof_pos: ``(rows, 12)`` joint positions.
        action_change: ``(rows, 12)`` ``clip(actions) - last_actions``.

    Returns:
        dict: The six :data:`GO2_REWARD_TERMS` and ``total``; ``float`` for a single row,
        ``np.ndarray`` of shape ``(rows,)`` for a batch.
    """
    _setup()
    import go2_legacy_reference as legacy

    base_pos = np.asarray(base_pos, dtype=np.float32)
    single = base_pos.ndim == 1
    arrays = {
        "base_pos": np.atleast_2d(base_pos),
        "base_quat": np.atleast_2d(np.asarray(base_quat, dtype=np.float32)),
        "base_lin_vel_world": np.atleast_2d(np.asarray(base_lin_vel_world, dtype=np.float32)),
        "base_ang_vel_world": np.atleast_2d(np.asarray(base_ang_vel_world, dtype=np.float32)),
        "dof_pos": np.atleast_2d(np.asarray(dof_pos, dtype=np.float32)),
        "action_change": np.atleast_2d(np.asarray(action_change, dtype=np.float64)),
    }
    rows = arrays["base_pos"].shape[0]
    names = (*GO2_REWARD_TERMS, "total")
    out = {name: np.empty(rows, dtype=np.float64) for name in names}
    for row in range(rows):
        terms = legacy.walking_reward_terms(
            arrays["base_pos"][row],
            arrays["base_quat"][row],
            arrays["base_lin_vel_world"][row],
            arrays["base_ang_vel_world"][row],
            arrays["dof_pos"][row],
            arrays["action_change"][row],
        )
        for name in names:
            out[name][row] = _scalar(terms[name])
    if single:
        return {name: float(values[0]) for name, values in out.items()}
    return out


def go2_done(base_quat_rel, error_mask, episode_length, max_episode_length):
    """Recompute the Go2 termination decision of one step, in the ``go2_env``/graph convention.

    ``tilted`` is a roll or pitch beyond ``TERMINATION_LIMIT_DEG`` measured by
    ``genesis.utils.geom.quat_to_xyz(quat, rpy=True, degrees=True)`` on the relative quaternion;
    ``timeout`` fires once the incremented counter would exceed ``max_episode_length``; the graph's
    ``done`` is ``alive_pre & (tilted | error | timeout)`` and the caller ANDs in ``alive``.

    Args:
        base_quat_rel: ``(rows, 4)`` base quaternion relative to the per-row init pose.
        error_mask: ``(rows,)`` bool, externally signalled solver failures.
        episode_length: ``(rows,)`` int32 pre-step counter.
        max_episode_length (int): Episode cap in control steps.

    Returns:
        tuple: ``(done, tilted, timeout)`` -- three ``(rows,)`` bool arrays, ``done`` including
        ``error_mask``.
    """
    _setup()
    import genesis as gs  # noqa: F401 - ensures the package is importable before geom
    from genesis.utils.geom import quat_to_xyz

    import go2_neat_kernels as kernels

    rpy = quat_to_xyz(np.asarray(base_quat_rel, dtype=np.float32), rpy=True, degrees=True)
    if hasattr(rpy, "detach"):
        rpy = rpy.detach().cpu().numpy()
    rpy_deg = np.asarray(rpy, dtype=np.float64)
    tilted = np.abs(rpy_deg[:, :2]).max(axis=1) > kernels.TERMINATION_LIMIT_DEG
    timeout = np.asarray(episode_length, dtype=np.int64) + 1 > int(max_episode_length)
    error = np.asarray(error_mask, dtype=bool)
    return tilted | error | timeout, tilted, timeout


def hover_oracle_step(**step_maths_kwargs) -> dict:
    """Recompute one hover step with ``hover_legacy_reference.step_maths``.

    Keyword arguments are forwarded verbatim (``base_pos``, ``last_base_pos``, ``base_quat``,
    ``base_lin_vel_world``, ``base_ang_vel_world``, ``commands``, ``actions``, ``last_actions``,
    ``episode_length``, ``resampled_commands``, ``error_mask``); see that function for the exact
    order of operations, which is the baseline's.

    Returns:
        dict: ``obs``, ``reward_terms`` (five pre-scale terms plus ``total``), ``crash``,
        ``timeout``, ``done``, ``at_target``, ``commands`` (next) and ``yaw_deg``.
    """
    _setup()
    import hover_legacy_reference as legacy

    return legacy.step_maths(**step_maths_kwargs)


def hover_graph_terms(reference_terms) -> dict:
    """Map ``hover_legacy_reference.reward_terms`` values to the convention the graph reports.

    ``hover_legacy_reference`` returns the five terms **pre-scale** and only folds the scales into
    ``total``, while ``hover_kernels.hover_advance`` returns every term already multiplied by its
    ``dt * REWARD_SCALES[name]`` factor (and by the same ``dt`` once for ``total``). Comparing the
    two therefore needs this one documented rescaling, which leaves ``total`` untouched because
    both sides already agree on it.

    Args:
        reference_terms (dict): The mapping returned by ``hover_legacy_reference.reward_terms``
            (``step_maths``'s ``reward_terms`` key).

    Returns:
        dict: The same keys with each :data:`HOVER_REWARD_TERMS` entry rescaled to the graph's
        convention, and ``total`` passed through.
    """
    _setup()
    import hover_legacy_reference as legacy

    scaled = {
        name: legacy.DT * legacy.REWARD_SCALES[name] * np.asarray(reference_terms[name], dtype=np.float32)
        for name in HOVER_REWARD_TERMS
    }
    scaled["total"] = np.asarray(reference_terms["total"], dtype=np.float32)
    return scaled


def hover_reset_obs(commands) -> np.ndarray:
    """Recompute the analytic post-``reset()`` hover observation for ``(rows, 3)`` commands.

    ``hover_legacy_reference.reset_observation`` pins the base to ``BASE_INIT_POS`` /
    ``BASE_INIT_QUAT`` and zeroes velocities, actions and the counter, which is exact while the
    generation's perturbation is the default all-zero one (what the replay harness measures).
    """
    _setup()
    import hover_legacy_reference as legacy

    return np.asarray(legacy.reset_observation(commands), dtype=np.float32)
