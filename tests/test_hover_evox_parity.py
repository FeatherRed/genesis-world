"""Numeric parity tests for the EvoX Compiler drone hover graphs.

The runnable drone stack lives in ``examples/drone``: the four task graphs of
``hover_kernels.py`` (controls, observation, reset observation, one-step advance) were
migrated out of the torch/PPO baseline ``hover_env.py``, whose pre-migration math is kept
verbatim in ``hover_legacy_reference.py``.

These tests only drive both implementations on the cheap ``numpy`` backend and compare
their outputs -- the legacy oracle is the single source of the old math, nothing is
re-derived here.  Every graph is built with ``backend="numpy"`` explicitly so the module
never touches a GPU, a display or the ``genesis`` simulator.

Measured agreement (``evoxc`` numpy backend vs the legacy oracle, from the parity table
in ``examples/drone/CONTEXT.md`` and the graphs' own ``_self_check``):

* observation, rotor RPMs, clipped actions, every termination mask and the analytic reset
  observation are **exact** (0.0 max abs difference);
* the five reward terms agree to ~1e-10 .. 3.6e-7 (the ``dt`` scaling is a separate
  float32 multiply on each side) and the total to ~3.576e-07;
* yaw agrees to ~1.5e-5 deg (the stored float32 angle's own resolution) against the
  documented parity tolerance of 5e-4 deg.

``evoxc`` is not a declared dependency of ``genesis-world``, so the whole module is
skipped where it is not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")

pytestmark = [pytest.mark.required]

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
for _dir in ("drone", "neuroevolution"):
    _path = str(EXAMPLES_DIR / _dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hover_kernels as kernels  # noqa: E402  (path set up above)
import hover_legacy_reference as ref  # noqa: E402  (path set up above)

# Measured maxima documented above, with the reward terms given the usual ~2x headroom
# over the worst observed value (3.6e-7) and the total its own measured 3.576e-07.
# ``YAW_ATOL_DEG`` is the documented parity tolerance (5e-4 deg), ~30x the measured
# 1.5e-5 deg, because the residual is the float32 minimax ``atan2`` plus one ulp of a
# float32 degree (1.526e-5 deg at +-180 deg).
OBS_ATOL = 1e-6
REWARD_TERM_ATOL = 1e-5
REWARD_TOTAL_ATOL = 1e-6
YAW_ATOL_DEG = 5e-4
NUM_RANDOM_STATES = 200
NUM_YAW_RANDOM = 20000
NUM_YAW_GIMBAL = 200


# --- Local array-building recipes (copied from the graph module's self-check) --------


def _f32(*rows):
    """A float32 ``(len(rows), len(rows[0]))`` array from row sequences."""
    return np.array(rows, dtype=np.float32)


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


def _random_commands(rng, n):
    """``(n, 3)`` commands from the ``hover_env`` ranges (``z`` pinned to 1.0)."""
    return np.stack([rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), np.ones(n)], axis=-1).astype(np.float32)


def _neutral_state(rows):
    """An all-zero hover state; a test overrides only the operands it exercises."""
    zero3 = np.zeros((rows, 3), dtype=np.float32)
    zero4 = np.zeros((rows, kernels.NUM_ACTIONS), dtype=np.float32)
    return dict(
        commands=zero3,
        base_pos=zero3,
        last_base_pos=zero3,
        resampled_commands=zero3,
        base_quat=np.tile(kernels.BASE_INIT_QUAT, (rows, 1)).astype(np.float32),
        base_lin_vel_world=zero3,
        base_ang_vel_world=zero3,
        actions=zero4,
        last_actions=zero4,
        episode_length=np.zeros((rows,), dtype=np.int32),
        error_mask=np.zeros((rows,), dtype=bool),
    )


def _latched_end_codes(end_code, done, timeout, error_mask):
    """Numpy replication of the graph's first-ending latch (error > timeout > crash, once)."""
    code_this = np.where(
        error_mask,
        kernels.END_REASONS["error"],
        np.where(timeout, kernels.END_REASONS["timeout"], kernels.END_REASONS["crash"]),
    )
    first = (np.asarray(end_code) == kernels.END_REASONS["unfinished"]) & done
    return np.where(first, code_this, end_code).astype(np.int8)


def _scaled_oracle_terms(reference):
    """The oracle's PRE-scale terms scaled by ``dt``, plus its already-``dt``-scaled total.

    ``hover_advance`` multiplies every term by ``dt`` itself, while ``ref.reward_terms``
    holds the five raw terms and only the ``total`` is already scaled -- exactly the rule
    ``hover_kernels._max_term_error`` applies.
    """
    expected = {
        name: ref.DT * ref.REWARD_SCALES[name] * reference["reward_terms"][name]
        for name in kernels.REWARD_TERM_NAMES
    }
    expected["total"] = reference["reward_terms"]["total"]
    return expected


def _randomized_state(rng, rows):
    """A plausible randomized advance state, with the at-target/crash/ground/timeout rows mixed in."""
    commands = _random_commands(rng, rows)
    offsets = np.stack(
        [rng.uniform(-0.4, 0.4, rows), rng.uniform(-0.4, 0.4, rows), rng.uniform(-0.5, 0.5, rows)], axis=-1
    ).astype(np.float32)
    base_pos = commands - offsets
    near_target = rng.random(rows) < 0.2
    base_pos[near_target] = commands[near_target] + rng.uniform(-0.05, 0.05, (int(near_target.sum()), 3)).astype(
        np.float32
    )
    base_pos[rng.random(rows) < 0.1] += np.array([4.0, 0.0, 0.0], dtype=np.float32)
    base_pos[rng.random(rows) < 0.1, 2] = 0.02
    episode_length = rng.integers(0, 1600, rows).astype(np.int32)
    # Park some counters right at the cap so the timeout boundary is exercised.
    episode_length[rng.random(rows) < 0.1] = ref.MAX_EPISODE_LENGTH
    episode_length[rng.random(rows) < 0.1] = ref.MAX_EPISODE_LENGTH - 1
    return {
        "commands": commands,
        "base_pos": base_pos,
        "last_base_pos": (base_pos + rng.uniform(-0.1, 0.1, (rows, 3))).astype(np.float32),
        "base_quat": _rpy_quat(
            rng.uniform(-1.5, 1.5, rows), rng.uniform(-1.4, 1.4, rows), rng.uniform(-3.1, 3.1, rows)
        ),
        "base_lin_vel_world": rng.uniform(-1.0, 1.0, (rows, 3)).astype(np.float32),
        "base_ang_vel_world": rng.uniform(-2.0, 2.0, (rows, 3)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, (rows, kernels.NUM_ACTIONS)).astype(np.float32),
        "last_actions": rng.uniform(-1.0, 1.0, (rows, kernels.NUM_ACTIONS)).astype(np.float32),
        "episode_length": episode_length,
        "resampled_commands": _random_commands(rng, rows),
        "error_mask": rng.random(rows) < 0.05,
    }


# --- Observation --------------------------------------------------------------------


def test_observation_parity_over_randomized_states():
    rng = np.random.default_rng(20241109)
    rows = NUM_RANDOM_STATES
    commands = _random_commands(rng, rows)
    base_pos = commands - np.stack(
        [rng.uniform(-0.4, 0.4, rows), rng.uniform(-0.4, 0.4, rows), rng.uniform(-0.5, 0.5, rows)], axis=-1
    ).astype(np.float32)
    # Unit quaternions, near-gimbal ones (pitch within 1e-3 rad of +-90 deg) and rows
    # scaled off the unit sphere -- Genesis hands back the raw, non-normalized quat.
    base_quat = _random_unit_quats(rng, rows)
    n_gimbal = rows // 3
    base_quat[:n_gimbal] = _rpy_quat(
        rng.uniform(-np.pi, np.pi, n_gimbal),
        np.pi / 2.0 - 1e-3 * rng.random(n_gimbal),
        rng.uniform(-np.pi, np.pi, n_gimbal),
    )
    scaled = rng.random(rows) < 0.25
    base_quat[scaled] *= rng.uniform(0.99, 1.01, (int(scaled.sum()), 1)).astype(np.float32)
    base_lin_vel_world = rng.uniform(-1.0, 1.0, (rows, 3)).astype(np.float32)
    base_ang_vel_world = rng.uniform(-2.0, 2.0, (rows, 3)).astype(np.float32)
    last_actions = rng.uniform(-1.0, 1.0, (rows, kernels.NUM_ACTIONS)).astype(np.float32)

    actual = kernels.run_obs(
        kernels.build_obs(rows, backend="numpy"),
        commands=commands,
        base_pos=base_pos,
        base_quat=base_quat,
        base_lin_vel_world=base_lin_vel_world,
        base_ang_vel_world=base_ang_vel_world,
        last_actions=last_actions,
    )
    lin_vel_body, ang_vel_body = ref.body_frame_velocities(base_lin_vel_world, base_ang_vel_world, base_quat)
    expected = ref.observation(commands - base_pos, base_quat, lin_vel_body, ang_vel_body, last_actions)

    assert actual["obs"].shape == (NUM_RANDOM_STATES, kernels.NUM_OBS)
    assert actual["obs"].dtype == np.float32
    max_error = np.abs(actual["obs"].astype(np.float64) - expected.astype(np.float64)).max()
    assert max_error <= OBS_ATOL, f"observation parity failed: {max_error:.3e}"


# --- Controls -----------------------------------------------------------------------


def test_rpm_affine_map_and_clipped_actions():
    raw = _f32(
        [0.0, 0.0, 0.0, 0.0],
        [5.0, -5.0, 2.0, -2.0],
        [0.5, -0.5, 0.25, -0.25],
        [1.0, -1.0, 1.0, -1.0],
        [-1.5, 1.5, 0.0, 0.75],
    )
    actual = kernels.run_controls(kernels.build_controls(raw.shape[0], backend="numpy"), raw)

    # The RPM map is affine in the clipped action: zero action is exactly the hover RPM.
    np.testing.assert_allclose(actual["rpm"][0], ref.HOVER_RPM, rtol=1e-7)
    assert np.all(actual["rpm"][0] == np.float32(ref.HOVER_RPM))
    # Raw actions beyond +-1 saturate instead of passing through.
    np.testing.assert_array_equal(actual["actions"][1], np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(actual["actions"], np.clip(raw, -kernels.CLIP_ACTIONS, kernels.CLIP_ACTIONS))
    # Documented as exact (the graph transcribes the oracle's float32 expression).
    np.testing.assert_allclose(actual["rpm"], ref.propeller_rpm(raw), rtol=1e-7)


# --- Reward terms -------------------------------------------------------------------


def test_reward_terms_parity_over_randomized_states():
    rng = np.random.default_rng(7)
    state = _randomized_state(rng, NUM_RANDOM_STATES)
    actual = kernels.run_advance(
        kernels.build_advance(NUM_RANDOM_STATES, backend="numpy"),
        **state,
        alive=np.ones(NUM_RANDOM_STATES, dtype=bool),
        returns=np.zeros(NUM_RANDOM_STATES, dtype=np.float32),
    )
    expected = _scaled_oracle_terms(ref.step_maths(**state))

    max_errors = {}
    for name in (*kernels.REWARD_TERM_NAMES, "total"):
        max_errors[name] = float(np.abs(actual["reward_terms"][name].astype(np.float64) - expected[name]).max())
    for name in kernels.REWARD_TERM_NAMES:
        assert max_errors[name] <= REWARD_TERM_ATOL, f"{name} reward parity failed: {max_errors}"
    assert max_errors["total"] <= REWARD_TOTAL_ATOL, f"total reward parity failed: {max_errors}"


# --- Masks and the end-code latch ---------------------------------------------------


def test_end_code_crash_timeout_done_at_target_masks():
    rows = 5
    # Row roles: 0 = pure x-crash, 1 = error mask, 2 = timeout at the 1500 cap,
    # 3 = already-latched (must never re-arm), 4 = at-target (takes the resampled draw).
    commands = _f32(
        [0.5, 0.0, 1.0],
        [-0.4, 0.3, 1.0],
        [0.2, -0.1, 1.0],
        [0.6, -0.2, 1.0],
        [0.0, 0.0, 1.0],
    )
    base_pos = _f32(
        [-3.0, 0.0, 0.7],
        [-0.9, 0.7, 0.4],
        [0.25, -0.05, 0.7],
        [0.65, -0.15, 0.9],
        [0.0, 0.0, 1.0],
    )
    state = {
        "commands": commands,
        "base_pos": base_pos,
        "last_base_pos": base_pos
        + _f32(
            [0.01, 0.0, 0.02],
            [0.0, -0.01, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ),
        "base_quat": _rpy_quat(
            np.array([0.05, -0.04, 0.3, 0.0, 0.1], dtype=np.float32),
            np.array([-0.03, 0.06, -0.2, 0.0, 0.02], dtype=np.float32),
            np.array([0.2, -0.1, 0.05, 0.0, -0.3], dtype=np.float32),
        ),
        "base_lin_vel_world": _f32(
            [0.1, -0.2, 0.05],
            [1.0, 2.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.05, 0.03, 0.02],
        ),
        "base_ang_vel_world": _f32(
            [0.02, -0.01, 0.03],
            [1.0, -1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.01, 0.02, -0.03],
        ),
        "actions": _f32(
            [0.1, -0.2, 0.3, 0.0],
            [1.0, -1.0, 0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, -0.1],
            [-0.4, 0.4, 0.0, 0.1],
        ),
        "last_actions": _f32(
            [0.0, 0.0, 0.0, 0.0],
            [0.9, -0.8, 0.4, -0.4],
            [0.1, 0.0, -0.1, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [-0.3, 0.2, 0.0, 0.0],
        ),
        "episode_length": np.array([10, 1400, ref.MAX_EPISODE_LENGTH, ref.MAX_EPISODE_LENGTH, 100], dtype=np.int32),
        "resampled_commands": _f32(
            [0.5, -0.5, 1.0],
            [0.1, 0.2, 1.0],
            [-0.7, 0.8, 1.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.9, 1.0],
        ),
        "error_mask": np.array([False, True, False, True, False]),
    }
    end_code = np.array(
        [
            kernels.END_REASONS["unfinished"],
            kernels.END_REASONS["unfinished"],
            kernels.END_REASONS["unfinished"],
            kernels.END_REASONS["crash"],
            kernels.END_REASONS["unfinished"],
        ],
        dtype=np.int8,
    )
    reference = ref.step_maths(**state)
    stepped = kernels.run_advance(
        kernels.build_advance(rows, max_episode_length=ref.MAX_EPISODE_LENGTH, backend="numpy"),
        **state,
        alive=np.ones(rows, dtype=bool),
        returns=np.zeros(rows, dtype=np.float32),
        end_code=end_code,
    )

    for name in ("crash", "timeout", "done", "at_target"):
        np.testing.assert_array_equal(stepped[name], reference[name])
    np.testing.assert_array_equal(stepped["commands"], reference["commands"])
    # The latch is a pure host-side rule: error > timeout > crash, once, never re-armed.
    np.testing.assert_array_equal(
        stepped["end_code"], _latched_end_codes(end_code, reference["done"], reference["timeout"], state["error_mask"])
    )
    # One row per ending: pure crash, error, timeout, already-latched (unchanged), at-target.
    np.testing.assert_array_equal(
        stepped["end_code"],
        np.array(
            [
                kernels.END_REASONS["crash"],
                kernels.END_REASONS["error"],
                kernels.END_REASONS["timeout"],
                kernels.END_REASONS["crash"],
                kernels.END_REASONS["unfinished"],
            ],
            dtype=np.int8,
        ),
    )
    assert stepped["end_code"].dtype == np.int8
    np.testing.assert_array_equal(stepped["episode_length"], state["episode_length"] + 1)
    assert stepped["episode_length"].dtype == np.int32
    np.testing.assert_array_equal(stepped["base_pos"], base_pos)

    # The same latch rule over ~200 randomized states, a third of which already carry a code.
    rng = np.random.default_rng(2024)
    rand_state = _randomized_state(rng, NUM_RANDOM_STATES)
    rand_end_code = np.where(
        rng.random(NUM_RANDOM_STATES) < 1.0 / 3.0,
        rng.choice(np.array(list(kernels.END_REASONS.values()), dtype=np.int8), NUM_RANDOM_STATES),
        np.int8(kernels.END_REASONS["unfinished"]),
    ).astype(np.int8)
    rand_reference = ref.step_maths(**rand_state)
    rand_out = kernels.run_advance(
        kernels.build_advance(NUM_RANDOM_STATES, backend="numpy"),
        **rand_state,
        alive=rng.random(NUM_RANDOM_STATES) < 0.9,
        returns=np.zeros(NUM_RANDOM_STATES, dtype=np.float32),
        end_code=rand_end_code,
    )
    for name in ("crash", "timeout", "done", "at_target", "commands"):
        np.testing.assert_array_equal(rand_out[name], rand_reference[name])
    np.testing.assert_array_equal(
        rand_out["end_code"],
        _latched_end_codes(rand_end_code, rand_reference["done"], rand_reference["timeout"], rand_state["error_mask"]),
    )
    assert np.count_nonzero((rand_end_code == kernels.END_REASONS["unfinished"]) & rand_reference["done"]) > 0, (
        "the randomized states never latch a new ending"
    )
    np.testing.assert_array_equal(rand_out["episode_length"], rand_state["episode_length"] + 1)
    assert rand_out["episode_length"].dtype == np.int32
    np.testing.assert_array_equal(rand_out["base_pos"], rand_state["base_pos"])


# --- Reset observation --------------------------------------------------------------


def test_analytic_reset_observation():
    rng = np.random.default_rng(11)
    rows = 6
    commands = _random_commands(rng, rows)
    init_pos = np.ascontiguousarray(np.tile(kernels.BASE_INIT_POS, (rows, 1)))
    init_quat = np.ascontiguousarray(np.tile(kernels.BASE_INIT_QUAT, (rows, 1)))
    executable = kernels.build_reset_obs(rows, backend="numpy")

    authored = kernels.run_reset_obs(executable, commands=commands, base_init_pos=init_pos, base_init_quat=init_quat)
    np.testing.assert_array_equal(authored["obs"], ref.reset_observation(commands))

    # The pose is a runtime operand: a perturbed reset pose lands directly in the row.
    moved_pos = (kernels.BASE_INIT_POS + np.float32(0.25)).astype(np.float32) * np.ones((rows, 1), dtype=np.float32)
    moved_quat = _rpy_quat(np.full(rows, 0.05), np.full(rows, -0.02), np.full(rows, 0.4))
    moved = kernels.run_reset_obs(executable, commands=commands, base_init_pos=moved_pos, base_init_quat=moved_quat)
    expected = np.concatenate(
        [
            np.clip((commands - moved_pos) * ref.OBS_SCALES["rel_pos"], -1.0, 1.0),
            moved_quat,
            np.zeros((rows, 10), np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    np.testing.assert_array_equal(moved["obs"], expected)
    assert moved["obs"].dtype == np.float32


# --- Returns bookkeeping ------------------------------------------------------------


def test_returns_bookkeeping():
    rng = np.random.default_rng(31)
    rows = 8
    state = _randomized_state(rng, rows)
    alive = np.array([True, True, False, True, False, True, True, True])
    returns = rng.uniform(-5.0, 5.0, rows).astype(np.float32)
    out = kernels.run_advance(
        kernels.build_advance(rows, backend="numpy"),
        **state,
        alive=alive,
        returns=returns,
    )

    credit = alive & ~state["error_mask"]
    assert credit.any() and not credit.all()
    total = out["reward_terms"]["total"]
    np.testing.assert_array_equal(out["returns"][~credit], returns[~credit])
    np.testing.assert_allclose(out["returns"][credit], returns[credit] + total[credit], rtol=1e-6, atol=1e-7)


# --- Yaw transcription --------------------------------------------------------------


def test_yaw_transcription_against_atan2():
    rng = np.random.default_rng(20241109)
    gimbal_quats = _rpy_quat(
        rng.uniform(-3.1, 3.1, NUM_YAW_GIMBAL),
        np.pi / 2.0 - 1e-3 * rng.random(NUM_YAW_GIMBAL),
        rng.uniform(-3.1, 3.1, NUM_YAW_GIMBAL),
    )
    quats = np.ascontiguousarray(
        np.concatenate([_random_unit_quats(rng, NUM_YAW_RANDOM), gimbal_quats], axis=0), dtype=np.float32
    )
    rows = quats.shape[0]
    out = kernels.run_advance(
        kernels.build_advance(rows, backend="numpy"),
        **{**_neutral_state(rows), "base_quat": quats},
        alive=np.ones(rows, dtype=bool),
        returns=np.zeros(rows, dtype=np.float32),
    )
    expected = ref.quat_to_xyz(
        ref.quat_mul(ref.inv_quat(ref.BASE_INIT_QUAT)[None, :], quats), rpy=True, degrees=True
    )[:, 2]
    max_error = float(np.abs(out["yaw_deg"].astype(np.float64) - expected.astype(np.float64)).max())
    assert max_error <= YAW_ATOL_DEG, f"yaw parity failed: {max_error:.3e} deg"
