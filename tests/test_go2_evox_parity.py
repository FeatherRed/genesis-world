"""Numeric parity tests for the EvoX Compiler Go2 demo graphs.

The runnable demo lives in ``examples/locomotion/try1.py``; its control and
reward math was migrated into the pure-``evoxc`` graphs of
``examples/locomotion/go2_evox_kernels.py``, with the pre-migration formulas
kept as ``examples/locomotion/go2_legacy_reference.py``.

These tests only drive both implementations and compare their outputs -- the
legacy oracle is the single source of the old math, nothing is re-derived here.
``evoxc`` is not a declared dependency of ``genesis-world``, so the whole module
is skipped where it is not installed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")

pytestmark = [pytest.mark.required]

EXAMPLES_DIR = Path(__file__).parents[1] / "examples" / "locomotion"


def _load_module(path: Path, name: str):
    """Load a plain example module by file path (the examples are not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernels = _load_module(EXAMPLES_DIR / "go2_evox_kernels.py", "go2_evox_kernels_under_test")
legacy = _load_module(EXAMPLES_DIR / "go2_legacy_reference.py", "go2_legacy_reference_under_test")

# Measured agreement (numpy evoxc backend vs the legacy oracle):
#   gait, 250 chained steps: targets ~2.6e-6, action ~1.3e-5, action_change ~1.3e-5
#   reward, 200 random states: every term <= ~1e-6, total ~1e-8
# The gait tolerance sits ~4x above the measured maxima because the graph runs in
# float32 while the phase keeps growing (up to ~56 rad after 250 steps), so the
# `sin`/`cos` and the chained `previous_action` accumulate float32 rounding.
GAIT_ATOL = 5e-5
REWARD_TERM_ATOL = 1e-5
REWARD_TOTAL_ATOL = 1e-6
NUM_GAIT_STEPS = 250
NUM_RANDOM_STATES = 200


def _scalar(value) -> float:
    """Materialize a numpy 0-d array / torch scalar / Python float as a float."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def test_gait_parity_over_250_chained_steps():
    executable = kernels.build_gait()
    previous_action = np.zeros(12, dtype=np.float32)
    max_errors = {"targets": 0.0, "action": 0.0, "action_change": 0.0}

    for step in range(NUM_GAIT_STEPS):
        actual = kernels.run_gait(executable, step, previous_action)
        expected = legacy.gait_step(step, previous_action)

        for key in max_errors:
            error = np.max(np.abs(actual[key].astype(np.float64) - expected[key]))
            max_errors[key] = max(max_errors[key], float(error))

        # Feed the compiled graph's action back, exactly like the demo does.
        previous_action = actual["action"]

    assert max_errors["targets"] <= GAIT_ATOL, max_errors
    assert max_errors["action"] <= GAIT_ATOL, max_errors
    assert max_errors["action_change"] <= GAIT_ATOL, max_errors


def test_gait_semantics_at_specific_steps():
    executable = kernels.build_gait()
    previous_action = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    standing = kernels.STANDING_FLAT

    # elapsed = max(0, step * dt - 0.5); ramp = min(1, 2 * elapsed) is exactly
    # zero for steps <= 25, so the targets are the standing pose bit-for-bit and
    # the action is zero (its change is simply the negated previous action).
    for step in (0, 25):
        actual = kernels.run_gait(executable, step, previous_action)
        np.testing.assert_array_equal(actual["targets"], standing)
        np.testing.assert_array_equal(actual["action"], np.zeros(12, dtype=np.float32))
        np.testing.assert_array_equal(actual["action_change"], -previous_action)

    # One step later the ramp is positive (2 * 0.02 = 0.04) and the trot starts;
    # only the thigh/calf columns move, the hips keep their standing value.
    actual = kernels.run_gait(executable, 26, np.zeros(12, dtype=np.float32))
    delta = actual["targets"] - standing
    assert np.max(np.abs(delta)) > 0.0
    np.testing.assert_array_equal(delta.reshape(4, 3)[:, 0], np.zeros(4, dtype=np.float32))


def test_reward_parity_over_randomized_states():
    executable = kernels.build_reward()
    rng = np.random.default_rng(0)
    max_errors = {name: 0.0 for name in kernels.REWARD_TERM_NAMES}

    for _ in range(NUM_RANDOM_STATES):
        base_pos = np.array(
            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(0.3, 0.45)], dtype=np.float32
        )
        # Unit-ish but deliberately not exactly normalized, like Genesis hands it back.
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        base_quat = (quat * rng.uniform(0.99, 1.01)).astype(np.float32)
        base_lin_vel_world = rng.normal(scale=0.5, size=3).astype(np.float32)
        base_ang_vel_world = rng.normal(scale=0.5, size=3).astype(np.float32)
        dof_pos = (kernels.STANDING_FLAT + rng.normal(scale=0.1, size=12)).astype(np.float32)
        action_change = rng.normal(scale=0.1, size=12).astype(np.float32)

        actual = kernels.run_reward(
            executable,
            base_pos,
            base_quat,
            base_lin_vel_world,
            base_ang_vel_world,
            dof_pos,
            action_change,
        )
        expected = legacy.walking_reward_terms(
            base_pos,
            base_quat,
            base_lin_vel_world,
            base_ang_vel_world,
            dof_pos,
            action_change,
        )

        for name in kernels.REWARD_TERM_NAMES:
            error = abs(_scalar(actual[name]) - _scalar(expected[name]))
            max_errors[name] = max(max_errors[name], error)

    for name in kernels.REWARD_TERM_NAMES:
        atol = REWARD_TOTAL_ATOL if name == "total" else REWARD_TERM_ATOL
        assert max_errors[name] <= atol, max_errors
