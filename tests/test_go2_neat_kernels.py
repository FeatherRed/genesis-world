"""Behavioural tests for the TensorNEAT Go2 compiled-graph layer.

``examples/locomotion/go2_neat_kernels.py`` owns the Go2 task graphs (observation,
advance and reward) of the TensorNEAT Go2-walking trainer, while the generic
policy/fitness graphs, the executable cache, the seed rules and the trace-only
problem come from the shared ``examples/neuroevolution`` layer.  These tests
drive those graphs directly -- no ``genesis`` scene, no simulator, no training
run -- and check the contracts the trainer relies on:

* the policy graph lays its rows out policy-major and consumes the genome payload
  policy-major;
* the advance graph books returns/``done``/``alive``/``episode_length`` with
  first-``done`` truncation, an error branch, a timeout branch and a tilt branch;
* fitness is the episode mean, with the zero-return episode still counted;
* the observation graph equals an independent NumPy reimplementation of
  ``go2_env._update_observation``;
* the reward equals the independent ``go2_legacy_reference`` oracle;
* the roll/pitch termination decision equals
  ``genesis.utils.geom.quat_to_xyz(..., rpy=True, degrees=True)``;
* ``ExecutableCache`` reuses shape-keyed executables and resets on ``clear()``.

The legacy oracle and Quaternion convention comments live in the module docstrings
under test; nothing here re-derives the engine's math from the implementation.
``evoxc`` / ``tensorneat`` are not declared dependencies of ``genesis-world``, so the
whole module is skipped where they are not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

pytestmark = [pytest.mark.required]

EXAMPLES_DIR = Path(__file__).parents[1] / "examples" / "locomotion"
NEUROEVOLUTION_DIR = Path(__file__).parents[1] / "examples" / "neuroevolution"
for _example_dir in (EXAMPLES_DIR, NEUROEVOLUTION_DIR):
    if str(_example_dir) not in sys.path:
        sys.path.insert(0, str(_example_dir))

from tensorneat import Pipeline, algorithm, genome
from tensorneat.common.tools import to_numpy

import go2_legacy_reference as legacy
import go2_neat_kernels as kernels
from evolution_graphs import ExecutableCache, build_fitness, run_fitness
from policy_graphs import build_policy, run_policy
from seeding import EPISODES_PER_POLICY
from trainer import TraceOnlyProblem

#: Policies per traced generation; rows are policy-major, so rows are ``POP_SIZE * 3``.
POP_SIZE = 3
ROWS = POP_SIZE * EPISODES_PER_POLICY

# Measured agreement (numpy evoxc backend vs the independent references):
#   observation, 8 non-unit quaternion rows: max abs error ~3e-8
#   reward, 50 random states: total error ~4e-9
#   roll/pitch decision, 4k random + near-threshold + pitch ~+-90 deg: 0 mismatches
# The observation/reward tolerances sit well above the float32 rounding of the graph
# while staying below any formula mistake, so they catch sign/index/scale errors.
OBS_ATOL = 1e-6
REWARD_ATOL = 1e-5
NUM_REWARD_STATES = 50
NUM_RANDOM_QUATS = 3000


@pytest.fixture(scope="module")
def policy_under_test():
    """Build the policy graph once from the same statics ``Go2NeatEnv`` traces."""
    generator = genome.DefaultGenome(
        num_inputs=kernels.NUM_OBS,
        num_outputs=kernels.NUM_ACTIONS,
        max_nodes=80,
        max_conns=640,
    )
    algo = algorithm.NEAT(generator, pop_size=POP_SIZE, species_size=min(3, POP_SIZE))
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(kernels.NUM_OBS, kernels.NUM_ACTIONS),
        seed=0,
        backend="numpy",
        is_detailed=False,
    )
    state = pipeline.setup()
    pop_arrays = tuple(np.array(to_numpy(leaf)) for leaf in state.pop)
    executable = build_policy(
        algo,
        pipeline._static_config(state),
        pop_arrays,
        ROWS,
        num_inputs=kernels.NUM_OBS,
        num_outputs=kernels.NUM_ACTIONS,
        episodes=EPISODES_PER_POLICY,
    )
    return executable, pop_arrays


def _scalar(value) -> float:
    """Materialize a numpy 0-d array / torch scalar / Python float as a float."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def _run_advance(executable, rows, **overrides):
    """Run one advance step with neutral defaults for every unspecified operand.

    The defaults are a non-tilted, non-failing, standing robot at the reward height,
    so a test only has to spell out the operands it is actually exercising.
    """
    defaults = {
        "base_pos": np.tile(np.array([0.0, 0.0, kernels.BASE_HEIGHT_TARGET], dtype=np.float32), (rows, 1)),
        "base_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (rows, 1)),
        "base_lin_vel_world": np.zeros((rows, 3), dtype=np.float32),
        "base_ang_vel_world": np.zeros((rows, 3), dtype=np.float32),
        "dof_pos": np.tile(kernels.STANDING_FLAT, (rows, 1)).astype(np.float32),
        "dof_vel": np.zeros((rows, kernels.NUM_ACTIONS), dtype=np.float32),
        "actions": np.zeros((rows, kernels.NUM_ACTIONS), dtype=np.float32),
        "last_actions": np.zeros((rows, kernels.NUM_ACTIONS), dtype=np.float32),
        "returns": np.zeros(rows, dtype=np.float32),
        "alive": np.ones(rows, dtype=bool),
        "episode_length": np.zeros(rows, dtype=np.int32),
        "error_mask": np.zeros(rows, dtype=bool),
        "commands": kernels.DEFAULT_COMMANDS,
    }
    defaults.update(overrides)
    return kernels.run_advance(executable, **defaults)


def _transform_by_inverse_quat(vectors, quats):
    """``transform_by_quat(v, inv(q))`` with an explicit w-first rotation matrix.

    Transcribed from the documented convention: the conjugate ``[w, -x, -y, -z]`` is
    used raw (never normalized) and the rotated vector is divided by ``|q|^2``.
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


def _reference_observations(base_ang_vel_world, base_quat, dof_pos, dof_vel, actions):
    """Independent NumPy reimplementation of ``go2_env._update_observation``.

    Layout: ``[body_ang_vel*0.25 (3), projected_gravity (3), commands*[2,2,0.25] (3),
    (dof_pos-standing)*1.0 (12), dof_vel*0.05 (12), actions (12)]``.
    """
    body_ang_vel = _transform_by_inverse_quat(base_ang_vel_world, base_quat)
    gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (base_quat.shape[0], 1))
    projected_gravity = _transform_by_inverse_quat(gravity, base_quat)
    scaled_commands = kernels.DEFAULT_COMMANDS * np.array([2.0, 2.0, 0.25], dtype=np.float32)
    return np.concatenate(
        (
            body_ang_vel * 0.25,
            projected_gravity,
            np.tile(scaled_commands, (base_quat.shape[0], 1)),
            (dof_pos - kernels.STANDING_FLAT) * 1.0,
            dof_vel * 0.05,
            actions,
        ),
        axis=1,
    ).astype(np.float32)


def _legacy_total(base_pos, base_quat, base_lin_vel_world, base_ang_vel_world, dof_pos, action_change):
    """The legacy reward total; ``walking_reward_terms`` is already ``dt``-scaled."""
    terms = legacy.walking_reward_terms(
        base_pos, base_quat, base_lin_vel_world, base_ang_vel_world, dof_pos, action_change
    )
    return _scalar(terms["total"])


def _roll_quat(degrees):
    """w-first quaternion of a pure roll rotation."""
    angle = np.deg2rad(degrees) * 0.5
    return np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)


def _pitch_quat(degrees):
    """w-first quaternion of a pure pitch rotation."""
    angle = np.deg2rad(degrees) * 0.5
    return np.array([np.cos(angle), 0.0, np.sin(angle), 0.0], dtype=np.float32)


def _small_tilt_quat(roll_deg, pitch_deg, yaw_deg):
    """w-first quaternion of ``yaw * pitch * roll``, angles in degrees."""
    half = np.deg2rad(np.array([roll_deg, pitch_deg, yaw_deg])) * 0.5

    def axis_quat(axis, angle):
        quat = np.zeros(4)
        quat[0] = np.cos(angle)
        quat[1 + axis] = np.sin(angle)
        return quat

    def multiply(lhs, rhs):
        w1, x1, y1, z1 = lhs
        w2, x2, y2, z2 = rhs
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    return multiply(multiply(axis_quat(2, half[2]), axis_quat(1, half[1])), axis_quat(0, half[0]))


# --- 1. Policy graph: policy-major rows ---------------------------------------------


def test_policy_rows_are_policy_major(policy_under_test):
    executable, pop_arrays = policy_under_test

    # The four payload leaves are (pop_size, ...): the leading dim is the population.
    leaf_shapes = tuple(np.shape(leaf) for leaf in pop_arrays)
    assert executable.pop_shapes == leaf_shapes
    assert all(shape[0] == POP_SIZE for shape in leaf_shapes)

    rng = np.random.default_rng(0)
    per_policy_obs = rng.normal(size=(POP_SIZE, kernels.NUM_OBS)).astype(np.float32)
    obs = np.repeat(per_policy_obs, EPISODES_PER_POLICY, axis=0)

    actions = run_policy(executable, pop_arrays, obs)
    assert actions.shape == (ROWS, kernels.NUM_ACTIONS)
    assert actions.dtype == np.float32

    # The three episodes of one policy see identical observations -> bit-identical actions.
    for policy in range(POP_SIZE):
        first = policy * EPISODES_PER_POLICY
        np.testing.assert_array_equal(actions[first], actions[first + 1])
        np.testing.assert_array_equal(actions[first], actions[first + 2])
    # Distinct policies receive distinct observations and produce distinct actions.
    for left in range(POP_SIZE):
        for right in range(left + 1, POP_SIZE):
            assert not np.array_equal(
                actions[left * EPISODES_PER_POLICY],
                actions[right * EPISODES_PER_POLICY],
            )

    # Overwrite policy 1's payload with policy 0's payload: only rows 3..5 may change.
    copied = [np.array(leaf, copy=True) for leaf in pop_arrays]
    for leaf in copied:
        leaf[1] = leaf[0]
    actions_copied = run_policy(executable, copied, obs)
    changed = np.any(actions_copied != actions, axis=1)
    assert changed[3:6].all()
    assert not changed[0:3].any()
    assert not changed[6:9].any()

    # With policy 0's observation fed to rows 3..5 too, identical payloads must now
    # produce identical actions, while policy 2 (untouched) keeps its original output.
    obs_same = obs.copy()
    obs_same[3:6] = obs[0:3]
    baseline_same = run_policy(executable, pop_arrays, obs_same)
    copied_same = run_policy(executable, copied, obs_same)
    assert not np.array_equal(baseline_same[0:3], baseline_same[3:6])
    np.testing.assert_array_equal(copied_same[0:3], copied_same[3:6])
    np.testing.assert_array_equal(baseline_same[6:9], copied_same[6:9])

    # Perturbing one observation row changes exactly that action row.
    perturbed = obs.copy()
    row = 4
    perturbed[row] += 1.0
    actions_perturbed = run_policy(executable, pop_arrays, perturbed)
    changed_rows = np.any(actions_perturbed != actions, axis=1)
    assert changed_rows[row]
    assert changed_rows.sum() == 1


# --- 2. Advance graph: first-done truncation, error and timeout branches ------------


def test_first_done_truncation_error_and_timeout():
    max_episode_length = 10
    rows = 3
    executable = kernels.build_advance(rows, max_episode_length=max_episode_length)

    # Row 0 never dies; row 1 is killed by error_mask on step 1; row 2 starts one step
    # away from the timeout (episode_length == max_episode_length -> next is max + 1).
    state = {
        "returns": np.zeros(rows, dtype=np.float32),
        "alive": np.ones(rows, dtype=bool),
        "episode_length": np.array([0, 0, max_episode_length], dtype=np.int32),
    }
    outcomes = []
    rewards = []
    for step in range(5):
        error_mask = np.zeros(rows, dtype=bool)
        if step == 1:
            error_mask[1] = True
        out = _run_advance(executable, rows, error_mask=error_mask, **state)
        outcomes.append(out)
        rewards.append(out["reward"].copy())
        state = {
            "returns": out["returns"],
            "alive": out["alive"],
            "episode_length": out["episode_length"],
        }

    # (a) Row 0 is never done, stays alive and accumulates every step's reward.
    for step, out in enumerate(outcomes):
        assert not out["done"][0]
        assert out["alive"][0]
        expected_return = float(np.sum([rewards[k][0] for k in range(step + 1)]))
        assert abs(float(out["returns"][0]) - expected_return) <= 1e-6

    # (b) Error branch: the dying step books its reward exactly once, done appears once.
    assert not outcomes[0]["done"][1]
    assert outcomes[1]["done"][1]
    dying_reward = float(rewards[1][1])
    assert abs(float(outcomes[1]["returns"][1]) - (float(outcomes[0]["returns"][1]) + dying_reward)) <= 1e-6
    assert not outcomes[1]["alive"][1]
    assert outcomes[1]["episode_length"][1] == 0
    for later in (2, 3, 4):
        assert not outcomes[later]["done"][1]
        assert not outcomes[later]["alive"][1]
        assert float(outcomes[later]["returns"][1]) == float(outcomes[1]["returns"][1])

    # (c) Timeout branch: episode_length == max fires done with max + 1 on the first step.
    assert outcomes[0]["done"][2]
    timeout_reward = float(rewards[0][2])
    assert abs(float(outcomes[0]["returns"][2]) - timeout_reward) <= 1e-6
    assert not outcomes[0]["alive"][2]
    assert outcomes[0]["episode_length"][2] == 0
    for later in (1, 2, 3, 4):
        assert not outcomes[later]["done"][2]
        assert not outcomes[later]["alive"][2]
        assert float(outcomes[later]["returns"][2]) == float(outcomes[0]["returns"][2])

    # Strong tilt fires done independently of error_mask (which stays all-False here).
    tilt = _run_advance(
        executable,
        rows,
        base_quat=np.stack((_roll_quat(45.0), _pitch_quat(-45.0), _roll_quat(-45.0))),
    )
    assert tilt["done"].all()


# --- 3. Fitness graph: per-policy episode mean --------------------------------------


def test_fitness_is_episode_mean_and_maximizes():
    n_policies = 4
    executable = build_fitness(n_policies)
    rng = np.random.default_rng(3)
    returns = rng.normal(size=(n_policies, EPISODES_PER_POLICY)).astype(np.float32)

    fitness = run_fitness(executable, returns.ravel())
    assert fitness.shape == (n_policies,)
    assert fitness.dtype == np.float32
    np.testing.assert_allclose(fitness, returns.mean(axis=1), rtol=0.0, atol=1e-6)

    # Maximization convention: all rows of policy 0 above all rows of policy 1.
    ordered = returns.copy()
    ordered[0] = np.abs(ordered[0]) + 10.0
    ordered[1] = -np.abs(ordered[1]) - 10.0
    ordered_fitness = run_fitness(executable, ordered.ravel())
    assert ordered_fitness[0] > ordered_fitness[1]

    # A dead (zero-return) episode is still divided by all three episodes.
    with_dead_episode = returns.copy()
    with_dead_episode[3] = np.array([2.0, 4.0, 0.0], dtype=np.float32)
    dead_fitness = run_fitness(executable, with_dead_episode.ravel())
    assert abs(float(dead_fitness[3]) - (2.0 + 4.0 + 0.0) / 3.0) <= 1e-6


# --- 4. Observation graph vs an independent NumPy reimplementation ------------------


def test_observation_parity_against_numpy_reimplementation():
    rows = 8
    executable = kernels.build_obs(rows)
    rng = np.random.default_rng(4)

    base_ang_vel_world = rng.normal(scale=0.7, size=(rows, 3)).astype(np.float32)
    quat = rng.normal(size=(rows, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # Deliberately non-unit, like Genesis hands the quaternion back.
    quat *= rng.uniform(0.6, 1.4, size=(rows, 1))
    base_quat = quat.astype(np.float32)
    dof_pos = (kernels.STANDING_FLAT + rng.normal(scale=0.15, size=(rows, kernels.NUM_ACTIONS))).astype(np.float32)
    dof_vel = rng.normal(scale=0.7, size=(rows, kernels.NUM_ACTIONS)).astype(np.float32)
    actions = rng.normal(size=(rows, kernels.NUM_ACTIONS)).astype(np.float32)

    actual = kernels.run_obs(
        executable,
        base_ang_vel_world=base_ang_vel_world,
        base_quat=base_quat,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        actions=actions,
        commands=kernels.DEFAULT_COMMANDS,
    )
    expected = _reference_observations(base_ang_vel_world, base_quat, dof_pos, dof_vel, actions)

    assert actual.shape == (rows, kernels.NUM_OBS)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=OBS_ATOL)


# --- 5. Reward graph vs the legacy oracle -------------------------------------------


def test_reward_parity_against_legacy_oracle():
    rows = NUM_REWARD_STATES
    executable = kernels.build_advance(rows, max_episode_length=10_000)
    rng = np.random.default_rng(5)

    base_pos = np.stack(
        (
            rng.uniform(-1.0, 1.0, size=rows),
            rng.uniform(-1.0, 1.0, size=rows),
            rng.uniform(0.25, 0.45, size=rows),
        ),
        axis=1,
    ).astype(np.float32)
    # Small tilts only (well below the 10 deg termination limit) with random yaw.
    small_tilt = np.stack(
        [
            _small_tilt_quat(roll, pitch, yaw)
            for roll, pitch, yaw in zip(
                rng.uniform(-8.0, 8.0, size=rows),
                rng.uniform(-8.0, 8.0, size=rows),
                rng.uniform(-180.0, 180.0, size=rows),
            )
        ]
    )
    # Deliberately not exactly normalized, like the legacy parity test.
    base_quat = (small_tilt * rng.uniform(0.99, 1.01, size=(rows, 1))).astype(np.float32)
    base_lin_vel_world = rng.normal(scale=0.5, size=(rows, 3)).astype(np.float32)
    base_ang_vel_world = rng.normal(scale=0.5, size=(rows, 3)).astype(np.float32)
    dof_pos = (kernels.STANDING_FLAT + rng.normal(scale=0.1, size=(rows, kernels.NUM_ACTIONS))).astype(np.float32)
    dof_vel = rng.normal(scale=0.5, size=(rows, kernels.NUM_ACTIONS)).astype(np.float32)
    # Raw actions stay far inside +-CLIP_ACTIONS, so the graph's clipping is a no-op.
    actions = rng.uniform(-1.5, 1.5, size=(rows, kernels.NUM_ACTIONS)).astype(np.float32)
    action_change = rng.normal(scale=0.1, size=(rows, kernels.NUM_ACTIONS)).astype(np.float32)
    last_actions = (actions - action_change).astype(np.float32)

    out = _run_advance(
        executable,
        rows,
        base_pos=base_pos,
        base_quat=base_quat,
        base_lin_vel_world=base_lin_vel_world,
        base_ang_vel_world=base_ang_vel_world,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        actions=actions,
        last_actions=last_actions,
    )
    assert out["alive"].all()
    assert not out["done"].any()

    max_error = 0.0
    for index in range(rows):
        expected = _legacy_total(
            base_pos[index],
            base_quat[index],
            base_lin_vel_world[index],
            base_ang_vel_world[index],
            dof_pos[index],
            action_change[index],
        )
        max_error = max(max_error, abs(float(out["reward"][index]) - expected))
    assert max_error <= REWARD_ATOL, max_error

    # Isolate three terms by varying only their input and comparing reward differences.
    # Fixed elsewhere: standing pose, identity orientation, zero velocities, zero actions.
    neutral_pos = np.tile(np.array([0.0, 0.0, kernels.BASE_HEIGHT_TARGET], dtype=np.float32), (rows, 1))
    neutral_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (rows, 1))
    zero_velocities = np.zeros((rows, 3), dtype=np.float32)
    neutral_dof = np.tile(kernels.STANDING_FLAT, (rows, 1)).astype(np.float32)
    neutral_actions = np.zeros((rows, kernels.NUM_ACTIONS), dtype=np.float32)
    deltas = np.linspace(-0.4, 0.4, rows, dtype=np.float32)

    baseline = _run_advance(
        executable,
        rows,
        base_pos=neutral_pos,
        base_quat=neutral_quat,
        base_lin_vel_world=zero_velocities,
        base_ang_vel_world=zero_velocities,
        dof_pos=neutral_dof,
        actions=neutral_actions,
        last_actions=neutral_actions,
    )
    baseline_legacy = _legacy_total(
        neutral_pos[0], neutral_quat[0], zero_velocities[0], zero_velocities[0], neutral_dof[0], neutral_actions[0]
    )

    def check_difference(varied_out, varied_pos, varied_dof, varied_change):
        errors = []
        for index in range(rows):
            legacy_varied = _legacy_total(
                varied_pos[index],
                neutral_quat[index],
                zero_velocities[index],
                zero_velocities[index],
                varied_dof[index],
                varied_change[index],
            )
            actual_difference = float(varied_out["reward"][index]) - float(baseline["reward"][index])
            expected_difference = legacy_varied - baseline_legacy
            errors.append(abs(actual_difference - expected_difference))
        return max(errors)

    # action_rate: vary only `actions` (last_actions stay zero, so action_change == actions).
    varied_actions = neutral_actions.copy()
    varied_actions[:, 4] = deltas
    out_actions = _run_advance(
        executable,
        rows,
        base_pos=neutral_pos,
        base_quat=neutral_quat,
        base_lin_vel_world=zero_velocities,
        base_ang_vel_world=zero_velocities,
        dof_pos=neutral_dof,
        actions=varied_actions,
        last_actions=neutral_actions,
    )
    assert (
        check_difference(out_actions, neutral_pos, neutral_dof, varied_actions - neutral_actions) <= REWARD_ATOL
    )

    # base_height: vary only base_pos[:, 2].
    varied_pos = neutral_pos.copy()
    varied_pos[:, 2] += deltas * 0.1
    out_height = _run_advance(
        executable,
        rows,
        base_pos=varied_pos,
        base_quat=neutral_quat,
        base_lin_vel_world=zero_velocities,
        base_ang_vel_world=zero_velocities,
        dof_pos=neutral_dof,
        actions=neutral_actions,
        last_actions=neutral_actions,
    )
    assert check_difference(out_height, varied_pos, neutral_dof, neutral_actions) <= REWARD_ATOL

    # similar_to_default: vary only dof_pos.
    varied_dof = neutral_dof.copy()
    varied_dof[:, 5] += deltas * 0.5
    out_pose = _run_advance(
        executable,
        rows,
        base_pos=neutral_pos,
        base_quat=neutral_quat,
        base_lin_vel_world=zero_velocities,
        base_ang_vel_world=zero_velocities,
        dof_pos=varied_dof,
        actions=neutral_actions,
        last_actions=neutral_actions,
    )
    assert check_difference(out_pose, neutral_pos, varied_dof, neutral_actions) <= REWARD_ATOL


# --- 6. Roll/pitch termination vs genesis.utils.geom.quat_to_xyz --------------------


def test_roll_pitch_termination_matches_quat_to_xyz():
    from genesis.utils.geom import quat_to_xyz

    rng = np.random.default_rng(6)
    random_quat = rng.normal(size=(NUM_RANDOM_QUATS, 4))
    random_quat /= np.linalg.norm(random_quat, axis=1, keepdims=True)
    random_quat *= rng.uniform(0.5, 1.5, size=(NUM_RANDOM_QUATS, 1))

    special = []
    # Near the 10 deg pitch limit from both sides (within +-1e-4 deg), both signs.
    for degrees in (10.0 + 1e-4, 10.0 - 1e-4, -(10.0 + 1e-4), -(10.0 - 1e-4)):
        special.append(_pitch_quat(degrees))
    # Near the 10 deg roll limit from both sides, both signs.
    for degrees in (10.0 + 1e-4, 10.0 - 1e-4, -(10.0 + 1e-4), -(10.0 - 1e-4)):
        special.append(_roll_quat(degrees))
    # Gimbal-lock-ish pitch ~ +-90 deg, both signs, plus a couple of far rolls.
    for degrees in (90.0, 90.0 - 1e-3, -90.0, -90.0 + 1e-3):
        special.append(_pitch_quat(degrees))
    for degrees in (179.0, -179.0):
        special.append(_roll_quat(degrees))

    quats = np.concatenate((random_quat.astype(np.float32), np.stack(special)), axis=0)
    rows = quats.shape[0]
    executable = kernels.build_advance(rows, max_episode_length=10**6)
    out = _run_advance(executable, rows, base_quat=quats)

    # Compare decisions against the documented oracle on the exact same values.
    rpy = quat_to_xyz(quats.astype(np.float64), rpy=True, degrees=True)
    expected = (np.abs(rpy[:, 0]) > kernels.TERMINATION_LIMIT_DEG) | (
        np.abs(rpy[:, 1]) > kernels.TERMINATION_LIMIT_DEG
    )
    done = out["done"]
    mismatches = np.nonzero(done != expected)[0]
    if mismatches.size:
        details = "\n".join(
            f"index={int(index)} quat={quats[index].tolist()} rpy_deg={rpy[index].tolist()} "
            f"graph_done={bool(done[index])} quat_to_xyz_done={bool(expected[index])}"
            for index in mismatches
        )
        pytest.fail(f"{mismatches.size} roll/pitch termination mismatches:\n{details}")

    # Guard against a vacuous comparison: the batch must exercise both decisions.
    assert expected.any()
    assert (~expected).any()


# --- 7. ExecutableCache -------------------------------------------------------------


def test_executable_cache_hits_misses_and_clear():
    cache = ExecutableCache()
    built = []
    executable = object()

    def builder():
        built.append(executable)
        return executable

    first = cache.get(("obs", 9), builder)
    assert first is executable
    assert (cache.hits, cache.misses) == (0, 1)

    second = cache.get(("obs", 9), builder)
    assert second is executable
    assert len(built) == 1
    assert (cache.hits, cache.misses) == (1, 1)

    other = object()
    third = cache.get(("obs", 12), lambda: other)
    assert third is other
    assert (cache.hits, cache.misses) == (1, 2)

    cache.clear()
    assert (cache.hits, cache.misses) == (0, 0)

    fourth = cache.get(("obs", 9), builder)
    assert fourth is executable
    assert len(built) == 2
    assert (cache.hits, cache.misses) == (0, 1)
