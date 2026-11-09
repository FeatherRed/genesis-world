"""Seeded reset and seed-stream tests for the TensorNEAT drone hover stack.

``examples/drone/hover_neat_env.py`` is the batched, seeded evaluation environment of the
TensorNEAT drone hover trainer.  The seed rules are not the drone's own: every task shares
``examples/neuroevolution/seeding.py``, which splits one master seed into an evolution stream
(:func:`seeding.evolution_seed`) and a per-generation episode-seed table
(:func:`seeding.generation_seed_table`), and draws each episode's reset perturbation from its
seed with :func:`seeding.perturbation_from_seed`.

These tests pin the pure helper contracts of the drone environment -- seed triples, supplied
tables, the zero-perturbation fixed reset, the policy-major slot mapping and the per-row
command streams -- without a simulator.  Nothing here re-derives the implementation: every
expected value comes from the shared seeding module, the documented ``[seed, generation]``
generator definition, or the explicit quaternion product ``_compose_quat`` documents.

The env layer is covered through its pure pieces only (seed table -> per-row init pose ->
per-row command stream): building a ``HoverNeatEnv`` compiles the graphs, imports Genesis and
builds a ``numpy``/``cpu`` scene, far past a single test's budget for what is a seed check.
A scene-free instance is therefore allocated with ``object.__new__`` and given just the five
attributes the helpers read.

The example modules are not a package (``hover_neat_env.py`` imports its siblings by bare
name), so ``examples/drone`` and ``examples/neuroevolution`` are put on ``sys.path`` before
importing them.  ``evoxc`` and ``tensorneat`` are not declared dependencies of
``genesis-world``, so the whole module is skipped where either is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

pytestmark = [pytest.mark.required]

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
for _dir in ("drone", "neuroevolution"):
    _path = str(EXAMPLES_DIR / _dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hover_neat_env as neat_env  # noqa: E402  (path set up above)
from hover_kernels import BASE_INIT_POS, BASE_INIT_QUAT  # noqa: E402  (path set up above)
from seeding import (  # noqa: E402  (path set up above)
    EPISODES_PER_POLICY,
    Perturbation,
    evolution_seed,
    generation_seed_table,
    perturbation_from_seed,
)


def _bare_env(master_seed=0, pop_size=2, perturbation=None, seed_table=None):
    """A scene-free ``HoverNeatEnv`` carrying only the attributes the pure helpers read.

    ``_seeds_for``, ``_init_poses``, ``_command_generators``, ``_draw_commands`` and
    ``_compose_quat`` touch ``master_seed``, ``pop_size``, ``rows``, ``seed_table`` and
    ``perturbation`` -- nothing else -- so the constructor's scene and graphs are not needed.
    """
    env = object.__new__(neat_env.HoverNeatEnv)
    env.master_seed = int(master_seed)
    env.pop_size = int(pop_size)
    env.rows = EPISODES_PER_POLICY * env.pop_size
    env.seed_table = None if seed_table is None else np.asarray(seed_table, dtype=np.int64)
    env.perturbation = Perturbation() if perturbation is None else perturbation
    return env


def test_module_reexports_the_shared_seeding_objects():
    # The task module must consume the shared layer, not carry a second copy of the rules.
    assert neat_env.generation_seed_table is generation_seed_table
    assert neat_env.perturbation_from_seed is perturbation_from_seed
    assert neat_env.EPISODES_PER_POLICY == EPISODES_PER_POLICY == 3
    assert isinstance(neat_env.Perturbation(), Perturbation)


def test_generation_seed_triples_match_shared_tables():
    for master_seed in (0, 7):
        env = _bare_env(master_seed=master_seed, pop_size=2)
        for generation in range(5):
            seeds = env._seeds_for(generation)
            expected = generation_seed_table(master_seed, generation + 1)[generation]
            np.testing.assert_array_equal(seeds, expected)
            assert seeds.shape == (EPISODES_PER_POLICY,)
            assert seeds.dtype == np.int64
            assert len(set(seeds.tolist())) == EPISODES_PER_POLICY

    # The documented reproduction guarantee, verbatim from ``seeding``'s docstring.
    np.testing.assert_array_equal(
        _bare_env(master_seed=0)._seeds_for(0),
        np.array([1454269175, 757617120, 1421746521], dtype=np.int64),
    )
    assert evolution_seed(0) == 3757552657


def test_supplied_seed_table_replaces_the_drawn_table():
    table = np.array(
        [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]],
        dtype=np.int64,
    )
    env = _bare_env(master_seed=1234, seed_table=table)
    for generation, row in enumerate(table):
        np.testing.assert_array_equal(env._seeds_for(generation), row)

    with pytest.raises(IndexError):
        env._seeds_for(len(table))
    with pytest.raises(IndexError):
        env._seeds_for(-1)


def test_zero_perturbation_is_bit_identical_across_rows():
    seeds = generation_seed_table(0, 1)[0]
    env = _bare_env(pop_size=3)
    assert env.perturbation.is_zero
    positions, quats = env._init_poses(seeds)

    assert positions.shape == (env.rows, 3) and positions.dtype == np.float32
    assert quats.shape == (env.rows, 4) and quats.dtype == np.float32
    for row in range(env.rows):
        np.testing.assert_array_equal(positions[row], BASE_INIT_POS)
        np.testing.assert_array_equal(quats[row], BASE_INIT_QUAT)
    for row in range(1, env.rows):
        np.testing.assert_array_equal(positions[row], positions[0])
        np.testing.assert_array_equal(quats[row], quats[0])

    again = env._init_poses(seeds)
    np.testing.assert_array_equal(positions, again[0])
    np.testing.assert_array_equal(quats, again[1])


def test_rows_are_policy_major_and_share_the_three_slots():
    seeds = generation_seed_table(3, 1)[0]
    perturbation = Perturbation(base_pos=0.05, base_rpy=0.1)
    env = _bare_env(pop_size=3, perturbation=perturbation)
    positions, quats = env._init_poses(seeds)

    # Row ``3p + e`` is policy ``p``'s episode ``e``, so the three slots repeat per policy.
    for row in range(env.rows):
        slot = row % EPISODES_PER_POLICY
        np.testing.assert_array_equal(positions[row], positions[slot])
        np.testing.assert_array_equal(quats[row], quats[slot])
    assert not np.array_equal(positions[0], positions[1])
    assert not np.array_equal(positions[1], positions[2])
    assert not np.array_equal(positions[0], positions[2])

    for row in range(env.rows):
        draw = perturbation_from_seed(int(seeds[row % EPISODES_PER_POLICY]), perturbation, num_dof=0)
        np.testing.assert_array_equal(positions[row], BASE_INIT_POS + draw["base_pos"])
        np.testing.assert_array_equal(quats[row], env._compose_quat(draw["base_quat"], BASE_INIT_QUAT))
        # The drone is a body with no joints, so the joint block of the draw is empty.
        assert draw["joint_pos"].shape == (0,)


def test_command_streams_are_row_and_step_order_independent():
    seeds = generation_seed_table(5, 1)[0]
    env = _bare_env(pop_size=2)  # 6 rows: two policies x three episode slots
    rngs = env._command_generators(seeds, 0)
    assert len(rngs) == env.rows

    # Rows of the same episode slot share the stream, across policies as well.
    for row in range(EPISODES_PER_POLICY):
        left = np.stack([env._draw_commands(rngs[row]) for _ in range(4)])
        right = np.stack([env._draw_commands(rngs[row + EPISODES_PER_POLICY]) for _ in range(4)])
        np.testing.assert_array_equal(left, right)
        assert left.shape == (4, 3) and left.dtype == np.float32
        assert np.all(left[:, 2] == 1.0)

    # Each row's stream is exactly ``default_rng([seed, generation])``'s three-uniform draw.
    fresh = env._command_generators(seeds, 0)
    first = []
    for row, rng in enumerate(fresh):
        expected_rng = np.random.default_rng([int(seeds[row % EPISODES_PER_POLICY]), 0])
        expected = np.array(
            [expected_rng.uniform(-1.0, 1.0), expected_rng.uniform(-1.0, 1.0), expected_rng.uniform(1.0, 1.0)],
            dtype=np.float32,
        )
        drawn = env._draw_commands(rng)
        np.testing.assert_array_equal(drawn, expected)
        first.append(drawn)

    # Drawing from one row's generator never shifts another row's stream.
    independent = env._command_generators(seeds, 0)
    for _ in range(2):
        env._draw_commands(independent[0])
    np.testing.assert_array_equal(env._draw_commands(independent[1]), first[1])

    # Recreating the generators replays every stream bit-for-bit.
    replayed = env._command_generators(seeds, 0)
    for row in range(env.rows):
        np.testing.assert_array_equal(env._draw_commands(replayed[row]), first[row])

    # The generation index is part of the seed, so a new generation is a new stream.
    next_generation = np.stack(
        [env._draw_commands(rng) for rng in env._command_generators(seeds, 1)]
    )
    assert not np.array_equal(next_generation, np.stack(first))
