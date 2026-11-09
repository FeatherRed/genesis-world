"""Seeded reset and seed-stream tests for the TensorNEAT Go2 walking stack.

``examples/locomotion/go2_neat_env.py`` is the batched, seeded evaluation environment of the
TensorNEAT Go2 walking trainer.  One master seed is split into two independent random streams --
the evolution stream (:func:`go2_neat_env.evolution_seed`, handed to the trainer's pipeline) and
the per-generation episode-seed table (:func:`go2_neat_env.generation_seed_table`) -- and every
episode seed is turned into a reset perturbation by :func:`go2_neat_env.perturbation_from_seed`
and :func:`go2_neat_env.perturbed_init_qpos`.

These tests pin the pure helper contracts (table shape / reproducibility / prefix stability, stream
independence, draw order, amplitude independence, rotation composition, seed pairing) without a
simulator.  Nothing here re-derives the implementation: every expected value comes from the
documented stream definition, an explicit rotation matrix, or the reference ``genesis.utils.geom``
helpers.

The env layer is covered through its pure pieces only (seed table -> per-row qpos mapping): building
a ``Go2NeatEnv`` (scene + four compiled graphs) costs 27 s on a warm compiler cache and 71 s cold,
far past a single test's budget, so the batched reset path is left to the training example itself.

The example modules are not a package (``go2_neat_env.py`` imports its siblings by bare name), so
``examples/locomotion`` is put on ``sys.path`` before importing them.  ``evoxc`` and ``tensorneat``
are not declared dependencies of ``genesis-world``, so the whole module is skipped where either is
missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

pytestmark = [pytest.mark.required]

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "locomotion"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import go2_neat_env as neat_env  # noqa: E402
from genesis.utils.geom import transform_by_quat, xyz_to_quat  # noqa: E402
from go2_neat_kernels import EPISODES_PER_POLICY, NUM_ACTIONS, Perturbation  # noqa: E402

MASTER_SEED = 20240601
EPISODE_SEED = 1234567
#: A plausible unperturbed Go2 qpos: base pos (3) | base quat (4, w-first) | joints (12).
INIT_QPOS = np.array(
    [
        0.0,
        0.0,
        0.42,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        1.0,
        -1.5,
        0.0,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)
QPOS_SIZE = 7 + NUM_ACTIONS


def _rotation_matrix(quat) -> np.ndarray:
    """Return the 3x3 rotation matrix of a w-first quaternion (``R(q) @ v`` rotates ``v``).

    This is the active convention of ``genesis.utils.geom`` (``transform_by_quat(v, q)`` is
    ``R(q) @ v``); the quaternion is normalized here, so a non-normalized input still yields an
    orthonormal matrix.
    """
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = w * w + x * x + y * y + z * z
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z) / norm, 2.0 * (x * y - w * z) / norm, 2.0 * (x * z + w * y) / norm],
            [2.0 * (x * y + w * z) / norm, 1.0 - 2.0 * (x * x + z * z) / norm, 2.0 * (y * z - w * x) / norm],
            [2.0 * (x * z - w * y) / norm, 2.0 * (y * z + w * x) / norm, 1.0 - 2.0 * (x * x + y * y) / norm],
        ]
    )


def _expected_row(init_qpos, seed, perturbation) -> np.ndarray:
    """Assemble the 19-vector reset of ``seed`` from the drawn offsets, with an identity init quat."""
    drawn = neat_env.perturbation_from_seed(int(seed), perturbation)
    row = np.empty(QPOS_SIZE, dtype=np.float32)
    row[:3] = init_qpos[:3] + drawn["base_pos"]
    row[3:7] = drawn["base_quat"]
    row[7:] = init_qpos[7:] + drawn["joint_pos"]
    return row


def test_generation_seed_table_shape_reproducibility_and_prefix_stability():
    table = neat_env.generation_seed_table(MASTER_SEED, 5)
    assert table.shape == (5, EPISODES_PER_POLICY)
    assert table.dtype == np.int64
    assert np.issubdtype(table.dtype, np.signedinteger)

    # Every generation owns `EPISODES_PER_POLICY` *strictly* distinct episode seeds.
    for generation, row in enumerate(table):
        assert len(set(row.tolist())) == EPISODES_PER_POLICY, (generation, row)

    # Same master seed -> byte-identical table; a different master seed -> a different table.
    np.testing.assert_array_equal(table, neat_env.generation_seed_table(MASTER_SEED, 5))
    assert not np.array_equal(table, neat_env.generation_seed_table(MASTER_SEED + 1, 5))

    # Consecutive generations differ.
    for generation in range(4):
        assert not np.array_equal(table[generation], table[generation + 1])

    # Prefix stability: rows are drawn one at a time, so asking for more generations must not
    # reshuffle the ones already materialized.
    longer = neat_env.generation_seed_table(MASTER_SEED, 8)
    np.testing.assert_array_equal(longer[:5], table)

    empty = neat_env.generation_seed_table(MASTER_SEED, 0)
    assert empty.shape == (0, EPISODES_PER_POLICY)
    assert empty.dtype == np.int64


def test_evolution_seed_is_independent_of_the_environment_stream():
    seed = neat_env.evolution_seed(MASTER_SEED)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
    assert seed == neat_env.evolution_seed(MASTER_SEED)  # stable
    assert seed != MASTER_SEED  # not the raw master seed
    assert neat_env.evolution_seed(MASTER_SEED + 1) != seed

    # The two children of `SeedSequence(master_seed).spawn(2)` never share values: the evolution
    # seed appears nowhere in the env table, however many generations are materialized.
    table = neat_env.generation_seed_table(MASTER_SEED, 12, episodes=8)
    assert seed not in set(table.ravel().tolist())

    # Consuming the environment stream (repeatedly) must not perturb the evolution stream.
    for num_generations in (1, 3, 16, 64):
        neat_env.generation_seed_table(MASTER_SEED, num_generations, episodes=EPISODES_PER_POLICY)
        assert neat_env.evolution_seed(MASTER_SEED) == seed


def test_perturbation_from_seed_determinism_and_bounds():
    perturbation = Perturbation(base_pos=0.1, base_rpy=0.2, joint_pos=0.3)
    drawn = neat_env.perturbation_from_seed(EPISODE_SEED, perturbation)
    repeated = neat_env.perturbation_from_seed(EPISODE_SEED, perturbation)

    assert set(drawn) == {"base_pos", "base_quat", "joint_pos"}
    for key, expected_shape in (("base_pos", (3,)), ("base_quat", (4,)), ("joint_pos", (NUM_ACTIONS,))):
        assert drawn[key].shape == expected_shape, key
        assert drawn[key].dtype == np.float32, key
        np.testing.assert_array_equal(drawn[key], repeated[key])  # bit-identical replay

    # Different seeds draw different perturbations.
    other = neat_env.perturbation_from_seed(EPISODE_SEED + 1, perturbation)
    for key in drawn:
        assert not np.array_equal(drawn[key], other[key]), key

    # Amplitudes bound the sampled offsets (the quaternion is a rotation, not an offset).
    assert np.all(np.abs(drawn["base_pos"]) <= perturbation.base_pos)
    assert np.all(np.abs(drawn["joint_pos"]) <= perturbation.joint_pos)
    assert np.isclose(np.linalg.norm(drawn["base_quat"]), 1.0, atol=1e-6)


def test_perturbation_from_seed_draw_order_is_amplitude_independent():
    # uniform(-a, +a) is a scaled uniform(-1, +1) draw, and the return value is float32, so
    # doubling an amplitude must double the sampled offset of that field exactly.
    for seed in (0, EPISODE_SEED, 2**31 - 1):
        for field in ("base_pos", "joint_pos"):
            for amplitude in (0.1, 0.2, 0.05):
                wide = neat_env.perturbation_from_seed(seed, Perturbation(**{field: amplitude}))[field]
                half = neat_env.perturbation_from_seed(seed, Perturbation(**{field: amplitude / 2}))[field]
                np.testing.assert_array_equal(wide, 2.0 * half, err_msg=f"{field} seed={seed} amp={amplitude}")

    # Zeroing one amplitude must not shift the other fields' draws: every field is drawn from the
    # same stream in the same (base_pos, base_rpy, joint_pos) order, whatever its amplitude.
    full = neat_env.perturbation_from_seed(EPISODE_SEED, Perturbation(0.1, 0.2, 0.3))
    no_rpy = neat_env.perturbation_from_seed(EPISODE_SEED, Perturbation(0.1, 0.0, 0.3))
    no_pos = neat_env.perturbation_from_seed(EPISODE_SEED, Perturbation(0.0, 0.2, 0.3))
    no_joints = neat_env.perturbation_from_seed(EPISODE_SEED, Perturbation(0.1, 0.2, 0.0))

    np.testing.assert_array_equal(full["base_pos"], no_rpy["base_pos"])
    np.testing.assert_array_equal(full["joint_pos"], no_rpy["joint_pos"])
    np.testing.assert_array_equal(no_rpy["base_quat"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    np.testing.assert_array_equal(full["base_quat"], no_pos["base_quat"])
    np.testing.assert_array_equal(full["joint_pos"], no_pos["joint_pos"])
    np.testing.assert_array_equal(no_pos["base_pos"], np.zeros(3, dtype=np.float32))

    np.testing.assert_array_equal(full["base_pos"], no_joints["base_pos"])
    np.testing.assert_array_equal(full["base_quat"], no_joints["base_quat"])
    np.testing.assert_array_equal(no_joints["joint_pos"], np.zeros(NUM_ACTIONS, dtype=np.float32))


def test_perturbation_from_seed_quaternion_matches_the_drawn_rpy():
    # Rebuild the documented stream by hand: three base_pos draws, then three base_rpy draws.
    perturbation = Perturbation(base_pos=0.1, base_rpy=0.25, joint_pos=0.3)
    drawn = neat_env.perturbation_from_seed(EPISODE_SEED, perturbation)

    rng = np.random.default_rng(EPISODE_SEED)
    base_pos = rng.uniform(-perturbation.base_pos, perturbation.base_pos, size=3)
    base_rpy = rng.uniform(-perturbation.base_rpy, perturbation.base_rpy, size=3)

    np.testing.assert_array_equal(np.asarray(base_pos, dtype=np.float32), drawn["base_pos"])
    expected_quat = np.asarray(xyz_to_quat(base_rpy, rpy=True), dtype=np.float32)
    np.testing.assert_array_equal(expected_quat, drawn["base_quat"])
    assert np.isclose(np.linalg.norm(drawn["base_quat"]), 1.0, atol=1e-6)


def test_zero_perturbation_reproduces_the_fixed_reset():
    zero = Perturbation()
    assert zero.is_zero
    assert not Perturbation(base_pos=0.05).is_zero
    assert not Perturbation(base_rpy=-0.05).is_zero
    assert not Perturbation(joint_pos=0.05).is_zero

    drawn = neat_env.perturbation_from_seed(EPISODE_SEED, zero)
    np.testing.assert_array_equal(drawn["base_pos"], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(drawn["base_quat"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(drawn["joint_pos"], np.zeros(NUM_ACTIONS, dtype=np.float32))

    seeds = [0, 11, 424242]
    rows = neat_env.perturbed_init_qpos(INIT_QPOS, seeds, zero)
    assert rows.shape == (len(seeds), QPOS_SIZE)
    for row in rows:
        np.testing.assert_array_equal(row, INIT_QPOS)  # bit-for-bit, not merely close

    # A non-zero perturbation breaks the fixed reset, but is still a pure function of the seed.
    perturbation = Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05)
    perturbed = neat_env.perturbed_init_qpos(INIT_QPOS, seeds, perturbation)
    for index in range(len(seeds)):
        assert not np.array_equal(perturbed[index], INIT_QPOS), index
    assert not np.array_equal(perturbed[0], perturbed[1])
    assert not np.array_equal(perturbed[1], perturbed[2])
    repeated = neat_env.perturbed_init_qpos(INIT_QPOS, [seeds[0], seeds[0], seeds[0]], perturbation)
    np.testing.assert_array_equal(repeated[0], repeated[1])
    np.testing.assert_array_equal(repeated[0], repeated[2])


def test_perturbed_init_qpos_row_layout():
    init_qpos = INIT_QPOS.copy()
    init_qpos[3:7] = np.array([0.9, 0.1, -0.2, 0.3], dtype=np.float32)  # non-identity, not unit
    init_qpos[3:7] /= np.linalg.norm(init_qpos[3:7])
    perturbation = Perturbation(base_pos=0.1, base_rpy=0.2, joint_pos=0.3)
    seeds = [11, 22, 33]
    rows = neat_env.perturbed_init_qpos(init_qpos, seeds, perturbation)

    assert rows.shape == (3, QPOS_SIZE)
    assert rows.dtype == np.float32
    for index, seed in enumerate(seeds):
        drawn = neat_env.perturbation_from_seed(seed, perturbation)
        np.testing.assert_array_equal(rows[index, :3], init_qpos[:3] + drawn["base_pos"])
        np.testing.assert_array_equal(rows[index, 7:], init_qpos[7:] + drawn["joint_pos"])
        # The base rotation is composed with the authored orientation, not copied from either.
        assert not np.array_equal(rows[index, 3:7], drawn["base_quat"])
        assert not np.array_equal(rows[index, 3:7], init_qpos[3:7])
        assert np.isclose(np.linalg.norm(rows[index, 3:7]), 1.0, atol=1e-6)

    for wrong_size in (QPOS_SIZE - 1, QPOS_SIZE + 1, 0):
        with pytest.raises(ValueError, match="init_qpos must hold"):
            neat_env.perturbed_init_qpos(np.zeros(wrong_size, dtype=np.float32), seeds, perturbation)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "perturbed_init_qpos post-multiplies the sampled base rotation by the authored init "
        "quaternion: transform_quat_by_quat(drawn, init) == quatmul(init, drawn), i.e. "
        "R(row) == R(init) @ R(pert). Its docstring documents the opposite (q_pert * q_init, "
        "'the perturbation acts in world axes after the authored orientation', "
        "R(row) == R(pert) @ R(init)). Invisible with the identity base_init_quat of the Go2 "
        "config; measured with init_quat=[0.5, 0.5, 0.5, 0.5] (90 deg about z) and "
        "pert_quat=[0.7071, 0.7071, 0, 0] (90 deg about x): expected [0.5, 0.5, -0.5, 0.5], "
        "actual [0.5, 0.5, 0.5, 0.5]."
    ),
)
def test_perturbed_init_qpos_composes_rotation_after_init_orientation():
    init_qpos = INIT_QPOS.copy()
    init_qpos[3:7] = np.array([0.9, 0.1, -0.2, 0.3], dtype=np.float32)
    init_qpos[3:7] /= np.linalg.norm(init_qpos[3:7])
    perturbation = Perturbation(base_pos=0.05, base_rpy=0.1, joint_pos=0.05)
    seed = 987654321

    # Sanity-check the explicit matrix against the genesis rotation convention: R(q) @ v is
    # transform_by_quat(v, q) for a body-frame vector v.
    probe = np.array([0.3, -0.5, 0.8], dtype=np.float32)
    probe_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    np.testing.assert_allclose(_rotation_matrix(probe_quat) @ probe, transform_by_quat(probe, probe_quat), atol=1e-6)

    row = neat_env.perturbed_init_qpos(init_qpos, [seed], perturbation)[0]
    drawn = neat_env.perturbation_from_seed(seed, perturbation)

    np.testing.assert_allclose(
        _rotation_matrix(row[3:7]),
        _rotation_matrix(drawn["base_quat"]) @ _rotation_matrix(init_qpos[3:7]),
        atol=1e-5,
    )


def test_perturbed_init_qpos_uses_the_seed_at_its_index():
    perturbation = Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05)
    seeds = [11, 22, 33]
    rows = neat_env.perturbed_init_qpos(INIT_QPOS, seeds, perturbation)

    # Row i is built from seeds[i] alone. INIT_QPOS carries the identity base quat, so the row
    # quaternion is the sampled rotation re-normalized by the composition (hence the 1e-6
    # tolerance on the quaternion slots, while the two offset blocks are exact).
    for index, seed in enumerate(seeds):
        expected = _expected_row(INIT_QPOS, seed, perturbation)
        np.testing.assert_array_equal(rows[index, :3], expected[:3])
        np.testing.assert_array_equal(rows[index, 7:], expected[7:])
        np.testing.assert_allclose(rows[index, 3:7], expected[3:7], rtol=0.0, atol=1e-6)
        single = neat_env.perturbed_init_qpos(INIT_QPOS, [seed], perturbation)
        np.testing.assert_array_equal(rows[index], single[0])
    assert not np.array_equal(rows[0], rows[1])

    # Permuting the seed order permutes the rows identically.
    order = [2, 0, 1]
    permuted = neat_env.perturbed_init_qpos(INIT_QPOS, [seeds[i] for i in order], perturbation)
    for target, source in enumerate(order):
        np.testing.assert_array_equal(permuted[target], rows[source])
