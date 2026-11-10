"""Self-check of the shared ``examples/neuroevolution`` layer: numpy, ``evoxc`` and TensorNEAT only.

Run it from the repository root::

    PYTHONPATH=$PWD/examples/neuroevolution /root/evox-robotics/.venv/bin/python examples/neuroevolution/selfcheck.py

Every check is one function that prints exactly one ``PASS <name>`` line; the first failure prints ``FAIL <name>: ...`` and makes the process exit non-zero.
No simulator, no ``torch``, no ``tensordict``, no pytest and no artifact outside ``$TMPDIR`` is involved, and the whole run takes a few seconds on the numpy backend.

What each check pins, in order:

``seeding``
    The documented seed constants bit for bit (``evolution_seed(0) == 3757552657``, the ``generation_seed_table(0, 3)`` table quoted from ``examples/locomotion/CONTEXT.md``), determinism, prefix stability, distinct seeds per row, dtype/shape, master-seed sensitivity, the independence of the evolution and env streams, and the perturbation draws (zero -> exact zeros, nonzero -> reproducible, bounded, ``num_dof``-sized, float32).

``env_base``
    The ``evaluate`` result contract: a well-formed synthetic result is accepted, an off-by-1e-3 fitness, a wrong ``returns`` length, a wrong ``seeds`` shape, duplicate seeds, an integer ``episode_returns`` and a missing required key are each rejected with ``ValueError``, while an off-by-1e-7 fitness is inside tolerance; plus ``rows_for`` and the runtime-checkable protocol.
    The non-finite case is pinned too: a row holding ``inf``, ``-inf`` or ``NaN`` with a fitness that counts it as 0 validates, while a fitness that keeps the infinity is rejected, and the rule has a single owner (``env_base._fitness_from_episode_returns``) that ``trainer._aggregate`` imports instead of restating it.

``graphs``
    ``build_fitness`` / ``run_fitness`` (the reshape-mean, and a cache hit on the second lookup), ``build_tell`` / ``run_tell`` (payload round trip, generation increment, randkey advance, rebuilt ``SpeciesState``) and ``build_policy`` / ``run_policy`` for both real genome widths (shape, dtype, the policy-major row tiling, distinct policies, and the width/row mismatches).

``trainer``
    End-to-end determinism of :class:`trainer.NeuroevolutionTrainer` against a simulator-free env: identical fitness history, records (timings aside) and seed table over two runs, the env is never closed, the ``output_dir`` artifacts, and the fitness tolerance.

``device_contract``
    The device result contract on the plumbing-only ``cpu:0`` device: ``validate_evaluation_device`` accepts a well-formed device result and rejects a wrong dtype, a wrong shape, a host array where a tensor is required, a ``cpu:0`` tensor when ``rocm:0`` is expected, and a missing key; ``evolution_graphs.fitness_rule_graph`` is bit-identical to ``env_base._fitness_from_episode_returns`` on ``NaN``/``+inf``/``-inf`` columns at the Go2 magnitude; ``build_verification`` accepts a correct result and rejects a ``1e-3`` fitness offset and a ``returns`` that is not the policy-major tiling, in both residencies; and ``device_runtime.device_setup`` is bitwise identical to ``Pipeline(...).setup()`` for both genome widths (the four payload leaves, all ten species leaves, the key and the generation counter).

``device_trainer``
    The device residency of the trainer on ``cpu:0``: two runs of one master seed share their fitness history and records (timings and cache counters aside), every runtime leaf is placed on ``cpu:0``, every ``evaluate`` call receives four ``evoxc`` tensors and no numpy array, the numpy residency of the same env agrees on the fitness history, a ``NaN`` episode is counted as 0 and a fitness off by ``1e-3`` is rejected.

``device_hardware``
    The same residency on the first non-CPU ``evoxc`` device (``iree`` unless ``EVOXC_BACKEND``/``EVOXC_DEVICE`` say otherwise): placement at initialization, at the fitness, at the tell output and on the next generation's payload, and a fitness history that matches the numpy residency within ``atol=rtol=1e-5`` (the measured absolute and relative diff are in the assertion message) -- exact equality is only promised for the integer leaves, ``randkey`` and ``generation``, because the backend's float drift accumulates with the population; with no accelerator the check prints ``SKIP device_hardware`` and the CPU checks alone decide the exit code.

``large_returns``
    The aggregation rule at the Go2 pilot's episode-return magnitude (~±100): an env whose float32 returns sit there and whose fitness is the real ``evolution_graphs`` fitness graph must be accepted, and the trainer's fitness must equal that graph bit for bit while a float64 mean of the same returns differs by more than the tolerance; the same env with the fitness off by 1e-3 must still be rejected, so the check is not vacuous.

``import_hygiene``
    An AST scan proving every sibling module of this directory -- ``selfcheck.py`` and the ``verify_device.py`` CLI included -- imports no ``torch`` / ``genesis`` / ``tensordict`` and no ``go2_*`` / ``hover_*`` task module, and that those packages are absent from ``sys.modules``.

Genome budgets are the smallest the dense initial network of a width allows, because ``DefaultGenome`` rejects a budget smaller than ``num_inputs + num_outputs`` nodes and ``num_inputs * num_outputs`` connections.

Only the ``device_hardware`` check needs an accelerator; every other check runs on ``cpu:0`` with the ``numpy`` backend, which is the plumbing-only mode of the device residency (placement, tiling and staging are exercised, no GPU is used).
It runs last on purpose: ``evoxc.devices()`` imports ``torch`` to probe the GPU runtimes, so the ``sys.modules`` half of ``import_hygiene`` has to be checked while only the numpy path has run.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import evoxc

from tensorneat import Pipeline, algorithm, genome
from tensorneat.algorithm.neat.species import SpeciesState
from tensorneat.common.tools import to_numpy

from device_env import SyntheticDeviceEnv
from device_runtime import (
    assert_payload_resident,
    device_placement,
    device_setup,
    is_device_tensor,
)
from env_base import (
    BatchedEpisodeEnv,
    _fitness_from_episode_returns,
    rows_for,
    validate_evaluation,
    validate_evaluation_device,
)
from evolution_graphs import (
    ExecutableCache,
    build_fitness,
    build_fitness_rule,
    build_tell,
    build_verification,
    run_fitness,
    run_fitness_rule,
    run_tell,
    run_verification,
)
from evox_runtime import _to_numpy, resolve_backend_device
from policy_graphs import build_policy, run_policy
from seeding import (
    EPISODES_PER_POLICY,
    Perturbation,
    evolution_seed,
    generation_seed_table,
    perturbation_from_seed,
)
from trainer import NeuroevolutionTrainer, TraceOnlyProblem, pop_arrays_of, species_leaves_of
from verify_device import ensure_pjrt_plugin, float_max_abs_diff, float_max_rel_diff, reachable_devices

#: Policies per generation in every check: small enough that the numpy backend stays at a few seconds.
POP_SIZE = 3
#: Generations the trainer check runs.
GENERATIONS = 3
#: Generations the large-return check runs: the failure mode reproduces on the first one, two keep it cheap.
LARGE_RETURN_GENERATIONS = 2
#: Episode-return magnitude of the large-return check, the Go2 pilot's ~±100 per episode.
LARGE_RETURN_SCALE = 100.0
#: Episodes per policy, the documented triple every generation shares.
EPISODES = EPISODES_PER_POLICY
#: ``(num_inputs, num_outputs)`` of the two real tasks: Go2 walking (45 -> 12) and drone hover (17 -> 4).
GENOME_WIDTHS = ((45, 12), (17, 4))
#: The one master seed the trainer check documents its reproducibility guarantee for.
MASTER_SEED = 0
#: ``evoxc`` backend and device of the plumbing-only device checks: the host, no compiler required.
PLUMBING_BACKEND = "numpy"
PLUMBING_DEVICE = "cpu:0"
#: Population size and generations of the plumbing-only device checks.
DEVICE_POP_SIZE = POP_SIZE
DEVICE_GENERATIONS = 2
#: Population size and generations of the accelerator check, at the Go2 pilot's population.
HARDWARE_POP_SIZE = 8
HARDWARE_GENERATIONS = 2
#: Absolute and relative tolerance of the accelerator check's numpy-parity fitness comparison.
#: The layer promises exactness only for the integer leaves, ``randkey`` and ``generation``; the
#: ``iree`` tell/init float drift accumulates with the population (exact at pop 8 and 32, ~1.5e-05
#: at pop 64), so the fitness history is compared with the same ``atol = rtol`` the CLI uses.
FITNESS_TOLERANCE = 1e-5
#: The sibling modules the import-hygiene check parses.
HYGIENE_FILES = (
    "seeding.py",
    "env_base.py",
    "evox_runtime.py",
    "evolution_graphs.py",
    "policy_graphs.py",
    "device_runtime.py",
    "device_env.py",
    "trainer.py",
    "verify_device.py",
)
#: Top-level packages the shared layer must stay importable without.
FORBIDDEN_PACKAGES = ("torch", "genesis", "tensordict")
#: Module prefixes of the task modules the shared layer must never import.
FORBIDDEN_MODULE_PREFIXES = ("go2_", "hover_")
#: Record and config fields derived from wall-clock time, stripped before two runs are compared.
TIMING_FIELDS = (
    "wall_time_s",
    "steps_per_s",
    "env_steps_per_s",
    "executable_build_time_s",
    "tell_build_time_s",
)
#: Record and config fields reflecting the state of an executable cache, stripped with the timings.
CACHE_FIELDS = (
    "cache_hits",
    "cache_misses",
    "cache_hits_delta",
    "cache_misses_delta",
)


# --- Helpers ------------------------------------------------------------------------


def _require(condition, message):
    """Raise ``AssertionError(message)`` when ``condition`` is falsy."""
    if not condition:
        raise AssertionError(message)


def _expect_value_error(thunk, label):
    """Run ``thunk``, require a ``ValueError``, and return its message."""
    try:
        result = thunk()
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"{label} must raise ValueError, got {result!r}")


def _strip_timings(value):
    """Return ``value`` without the wall-clock-derived and cache-derived fields, recursively."""
    if isinstance(value, dict):
        return {
            key: _strip_timings(item)
            for key, item in value.items()
            if key not in TIMING_FIELDS and key not in CACHE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_timings(item) for item in value]
    return value


def _array_equal(left, right):
    """Exact equality of two arrays, treating the NaN padding of an unused genome slot as equal."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left, right, equal_nan=True))


def _plumbing_device():
    """The ``cpu:0`` ``evoxc`` device object of the plumbing-only device checks."""
    return resolve_backend_device(PLUMBING_BACKEND, PLUMBING_DEVICE)[1]


def _stage(array):
    """Stage one host array as a ``cpu:0`` device tensor, the plumbing mode's only transfer."""
    evoxc.backends.get(PLUMBING_BACKEND)
    return evoxc.from_numpy(np.ascontiguousarray(np.asarray(array)))


def _device_result(episode_returns, reported_fitness=None):
    """Build a well-formed device ``evaluate`` result of ``episode_returns`` on ``cpu:0``."""
    episode_returns = np.asarray(episode_returns, dtype=np.float32)
    pop_size, episodes = episode_returns.shape
    fitness = _fitness_from_episode_returns(episode_returns) if reported_fitness is None else reported_fitness
    return {
        "seeds": _stage(np.arange(11, 11 + episodes, dtype=np.int64)),
        "episode_returns": _stage(episode_returns),
        "returns": _stage(episode_returns.reshape(-1)),
        "fitness": _stage(fitness),
        "steps": 25,
        "env_steps": 25 * rows_for(pop_size, episodes),
        "wall_time_s": 0.01,
    }


class _EnvWrapper:
    """Delegate every attribute to the wrapped env, so a wrapper stays a ``BatchedEpisodeEnv``."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def close(self):
        """Close the wrapped env."""
        self.env.close()


class _PayloadRecorder(_EnvWrapper):
    """Record the genome payload every ``evaluate`` call receives, without changing anything else."""

    def __init__(self, env):
        super().__init__(env)
        self.payloads = []

    def evaluate(self, pop_arrays, generation):
        """Record the payload of one generation and run it on the wrapped env."""
        self.payloads.append(tuple(pop_arrays))
        return self.env.evaluate(pop_arrays, generation)


class _FitnessOffsetEnv(_EnvWrapper):
    """Report the wrapped env's fitness shifted by ``offset``, to drive the device verification's tolerance."""

    def __init__(self, env, offset):
        super().__init__(env)
        self.offset = np.float32(offset)

    def evaluate(self, pop_arrays, generation):
        """Run one generation and shift the device fitness tensor by the offset."""
        result = self.env.evaluate(pop_arrays, generation)
        return dict(result, fitness=_stage(_to_numpy(result["fitness"]) + self.offset))


def _imported_modules(tree):
    """Return every dotted module path an ``ast`` tree imports, ``from`` target names included."""
    paths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            paths.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                paths.append(module)
            paths.extend(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
    return paths


def _genome(num_inputs, num_outputs):
    """Return the genome of one width with the smallest node/connection budget it admits.

    ``DefaultGenome`` must hold the dense initial network, i.e. ``num_inputs + num_outputs`` nodes and ``num_inputs * num_outputs`` connections, so the nominal 20/40 trace budget cannot express the real 45 -> 12 width (57 nodes, 540 connections).
    The minimum feasible budget keeps the check fast and still leaves the genome free to mutate.
    """
    return genome.DefaultGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=num_inputs + num_outputs,
        max_conns=num_inputs * num_outputs,
    )


def _build_pipeline(num_inputs, num_outputs, seed=MASTER_SEED):
    """Set up the TensorNEAT pipeline of one genome width; returns ``(algo, config, state)``."""
    algo = algorithm.NEAT(_genome(num_inputs, num_outputs), pop_size=POP_SIZE, species_size=min(3, POP_SIZE))
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(num_inputs, num_outputs),
        seed=seed,
        backend="numpy",
        is_detailed=False,
    )
    state = pipeline.setup()
    return algo, NeuroevolutionTrainer._static_config(pipeline, state), state


# --- 1. Seeding ---------------------------------------------------------------------


def check_seeding():
    """Pin the documented seed rules, the stream independence and the perturbation draws."""
    _require(EPISODES_PER_POLICY == 3, f"EPISODES_PER_POLICY must stay 3, got {EPISODES_PER_POLICY}")
    _require(
        evolution_seed(0) == 3757552657,
        f"evolution_seed(0) must be 3757552657, got {evolution_seed(0)}",
    )
    documented = [
        [1454269175, 757617120, 1421746521],
        [1313752747, 908599966, 53557616],
        [1683657442, 1375977993, 1654802006],
    ]
    table = generation_seed_table(0, 3)
    _require(table.tolist() == documented, f"generation_seed_table(0, 3) must be {documented}, got {table.tolist()}")
    _require(table.dtype == np.int64, f"the seed table must be int64, got {table.dtype}")
    _require(table.shape == (3, 3), f"the seed table must be (3, 3), got {table.shape}")
    _require(np.array_equal(table, generation_seed_table(0, 3)), "two identical seed-table calls must compare equal")
    _require(
        np.array_equal(generation_seed_table(0, 5)[:3], table),
        "generation_seed_table(0, 5)[:3] must equal generation_seed_table(0, 3) (prefix stability)",
    )
    for row_index, row in enumerate(table):
        _require(
            np.unique(row).size == row.size,
            f"row {row_index} seeds must be pairwise distinct, got {row.tolist()}",
        )

    other_table = generation_seed_table(1, 3)
    _require(other_table.shape == (3, 3) and other_table.dtype == np.int64, "master_seed=1 must keep the dtype/shape")
    _require(not np.array_equal(other_table, table), "master_seed=1 must give a different seed table")
    _require(evolution_seed(1) != evolution_seed(0), "master_seed=1 must give a different evolution_seed")

    wider = generation_seed_table(0, 3, episodes=4)
    _require(
        wider.shape == (3, 4) and wider.dtype == np.int64,
        f"episodes=4 must give a (3, 4) int64 table, got {wider.shape}",
    )
    _require(not np.array_equal(wider[:, :3], table), "a different episode count must draw a different env table")
    _require(
        evolution_seed(0) == 3757552657,
        "the evolution stream must not move when only the episode count differs",
    )

    zero = Perturbation()
    _require(zero.is_zero, "Perturbation() must be the zero perturbation")
    zero_draw = perturbation_from_seed(12345, zero, 12)
    _require(
        np.array_equal(zero_draw["base_pos"], np.zeros(3, dtype=np.float32)),
        f"a zero perturbation must draw a zero base_pos, got {zero_draw['base_pos']}",
    )
    _require(
        np.array_equal(zero_draw["base_quat"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        f"a zero perturbation must draw base_quat [1, 0, 0, 0], got {zero_draw['base_quat']}",
    )
    _require(
        np.array_equal(zero_draw["joint_pos"], np.zeros(12, dtype=np.float32)),
        f"a zero perturbation must draw zero joint_pos, got {zero_draw['joint_pos']}",
    )
    for key, value in zero_draw.items():
        _require(value.dtype == np.float32, f"the zero draw {key} must be float32, got {value.dtype}")

    nonzero = Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05)
    _require(not nonzero.is_zero, "a nonzero Perturbation must not report is_zero")
    first = perturbation_from_seed(7, nonzero, 12)
    replay = perturbation_from_seed(7, nonzero, 12)
    _require(
        all(np.array_equal(first[key], replay[key]) for key in first),
        "the same seed must reproduce the same perturbation bit for bit",
    )
    elsewhere = perturbation_from_seed(8, nonzero, 12)
    _require(
        not np.array_equal(first["base_pos"], elsewhere["base_pos"]),
        "different seeds must give a different base_pos",
    )
    _require(
        not np.array_equal(first["joint_pos"], elsewhere["joint_pos"]),
        "different seeds must give a different joint_pos",
    )
    for key, shape in (("base_pos", (3,)), ("base_quat", (4,)), ("joint_pos", (12,))):
        _require(first[key].shape == shape, f"the drawn {key} must have shape {shape}, got {first[key].shape}")
        _require(first[key].dtype == np.float32, f"the drawn {key} must be float32, got {first[key].dtype}")
    _require(float(np.abs(first["base_pos"]).max()) <= 0.05, f"|base_pos| must stay <= 0.05, got {first['base_pos']}")
    _require(
        float(np.abs(first["joint_pos"]).max()) <= 0.05,
        f"|joint_pos| must stay <= 0.05, got {first['joint_pos']}",
    )
    narrow = perturbation_from_seed(7, nonzero, 4)
    _require(
        narrow["joint_pos"].shape == (4,),
        f"num_dof=4 must give a (4,) joint_pos, got {narrow['joint_pos'].shape}",
    )
    _require(narrow["joint_pos"].size != first["joint_pos"].size, "num_dof must size the joint_pos vector")
    _require(
        np.array_equal(narrow["joint_pos"], first["joint_pos"][:4]),
        "the joint_pos draws must not shift with num_dof (the first num_dof offsets are shared)",
    )


# --- 2. Env contract ----------------------------------------------------------------


class _MinimalEnv:
    """The smallest object that satisfies :class:`env_base.BatchedEpisodeEnv`.

    It never runs anything; the check only asks whether the runtime-checkable protocol accepts it, so it declares the numpy residency and no device of its own.
    """

    residency = "numpy"
    backend = None
    device = None

    def __init__(self):
        self.rows = rows_for(1, EPISODES)
        self.episodes_per_policy = EPISODES
        self.max_episode_length = 1
        self.cache = None

    def evaluate(self, pop_arrays, generation):
        """Raise: the minimal env exists for the protocol check only."""
        raise NotImplementedError("the minimal env is never evaluated")

    def close(self):
        """Do nothing: the minimal env owns no resource."""


def check_env_base():
    """Pin ``validate_evaluation`` acceptance/rejection and the row layout."""
    pop_size = 4
    _require(rows_for(4) == 12, f"rows_for(4) must be 12, got {rows_for(4)}")
    _require(isinstance(_MinimalEnv(), BatchedEpisodeEnv), "a minimal env must satisfy BatchedEpisodeEnv")

    seeds = generation_seed_table(MASTER_SEED, 1)[0]
    episode_returns = np.array(
        [
            [0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0],
            [3.5, 4.0, 4.5],
            [5.0, 5.5, 6.0],
        ],
        dtype=np.float32,
    )
    well_formed = {
        "seeds": seeds,
        "episode_returns": episode_returns,
        "returns": episode_returns.reshape(-1),
        "fitness": episode_returns.mean(axis=1),
        "steps": pop_size * EPISODES * 5,
        "env_steps": pop_size * EPISODES * 5,
        "wall_time_s": 0.25,
    }
    _require(validate_evaluation(well_formed, pop_size) is None, "a well-formed result must be accepted")
    _require(
        np.array_equal(well_formed["fitness"], episode_returns.mean(axis=1)),
        "the synthetic fitness must be the exact episode mean",
    )

    off_by_tolerance = dict(well_formed, fitness=well_formed["fitness"] + np.float32(1e-7))
    _require(
        validate_evaluation(off_by_tolerance, pop_size) is None,
        "a fitness off by 1e-7 is inside tolerance and must be accepted",
    )
    off_by_1e3 = dict(well_formed, fitness=well_formed["fitness"] + np.float32(1e-3))
    message = _expect_value_error(lambda: validate_evaluation(off_by_1e3, pop_size), "a fitness off by 1e-3")
    _require("fitness" in message, f"the rejection message must name the fitness, got {message}")

    non_finite = np.array(episode_returns, dtype=np.float32)
    non_finite[1, 2] = np.inf
    non_finite[2, 0] = -np.inf
    non_finite[3, 1] = np.nan
    scored = dict(
        well_formed,
        episode_returns=non_finite,
        returns=non_finite.reshape(-1),
        fitness=np.array([1.0, 1.5, 8.5 / 3.0, 11.0 / 3.0], dtype=np.float32),
    )
    _require(
        validate_evaluation(scored, pop_size) is None,
        "a non-finite episode return with a fitness that counts it as 0 must be accepted",
    )
    unscored = dict(scored, fitness=np.array([1.0, np.inf, 8.5 / 3.0, 11.0 / 3.0], dtype=np.float32))
    with np.errstate(invalid="ignore"):
        message = _expect_value_error(
            lambda: validate_evaluation(unscored, pop_size), "a fitness that keeps an infinite episode return"
        )
    _require("fitness" in message, f"the rejection message must name the fitness, got {message}")

    integer_returns = episode_returns.astype(np.int32)
    _expect_value_error(
        lambda: validate_evaluation(
            dict(well_formed, episode_returns=integer_returns, returns=integer_returns.reshape(-1)), pop_size
        ),
        "an integer episode_returns",
    )
    _expect_value_error(
        lambda: validate_evaluation(dict(well_formed, returns=well_formed["returns"][:-1]), pop_size),
        "a wrong returns length",
    )
    _expect_value_error(
        lambda: validate_evaluation(dict(well_formed, seeds=seeds.reshape(1, -1)), pop_size),
        "a wrong seeds shape",
    )
    _expect_value_error(
        lambda: validate_evaluation(dict(well_formed, seeds=np.array([seeds[0], seeds[0], seeds[2]])), pop_size),
        "duplicate seeds",
    )
    missing = {key: value for key, value in well_formed.items() if key != "env_steps"}
    message = _expect_value_error(lambda: validate_evaluation(missing, pop_size), "a missing required key")
    _require("env_steps" in message, f"the rejection message must name the missing key, got {message}")


# --- 2b. Device result contract ------------------------------------------------------


def _host_init_state(num_inputs, num_outputs, pop_size):
    """Return the ``Pipeline.setup()`` state of one width: the host reference of device-first initialization."""
    algo = algorithm.NEAT(_genome(num_inputs, num_outputs), pop_size=pop_size, species_size=min(3, pop_size))
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(num_inputs, num_outputs),
        seed=evolution_seed(MASTER_SEED),
        backend="numpy",
        is_detailed=False,
    )
    return pipeline.setup()


def check_device_contract():
    """Pin the device result contract, the device fitness rule, the verification graph and device-first init."""
    pop_size = DEVICE_POP_SIZE
    episode_returns = np.array(
        [
            [99.5, -42.25, 7.0],
            [np.nan, 100.0, -100.0],
            [np.inf, -np.inf, 3.5],
        ],
        dtype=np.float32,
    )
    _require(
        episode_returns.shape == (pop_size, EPISODES),
        f"the device contract check needs a ({pop_size}, {EPISODES}) return matrix, got {episode_returns.shape}",
    )
    well_formed = _device_result(episode_returns)
    _require(
        validate_evaluation_device(well_formed, pop_size) is None,
        "a well-formed device result must be accepted",
    )
    _require(
        validate_evaluation_device(well_formed, pop_size, device=_plumbing_device()) is None,
        "a well-formed device result on cpu:0 must be accepted when cpu:0 is the expected device",
    )

    message = _expect_value_error(
        lambda: validate_evaluation_device(
            dict(
                well_formed,
                episode_returns=_stage(
                    np.nan_to_num(episode_returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
                ),
            ),
            pop_size,
        ),
        "an integer episode_returns tensor",
    )
    _require("episode_returns" in message, f"the rejection message must name episode_returns, got {message}")
    message = _expect_value_error(
        lambda: validate_evaluation_device(
            dict(well_formed, returns=_stage(episode_returns.reshape(-1)[:-1])), pop_size
        ),
        "a wrong returns shape",
    )
    _require("returns" in message, f"the rejection message must name returns, got {message}")
    message = _expect_value_error(
        lambda: validate_evaluation_device(dict(well_formed, episode_returns=episode_returns), pop_size),
        "a host array where a device tensor is required",
    )
    _require("episode_returns" in message, f"the rejection message must name episode_returns, got {message}")
    message = _expect_value_error(
        lambda: validate_evaluation_device(
            well_formed, pop_size, device=resolve_backend_device(PLUMBING_BACKEND, "rocm:0")[1]
        ),
        "a cpu:0 tensor when the generation is expected on rocm:0",
    )
    _require("rocm:0" in message, f"the rejection message must name rocm:0, got {message}")
    message = _expect_value_error(
        lambda: validate_evaluation_device(
            {key: value for key, value in well_formed.items() if key != "env_steps"}, pop_size
        ),
        "a device result missing a required key",
    )
    _require("env_steps" in message, f"the rejection message must name the missing key, got {message}")

    rule = build_fitness_rule(pop_size, episodes_per_policy=EPISODES, backend=PLUMBING_BACKEND, device=PLUMBING_DEVICE)
    graph_fitness = run_fitness_rule(rule, episode_returns.reshape(-1))
    host_fitness = _fitness_from_episode_returns(episode_returns)
    _require(
        np.asarray(graph_fitness).shape == (pop_size,) and np.asarray(graph_fitness).dtype == np.float32,
        f"the device fitness rule must return ({pop_size},) float32, got "
        f"{np.asarray(graph_fitness).shape} {np.asarray(graph_fitness).dtype}",
    )
    _require(
        np.array_equal(np.asarray(graph_fitness), host_fitness),
        f"the device fitness rule {np.asarray(graph_fitness).tolist()} must equal "
        f"env_base._fitness_from_episode_returns {host_fitness.tolist()} bit for bit",
    )
    _require(
        np.all(np.isfinite(host_fitness)) and host_fitness[1] == np.float32(0.0),
        f"a non-finite episode must count as 0 and never propagate, got {host_fitness.tolist()}",
    )
    exact_returns = np.array(
        [
            [np.inf, 2.0, 4.0],
            [-np.inf, np.nan, 3.0],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    _require(
        np.array_equal(
            np.asarray(run_fitness_rule(rule, exact_returns.reshape(-1))),
            np.array([2.0, 1.0, 0.0], dtype=np.float32),
        ),
        "the device rule must zero every non-finite episode before the float32 mean, so [inf, 2, 4] -> 2, "
        f"[-inf, nan, 3] -> 1 and an all-NaN row -> 0, got "
        f"{np.asarray(run_fitness_rule(rule, exact_returns.reshape(-1))).tolist()}",
    )

    verification = build_verification(
        pop_size, episodes_per_policy=EPISODES, backend=PLUMBING_BACKEND, device=PLUMBING_DEVICE
    )
    returns = episode_returns.reshape(-1)
    accepted = run_verification(verification, returns=returns, episode_returns=episode_returns, fitness=host_fitness)
    _require(accepted["returns_match"], f"a correct device result must match its tiling, got {accepted}")
    _require(accepted["fitness_match"], f"a correct device result must match the fitness rule, got {accepted}")
    _require(
        accepted["fitness_max_abs_diff"] == 0.0,
        f"a correct device result must have no fitness deviation, got {accepted['fitness_max_abs_diff']}",
    )
    offset = run_verification(
        verification,
        returns=returns,
        episode_returns=episode_returns,
        fitness=host_fitness + np.float32(1e-3),
    )
    _require(
        offset["returns_match"] and not offset["fitness_match"], f"a 1e-3 fitness offset must be rejected: {offset}"
    )
    _require(
        offset["fitness_max_abs_diff"] >= 1e-3,
        f"the reported deviation must carry the offset, got {offset['fitness_max_abs_diff']}",
    )
    rolled = run_verification(
        verification, returns=np.roll(returns, 1), episode_returns=episode_returns, fitness=host_fitness
    )
    _require(not rolled["returns_match"], f"a returns tensor that is not the tiling must be rejected: {rolled}")

    staged = run_verification(
        verification,
        returns=_stage(returns),
        episode_returns=_stage(episode_returns),
        fitness=_stage(host_fitness),
        residency="device",
    )
    _require(
        staged["returns_match"] and staged["fitness_match"] and staged["fitness_max_abs_diff"] == 0.0,
        f"the verification graph must accept the same result as device tensors, got {staged}",
    )
    staged_offset = run_verification(
        verification,
        returns=_stage(returns),
        episode_returns=_stage(episode_returns),
        fitness=_stage(host_fitness + np.float32(1e-3)),
        residency="device",
    )
    _require(
        staged_offset["returns_match"] and not staged_offset["fitness_match"],
        f"the verification graph must reject a 1e-3 fitness offset on device tensors, got {staged_offset}",
    )

    for num_inputs, num_outputs in GENOME_WIDTHS:
        host_state = _host_init_state(num_inputs, num_outputs, pop_size)
        algo = algorithm.NEAT(_genome(num_inputs, num_outputs), pop_size=pop_size, species_size=min(3, pop_size))
        _config, device_state = device_setup(
            algo,
            TraceOnlyProblem(num_inputs, num_outputs),
            seed=evolution_seed(MASTER_SEED),
            backend=PLUMBING_BACKEND,
            device=PLUMBING_DEVICE,
        )
        device_pop = tuple(_to_numpy(leaf) for leaf in device_state.pop)
        device_species = tuple(_to_numpy(leaf) for leaf in evoxc.tree_leaves(device_state.species))
        assert_payload_resident(
            tuple(device_state.pop),
            tuple(evoxc.tree_leaves(device_state.species)),
            device_state.randkey,
            device_state.generation,
            _plumbing_device(),
        )
        host_pop = pop_arrays_of(host_state)
        host_species = tuple(_to_numpy(leaf) for leaf in species_leaves_of(host_state.species))
        _require(
            len(device_pop) == len(host_pop),
            f"the {num_inputs} -> {num_outputs} payload must have {len(host_pop)} leaves, got {len(device_pop)}",
        )
        for index, (host_leaf, device_leaf) in enumerate(zip(host_pop, device_pop)):
            _require(
                _array_equal(host_leaf, device_leaf),
                f"the {num_inputs} -> {num_outputs} payload leaf {index} of device-first init must equal "
                f"Pipeline.setup() bit for bit (max abs diff {float_max_abs_diff(host_leaf, device_leaf)!r})",
            )
        _require(
            len(device_species) == len(host_species),
            f"the species pytree must keep its {len(host_species)} leaves, got {len(device_species)}",
        )
        for index, (host_leaf, device_leaf) in enumerate(zip(host_species, device_species)):
            _require(
                _array_equal(host_leaf, device_leaf),
                f"the {num_inputs} -> {num_outputs} species leaf {index} of device-first init must equal "
                f"Pipeline.setup() bit for bit (max abs diff {float_max_abs_diff(host_leaf, device_leaf)!r})",
            )
        _require(
            int(_to_numpy(host_state.randkey)) == int(_to_numpy(device_state.randkey)),
            f"the {num_inputs} -> {num_outputs} randkey must survive device-first init, got "
            f"{int(_to_numpy(device_state.randkey))} instead of {int(_to_numpy(host_state.randkey))}",
        )
        _require(
            float(_to_numpy(host_state.generation)) == float(_to_numpy(device_state.generation)),
            f"the {num_inputs} -> {num_outputs} generation counter must start at the host value, got "
            f"{float(_to_numpy(device_state.generation))}",
        )


# --- 3. Graphs ----------------------------------------------------------------------


def check_graphs():
    """Pin the fitness, tell and policy graphs of both real genome widths."""
    built = {}
    for num_inputs, num_outputs in GENOME_WIDTHS:
        algo, config, state = _build_pipeline(num_inputs, num_outputs)
        built[(num_inputs, num_outputs)] = (
            algo,
            config,
            state,
            pop_arrays_of(state),
            species_leaves_of(state.species),
        )

    rows = POP_SIZE * EPISODES
    returns = np.arange(rows, dtype=np.float32) * np.float32(0.5) - np.float32(1.0)
    expected = returns.reshape(POP_SIZE, EPISODES).mean(axis=1, dtype=np.float32)
    _require(expected.dtype == np.float32, f"the expected fitness must be the float32 mean, got {expected.dtype}")
    cache = ExecutableCache()
    fitness_key = ("fitness", rows, EPISODES, "numpy", "cpu:0")
    fitness_executable = cache.get(fitness_key, lambda: build_fitness(POP_SIZE, episodes_per_policy=EPISODES))
    fitness = run_fitness(fitness_executable, returns)
    _require(fitness.shape == (POP_SIZE,), f"fitness must have shape ({POP_SIZE},), got {fitness.shape}")
    _require(fitness.dtype == np.float32, f"fitness must be float32, got {fitness.dtype}")
    _require(np.allclose(fitness, expected, atol=1e-6, rtol=0), f"fitness must be {expected}, got {fitness}")
    _require(
        cache.get(fitness_key, lambda: build_fitness(POP_SIZE, episodes_per_policy=EPISODES)) is fitness_executable,
        "the second fitness lookup must return the cached executable",
    )
    _require(
        (cache.misses, cache.hits) == (1, 1),
        f"the fitness cache must report one miss and one hit, got {cache.misses} and {cache.hits}",
    )

    algo, config, state, pop_arrays, species_leaves = built[GENOME_WIDTHS[0]]
    tell_cache = ExecutableCache()
    tell_key = ("tell", POP_SIZE, "numpy", "cpu:0")
    tell_executable = tell_cache.get(
        tell_key,
        lambda: build_tell(algo, config, pop_arrays, species_leaves, POP_SIZE),
    )
    _require(
        tell_cache.get(
            tell_key,
            lambda: build_tell(algo, config, pop_arrays, species_leaves, POP_SIZE),
        )
        is tell_executable,
        "the second tell lookup must return the cached executable",
    )
    _require(
        (tell_cache.misses, tell_cache.hits) == (1, 1),
        f"the tell cache must report one miss and one hit, got {tell_cache.misses} and {tell_cache.hits}",
    )
    tell = run_tell(
        tell_executable,
        randkey=state.randkey,
        pop_arrays=pop_arrays,
        generation=state.generation,
        fitness=np.array([0.25, 0.5, 0.75], dtype=np.float32),
        species_leaves=species_leaves,
    )
    _require(
        sorted(tell) == ["generation", "pop", "randkey", "species"],
        f"run_tell must return generation/pop/randkey/species, got {sorted(tell)}",
    )
    _require(len(tell["pop"]) == len(pop_arrays), f"run_tell must return the 4 payload leaves, got {len(tell['pop'])}")
    for leaf_index, (before, after) in enumerate(zip(pop_arrays, tell["pop"])):
        _require(
            np.asarray(after).shape == np.asarray(before).shape,
            f"payload leaf {leaf_index} must keep its shape {np.asarray(before).shape}, got {np.asarray(after).shape}",
        )
    _require(
        float(to_numpy(tell["generation"])) == float(to_numpy(state.generation)) + 1.0,
        f"run_tell must increment the generation, got {to_numpy(tell['generation'])!r}",
    )
    _require(
        int(to_numpy(tell["randkey"])) != int(to_numpy(state.randkey)),
        "run_tell must advance the randkey",
    )
    _require(
        isinstance(tell["species"], SpeciesState),
        f"run_tell must rebuild a SpeciesState, got {type(tell['species'])}",
    )
    _require(
        len(evoxc.tree_leaves(tell["species"])) == len(species_leaves),
        "the rebuilt species pytree must keep its leaf count",
    )

    for (num_inputs, num_outputs), (algo, config, _state, pop_arrays, _species) in built.items():
        width_rows = POP_SIZE * EPISODES
        policy = build_policy(
            algo,
            config,
            pop_arrays,
            width_rows,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            episodes=EPISODES,
        )
        obs = np.tile(np.linspace(-1.0, 1.0, num_inputs, dtype=np.float32), (width_rows, 1))
        actions = run_policy(policy, pop_arrays, obs)
        _require(
            actions.shape == (width_rows, num_outputs),
            f"the {num_inputs} -> {num_outputs} actions must be ({width_rows}, {num_outputs}), got {actions.shape}",
        )
        _require(actions.dtype == np.float32, f"the actions must be float32, got {actions.dtype}")
        for policy_index in range(POP_SIZE):
            block = actions[EPISODES * policy_index : EPISODES * (policy_index + 1)]
            _require(
                np.array_equal(block, np.broadcast_to(block[:1], block.shape)),
                f"the {EPISODES} rows of policy {policy_index} must agree exactly on identical observations "
                f"({num_inputs} -> {num_outputs}), got {block}",
            )
        first_rows = actions[::EPISODES]
        _require(
            len({row.tobytes() for row in first_rows}) > 1,
            f"distinct policies of the {num_inputs} -> {num_outputs} genome must not all act identically",
        )
        _expect_value_error(
            lambda: build_policy(
                algo,
                config,
                pop_arrays,
                width_rows,
                num_inputs=num_inputs + 1,
                num_outputs=num_outputs,
                episodes=EPISODES,
            ),
            "a num_inputs mismatch",
        )
        _expect_value_error(
            lambda: build_policy(
                algo,
                config,
                pop_arrays,
                width_rows,
                num_inputs=num_inputs,
                num_outputs=num_outputs + 1,
                episodes=EPISODES,
            ),
            "a num_outputs mismatch",
        )
        _expect_value_error(
            lambda: build_policy(
                algo,
                config,
                pop_arrays,
                width_rows + 1,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                episodes=EPISODES,
            ),
            "a rows mismatch",
        )


# --- 4. Trainer ---------------------------------------------------------------------


class FakeEnv:
    """A simulator-free :class:`env_base.BatchedEpisodeEnv` whose returns are an exact float32 formula.

    ``episode_returns[policy, episode]`` is ``((seed * 31 + generation * 17 + policy * 7 + episode) % 64) / 64``, so a rollout is deterministic in ``(generation, policy index, seed)`` alone and deliberately independent of the genome payload.
    The values are multiples of 1/64 below 1, which keeps the float32 and the float64 episode means within a rounding step of each other and keeps a 1e-7 fitness offset representable in float32.
    ``fitness_offset`` shifts the reported fitness, which is how the check drives the trainer's tolerance.
    """

    def __init__(self, pop_size=POP_SIZE, generations=GENERATIONS, master_seed=MASTER_SEED, fitness_offset=0.0):
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.master_seed = int(master_seed)
        self.fitness_offset = float(fitness_offset)
        self.episodes_per_policy = EPISODES
        self.max_episode_length = 5
        self.rows = rows_for(self.pop_size, EPISODES)
        self.cache = None
        self.closed = False
        self.seed_table = generation_seed_table(self.master_seed, self.generations, episodes=EPISODES)
        self.evaluated = []

    @staticmethod
    def _episode_return(generation, policy, seed, episode):
        """Return the deterministic float32 episode return of one row."""
        return np.float32(((int(seed) * 31 + int(generation) * 17 + int(policy) * 7 + int(episode)) % 64) / 64.0)

    def evaluate(self, pop_arrays, generation):
        """Run one fake generation: the seed row of ``generation``, a constant return formula."""
        generation = int(generation)
        seeds = self.seed_table[generation]
        start = time.perf_counter()
        episode_returns = self._episode_returns(generation, seeds)
        wall_time_s = max(time.perf_counter() - start, 1e-6)
        steps = self.rows * self.max_episode_length
        self.evaluated.append(generation)
        return {
            "seeds": seeds,
            "episode_returns": episode_returns,
            "returns": episode_returns.reshape(-1),
            "fitness": self._report_fitness(episode_returns),
            "steps": steps,
            "env_steps": steps,
            "wall_time_s": wall_time_s,
            "steps_per_s": steps / wall_time_s,
            "env_steps_per_s": steps / wall_time_s,
            "end_reasons": {"timeout": self.rows},
            "mean_episode_return": float(episode_returns.mean()),
        }

    def _episode_returns(self, generation, seeds):
        """Return the ``(pop_size, episodes)`` float32 episode returns of one generation."""
        return np.array(
            [
                [self._episode_return(generation, policy, seed, episode) for episode, seed in enumerate(seeds)]
                for policy in range(self.pop_size)
            ],
            dtype=np.float32,
        )

    def _report_fitness(self, episode_returns):
        """Return the reported fitness: the float32 episode mean plus ``fitness_offset``."""
        return episode_returns.mean(axis=1) + np.float32(self.fitness_offset)

    def close(self):
        """Record that the caller closed the env; the trainer must never do this."""
        self.closed = True


class LargeReturnEnv(FakeEnv):
    """A :class:`FakeEnv` at the Go2 pilot's episode-return magnitude, reporting the compiled fitness graph.

    ``episode_returns`` are float32 draws at the ~±100 per-episode magnitude of the Go2 pilot, two positive episodes and one negative per policy, deterministic in ``(master_seed, generation)`` and still independent of the genome payload.
    The reported fitness is what a task env produces: :func:`run_fitness` on ``episode_returns.reshape(-1)``, i.e. the float32 ``evoxc.reduce_mean`` of :func:`build_fitness`, plus ``fitness_offset``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fitness_executable = None

    def _episode_returns(self, generation, seeds):
        """Return the float32 returns of one generation at the Go2 magnitude, mixed signs."""
        rng = np.random.default_rng((self.master_seed, int(generation)))
        base = rng.uniform(0.5 * LARGE_RETURN_SCALE, LARGE_RETURN_SCALE - 1.0, size=(self.pop_size, 1))
        signs = np.ones((1, EPISODES), dtype=np.float32)
        signs[0, -1] = -1.0
        jitter = rng.uniform(-1.0, 1.0, size=(self.pop_size, EPISODES))
        return (base.astype(np.float32) * signs + jitter.astype(np.float32)).astype(np.float32)

    def _report_fitness(self, episode_returns):
        """Return the compiled fitness graph's answer plus ``fitness_offset``."""
        if self._fitness_executable is None:
            self._fitness_executable = build_fitness(self.pop_size, episodes_per_policy=EPISODES)
        graph_fitness = run_fitness(self._fitness_executable, episode_returns.reshape(-1))
        return graph_fitness + np.float32(self.fitness_offset)


def check_trainer():
    """Pin end-to-end determinism, the artifacts and the fitness tolerance of the trainer."""

    def train(env_kwargs=None, **trainer_kwargs):
        """Run one full trainer pass against a fresh fake env; returns ``(trainer, env)``."""
        env = FakeEnv(**(env_kwargs or {}))
        trainer = NeuroevolutionTrainer(
            env,
            POP_SIZE,
            GENERATIONS,
            num_inputs=6,
            num_outputs=2,
            max_nodes=20,
            max_conns=40,
            **trainer_kwargs,
        )
        trainer.run()
        return trainer, env

    trainer_a, env_a = train()
    trainer_b, env_b = train()
    _require(len(trainer_a.records) == GENERATIONS, f"the trainer must record {GENERATIONS} generations")
    _require(
        len(trainer_a.fitness_history) == GENERATIONS,
        f"the trainer must record {GENERATIONS} fitness values, got {len(trainer_a.fitness_history)}",
    )
    _require(
        trainer_a.fitness_history == trainer_b.fitness_history,
        f"two identical runs must share their fitness history, got {trainer_a.fitness_history} and "
        f"{trainer_b.fitness_history}",
    )
    _require(
        _strip_timings(trainer_a.records) == _strip_timings(trainer_b.records),
        "two identical runs must share their generation records (timings aside)",
    )
    _require(
        np.array_equal(trainer_a.seed_table, trainer_b.seed_table),
        "two identical runs must share their seed table",
    )
    for generation, record in enumerate(trainer_a.records):
        _require(
            record["seeds"] == trainer_a.seed_table[generation].tolist(),
            f"generation {generation} must run the seed row of the seed table, got {record['seeds']}",
        )
    _require(env_a.closed is False and env_b.closed is False, "the trainer must never close the env it was handed")
    trainer_a.close()
    _require(env_a.closed is False, "NeuroevolutionTrainer.close() must not close the caller's env")

    output_dir = Path(tempfile.mkdtemp(prefix="neuroevolution-selfcheck-"))
    try:
        train(output_dir=output_dir)
        history_lines = (output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        _require(
            len(history_lines) == GENERATIONS,
            f"history.jsonl must hold one line per generation, got {len(history_lines)}",
        )
        _require(
            all(isinstance(json.loads(line), dict) for line in history_lines),
            "every history.jsonl line must be a JSON object",
        )
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        for key in ("config", "generations", "totals"):
            _require(key in summary, f"summary.json must hold {key!r}, got {sorted(summary)}")
        _require(
            len(summary["generations"]) == GENERATIONS,
            f"the summary must hold {GENERATIONS} generations, got {len(summary['generations'])}",
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    _require(not output_dir.exists(), f"the temporary output dir {output_dir} must be removed")
    _require(trainer_a.history_path is None, "output_dir=None must keep history.jsonl in memory")
    _require(trainer_a.summary_path is None, "output_dir=None must keep summary.json in memory")

    message = _expect_value_error(
        lambda: train(env_kwargs={"fitness_offset": 1e-3}),
        "a fitness off by 1e-3",
    )
    _require("fitness" in message, f"the rejection message must name the fitness, got {message}")
    train(env_kwargs={"fitness_offset": 1e-7})

    other_seed, _ = train(master_seed=MASTER_SEED + 1)
    _require(
        not np.array_equal(other_seed.seed_table, trainer_a.seed_table),
        "a nonzero master seed must change the seed table",
    )


# --- 4b. Device trainer (plumbing-only device) --------------------------------------


def check_device_trainer():
    """Pin the device residency of the trainer on ``cpu:0``: determinism, placement and the fitness rule."""

    def train(residency, *, nan_episodes=(), fitness_offset=None, record_payloads=False):
        """Run one trainer pass against a fresh synthetic env; returns ``(trainer, env, recorder)``."""
        env = SyntheticDeviceEnv(
            DEVICE_POP_SIZE,
            residency=residency,
            backend=PLUMBING_BACKEND,
            device=PLUMBING_DEVICE,
            nan_episodes=nan_episodes,
        )
        wrapped = env if fitness_offset is None else _FitnessOffsetEnv(env, fitness_offset)
        recorder = None
        if record_payloads:
            recorder = _PayloadRecorder(wrapped)
            wrapped = recorder
        trainer = NeuroevolutionTrainer(
            wrapped,
            DEVICE_POP_SIZE,
            DEVICE_GENERATIONS,
            master_seed=MASTER_SEED,
            num_inputs=45,
            num_outputs=12,
            residency=residency,
        )
        trainer.run()
        return trainer, env, recorder

    trainer_a, env_a, recorder = train("device", record_payloads=True)
    trainer_b, env_b, _ = train("device")
    _require(
        len(trainer_a.records) == DEVICE_GENERATIONS,
        f"the device trainer must record {DEVICE_GENERATIONS} generations, got {len(trainer_a.records)}",
    )
    _require(
        trainer_a.fitness_history == trainer_b.fitness_history,
        f"two device runs of one master seed must share their fitness history, got "
        f"{trainer_a.fitness_history} and {trainer_b.fitness_history}",
    )
    _require(
        _strip_timings(trainer_a.records) == _strip_timings(trainer_b.records),
        "two device runs of one master seed must share their generation records (timings and cache aside)",
    )

    placement = trainer_a.placement
    target = PLUMBING_DEVICE
    for key in ("randkey", "generation", "fitness", "episode_returns", "returns", "seeds"):
        _require(
            placement.get(key) == target,
            f"the device residency must keep {key} on {target}, got {placement.get(key)!r}",
        )
    _require(
        len(placement.get("pop", ())) == 4 and set(placement["pop"]) == {target},
        f"the four payload leaves must live on {target}, got {placement.get('pop')!r}",
    )
    _require(
        len(placement.get("species", ())) == 10 and set(placement["species"]) == {target},
        f"the ten species leaves must live on {target}, got {placement.get('species')!r}",
    )

    _require(
        recorder is not None and len(recorder.payloads) == DEVICE_GENERATIONS,
        f"every generation must hand a payload to the env, got {0 if recorder is None else len(recorder.payloads)}",
    )
    for generation, payload in enumerate(recorder.payloads):
        _require(
            len(payload) == 4,
            f"generation {generation} must receive the four payload leaves, got {len(payload)}",
        )
        for index, leaf in enumerate(payload):
            _require(
                not isinstance(leaf, np.ndarray) and is_device_tensor(leaf),
                f"generation {generation} payload leaf {index} must be an evoxc tensor under device "
                f"residency, got {type(leaf).__name__}",
            )
            _require(
                device_placement(leaf) == target,
                f"generation {generation} payload leaf {index} must live on {target}, got {device_placement(leaf)}",
            )

    trainer_numpy, env_numpy, _ = train("numpy")
    _require(
        trainer_numpy.fitness_history == trainer_a.fitness_history,
        f"the numpy residency of the same env must reproduce the device fitness history, got "
        f"{trainer_numpy.fitness_history} and {trainer_a.fitness_history}",
    )
    _require(
        trainer_numpy.placement.get("randkey") == "host" and trainer_numpy.placement.get("fitness") == "host",
        f"the numpy residency must report host placement, got {trainer_numpy.placement}",
    )

    trainer_nan, env_nan, _ = train("device", nan_episodes=(0,))
    nan_record = trainer_nan.records[0]
    nan_episode_returns = np.asarray(nan_record["episode_returns"], dtype=np.float32)
    _require(
        bool(np.isnan(nan_episode_returns[:, 0]).all()),
        f"the unscored episode column must reach the record as NaN, got {nan_episode_returns[:, 0]}",
    )
    _require(
        np.array_equal(
            np.asarray(nan_record["fitness"], dtype=np.float32),
            _fitness_from_episode_returns(nan_episode_returns),
        ),
        f"a NaN episode must be counted as 0 on the device, got {nan_record['fitness']}",
    )

    message = _expect_value_error(
        lambda: train("device", fitness_offset=1e-3),
        "a device fitness off by 1e-3",
    )
    _require("fitness" in message, f"the rejection message must name the fitness, got {message}")

    for trainer, env in (
        (trainer_a, env_a),
        (trainer_b, env_b),
        (trainer_numpy, env_numpy),
        (trainer_nan, env_nan),
    ):
        trainer.close()
        # A closed SyntheticDeviceEnv refuses to hand out its executables, so this is the lifecycle check.
        _require(env.executables() is not None, "the trainer must never close the env it was handed")
        env.close()


# --- 4c. Device hardware ------------------------------------------------------------


def check_device_hardware():
    """Verify the device residency on the first reachable non-CPU device, or print ``SKIP``."""
    devices = evoxc.devices()
    available = [f"{entry.kind}:{entry.index}" for entry in devices]
    backend = os.environ.get("EVOXC_BACKEND") or "iree"
    device_spec = os.environ.get("EVOXC_DEVICE") or None
    plugin_path = ensure_pjrt_plugin(backend)
    if backend == "xla" and plugin_path is None:
        raise AssertionError(
            "EVOXC_BACKEND=xla needs its PJRT plugin visible: set EVOXC_PJRT_PLUGIN to the xla_rocm_plugin.so "
            "or put its jax_plugins/ directory on PYTHONPATH (the xla adapter's placement provider reads no "
            "compile options)"
        )
    usable, skipped = reachable_devices(devices, backend)
    if device_spec is not None:
        device = resolve_backend_device(backend, device_spec)[1]
    elif usable:
        device = usable[0]
    else:
        reasons = "; ".join(f"{entry['device']} unreachable ({entry['error']})" for entry in skipped)
        print(
            f"SKIP device_hardware: no non-CPU evoxc device is reachable through {backend} "
            f"(available: {available}{'; ' + reasons if reasons else ''})"
        )
        return
    if device.kind == "cpu":
        print(f"SKIP device_hardware: EVOXC_DEVICE={device_spec} names a cpu-kind device; the CPU checks cover it")
        return
    target = f"{device.kind}:{device.index}"
    backend, device = resolve_backend_device(backend, device)

    env = SyntheticDeviceEnv(
        HARDWARE_POP_SIZE,
        residency="device",
        backend=backend,
        device=device,
    )
    trainer = NeuroevolutionTrainer(
        env,
        HARDWARE_POP_SIZE,
        HARDWARE_GENERATIONS,
        master_seed=MASTER_SEED,
        num_inputs=45,
        num_outputs=12,
        residency="device",
        plugin_path=plugin_path,
    )
    trainer.setup()
    assert_payload_resident(
        trainer.pop_arrays,
        trainer.species_leaves,
        trainer.randkey,
        trainer.generation_counter,
        device,
    )
    _require(
        set(trainer.placement["pop"]) == {target} and set(trainer.placement["species"]) == {target},
        f"initialization must be device-resident on {target}, got {trainer.placement}",
    )
    _require(
        trainer.placement["randkey"] == target and trainer.placement["generation"] == target,
        f"the key and the generation counter must be on {target}, got {trainer.placement}",
    )

    trainer.run()
    placement = trainer.placement
    for key in ("fitness", "episode_returns", "returns", "seeds"):
        _require(
            placement.get(key) == target,
            f"the device result's {key} must be on {target}, got {placement.get(key)!r}",
        )
    assert_payload_resident(
        trainer.pop_arrays,
        trainer.species_leaves,
        trainer.randkey,
        trainer.generation_counter,
        device,
    )
    _require(
        set(placement["pop"]) == {target} and set(placement["species"]) == {target},
        f"the tell graph's output payload must stay on {target}, got {placement}",
    )

    reference_env = SyntheticDeviceEnv(HARDWARE_POP_SIZE, residency="numpy", backend="numpy", device="cpu:0")
    reference = NeuroevolutionTrainer(
        reference_env,
        HARDWARE_POP_SIZE,
        HARDWARE_GENERATIONS,
        master_seed=MASTER_SEED,
        num_inputs=45,
        num_outputs=12,
        residency="numpy",
    )
    reference.run()
    _require(
        np.allclose(
            np.asarray([float(value) for value in trainer.fitness_history], dtype=np.float64),
            np.asarray([float(value) for value in reference.fitness_history], dtype=np.float64),
            atol=FITNESS_TOLERANCE,
            rtol=FITNESS_TOLERANCE,
        ),
        f"the {target} residency must reproduce the numpy residency's fitness history within "
        f"atol=rtol={FITNESS_TOLERANCE:.1e}, got max |diff| "
        f"{float_max_abs_diff(trainer.fitness_history, reference.fitness_history):.3e}, max rel "
        f"{float_max_rel_diff(trainer.fitness_history, reference.fitness_history):.3e} between "
        f"{trainer.fitness_history} and {reference.fitness_history}",
    )
    _require(
        not np.isnan(np.asarray(trainer.fitness_history, dtype=np.float32)).any(),
        f"the {target} fitness history must be finite, got {trainer.fitness_history}",
    )
    print(
        f"  device_hardware: {backend}@{target} pop {HARDWARE_POP_SIZE}, "
        f"fitness {[round(value, 6) for value in trainer.fitness_history]} "
        f"(numpy parity max |diff| {float_max_abs_diff(trainer.fitness_history, reference.fitness_history):.1e}, "
        f"max rel {float_max_rel_diff(trainer.fitness_history, reference.fitness_history):.1e} "
        f"within atol=rtol={FITNESS_TOLERANCE:.1e})"
    )
    trainer.close()
    env.close()
    reference.close()
    reference_env.close()


# --- 5. Large-return aggregation ----------------------------------------------------


def check_large_returns():
    """Pin the float32 aggregation rule at the Go2 pilot's episode-return magnitude (~±100)."""

    def train(env_kwargs=None):
        """Run one full trainer pass against a fresh large-return env; returns ``(trainer, env)``."""
        env = LargeReturnEnv(**(env_kwargs or {}))
        trainer = NeuroevolutionTrainer(
            env,
            POP_SIZE,
            LARGE_RETURN_GENERATIONS,
            num_inputs=6,
            num_outputs=2,
            max_nodes=20,
            max_conns=40,
        )
        trainer.run()
        return trainer, env

    trainer, env = train()
    _require(
        len(trainer.records) == LARGE_RETURN_GENERATIONS,
        f"the trainer must record {LARGE_RETURN_GENERATIONS} generations, got {len(trainer.records)}",
    )
    fitness_executable = build_fitness(POP_SIZE, episodes_per_policy=EPISODES)
    worst_float64_gap = 0.0
    for generation, record in enumerate(trainer.records):
        episode_returns = np.asarray(record["episode_returns"], dtype=np.float32)
        magnitude = float(np.max(np.abs(episode_returns)))
        _require(
            episode_returns.dtype == np.float32 and 50.0 < magnitude <= LARGE_RETURN_SCALE,
            f"generation {generation} must run float32 returns at the Go2 magnitude (|return| <= "
            f"{LARGE_RETURN_SCALE}), got {episode_returns.dtype} up to {magnitude}",
        )
        graph_fitness = run_fitness(fitness_executable, episode_returns.reshape(-1))
        _require(
            np.array_equal(np.asarray(record["fitness"], dtype=np.float32), graph_fitness),
            f"generation {generation}: the trainer's fitness {record['fitness']} must equal the compiled "
            f"fitness graph's answer {graph_fitness.tolist()} exactly",
        )
        float64_mean = episode_returns.mean(axis=1, dtype=np.float64).astype(np.float32)
        worst_float64_gap = max(worst_float64_gap, float(np.max(np.abs(float64_mean - graph_fitness))))
    _require(
        worst_float64_gap > 1e-6,
        f"the float64 mean must differ from the compiled graph by more than the 1e-6 tolerance, got "
        f"{worst_float64_gap}; otherwise this check would not exercise the aggregation defect",
    )
    trainer.close()
    _require(env.closed is False, "NeuroevolutionTrainer.close() must not close the caller's env")

    message = _expect_value_error(
        lambda: train(env_kwargs={"fitness_offset": 1e-3}),
        "a graph fitness off by 1e-3 at the Go2 magnitude",
    )
    _require("fitness" in message, f"the rejection message must name the fitness, got {message}")


# --- 6. Import hygiene --------------------------------------------------------------


def check_import_hygiene():
    """Assert the sibling modules import no simulator/torch/task module and none is loaded."""
    root = Path(__file__).parent
    for filename in HYGIENE_FILES:
        path = root / filename
        _require(path.is_file(), f"{filename} must sit next to selfcheck.py, got {path}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for dotted in _imported_modules(tree):
            parts = dotted.split(".")
            _require(parts[0] not in FORBIDDEN_PACKAGES, f"{filename} must not import {dotted}")
            _require(
                not any(part.startswith(FORBIDDEN_MODULE_PREFIXES) for part in parts),
                f"{filename} must not import the task module {dotted}",
            )
    for name in FORBIDDEN_PACKAGES:
        _require(name not in sys.modules, f"{name} must not be imported by the neuroevolution layer")


# --- Entry point --------------------------------------------------------------------

#: The checks, in the order ``main`` runs them, with the name printed on their PASS/FAIL line.
CHECKS = (
    ("seeding", check_seeding),
    ("env_base", check_env_base),
    ("device_contract", check_device_contract),
    ("graphs", check_graphs),
    ("trainer", check_trainer),
    ("device_trainer", check_device_trainer),
    ("large_returns", check_large_returns),
    ("import_hygiene", check_import_hygiene),
    ("device_hardware", check_device_hardware),
)


def main():
    """Run every check in order; return 0 when all pass and 1 on the first failure."""
    for name, check in CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - a check runner must report, not abort
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            return 1
        print(f"PASS {name}")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
