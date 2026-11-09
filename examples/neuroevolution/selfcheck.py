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

``large_returns``
    The aggregation rule at the Go2 pilot's episode-return magnitude (~±100): an env whose float32 returns sit there and whose fitness is the real ``evolution_graphs`` fitness graph must be accepted, and the trainer's fitness must equal that graph bit for bit while a float64 mean of the same returns differs by more than the tolerance; the same env with the fitness off by 1e-3 must still be rejected, so the check is not vacuous.

``import_hygiene``
    An AST scan proving the six sibling modules import no ``torch`` / ``genesis`` / ``tensordict`` and no ``go2_*`` / ``hover_*`` task module, and that those packages are absent from ``sys.modules``.

Genome budgets are the smallest the dense initial network of a width allows, because ``DefaultGenome`` rejects a budget smaller than ``num_inputs + num_outputs`` nodes and ``num_inputs * num_outputs`` connections.
"""

from __future__ import annotations

import ast
import json
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

from env_base import BatchedEpisodeEnv, rows_for, validate_evaluation
from evolution_graphs import ExecutableCache, build_fitness, build_tell, run_fitness, run_tell
from policy_graphs import build_policy, run_policy
from seeding import (
    EPISODES_PER_POLICY,
    Perturbation,
    evolution_seed,
    generation_seed_table,
    perturbation_from_seed,
)
from trainer import NeuroevolutionTrainer, TraceOnlyProblem, pop_arrays_of, species_leaves_of

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
#: The sibling modules the import-hygiene check parses.
HYGIENE_FILES = (
    "seeding.py",
    "env_base.py",
    "evox_runtime.py",
    "evolution_graphs.py",
    "policy_graphs.py",
    "trainer.py",
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
    """Return ``value`` without the wall-clock-derived fields, recursively."""
    if isinstance(value, dict):
        return {key: _strip_timings(item) for key, item in value.items() if key not in TIMING_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_strip_timings(item) for item in value]
    return value


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

    It never runs anything; the check only asks whether the runtime-checkable protocol accepts it.
    """

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
    ("graphs", check_graphs),
    ("trainer", check_trainer),
    ("large_returns", check_large_returns),
    ("import_hygiene", check_import_hygiene),
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
