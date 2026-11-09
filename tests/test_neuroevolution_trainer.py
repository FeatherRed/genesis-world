"""Behavioural tests for the shared, task-agnostic neuroevolution trainer contract.

``examples/neuroevolution`` is the task-agnostic half of the neuroevolution stack: the
seed rules (``seeding``), the batched-episode env contract (``env_base``), the
width-generic policy graph (``policy_graphs``) and the trainer itself (``trainer``).
These tests pin that shared contract directly -- no ``genesis`` scene, no simulator, no
network, no display and no training run -- so every task that plugs into it inherits
the same guarantees:

* ``env_base.validate_evaluation`` accepts one well-formed generation and rejects each
  single-rule violation (shape, dtype, duplicate seeds, a missing key, a disagreeing
  fitness, a ``returns`` vector that is not the policy-major flattening);
* the documented seed constants hold bit for bit, the seed table is prefix-stable, and
  the evolution and environment streams of one master seed are independent;
* the policy graph is width-generic (both real widths, 45 -> 12 and 17 -> 4), lays its
  rows out policy-major, is deterministic, and the Go2-specific adapter delegates to it
  unchanged;
* ``NeuroevolutionTrainer`` is reproducible from the master seed alone, never closes the
  env it was handed, draws its seed table from ``seeding``, and enforces the single
  float32 episode-mean aggregation rule to ``1e-6``.

``evoxc`` and ``tensorneat`` are not declared dependencies of ``genesis-world``, so the
whole module is skipped where they are not installed.

Run it from the repository root::

    PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \
        tests/test_neuroevolution_trainer.py -q -n 0

Everything runs on the numpy backend with the smallest genome budgets a dense initial
network admits, so the module stays far below a minute on CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

pytestmark = [pytest.mark.required]

#: ``examples/`` is a set of plain scripts rather than a package, so both the shared
#: neuroevolution layer and the Go2 kernels import their siblings by bare module name.
EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
NEUROEVOLUTION_DIR = EXAMPLES_DIR / "neuroevolution"
LOCOMOTION_DIR = EXAMPLES_DIR / "locomotion"
for _directory in (NEUROEVOLUTION_DIR, LOCOMOTION_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from tensorneat import Pipeline, algorithm, genome

import go2_neat_kernels
from env_base import REQUIRED_KEYS, rows_for, validate_evaluation
from policy_graphs import build_policy, run_policy
from seeding import EPISODES_PER_POLICY, evolution_seed, generation_seed_table
from trainer import NeuroevolutionTrainer, TraceOnlyProblem, pop_arrays_of

#: Policies per generation in every check: small enough that the numpy backend stays cheap.
POP_SIZE = 3
#: Episodes per policy, the documented triple every generation shares.
EPISODES = EPISODES_PER_POLICY
#: Rows one generation simulates, policy-major.
ROWS = rows_for(POP_SIZE, EPISODES)
#: Generations the trainer checks run.
GENERATIONS = 3
#: The master seed the reproducibility guarantee is documented for.
MASTER_SEED = 0
#: ``(num_inputs, num_outputs)`` of the two real tasks: Go2 walking (45 -> 12) and drone hover (17 -> 4).
GENOME_WIDTHS = ((45, 12), (17, 4))

#: The fitness tolerance ``env_base.validate_evaluation`` and ``trainer._aggregate`` hard-code,
#: absolute and relative-free.
FITNESS_ATOL = 1e-6

#: A well-formed generation of episode returns: distinct float32 values, none of them non-finite.
EPISODE_RETURNS = np.array(
    [
        [0.5, 0.25, 0.75],
        [0.125, 0.0, 0.375],
        [1.0, 0.5, 0.25],
    ],
    dtype=np.float32,
)
#: The generation's ordered, pairwise-distinct episode seeds, int64 like the seed table.
SEEDS = np.array([1454269175, 757617120, 1421746521], dtype=np.int64)

#: Record and summary fields derived from wall-clock time, stripped before two runs are compared.
TIMING_FIELDS = (
    "wall_time_s",
    "steps_per_s",
    "env_steps_per_s",
    "executable_build_time_s",
    "tell_build_time_s",
)


# --- Helpers ------------------------------------------------------------------------


def _fitness_rule(episode_returns):
    """Return the fitness the contract asks for: the float32 episode mean with NaN counted as 0.

    The expression is the one ``env_base.validate_evaluation`` recomputes, so a result handed
    back by this helper is well formed by construction.
    """
    return np.nan_to_num(episode_returns, nan=0.0).mean(axis=1)


def _evaluation(episode_returns=None, **overrides):
    """Return a well-formed ``evaluate`` result for one generation; ``overrides`` replaces fields."""
    episode_returns = np.array(EPISODE_RETURNS if episode_returns is None else episode_returns, dtype=np.float32)
    result = {
        "seeds": SEEDS.copy(),
        "episode_returns": episode_returns,
        "returns": episode_returns.reshape(-1),
        "fitness": _fitness_rule(episode_returns),
        "steps": ROWS * 5,
        "env_steps": ROWS * 5,
        "wall_time_s": 0.25,
        "end_reasons": {"timeout": ROWS},
    }
    result.update(overrides)
    return result


def _strip_timings(value):
    """Return ``value`` without the wall-clock-derived fields, recursively."""
    if isinstance(value, dict):
        return {key: _strip_timings(item) for key, item in value.items() if key not in TIMING_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_strip_timings(item) for item in value]
    return value


def _trace_width(num_inputs, num_outputs, seed=0):
    """Trace one genome width exactly as the trainer does; returns ``(algo, config, pop_arrays)``.

    ``DefaultGenome`` must be able to hold the dense initial network, i.e. ``num_inputs +
    num_outputs`` nodes and ``num_inputs * num_outputs`` connections, so that is the budget:
    the nominal trainer budget (20/40) cannot express the real 45 -> 12 width (57 nodes, 540
    connections).  ``config`` is the tensor-free ``Pipeline._static_config`` every compiled
    graph is traced with, the same private extractor the trainer uses.
    """
    generator = genome.DefaultGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=num_inputs + num_outputs,
        max_conns=num_inputs * num_outputs,
    )
    algo = algorithm.NEAT(generator, pop_size=POP_SIZE, species_size=min(3, POP_SIZE))
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(num_inputs, num_outputs),
        seed=seed,
        backend="numpy",
        is_detailed=False,
    )
    state = pipeline.setup()
    return algo, pipeline._static_config(state), pop_arrays_of(state)


class FakeEnv:
    """A simulator-free ``BatchedEpisodeEnv`` whose returns are an exact float32 formula.

    ``episode_returns[policy, episode]`` is ``((seed * 31 + generation * 17 + policy * 7 +
    episode) % 64) / 64``, so a rollout is deterministic in ``(generation, policy index,
    seed)`` alone and deliberately independent of the genome payload -- which makes the
    trainer's reproducibility observable and nothing else.
    The values are multiples of 1/64 below 1, which keeps the float32 and float64 episode
    means within a rounding step of each other and keeps a 1e-7 fitness offset representable
    in float32, so the tolerance check is not vacuous.
    ``fitness_offset`` shifts the reported fitness, which is how the tolerance is driven.
    ``close`` only records the call: the trainer must never make it.
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

    def evaluate(self, pop_arrays, generation):
        """Run one fake generation: the seed row of ``generation``, a constant return formula."""
        generation = int(generation)
        seeds = self.seed_table[generation]
        start = time.perf_counter()
        episode_returns = self._episode_returns(generation, seeds)
        # A genuinely measured wall time, never a sleep: it is what ``_strip_timings`` exists for.
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

    def close(self):
        """Record that the caller closed the env; the trainer must never do this."""
        self.closed = True


def _train(env_kwargs=None, **trainer_kwargs):
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


# --- 1. The evaluate result contract ------------------------------------------------


def test_validate_evaluation_accepts_valid_and_rejects_bad_results():
    """Pin the ``evaluate`` result contract: one valid result, every single-rule violation rejected.

    Tolerances are the contract's own: ``1e-6`` absolute on both the fitness rule and the
    ``returns``/``episode_returns`` agreement, and relative-free, so a ``1e-7`` fitness
    offset is inside the tolerance while ``1e-3`` is not.
    """
    result = _evaluation()
    assert result["returns"].shape == (ROWS,)
    assert result["episode_returns"].shape == (POP_SIZE, EPISODES)
    assert result["fitness"].shape == (POP_SIZE,)
    assert validate_evaluation(result, POP_SIZE) is None

    # Every malformed variant below breaks exactly one rule of the contract.
    cases = {
        "returns length": dict(result, returns=np.asarray(result["returns"])[:-1].copy()),
        "episode_returns shape": dict(result, episode_returns=EPISODE_RETURNS.reshape(ROWS, 1)),
        "seeds shape": dict(result, seeds=SEEDS[:-1].copy()),
        "duplicate seeds": dict(result, seeds=np.array([SEEDS[0], SEEDS[0], SEEDS[2]], dtype=np.int64)),
        "fitness off by 1e-3": dict(result, fitness=result["fitness"] + np.float32(1e-3)),
        "returns not the policy-major flattening": dict(result, returns=np.asarray(result["returns"])[::-1].copy()),
    }
    for label, case in cases.items():
        with pytest.raises(ValueError):
            validate_evaluation(case, POP_SIZE)

    # A missing required key is rejected too, and the message names the key.
    for key in REQUIRED_KEYS:
        without_key = {name: value for name, value in result.items() if name != key}
        with pytest.raises(ValueError) as missing:
            validate_evaluation(without_key, POP_SIZE)
        assert key in str(missing.value)

    # 1e-7 is inside the 1e-6 tolerance, and the offset survives float32 so the check is real.
    tiny_offset = result["fitness"] + np.float32(1e-7)
    assert not np.array_equal(tiny_offset, result["fitness"])
    assert validate_evaluation(_evaluation(fitness=tiny_offset), POP_SIZE) is None

    # A non-finite episode is allowed: NaN is what an unscored episode reports, and the rule counts it as 0.
    nan_returns = np.array(EPISODE_RETURNS, dtype=np.float32)
    nan_returns[1, 1] = np.nan
    nan_result = _evaluation(nan_returns)
    assert np.isfinite(nan_result["fitness"]).all()
    assert nan_result["fitness"][1] == np.nan_to_num(nan_returns[1], nan=0.0).mean()
    assert validate_evaluation(nan_result, POP_SIZE) is None

    # +inf is accepted as well; the validator's rule substitutes only NaN, so inf saturates to the
    # float32 maximum instead of 0 (the trainer's own aggregation maps +-inf to 0 -- see test 5).
    inf_returns = np.array(EPISODE_RETURNS, dtype=np.float32)
    inf_returns[1, 1] = np.inf
    inf_result = _evaluation(inf_returns)
    assert np.isfinite(inf_result["fitness"]).all()
    assert inf_result["fitness"][1] > np.float32(1e37)
    assert validate_evaluation(inf_result, POP_SIZE) is None


# --- 2. The seed rules --------------------------------------------------------------


def test_seeding_reproduces_the_documented_values():
    """Pin the documented seed constants, the table layout and the two-stream independence."""
    assert EPISODES_PER_POLICY == 3

    # The constants quoted in the ``seeding`` docstring, bit for bit.
    assert evolution_seed(0) == 3757552657
    documented = [
        [1454269175, 757617120, 1421746521],
        [1313752747, 908599966, 53557616],
        [1683657442, 1375977993, 1654802006],
    ]
    table = generation_seed_table(0, 3)
    assert table.tolist() == documented
    assert table.dtype == np.int64
    assert table.shape == (3, 3)

    # Every row is pairwise distinct and a full 31-bit draw, and no two rows repeat.
    long_table = generation_seed_table(0, 8)
    for generation, row in enumerate(long_table):
        assert len(set(row.tolist())) == EPISODES, f"generation {generation} repeats a seed: {row.tolist()}"
    assert np.all((long_table >= 0) & (long_table < 2**31))
    assert np.unique(long_table, axis=0).shape[0] == long_table.shape[0]

    # Prefix stability: asking for more generations never changes a row already materialized.
    for master_seed in (0, 1, 12345):
        full = generation_seed_table(master_seed, 8)
        for count in (1, 2, 5, 8):
            assert np.array_equal(full[:count], generation_seed_table(master_seed, count))

    # The master seed actually selects the stream.
    assert not np.array_equal(generation_seed_table(0, 3), generation_seed_table(1, 3))
    assert evolution_seed(0) != evolution_seed(1)

    # The evolution stream is independent of the env stream: it never appears in a large table ...
    for master_seed in (0, 1, 7):
        env_seeds = set(generation_seed_table(master_seed, 200).ravel().tolist())
        assert evolution_seed(master_seed) not in env_seeds
        # ... and materializing many env tables cannot move it, because it draws from its own stream.
        before = evolution_seed(master_seed)
        for _ in range(5):
            generation_seed_table(master_seed, 50)
        assert evolution_seed(master_seed) == before


# --- 3. The width-generic policy graph ----------------------------------------------


def test_policy_graph_is_width_generic_and_identical_at_both_widths():
    """Pin the shared policy graph: one implementation, both real widths, policy-major and deterministic.

    The agreement with the Go2 adapter is exact (max abs difference ``0.0``), because the
    adapter only binds the task widths around the shared builder; anything above zero would
    mean the adapter has grown a graph of its own.
    """
    rows = POP_SIZE * EPISODES
    for num_inputs, num_outputs in GENOME_WIDTHS:
        algo, config, pop_arrays = _trace_width(num_inputs, num_outputs)
        # One observation repeated on every row: the rows of one policy must then agree bit for bit.
        obs = np.tile(np.linspace(-1.0, 1.0, num_inputs, dtype=np.float32), (rows, 1))
        executable = build_policy(
            algo,
            config,
            pop_arrays,
            rows,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            episodes=EPISODES,
        )
        actions = run_policy(executable, pop_arrays, obs)
        assert actions.shape == (rows, num_outputs)
        assert actions.dtype == np.float32

        # Policy-major tiling: rows episodes*p .. episodes*p+episodes-1 are one policy's episodes.
        for policy in range(POP_SIZE):
            block = actions[EPISODES * policy : EPISODES * (policy + 1)]
            assert np.array_equal(block, np.broadcast_to(block[:1], block.shape)), (
                f"the {EPISODES} rows of policy {policy} must agree exactly on identical observations "
                f"({num_inputs} -> {num_outputs})"
            )
        # Distinct policies of one population, so the graph really reads the genome payload.
        assert len({row.tobytes() for row in actions[::EPISODES]}) > 1

        # Determinism: the same genome and the same observation tile reproduce the actions bit for bit.
        assert np.array_equal(actions, run_policy(executable, pop_arrays, obs))

        # At the Go2 width only, the task adapter must be a pure delegation to this shared graph.
        if (num_inputs, num_outputs) == GENOME_WIDTHS[0]:
            adapter = getattr(go2_neat_kernels, "build_policy", None)
            if adapter is not None:
                delegated = run_policy(adapter(algo, config, pop_arrays, rows), pop_arrays, obs)
                assert np.array_equal(actions, delegated)
                assert float(np.abs(actions - delegated).max()) == 0.0
            # else: the adapter was removed, so there is nothing left to delegate-check.


# --- 4. Trainer determinism ---------------------------------------------------------


def test_trainer_is_deterministic_for_the_same_master_seed_and_differs_otherwise():
    """Pin reproducibility from the master seed alone, the seed table and the env's ownership.

    Two runs of the same configuration must agree on the fitness history and on every record
    once the wall-clock-derived fields are stripped; a different master seed changes the seed
    table and therefore the records.
    """
    first, first_env = _train()
    second, second_env = _train()
    assert len(first.records) == GENERATIONS
    assert len(first.fitness_history) == GENERATIONS
    assert len(second.records) == GENERATIONS
    assert first.fitness_history == second.fitness_history
    assert _strip_timings(first.records) == _strip_timings(second.records)
    assert np.array_equal(first.seed_table, second.seed_table)

    # The trainer draws its episode seeds from the documented env stream, not from its own RNG.
    assert np.array_equal(first.seed_table, generation_seed_table(MASTER_SEED, GENERATIONS, episodes=EPISODES))
    for generation, record in enumerate(first.records):
        assert record["seeds"] == first.seed_table[generation].tolist()

    # output_dir=None keeps the whole run in memory: no history file, no summary file.
    assert first.history_path is None
    assert first.summary_path is None

    # The env belongs to the caller: the trainer never closes it, not even in close().
    assert first_env.closed is False
    assert second_env.closed is False
    first.close()
    assert first_env.closed is False

    # A different master seed changes the seed table and, through it, the records.
    # The task env derives its own table from the same master seed (it must, per ``seeding``),
    # so the seed is moved on both sides.
    other, _ = _train(env_kwargs={"master_seed": MASTER_SEED + 1}, master_seed=MASTER_SEED + 1)
    assert not np.array_equal(other.seed_table, first.seed_table)
    assert other.records[0]["seeds"] != first.records[0]["seeds"]
    assert _strip_timings(other.records) != _strip_timings(first.records)


# --- 5. The fitness aggregation rule ------------------------------------------------


def test_trainer_enforces_the_float32_episode_mean_fitness():
    """Pin the single-owner aggregation rule: the float32 episode mean, agreed to 1e-6.

    The rule is stated in float32 on purpose.  At the Go2 pilot's ~+-100 per-episode
    magnitude, the float64 mean of the very same float32 returns differs from the float32
    mean by ~1.27e-6 (measured below), i.e. beyond the 1e-6 tolerance, so a task that
    aggregated in float64 would be rejected as a second, different fitness.
    """
    with pytest.raises(ValueError) as rejected:
        _train(env_kwargs={"fitness_offset": 1e-3})
    assert "fitness" in str(rejected.value)

    # A 1e-7 offset is inside the tolerance and must be accepted.
    trainer, _ = _train(env_kwargs={"fitness_offset": 1e-7})

    # The record's fitness is the trainer's own aggregation of episode_returns.  This env reported
    # that mean plus the 1e-7 offset, so the exact equality proves the trainer recomputed the rule
    # instead of echoing the reported vector (and, with the offset still representable in float32,
    # that it is the float32 mean).
    episode_returns = FakeEnv()._episode_returns(0, trainer.seed_table[0])
    assert trainer.records[0]["fitness"] == episode_returns.mean(axis=1).tolist()

    # The float64-versus-float32 episode mean at Go2 magnitude, the reason the rule is float32.
    large_returns = np.array([[100.5, -99.5, 100.25], [-100.75, 99.75, -100.125]], dtype=np.float32)
    float32_mean = large_returns.mean(axis=1, dtype=np.float32)
    float64_mean = large_returns.mean(axis=1, dtype=np.float64)
    assert float(np.abs(float32_mean - float64_mean).max()) > FITNESS_ATOL
    assert not np.allclose(float32_mean, float64_mean, atol=FITNESS_ATOL, rtol=0)
