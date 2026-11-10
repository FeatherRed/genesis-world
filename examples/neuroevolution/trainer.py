"""The task-agnostic neuroevolution trainer of ``examples/neuroevolution``.

A trainer needs exactly two things from a task: the genome widths (``num_inputs`` / ``num_outputs``) and an object implementing :class:`env_base.BatchedEpisodeEnv`.
Everything else -- the initial population, the NEAT ``tell`` transition, the fitness aggregation rule, the seed streams, the per-generation records and the run artifacts -- is owned here and is identical for every task.

The loop, per generation::

    result = env.evaluate(pop_arrays, generation)          # the task simulates the rows
    fitness = _fitness_from_episode_returns(result["episode_returns"])  # env_base owns the rule
    tell = run_tell(randkey, pop, generation, fitness, species)

Two residencies
---------------
``residency="numpy"`` is the reference path: :meth:`BatchedEpisodeEnv.evaluate` hands back host arrays, the trainer aggregates them with :func:`env_base._fitness_from_episode_returns`, and ``run_tell`` stages every operand and materializes every result.
``residency="device"`` keeps the four genome payload leaves, the ``SpeciesState`` leaves, the RNG key and the generation counter on ``env.device``: the trainer initializes them with :func:`device_runtime.device_setup`, hands the device payload straight to ``env.evaluate``, produces the fitness with the device transcription of the rule (:func:`evolution_graphs.run_fitness_rule`), verifies the env's result in-graph (:func:`evolution_graphs.run_verification`) and feeds the tell graph without staging anything.
Only the per-generation metrics a record needs (``seeds``, ``episode_returns``, ``returns``, ``fitness``) are copied to the host, after the tell graph consumed the device tensors.
A cpu-kind device is accepted as the plumbing-only mode of the device residency: placement, tiling and staging are exercised, no accelerator is used.
Device residency is single-backend -- an ``IreeDevicePayload`` cannot enter an xla executable and vice versa, and no device-to-device transfer exists -- so an explicit ``tell_backend``/``tell_device`` must resolve to the env's own backend and device.

Only ``numpy``, ``evoxc``, ``TensorNEAT`` and the sibling modules of this directory are imported: no ``torch``, no ``genesis``, no ``tensordict``, so the trainer is testable without a simulator.
The environment belongs to the caller: :meth:`NeuroevolutionTrainer.close` never closes it.

Seeds come from one master seed in two independent streams (see :mod:`seeding`): the evolution stream seeds the TensorNEAT pipeline, and the env stream materializes the per-generation episode seed table.
The trainer therefore adds no RNG of its own, so two runs of the same configuration and the same environment are identical.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

import evoxc

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common.tools import to_numpy

from env_base import (
    DEVICE_KEYS,
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    RESIDENCIES,
    _fitness_from_episode_returns,
    rows_for,
    validate_evaluation,
    validate_evaluation_device,
)
from device_runtime import (
    assert_device_resident,
    assert_payload_resident,
    device_payload,
    device_placement,
    device_setup,
    device_species_leaves,
    placement_map,
)
from evolution_graphs import (
    ExecutableCache,
    build_fitness_rule,
    build_tell,
    build_verification,
    run_fitness_rule,
    run_tell,
    run_verification,
)
from evox_runtime import _device_name, _to_numpy, resolve_backend_device
from seeding import EPISODES_PER_POLICY, evolution_seed, generation_seed_table

__all__ = [
    "NeuroevolutionTrainer",
    "TraceOnlyProblem",
    "pop_arrays_of",
    "species_leaves_of",
]

#: Environment attributes copied verbatim into the summary ``config`` when the env exposes them.
ENV_CONFIG_FIELDS = ("max_episode_length",)

#: Fitness tolerance of the in-graph device verification (the numpy path keeps ``1e-6``).
DEVICE_VERIFICATION_ATOL = 1e-5


class TraceOnlyProblem(problem.BaseProblem):
    """Stand-in problem for a pipeline that only supplies the genome statics.

    ``Pipeline.setup()`` needs a problem object, but the fitness always comes from the simulator here, never from TensorNEAT.
    ``input_shape`` / ``output_shape`` are the per-sample genome widths and are read-only properties of ``BaseProblem``, so this class exposes the constructor values through overridden properties.

    Args:
        num_inputs (int): Observation width of the task's genome.
        num_outputs (int): Action width of the task's genome.
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self._input_shape = (int(num_inputs),)
        self._output_shape = (int(num_outputs),)

    @property
    def input_shape(self):
        """``(num_inputs,)``: the per-sample observation shape."""
        return self._input_shape

    @property
    def output_shape(self):
        """``(num_outputs,)``: the per-sample action shape."""
        return self._output_shape

    def evaluate(self, state, keys, act_fn, transformed):
        """Raise: the trainer never asks TensorNEAT for a fitness."""
        raise NotImplementedError(
            "NeuroevolutionTrainer only traces the genome pipeline, it never evaluates a problem"
        )


class NeuroevolutionTrainer:
    """Runs the NEAT ``ask``/``tell`` loop against a batched-episode environment.

    The trainer owns the genome pipeline, the evolution step, the fitness aggregation rule, the records and the artifacts; the task owns the rollout.
    Per generation it calls ``env.evaluate(pop_arrays, generation)``, validates that result (``env_base.validate_evaluation`` for the numpy residency, ``env_base.validate_evaluation_device`` for the device residency), aggregates the fitness itself, and feeds the result into the compiled ``tell`` graph.

    Seeds: the TensorNEAT pipeline is seeded with :func:`seeding.evolution_seed` and the env receives the episode seeds of :func:`seeding.generation_seed_table`, so the two streams are independent and a run is reproducible from ``master_seed`` alone.

    Residency: ``self.residency`` inherits ``env.residency`` (``"numpy"`` when the env declares none) and decides both the payload form and the result contract.
    Under ``"device"``, ``evaluate`` receives the four :data:`env_base.DEVICE_KEYS`' payload leaves as tensors on ``env.device`` and must return ``episode_returns`` ``(pop_size, episodes)``, ``returns`` ``(rows,)``, ``fitness`` ``(pop_size,)`` and ``seeds`` ``(episodes,)`` as tensors on that same device, with only the logging values (``steps``, ``env_steps``, ``wall_time_s`` and friends) on the host.
    The trainer then computes the fitness with the device transcription of the single aggregation rule and verifies the env's ``returns`` tiling and reported ``fitness`` in-graph, so neither is recomputed on the host.

    The environment belongs to the caller: this class never closes it, and :meth:`close` only drops a cache the trainer created itself.

    Args:
        env (BatchedEpisodeEnv): The task's batched-episode environment; it must expose ``rows``, ``episodes_per_policy``, ``max_episode_length``, ``cache``, ``evaluate`` and ``close``, plus ``residency``/``backend``/``device`` for the device residency.
        pop_size (int): Policies per generation, hence ``pop_size * episodes_per_policy`` simulated rows.
        generations (int): Generations to run.
        master_seed (int): Seed of both RNG streams; see :mod:`seeding`.
        num_inputs (int): Observation width of the task's genome (45 for Go2, 17 for the drone).
        num_outputs (int): Action width of the task's genome (12 for Go2, 4 for the drone).
        max_nodes (int): Genome node budget.
        max_conns (int): Genome connection budget.
        species_size (int): NEAT species size, clamped to ``pop_size``.
        residency (str | None): ``"numpy"`` or ``"device"``; defaults to ``env.residency`` (``"numpy"`` when the env declares none), and ``"device"`` requires the env to declare that residency too.
        tell_backend (str | None): Evolution backend; defaults to the env backend, except ROCm/HIP where it defaults to ``"xla"`` because IREE/ROCm tell compilation does not terminate -- under the device residency it is the env backend and an explicit different value raises.
        tell_device (str | evoxc.Device | None): Evolution device; defaults to ``cpu:0`` on the numpy backend, otherwise the env device -- under the device residency it is the env device and an explicit different value raises.
        plugin_path (str | os.PathLike | None): XLA/PJRT plugin, forwarded to :func:`evolution_graphs.build_tell` (numpy residency) or to ``device_runtime.device_setup``'s compile options (device residency, ``xla`` only).
        output_dir (str | os.PathLike | None): Artifact directory; ``None`` keeps the whole run in memory and writes no file.
        logger (logging.Logger | None): Logger of the run; defaults to this module's logger.
        problem (problem.BaseProblem | None): Pipeline problem; defaults to :class:`TraceOnlyProblem` with the given genome widths.
        seed_table (array-like | None): ``(generations, episodes_per_policy)`` episode-seed table; when given it replaces the table :func:`seeding.generation_seed_table` would draw.

    Attributes:
        residency (str): ``"numpy"`` or ``"device"``; the payload form and result contract of every generation.
        cache (ExecutableCache): ``env.cache`` when the env has one, else a private cache.
        seed_table (np.ndarray): The episode seeds the env is expected to run, one row per generation.
        records (list): The per-generation records of the last :meth:`run`.
        fitness_history (list): The ``best_fitness`` of every record, in generation order.
        summary (dict): The summary of the last :meth:`run`.
        placement (dict): Placement evidence of every runtime leaf -- ``"pop"``/``"species"`` (one device string per leaf), ``"randkey"``, ``"generation"``, ``"fitness"``, ``"episode_returns"``, ``"returns"`` and ``"seeds"`` (``"host"``); refreshed by :meth:`setup` and at the end of every generation.
        build_time_s (float): Wall-clock seconds of :meth:`setup`, including every graph compilation.
        tell_build_time_s (float): Wall-clock seconds of the ``tell`` compilation.
        rule_build_time_s (float): Wall-clock seconds of the device fitness-rule compilation (0 for the numpy residency).
        verification_build_time_s (float): Wall-clock seconds of the device verification compilation (0 for the numpy residency).
        verification_atol (float): The fitness tolerance of the device verification, :data:`DEVICE_VERIFICATION_ATOL`.
    """

    def __init__(
        self,
        env,
        pop_size,
        generations,
        master_seed=0,
        *,
        num_inputs,
        num_outputs,
        max_nodes=80,
        max_conns=640,
        species_size=3,
        residency=None,
        tell_backend=None,
        tell_device=None,
        plugin_path=None,
        output_dir=None,
        logger=None,
        problem=None,
        seed_table=None,
    ):
        self.env = env
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        if self.pop_size < 1:
            raise ValueError(f"pop_size must be positive, got {pop_size!r}")
        if self.generations < 1:
            raise ValueError(f"generations must be positive, got {generations!r}")
        self.master_seed = int(master_seed)
        self.evolution_seed = evolution_seed(self.master_seed)
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        if self.num_inputs < 1 or self.num_outputs < 1:
            raise ValueError(
                f"num_inputs and num_outputs must be positive, got {num_inputs!r} and {num_outputs!r}"
            )
        self.max_nodes = int(max_nodes)
        self.max_conns = int(max_conns)
        self.species_size = min(int(species_size), self.pop_size)
        if self.species_size < 1:
            raise ValueError(f"species_size must be positive, got {species_size!r}")
        self.episodes_per_policy = int(getattr(env, "episodes_per_policy", EPISODES_PER_POLICY))
        self.residency = self._resolve_residency(residency)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.problem = problem if problem is not None else TraceOnlyProblem(self.num_inputs, self.num_outputs)
        self.plugin_path = plugin_path

        env_cache = getattr(env, "cache", None)
        self.cache = env_cache if env_cache is not None else ExecutableCache()
        self._owns_cache = env_cache is None

        self.tell_backend, self.tell_device = self._resolve_tell(tell_backend, tell_device)
        self.tell_device_name = _device_name(self.tell_device)
        self.tell_key = (
            "tell",
            self.pop_size,
            self.episodes_per_policy,
            self.tell_backend,
            self.tell_device_name,
            str(plugin_path),
            self.residency,
        )
        self.verification_atol = DEVICE_VERIFICATION_ATOL
        self.rule_key = (
            "fitness_rule",
            self.pop_size,
            self.episodes_per_policy,
            self.tell_backend,
            self.tell_device_name,
            self.residency,
        )
        self.verification_key = (
            "verification",
            self.pop_size,
            self.episodes_per_policy,
            self.tell_backend,
            self.tell_device_name,
            self.verification_atol,
            self.residency,
        )

        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = None if self.output_dir is None else self.output_dir / "history.jsonl"
        self.summary_path = None if self.output_dir is None else self.output_dir / "summary.json"

        self.seed_table = (
            generation_seed_table(self.master_seed, self.generations, episodes=self.episodes_per_policy)
            if seed_table is None
            else np.asarray(seed_table, dtype=np.int64)
        )

        self.algo = None
        self.config = None
        self.pop_arrays = None
        self.species_leaves = None
        self.randkey = None
        self.generation_counter = None
        self.records = []
        self.fitness_history = []
        self.summary = {}
        self.placement = {}
        self.build_time_s = 0.0
        self.tell_build_time_s = 0.0
        self.rule_build_time_s = 0.0
        self.verification_build_time_s = 0.0
        self._setup_done = False

    # --- Configuration ---------------------------------------------------------------

    def _resolve_residency(self, residency):
        """Resolve and validate the residency, inheriting it from the env when not given.

        ``residency="device"`` is only accepted from an env that declares it as well and exposes ``backend``/``device``, because the trainer drives that env's payload form and result contract.
        A cpu-kind device is allowed: it is the plumbing-only mode of the device residency.
        """
        value = residency if residency is not None else getattr(self.env, "residency", "numpy")
        if value not in RESIDENCIES:
            raise ValueError(f"residency must be one of {RESIDENCIES}, got {residency!r}")
        if value == "device":
            if getattr(self.env, "residency", None) != "device":
                raise ValueError(
                    f"device residency of the trainer requires the env to declare residency='device', "
                    f"but {type(self.env).__name__} declares {getattr(self.env, 'residency', None)!r}; "
                    "the env is the owner of the payload form and of the device result contract"
                )
            if getattr(self.env, "backend", None) is None or getattr(self.env, "device", None) is None:
                raise ValueError(
                    f"device residency requires {type(self.env).__name__}.backend and .device, the evoxc "
                    "backend and device the env's graphs (and therefore the trainer's) run on"
                )
        return value

    def _resolve_tell(self, tell_backend, tell_device):
        """Resolve the evolution ``(backend, device)`` the same way the Go2 trainer does.

        An explicit ``tell_backend`` wins; a numpy env stays numpy; a ROCm/HIP env becomes ``xla`` because IREE's ROCm compile of the tell graph does not terminate; otherwise the env backend is used.
        The device is ``cpu:0`` on numpy and the env device everywhere else.
        Under the device residency the env's own backend and device are used and an explicit different pair raises: evoxc payloads are not interchangeable (an IREE payload cannot enter an XLA executable and vice versa) and no device-to-device transfer exists.
        """
        env_backend = getattr(self.env, "backend", None) or "numpy"
        env_device = getattr(self.env, "device", None)
        if self.residency == "device":
            backend = env_backend if tell_backend is None else str(tell_backend)
            device = env_device if tell_device is None else tell_device
            resolved_backend, resolved_device = resolve_backend_device(backend, device)
            if resolved_backend != env_backend or _device_name(resolved_device) != _device_name(env_device):
                raise ValueError(
                    f"device residency is single-backend: the fitness, verification and tell graphs must run on "
                    f"{type(self.env).__name__}'s own {env_backend}@{_device_name(env_device)}, but "
                    f"tell_backend={resolved_backend!r} tell_device={_device_name(resolved_device)!r} was "
                    "requested; evoxc device payloads are not interchangeable (an IreeDevicePayload cannot "
                    "enter an xla executable and vice versa) and there is no device-to-device transfer"
                )
            return resolved_backend, resolved_device
        if tell_backend is not None:
            backend = str(tell_backend)
        elif env_backend == "numpy":
            backend = "numpy"
        elif env_device is not None and env_device.kind in ("rocm", "hip"):
            backend = "xla"
        else:
            backend = env_backend
        if tell_device is not None:
            device = tell_device
        elif backend == "numpy" or env_device is None:
            device = evoxc.Device("cpu", 0)
        else:
            device = env_device
        return resolve_backend_device(backend, device)

    @staticmethod
    def _static_config(pipeline, state):
        """Return the tensor-free statics the compiled graphs are traced with.

        TensorNEAT's ``Pipeline._static_config`` is private API, so the lookup is explicit and a future rename fails loudly instead of silently tracing a graph without the pipeline statics.
        The returned state holds the values the graphs need at trace time and no tensors, which is exactly what ``build_tell`` registers its runtime payload into.

        Args:
            pipeline (Pipeline): The pipeline whose statics are wanted.
            state (State): The ``pipeline.setup()`` state.

        Returns:
            State: The tensor-free static config.
        """
        extractor = getattr(pipeline, "_static_config", None)
        if not callable(extractor):
            raise RuntimeError(
                "TensorNEAT's private Pipeline._static_config is missing; the neuroevolution trainer "
                "needs it to trace the compiled graphs (pinned TensorNEAT version changed?)"
            )
        return extractor(state)

    # --- Setup -----------------------------------------------------------------------

    def setup(self):
        """Build the genome pipeline and every evolution-side executable (idempotent).

        Under the numpy residency the pipeline exists for its statics only: the initial population, the species pytree, the RNG key, the generation counter and the tensor-free config.
        Under the device residency the generation-0 state is built by :meth:`_setup_device` on the evolution device instead, so no leaf ever crosses the host.
        Its seed is the evolution stream only -- the episode seeds come from :attr:`seed_table` -- so the evolution RNG is independent of the environment RNG.
        The ``tell`` graph, and for the device residency the fitness-rule and verification graphs, are compiled once through the cache, and the environment is warmed up, if it exposes ``executables()``, so no task graph compiles inside a generation.

        Returns:
            NeuroevolutionTrainer: ``self``.
        """
        if self._setup_done:
            return self
        build_start = time.perf_counter()
        gen = genome.DefaultGenome(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            max_nodes=self.max_nodes,
            max_conns=self.max_conns,
        )
        self.algo = algorithm.NEAT(gen, pop_size=self.pop_size, species_size=self.species_size)
        if self.residency == "device":
            self._setup_device()
        else:
            pipeline = Pipeline(
                self.algo,
                self.problem,
                seed=self.evolution_seed,
                backend="numpy",
                is_detailed=False,
            )
            state = pipeline.setup()
            self.config = self._static_config(pipeline, state)
            self.pop_arrays = pop_arrays_of(state)
            self.species_leaves = species_leaves_of(state.species)
            self.randkey = state.randkey
            self.generation_counter = state.generation
        self._setup_done = True

        self.logger.info(
            "Building the TensorNEAT tell graph on %s@%s", self.tell_backend, self.tell_device_name
        )
        tell_start = time.perf_counter()
        self._tell_executable()
        self.tell_build_time_s = time.perf_counter() - tell_start

        if self.residency == "device":
            rule_start = time.perf_counter()
            self._fitness_rule_executable()
            self.rule_build_time_s = time.perf_counter() - rule_start
            verification_start = time.perf_counter()
            self._verification_executable()
            self.verification_build_time_s = time.perf_counter() - verification_start

        env_executables = getattr(self.env, "executables", None)
        if callable(env_executables):
            env_executables()

        self._refresh_placement()
        self.build_time_s = time.perf_counter() - build_start
        self.logger.info(
            "Setup done in %.3f s (%s residency, tell %.3f s, rule %.3f s, verification %.3f s, "
            "cache %d hits / %d misses)",
            self.build_time_s,
            self.residency,
            self.tell_build_time_s,
            self.rule_build_time_s,
            self.verification_build_time_s,
            self._cache_hits(),
            self._cache_misses(),
        )
        return self

    def _setup_device(self):
        """Build the generation-0 state on the evolution device, without a host roundtrip.

        ``device_runtime.device_setup`` registers the RNG key and the host formulation triple on a bare state and runs TensorNEAT's own initialization graph on the target device, so the payload, the species leaves, the key and the generation counter come back device-resident.
        The xla backend additionally needs its PJRT plugin at compile time; it is forwarded as the ``plugin_path`` compile option and no other backend accepts it.
        """
        compile_options = None
        if self.tell_backend == "xla" and self.plugin_path is not None:
            compile_options = {"plugin_path": str(self.plugin_path)}
        self.config, state = device_setup(
            self.algo,
            self.problem,
            seed=self.evolution_seed,
            backend=self.tell_backend,
            device=self.tell_device,
            compile_options=compile_options,
        )
        self.pop_arrays = device_payload(state)
        self.species_leaves = device_species_leaves(state)
        self.randkey = state.randkey
        self.generation_counter = state.generation
        assert_payload_resident(
            self.pop_arrays, self.species_leaves, self.randkey, self.generation_counter, self.tell_device
        )

    def _tell_executable(self):
        """Return the cached tell executable, compiling it on the first call."""
        return self.cache.get(
            self.tell_key,
            lambda: build_tell(
                self.algo,
                self.config,
                self.pop_arrays,
                self.species_leaves,
                self.pop_size,
                backend=self.tell_backend,
                device=self.tell_device,
                plugin_path=self.plugin_path,
            ),
        )

    def _fitness_rule_executable(self):
        """Return the cached device fitness-rule graph, compiling it on the first call."""
        return self.cache.get(
            self.rule_key,
            lambda: build_fitness_rule(
                self.pop_size,
                episodes_per_policy=self.episodes_per_policy,
                backend=self.tell_backend,
                device=self.tell_device,
            ),
        )

    def _verification_executable(self):
        """Return the cached device verification graph, compiling it on the first call."""
        return self.cache.get(
            self.verification_key,
            lambda: build_verification(
                self.pop_size,
                episodes_per_policy=self.episodes_per_policy,
                backend=self.tell_backend,
                device=self.tell_device,
                atol=self.verification_atol,
            ),
        )

    # --- Run -------------------------------------------------------------------------

    def run(self):
        """Run every generation and return the summary dict.

        Each generation: evaluate the environment, validate its result, aggregate the fitness, run the tell graph, write the record, and advance the RNG key, the generation counter, the population and the species pytree.
        A record's ``fitness`` is the trainer's own aggregation of ``episode_returns``; a disagreeing ``result["fitness"]`` raises, because exactly one owner of the aggregation rule keeps tasks comparable.
        Under the device residency the loop is :meth:`_run_device`, which hands the device payload straight to ``env.evaluate`` and stages nothing; both loops produce the same records and artifacts.

        Returns:
            dict: ``config``, ``generations`` (the records) and ``totals``; also stored as :attr:`summary`.
        """
        if not self._setup_done:
            self.setup()
        if self.residency == "device":
            return self._run_device()
        records = []
        previous_hits, previous_misses = self._cache_hits(), self._cache_misses()
        run_start = time.perf_counter()
        history = None if self.history_path is None else self.history_path.open("w", encoding="utf-8")
        try:
            for generation in range(self.generations):
                result = self.env.evaluate(self.pop_arrays, generation)
                validate_evaluation(result, self.pop_size, episodes=self.episodes_per_policy)
                episode_returns, fitness = self._aggregate(result)
                tell = run_tell(
                    self._tell_executable(),
                    randkey=self.randkey,
                    pop_arrays=self.pop_arrays,
                    generation=self.generation_counter,
                    fitness=fitness,
                    species_leaves=self.species_leaves,
                )
                record = self._build_record(
                    generation,
                    result,
                    episode_returns,
                    fitness,
                    previous_hits,
                    previous_misses,
                )
                records.append(record)
                if history is not None:
                    history.write(json.dumps(_jsonable(record)) + "\n")
                    history.flush()
                self.logger.info("%s", _format_generation(record))
                previous_hits, previous_misses = record["cache_hits"], record["cache_misses"]
                self.randkey = tell["randkey"]
                self.generation_counter = tell["generation"]
                self.pop_arrays = tuple(np.array(leaf, copy=True) for leaf in tell["pop"])
                self.species_leaves = species_leaves_of(tell["species"])
        finally:
            if history is not None:
                history.close()
        wall_time_s = time.perf_counter() - run_start
        return self._finish(records, wall_time_s)

    def _run_device(self):
        """Run every generation device-resident: the device payload in, the tell graph's device output back.

        The env receives the four device payload leaves and returns the four device metric tensors; the trainer computes the fitness on the device, verifies the env's result in-graph, and only reads the four metrics back (plus the host logging scalars) once the tell graph has consumed the device tensors.
        The record schema, the artifacts and the logger lines are the numpy loop's.
        """
        records = []
        previous_hits, previous_misses = self._cache_hits(), self._cache_misses()
        run_start = time.perf_counter()
        history = None if self.history_path is None else self.history_path.open("w", encoding="utf-8")
        try:
            for generation in range(self.generations):
                result = self.env.evaluate(self.pop_arrays, generation)
                validate_evaluation_device(
                    result,
                    self.pop_size,
                    episodes=self.episodes_per_policy,
                    device=self.tell_device,
                )
                episode_returns, fitness = self._aggregate_device(result)
                tell = run_tell(
                    self._tell_executable(),
                    randkey=self.randkey,
                    pop_arrays=self.pop_arrays,
                    generation=self.generation_counter,
                    fitness=fitness,
                    species_leaves=self.species_leaves,
                    residency="device",
                )
                record = self._build_record_device(
                    generation,
                    result,
                    episode_returns,
                    fitness,
                    previous_hits,
                    previous_misses,
                )
                records.append(record)
                if history is not None:
                    history.write(json.dumps(_jsonable(record)) + "\n")
                    history.flush()
                self.logger.info("%s", _format_generation(record))
                previous_hits, previous_misses = record["cache_hits"], record["cache_misses"]
                self.randkey = tell["randkey"]
                self.generation_counter = tell["generation"]
                self.pop_arrays = tuple(tell["pop"])
                self.species_leaves = species_leaves_of(tell["species"])
                self._refresh_placement(result=result, fitness=fitness)
        finally:
            if history is not None:
                history.close()
        wall_time_s = time.perf_counter() - run_start
        return self._finish(records, wall_time_s)

    def _finish(self, records, wall_time_s):
        """Store the records and the summary, write ``summary.json``, log the run and return the summary."""
        self.records = records
        self.fitness_history = [record["best_fitness"] for record in records]
        self.summary = self._build_summary(records, wall_time_s)
        if self.summary_path is not None:
            self.summary_path.write_text(json.dumps(_jsonable(self.summary), indent=2) + "\n", encoding="utf-8")
        self.logger.info(
            "Done: %d generations, %d steps, %.1f s total, best fitness %s",
            len(records),
            self.summary["totals"]["steps"],
            wall_time_s,
            "n/a" if self.summary["totals"]["best_fitness"] is None else f"{self.summary['totals']['best_fitness']:+.4f}",
        )
        return self.summary

    def _aggregate(self, result):
        """Return ``(episode_returns, fitness)`` with the trainer's aggregation rule applied.

        An unscored or otherwise non-finite episode (NaN or +/-inf) counts as 0, and the mean is taken in float32.
        The rule itself is owned by ``env_base._fitness_from_episode_returns`` and imported here, so the trainer and ``env_base.validate_evaluation`` cannot disagree about an env's fitness.
        The float32 arithmetic is what the compiled ``fitness_graph`` (``evoxc.reduce_mean``) the env reports uses; a float64 mean differs from that graph's answer by more than the ``1e-6`` tolerance at episode-return magnitudes of ~100, so it would reject a legitimate env.
        ``result["fitness"]`` must agree with that rule to ``1e-6``, otherwise the env is computing a second, possibly different fitness.

        Args:
            result (Mapping): A validated :meth:`env_base.BatchedEpisodeEnv.evaluate` result.

        Returns:
            tuple: ``episode_returns`` ``(pop_size, episodes)`` float32 and ``fitness`` ``(pop_size,)`` float32.
        """
        episode_returns = np.asarray(result["episode_returns"], dtype=np.float32)
        fitness = _fitness_from_episode_returns(episode_returns)
        reported = np.asarray(result["fitness"], dtype=np.float32)
        if reported.shape != fitness.shape or not np.allclose(reported, fitness, atol=1e-6, rtol=0):
            raise ValueError(
                f"{type(self.env).__name__}.evaluate reported fitness {reported.tolist()} but the trainer's "
                f"episode-mean aggregation of episode_returns is {fitness.tolist()}; the aggregation rule "
                f"has exactly one owner, so the env must report the same vector"
            )
        return episode_returns, fitness

    def _aggregate_device(self, result):
        """Return the device ``(episode_returns, fitness)`` and verify the env's device result in-graph.

        The trainer owns the arithmetic exactly as on the numpy path: the fitness is :func:`evolution_graphs.run_fitness_rule` of the policy-major ``returns`` -- the device transcription of ``env_base._fitness_from_episode_returns`` (non-finite episodes counted as 0, float32 mean).
        :func:`evolution_graphs.run_verification` then checks in-graph that ``returns`` is the policy-major tiling of ``episode_returns`` and that the env's reported ``fitness`` matches that rule to :attr:`verification_atol`; only its three scalars are read back.
        Nothing is materialized here: both returned values are tensors on :attr:`tell_device`.

        Args:
            result (Mapping): A validated :meth:`env_base.BatchedEpisodeEnv.evaluate` device result.

        Returns:
            tuple: ``episode_returns`` ``(pop_size, episodes)`` and ``fitness`` ``(pop_size,)``, both float32 on the tell device.
        """
        for key in DEVICE_KEYS:
            assert_device_resident(result[key], self.tell_device, key)
        episode_returns = result["episode_returns"]
        fitness = run_fitness_rule(self._fitness_rule_executable(), result["returns"], residency="device")
        verification = run_verification(
            self._verification_executable(),
            returns=result["returns"],
            episode_returns=episode_returns,
            fitness=result["fitness"],
            residency="device",
        )
        if not verification["returns_match"]:
            raise ValueError(
                f"{type(self.env).__name__}.evaluate returned a 'returns' tensor that is not the policy-major "
                f"flattening of 'episode_returns'; the row tiling is part of the environment contract"
            )
        if not verification["fitness_match"]:
            raise ValueError(
                f"{type(self.env).__name__}.evaluate reported a 'fitness' that differs from the trainer's "
                f"device aggregation of 'returns' by {verification['fitness_max_abs_diff']:.3e}, more than "
                f"atol {self.verification_atol}; the aggregation rule has exactly one owner, so the env must "
                f"report the float32 mean of its episode returns with every non-finite episode counted as 0"
            )
        return episode_returns, fitness

    def _build_record(self, generation, result, episode_returns, fitness, previous_hits, previous_misses):
        """Return the JSON-safe record of one generation, including the unconsumed env diagnostics."""
        return self._record(
            generation,
            result,
            np.asarray(result["seeds"]),
            np.asarray(episode_returns, dtype=np.float32),
            np.asarray(result["returns"], dtype=np.float32),
            np.asarray(fitness, dtype=np.float32),
            previous_hits,
            previous_misses,
        )

    def _build_record_device(self, generation, result, episode_returns, fitness, previous_hits, previous_misses):
        """Return the same record from a device result, materializing only its four metrics.

        ``seeds``, ``episode_returns``, ``returns`` and ``fitness`` are copied to the host here, after the tell graph consumed the device tensors; everything else is already a host value.
        """
        return self._record(
            generation,
            result,
            _to_numpy(result["seeds"]),
            np.asarray(_to_numpy(episode_returns), dtype=np.float32),
            np.asarray(_to_numpy(result["returns"]), dtype=np.float32),
            np.asarray(_to_numpy(fitness), dtype=np.float32),
            previous_hits,
            previous_misses,
        )

    def _record(self, generation, result, seeds, episode_returns, returns, fitness, previous_hits, previous_misses):
        """Assemble the JSON-safe record of one generation from host metrics."""
        wall_time_s = float(result["wall_time_s"])
        steps = int(result["steps"])
        env_steps = int(result["env_steps"])
        steps_per_s = result.get("steps_per_s")
        env_steps_per_s = result.get("env_steps_per_s")
        if steps_per_s is None:
            steps_per_s = steps / wall_time_s if wall_time_s > 0.0 else 0.0
        if env_steps_per_s is None:
            env_steps_per_s = env_steps / wall_time_s if wall_time_s > 0.0 else 0.0
        hits, misses = self._cache_hits(), self._cache_misses()
        consumed = frozenset(REQUIRED_KEYS) | frozenset(OPTIONAL_KEYS)
        diagnostics = {key: value for key, value in result.items() if key not in consumed}
        end_reasons = result.get("end_reasons") or {}
        return {
            "generation": int(generation),
            "seeds": [int(seed) for seed in np.asarray(seeds).tolist()],
            "episode_returns": episode_returns.tolist(),
            "returns": np.asarray(returns, dtype=np.float32).tolist(),
            "fitness": fitness.tolist(),
            "best_fitness": float(fitness.max()) if fitness.size else None,
            "median_fitness": float(np.median(fitness)) if fitness.size else None,
            "wall_time_s": wall_time_s,
            "steps": steps,
            "env_steps": env_steps,
            "steps_per_s": float(steps_per_s),
            "env_steps_per_s": float(env_steps_per_s),
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hits_delta": int(hits - previous_hits),
            "cache_misses_delta": int(misses - previous_misses),
            "end_reasons": {str(name): int(count) for name, count in end_reasons.items()},
            "diagnostics": _jsonable(diagnostics),
        }

    def _refresh_placement(self, result=None, fitness=None):
        """Refresh :attr:`placement`, the placement evidence of every runtime leaf of the last generation.

        The numpy residency reports ``"host"`` for every entry; the device residency reports the device string (``"rocm:0"``, ``"cpu:0"``) of the payload, the species leaves, the key and the generation counter, plus the four result metrics once a generation has run.
        """
        if self.residency != "device":
            self.placement = {
                "pop": ["host"] * len(self.pop_arrays),
                "species": ["host"] * len(self.species_leaves),
                "randkey": "host",
                "generation": "host",
                "fitness": "host",
                "episode_returns": "host",
                "returns": "host",
                "seeds": "host",
            }
            return
        placement = dict(placement_map({"randkey": self.randkey, "generation": self.generation_counter}))
        placement["pop"] = [device_placement(leaf) for leaf in self.pop_arrays]
        placement["species"] = [device_placement(leaf) for leaf in self.species_leaves]
        placement["fitness"] = device_placement(fitness)
        for key in ("episode_returns", "returns", "seeds"):
            placement[key] = "host" if result is None else device_placement(result[key])
        self.placement = placement

    def _build_summary(self, records, wall_time_s):
        """Return the summary dict: ``config``, the generation records and the ``totals``."""
        config = {
            "pop_size": self.pop_size,
            "rows": int(getattr(self.env, "rows", rows_for(self.pop_size, self.episodes_per_policy))),
            "episodes_per_policy": self.episodes_per_policy,
            "residency": self.residency,
            "generations": self.generations,
            "master_seed": self.master_seed,
            "evolution_seed": self.evolution_seed,
            "seed_table": np.asarray(self.seed_table).tolist(),
            "genome": {
                "num_inputs": self.num_inputs,
                "num_outputs": self.num_outputs,
                "max_nodes": self.max_nodes,
                "max_conns": self.max_conns,
                "species_size": self.species_size,
            },
            "tell_backend": self.tell_backend,
            "tell_device": self.tell_device_name,
            "executable_build_time_s": self.build_time_s,
            "tell_build_time_s": self.tell_build_time_s,
        }
        for name in ENV_CONFIG_FIELDS:
            value = getattr(self.env, name, None)
            if value is not None:
                config[name] = int(value)
        fitness_history = [record["best_fitness"] for record in records]
        return {
            "config": config,
            "generations": records,
            "totals": {
                "wall_time_s": wall_time_s,
                "steps": sum(record["steps"] for record in records),
                "env_steps": sum(record["env_steps"] for record in records),
                "best_fitness": max(fitness_history) if fitness_history else None,
                "final_best_fitness": fitness_history[-1] if fitness_history else None,
                "fitness_history": fitness_history,
                "cache_hits": self._cache_hits(),
                "cache_misses": self._cache_misses(),
            },
        }

    # --- Cache / lifecycle ------------------------------------------------------------

    def _cache_hits(self):
        """Return the cache hit counter (0 when the cache does not track hits)."""
        return int(getattr(self.cache, "hits", 0))

    def _cache_misses(self):
        """Return the cache miss counter (0 when the cache does not track misses)."""
        return int(getattr(self.cache, "misses", 0))

    def close(self):
        """Drop a cache the trainer created itself; the environment is never closed here.

        The caller owns :attr:`env`, so it also owns its lifecycle: closing it is the caller's decision, made exactly once after the last generation.
        """
        if self._owns_cache:
            self.cache.clear()

    def __enter__(self):
        """Return ``self`` so the trainer can be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the trainer's own cache; the environment is left open."""
        self.close()
        return False


# --- Payload helpers ----------------------------------------------------------------


def pop_arrays_of(state):
    """Return the four genome payload arrays of a pipeline state as host numpy.

    Args:
        state: A ``Pipeline.setup()`` state.

    Returns:
        tuple: The four ``GenomeTensors`` payload leaves, in genome order.
    """
    return tuple(np.array(to_numpy(leaf)) for leaf in state.pop)


def species_leaves_of(species):
    """Return the species pytree leaves in ``evoxc`` pytree order.

    Args:
        species: A TensorNEAT ``SpeciesState`` (or the rebuilt one ``run_tell`` hands back).

    Returns:
        tuple: The species leaves, in ``evoxc.tree_leaves`` order.
    """
    return tuple(evoxc.tree_leaves(species))


# --- Artifacts ----------------------------------------------------------------------


def _jsonable(value):
    """Convert numpy scalars/arrays, evoxc tensors (and containers of them) to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "device") and hasattr(value, "numpy"):
        # An evoxc tensor handed back as a task diagnostic: materialize it instead of failing.
        return _to_numpy(value).tolist()
    return value


def _format_generation(record):
    """One compact log line for a generation record."""
    return (
        f"gen {record['generation']} | seeds {record['seeds']} | "
        f"best {record['best_fitness']:+.4f} median {record['median_fitness']:+.4f} | "
        f"{record['steps']} steps {record['wall_time_s']:.3f} s "
        f"({record['steps_per_s']:.1f} steps/s, {record['env_steps_per_s']:.0f} env-steps/s) | "
        f"cache +{record['cache_hits_delta']}/+{record['cache_misses_delta']} | "
        f"ends {record['end_reasons']}"
    )
