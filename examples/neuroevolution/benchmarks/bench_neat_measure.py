"""Phase-3 measurement: one full TensorNEAT training run, timed region by region.

``bench_envs.py`` (Phase 1) checks the migrated evaluation maths and ``bench_runtime.py`` (Phase 2)
measures the environment's throughput and residency; this module measures the *trainer* on top of
them: build the pilot env, build :class:`trainer.NeuroevolutionTrainer`, compile the ``tell`` graph,
then run generations and time the two regions a generation is made of -- ``env.evaluate`` (the seed
draw, the reset, the steps and the fitness graph) and the ``tell`` transition -- plus the
``generation`` wall clock that contains them and the residual bookkeeping the trainer does between
them.  Nothing here writes a record; ``bench_neat.py`` owns the CLI and the record schema.

The instrumentation is the Phase-2 instrumentation: :func:`runtime_probes._wrap_executable_cache`
times every graph build through ``ExecutableCache.get`` (so the record has one build time per graph
name, ``tell`` included), :func:`common.wrap_method` / :func:`common.wrap_function` time the two
regions through the trainer's own ``env.evaluate`` and ``trainer.run_tell`` call sites, and
:func:`common.sync` brackets every region so a GPU process reports device completion inside the
region instead of in the next one.

Held-out evaluation happens in the same process, after ``trainer.run()``: every evaluated
population was copied out of the ``env.evaluate`` call, and the first (initial) and the last
(trained) of them are scored on one fresh seed triple that no training generation used.  The
environment's ``seed_table`` is restored afterwards, so the swap is invisible to anything else.

All the sibling files are read-only here: the patches are runtime monkey-patching and every one of
them is restored before the environment is closed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time

import numpy as np

from bench_envs import _build_env, _load_task, _r
from common import Timer, setup_paths, stats, sync, wrap_function, wrap_method
from runtime_probes import _wrap_executable_cache

#: Genome widths of the two pilots: ``(num_inputs, num_outputs)``.
TASK_GENOME = {"go2": (45, 12), "hover": (17, 4)}

#: Genome budgets shared by both pilots (``go2_neat_env.MAX_NODES`` / ``hover_neat_env.MAX_NODES``).
MAX_NODES = 80
MAX_CONNS = 640
SPECIES_SIZE = 3

#: Offset of the held-out seed stream: ``generation_seed_table(master_seed + 1000, 1)`` is a triple
#: no training generation ever drew, and the two streams stay independent of the training one.
HOLDOUT_SEED_OFFSET = 1000

#: The timeout each pilot's episode is capped at, taken from the env default (go2 20 s, hover 15 s).
EPISODE_LENGTH_NOTE = "env default: Go2 20 s (max_episode_length 1000), hover 15 s (1500)"


def perturbation_for(task: str):
    """Return the reset perturbation the pilot trainer CLI defaults to.

    Go2 perturbs the base position, the base orientation and the joints (0.05 each); the drone has
    no joints, so only its base pose is perturbed.
    """
    setup_paths()  # ``seeding`` is a bare-name sibling module of examples/neuroevolution
    from seeding import Perturbation

    if task == "go2":
        return Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05)
    if task == "hover":
        return Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.0)
    raise ValueError(f"unknown task {task!r}; expected 'go2' or 'hover'")


def genome_of(task: str, module) -> dict:
    """Return the genome block, after checking the task module's own widths against this harness."""
    num_inputs, num_outputs = TASK_GENOME[task]
    for name, expected in (("NUM_OBS", num_inputs), ("NUM_ACTIONS", num_outputs)):
        observed = getattr(module, name, None)
        if observed is not None and int(observed) != int(expected):
            raise ValueError(
                f"{task}: {module.__name__}.{name} is {observed}, this harness is written for {expected}"
            )
    return {
        "num_inputs": int(num_inputs),
        "num_outputs": int(num_outputs),
        "max_nodes": MAX_NODES,
        "max_conns": MAX_CONNS,
        "species_size": SPECIES_SIZE,
    }


def logger() -> logging.Logger:
    """Return the benchmark's logger (the trainer logs one line per generation through it)."""
    return logging.getLogger("bench_neat")


# --- region probes -----------------------------------------------------------------


class RegionProbe:
    """Stamp one timed region from the ``on_call`` hook of :func:`common.wrap_method` / ``wrap_function``.

    ``common``'s wrappers add the seconds they measured to a :class:`common.Timer` and call
    ``on_call`` *outside* the timed region, so the region's host time is recovered from the timer's
    cumulative total and its start from the moment the callback runs.  ``sync()`` runs inside the
    callback: on a GPU process the device completion of a region then belongs to that region, and
    the next region starts after it, which is what the end-to-end wall clock means.

    Args:
        timer (Timer): The accumulator the wrapper fills under ``tag``.
        tag (str): Accumulator name, ``"evaluate"`` or ``"tell"``.
    """

    def __init__(self, timer: Timer, tag: str):
        self.timer = timer
        self.tag = tag
        self.previous = 0.0
        self.samples = []

    def __call__(self, args, kwargs, result) -> None:
        cumulative = float(self.timer.totals().get(self.tag, 0.0))
        host_s = cumulative - self.previous
        self.previous = cumulative
        sync_start = time.perf_counter()
        synchronised = sync()
        sync_s = time.perf_counter() - sync_start
        end = time.perf_counter()
        self.samples.append(
            {
                "tag": self.tag,
                "start": end - host_s - sync_s,
                "end": end,
                "host_s": host_s,
                "sync_s": sync_s,
                "s": host_s + sync_s,
                "synchronised": bool(synchronised),
            }
        )


class PopulationCapture:
    """Copy the population handed to ``env.evaluate``, one copy per training generation.

    The capture is the ``on_call`` hook of the ``env.evaluate`` wrapper, so the copy is taken once
    the generation has been evaluated (and validated) and before the trainer's ``tell`` replaces
    the population.  Held-out evaluation needs exactly these arrays: the trained population is the
    one that was *evaluated* last, not the offspring ``trainer.pop_arrays`` holds after the tell.
    """

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.populations = []

    def __call__(self, args, kwargs, result) -> None:
        if len(self.populations) >= self.limit or len(args) < 2:
            return
        self.populations.append(tuple(np.array(leaf, copy=True) for leaf in args[1]))


# --- held-out evaluation -----------------------------------------------------------


def _holdout_table(master_seed: int) -> np.ndarray:
    """The ``(1, episodes)`` held-out seed table of a run: a stream no training generation drew."""
    from seeding import generation_seed_table

    return generation_seed_table(int(master_seed) + HOLDOUT_SEED_OFFSET, 1)


def _side(result: dict) -> dict:
    """Summarise one held-out evaluation (the same fields for the initial and the trained side)."""
    fitness = np.asarray(result["fitness"], dtype=np.float64)
    episode_returns = np.asarray(result["episode_returns"], dtype=np.float64)
    return {
        "seeds": [int(seed) for seed in np.asarray(result["seeds"]).reshape(-1)],
        "fitness": [_r(value) for value in fitness],
        "episode_returns": [[_r(value) for value in row] for row in episode_returns],
        "best_fitness": _r(float(fitness.max())) if fitness.size else None,
        "median_fitness": _r(float(np.median(fitness))) if fitness.size else None,
        "steps": int(result["steps"]),
        "env_steps": int(result["env_steps"]),
        "end_reasons": {str(name): int(count) for name, count in (result.get("end_reasons") or {}).items()},
    }


def _evaluate_once(env, pop_arrays, generation: int) -> tuple[dict, float]:
    """Evaluate one population on the held-out table and return ``(result, seconds)``."""
    sync()
    start = time.perf_counter()
    result = env.evaluate(pop_arrays, generation)
    sync()
    return result, time.perf_counter() - start


def holdout(env, populations: list, records: list, training_table, master_seed: int) -> dict:
    """Score the initial and the trained population on one fresh seed triple, then restore the env.

    Args:
        env: The pilot environment (its ``seed_table`` is swapped and restored here).
        populations (list): The populations copied out of every training generation.
        records (list): The trainer's per-generation records (the last one holds the train fitness).
        training_table (array-like): The ``(generations, episodes)`` table the run trained on.
        master_seed (int): The run seed; the held-out stream is ``master_seed + 1000``.

    Returns:
        dict: The ``holdout`` block of the record: both fitness vectors, the per-policy deltas, the
        best policy of the last training generation with its train and held-out fitness, and the
        timing of the two evaluations.  A single-generation run has only one evaluated population,
        so both sides are that population and ``separates_initial_from_trained`` says so.
    """
    if not populations:
        raise RuntimeError("held-out evaluation needs at least the initial population, captured none")
    initial_population = populations[0]
    trained_population = populations[-1]
    table = _holdout_table(master_seed)
    training_seeds = {int(seed) for seed in np.asarray(training_table).reshape(-1)}
    heldout_seeds = {int(seed) for seed in table.reshape(-1)}
    original_table = env.seed_table
    env.seed_table = np.asarray(table, dtype=np.int64)
    try:
        initial, initial_s = _evaluate_once(env, initial_population, 0)
        trained, trained_s = _evaluate_once(env, trained_population, 0)
    finally:
        env.seed_table = original_table

    train_fitness = np.asarray(records[-1]["fitness"], dtype=np.float64)
    best_index = int(np.argmax(train_fitness)) if train_fitness.size else 0
    initial_fitness = np.asarray(initial["fitness"], dtype=np.float64)
    trained_fitness = np.asarray(trained["fitness"], dtype=np.float64)
    delta = trained_fitness - initial_fitness
    return {
        "seed_table": table.tolist(),
        "source": f"seeding.generation_seed_table(master_seed + {HOLDOUT_SEED_OFFSET}, 1)",
        "episodes_per_policy": int(env.episodes_per_policy),
        "disjoint_from_training": not (heldout_seeds & training_seeds),
        "separates_initial_from_trained": len(populations) > 1,
        "evaluations": 2,
        "initial": _side(initial),
        "trained": _side(trained),
        "best_policy_index": best_index,
        "best_policy": {
            "index": best_index,
            "train_fitness": _r(train_fitness[best_index]) if train_fitness.size else None,
            "heldout_fitness": _r(trained_fitness[best_index]) if trained_fitness.size else None,
            "delta": _r(delta[best_index]) if delta.size else None,
        },
        "per_policy_delta": [_r(value) for value in delta],
        "median_initial": _r(float(np.median(initial_fitness))) if initial_fitness.size else None,
        "median_trained": _r(float(np.median(trained_fitness))) if trained_fitness.size else None,
        "median_delta": _r(float(np.median(delta))) if delta.size else None,
        "evaluate_s": {"initial": _r(initial_s), "trained": _r(trained_s)},
        "note": (
            "one held-out triple (3 episodes per policy) because every episode is a full-horizon "
            "episode of the pilot's own episode length; the triple is never used for training, and "
            "the trained side is the population that was evaluated in the last training generation "
            "(with a single generation the two sides are necessarily the same population)"
        ),
    }


# --- per-generation entry ----------------------------------------------------------


def generation_entry(index, record, evaluate_sample, tell_sample, generation_s) -> dict:
    """Assemble one generation's record entry: fitness/seeds plus the harness-measured regions."""
    fitness = [float(value) for value in record["fitness"]]
    evaluate_s = evaluate_sample["s"]
    tell_s = tell_sample["s"]
    return {
        "generation": int(index),
        "seeds": [int(seed) for seed in record["seeds"]],
        "fitness": [_r(value) for value in fitness],
        "episode_returns": [[_r(value) for value in row] for row in record["episode_returns"]],
        "best_fitness": _r(record["best_fitness"]),
        "median_fitness": _r(record["median_fitness"]),
        "evaluate_s": _r(evaluate_s),
        "env_wall_s": _r(record["wall_time_s"]),
        "tell_s": _r(tell_s),
        "generation_s": _r(generation_s),
        "bookkeeping_s": _r(max(0.0, generation_s - evaluate_s - tell_s)),
        "evaluate_share": _r(evaluate_s / generation_s) if generation_s > 0.0 else None,
        "sync_s": _r(evaluate_sample["sync_s"] + tell_sample["sync_s"]),
        "synchronised": bool(evaluate_sample["synchronised"] or tell_sample["synchronised"]),
        "steps": int(record["steps"]),
        "env_steps": int(record["env_steps"]),
        "steps_per_s": _r(record["steps_per_s"]),
        "env_steps_per_s": _r(record["env_steps_per_s"]),
        "env_steps_per_s_harness": _r(record["env_steps"] / evaluate_s) if evaluate_s > 0.0 else None,
        "cache_hits": int(record["cache_hits"]),
        "cache_misses": int(record["cache_misses"]),
        "cache_hits_delta": int(record["cache_hits_delta"]),
        "cache_misses_delta": int(record["cache_misses_delta"]),
        "end_reasons": {str(name): int(count) for name, count in record["end_reasons"].items()},
        "diagnostics": record.get("diagnostics") or {},
    }


# --- determinism fingerprint -------------------------------------------------------


def fingerprint_payload(seed_table, generations: list, holdout_block: dict) -> dict:
    """Return the canonical value the determinism fingerprint is taken over.

    It is the four things a second process must reproduce exactly: the episode seed table, the
    seeds of every generation, the fitness vector of every generation (rounded to 6 decimals, the
    precision of the Phase-1 fingerprint) and the held-out fitness of both sides.
    """
    return {
        "seed_table": np.asarray(seed_table, dtype=np.int64).tolist(),
        "generation_seeds": [[int(seed) for seed in entry["seeds"]] for entry in generations],
        "fitness": [[round(float(value), 6) for value in entry["fitness"]] for entry in generations],
        "holdout": {
            "initial": [round(float(value), 6) for value in holdout_block["initial"]["fitness"]],
            "trained": [round(float(value), 6) for value in holdout_block["trained"]["fitness"]],
        },
    }


def fingerprint(seed_table, generations: list, holdout_block: dict) -> str:
    """sha256 over :func:`fingerprint_payload`, as the record's ``determinism_fingerprint``."""
    payload = fingerprint_payload(seed_table, generations, holdout_block)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the measurement ---------------------------------------------------------------


def measure(args) -> dict:
    """Build, train and measure one pilot run; returns the measurement block of the record.

    Args:
        args (argparse.Namespace): The parsed ``train`` subcommand.

    Returns:
        dict: The measurement block: resolved targets, per-generation entries, totals, phases,
        the trainer's own summary config, the held-out block and the checks.
    """
    task = str(args.task)
    setup_paths()
    import_start = time.perf_counter()
    module, env_cls, _payload = _load_task(task)
    import_s = time.perf_counter() - import_start
    import trainer as trainer_module  # noqa: PLC0415 - needs the sibling paths of ``setup_paths``

    genome = genome_of(task, module)
    perturbation = perturbation_for(task)
    from seeding import generation_seed_table  # noqa: PLC0415 - same reason

    seed_table = generation_seed_table(int(args.master_seed), int(args.generations))
    started = time.perf_counter()
    timer = Timer(("evaluate", "tell"))
    builds = {}
    restore_cache = _wrap_executable_cache(builds)
    env = None
    trainer = None
    close_s = None
    try:
        build_start = time.perf_counter()
        env = _build_env(
            env_cls,
            int(args.pop_size),
            int(args.master_seed),
            int(args.horizon),
            residency=str(args.residency),
            backend=str(args.backend),
            device=args.device,
            stop_when_done=False,
            perturbation=perturbation,
        )
        build_s = time.perf_counter() - build_start
        env_info = {
            "rows": int(env.rows),
            "episodes_per_policy": int(env.episodes_per_policy),
            "max_episode_length": int(env.max_episode_length),
            "backend": str(env.backend),
            "device_name": str(env.device_name),
        }
        # The env reads ``seed_table`` in ``_seeds_for`` only, so handing it the trainer's table is
        # what makes the episode seeds the record claims verifiable; the held-out swap restores it.
        env.seed_table = np.asarray(seed_table, dtype=np.int64)

        trainer = trainer_module.NeuroevolutionTrainer(
            env,
            pop_size=int(args.pop_size),
            generations=int(args.generations),
            master_seed=int(args.master_seed),
            num_inputs=genome["num_inputs"],
            num_outputs=genome["num_outputs"],
            max_nodes=MAX_NODES,
            max_conns=MAX_CONNS,
            species_size=SPECIES_SIZE,
            tell_backend=args.tell_backend,
            plugin_path=args.pjrt_plugin,
            output_dir=None,
            logger=logger(),
            seed_table=seed_table,
        )
        setup_start = time.perf_counter()
        trainer.setup()
        setup_s = time.perf_counter() - setup_start
        builds_before_run = {name: int(slot["builds"]) for name, slot in builds.items()}

        evaluate_probe = RegionProbe(timer, "evaluate")
        capture = PopulationCapture(trainer.generations)

        def on_evaluate(call_args, call_kwargs, call_result):
            evaluate_probe(call_args, call_kwargs, call_result)
            capture(call_args, call_kwargs, call_result)

        tell_probe = RegionProbe(timer, "tell")
        restores = [
            wrap_method(type(env), "evaluate", timer, "evaluate", on_call=on_evaluate),
            wrap_function(trainer_module, "run_tell", timer, "tell", on_call=tell_probe),
        ]
        sync()
        run_start = time.perf_counter()
        try:
            summary = trainer.run()
        finally:
            run_end = time.perf_counter()
            for restore in reversed(restores):
                restore()
        loop_s = run_end - run_start
        builds_after_run = {name: int(slot["builds"]) for name, slot in builds.items()}
        records = list(trainer.records)

        entries = []
        generation_start = run_start
        for index, record in enumerate(records):
            evaluate_sample = evaluate_probe.samples[index]
            tell_sample = tell_probe.samples[index]
            generation_s = tell_sample["end"] - generation_start
            entries.append(generation_entry(index, record, evaluate_sample, tell_sample, generation_s))
            generation_start = tell_sample["end"]

        holdout_block = holdout(env, capture.populations, records, seed_table, int(args.master_seed))
        cache_final = {"hits": int(env.cache.hits), "misses": int(env.cache.misses)}

        checks = {
            "determinism_fingerprint": fingerprint(seed_table, entries, holdout_block),
            "fingerprint_payload": (
                "sha256 over seed_table, the per-generation seeds, the per-generation fitness "
                "(6 dp) and the held-out fitness of both sides (6 dp)"
            ),
            "seed_table_matches_seeding": bool(
                np.array_equal(np.asarray(seed_table), generation_seed_table(int(args.master_seed), int(args.generations)))
            ),
            "seed_table_matches_trainer": bool(
                np.array_equal(np.asarray(trainer.seed_table), np.asarray(seed_table))
            ),
            "env_seed_table_restored": bool(
                env.seed_table is not None and np.array_equal(np.asarray(env.seed_table), np.asarray(seed_table))
            ),
            "distinct_seeds_per_generation": bool(
                all(len(set(entry["seeds"])) == len(entry["seeds"]) for entry in entries)
            ),
            "holdout_disjoint": bool(holdout_block["disjoint_from_training"]),
            "no_graph_build_during_run": builds_after_run == builds_before_run,
            "graphs_built_after_setup": sorted(
                name
                for name in set(builds_before_run) | set(builds_after_run)
                if builds_after_run.get(name, 0) != builds_before_run.get(name, 0)
            ),
            "cache_hits": cache_final["hits"],
            "cache_misses": cache_final["misses"],
        }
        trainer.close()
    finally:
        restore_cache()
        if env is not None:
            close_start = time.perf_counter()
            env.close()
            close_s = time.perf_counter() - close_start

    steady = entries[1:]
    eval_seconds = [entry["evaluate_s"] for entry in entries]
    tell_seconds = [entry["tell_s"] for entry in entries]
    generation_seconds = [entry["generation_s"] for entry in entries]
    env_steps_total = sum(entry["env_steps"] for entry in entries)
    steps_total = sum(entry["steps"] for entry in entries)
    evaluate_total = sum(eval_seconds)
    learning_curve = _learning_curve(entries)
    return {
        "task": task,
        "rows": env_info["rows"],
        "pop_size": int(args.pop_size),
        "episodes_per_policy": env_info["episodes_per_policy"],
        "max_episode_length": env_info["max_episode_length"],
        "horizon": int(args.horizon),
        "residency": str(args.residency),
        "backend": env_info["backend"],
        "device_name": env_info["device_name"],
        "perturbation": {
            "base_pos": float(perturbation.base_pos),
            "base_rpy": float(perturbation.base_rpy),
            "joint_pos": float(perturbation.joint_pos),
            "is_zero": bool(perturbation.is_zero),
            "source": "the pilot trainer CLI defaults (go2_neat_train.py / hover_neat_train.py)",
        },
        "genome": genome,
        "seed_table": np.asarray(seed_table, dtype=np.int64).tolist(),
        "generations": entries,
        "totals": {
            "generations": len(entries),
            "loop_s": _r(loop_s),
            "generation_s": _r(sum(generation_seconds)),
            "evaluate_s": _r(evaluate_total),
            "tell_s": _r(sum(tell_seconds)),
            "env_wall_s": _r(sum(entry["env_wall_s"] for entry in entries)),
            "unattributed_s": _r(max(0.0, loop_s - sum(generation_seconds))),
            "steps": int(steps_total),
            "env_steps": int(env_steps_total),
            "env_steps_per_s_loop": _r(env_steps_total / loop_s) if loop_s > 0.0 else None,
            "env_steps_per_s_evaluate": _r(env_steps_total / evaluate_total) if evaluate_total > 0.0 else None,
            "steps_per_s": {"stats": _stats([entry["steps_per_s"] for entry in entries])},
            "env_steps_per_s": {"stats": _stats([entry["env_steps_per_s"] for entry in entries])},
            "evaluate_s_stats": _stats(eval_seconds),
            "tell_s_stats": _stats(tell_seconds),
            "generation_s_stats": _stats(generation_seconds),
            "best_fitness": summary["totals"].get("best_fitness"),
            "final_best_fitness": summary["totals"].get("final_best_fitness"),
            "cache_hits": int(cache_final["hits"]),
            "cache_misses": int(cache_final["misses"]),
            "note": (
                "evaluate_s/tell_s are common.sync()-bracketed wall clock of the two trainer regions; "
                "generation_s is the whole generation (tell end minus the previous tell end, or the "
                "run start) and bookkeeping_s is what is left of it between the regions"
            ),
        },
        "phases": {
            "import_s": _r(import_s),
            "build_s": _r(build_s),
            "graph_build_s": {
                name: {
                    "calls": int(slot["calls"]),
                    "builds": int(slot["builds"]),
                    "build_s": _r(slot["build_s"]),
                    "build_s_per_build": _r(slot["build_s"] / slot["builds"]) if slot["builds"] else None,
                    "lookup_s": _r(slot["lookup_s"]),
                }
                for name, slot in sorted(builds.items())
            },
            "graph_build_total_s": _r(sum(slot["build_s"] for slot in builds.values())),
            "setup_s": _r(setup_s),
            "trainer_build_s": _r(trainer.build_time_s),
            "tell_build_s": _r(trainer.tell_build_time_s),
            "first_generation_s": _r(generation_seconds[0]) if generation_seconds else None,
            "steady_generation_median_s": _r(statistics.median([entry["generation_s"] for entry in steady]))
            if steady
            else None,
            "holdout_s": _r(sum(holdout_block["evaluate_s"].values())),
            "close_s": _r(close_s),
            "totals_s": _r(time.perf_counter() - started),
            "import_note": (
                "the first import of the pilot env module and its kernels: genesis itself was "
                "already imported (and initialised) by the CLI's _ensure_gs_init, and the "
                "evoxc/TensorNEAT stack the trainer adds is imported right after, outside this timer"
            ),
            "graph_build_note": (
                "measured inside ExecutableCache.get while the env and the trainer are built; the "
                "env compiles its own graphs in __init__, the trainer compiles the tell graph in "
                "setup(), and every later generation is a cache lookup"
            ),
            "steady_note": "first_generation_s is generation 0 (initial population); steady is generations 1..",
        },
        "measurements": {
            "learning_curve": learning_curve,
            "holdout": holdout_block,
            "checks": checks,
            "tell": {
                "backend": str(trainer.tell_backend),
                "device": str(trainer.tell_device_name),
                "key": [str(part) for part in trainer.tell_key],
            },
            "config": trainer.summary.get("config", {}),
            "trainer_totals": {
                key: trainer.summary.get("totals", {}).get(key)
                for key in ("wall_time_s", "steps", "env_steps", "best_fitness", "final_best_fitness", "cache_hits", "cache_misses")
            },
        },
        "episode_length_note": EPISODE_LENGTH_NOTE,
        "sync_note": (
            "common.sync() runs before and after each timed region; it calls torch.cuda.synchronize() "
            "only when a HIP/CUDA device is available and is a no-op on a CPU-resident run, where "
            "every timed region is host wall clock"
        ),
    }


def _stats(values) -> dict:
    """Return :func:`common.stats` of ``values`` (``n`` 0 and every field ``None`` when empty)."""
    return stats(values)


def _learning_curve(entries: list) -> dict:
    """Per-generation best/median fitness with cumulative env steps and cumulative loop seconds."""
    cumulative_steps = []
    cumulative_loop = []
    env_steps = 0
    loop_s = 0.0
    for entry in entries:
        env_steps += int(entry["env_steps"])
        loop_s += float(entry["generation_s"])
        cumulative_steps.append(env_steps)
        cumulative_loop.append(_r(loop_s))
    return {
        "generations": [int(entry["generation"]) for entry in entries],
        "best_fitness": [entry["best_fitness"] for entry in entries],
        "median_fitness": [entry["median_fitness"] for entry in entries],
        "cumulative_env_steps": cumulative_steps,
        "cumulative_loop_s": cumulative_loop,
        "x_axes": "cumulative_env_steps and cumulative_loop_s are the two x-axes of the report figure",
    }
