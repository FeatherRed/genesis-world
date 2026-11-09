"""TensorNEAT trainer of the Go2 *walking* task on top of the compiled EvoX Compiler graphs.

This is the entry point of the TensorNEAT/``evoxc`` walking stack.  Every tensor op that
matters -- observation assembly, the genome forward pass, the reward/termination step, the
episode-mean fitness and the NEAT ``tell`` transition -- runs as a compiled graph from
``go2_neat_kernels.py``.  Genesis owns the scene and the physics; this module owns only the
async gym loop, the generation bookkeeping and the artifacts.

What is where
-------------
``examples/locomotion/go2_env.py``   the behaviour being reproduced (env semantics).
``examples/locomotion/go2_train.py`` the untouched PPO control path (``rsl-rl``); read its
                                    ``get_cfgs()`` for the task constants, never import it --
                                    it raises at import time without ``rsl-rl-lib``.
``go2_neat_env.py``                 the batched seeded evaluation env (``Go2NeatEnv``).
``go2_neat_kernels.py``             the five compiled graphs and ``ExecutableCache``.

Manual ask/tell loop
--------------------
``tensorneat.pipeline.Pipeline`` owns a jit-compiled generation loop that calls
``problem.evaluate``; the evaluation here is a simulator, so the pipeline is only used to
supply its *statics*: ``pipe.setup()`` gives the initial population (``state.pop``), the
species pytree (``state.species``), the RNG key (``state.randkey``) and the generation
counter (``state.generation``), and ``pipe._static_config(state)`` gives the tensor-free
config the graphs register their runtime state in.  The problem object passed to the
pipeline is a stub (:class:`~go2_neat_env.TraceOnlyProblem`) whose ``evaluate`` raises:
the fitness always comes from the simulation, never from TensorNEAT.

The loop is therefore, per generation::

    simulate (Go2NeatEnv.evaluate)  ->  fitness (pop_size,)
    run_tell(randkey, pop, generation, fitness, species) -> next randkey/pop/generation/species

``run_tell`` feeds ``state.randkey`` *as-is* (no ``eval_key``/``tell_key`` split), so the
mutation stream is reproducible from ``evolution_seed(master_seed)`` but differs from the
one ``Pipeline.step`` would consume.

Seeds -- two independent streams from one master seed
----------------------------------------------------
``np.random.SeedSequence(master_seed).spawn(2)``:
* the **evolution** stream -> ``evolution_seed(master_seed)`` -> ``Pipeline(seed=...)``;
* the **environment** stream -> ``generation_seed_table(master_seed, ...)``, one row of
  ``EPISODES_PER_POLICY`` distinct reset seeds per generation, prefix-stable.  Every policy
  of a generation shares the generation's seed triple (its three rows use the three seeds),
  so a generation compares policies on the same three initial states.

Compile once, reuse
-------------------
All five executables go through one ``ExecutableCache`` (``env.cache``): the four
``Go2NeatEnv`` graphs are built in its ``__init__`` and the ``tell`` graph here.  Each
generation calls ``env.executables()`` and the ``tell`` lookup again, which must report
hits -- that is the per-generation ``cache`` field of the log line.

Backends
--------
CPU/``numpy`` is the default. ``-g/--gpu`` runs Genesis on ``gs.gpu`` and the rollout and
evolution graphs with IREE on ``cuda:0``. ``--tell-backend numpy`` explicitly runs
evolution on CPU. On ROCm (``--evoxc-device rocm:0``) the rollout runs on IREE and the
evolution defaults to XLA on ``rocm:0``, with an explicit ``--pjrt-plugin`` (or the
``EVOXC_PJRT_PLUGIN`` environment variable) selecting the PJRT plugin, because IREE's ROCm
compile of the ``tell`` graph does not terminate.

Artifacts (git-ignored, never committed)
----------------------------------------
``<output-dir>/history.jsonl``  one JSON object per generation.
``<output-dir>/summary.json``   resolved config + every generation record + totals.
``-o`` defaults to ``out/go2_neat``; ``-e/--exp-name`` appends a subdirectory.

Under ``PYTEST_VERSION`` the run shrinks to one generation of 20 steps unless the matching
clicks are given explicitly, so a future smoke test cannot hang.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import time

import numpy as np

import evoxc
from tensorneat import Pipeline, algorithm, genome
from tensorneat.common.tools import to_numpy

import genesis as gs

from go2_neat_env import (
    MAX_CONNS,
    MAX_NODES,
    SPECIES_SIZE,
    Go2NeatEnv,
    TraceOnlyProblem,
    evolution_seed,
    generation_seed_table,
)
from go2_neat_kernels import (
    EPISODES_PER_POLICY,
    NUM_ACTIONS,
    NUM_OBS,
    Perturbation,
    build_tell,
    run_tell,
)

logger = logging.getLogger(__name__)


# --- Genome pipeline ----------------------------------------------------------------


def build_genome_pipeline(pop_size, master_seed):
    """Create the NEAT algorithm and the static graph config of a ``Pipeline.setup()``.

    The problem passed to the pipeline is never evaluated (see the module docstring): the
    pipeline exists to produce the initial population, the species pytree and the statics.

    Args:
        pop_size (int): Population size, i.e. policies per generation.
        master_seed (int): Seed of the run; the pipeline uses the evolution stream.

    Returns:
        tuple: ``(algo, config, state)`` -- the ``NEAT`` instance, the tensor-free
        ``_static_config`` handed to every compiled graph, and the initial pipeline state.
    """
    gen = genome.DefaultGenome(
        num_inputs=NUM_OBS,
        num_outputs=NUM_ACTIONS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
    )
    algo = algorithm.NEAT(gen, pop_size=pop_size, species_size=min(SPECIES_SIZE, pop_size))
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(),
        seed=evolution_seed(master_seed),
        backend="numpy",
        is_detailed=False,
    )
    state = pipeline.setup()
    return algo, pipeline._static_config(state), state


def pop_arrays_of(state):
    """Return the four genome payload arrays of a pipeline state as host numpy."""
    return tuple(np.array(to_numpy(leaf)) for leaf in state.pop)


def species_leaves_of(species):
    """Return the species pytree leaves in ``evoxc`` pytree order."""
    return tuple(evoxc.tree_leaves(species))


# --- Artifacts ----------------------------------------------------------------------


def _jsonable(value):
    """Convert numpy scalars/arrays (and containers of them) to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_generation(record):
    """One compact log line for a generation record."""
    ends = record["end_reasons"]
    best_index = int(np.argmax(record["fitness"]))
    best_episodes = " ".join(f"{value:+.3f}" for value in record["episode_returns"][best_index])
    return (
        f"gen {record['generation']} | seeds {record['seeds']} | "
        f"best {record['best_fitness']:+.4f} median {record['median_fitness']:+.4f} | "
        f"best p{best_index} [{best_episodes}] | "
        f"{record['steps']} steps {record['wall_time_s']:.3f} s "
        f"({record['steps_per_s']:.1f} steps/s, {record['env_steps_per_s']:.0f} env-steps/s) | "
        f"cache +{record['cache_hits_delta']} hits/+{record['cache_misses_delta']} misses "
        f"(total {record['cache_hits']} hits/{record['cache_misses']} misses) | "
        f"ends timeout={ends['timeout']} tilted={ends['tilted']} error={ends['error']} "
        f"unfinished={ends['unfinished']}"
    )


# --- CLI ----------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a TensorNEAT Go2 walking population with Genesis and EvoX Compiler graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  python examples/locomotion/go2_neat_train.py --pop-size 8 --generations 3 --horizon 25\n"
            "  python examples/locomotion/go2_neat_train.py --pop-size 8 --generations 3\n"
            "  python examples/locomotion/go2_neat_train.py --pop-size 8 --generations 3 -g\n"
            "  python examples/locomotion/go2_neat_train.py -g --evoxc-device rocm:0  # AMD recipe\n"
        ),
    )
    parser.add_argument("--pop-size", type=int, default=8, help="Policies per generation (rows = 3 * pop_size).")
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Generations to run; default 2 (1 under PYTEST_VERSION).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="scene.step() calls per generation; default the full episode (episode_length_s / dt).",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=20.0,
        help="Episode cap in seconds; it fixes the max_episode_length of the advance graph.",
    )
    parser.add_argument("--master-seed", type=int, default=0, help="Seed of both RNG streams.")
    parser.add_argument("--perturb-base-pos", type=float, default=0.05, help="Max |base position| reset offset [m].")
    parser.add_argument("--perturb-base-rpy", type=float, default=0.05, help="Max |roll/pitch/yaw| reset offset [rad].")
    parser.add_argument("--perturb-joint-pos", type=float, default=0.05, help="Max |joint| reset offset [rad].")
    parser.add_argument(
        "--no-perturbation",
        action="store_true",
        help="Zero all three reset offsets, reproducing the fixed reset of go2_env.",
    )
    parser.add_argument(
        "--stop-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a generation as soon as every row has ended (disable to force the full horizon).",
    )
    parser.add_argument("-e", "--exp-name", type=str, default=None, help="Experiment name, a subdirectory of -o.")
    parser.add_argument("-o", "--output-dir", type=str, default="out/go2_neat", help="Directory of the artifacts.")
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Init Genesis on gs.gpu and default the EvoX Compiler graphs to IREE on cuda:0.",
    )
    parser.add_argument("--evoxc-backend", type=str, default=None, help="Rollout backend override (numpy or iree).")
    parser.add_argument("--evoxc-device", type=str, default=None, help="Rollout device override, e.g. cuda:0 or rocm:0.")
    parser.add_argument(
        "--tell-backend",
        choices=("numpy", "iree", "xla"),
        help=(
            "Evolution backend override. Defaults to the rollout backend and device, except on "
            "ROCm where it defaults to xla. numpy = CPU; iree = IREE; xla = PJRT/XLA."
        ),
    )
    parser.add_argument(
        "--pjrt-plugin",
        type=str,
        default=None,
        help=(
            "Path to the XLA/PJRT plugin .so used by the xla tell backend; when omitted evoxc's "
            "own discovery applies (EVOXC_PJRT_PLUGIN or the installed jax_plugins/*/*.so)."
        ),
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Open the Genesis viewer.")
    parser.add_argument("--quiet", action="store_true", help="Write the artifacts without per-generation lines.")
    return parser.parse_args(argv)


# --- Main loop ----------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pytest_mode = "PYTEST_VERSION" in os.environ
    generations = args.generations if args.generations is not None else (1 if pytest_mode else 2)
    if generations < 1:
        raise ValueError(f"--generations must be positive, got {generations}")
    horizon = args.horizon if args.horizon is not None else (20 if pytest_mode else None)
    if horizon is not None and horizon < 1:
        raise ValueError(f"--horizon must be positive, got {horizon}")
    if args.pop_size < 1:
        raise ValueError(f"--pop-size must be positive, got {args.pop_size}")

    perturbation = (
        Perturbation()
        if args.no_perturbation
        else Perturbation(
            base_pos=args.perturb_base_pos,
            base_rpy=args.perturb_base_rpy,
            joint_pos=args.perturb_joint_pos,
        )
    )
    backend = args.evoxc_backend or ("iree" if args.gpu else "numpy")
    device = args.evoxc_device or ("cuda:0" if args.gpu and backend != "numpy" else None)
    show_viewer = bool(args.vis) and not pytest_mode

    if not gs._initialized:
        if args.gpu:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)
        else:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    output_dir = Path(args.output_dir)
    if args.exp_name:
        output_dir = output_dir / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    summary_path = output_dir / "summary.json"

    build_start = time.perf_counter()
    env = Go2NeatEnv(
        pop_size=args.pop_size,
        master_seed=args.master_seed,
        perturbation=perturbation,
        horizon=horizon,
        backend=backend,
        device=device,
        show_viewer=show_viewer,
        episode_length_s=args.episode_length_s,
        stop_when_done=args.stop_when_done,
    )
    algo, config, state = build_genome_pipeline(args.pop_size, args.master_seed)
    pop_arrays = pop_arrays_of(state)
    species_leaves = species_leaves_of(state.species)
    randkey = state.randkey
    generation_counter = state.generation
    if args.tell_backend:
        tell_backend = args.tell_backend
    elif env.backend == "numpy":
        tell_backend = "numpy"
    elif env.device.kind in ("rocm", "hip"):
        tell_backend = "xla"  # IREE/ROCm tell compilation does not terminate
    else:
        tell_backend = env.backend
    tell_device = evoxc.Device("cpu", 0) if tell_backend == "numpy" else env.device
    tell_device_name = f"{tell_device.kind}:{tell_device.index}"
    tell_key = ("tell", args.pop_size, tell_backend, tell_device_name, args.pjrt_plugin)

    def tell_executable():
        """Look the tell executable up in the shared cache (compiled on the first call)."""
        return env.cache.get(
            tell_key,
            lambda: build_tell(
                algo,
                config,
                pop_arrays,
                species_leaves,
                args.pop_size,
                backend=tell_backend,
                device=tell_device,
                plugin_path=args.pjrt_plugin,
            ),
        )

    logger.info("Building TensorNEAT tell on %s@%s", tell_backend, tell_device_name)
    tell_build_start = time.perf_counter()
    tell_executable()
    tell_build_time = time.perf_counter() - tell_build_start
    build_time = time.perf_counter() - build_start

    if not args.quiet:
        print(
            f"Go2 NEAT walking: pop_size={args.pop_size} rows={env.rows} generations={generations} "
            f"horizon={env.horizon} max_episode_length={env.max_episode_length} "
            f"genesis={'gpu' if args.gpu else 'cpu'} evoxc={env.backend}@{env.device_name} "
            f"tell={tell_backend}@{tell_device_name} "
            f"master_seed={args.master_seed} evolution_seed={evolution_seed(args.master_seed)} "
            f"perturbation={'zero' if perturbation.is_zero else 'nonzero'}"
        )
        print(
            f"built 5 executables in {build_time:.3f} s (tell {tell_build_time:.3f} s) "
            f"(cache {env.cache.hits} hits / {env.cache.misses} misses)"
        )

    prev_hits, prev_misses = env.cache.hits, env.cache.misses
    records = []
    run_start = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history:
        for generation_index in range(generations):
            env.executables()  # four cache hits: the env graphs are compiled only once
            tell_executable()  # one more cache hit
            result = env.evaluate(pop_arrays, generation_index)
            fitness = np.asarray(result["fitness"], dtype=np.float32)
            tell = run_tell(
                tell_executable(),
                randkey=randkey,
                pop_arrays=pop_arrays,
                generation=generation_counter,
                fitness=fitness,
                species_leaves=species_leaves,
            )
            hits, misses = env.cache.hits, env.cache.misses
            record = {
                "generation": generation_index,
                "seeds": [int(seed) for seed in result["seeds"]],
                "episode_returns": result["episode_returns"].tolist(),
                "returns": result["returns"].tolist(),
                "fitness": fitness.tolist(),
                "best_fitness": float(fitness.max()),
                "median_fitness": float(np.median(fitness)),
                "wall_time_s": float(result["wall_time_s"]),
                "steps": int(result["steps"]),
                "env_steps": int(result["env_steps"]),
                "steps_per_s": float(result["steps_per_s"]),
                "env_steps_per_s": float(result["env_steps_per_s"]),
                "cache_hits": int(hits),
                "cache_misses": int(misses),
                "cache_hits_delta": int(hits - prev_hits),
                "cache_misses_delta": int(misses - prev_misses),
                "end_reasons": {name: int(count) for name, count in result["end_reasons"].items()},
                "init_qpos": result["init_qpos"].tolist(),
            }
            records.append(record)
            history.write(json.dumps(_jsonable(record)) + "\n")
            history.flush()
            if not args.quiet:
                print(_format_generation(record))
            prev_hits, prev_misses = hits, misses
            randkey = tell["randkey"]
            generation_counter = tell["generation"]
            pop_arrays = tuple(np.array(leaf, copy=True) for leaf in tell["pop"])
            species_leaves = species_leaves_of(tell["species"])
    wall_time_s = time.perf_counter() - run_start

    fitness_history = [record["best_fitness"] for record in records]
    summary = {
        "config": {
            "pop_size": args.pop_size,
            "rows": env.rows,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "generations": generations,
            "horizon": int(env.horizon),
            "max_episode_length": int(env.max_episode_length),
            "episode_length_s": float(args.episode_length_s),
            "master_seed": args.master_seed,
            "evolution_seed": evolution_seed(args.master_seed),
            "seed_table": generation_seed_table(args.master_seed, generations).tolist(),
            "perturbation": {
                "base_pos": float(perturbation.base_pos),
                "base_rpy": float(perturbation.base_rpy),
                "joint_pos": float(perturbation.joint_pos),
                "is_zero": bool(perturbation.is_zero),
            },
            "genesis_backend": "gpu" if args.gpu else "cpu",
            "evoxc_backend": env.backend,
            "evoxc_device": env.device_name,
            "tell_backend": tell_backend,
            "tell_device": tell_device_name,
            "show_viewer": show_viewer,
            "stop_when_done": bool(args.stop_when_done),
            "genome": {
                "num_inputs": NUM_OBS,
                "num_outputs": NUM_ACTIONS,
                "max_nodes": MAX_NODES,
                "max_conns": MAX_CONNS,
                "species_size": min(SPECIES_SIZE, args.pop_size),
            },
            "executable_build_time_s": build_time,
            "tell_build_time_s": tell_build_time,
        },
        "generations": records,
        "totals": {
            "wall_time_s": wall_time_s,
            "steps": sum(record["steps"] for record in records),
            "env_steps": sum(record["env_steps"] for record in records),
            "best_fitness": max(fitness_history) if fitness_history else None,
            "final_best_fitness": fitness_history[-1] if fitness_history else None,
            "fitness_history": fitness_history,
            "cache_hits": int(env.cache.hits),
            "cache_misses": int(env.cache.misses),
        },
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")

    env.close()
    if not args.quiet:
        best = summary["totals"]["best_fitness"]
        print(
            f"done: {generations} generations, {summary['totals']['steps']} steps, "
            f"{wall_time_s:.1f} s total, best fitness {best:+.4f}; "
            f"wrote {history_path} and {summary_path}"
        )
    if tell_backend == "xla" and tell_device.kind != "cpu":
        # A live PJRT client makes the ROCm plugin abort at interpreter exit (after the work
        # is done, so the artifacts above are already on disk); skip the destructor.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
