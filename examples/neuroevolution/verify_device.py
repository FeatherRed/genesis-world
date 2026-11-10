"""Hardware verification and measurement of the neuroevolution layer's device residency.

Device residency (``residency="device"``) is a claim about placement: the four genome payload leaves, the ten ``SpeciesState`` leaves, the RNG key and the generation counter live on one ``evoxc`` device, the fitness and the verification of it are computed there, and the tell graph's output stays there too.
This CLI is the executable form of that claim on a real accelerator: it builds the simulator-free :class:`device_env.SyntheticDeviceEnv` at every requested population size, drives :class:`trainer.NeuroevolutionTrainer` device-resident, asserts the placement of every runtime leaf, and compares the run against the numpy residency and against ``Pipeline.setup()``.

Run it from the repository root::

    PYTHONPATH=$PWD/examples/neuroevolution /root/evox-robotics/.venv/bin/python examples/neuroevolution/verify_device.py

The default target is ``iree`` on the first non-CPU ``evoxc`` device (``EVOXC_BACKEND`` / ``EVOXC_DEVICE`` override it, ``--backend`` / ``--device`` override those).
``iree`` is the default because its fitness rule is bit-identical to the host rule; an ``xla`` target needs its PJRT plugin visible to the options-less discovery chain (``EVOXC_PJRT_PLUGIN`` or a ``jax_plugins/*/*.so`` on ``sys.path``) and :func:`ensure_pjrt_plugin` sets the variable when it can find the library.
A host without a non-CPU device prints one ``SKIP`` line and exits 0, so a bare invocation stays harmless in CI; ``--allow-cpu`` runs the same checks in the plumbing-only ``cpu:0`` mode.

What is measured and printed per population size:

* the ``setup()`` breakdown -- total, ``tell`` build, fitness-rule build, verification build;
* the per-generation wall clock (from one ``env.evaluate`` entry to the next, so the tell graph and the record bookkeeping are inside it), its median, and the resulting steps/s and env-steps/s;
* the placement evidence of the initialization, of the fitness, of the tell output and of the next generation's payload;
* the parity against the numpy residency (fitness history exactly, the record's float leaves within ``--tolerance``) and against ``Pipeline(...).setup()`` (integer leaves, key and generation exact, float leaves within ``--tolerance``).

Both parities use ``--tolerance`` as one absolute bound, except that the record leaves are compared with ``numpy.allclose``'s convention (``atol = rtol = --tolerance``): at the Go2 return magnitude of ~100 a 1e-5 absolute bound is barely one ulp, and the device init drift of ~5e-07 already moves a generation's returns by ~1.5e-05 by way of the genome.

``--json PATH`` writes the same numbers as one record of plain Python types.
Nothing here imports ``torch``, ``genesis`` or ``tensordict``: the imports are ``numpy``, ``evoxc``, ``TensorNEAT`` and the sibling modules of this directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import evoxc

from tensorneat import Pipeline, algorithm, genome

from device_env import SyntheticDeviceEnv
from device_runtime import assert_payload_resident
from evox_runtime import _to_numpy, resolve_backend_device
from seeding import evolution_seed
from trainer import NeuroevolutionTrainer, TraceOnlyProblem, pop_arrays_of, species_leaves_of

__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_POPS",
    "Reporter",
    "ensure_pjrt_plugin",
    "float_max_abs_diff",
    "main",
    "measure_pop",
    "resolve_target",
]

#: Schema tag of the ``--json`` record.
SCHEMA = "neuroevolution-device-verify/1"

#: Backend a target is measured on unless ``--backend``/``EVOXC_BACKEND`` name another one.
DEFAULT_BACKEND = "iree"

#: Population sizes measured unless ``--pops`` says otherwise (the Go2 pilot's pop and a 4x batch).
DEFAULT_POPS = (8, 32)

#: Generations per population size, the shortest loop that exercises a tell transition and a replay.
DEFAULT_GENERATIONS = 2

#: Episodes per policy, episodes per generation.
DEFAULT_EPISODES = 3

#: Genome widths of the Go2 pilot, the widest task the layer serves (45 -> 12).
DEFAULT_NUM_INPUTS = 45
DEFAULT_NUM_OUTPUTS = 12

#: Master seed of every run; the evolution and env streams are both derived from it.
DEFAULT_MASTER_SEED = 0

#: Fitness tolerance of the parities (the device verification's own ``atol``).
DEFAULT_TOLERANCE = 1e-5

#: Genome budgets and species size the trainers build (``trainer.NeuroevolutionTrainer`` defaults).
MAX_NODES = 80
MAX_CONNS = 640
SPECIES_SIZE = 3

#: Prefix of the line printed when the host offers nothing to verify.
SKIP_PREFIX = "SKIP device residency:"


# --- xla plugin discovery ------------------------------------------------------------


def ensure_pjrt_plugin(backend):
    """Make the ``xla`` adapter's PJRT plugin discoverable through the environment.

    The xla adapter finds its plugin through an options-less chain only -- ``EVOXC_PJRT_PLUGIN``
    or a ``jax_plugins/*/*.so`` under a ``sys.path`` entry -- and the placement provider behind
    ``Tensor.to(rocm:0)`` never sees a compile option, so device-first initialization on
    ``xla@<accelerator>`` needs the variable even when the tell graph is built with
    ``plugin_path=``.

    Args:
        backend (str): ``evoxc`` backend name; the lookup only happens for ``"xla"``.

    Returns:
        str | None: The plugin library the environment now names, or ``None`` when the backend
        needs none or none was found.
    """
    if backend != "xla":
        return None
    existing = os.environ.get("EVOXC_PJRT_PLUGIN")
    if existing:
        return existing
    for entry in sys.path:
        if not entry:
            continue
        for candidate in sorted(Path(entry).glob("jax_plugins/*/*.so")):
            os.environ["EVOXC_PJRT_PLUGIN"] = str(candidate)
            return str(candidate)
    return None


# --- Numeric comparison -------------------------------------------------------------


def float_max_abs_diff(left, right):
    """Largest finite absolute difference of two float arrays; ``inf`` on a shape mismatch.

    The NaN padding of an unused genome slot is ignored, and two arrays that only differ there
    report ``0.0``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    with np.errstate(invalid="ignore"):
        # The NaN padding of an unused genome slot makes ``inf - inf``; those entries are skipped below.
        differences = np.abs(left - right)
    differences = differences[np.isfinite(differences)]
    return float(differences.max()) if differences.size else 0.0


def _array_equal(left, right):
    """Exact equality of two arrays, treating the NaN padding of an unused genome slot as equal."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left, right, equal_nan=True))


def _plain(value):
    """Convert numpy scalars/arrays (and containers of them) to plain JSON types."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# --- Reporting ----------------------------------------------------------------------


class Reporter:
    """Collects the ``PASS``/``FAIL`` lines of a run; :func:`main` exits non-zero if any failed."""

    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail):
        """Record and print one assertion.

        Args:
            name (str): Short name of the assertion.
            ok (bool): Its outcome.
            detail (str): The evidence, printed on the same line.

        Returns:
            bool: ``ok``, so a caller can branch on the assertion.
        """
        self.checks.append({"name": str(name), "ok": bool(ok), "detail": str(detail)})
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
        return bool(ok)

    @property
    def ok(self):
        """True when every recorded assertion passed."""
        return all(check["ok"] for check in self.checks)


def _flatten_placement(placement):
    """Flatten a trainer placement mapping into ``name -> device string`` (its leaf lists included)."""
    flat = {}
    for key, value in placement.items():
        if isinstance(value, (list, tuple)):
            flat.update({f"{key}[{index}]": item for index, item in enumerate(value)})
        else:
            flat[key] = value
    return flat


def check_placement(reporter, name, placement, expected, keys=None):
    """Report one assertion that every entry of a placement mapping is placed on ``expected``.

    Args:
        reporter (Reporter): The run's reporter.
        name (str): Short name of the assertion.
        placement (Mapping): A trainer placement mapping (leaf lists included).
        expected (str): The expected ``"kind:index"`` placement.
        keys (Iterable[str] | None): Restrict the assertion to these top-level keys.

    Returns:
        bool: True when every checked entry matches.
    """
    flat = _flatten_placement(placement)
    if keys is not None:
        wanted = {str(key) for key in keys}
        flat = {key: value for key, value in flat.items() if key.split("[")[0] in wanted}
    wrong = {key: value for key, value in flat.items() if value != expected}
    if wrong:
        return reporter.check(
            name,
            False,
            f"{len(wrong)}/{len(flat)} entries are not on {expected}: {dict(sorted(wrong.items())[:4])}",
        )
    return reporter.check(name, True, f"{len(flat)} entries on {expected}")


# --- Environment wrapper ------------------------------------------------------------


class _TimedEnv:
    """Delegate to the env and timestamp every ``evaluate`` entry, which brackets one generation."""

    def __init__(self, env):
        self.env = env
        self.starts = []

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def evaluate(self, pop_arrays, generation):
        """Record the generation's start and run it on the wrapped env."""
        self.starts.append(time.perf_counter())
        return self.env.evaluate(pop_arrays, generation)

    def close(self):
        """Close the wrapped env."""
        self.env.close()


# --- Parity -------------------------------------------------------------------------


def initialization_parity(reporter, initial, pop_size, args, tolerance):
    """Compare the device-first initialization against ``Pipeline(...).setup()`` on the host.

    ``device_runtime.device_setup`` must build the generation-0 state TensorNEAT's own
    initialization graph builds on a numpy pipeline: the integer leaves, the RNG key and the
    generation counter bit for bit, and the float leaves within ``tolerance``. The numpy
    backend is bitwise identical to the host and ``iree@rocm:0`` was measured at 4.8e-07, so a
    ``tolerance`` of 1e-5 has room for the backend's own reduction order.

    Args:
        reporter (Reporter): The run's reporter.
        initial (Mapping): ``pop``/``species``/``randkey``/``generation`` of the device state.
        pop_size (int): Population size of the state.
        args (argparse.Namespace): The CLI arguments (genome widths and master seed).
        tolerance (float): Absolute tolerance of the float leaves.

    Returns:
        dict: The JSON-able evidence block of the comparison.
    """
    algo = algorithm.NEAT(
        genome.DefaultGenome(
            num_inputs=args.num_inputs,
            num_outputs=args.num_outputs,
            max_nodes=MAX_NODES,
            max_conns=MAX_CONNS,
        ),
        pop_size=pop_size,
        species_size=min(SPECIES_SIZE, pop_size),
    )
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(args.num_inputs, args.num_outputs),
        seed=evolution_seed(args.master_seed),
        backend="numpy",
        is_detailed=False,
    )
    host_state = pipeline.setup()
    groups = (
        ("payload", list(zip(pop_arrays_of(host_state), initial["pop"], strict=True))),
        (
            "species",
            list(
                zip(
                    [_to_numpy(leaf) for leaf in species_leaves_of(host_state.species)], initial["species"], strict=True
                )
            ),
        ),
    )
    problems = []
    float_diff = 0.0
    integer_leaves = 0
    integer_exact = True
    for group_name, pairs in groups:
        for index, (host_leaf, device_leaf) in enumerate(pairs):
            host_leaf = np.asarray(host_leaf)
            device_leaf = np.asarray(_to_numpy(device_leaf))
            if host_leaf.shape != device_leaf.shape:
                problems.append(f"{group_name} leaf {index} shape {host_leaf.shape} != {device_leaf.shape}")
            elif np.issubdtype(host_leaf.dtype, np.integer):
                integer_leaves += 1
                if not np.array_equal(host_leaf, device_leaf):
                    integer_exact = False
                    problems.append(f"{group_name} leaf {index} (integer) differs from Pipeline.setup()")
            else:
                diff = float_max_abs_diff(host_leaf, device_leaf)
                float_diff = max(float_diff, diff)
                if diff > tolerance:
                    problems.append(
                        f"{group_name} leaf {index} differs by {diff:.3e} from Pipeline.setup(), "
                        f"more than {tolerance:.1e}"
                    )
                elif not _array_equal(host_leaf, device_leaf) and diff == 0.0:
                    problems.append(f"{group_name} leaf {index} differs in more than finite values")
    randkey_exact = int(_to_numpy(host_state.randkey)) == int(_to_numpy(initial["randkey"]))
    if not randkey_exact:
        problems.append(f"randkey {int(_to_numpy(initial['randkey']))} != {int(_to_numpy(host_state.randkey))}")
    generation_exact = float(_to_numpy(host_state.generation)) == float(_to_numpy(initial["generation"]))
    if not generation_exact:
        problems.append(f"generation {float(_to_numpy(initial['generation']))} != 0")
    evidence = {
        "integer_leaves": integer_leaves,
        "integer_leaves_exact": integer_exact,
        "float_max_abs_diff": float_diff,
        "randkey_exact": randkey_exact,
        "generation_exact": generation_exact,
        "tolerance": float(tolerance),
    }
    detail = (
        f"{integer_leaves} integer leaves exact, float leaves max |diff| {float_diff:.3e} <= {tolerance:.1e}, "
        f"randkey {'exact' if randkey_exact else 'DIFFERS'}, generation {'exact' if generation_exact else 'DIFFERS'}"
    )
    if problems:
        detail = "; ".join(problems[:4])
    reporter.check("parity/initialization", not problems, detail)
    return evidence


def residency_parity(reporter, trainer, pop_size, args, tolerance):
    """Run the numpy residency of the same env and compare it against the device run.

    The fitness history must be identical to the bit (the device fitness is the same
    aggregation of the same episode returns), and the record's float leaves are compared with
    ``atol = rtol = tolerance``: the device init drift of ~5e-07 moves a later generation's
    genome, and with it its returns, by more than a 1e-5 absolute bound at the Go2 magnitude.

    Args:
        reporter (Reporter): The run's reporter.
        trainer (NeuroevolutionTrainer): The finished device-resident trainer.
        pop_size (int): Population size of both runs.
        args (argparse.Namespace): The CLI arguments.
        tolerance (float): Absolute tolerance of the record's float leaves.

    Returns:
        dict: The JSON-able evidence block of the comparison.
    """
    reference_env = SyntheticDeviceEnv(
        pop_size,
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        episodes=args.episodes,
        residency="numpy",
        backend="numpy",
        device="cpu:0",
        seed=args.master_seed,
    )
    reference = NeuroevolutionTrainer(
        reference_env,
        pop_size,
        args.generations,
        master_seed=args.master_seed,
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        species_size=SPECIES_SIZE,
        residency="numpy",
    )
    try:
        reference.run()
        same_history = [float(value) for value in trainer.fitness_history] == [
            float(value) for value in reference.fitness_history
        ]
        reporter.check(
            "parity/fitness_history",
            same_history,
            "identical to the numpy residency"
            if same_history
            else f"device {trainer.fitness_history} != numpy {reference.fitness_history}",
        )
        differences = {}
        within_tolerance = True
        for key in ("fitness", "episode_returns"):
            device_values = np.asarray([record[key] for record in trainer.records], dtype=np.float64)
            numpy_values = np.asarray([record[key] for record in reference.records], dtype=np.float64)
            ok = bool(np.allclose(device_values, numpy_values, atol=tolerance, rtol=tolerance))
            within_tolerance = within_tolerance and ok
            differences[key] = {
                "max_abs_diff": float_max_abs_diff(device_values, numpy_values),
                "max_abs_value": float(np.max(np.abs(numpy_values))) if numpy_values.size else 0.0,
                "within_tolerance": ok,
            }
        reporter.check(
            "parity/records",
            within_tolerance,
            f"max |device - numpy| fitness {differences['fitness']['max_abs_diff']:.3e}, "
            f"episode_returns {differences['episode_returns']['max_abs_diff']:.3e} "
            f"(values up to {differences['episode_returns']['max_abs_value']:.3g}) "
            f"within atol=rtol={tolerance:.1e}",
        )
        return {
            "fitness_history_identical": same_history,
            "numpy_fitness_history": [float(value) for value in reference.fitness_history],
            "record_differences": differences,
            "tolerance": float(tolerance),
        }
    finally:
        reference.close()
        reference_env.close()


# --- Target resolution --------------------------------------------------------------


def reachable_devices(devices, backend):
    """Split the non-CPU devices into the ones ``backend`` actually reaches and the ones it does not.

    ``evoxc.devices()`` enumerates every runtime the wheel was built with, so a ROCm-only host can
    still list ``cuda:0``; the only honest test of reachability is to move a tensor there, which is
    what this probe does -- one float32 element per device, through the backend's own placement
    provider (re-selected first, since the provider is per backend).

    Args:
        devices (Iterable[evoxc.Device]): The devices ``evoxc.devices()`` reported.
        backend (str): The ``evoxc`` backend the run uses.

    Returns:
        tuple: ``(usable, skipped)`` -- the reachable non-CPU devices in enumeration order, and one
        ``{"device", "error"}`` entry per unreachable device.
    """
    evoxc.backends.get(backend)
    usable = []
    skipped = []
    for device in devices:
        if device.kind == "cpu":
            continue
        name = f"{device.kind}:{device.index}"
        try:
            evoxc.from_numpy(np.zeros(1, dtype=np.float32)).to(device)
        except Exception as exc:  # noqa: BLE001 - an unreachable runtime is an expected outcome to report
            skipped.append({"device": name, "error": f"{type(exc).__name__}: {exc}".splitlines()[0]})
            continue
        usable.append(device)
    return usable, skipped


def resolve_target(args):
    """Resolve the backend and device to measure, or ``None`` when the host offers nothing to verify.

    Args:
        args (argparse.Namespace): The CLI arguments.

    Returns:
        dict | None: ``backend``, ``device``, ``device_name``, ``plugin_path``, ``available`` and
        ``skipped``, or ``None`` when the run should stop with exit code 0.
    """
    devices = evoxc.devices()
    available = [f"{device.kind}:{device.index}" for device in devices]
    explicit_backend = args.backend or os.environ.get("EVOXC_BACKEND") or ""
    backend = explicit_backend or DEFAULT_BACKEND
    plugin_path = ensure_pjrt_plugin(backend)
    if backend == "xla" and plugin_path is None:
        print(
            "xla adapter: no PJRT plugin is discoverable; set EVOXC_PJRT_PLUGIN to the "
            "xla_<kind>_plugin.so or put its jax_plugins/ directory on PYTHONPATH "
            "(the xla adapter's placement provider reads no compile options)"
        )
        raise SystemExit(2)
    usable, skipped = reachable_devices(devices, backend)
    device_spec = args.device or os.environ.get("EVOXC_DEVICE") or ""
    if device_spec:
        device = resolve_backend_device(backend, device_spec)[1]
    elif usable:
        device = usable[0]
    else:
        device = None
    if device is None or device.kind == "cpu":
        if device is not None:
            print(f"note: {device_spec} names a cpu-kind device; measuring the plumbing-only mode")
        elif not args.allow_cpu:
            reasons = "; ".join(f"{entry['device']} unreachable ({entry['error']})" for entry in skipped)
            print(
                f"{SKIP_PREFIX} no non-CPU evoxc device is reachable through {backend} "
                f"(available: {available}{'; ' + reasons if reasons else ''}); "
                "pass --allow-cpu to measure the plumbing-only cpu:0 mode"
            )
            return None
        else:
            print("note: --allow-cpu: measuring the plumbing-only cpu:0 mode")
        device = evoxc.Device("cpu", 0)
        backend = explicit_backend or "numpy"
    backend, device = resolve_backend_device(backend, device)
    return {
        "backend": backend,
        "device": device,
        "device_name": f"{device.kind}:{device.index}",
        "plugin_path": plugin_path,
        "available": available,
        "skipped": skipped,
    }


# --- One population size ------------------------------------------------------------


def measure_pop(pop_size, target, args, reporter):
    """Build, verify and measure the device residency at one population size.

    Args:
        pop_size (int): Policies per generation.
        target (dict): The resolved target (see :func:`resolve_target`).
        args (argparse.Namespace): The CLI arguments.
        reporter (Reporter): The run's reporter.

    Returns:
        dict: The JSON-able measurement of this population size.
    """
    device = target["device"]
    device_name = target["device_name"]
    tolerance = float(args.tolerance)
    print(f"pop {pop_size}")
    started = time.perf_counter()
    env = SyntheticDeviceEnv(
        pop_size,
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        episodes=args.episodes,
        backend=target["backend"],
        device=device,
        residency="device",
        seed=args.master_seed,
    )
    timed = _TimedEnv(env)
    trainer = NeuroevolutionTrainer(
        timed,
        pop_size,
        args.generations,
        master_seed=args.master_seed,
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        species_size=SPECIES_SIZE,
        residency="device",
        plugin_path=target["plugin_path"],
    )
    try:
        trainer.setup()
        setup_s = time.perf_counter() - started
        initial = {
            "pop": tuple(trainer.pop_arrays),
            "species": tuple(trainer.species_leaves),
            "randkey": trainer.randkey,
            "generation": trainer.generation_counter,
        }
        assert_payload_resident(initial["pop"], initial["species"], initial["randkey"], initial["generation"], device)
        check_placement(
            reporter,
            "initialization",
            dict(trainer.placement, pop=list(trainer.placement["pop"]), species=list(trainer.placement["species"])),
            device_name,
            keys=("pop", "species", "randkey", "generation"),
        )

        run_start = time.perf_counter()
        trainer.run()
        run_s = time.perf_counter() - run_start
        starts = timed.starts
        generation_s = [later - earlier for earlier, later in itertools.pairwise(starts)]
        if starts:
            generation_s.append(time.perf_counter() - starts[-1])

        assert_payload_resident(
            trainer.pop_arrays, trainer.species_leaves, trainer.randkey, trainer.generation_counter, device
        )
        placement = trainer.placement
        check_placement(
            reporter,
            "fitness",
            placement,
            device_name,
            keys=("fitness", "episode_returns", "returns", "seeds"),
        )
        check_placement(reporter, "tell", placement, device_name, keys=("randkey", "generation"))
        check_placement(reporter, "next-generation payload", placement, device_name, keys=("pop", "species"))

        steps = [int(record["steps"]) for record in trainer.records]
        env_steps = [int(record["env_steps"]) for record in trainer.records]
        steady = generation_s[1:] if len(generation_s) > 1 else generation_s
        median_s = statistics.median(steady) if steady else 0.0
        steps_per_s = (
            statistics.median([step / seconds for step, seconds in zip(steps, generation_s, strict=True)])
            if generation_s
            else 0.0
        )
        env_steps_per_s = (
            statistics.median([step / seconds for step, seconds in zip(env_steps, generation_s, strict=True)])
            if generation_s
            else 0.0
        )
        reporter.check(
            "throughput",
            all(seconds > 0.0 for seconds in generation_s),
            f"{len(generation_s)} generations, median {median_s:.4f} s/generation, "
            f"{steps_per_s:.1f} steps/s, {env_steps_per_s:.0f} env-steps/s on {device_name}",
        )

        parity_initialization = initialization_parity(reporter, initial, pop_size, args, tolerance)
        parity_residency = residency_parity(reporter, trainer, pop_size, args, tolerance)
        return {
            "pop_size": int(pop_size),
            "rows": int(trainer.pop_size * trainer.episodes_per_policy),
            "setup_s": setup_s,
            "build_time_s": trainer.build_time_s,
            "tell_build_time_s": trainer.tell_build_time_s,
            "rule_build_time_s": trainer.rule_build_time_s,
            "verification_build_time_s": trainer.verification_build_time_s,
            "run_s": run_s,
            "generation_wall_s": generation_s,
            "generation_wall_median_s": median_s,
            "steps_per_generation": steps,
            "env_steps_per_generation": env_steps,
            "steps_per_s": steps_per_s,
            "env_steps_per_s": env_steps_per_s,
            "fitness_history": [float(value) for value in trainer.fitness_history],
            "placement": dict(trainer.placement.items()),
            "parity": {"initialization": parity_initialization, "residency": parity_residency},
        }
    finally:
        trainer.close()
        env.close()


# --- CLI ----------------------------------------------------------------------------


def parse_args(argv=None):
    """Parse the CLI arguments.

    Args:
        argv (Sequence[str] | None): Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        default=None,
        help=f"evoxc backend of the target (default: $EVOXC_BACKEND or {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="evoxc device of the target (default: $EVOXC_DEVICE or the first non-CPU device)",
    )
    parser.add_argument(
        "--pops",
        type=int,
        nargs="+",
        default=list(DEFAULT_POPS),
        help=f"population sizes to measure (default: {' '.join(str(pop) for pop in DEFAULT_POPS)})",
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="generations per population")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="episodes per policy")
    parser.add_argument("--num-inputs", type=int, default=DEFAULT_NUM_INPUTS, help="genome observation width")
    parser.add_argument("--num-outputs", type=int, default=DEFAULT_NUM_OUTPUTS, help="genome action width")
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED, help="master seed of every run")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="absolute tolerance of the float parities",
    )
    parser.add_argument("--json", default=None, help="write the measurements to this JSON file")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="measure the plumbing-only cpu:0 mode when the host has no accelerator",
    )
    return parser.parse_args(argv)


def _format_measurement(entry):
    """One compact ``DEVICE MEASUREMENTS`` line for a population size."""
    generations = ", ".join(f"{seconds:.4f}" for seconds in entry["generation_wall_s"])
    placement = entry["placement"]
    return (
        f"pop {entry['pop_size']}: setup {entry['setup_s']:.2f} s "
        f"(tell {entry['tell_build_time_s']:.2f}, rule {entry['rule_build_time_s']:.2f}, "
        f"verification {entry['verification_build_time_s']:.2f}) | "
        f"generation median {entry['generation_wall_median_s']:.4f} s over "
        f"{len(entry['generation_wall_s'])} [{generations}] | "
        f"{entry['steps_per_generation'][0] if entry['steps_per_generation'] else 0} steps/generation, "
        f"{entry['steps_per_s']:.1f} steps/s | "
        f"{entry['env_steps_per_generation'][0] if entry['env_steps_per_generation'] else 0} env-steps/generation, "
        f"{entry['env_steps_per_s']:.0f} env-steps/s | "
        f"leaves {placement['randkey']}, fitness {placement['fitness']}"
    )


def main(argv=None):
    """Run every measurement; return 0 when all checks passed and 1 otherwise."""
    args = parse_args(argv)
    target = resolve_target(args)
    if target is None:
        return 0
    print(
        f"target: {target['backend']}@{target['device_name']} "
        f"(available: {', '.join(target['available'])}; pops {list(args.pops)}; "
        f"{args.generations} generations; tolerance {args.tolerance:g})"
    )
    reporter = Reporter()
    entries = []
    started = time.perf_counter()
    for pop_size in args.pops:
        try:
            entries.append(measure_pop(pop_size, target, args, reporter))
        except Exception as exc:  # noqa: BLE001 - one failing population must not hide the others
            reporter.check(f"measure[pop {pop_size}]", False, f"{type(exc).__name__}: {exc}")
    total_s = time.perf_counter() - started

    print(f"DEVICE MEASUREMENTS ({target['backend']}@{target['device_name']}, {total_s:.2f} s total)")
    for entry in entries:
        print(f"  {_format_measurement(entry)}")
    passed = reporter.ok and bool(entries)
    if args.json:
        payload = {
            "schema": SCHEMA,
            "target": {
                "backend": target["backend"],
                "device": target["device_name"],
                "device_kind": target["device"].kind,
                "plugin_path": target["plugin_path"],
                "available_devices": target["available"],
            },
            "arguments": {
                "pops": [int(pop) for pop in args.pops],
                "generations": int(args.generations),
                "episodes": int(args.episodes),
                "num_inputs": int(args.num_inputs),
                "num_outputs": int(args.num_outputs),
                "master_seed": int(args.master_seed),
                "tolerance": float(args.tolerance),
                "allow_cpu": bool(args.allow_cpu),
            },
            "pops": entries,
            "checks": reporter.checks,
            "passed": passed,
            "total_s": total_s,
        }
        path = Path(args.json)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if not passed:
        print("DEVICE CHECKS FAILED")
        return 1
    print("ALL DEVICE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
