"""Self-contained worker of the ``bench_probe.py tell-backend`` subcommand.

The Stage-3 Phase-4 probe has to answer one question per compiler backend: *does the TensorNEAT
``tell`` graph compile and run on this device, and if not, how far does it get before it stops?*
A hang is the documented outcome for ``iree@rocm:0``, so the measurement cannot live in the probe
process -- it has to be a child process that prints a phase marker before every step, so the parent
can kill the whole process group and still say which phase was reached.

The worker builds the smallest honest pipeline: ``DefaultGenome(45 -> 12)`` (the Go2 widths),
``NEAT(pop_size=4, species_size=2)``, ``Pipeline(..., seed=0, backend="numpy")`` -- the evolution
side is always NumPy, only the compiled ``tell`` target is the requested backend -- then
``evolution_graphs.build_tell`` and one ``run_tell``.

Output protocol (consumed by ``bench_probe._run_tell_entry``):

* ``PHASE {"phase": name, "elapsed_s": ...}`` before every step, flushed immediately;
* ``RESULT {...}`` once, with the build/run timings and the returned shapes;
* non-zero exit code and a ``.error`` block on failure.

Nothing here imports ``genesis`` or ``torch``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

PHASE_MARKER = "PHASE"
RESULT_MARKER = "RESULT"


def _emit(marker: str, payload: dict) -> None:
    """Write one machine-readable line and flush it (a killed process must not lose it)."""
    print(f"{marker} {json.dumps(payload, sort_keys=True)}", flush=True)


def _phase(name: str, started: float, **extra) -> None:
    """Print one phase marker with the seconds since the worker started."""
    _emit(PHASE_MARKER, {"phase": name, "elapsed_s": round(time.perf_counter() - started, 6), **extra})


def main(argv=None) -> int:
    """Build the tell graph on the requested backend, run it once, print the phase trail."""
    parser = argparse.ArgumentParser(prog="_tell_worker.py", description="tell-graph backend probe worker")
    parser.add_argument("--backend", required=True, help="evoxc backend of the tell graph (numpy/iree/xla)")
    parser.add_argument("--device", default=None, help="evoxc device, e.g. cpu:0 or rocm:0")
    parser.add_argument("--pjrt-plugin", default=None, help="XLA/PJRT plugin .so (xla backend only)")
    parser.add_argument("--num-inputs", type=int, default=45, help="genome inputs (Go2 observation width)")
    parser.add_argument("--num-outputs", type=int, default=12, help="genome outputs (Go2 action width)")
    parser.add_argument("--pop-size", type=int, default=4, help="policies in the tiny pipeline")
    parser.add_argument("--species-size", type=int, default=2, help="TensorNEAT species size")
    parser.add_argument("--max-nodes", type=int, default=80, help="genome node budget (57 nodes are needed for 45 -> 12)")
    parser.add_argument("--max-conns", type=int, default=640, help="genome connection budget")
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        # The ``examples/**/*.py`` smoke harness runs every script; a worker without arguments is
        # not a failing example, so print the usage and exit 0 like the benchmark CLIs do.
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    started = time.perf_counter()
    _phase("start", started, backend=args.backend, device=args.device)

    import numpy as np

    _phase("numpy_imported", started)
    import evoxc

    _phase("evoxc_imported", started, evoxc_version=getattr(evoxc, "__version__", None))
    if args.pjrt_plugin:
        # ``_to_device_array`` re-selects the backend before staging, so the option-less PJRT
        # discovery must see the same plugin the tell build is given.
        os.environ.setdefault("EVOXC_PJRT_PLUGIN", str(args.pjrt_plugin))
    from tensorneat import Pipeline, algorithm, genome
    from tensorneat.common.tools import to_numpy

    _phase("tensorneat_imported", started)
    from evolution_graphs import build_tell, run_tell
    from trainer import NeuroevolutionTrainer, TraceOnlyProblem, pop_arrays_of, species_leaves_of

    _phase("worker_imports_done", started)

    setup_start = time.perf_counter()
    gen = genome.DefaultGenome(
        num_inputs=args.num_inputs,
        num_outputs=args.num_outputs,
        max_nodes=args.max_nodes,
        max_conns=args.max_conns,
    )
    algo = algorithm.NEAT(gen, pop_size=args.pop_size, species_size=args.species_size)
    pipeline = Pipeline(
        algo,
        TraceOnlyProblem(args.num_inputs, args.num_outputs),
        seed=0,
        backend="numpy",
        is_detailed=False,
    )
    state = pipeline.setup()
    config = NeuroevolutionTrainer._static_config(pipeline, state)
    pop_arrays = pop_arrays_of(state)
    species_leaves = species_leaves_of(state.species)
    setup_s = time.perf_counter() - setup_start
    _phase(
        "pipeline_ready",
        started,
        setup_s=round(setup_s, 6),
        pop_shapes=[list(np.shape(leaf)) for leaf in pop_arrays],
        species_leaves=len(species_leaves),
        randkey=int(to_numpy(state.randkey)),
        generation=float(to_numpy(state.generation)),
    )

    _phase("build_tell_start", started, backend=args.backend, device=args.device)
    build_start = time.perf_counter()
    executable = build_tell(
        algo,
        config,
        pop_arrays,
        species_leaves,
        int(algo.pop_size),
        backend=args.backend,
        device=args.device,
        plugin_path=args.pjrt_plugin,
    )
    build_s = time.perf_counter() - build_start
    _phase(
        "build_tell_done",
        started,
        build_s=round(build_s, 6),
        resolved_backend=executable.backend,
        resolved_device=executable.device_name,
    )

    fitness = np.zeros(int(algo.pop_size), dtype=np.float32)
    _phase("run_tell_start", started)
    run_start = time.perf_counter()
    result = run_tell(
        executable,
        randkey=state.randkey,
        pop_arrays=pop_arrays,
        generation=state.generation,
        fitness=fitness,
        species_leaves=species_leaves,
    )
    run_s = time.perf_counter() - run_start
    _phase("run_tell_done", started, run_s=round(run_s, 6))

    _emit(
        RESULT_MARKER,
        {
            "status": "ok",
            "backend": args.backend,
            "device": args.device,
            "resolved_backend": executable.backend,
            "resolved_device": executable.device_name,
            "setup_s": round(setup_s, 6),
            "build_s": round(build_s, 6),
            "run_s": round(run_s, 6),
            "total_s": round(time.perf_counter() - started, 6),
            "population": [
                {"shape": list(np.shape(leaf)), "dtype": str(np.asarray(leaf).dtype)} for leaf in result["pop"]
            ],
            "species_leaves": len(species_leaves_of(result["species"])),
            "generation_out": float(result["generation"]),
            "randkey_out": int(result["randkey"]),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - the phase trail is the payload; report and exit 1
        _emit(
            RESULT_MARKER,
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
