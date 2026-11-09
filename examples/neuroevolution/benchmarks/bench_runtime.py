"""Stage-3 Phase-2 runtime benchmarks: ``throughput`` and ``residency-parity``.

``bench_envs.py`` answers "does the migrated evaluation environment still compute the
pre-migration maths?"; this module answers the two runtime questions that follow:

* **throughput** -- how many ``scene.step()`` calls per second (and how many environment steps
  per second) does one pilot environment sustain, split by the timer groups of
  :data:`bench_envs.TIMER_GROUPS`, with the per-graph compile times that precede them.  The
  protocol is one build, generation 0 as a warm-up (recorded separately, never in the headline),
  then generations 1..3 as the three timed repeats; :func:`common.sync` brackets every generation,
  and the record reports the median and the spread of the repeats, never a single number.
* **residency-parity** -- the same generation evaluated by the host-staged (NumPy) rollout and by
  the device-resident one, in the same process on the same ``genesis`` backend, compared field by
  field (``seeds``, ``returns``, ``fitness``, ``steps``, ``end_reasons``, ``episode_returns`` and
  the task payload -- ``init_qpos`` for Go2, ``commands`` for hover).  The record carries the max
  absolute deviations, the bit-identical flags, both fingerprints, both throughput blocks and the
  per-step host-call/byte/sync counters that prove what the device residency removed.

Everything here reuses the Phase-1 harness: :func:`bench_envs._load_task`,
:func:`bench_envs._build_env`, :func:`bench_envs._instrument` and
:func:`bench_envs._generation_entry` are the pilot-env plumbing, :mod:`common` owns the record
schema, the timers and the traffic counters, and nothing on disk outside this directory is
touched (all instrumentation is runtime monkey-patching that is restored afterwards).

Usage (from the repository root)::

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_runtime.py throughput --task go2 --pop-size 8 \
        --horizon 200 --residency numpy --backend numpy --device cpu:0 --repeat 0 \
        --process-role only

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_runtime.py residency-parity --task hover --pop-size 8 \
        --horizon 200 --gpu --repeat 0 --process-role only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import bench_envs
import common
from bench_envs import _fingerprint, _r
from common import run_guarded, write_record
from runtime_measure import _ensure_gs_init, _filter_own_gpu_pids, _measure_env
from runtime_probes import (
    DEFAULT_GENERATIONS,
    RESULTS_DIR,
    RUNTIME_TIMER_DESCRIPTIONS,
    TIMED_GENERATIONS,
    WARMUP_GENERATION,
)


def _variant(args) -> str:
    """The measurement variant label: ``baseline``, or the runtime override this run needed."""
    if getattr(args, "collision_cap", None) is not None:
        return f"collision_cap_{int(args.collision_cap)}"
    return "baseline"


def _rigid_override(args):
    """The ``RigidOptions`` runtime override of this run (``None`` for the unmodified build)."""
    cap = getattr(args, "collision_cap", None)
    if cap is None:
        return None
    return {
        "max_collision_pairs": int(cap),
        "how": (
            "harness-level runtime override: gs.options.RigidOptions is wrapped for the duration of "
            "this env's construction so max_collision_pairs is at least this value; "
            "examples/locomotion/go2_neat_env.py is not modified and passes its own 20 explicitly"
        ),
        "why": (
            "the sibling env hardcodes max_collision_pairs=20 outside CFG and raises when the narrow "
            "phase of a larger batch exceeds it; this variant locates the ceiling instead of editing "
            "the read-only sibling"
        ),
    }


def _throughput_workload(args) -> dict:
    """The ``workload`` block of a throughput record."""
    physics = "gpu" if args.gpu else "cpu"
    return {
        "task": str(args.task),
        "pop_size": int(args.pop_size),
        "horizon": int(args.horizon),
        "generations": int(args.generations),
        "warmup_generation": WARMUP_GENERATION,
        "timed_generations": TIMED_GENERATIONS,
        "residency": str(args.residency),
        "backend": str(args.backend),
        "device": str(args.device) if args.device is not None else "cpu:0",
        "physics_backend": physics,
        "stop_when_done": False,
        "perturbation": "zero (default seeding.Perturbation)",
        "payload": "env._trace_pop (go2) / env.initial_pop_arrays() (hover)",
        "timer_groups": {**bench_envs.TIMER_DESCRIPTIONS, **RUNTIME_TIMER_DESCRIPTIONS},
        "sync": "common.sync() before and after every generation",
        "variant": _variant(args),
        "rigid_options_override": _rigid_override(args),
        "byte_estimate": (
            "tensor numel * element_size, numpy .nbytes; a graph call counts its operands and its "
            "result, so the figure is an estimate of the host <-> graph traffic, not a PCIe counter"
        ),
    }


def _throughput(args) -> dict:
    """Run the throughput measurement and return the finished record."""
    task = str(args.task)
    # Before the record: ``common.new_record`` -> ``gpu_info()`` queries torch.cuda, which pins
    # torch's bundled HSA runtime and makes genesis skip its re-exec repair (a mixed-runtime
    # process crashes with a malformed AQL packet on this host). Initialising genesis first lets
    # that repair run.
    init_info = _ensure_gs_init(bool(args.gpu))
    record = common.new_record(
        kind="env_throughput",
        command=sys.argv,
        task=task,
        seed=args.master_seed,
        repeat=args.repeat,
        process_role=args.process_role,
        workload=_throughput_workload(args),
    )
    record["genesis_init"] = init_info
    _filter_own_gpu_pids(record)
    run_guarded(lambda rec: _run_throughput(args, rec), record)
    out_path = Path(args.out) if args.out else _throughput_out_path(args)
    write_record(out_path, record)
    _print_summary(record, out_path)
    return record


def _run_throughput(args, record: dict) -> None:
    """The body of :func:`_throughput`: build, instrument, warm up, time the repeats."""
    measurement = _measure_env(
        str(args.task),
        pop_size=args.pop_size,
        master_seed=args.master_seed,
        horizon=args.horizon,
        residency=str(args.residency),
        backend=str(args.backend),
        device=args.device,
        generations=int(args.generations),
        device_path=str(args.residency) == "device",
        collision_cap=args.collision_cap,
    )
    record["workload"].update(
        {
            "rows": measurement["rows"],
            "episodes_per_policy": measurement["episodes_per_policy"],
            "max_episode_length": measurement["max_episode_length"],
            "resolved_backend": measurement["backend"],
            "resolved_device": measurement["device_name"],
        }
    )
    record["phases"] = measurement["phases"]
    record["timings"] = {
        "groups": measurement["timer_totals"],
        "note": "per-generation breakdowns live in measurements.generations[*].timings",
    }
    record["measurements"] = {
        "task": str(args.task),
        "residency": measurement["residency"],
        "device_name": measurement["device_name"],
        "generations": measurement["entries"],
        "throughput": measurement["throughput"],
        "per_step_means": measurement["per_step_means"],
        "cache": measurement["cache"],
        "traffic": measurement["traffic"],
        "device_resident_intermediates": measurement["device_resident_intermediates"],
        "state_read_parity": measurement["state_read_parity"],
        "checks": measurement["checks"],
        "host_calls_per_step": _sum_traffic(measurement["traffic"], "calls_per_step"),
        "host_bytes_per_step": _sum_traffic(measurement["traffic"], "bytes_per_step"),
    }


def _sum_traffic(traffic: dict, field: str):
    """Sum one per-step traffic field over the counters (``None`` when nothing crossed)."""
    values = [value[field] for value in traffic.values() if value.get(field) is not None]
    return _r(sum(values)) if values else None


# --- residency parity --------------------------------------------------------------


def _compare_arrays(reference, observed) -> dict:
    """Max abs deviation and the bit-identical flag of two result arrays."""
    ref = np.asarray(reference)
    obs = np.asarray(observed)
    if ref.shape != obs.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(ref.shape),
            "observed_shape": list(obs.shape),
            "pass": False,
        }
    difference = float(np.max(np.abs(ref.astype(np.float64) - obs.astype(np.float64)))) if ref.size else 0.0
    return {
        "shape_equal": True,
        "max_abs_diff": _r(difference),
        "bit_identical": bool(np.array_equal(ref, obs)),
        "pass": bool(difference == 0.0),
    }


def _parity_block(numpy_result, device_result, task) -> dict:
    """Field-by-field comparison of one generation evaluated by both residencies."""
    payload_key = "init_qpos" if task == "go2" else "commands"
    arrays = {}
    for key in ("seeds", "returns", "fitness", "episode_returns", payload_key):
        if key in numpy_result and key in device_result:
            arrays[key] = _compare_arrays(numpy_result[key], device_result[key])
    numpy_fp, numpy_src = _fingerprint(numpy_result)
    device_fp, device_src = _fingerprint(device_result)
    scalars = {
        "steps": {"reference": int(numpy_result["steps"]), "observed": int(device_result["steps"])},
        "env_steps": {"reference": int(numpy_result["env_steps"]), "observed": int(device_result["env_steps"])},
        "end_reasons": {
            "reference": {name: int(count) for name, count in numpy_result["end_reasons"].items()},
            "observed": {name: int(count) for name, count in device_result["end_reasons"].items()},
        },
    }
    scalars["steps"]["pass"] = scalars["steps"]["reference"] == scalars["steps"]["observed"]
    scalars["env_steps"]["pass"] = scalars["env_steps"]["reference"] == scalars["env_steps"]["observed"]
    scalars["end_reasons"]["pass"] = scalars["end_reasons"]["reference"] == scalars["end_reasons"]["observed"]
    all_pass = all(entry["pass"] for entry in arrays.values()) and all(
        entry["pass"] for entry in scalars.values()
    )
    return {
        "generation": 0,
        "arrays": arrays,
        "scalars": scalars,
        "fingerprint": {
            "numpy": numpy_fp,
            "device": device_fp,
            "equal": bool(numpy_fp == device_fp),
            "numpy_source": numpy_src,
            "device_source": device_src,
            "rule": "sha256 over seeds + returns + fitness rounded to 6 decimals",
        },
        "all_pass": bool(all_pass),
        "note": (
            "both residencies run the same graphs on the same values in the same order; only the "
            "location of the operands differs, so every field is expected to be bit-identical. A "
            "nonzero entry means the two sides handed the graphs different inputs: check "
            "sides.*.state_read_parity (Go2 reads the state through the entity getters on the "
            "numpy side and through the live solver buffers on the device side) before blaming a "
            "graph"
        ),
    }


def _residency_workload(args) -> dict:
    """The ``workload`` block of a residency record (the workload both sides share)."""
    return {
        "task": str(args.task),
        "pop_size": int(args.pop_size),
        "horizon": int(args.horizon),
        "generations": int(args.generations),
        "warmup_generation": WARMUP_GENERATION,
        "timed_generations": TIMED_GENERATIONS,
        "residencies": ["numpy", "device"],
        "backend": str(args.backend),
        "device": str(args.device) if args.device is not None else ("rocm:0" if args.gpu else "cpu:0"),
        "physics_backend": "gpu" if args.gpu else "cpu",
        "stop_when_done": False,
        "perturbation": "zero (default seeding.Perturbation), identical on both sides",
        "payload": "env._trace_pop (go2) / env.initial_pop_arrays() (hover), same object on both sides",
        "parity_generation": 0,
        "timer_groups": {**bench_envs.TIMER_DESCRIPTIONS, **RUNTIME_TIMER_DESCRIPTIONS},
        "sync": "common.sync() before and after every generation",
    }


def _residency(args) -> dict:
    """Run the residency comparison and return the finished record."""
    task = str(args.task)
    # See ``_throughput``: genesis must initialise (and possibly re-exec for the HSA repair)
    # before the record's provenance block queries torch.cuda.
    init_info = _ensure_gs_init(bool(args.gpu))
    record = common.new_record(
        kind="env_residency",
        command=sys.argv,
        task=task,
        seed=args.master_seed,
        repeat=args.repeat,
        process_role=args.process_role,
        workload=_residency_workload(args),
    )
    record["genesis_init"] = init_info
    _filter_own_gpu_pids(record)
    run_guarded(lambda rec: _run_residency(args, rec), record)
    out_path = Path(args.out) if args.out else _residency_out_path(args)
    write_record(out_path, record)
    _print_summary(record, out_path)
    return record


def _run_residency(args, record: dict) -> None:
    """Build both residencies in one process, measure each, and compare generation 0."""
    sides = {}
    for residency in ("numpy", "device"):
        sides[residency] = _measure_env(
            str(args.task),
            pop_size=args.pop_size,
            master_seed=args.master_seed,
            horizon=args.horizon,
            residency=residency,
            backend=str(args.backend),
            device=args.device,
            generations=int(args.generations),
            device_path=residency == "device",
            collision_cap=args.collision_cap,
            keep_results=True,
        )
    numpy_side, device_side = sides["numpy"], sides["device"]
    if WARMUP_GENERATION not in numpy_side["results"] or WARMUP_GENERATION not in device_side["results"]:
        raise RuntimeError("both residencies must evaluate generation 0 for the parity comparison")
    parity = _parity_block(
        numpy_side["results"][WARMUP_GENERATION],
        device_side["results"][WARMUP_GENERATION],
        str(args.task),
    )
    record["workload"].update(
        {
            "rows": numpy_side["rows"],
            "episodes_per_policy": numpy_side["episodes_per_policy"],
            "max_episode_length": numpy_side["max_episode_length"],
            "resolved_backend": numpy_side["backend"],
            "resolved_device": numpy_side["device_name"],
            "device_side_resolved": f"{device_side['backend']}@{device_side['device_name']}",
        }
    )
    record["phases"] = {
        "numpy": numpy_side["phases"],
        "device": device_side["phases"],
        "note": "one row per residency; the two environments are built one after the other in this process",
    }
    record["timings"] = {
        "numpy": {"groups": numpy_side["timer_totals"]},
        "device": {"groups": device_side["timer_totals"]},
        "note": "process totals; per-generation breakdowns live in measurements.sides[*].generations",
    }
    record["measurements"] = {
        "task": str(args.task),
        "generation": WARMUP_GENERATION,
        "parity": parity,
        "sides": {
            name: {
                "residency": side["residency"],
                "backend": side["backend"],
                "device_name": side["device_name"],
                "build_s": side["phases"]["build_s"],
                "graph_build_s": side["phases"]["graph_build_s"],
                "generations": side["entries"],
                "throughput": side["throughput"],
                "per_step_means": side["per_step_means"],
                "cache": side["cache"],
                "traffic": side["traffic"],
                "host_calls_per_step": _sum_traffic(side["traffic"], "calls_per_step"),
                "host_bytes_per_step": _sum_traffic(side["traffic"], "bytes_per_step"),
                "device_resident_intermediates": side["device_resident_intermediates"],
                "state_read_parity": side["state_read_parity"],
                "checks": side["checks"],
            }
            for name, side in sides.items()
        },
        "comparison": _residency_comparison(numpy_side, device_side),
    }


def _residency_comparison(numpy_side, device_side) -> dict:
    """The headline ratio: device-residency steps/s against the NumPy baseline of the same process."""
    numpy_median = numpy_side["throughput"]["steps_per_s"]["median"]
    device_median = device_side["throughput"]["steps_per_s"]["median"]
    return {
        "steps_per_s_median": {"numpy": numpy_median, "device": device_median},
        "env_steps_per_s_median": {
            "numpy": numpy_side["throughput"]["env_steps_per_s"]["median"],
            "device": device_side["throughput"]["env_steps_per_s"]["median"],
        },
        "speedup": _r(device_median / numpy_median) if numpy_median and device_median else None,
        "host_calls_per_step": {
            "numpy": _sum_traffic(numpy_side["traffic"], "calls_per_step"),
            "device": _sum_traffic(device_side["traffic"], "calls_per_step"),
        },
        "host_bytes_per_step": {
            "numpy": _sum_traffic(numpy_side["traffic"], "bytes_per_step"),
            "device": _sum_traffic(device_side["traffic"], "bytes_per_step"),
        },
        "note": (
            "one process, one genesis backend, the same master seed and perturbation; the only "
            "difference between the two rows is where the graph operands live"
        ),
    }


# --- output paths and summary ------------------------------------------------------


def _suffix(repeat: int) -> str:
    """``""`` for the primary replica and ``_r<n>`` for a re-run in a fresh process."""
    return "" if int(repeat) == 0 else f"_r{int(repeat)}"


def _throughput_out_path(args) -> Path:
    """``results/env_<task>_throughput_<residency>_<physics>_pop<n>[_r<repeat>].json``."""
    physics = "gpu" if args.gpu else "cpu"
    return RESULTS_DIR / (
        f"env_{args.task}_throughput_{args.residency}_{physics}_pop{int(args.pop_size)}{_suffix(args.repeat)}.json"
    )


def _residency_out_path(args) -> Path:
    """``results/residency_<task>_gpu[_r<repeat>].json`` (``_device_cpu`` for a CPU-only run)."""
    label = "gpu" if args.gpu else "device_cpu"
    return RESULTS_DIR / f"residency_{args.task}_{label}{_suffix(args.repeat)}.json"


def _print_summary(record: dict, out_path: Path) -> None:
    """Print the reviewable one-screen summary of a finished (or failed) record."""
    print(f"[bench] {record['kind']} task={record['task']} seed={record['seed']} repeat={record['repeat']}")
    print(f"[bench] status={record['status']} git={record['git']['head'][:12]} dirty={record['git']['dirty']}")
    print(
        f"[bench] host: loadavg={record['host'].get('loadavg_1m')} "
        f"gpu_pids={record['host'].get('rocm_pids')} contended={record['contended']}"
    )
    print(f"[bench] record -> {out_path}")
    if record["status"] != "ok":
        error = record.get("error") or {}
        print(f"[bench] FAILED {error.get('type')}: {error.get('message')}")
        return
    measurements = record["measurements"]

    def report_side(name, side):
        throughput = side["throughput"]
        steps = throughput["steps_per_s"]
        env_steps = throughput["env_steps_per_s"]
        print(
            f"[bench] {name}: {steps['median']:.2f} +/- {steps['spread']:.2f} steps/s "
            f"({env_steps['median']:.1f} env-steps/s) rows={throughput['rows']} "
            f"build={side['build_s']:.2f}s "
            f"graph_build={sum(slot['build_s'] for slot in side['graph_build_s'].values()):.2f}s "
            f"calls/step={side['host_calls_per_step']} bytes/step={side['host_bytes_per_step']}"
        )
        for entry in side["generations"]:
            marker = "warmup" if entry["warmup"] else "timed "
            residual = entry["residual_fraction"]
            share = "n/a" if residual is None else f"{100.0 * residual:.1f}%"
            print(
                f"[bench]   gen {entry['generation']} {marker} steps={entry['steps']} "
                f"{entry['steps_per_s']:.2f} steps/s {entry['env_steps_per_s']:.1f} env-steps/s "
                f"wall={entry['wall_time_s']:.3f}s residual={share}"
            )
        graphs = ", ".join(
            f"{name}={slot['build_s']:.2f}s" for name, slot in side["graph_build_s"].items()
        )
        print(f"[bench]   compile: {graphs}")
        checks = side["checks"]["checks"]
        if checks:
            print(
                "[bench]   checks: "
                + ", ".join(f"{'ok' if check['pass'] else 'MISMATCH'}" for check in checks)
            )

    if record["kind"] == "env_throughput":
        side = {
            "build_s": record["phases"]["build_s"],
            "graph_build_s": record["phases"]["graph_build_s"],
            "generations": measurements["generations"],
            "throughput": measurements["throughput"],
            "host_calls_per_step": measurements["host_calls_per_step"],
            "host_bytes_per_step": measurements["host_bytes_per_step"],
            "checks": measurements["checks"],
        }
        report_side(measurements["residency"], side)
        return
    for name in ("numpy", "device"):
        report_side(name, measurements["sides"][name])
    parity = measurements["parity"]
    for key, entry in parity["arrays"].items():
        bit = "bit-identical" if entry.get("bit_identical") else f"max|d|={entry.get('max_abs_diff')}"
        print(f"[bench] parity {key}: {bit} pass={entry['pass']}")
    print(
        f"[bench] parity steps={parity['scalars']['steps']['reference']} "
        f"end_reasons match={parity['scalars']['end_reasons']['pass']} "
        f"fingerprint match={parity['fingerprint']['equal']} all_pass={parity['all_pass']}"
    )
    comparison = measurements["comparison"]
    print(
        f"[bench] device vs numpy: {comparison['steps_per_s_median']['device']} vs "
        f"{comparison['steps_per_s_median']['numpy']} steps/s (speedup {comparison['speedup']}), "
        f"host calls/step {comparison['host_calls_per_step']}, bytes/step {comparison['host_bytes_per_step']}"
    )


# --- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the Phase-2 runtime benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_runtime.py",
        description="Stage-3 runtime benchmarks: throughput and residency parity of the pilot envs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    throughput = subparsers.add_parser(
        "throughput",
        help="time one pilot env over a warm-up and three timed generations",
        description=(
            "Build one pilot environment (optionally on the GPU), call env.executables() once per "
            "generation as the compile-once proof, run generation 0 as a warm-up and generations "
            "1-3 as the timed repeats, and record per-graph compile times, per-generation "
            "timings, throughput statistics and the host-traffic counters."
        ),
    )
    throughput.add_argument("--task", required=True, choices=("go2", "hover"), help="pilot task to measure")
    throughput.add_argument("--pop-size", type=int, default=8, help="policies per generation (rows = 3 x this)")
    throughput.add_argument("--horizon", type=int, default=200, help="scene.step() calls per generation")
    throughput.add_argument(
        "--residency", default="numpy", choices=("numpy", "device"), help="graph operand residency"
    )
    throughput.add_argument("--backend", default="numpy", help="evoxc backend of the compiled graphs")
    throughput.add_argument("--device", default="cpu:0", help="evoxc device of the compiled graphs")
    throughput.add_argument("--gpu", action="store_true", help="initialise genesis on the GPU (gs.gpu physics)")
    throughput.add_argument("--master-seed", type=int, default=0, help="run seed (episode seeds derive from it)")
    throughput.add_argument("--repeat", type=int, default=0, help="replica index; nonzero marks a re-run")
    throughput.add_argument(
        "--process-role",
        default="only",
        choices=("cold", "warm", "only"),
        help="where this process sits in the warm-up chain (cold/warm compile cache)",
    )
    throughput.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="warm-up + timed generations")
    throughput.add_argument("--out", default=None, help="record path (default: results/env_<task>_throughput_*.json)")
    throughput.add_argument(
        "--collision-cap",
        type=int,
        default=None,
        help="raise RigidOptions.max_collision_pairs at runtime (labeled variant=collision_cap_<n>)",
    )

    residency = subparsers.add_parser(
        "residency-parity",
        help="compare the numpy and device residencies of one generation, and time both",
        description=(
            "Build a numpy-residency environment and a device-residency environment in the same "
            "process on the same genesis backend, evaluate the same generation index on both, and "
            "record the field-by-field parity, both throughput blocks and the host-traffic counters."
        ),
    )
    residency.add_argument("--task", required=True, choices=("go2", "hover"), help="pilot task to compare")
    residency.add_argument("--pop-size", type=int, default=8, help="policies per generation (rows = 3 x this)")
    residency.add_argument("--horizon", type=int, default=200, help="scene.step() calls per generation")
    residency.add_argument("--backend", default="iree", help="evoxc backend of the compiled graphs")
    residency.add_argument("--device", default=None, help="evoxc device (default: rocm:0 with --gpu, else cpu:0)")
    residency.add_argument("--gpu", action="store_true", help="initialise genesis on the GPU (gs.gpu physics)")
    residency.add_argument("--master-seed", type=int, default=0, help="run seed (episode seeds derive from it)")
    residency.add_argument("--repeat", type=int, default=0, help="replica index; nonzero marks a re-run")
    residency.add_argument(
        "--process-role",
        default="only",
        choices=("cold", "warm", "only"),
        help="where this process sits in the warm-up chain",
    )
    residency.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="warm-up + timed generations")
    residency.add_argument("--out", default=None, help="record path (default: results/residency_<task>_<label>.json)")
    residency.add_argument(
        "--collision-cap",
        type=int,
        default=None,
        help="raise RigidOptions.max_collision_pairs at runtime (labeled variant=collision_cap_<n>)",
    )
    return parser


def main(argv=None) -> int:
    """Parse ``argv`` (``sys.argv[1:]`` by default) and run the requested subcommand.

    A bare invocation prints the usage and exits 0: ``tests/test_examples.py`` runs every
    ``examples/**/*.py`` as a subprocess, and a command-line tool without a subcommand is not a
    failing example.  Anything that does name a subcommand is parsed strictly (exit 2 on misuse).
    """
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.device is None and args.command == "residency-parity":
        args.device = "rocm:0" if args.gpu else "cpu:0"
    if args.command == "throughput":
        record = _throughput(args)
    elif args.command == "residency-parity":
        record = _residency(args)
    else:
        raise SystemExit(f"unsupported subcommand {args.command!r}")
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
