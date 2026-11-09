"""Stage-3 Phase-3 trainer benchmark: ``train`` and ``check-determinism``.

``bench_envs.py`` (Phase 1) checks the migrated evaluation maths, ``bench_runtime.py`` (Phase 2)
measures the environment's throughput and residency, and this module measures the thing that sits
on top of both: one :class:`trainer.NeuroevolutionTrainer` run, with the two regions of every
generation timed separately.

``train`` -- build the pilot env, build the trainer, ``trainer.setup()`` (which compiles the
``tell`` graph), then run the generations.  Each generation is timed as:

* ``evaluate_s`` -- the ``env.evaluate`` call: seed draw, reset, ``scene.step()`` loop, the fitness
  graph (``env_wall_s`` is the environment's own counter of the simulation part);
* ``tell_s`` -- the compiled NEAT transition (``trainer.run_tell``);
* ``generation_s`` -- the whole generation, from the previous tell's end to this one's, with
  ``bookkeeping_s`` the residual the trainer spends between the regions.

Every region is bracketed by :func:`common.sync` and every patch is restored before the
environment is closed.  After ``trainer.run()`` the record's held-out block scores the initial and
the trained population on one fresh seed triple (``master_seed + 1000``) that no training
generation used.

``check-determinism`` -- compare two ``train`` records of the same task and master seed: the seed
table, the per-generation seeds, every fitness vector, the held-out fitness and the sha256
fingerprint must be *identical* (not merely close), and the record's ``status`` is ``ok`` only when
they are.

Usage (from the repository root)::

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_neat.py train --task go2 --master-seed 0 \
        --pop-size 8 --generations 3 --horizon 200 --residency numpy --backend numpy \
        --device cpu:0 --tell-backend numpy --process-role only

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_neat.py check-determinism \
        results/neat_go2_seed0.json results/neat_go2_seed0_r1.json --out results/neat_determinism_go2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import bench_neat_measure as measure_module
import common
from bench_envs import _r
from common import run_guarded, write_record
from runtime_measure import _ensure_gs_init, _filter_own_gpu_pids
from runtime_probes import RESULTS_DIR


def _suffix(repeat: int) -> str:
    """``""`` for the primary replica and ``_r<n>`` for a re-run in a fresh process."""
    return "" if int(repeat) == 0 else f"_r{int(repeat)}"


def _default_out_path(args) -> Path:
    """``results/neat_<task>_seed<n>[_gpu][_<residency>][_<tell>_tell][_r<repeat>].json``.

    The plain CPU/NumPy/NumPy-tell run keeps the short ``neat_<task>_seed<n>.json`` name; every
    deviation from that configuration adds its label so no two measurements can share a file.
    """
    name = f"neat_{args.task}_seed{int(args.master_seed)}"
    if args.gpu:
        name += "_gpu"
    if str(args.residency) != "numpy":
        name += f"_{args.residency}"
    if args.tell_backend and str(args.tell_backend) != "numpy":
        name += f"_{args.tell_backend}_tell"
    return RESULTS_DIR / f"{name}{_suffix(args.repeat)}.json"


# --- train --------------------------------------------------------------------------


def _workload(args) -> dict:
    """The ``workload`` block of a training record (everything that defines the measured run)."""
    perturbation = measure_module.perturbation_for(str(args.task))
    genome = {
        "num_inputs": measure_module.TASK_GENOME[args.task][0],
        "num_outputs": measure_module.TASK_GENOME[args.task][1],
        "max_nodes": measure_module.MAX_NODES,
        "max_conns": measure_module.MAX_CONNS,
        "species_size": measure_module.SPECIES_SIZE,
    }
    return {
        "task": str(args.task),
        "pop_size": int(args.pop_size),
        "rows": 3 * int(args.pop_size),
        "generations": int(args.generations),
        "horizon": int(args.horizon),
        "episode_length": measure_module.EPISODE_LENGTH_NOTE,
        "residency": str(args.residency),
        "backend": str(args.backend),
        "device": str(args.device) if args.device is not None else "cpu:0",
        "tell_backend": str(args.tell_backend) if args.tell_backend else "trainer default (env backend; xla on ROCm)",
        "pjrt_plugin": str(args.pjrt_plugin) if args.pjrt_plugin else None,
        "physics_backend": "gpu" if args.gpu else "cpu",
        "stop_when_done": False,
        "perturbation": {
            "base_pos": float(perturbation.base_pos),
            "base_rpy": float(perturbation.base_rpy),
            "joint_pos": float(perturbation.joint_pos),
            "source": "the pilot trainer CLI defaults",
        },
        "genome": genome,
        "holdout": {
            "seed_stream": f"seeding.generation_seed_table(master_seed + {measure_module.HOLDOUT_SEED_OFFSET}, 1)",
            "episodes_per_policy": 3,
            "populations": "the generation-0 (initial) and the last-generation (trained) population",
        },
        "timer_groups": {
            "evaluate": "env.evaluate: seed draw + reset + scene.step() loop + fitness graph, sync-bracketed",
            "tell": "trainer.run_tell: one compiled NEAT generation transition, sync-bracketed",
            "generation_total": "the whole generation, tell end minus the previous tell end",
        },
        "sync": "common.sync() before and after each timed region",
        "note": (
            "generations are real generations, not warm-ups: generation 0 trains on the initial "
            "population, so it is reported separately from the steady generations 1.."
        ),
    }


def _train(args) -> dict:
    """Run the ``train`` subcommand and return the written record."""
    task = str(args.task)
    # Before the record: ``common.new_record`` -> ``gpu_info()`` queries torch.cuda, which pins
    # torch's bundled HSA runtime and makes genesis skip its re-exec repair. Initialising genesis
    # first lets that repair run.
    init_info = _ensure_gs_init(bool(args.gpu))
    if args.pjrt_plugin:
        # ``Tensor.to`` passes no options, so the option-less PJRT discovery used for device
        # staging must see the same plugin as the xla tell build.
        os.environ.setdefault("EVOXC_PJRT_PLUGIN", str(args.pjrt_plugin))
    record = common.new_record(
        kind="neat_train",
        command=sys.argv,
        task=task,
        seed=args.master_seed,
        repeat=args.repeat,
        process_role=args.process_role,
        workload=_workload(args),
    )
    record["genesis_init"] = init_info
    _filter_own_gpu_pids(record)
    run_guarded(lambda rec: _fill_train(args, rec), record)
    out_path = Path(args.out) if args.out else _default_out_path(args)
    write_record(out_path, record)
    _print_summary(record, out_path)
    return record


def _fill_train(args, record: dict) -> None:
    """The body of :func:`_train`: run the measurement and fill the record's blocks."""
    measurement = measure_module.measure(args)
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
        "note": "per-generation timings live in measurements.generations[*]; phases holds the build side"
    }
    record["measurements"] = {
        "task": measurement["task"],
        "rows": measurement["rows"],
        "pop_size": measurement["pop_size"],
        "residency": measurement["residency"],
        "backend": measurement["backend"],
        "device_name": measurement["device_name"],
        "seed_table": measurement["seed_table"],
        "genome": measurement["genome"],
        "generations": measurement["generations"],
        "totals": measurement["totals"],
        "learning_curve": measurement["measurements"]["learning_curve"],
        "holdout": measurement["measurements"]["holdout"],
        "checks": measurement["measurements"]["checks"],
        "tell": measurement["measurements"]["tell"],
        "config": measurement["measurements"]["config"],
        "trainer_totals": measurement["measurements"]["trainer_totals"],
        "perturbation": measurement["perturbation"],
        "sync_note": measurement["sync_note"],
        "episode_length_note": measurement["episode_length_note"],
    }


# --- check-determinism -------------------------------------------------------------


def _record_arrays(record: dict) -> dict:
    """Extract the five arrays a second process must reproduce from one ``train`` record."""
    measurements = record.get("measurements") or {}
    generations = measurements.get("generations") or []
    checks = measurements.get("checks") or {}
    holdout = measurements.get("holdout") or {}
    return {
        "seed_table": measurements.get("seed_table") or (measurements.get("config") or {}).get("seed_table"),
        "generation_seeds": [entry.get("seeds") for entry in generations],
        "fitness": [entry.get("fitness") for entry in generations],
        "holdout_initial": (holdout.get("initial") or {}).get("fitness"),
        "holdout_trained": (holdout.get("trained") or {}).get("fitness"),
        "fingerprint": checks.get("determinism_fingerprint"),
    }


def _array_compare(reference, observed) -> dict:
    """Compare two numeric blocks exactly; ``match`` needs a zero maximum absolute difference."""
    if reference is None or observed is None:
        return {"match": False, "max_abs_diff": None, "reason": "missing on one side"}
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(observed, dtype=np.float64)
    if left.shape != right.shape:
        return {"match": False, "max_abs_diff": None, "reason": "shape mismatch", "shapes": [list(left.shape), list(right.shape)]}
    diff = float(np.max(np.abs(left - right))) if left.size else 0.0
    return {
        "match": diff == 0.0,
        "max_abs_diff": _r(diff),
        "match_within_1e-6": bool(diff <= 1e-6),
        "shape": list(left.shape),
    }


def _seeds_compare(reference, observed) -> dict:
    """Compare the per-generation seed rows of two runs (integers, so exact equality)."""
    left = list(reference or [])
    right = list(observed or [])
    if len(left) != len(right):
        return {"match": False, "reason": "different generation counts", "generations": [len(left), len(right)]}
    mismatched = [
        index for index, (row_a, row_b) in enumerate(zip(left, right)) if [int(seed) for seed in row_a] != [int(seed) for seed in row_b]
    ]
    return {"match": not mismatched, "generations": len(left), "mismatched_generations": mismatched}


def _check(args) -> dict:
    """Run the ``check-determinism`` subcommand and return the written record."""
    init_info = _ensure_gs_init(False)
    first = json.loads(Path(args.a).read_text(encoding="utf-8"))
    second = json.loads(Path(args.b).read_text(encoding="utf-8"))
    record = common.new_record(
        kind="neat_determinism",
        command=sys.argv,
        task=first.get("task", "unknown"),
        seed=int(first.get("seed", 0)),
        repeat=int(second.get("repeat", 0)),
        process_role="only",
        workload={
            "records": [str(args.a), str(args.b)],
            "rule": (
                "the seed table, the per-generation seeds, every fitness vector, the held-out "
                "fitness of both sides and the sha256 fingerprint must be identical, not merely close"
            ),
            "record_kinds": [first.get("kind"), second.get("kind")],
            "process_roles": [first.get("process_role"), second.get("process_role")],
        },
    )
    record["genesis_init"] = init_info
    _filter_own_gpu_pids(record)
    run_guarded(lambda rec: _fill_check(args, first, second, rec), record)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"neat_determinism_{record['task']}.json"
    write_record(out_path, record)
    _print_check(record, out_path)
    return record


def _fill_check(args, first: dict, second: dict, record: dict) -> None:
    """Compare the two records and set the record's verdict (``status`` fails on any mismatch)."""
    for label, side in (("A", first), ("B", second)):
        if side.get("status") != "ok":
            raise ValueError(f"record {label} ({args.a if label == 'A' else args.b}) is not a successful run")
    if first.get("task") != second.get("task"):
        raise ValueError(f"records measure different tasks: {first.get('task')!r} vs {second.get('task')!r}")
    if int(first.get("seed", 0)) != int(second.get("seed", 0)):
        raise ValueError(f"records use different master seeds: {first.get('seed')} vs {second.get('seed')}")
    left = _record_arrays(first)
    right = _record_arrays(second)
    checks = {
        "task": {"match": True, "value": first.get("task")},
        "seed": {"match": True, "value": int(first.get("seed", 0))},
        "seed_table": _array_compare(left["seed_table"], right["seed_table"]),
        "generation_seeds": _seeds_compare(left["generation_seeds"], right["generation_seeds"]),
        "fitness": _array_compare(left["fitness"], right["fitness"]),
        "holdout_initial": _array_compare(left["holdout_initial"], right["holdout_initial"]),
        "holdout_trained": _array_compare(left["holdout_trained"], right["holdout_trained"]),
        "fingerprint": {
            "match": bool(left["fingerprint"]) and left["fingerprint"] == right["fingerprint"],
            "a": left["fingerprint"],
            "b": right["fingerprint"],
        },
    }
    failures = [name for name, entry in checks.items() if not entry["match"]]
    record["measurements"] = {
        "records": {
            "a": {
                "path": str(args.a),
                "kind": first.get("kind"),
                "status": first.get("status"),
                "repeat": first.get("repeat"),
                "process_role": first.get("process_role"),
                "created_utc": first.get("created_utc"),
            },
            "b": {
                "path": str(args.b),
                "kind": second.get("kind"),
                "status": second.get("status"),
                "repeat": second.get("repeat"),
                "process_role": second.get("process_role"),
                "created_utc": second.get("created_utc"),
            },
        },
        "checks": checks,
        "verdict": {
            "ok": not failures,
            "failures": failures,
            "rule": "exact equality of the seed table, seeds, fitness, held-out fitness and fingerprint",
        },
    }
    if failures:
        record["status"] = "failed"
        record["error"] = {
            "type": "DeterminismMismatch",
            "message": f"{len(failures)} check(s) differ between the two processes: {', '.join(failures)}",
            "traceback_tail": [],
        }


# --- printing ----------------------------------------------------------------------


def _print_summary(record: dict, out_path: Path) -> None:
    """Print the reviewable one-screen summary of a finished (or failed) training record."""
    print(f"[bench] {record['kind']} task={record['task']} seed={record['seed']} repeat={record['repeat']} role={record['process_role']}")
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
    phases = record["phases"]
    print(
        f"[bench] {measurements['task']}: pop={measurements['pop_size']} rows={measurements['rows']} "
        f"horizon={record['workload']['horizon']} residency={measurements['residency']} "
        f"{measurements['backend']}@{measurements['device_name']} tell={measurements['tell']['backend']}"
        f"@{measurements['tell']['device']}"
    )
    print(
        f"[bench] phases: import={phases['import_s']}s build={phases['build_s']}s "
        f"graph={phases['graph_build_total_s']}s setup={phases['setup_s']}s "
        f"tell_build={phases['tell_build_s']}s first_gen={phases['first_generation_s']}s "
        f"steady={phases['steady_generation_median_s']}s close={phases['close_s']}s"
    )
    graphs = ", ".join(f"{name}={slot['build_s']:.2f}s" for name, slot in phases["graph_build_s"].items())
    print(f"[bench]   compile: {graphs}")
    for entry in measurements["generations"]:
        print(
            f"[bench]   gen {entry['generation']} best={entry['best_fitness']:+.4f} "
            f"median={entry['median_fitness']:+.4f} eval={entry['evaluate_s']:.3f}s "
            f"tell={entry['tell_s']:.3f}s gen={entry['generation_s']:.3f}s "
            f"steps={entry['steps']} {entry['env_steps_per_s']:.0f} env-steps/s "
            f"(harness {entry['env_steps_per_s_harness']:.0f})"
        )
    totals = measurements["totals"]
    print(
        f"[bench] totals: {totals['steps']} steps {totals['env_steps']} env-steps in {totals['loop_s']}s "
        f"({totals['env_steps_per_s_loop']} env-steps/s); generation median "
        f"{totals['generation_s_stats']['median']}s +/- {totals['generation_s_stats']['spread']}s; "
        f"best {totals['best_fitness']}"
    )
    holdout = measurements["holdout"]
    print(
        f"[bench] held-out {holdout['seed_table'][0]}: median initial={holdout['median_initial']} "
        f"trained={holdout['median_trained']} delta={holdout['median_delta']}; best policy "
        f"{holdout['best_policy']['index']} train={holdout['best_policy']['train_fitness']} "
        f"heldout={holdout['best_policy']['heldout_fitness']}"
    )
    checks = measurements["checks"]
    print(
        f"[bench] checks: no_build_during_run={checks['no_graph_build_during_run']} "
        f"holdout_disjoint={checks['holdout_disjoint']} fingerprint={checks['determinism_fingerprint'][:16]}"
    )


def _print_check(record: dict, out_path: Path) -> None:
    """Print the verdict of a determinism comparison."""
    print(f"[bench] {record['kind']} task={record['task']} seed={record['seed']}")
    print(f"[bench] status={record['status']} git={record['git']['head'][:12]} dirty={record['git']['dirty']}")
    print(f"[bench] record -> {out_path}")
    measurements = record.get("measurements") or {}
    for name, entry in (measurements.get("checks") or {}).items():
        detail = ""
        if not entry.get("match"):
            differences = {key: value for key, value in entry.items() if key != "match"}
            detail = f" ({differences})"
        print(f"[bench]   {'ok  ' if entry.get('match') else 'FAIL'} {name}{detail}")
    verdict = measurements.get("verdict") or {}
    print(f"[bench] verdict: {'identical' if verdict.get('ok') else 'DIFFERENT'} ({verdict.get('failures')})")


# --- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the Phase-3 trainer benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_neat.py",
        description="Stage-3 trainer benchmark: one TensorNEAT training run, timed per region.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="time one pilot training run (evaluate + tell per generation) and its held-out score",
        description=(
            "Build one pilot environment, build NeuroevolutionTrainer, run the generations with "
            "env.evaluate and the tell graph timed separately, then score the initial and trained "
            "populations on a fresh held-out seed triple."
        ),
    )
    train.add_argument("--task", required=True, choices=("go2", "hover"), help="pilot task to train")
    train.add_argument("--master-seed", type=int, default=0, help="seed of both RNG streams")
    train.add_argument("--pop-size", type=int, default=8, help="policies per generation (rows = 3 x this)")
    train.add_argument("--generations", type=int, default=3, help="generations to train (and time)")
    train.add_argument("--horizon", type=int, default=200, help="scene.step() calls per generation")
    train.add_argument("--residency", default="numpy", choices=("numpy", "device"), help="graph operand residency")
    train.add_argument("--backend", default="numpy", help="evoxc backend of the rollout graphs")
    train.add_argument("--device", default=None, help="evoxc device of the rollout graphs (default: env resolution)")
    train.add_argument(
        "--tell-backend",
        choices=("numpy", "xla"),
        default=None,
        help="tell backend; default the trainer's own resolution (numpy env -> numpy, ROCm -> xla)",
    )
    train.add_argument("--pjrt-plugin", default=None, help="XLA/PJRT plugin .so for the xla tell backend")
    train.add_argument("--gpu", action="store_true", help="initialise genesis on the GPU (gs.gpu physics)")
    train.add_argument("--repeat", type=int, default=0, help="replica index; nonzero marks a re-run")
    train.add_argument(
        "--process-role",
        default="only",
        choices=("cold", "warm", "only"),
        help="where this process sits in the warm-up chain (cold/warm compile cache)",
    )
    train.add_argument("--out", default=None, help="record path (default: results/neat_<task>_seed<n>*.json)")

    check = subparsers.add_parser(
        "check-determinism",
        help="compare two train records of the same task/seed and record the verdict",
        description=(
            "Compare the seed table, the per-generation seeds, every fitness vector, the held-out "
            "fitness and the sha256 fingerprint of two train records; the new record's status is "
            "ok only when every one of them is identical."
        ),
    )
    check.add_argument("a", help="first train record (usually the primary replica)")
    check.add_argument("b", help="second train record (a re-run in a fresh process)")
    check.add_argument("--out", default=None, help="record path (default: results/neat_determinism_<task>.json)")
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
    if args.command == "train":
        if args.pop_size < 1 or args.generations < 1 or args.horizon < 1:
            raise ValueError("--pop-size, --generations and --horizon must be positive")
        record = _train(args)
    elif args.command == "check-determinism":
        record = _check(args)
    else:
        raise SystemExit(f"unsupported subcommand {args.command!r}")
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
