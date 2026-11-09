"""Stage-3 Phase-4 probes: unmatched baselines, backend capability and the torch/CPU boundary.

The Phase 1-3 harness measured the *migrated* environments and trainer. This CLI records what the
migration did **not** measure, so the Stage-3 report states its gaps instead of implying coverage:

``unmatched``
    The pre-migration baselines (``examples/locomotion/go2_env.py`` + ``go2_train.py`` and
    ``examples/drone/hover_env.py`` + ``hover_train.py``) cannot be driven on this host: they import
    ``tensordict``/``rsl_rl``/``rsl-rl-lib`` at import time and none is installed. Every import,
    both thin-adapter attempts and one offline install attempt run as separate subprocesses with a
    hard timeout and a process-group kill, and each failure is recorded verbatim. The structured
    ``interface_mismatch`` block states why an adapter would not be enough even with the
    dependencies installed; PPO is recorded separately as an unmatched *algorithm* baseline.

``tell-backend``
    The documented Phase-3 finding is that the IREE/ROCm ``tell`` graph never compiles. The probe
    reproduces that in a child process (:mod:`_tell_worker`) that prints a phase marker before every
    step, so a timeout records how far the build got; the NumPy control and the XLA/ROCm positive
    control run through the same worker. The process group is killed on timeout and
    ``rocm-smi --showpids`` is recorded afterwards, proving nothing is left running.

``torch-cpu``
    The torch/CPU boundary of the migrated stack, from two independent sources: an AST scan of the
    neuroevolution and pilot modules plus one fresh-subprocess ``import genesis`` probe
    (:mod:`probe_sources`, :func:`_runtime_probe`), and the already-committed boundary numbers of
    ``results/`` cited with their source filename and git head (never re-measured).

Usage (from the repository root)::

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_probe.py unmatched
    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_probe.py tell-backend --all
    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_probe.py torch-cpu

A bare invocation prints the usage and exits 0, like the other benchmark CLIs, so the
``examples/**/*.py`` smoke harness does not treat this tool as a failing example.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import common
import probe_sources
from bench_envs import _r
from common import run_guarded, write_record
from probe_runtime import (
    CASE_TIMEOUT_S,
    DEFAULT_XLA_PLUGIN,
    INSTALL_TIMEOUT_S,
    TELL_TIMEOUT_S,
    _exception_from_stderr,
    gpu_after,
    import_case,
    install_attempt,
    probe_env,
    run_capture,
    run_tell_entry,
)
from probe_sources import BASELINES, RESULTS_DIR, interface_mismatch, measured_boundary, torch_scan
from runtime_measure import _filter_own_gpu_pids


def _new_probe_record(kind: str, task: str, workload: dict) -> dict:
    """Build a probe record with the standard provenance block, own GPU PIDs filtered out."""
    record = common.new_record(
        kind=kind,
        command=sys.argv,
        task=task,
        seed=0,
        repeat=0,
        process_role="only",
        workload=workload,
    )
    _filter_own_gpu_pids(record)
    return record


# --- unmatched baselines ----------------------------------------------------------
def _ppo_block(import_cases: dict) -> dict:
    """The unmatched PPO algorithm baseline, kept separate from the environment comparison."""
    rsl = import_cases.get("rsl_rl", {})
    train = import_cases.get("go2_train", {})
    return {
        "status": "unmatched",
        "reason": (
            "PPO is trained by rsl-rl against the torch env; rsl_rl is not installed and go2_train/"
            "hover_train refuse to import without rsl-rl-lib>=5.0.0, so no PPO rollout was run"
        ),
        "error": {
            "type": (train.get("exception") or {}).get("type"),
            "message": (train.get("exception") or {}).get("message"),
            "source": "import go2_train",
        },
        "module_error": {
            "type": (rsl.get("exception") or {}).get("type"),
            "message": (rsl.get("exception") or {}).get("message"),
            "source": "import rsl_rl",
        },
        "required": ["rsl-rl-lib>=5.0.0", "tensordict", "tensorboard"],
        "training_run": False,
        "evaluation_run": False,
        "note": "algorithm comparison only; not substituted for the environment comparison",
    }


def _fill_unmatched(args, record: dict) -> None:
    """Run every baseline case, the adapter attempts and the install attempt; fill the record."""
    import_cases = {}
    started = time.perf_counter()
    for name, code, extra_dir in (
        ("tensordict", "import tensordict", None),
        ("rsl_rl", "import rsl_rl", None),
        ("go2_env", "import go2_env", BASELINES["go2"]["import_dir"]),
        ("hover_env", "import hover_env", BASELINES["hover"]["import_dir"]),
        ("go2_train", "import go2_train", BASELINES["go2"]["import_dir"]),
        ("hover_train", "import hover_train", BASELINES["hover"]["import_dir"]),
    ):
        import_cases[name] = import_case(name, code, extra_dir, args.timeout)
    imports_s = time.perf_counter() - started

    started = time.perf_counter()
    adapter_cases = {}
    for task, spec in BASELINES.items():
        code = f"from {spec['env_module']} import {spec['env_class']}; {spec['env_class']}(1, {{}}, {{}}, {{}}, {{}})"
        adapter_cases[task] = import_case(f"{spec['env_module']}_adapter", code, spec["import_dir"], args.timeout)
    adapters_s = time.perf_counter() - started

    started = time.perf_counter()
    install = install_attempt(sys.executable)
    install_s = time.perf_counter() - started

    started = time.perf_counter()
    interface = {
        task: interface_mismatch(task, import_cases[spec["env_module"]], adapter_cases[task])
        for task, spec in BASELINES.items()
    }
    scan_s = time.perf_counter() - started

    record["workload"].update(
        {
            "python": sys.executable,
            "case_timeout_s": args.timeout,
            "install_timeout_s": INSTALL_TIMEOUT_S,
            "pythonpath_roots": {
                "repo": str(common.repo_root()),
                "go2": str(common.repo_root() / BASELINES["go2"]["import_dir"]),
                "hover": str(common.repo_root() / BASELINES["hover"]["import_dir"]),
            },
            "baselines": {task: spec["env"] for task, spec in BASELINES.items()},
            "adapter_attempts": {
                "go2": "from go2_env import Go2Env; Go2Env(1, {}, {}, {}, {})",
                "hover": "from hover_env import HoverEnv; HoverEnv(1, {}, {}, {}, {})",
            },
            "note": (
                "every case runs in its own session with a hard timeout and a process-group kill; "
                "the import failure is the measurement"
            ),
        }
    )
    record["phases"] = {
        "imports_s": _r(imports_s),
        "adapters_s": _r(adapters_s),
        "install_s": _r(install_s),
        "scan_s": _r(scan_s),
    }
    record["timings"] = {"note": "phase seconds are in phases; per-case elapsed_s is in measurements.imports/adapter_attempts"}
    record["measurements"] = {
        "imports": import_cases,
        "adapter_attempts": adapter_cases,
        "adapter_verdict": {
            task: {
                "attempted": True,
                "blocked_at": "import",
                "observed_exception": entry["exception"],
                "verdict": interface[task]["verdict"],
            }
            for task, entry in adapter_cases.items()
        },
        "interface_mismatch": interface,
        "ppo": _ppo_block(import_cases),
        "install_attempt": install,
        "verdict": {
            "measured_environments": [],
            "unmatched_environments": list(BASELINES),
            "unmatched_algorithms": ["ppo"],
            "rule": (
                "an environment is 'measured' only when the trainer drove it through "
                "env.evaluate(pop_arrays, generation); neither baseline could be imported, so both are unmatched"
            ),
        },
    }


# --- tell backend -----------------------------------------------------------------
def _tell_specs(args) -> list[dict]:
    """Return the (backend, device, plugin) entries to run: one explicit spec or the 3-entry matrix."""
    if args.backend and not args.all:
        return [{"backend": args.backend, "device": args.device, "plugin_path": args.pjrt_plugin}]
    plugin = args.pjrt_plugin
    if plugin is None and Path(DEFAULT_XLA_PLUGIN).is_file():
        plugin = DEFAULT_XLA_PLUGIN
    device = "rocm:0" if args.device is None else args.device
    return [
        {"backend": "numpy", "device": "cpu:0", "plugin_path": None},
        {"backend": "iree", "device": device, "plugin_path": None},
        {"backend": "xla", "device": device, "plugin_path": plugin},
    ]


def _fill_tell_backend(args, record: dict) -> None:
    """Run each backend entry, then record the post-probe GPU/process state and the verdict."""
    specs = _tell_specs(args)
    matrix = args.all or args.backend is None
    entries = {}
    started = time.perf_counter()
    for spec in specs:
        entries[f"{spec['backend']}@{spec['device']}"] = run_tell_entry(spec, args.timeout)
    record["workload"].update(
        {
            "worker": "_tell_worker.py",
            "entries": [f"{spec['backend']}@{spec['device']}" for spec in specs],
            "genome": {"num_inputs": 45, "num_outputs": 12, "pop_size": 4, "species_size": 2, "max_nodes": 80, "max_conns": 640},
            "pipeline": "Pipeline(DefaultGenome(45,12), TraceOnlyProblem(45,12), seed=0, backend='numpy')",
            "per_entry_timeout_s": args.timeout,
            "note": (
                "each entry is a separate process group; a timeout kills the group (the IREE compile "
                "may spawn children) and records the last printed phase marker"
            ),
        }
    )
    record["phases"] = {"probe_total_s": _r(time.perf_counter() - started), "entries": len(entries)}
    record["timings"] = {"note": "per-entry build/run seconds are in measurements.entries[*].result"}
    record["measurements"] = {"entries": entries, "gpu_after": gpu_after()}
    expectations = {"numpy": "ok", "iree": "timeout", "xla": "ok"} if matrix else {}
    verdicts = {}
    for label, entry in entries.items():
        expected = expectations.get(entry["backend"])
        verdicts[label] = {
            "status": entry["status"],
            "expected": expected,
            "matches_expectation": (expected is None or entry["status"] == expected)
            or (expected == "ok" and entry["status"] == "skipped"),
        }
    iree_entry = next((entry for entry in entries.values() if entry["backend"] == "iree"), None)
    record["measurements"]["verdict"] = {
        "matrix": matrix,
        "expectations": expectations,
        "entries": verdicts,
        "iree_rocm_documented_failure_reproduced": (
            iree_entry is not None and iree_entry["status"] == "timeout" if matrix else None
        ),
        "gpu_free_after": record["measurements"]["gpu_after"]["gpu_free"],
        "no_worker_left": not any(entry.get("worker_processes_left") for entry in entries.values()),
    }
    failures = [label for label, entry in verdicts.items() if not entry["matches_expectation"]]
    if failures:
        record["status"] = "failed"
        record["error"] = {
            "type": "UnexpectedTellOutcome",
            "message": f"tell backend(s) did not behave as documented: {', '.join(failures)}",
            "traceback_tail": [],
        }


# --- torch / CPU boundary ---------------------------------------------------------
_RUNTIME_PROBE = r"""
import json
import sys
import time

report = {"torch_in_sys_modules_before_genesis": "torch" in sys.modules}
started = time.perf_counter()
import genesis
report["genesis_import_s"] = round(time.perf_counter() - started, 3)
report["genesis_version"] = getattr(genesis, "__version__", None)
report["genesis_initialized"] = bool(getattr(genesis, "_initialized", False))
report["genesis_device"] = repr(getattr(genesis, "device", None))
report["torch_in_sys_modules_after_import_genesis"] = "torch" in sys.modules
import torch
report["torch_version"] = torch.__version__
report["torch_hip_version"] = getattr(torch.version, "hip", None)
report["torch_cuda_version"] = getattr(torch.version, "cuda", None)
available = bool(torch.cuda.is_available())
report["torch_cuda_is_available"] = available
report["hip_device_visible"] = bool(available and getattr(torch.version, "hip", None))
report["torch_cuda_device_count"] = torch.cuda.device_count() if available else 0
if available:
    try:
        report["hip_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["hip_device_error"] = f"{type(exc).__name__}: {exc}"
import importlib.util
report["find_spec"] = {}
for name in ("torch", "tensordict", "rsl_rl", "evoxc", "tensorneat"):
    try:
        report["find_spec"][name] = importlib.util.find_spec(name) is not None
    except Exception as exc:
        report["find_spec"][name] = f"{type(exc).__name__}: {exc}"
print("PROBE " + json.dumps(report))
"""


def _runtime_probe(timeout: float) -> dict:
    """Import ``genesis`` in a fresh process and record whether it pulls ``torch`` and HIP."""
    run = run_capture([sys.executable, "-c", _RUNTIME_PROBE], timeout=timeout, env=probe_env(), cwd=common.repo_root())
    stdout = run.pop("_stdout")
    stderr = run.pop("_stderr")
    report = None
    for line in stdout.splitlines():
        if line.startswith("PROBE "):
            try:
                report = json.loads(line[len("PROBE ") :])
            except json.JSONDecodeError:
                report = None
    return {
        "method": "fresh subprocess, import genesis + torch, no scene",
        "command": run["command"],
        "returncode": run["returncode"],
        "timed_out": run["timed_out"],
        "elapsed_s": run["elapsed_s"],
        "report": report,
        "exception": None if report else _exception_from_stderr(stderr),
        "stderr_tail": run["stderr_tail"],
    }


def _fill_torch_cpu(args, record: dict) -> None:
    """Fill the AST scan, the runtime probe and the committed measured boundary numbers."""
    started = time.perf_counter()
    scan = torch_scan()
    scan_s = time.perf_counter() - started
    started = time.perf_counter()
    runtime = _runtime_probe(args.timeout)
    probe_s = time.perf_counter() - started
    record["workload"].update(
        {
            "scan_targets": probe_sources.scan_targets(),
            "rule": (
                "AST evidence plus one fresh-subprocess import probe; the committed records are cited, "
                "never re-measured, so a number can be traced to its source file and git head"
            ),
            "measured_record_glob": [
                "results/env_<task>_replay_seed0.json",
                "results/env_<task>_throughput_numpy_cpu_pop8.json",
                "results/residency_<task>_gpu.json",
            ],
            "runtime_probe_timeout_s": args.timeout,
        }
    )
    record["phases"] = {"ast_scan_s": _r(scan_s), "runtime_probe_s": _r(probe_s)}
    record["timings"] = {"note": "phase seconds are in phases; boundary numbers are copied from the source records"}
    record["measurements"] = {"ast_scan": scan, "runtime_probe": runtime, "measured_boundary": measured_boundary()}


# --- printing ---------------------------------------------------------------------
def _print_unmatched(record: dict, out_path: Path) -> None:
    """Print the reviewable summary of the unmatched-baseline record."""
    print(f"[probe] {record['kind']} status={record['status']} git={record['git']['head'][:12]}")
    print(f"[probe] record -> {out_path}")
    measurements = record.get("measurements") or {}
    for name, entry in (measurements.get("imports") or {}).items():
        exc = entry.get("exception") or {}
        print(f"[probe]   import {name:<12} rc={entry['returncode']} {exc.get('type')}: {exc.get('message')}")
    for name, entry in (measurements.get("adapter_attempts") or {}).items():
        exc = entry.get("exception") or {}
        print(f"[probe]   adapter {name:<6} rc={entry['returncode']} blocked_at=import {exc.get('type')}: {exc.get('message')}")
    install = measurements.get("install_attempt") or {}
    print(
        f"[probe]   install: status={install.get('status')} pip={install.get('pip', {}).get('returncode')} "
        f"uv={install.get('uv', {}).get('available')}"
    )
    print(f"[probe]   unmatched={measurements.get('verdict', {}).get('unmatched_environments')} ppo={measurements.get('ppo', {}).get('status')}")


def _print_tell(record: dict, out_path: Path) -> None:
    """Print one line per backend entry and the documented-failure verdict."""
    print(f"[probe] {record['kind']} status={record['status']} git={record['git']['head'][:12]}")
    print(f"[probe] record -> {out_path}")
    measurements = record.get("measurements") or {}
    for label, entry in (measurements.get("entries") or {}).items():
        result = entry.get("result") or {}
        print(
            f"[probe]   {label:<14} status={entry['status']:<8} elapsed={entry.get('elapsed_s')}s "
            f"last_phase={entry.get('last_phase')} build={result.get('build_s')}s run={result.get('run_s')}s"
        )
    gpu = measurements.get("gpu_after") or {}
    print(f"[probe]   gpu_free_after={gpu.get('gpu_free')} pids={gpu.get('rocm_pids')}")
    print(f"[probe]   verdict={measurements.get('verdict', {}).get('iree_rocm_documented_failure_reproduced')}")


def _print_torch(record: dict, out_path: Path) -> None:
    """Print the AST summary and the runtime-probe facts."""
    print(f"[probe] {record['kind']} status={record['status']} git={record['git']['head'][:12]}")
    print(f"[probe] record -> {out_path}")
    measurements = record.get("measurements") or {}
    summary = (measurements.get("ast_scan") or {}).get("summary") or {}
    print(f"[probe]   torch at import: {summary.get('torch_at_import_time')}")
    print(f"[probe]   torch only inside functions: {summary.get('torch_only_inside_functions')}")
    report = (measurements.get("runtime_probe") or {}).get("report") or {}
    print(
        f"[probe]   genesis pulls torch: {report.get('torch_in_sys_modules_after_import_genesis')} "
        f"torch={report.get('torch_version')} hip={report.get('torch_hip_version')} "
        f"cuda_available={report.get('torch_cuda_is_available')}"
    )


# --- CLI --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the Phase-4 probe CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_probe.py",
        description="Stage-3 Phase-4 probes: unmatched baselines, tell backends and the torch/CPU boundary.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    unmatched = subparsers.add_parser(
        "unmatched", help="record the import/adapter/install failures of the torch baselines and the PPO baseline"
    )
    unmatched.add_argument("--timeout", type=float, default=CASE_TIMEOUT_S, help="hard timeout of one case subprocess")
    unmatched.add_argument("--out", default=str(RESULTS_DIR / "unmatched_baselines.json"))
    tell = subparsers.add_parser(
        "tell-backend", help="probe the tell-graph compile per backend (iree@rocm:0 is expected to time out)"
    )
    tell.add_argument("--backend", default=None, help="run only this evoxc backend (default: the 3-entry matrix)")
    tell.add_argument("--device", default=None, help="evoxc device (default: cpu:0 numpy, rocm:0 gpu)")
    tell.add_argument("--pjrt-plugin", default=None, help=f"XLA/PJRT plugin .so (default: {DEFAULT_XLA_PLUGIN})")
    tell.add_argument("--timeout", type=float, default=TELL_TIMEOUT_S, help="hard timeout of one worker process")
    tell.add_argument("--all", action="store_true", help="run the numpy/iree/xla matrix even with --backend")
    tell.add_argument("--out", default=str(RESULTS_DIR / "tell_backend_rocm.json"))
    torch_cpu = subparsers.add_parser(
        "torch-cpu", help="AST-scan the torch boundary and cite the committed per-step boundary numbers"
    )
    torch_cpu.add_argument("--timeout", type=float, default=180.0, help="hard timeout of the import genesis probe")
    torch_cpu.add_argument("--out", default=str(RESULTS_DIR / "torch_cpu_boundary.json"))
    return parser


def main(argv=None) -> int:
    """Parse ``argv`` and run the requested probe; a bare invocation prints usage and exits 0."""
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    common.setup_paths()
    if args.command == "unmatched":
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        record = _new_probe_record(
            kind="probe_unmatched_baselines",
            task="go2+hover",
            workload={"probe": "unmatched", "cases": ["tensordict", "rsl_rl", "go2_env", "hover_env", "go2_train", "hover_train"]},
        )
        run_guarded(lambda rec: _fill_unmatched(args, rec), record)
    elif args.command == "tell-backend":
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        record = _new_probe_record(
            kind="probe_tell_backend", task="go2", workload={"probe": "tell_backend", "matrix": args.all or args.backend is None}
        )
        run_guarded(lambda rec: _fill_tell_backend(args, rec), record)
    elif args.command == "torch-cpu":
        record = _new_probe_record(kind="probe_torch_cpu_boundary", task="go2+hover", workload={"probe": "torch_cpu"})
        run_guarded(lambda rec: _fill_torch_cpu(args, rec), record)
    else:
        raise SystemExit(f"unsupported subcommand {args.command!r}")
    out_path = Path(args.out)
    write_record(out_path, record)
    {"unmatched": _print_unmatched, "tell-backend": _print_tell, "torch-cpu": _print_torch}[args.command](record, out_path)
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
