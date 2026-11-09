"""Shared plumbing of the Stage-3 neuroevolution benchmark harness (records, host facts, timers).

This module is the part of ``examples/neuroevolution/benchmarks`` that every benchmark subcommand
needs and that must work **without importing a simulator**: nothing here imports ``genesis``,
``torch``, ``tensordict`` or ``evoxc`` at module level, so the record schema, the statistics
helpers and the timer wrappers are testable on a machine with none of them installed.

What it owns
------------
* :func:`setup_paths` -- the ``sys.path`` shim that makes the sibling example modules
  (``examples/locomotion``, ``examples/drone``, ``examples/neuroevolution``) importable by bare
  name, exactly as the example tree imports them.
* :func:`git_info`, :func:`packages`, :func:`host_info`, :func:`gpu_info` -- the provenance block
  of a record: which revision produced it, what was installed, and what else was running on the
  host when it was measured (``/proc/loadavg`` and ``rocm-smi --showpids`` are the two contention
  signals the Stage-3 plan asks for).
* :func:`new_record` / :func:`write_record` -- the one JSON schema every record file uses, so a
  later phase can diff records across tasks and residencies.
* :func:`stats`, :class:`Timer`, :func:`wrap_function` / :func:`wrap_method`, :func:`sync`,
  :func:`run_guarded` -- measurement mechanics: named accumulators filled by runtime
  monkey-patching of read-only sibling modules (files stay untouched; the patches are restored by
  the returned callable).

Conventions
-----------
Records are JSON with ``indent=2`` and sorted keys; :func:`write_record` caps sample lists so a
record stays well under ~150 KB, because these files are committed next to the code.
Every timer stores seconds and call counts, never per-call samples, for the same reason.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

#: Git revision of the pre-migration (PyTorch) environments the benchmarks compare against.
#: ``git diff ad2f07e5 -- examples/locomotion/go2_env.py examples/drone/hover_env.py
#: examples/locomotion/go2_train.py examples/drone/hover_train.py`` is empty (recorded per run).
BASELINE_COMMIT = "ad2f07e5"

#: Paths of the four baseline files that define the pre-migration behaviour; the diff against
#: :data:`BASELINE_COMMIT` is recorded so a reader can see the baseline is unchanged.
BASELINE_ENV_PATHS = (
    "examples/locomotion/go2_env.py",
    "examples/drone/hover_env.py",
    "examples/locomotion/go2_train.py",
    "examples/drone/hover_train.py",
)

#: Distribution names recorded in every record (the migration stack of this benchmark).
PACKAGE_NAMES = (
    "evoxc",
    "tensorneat",
    "iree-base-compiler",
    "iree-base-runtime",
    "torch",
    "quadrants",
    "genesis-world",
    "numpy",
)

#: Version of the record schema. Bump it when a key is renamed or removed, never for an addition.
SCHEMA = "neuroevolution-benchmark/1"

#: 1-minute load average above which a measurement is flagged ``contended`` (the GPU is shared).
CONTENTION_LOADAVG = 4.0

#: Hard ceiling on a serialised record; sample lists are truncated to stay below it.
MAX_RECORD_BYTES = 150_000

#: The four sibling directories a benchmark run needs on ``sys.path``.
_SIBLING_DIRS = ("examples/locomotion", "examples/drone", "examples/neuroevolution")


def repo_root() -> Path:
    """Return the repository root of this worktree (``.../examples/neuroevolution/benchmarks``)."""
    return Path(__file__).resolve().parents[3]


def setup_paths() -> list[str]:
    """Put the repository root and the three sibling example directories on ``sys.path``.

    The example tree imports its shared modules by bare name (``from seeding import ...``), so a
    benchmark script has to reproduce that layout itself; the CLI is launched as a real file from
    the repository root, which already puts this directory first.

    Returns:
        list[str]: The entries that were missing and got inserted, in insertion order (empty when
        :func:`setup_paths` was already called).
    """
    root = repo_root()
    wanted = [str(root)] + [str(root / relative) for relative in _SIBLING_DIRS]
    added = []
    for entry in wanted:
        if entry not in sys.path:
            sys.path.insert(0, entry)
            added.append(entry)
    return added


def _git(*args: str) -> str:
    """Run one ``git`` command in the repository root and return its stripped stdout."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_info() -> dict:
    """Describe the revision a record was produced from.

    Returns:
        dict: ``head`` (full sha), ``branch``, ``dirty`` (whether ``git status --porcelain``
        reports anything, i.e. whether the worktree differs from ``head`` while running),
        ``dirty_files`` (first 20 porcelain entries), ``baseline`` (:data:`BASELINE_COMMIT`),
        ``baseline_sha`` (its resolved full sha), ``baseline_env_diff`` (the diff of
        :data:`BASELINE_ENV_PATHS` against the baseline) and ``baseline_env_diff_empty`` (whether
        the pre-migration environments are unchanged, i.e. the baseline is what we think it is).
    """
    porcelain = _git("status", "--porcelain", "--untracked-files=normal")
    diff = _git("diff", BASELINE_COMMIT, "--", *BASELINE_ENV_PATHS)
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(porcelain),
        "dirty_files": porcelain.splitlines()[:20],
        "baseline": BASELINE_COMMIT,
        "baseline_sha": _git("rev-parse", BASELINE_COMMIT),
        "baseline_env_diff": diff,
        "baseline_env_diff_empty": diff == "",
    }


def packages() -> dict:
    """Return ``{distribution: version or None}`` for :data:`PACKAGE_NAMES`."""
    found = {}
    for name in PACKAGE_NAMES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
        except Exception:  # noqa: BLE001 - metadata can raise on a broken dist
            found[name] = None
    return found


def _first_lines(text: str, limit: int = 24) -> list[str]:
    """Return the first ``limit`` non-empty lines of ``text``, without the banner rules."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[:limit]


def _run_tool(argv: list[str], timeout: float = 30.0) -> str:
    """Run an external tool and return its combined output (``""`` when it cannot run)."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def rocm_pids(text: str) -> list[str]:
    """Extract the PIDs listed in a ``rocm-smi --showpids`` dump.

    The tool prints either ``No KFD PIDs currently running`` (an idle GPU) or one table row per
    running process; both are reduced to a list of PID strings here so a record can say whether
    somebody else was using the GPU during a measurement.
    """
    if "No KFD PIDs currently running" in text:
        return []
    pids = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("=", "WARNING", "GPU[")):
            continue
        head = stripped.split()[0]
        if head.isdigit():
            pids.append(head)
    return pids


def host_info(with_rocm: bool = True) -> dict:
    """Return the host facts recorded with every measurement (load, cores, GPU driver, GPU users).

    Args:
        with_rocm (bool): Run the ``rocm-smi`` queries. ``False`` skips them for a fast record
            (the acceptance check of the harness itself does not need them).

    Returns:
        dict: ``loadavg_text`` / ``loadavg_1m`` (the first field of ``/proc/loadavg``),
        ``cpu_count``, ``python``, ``platform``, ``rocm_product`` / ``rocm_driver`` (first lines of
        the two ``rocm-smi`` queries), ``rocm_pids`` (PIDs listed by ``rocm-smi --showpids``) and
        ``gpu_busy`` (whether that list is non-empty).
    """
    loadavg_text = ""
    try:
        loadavg_text = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    except OSError:
        loadavg_text = ""
    info = {
        "loadavg_text": loadavg_text,
        "loadavg_1m": float(loadavg_text.split()[0]) if loadavg_text else None,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    if with_rocm:
        product = _run_tool(["rocm-smi", "--showproductname"])
        driver = _run_tool(["rocm-smi", "--showdriverversion"])
        pids_text = _run_tool(["rocm-smi", "--showpids"])
        pids = rocm_pids(pids_text)
        info.update(
            {
                "rocm_product": _first_lines(product),
                "rocm_driver": _first_lines(driver),
                "rocm_pids": pids,
                "gpu_busy": bool(pids),
            }
        )
    return info


def gpu_info() -> dict:
    """Return the ``torch`` view of the GPU (empty ``{}`` when torch cannot be imported).

    ``torch.cuda`` is HIP on this host, so ``torch.cuda.is_available()`` maps to ``hip``; the
    record keeps both the raw values and a ``torch_error`` when the import fails.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch is optional for this harness
        return {"available": False, "torch_error": f"{type(exc).__name__}: {exc}"}
    info = {"available": bool(torch.cuda.is_available()), "torch": torch.__version__}
    if info["available"]:
        try:
            info["device_count"] = torch.cuda.device_count()
            info["name"] = torch.cuda.get_device_name(0)
            info["hip"] = getattr(torch.version, "hip", None)
        except Exception as exc:  # noqa: BLE001 - a broken driver must not kill the record
            info["device_error"] = f"{type(exc).__name__}: {exc}"
    return info


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_record(kind: str, command, task: str, seed: int, repeat: int, process_role: str, workload: dict) -> dict:
    """Build the empty record skeleton of one measurement.

    Args:
        kind (str): Measurement kind, e.g. ``"env_replay"``.
        command (list[str]): The exact process command (``sys.argv``), recorded verbatim.
        task (str): Task name, ``"go2"`` or ``"hover"``.
        seed (int): The ``master_seed`` of the measured run.
        repeat (int): Replica index; a nonzero repeat is a determinism cross-check, not a new run.
        process_role (str): ``"cold"`` / ``"warm"`` / ``"only"`` -- where the process sits in the
            warm-up chain (a compile cache warmed by an earlier process is the difference).
        workload (dict): What was simulated (pop size, horizon, rows, residency, ...).

    Returns:
        dict: The record with every provenance block filled, ``phases`` / ``timings`` /
        ``measurements`` empty, ``status`` ``"ok"`` and ``error`` ``None``; a caller that catches an
        exception through :func:`run_guarded` gets ``"failed"`` and the error instead.
        ``contended`` is ``True`` when ``rocm-smi --showpids`` lists a process or the 1-minute load
        average exceeds :data:`CONTENTION_LOADAVG`, with the reasons in ``contention_reasons``, so a
        reader knows the timings were taken on a shared host.
    """
    host = host_info()
    load = host.get("loadavg_1m")
    reasons = []
    if host.get("gpu_busy"):
        reasons.append(f"rocm-smi --showpids lists {len(host.get('rocm_pids') or [])} process(es)")
    if load is not None and load > CONTENTION_LOADAVG:
        reasons.append(f"/proc/loadavg 1-minute average {load} > {CONTENTION_LOADAVG}")
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "kind": str(kind),
        "command": [str(part) for part in command],
        "git": git_info(),
        "packages": packages(),
        "host": host,
        "gpu": gpu_info(),
        "contended": bool(reasons),
        "contention_reasons": reasons,
        "task": str(task),
        "seed": int(seed),
        "repeat": int(repeat),
        "process_role": str(process_role),
        "workload": dict(workload),
        "phases": {},
        "timings": {},
        "measurements": {},
        "status": "ok",
        "error": None,
    }


def _cap_samples(node, limit: int):
    """Recursively truncate numeric sample lists longer than ``limit`` (safety net for file size)."""
    if isinstance(node, dict):
        return {key: _cap_samples(value, limit) for key, value in node.items()}
    if isinstance(node, list):
        if len(node) > limit and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in node):
            head = list(node[: limit // 2])
            tail = list(node[-limit // 2 :])
            return [*head, {"truncated_between": len(node) - limit}, *tail]
        return [_cap_samples(item, limit) for item in node]
    return node


def write_record(path, record: dict) -> Path:
    """Serialise ``record`` to ``path`` as sorted, indented JSON, creating parent directories.

    Sample lists are capped (progressively harder) until the encoded file is below
    :data:`MAX_RECORD_BYTES`; ``record["truncated"]`` records that a cap was applied so a reader
    knows the file is not lossless.

    Args:
        path (str | Path): Destination file, usually under ``examples/neuroevolution/benchmarks/results``.
        record (dict): The record built by :func:`new_record` (mutated only by the truncation note).

    Returns:
        Path: The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = None
    for limit in (256, 64, 16, 4):
        candidate = _cap_samples(record, limit)
        if limit != 256:
            candidate["truncated"] = {"sample_limit": limit}
        if len(json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False)) <= MAX_RECORD_BYTES:
            payload = candidate
            break
    if payload is None:
        # Last resort: drop the heaviest provenance block rather than write a 1 MB record.
        payload = _cap_samples(record, 4)
        payload["host"] = {key: payload["host"].get(key) for key in ("loadavg_1m", "cpu_count", "gpu_busy")}
        payload["truncated"] = {"sample_limit": 4, "dropped": "host.rocm_*"}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _finite(values) -> list[float]:
    """Return the finite floats of ``values`` (``None`` and non-finite entries dropped)."""
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def stats(values) -> dict:
    """Return the summary statistics of ``values`` (``n`` 0 and every field ``None`` when empty).

    ``spread`` is ``max - min`` and ``stdev`` the population standard deviation (``pstdev``), so a
    single sample gives ``0.0`` rather than raising.
    """
    sample = _finite(values)
    if not sample:
        return {"n": 0, "min": None, "max": None, "median": None, "mean": None, "spread": None, "stdev": None}
    return {
        "n": len(sample),
        "min": min(sample),
        "max": max(sample),
        "median": statistics.median(sample),
        "mean": statistics.fmean(sample),
        "spread": max(sample) - min(sample),
        "stdev": statistics.pstdev(sample),
    }


class Timer:
    """Named accumulators of seconds and call counts, filled by monkey-patched wrappers.

    A ``Timer`` never stores per-call samples: a benchmark generation can have thousands of calls
    and the record has to stay small, so only ``seconds`` and ``calls`` per name are kept, and
    :meth:`as_dict` derives ``mean_s``.

    Args:
        names (iterable of str, optional): Pre-create these accumulators at zero, so a record shows
            every measured group even when a task never calls it.
    """

    def __init__(self, names=()):
        self._data = {}
        self._order = []
        for name in names:
            self.add(name, 0.0, calls=0)

    def add(self, name: str, seconds: float, calls: int = 1) -> None:
        """Add ``seconds`` and ``calls`` to the accumulator ``name``."""
        entry = self._data.get(name)
        if entry is None:
            entry = {"seconds": 0.0, "calls": 0}
            self._data[name] = entry
            self._order.append(name)
        entry["seconds"] += float(seconds)
        entry["calls"] += int(calls)

    def count(self, name: str) -> int:
        """Return the number of calls recorded for ``name``."""
        entry = self._data.get(name)
        return 0 if entry is None else entry["calls"]

    def totals(self) -> dict:
        """Return ``{name: seconds}`` for every accumulator."""
        return {name: entry["seconds"] for name, entry in self._data.items()}

    def snapshot(self) -> dict:
        """Return an immutable copy of the accumulators (``{name: (seconds, calls)}``)."""
        return {name: (entry["seconds"], entry["calls"]) for name, entry in self._data.items()}

    def delta(self, before: dict, after: dict | None = None) -> dict:
        """Return ``{name: (seconds, calls)}`` between two :meth:`snapshot` results."""
        after = self.snapshot() if after is None else after
        names = list(dict.fromkeys([*before, *after]))
        out = {}
        for name in names:
            seconds, calls = after.get(name, (0.0, 0))
            seconds0, calls0 = before.get(name, (0.0, 0))
            out[name] = (seconds - seconds0, calls - calls0)
        return out

    def as_dict(self, snapshot: dict | None = None) -> dict:
        """Return ``{name: {"seconds", "calls", "mean_s"}}``, derived from :meth:`snapshot` data."""
        data = self.snapshot() if snapshot is None else snapshot
        out = {}
        for name, (seconds, calls) in data.items():
            out[name] = {
                "seconds": seconds,
                "calls": calls,
                "mean_s": (seconds / calls) if calls else None,
            }
        return out


def wrap_function(module, name: str, timer: Timer, tag: str, on_call=None, exclusive: bool = False):
    """Monkey-patch the module-level function ``module.name`` to accumulate into ``timer[tag]``.

    The wrapper keeps the signature transparent (``*args, **kwargs``) and the original name, and
    the elapsed time is added to ``timer`` in a ``finally`` block so a raising call still counts.

    Args:
        module: The module object owning the function (patched in place).
        name (str): Attribute name, e.g. ``"run_advance"``.
        timer (Timer): The accumulator receiving the measurements.
        tag (str): Accumulator name, e.g. ``"graph_advance"``.
        on_call (callable, optional): Called as ``on_call(args, kwargs, result)`` after a successful
            call, **outside** the timed region (used to capture parity operands).
        exclusive (bool): Subtract the time that nested tagged wrappers accumulated during the
            call, so ``tag`` holds only the time this frame itself spent.

    Returns:
        callable: A restore function that puts the original back (idempotent enough to call once).
    """
    original = getattr(module, name)
    wrapper = _make_wrapper(original, timer, tag, on_call, exclusive)
    if isinstance(inspect.getattr_static(module, name), staticmethod):
        setattr(module, name, staticmethod(wrapper))
    else:
        setattr(module, name, wrapper)

    def restore():
        setattr(module, name, original)

    return restore


def wrap_method(container, name: str, timer: Timer, tag: str, on_call=None, exclusive: bool = False):
    """Monkey-patch the method ``container.name`` (instance or class attribute) like :func:`wrap_function`.

    ``container`` may be a class or, as in the benchmark, the class object itself, which is what
    patches every instance including the ones created later. A ``staticmethod`` is re-installed as
    a ``staticmethod`` so ``self._draw_commands(rng)`` keeps calling it without a ``self`` argument.
    """
    was_static = isinstance(inspect.getattr_static(container, name), staticmethod)
    original = getattr(container, name)
    wrapper = _make_wrapper(original, timer, tag, on_call, exclusive)
    if was_static:
        setattr(container, name, staticmethod(wrapper))
    else:
        setattr(container, name, wrapper)

    def restore():
        if was_static:
            setattr(container, name, staticmethod(original))
        else:
            setattr(container, name, original)

    return restore


def _make_wrapper(original, timer: Timer, tag: str, on_call, exclusive: bool):
    """Build the transparent timing wrapper around ``original``."""

    def wrapper(*args, **kwargs):
        before = timer.snapshot() if exclusive else None
        start = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            if exclusive:
                nested = sum(
                    seconds
                    for name, (seconds, _calls) in timer.delta(before).items()
                    if name != tag and name != "generation_total"
                )
                timer.add(tag, max(0.0, elapsed - nested))
            else:
                timer.add(tag, elapsed)
        if on_call is not None:
            on_call(args, kwargs, result)
        return result

    wrapper.__name__ = getattr(original, "__name__", "wrapper")
    wrapper.__wrapped__ = original
    wrapper.__doc__ = getattr(original, "__doc__", None)
    return wrapper


def sync() -> bool:
    """Synchronise the accelerator, if there is one, and report whether anything was synchronised.

    ``torch.cuda.synchronize()`` is the HIP synchronisation on a ROCm host, so this is the one
    call the timing sections need; without an available device (or without torch) it is a no-op.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means a CPU-only measurement
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    return True


def error_block(exc: BaseException) -> dict:
    """Return the ``error`` block of a failed record: type, message and the traceback tail."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().strip().splitlines()[-40:],
    }


def run_guarded(fn, record: dict):
    """Call ``fn(record)``; on an exception mark the record failed and return ``None``.

    The record keeps whatever ``fn`` filled in before raising, plus ``status="failed"`` and the
    :func:`error_block`, so a broken measurement is still a committable data point.
    """
    try:
        return fn(record)
    except Exception as exc:  # noqa: BLE001 - a failed benchmark must be recorded, not crash
        record["status"] = "failed"
        record["error"] = error_block(exc)
        return None
