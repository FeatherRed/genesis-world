"""Measurement engine of the Phase-2 runtime benchmarks.

This module owns everything that touches a live pilot environment: the provenance helpers that
must run before and after a GPU measurement, the per-generation record entry, the per-step means
and traffic totals, :func:`_measure_env`, which builds one environment, warms it up on generation
0 and times generations 1..3, and the consistency checks that validate those counters.

The probes themselves (timers, traffic wrappers, graph-build timing) live in :mod:`runtime_probes`
and the CLI and record builder in :mod:`bench_runtime`.  Nothing in this module is imported by the
pilot environments, and no patch it installs survives the environment it was installed for.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import numpy as np

import common
from bench_envs import _build_env, _fingerprint, _load_task, _r
from common import Counter, Timer, sync
from runtime_probes import (
    DEVICE_RESIDENT_INTERMEDIATES,
    TIMER_GROUPS,
    TRAFFIC_NAMES,
    WARMUP_GENERATION,
    _SCOPE_TAGS,
    _instrument_device_path,
    _instrument_numpy_path,
    _patch_collision_cap,
    _state_read_parity,
    _wrap_executable_cache,
)

def _process_tree() -> set:
    """Return the PIDs of this process and its descendants (cheap, ``/proc`` based)."""
    tree = {os.getpid()}
    parents = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text(encoding="utf-8")
            parents[int(entry)] = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in tree and parent in tree:
                tree.add(pid)
                changed = True
    return tree


def _filter_own_gpu_pids(record: dict) -> None:
    """Remove this benchmark's own GPU process from the contention signal.

    ``genesis`` has to initialise (and possibly re-exec for its HSA repair) *before* the record's
    provenance block, and :func:`common.gpu_info` then queries ``torch.cuda``; either way this
    process already holds a GPU context when ``rocm-smi --showpids`` runs, so its own PID would be
    reported as contention.  Own-tree PIDs are therefore dropped -- the raw list is kept under
    ``host.rocm_pids_all`` and the dropped ones under ``host.owned_gpu_pids``, so nothing is hidden
    -- and ``contended`` is recomputed from the foreign PIDs and the load average.
    """
    host = record.get("host") or {}
    listed = [str(pid) for pid in host.get("rocm_pids") or []]
    if not listed:
        return
    own = _process_tree()
    foreign = [pid for pid in listed if int(pid) not in own]
    host["rocm_pids_all"] = listed
    host["rocm_pids"] = foreign
    host["owned_gpu_pids"] = sorted({int(pid) for pid in listed if int(pid) in own})
    host["gpu_busy"] = bool(foreign)
    host["gpu_pid_filter_note"] = (
        "genesis initialises the GPU (and torch.cuda is queried by this record) before the "
        "provenance block runs, so this process' own PIDs are excluded from rocm_pids and from "
        "the contention test; they stay listed in rocm_pids_all / owned_gpu_pids"
    )
    reasons = [reason for reason in record.get("contention_reasons") or [] if not reason.startswith("rocm-smi")]
    if foreign:
        reasons.insert(0, f"rocm-smi --showpids lists {len(foreign)} foreign process(es)")
    record["contention_reasons"] = reasons
    record["contended"] = bool(reasons)


# --- one measured environment ------------------------------------------------------


def _ensure_gs_init(use_gpu: bool) -> dict:
    """Initialise ``genesis`` once, before any scene is built; returns what was done.

    The pilot envs only call ``gs.init`` when genesis is not initialised yet, so a GPU run has to
    do it first (and it has to happen in this real script file, which is what the HSA repair of
    ``gs.init(backend=gs.gpu)`` re-executes).
    """
    import genesis as gs

    requested = "gpu" if use_gpu else "cpu"
    if gs._initialized:
        return {"requested": requested, "already_initialized": True, "backend": str(gs.device)}
    if use_gpu:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    else:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    return {"requested": requested, "already_initialized": False, "backend": str(gs.device)}


def _generation_entry(index, result, delta, traffic_delta, *, warmup):
    """Assemble one generation's throughput/traffic entry (the Phase-2 counterpart of replay's)."""
    steps = int(result["steps"])
    groups = {}
    for tag, (seconds, calls) in delta.items():
        groups[tag] = {
            "seconds": _r(seconds),
            "calls": int(calls),
            "per_step_s": _r(seconds / steps) if steps else None,
            "per_call_s": _r(seconds / calls) if calls else None,
        }
    total_seconds = float(delta.get("generation_total", (0.0, 0))[0])
    measured = sum(seconds for tag, (seconds, _c) in delta.items() if tag not in _SCOPE_TAGS)
    traffic = {}
    for tag, (calls, nbytes) in traffic_delta.items():
        if not calls and not nbytes:
            continue
        traffic[tag] = {
            "calls": int(calls),
            "nbytes": int(nbytes),
            "calls_per_step": _r(calls / steps) if steps else None,
            "bytes_per_step": _r(nbytes / steps) if steps else None,
        }
    fingerprint, _source = _fingerprint(result)
    return {
        "generation": int(index),
        "warmup": bool(warmup),
        "steps": steps,
        "env_steps": int(result["env_steps"]),
        "wall_time_s": _r(result["wall_time_s"]),
        "steps_per_s": _r(result["steps_per_s"]),
        "steps_per_s_harness": _r(steps / total_seconds) if total_seconds else None,
        "env_steps_per_s": _r(result["env_steps_per_s"]),
        "cache_hits": int(result["cache_hits"]),
        "cache_misses": int(result["cache_misses"]),
        "end_reasons": {name: int(count) for name, count in result["end_reasons"].items()},
        "seeds": [int(value) for value in np.asarray(result["seeds"]).reshape(-1)],
        "fitness": [_r(value) for value in np.asarray(result["fitness"]).reshape(-1)],
        "timings": {"groups": groups, "measured_s": _r(measured)},
        "residual_s": _r(total_seconds - measured),
        "residual_fraction": _r((total_seconds - measured) / total_seconds) if total_seconds else None,
        "traffic": traffic,
        "fingerprint": fingerprint,
    }


def _per_step_means(entries) -> dict:
    """Mean per-step seconds of every timer group over the timed (non-warm-up) generations."""
    timed = [entry for entry in entries if not entry["warmup"]]
    steps = sum(entry["steps"] for entry in timed)
    out = {}
    for tag in TIMER_GROUPS:
        if tag in _SCOPE_TAGS:
            continue
        seconds = sum(entry["timings"]["groups"].get(tag, {}).get("seconds") or 0.0 for entry in timed)
        if not seconds:
            continue
        out[tag] = {
            "seconds": _r(seconds),
            "calls": sum(entry["timings"]["groups"].get(tag, {}).get("calls") or 0 for entry in timed),
            "per_step_s": _r(seconds / steps) if steps else None,
            "per_generation_median_s": _r(
                statistics.median(
                    [entry["timings"]["groups"].get(tag, {}).get("seconds") or 0.0 for entry in timed]
                )
            ),
        }
    return out


def _traffic_totals(entries, steps: int) -> dict:
    """Sum the per-generation traffic counters of the timed generations, per step."""
    totals = {}
    for entry in entries:
        if entry["warmup"]:
            continue
        for tag, value in entry["traffic"].items():
            slot = totals.setdefault(tag, {"calls": 0, "nbytes": 0})
            slot["calls"] += value["calls"]
            slot["nbytes"] += value["nbytes"]
    for slot in totals.values():
        slot["calls_per_step"] = _r(slot["calls"] / steps) if steps else None
        slot["bytes_per_step"] = _r(slot["nbytes"] / steps) if steps else None
    return totals


def _measure_env(
    task,
    *,
    pop_size,
    master_seed,
    horizon,
    residency,
    backend,
    device,
    generations,
    device_path,
    collision_cap=None,
    keep_results=False,
) -> dict:
    """Build, instrument, run and close one pilot environment.

    Generation 0 is the warm-up (compile/cache dominated) and generation 1.. are the timed
    repeats.  Every patch installed here is restored when the environment is closed, so two
    environments (the residency comparison) can be measured one after the other in one process.
    """
    module, env_cls, payload_of = _load_task(task)
    builds = {}
    restore_cache = _wrap_executable_cache(builds)
    restore_cap = _patch_collision_cap(module, int(collision_cap)) if collision_cap is not None else None
    started = time.perf_counter()
    build_start = time.perf_counter()
    try:
        env = _build_env(
            env_cls,
            pop_size,
            master_seed,
            horizon,
            residency=residency,
            backend=backend,
            device=device,
            stop_when_done=False,
        )
    finally:
        restore_cache()
        if restore_cap is not None:
            restore_cap()
    build_s = time.perf_counter() - build_start
    payload = payload_of(env)
    timer = Timer(TIMER_GROUPS)
    traffic = Counter(TRAFFIC_NAMES)
    # ``_instrument_numpy_path`` installs the ``genesis.Scene.step`` timer itself, so it is not
    # wrapped here a second time (a double wrapper would count every physics step twice).
    restores = _instrument_numpy_path(task, env, timer, traffic)
    if device_path:
        restores += _instrument_device_path(task, env, timer, traffic)

    entries = []
    results = {}
    cache_calls = []
    synchronized = False
    close_s = None
    cache_final = {"hits": 0, "misses": 0}
    try:
        for generation in range(int(generations)):
            hits_before, misses_before = int(env.cache.hits), int(env.cache.misses)
            env.executables()  # compile-once proof: every call after __init__ must be a hit
            cache_calls.append(
                {
                    "generation": int(generation),
                    "hits": int(env.cache.hits) - hits_before,
                    "misses": int(env.cache.misses) - misses_before,
                }
            )
            sync()
            before = timer.snapshot()
            before_traffic = traffic.snapshot()
            start = time.perf_counter()
            result = env.evaluate(payload, generation)
            elapsed = time.perf_counter() - start
            synchronized = sync() or synchronized
            timer.add("generation_total", elapsed)
            after = timer.snapshot()
            after_traffic = traffic.snapshot()
            entries.append(
                _generation_entry(
                    generation,
                    result,
                    timer.delta(before, after),
                    traffic.delta(before_traffic, after_traffic),
                    warmup=generation == WARMUP_GENERATION,
                )
            )
            if keep_results:
                results[int(generation)] = result
        # Go2's two residencies read the state through different objects; measure the gap while
        # the scene is still alive (hover reads the same link buffer on both sides).
        state_read_parity = _state_read_parity(env) if task == "go2" else None
    finally:
        for restore in reversed(restores):
            restore()
        # ``close()`` clears the executable cache (and its counters), so read them first.
        cache_final = {"hits": int(env.cache.hits), "misses": int(env.cache.misses)}
        close_start = time.perf_counter()
        env.close()
        close_s = time.perf_counter() - close_start

    timed = [entry for entry in entries if not entry["warmup"]]
    steps_total = sum(entry["steps"] for entry in timed)
    return {
        "rows": int(env.rows),
        "pop_size": int(env.pop_size),
        "episodes_per_policy": int(env.episodes_per_policy),
        "max_episode_length": int(env.max_episode_length),
        "residency": str(env.residency),
        "backend": str(env.backend),
        "device_name": str(env.device_name),
        "entries": entries,
        "results": results,
        "builds": builds,
        "cache_calls": cache_calls,
        "timer_totals": timer.as_dict(),
        "phases": {
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
            "graph_build_note": (
                "per-graph build time is measured inside ExecutableCache.get while the env is "
                "constructed; the env builds every executable in __init__, so this table is the "
                "compile cost and every later generation is a cache hit"
            ),
            "first_generation_s": entries[0]["wall_time_s"] if entries else None,
            "first_generation_total_s": entries[0]["timings"]["groups"]["generation_total"]["seconds"]
            if entries
            else None,
            "steady_generation_median_s": _r(statistics.median([entry["wall_time_s"] for entry in timed]))
            if timed
            else None,
            "steady_generation_total_median_s": _r(
                statistics.median([entry["timings"]["groups"]["generation_total"]["seconds"] for entry in timed])
            )
            if timed
            else None,
            "close_s": _r(close_s),
            "total_s": _r(time.perf_counter() - started),
            "synchronized_before_and_after_each_generation": bool(synchronized),
            "generation_total_scope": (
                "generation_total is env.evaluate end to end (host seed draw + reset + steps + "
                "run_fitness) measured by the harness; measurements.generations[*].wall_time_s is "
                "the environment's own counter, which starts after the host seed draw and stops "
                "before run_fitness"
            ),
            "sync_note": (
                "common.sync() runs before and after every generation; it calls "
                "torch.cuda.synchronize() only when a HIP/CUDA device is available (it reported "
                f"{synchronized} here) and is a no-op on a CPU-resident run, where every timed "
                "region is host wall clock (time.perf_counter anyway)"
            ),
        },
        "throughput": {
            "timed_generations": [entry["generation"] for entry in timed],
            "warmup_generation": WARMUP_GENERATION,
            "rows": int(env.rows),
            "steps_total": int(steps_total),
            "steps_per_s": common.stats([entry["steps_per_s"] for entry in timed]),
            "env_steps_per_s": common.stats([entry["env_steps_per_s"] for entry in timed]),
            "wall_time_s": common.stats([entry["wall_time_s"] for entry in timed]),
            "steps_per_s_harness": common.stats([entry["steps_per_s_harness"] for entry in timed]),
            "note": (
                "median and spread over the timed generations of this process; the warm-up "
                "generation is excluded everywhere (it carries the first-step JIT of the compiled "
                "kernels and the empty caches)"
            ),
        },
        "per_step_means": _per_step_means(entries),
        "traffic": _traffic_totals(entries, steps_total),
        "cache": {
            "executables_call": "env.executables() once per generation (compile-once proof)",
            "per_generation": cache_calls,
            "hits": cache_final["hits"],
            "misses": cache_final["misses"],
            "compile_once": all(entry["misses"] == 0 for entry in cache_calls[1:]),
        },
        "device_resident_intermediates": DEVICE_RESIDENT_INTERMEDIATES[task] if device_path else None,
        "state_read_parity": state_read_parity,
        "checks": _consistency_checks(task, residency, entries, timed, steps_total),
    }


def _consistency_checks(task, residency, entries, timed, steps_total) -> dict:
    """Check the traffic counters against what each residency's design says must happen.

    A device residency must not download a graph output per step: Go2 reads back only the ``done``
    mask (plus ``returns`` once at the end), hover downloads nothing before the end of the
    generation.  A NumPy residency must read the state and write the actuators exactly once per
    step.  The checks are informational -- a mismatch is recorded, never raised, because a genuine
    behaviour change is exactly what a reader has to see.
    """

    def calls(tag):
        return sum(entry["traffic"].get(tag, {}).get("calls", 0) for entry in timed)

    checks = []
    if not timed:
        return {"checks": checks, "pass": False, "note": "no timed generation"}
    steps = int(steps_total)
    generations = len(timed)
    if residency == "device":
        if task == "go2":
            observed, expected = calls("download"), steps + generations
            checks.append(
                {
                    "name": "go2 device: one done-mask readback per step, returns once per generation",
                    "observed_calls": observed,
                    "expected_calls": expected,
                    "pass": observed == expected,
                }
            )
            expected_sync = 2 * steps + generations
            observed_sync = calls("sync")
            checks.append(
                {
                    "name": "go2 device: two fences per step plus one per generation reset",
                    "observed_calls": observed_sync,
                    "expected_calls": expected_sync,
                    "pass": observed_sync == expected_sync,
                }
            )
        else:
            observed, expected = calls("download"), 4 * generations
            checks.append(
                {
                    "name": "hover device: only end_code/returns/fitness/commands are downloaded, once per generation",
                    "observed_calls": observed,
                    "expected_calls": expected,
                    "pass": observed == expected,
                }
            )
    else:
        for tag, label in (
            ("read_state", "state read"),
            ("write_actuators", "actuator write"),
        ):
            observed = calls(tag)
            checks.append(
                {
                    "name": f"numpy residency: one {label} per step",
                    "observed_calls": observed,
                    "expected_calls": steps,
                    "pass": observed == steps,
                }
            )
    horizon_ok = all(entry["steps"] > 0 for entry in timed)
    return {"checks": checks, "pass": bool(all(check["pass"] for check in checks) and horizon_ok)}


# --- throughput --------------------------------------------------------------------


