"""Stage-3 Phase-5 aggregation: committed records -> ``results/summary.json`` + ``results/tables.md``.

Every measurement of this node is a JSON record under ``results/`` (written by
:func:`common.write_record`). This CLI loads all of them, validates their provenance, aggregates
them by ``(kind, task, workload signature)`` and writes:

* ``results/summary.json`` -- the validated aggregates (record counts, per-configuration medians and
  spreads, per-generation values, parity diffs, probe verdicts) plus the list of source records, so
  a reader can trace every number back to a file;
* ``results/tables.md`` -- the GitHub-flavoured numeric tables (rendered by :mod:`summary_tables`)
  that ``RESULTS.md`` embeds verbatim, so no number in the report is hand-transcribed.

Usage (from the repository root)::

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/summarize.py --markdown examples/neuroevolution/benchmarks/results/tables.md
    PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/summarize.py check

A bare invocation prints the usage and exits 0, like the other benchmark CLIs, so the
``examples/**/*.py`` smoke harness does not treat this tool as a failing example.

Nothing here imports ``genesis``, ``torch``, ``tensordict`` or ``evoxc``: aggregation is plain
``json`` over the committed records plus :mod:`common` (statistics, schema and baseline constants).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common
from summary_tables import _entries, _f, _mean_spread, _num, _pick, markdown_document

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
TABLES_PATH = RESULTS_DIR / "tables.md"

#: Full sha of :data:`common.BASELINE_COMMIT`; every record resolves the short sha to this one.
BASELINE_SHA = "ad2f07e5aec1ba931f5a6704253f70e6fa6370b2"

#: Top-level keys every record must carry (``new_record`` writes them).
REQUIRED_KEYS = (
    "schema", "created_utc", "kind", "command", "git", "packages", "host", "gpu", "contended",
    "contention_reasons", "task", "seed", "repeat", "process_role", "workload", "phases", "timings",
    "measurements", "status", "error",
)  # fmt: skip

#: Record families in report order.
KIND_ORDER = (
    "env_replay",
    "env_throughput",
    "env_residency",
    "neat_train",
    "neat_determinism",
    "probe_unmatched_baselines",
    "probe_tell_backend",
    "probe_torch_cpu_boundary",
)

#: Hard ceiling on ``summary.json`` (aggregates and small arrays only).
MAX_SUMMARY_BYTES = 200_000

#: Per-run trainer checks kept in the summary (the rest is cache/seed bookkeeping).
NEAT_CHECKS = (
    "determinism_fingerprint", "cache_hits", "cache_misses", "no_graph_build_during_run",
    "seed_table_matches_trainer", "holdout_disjoint", "distinct_seeds_per_generation",
)  # fmt: skip

#: Scalar workload fields kept in the summary (the timer-group prose is dropped).
WORKLOAD_SCALARS = (
    "task", "backend", "device", "resolved_backend", "resolved_device", "device_side_resolved",
    "physics_backend", "residency", "residencies", "pop_size", "rows", "horizon", "max_episode_length",
    "episodes_per_policy", "generations", "timed_generations", "warmup_generation", "parity_generation",
    "stop_when_done", "variant", "tell_backend", "pjrt_plugin", "probe", "capture_limit",
)  # fmt: skip


def _stem(path):
    """Record filename without its ``.json`` suffix (the name used in every table)."""
    return Path(path).stem


def _workload_view(workload):
    """Keep only the scalar/list-of-scalar workload fields."""
    return {key: workload[key] for key in WORKLOAD_SCALARS if key in workload}


# ---------------------------------------------------------------------------------------------
# loading + validation
# ---------------------------------------------------------------------------------------------
def load_records(results_dir=RESULTS_DIR):
    """Load every ``*.json`` of ``results_dir`` except the generated ``summary.json``.

    Returns:
        list[tuple[str, dict]]: ``("results/<file>", record)`` in filename order.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"no results directory at {results_dir}")
    loaded = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == SUMMARY_PATH.name:
            continue
        try:
            loaded.append((f"results/{path.name}", json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name}: not valid JSON ({exc})") from exc
    if not loaded:
        raise SystemExit(f"no records under {results_dir}")
    return loaded


def validate_records(loaded):
    """Validate the provenance of every record and return the report's ``validation`` block."""
    warnings, heads, kinds, schema_counts = [], {}, {}, {}
    n_ok = n_failed = n_contended = 0
    diff_empty_all = True
    for name, record in loaded:
        kind = record.get("kind")
        kinds[kind] = kinds.get(kind, 0) + 1
        heads.setdefault(str(kind), set()).add(_pick(record, "git", "head"))
        schema_counts[record.get("schema")] = schema_counts.get(record.get("schema"), 0) + 1
        missing = [key for key in REQUIRED_KEYS if key not in record]
        if missing:
            warnings.append(f"{name}: missing top-level key(s) {', '.join(missing)}")
        if record.get("schema") != common.SCHEMA:
            warnings.append(f"{name}: schema {record.get('schema')!r} != {common.SCHEMA!r}")
        if not str(record.get("task") or "").strip():
            warnings.append(f"{name}: empty task")
        workload = record.get("workload")
        if not isinstance(workload, dict) or not workload:
            warnings.append(f"{name}: missing workload")
        if _pick(record, "git", "baseline_env_diff_empty") is not True:
            diff_empty_all = False
            warnings.append(f"{name}: git.baseline_env_diff_empty is {_pick(record, 'git', 'baseline_env_diff_empty')!r}")
        if _pick(record, "git", "baseline") != common.BASELINE_COMMIT:
            warnings.append(f"{name}: git.baseline {_pick(record, 'git', 'baseline')!r} != {common.BASELINE_COMMIT!r}")
        if _pick(record, "git", "baseline_sha") != BASELINE_SHA:
            warnings.append(f"{name}: git.baseline_sha {_pick(record, 'git', 'baseline_sha')!r} != {BASELINE_SHA!r}")
        if kind not in KIND_ORDER:
            warnings.append(f"{name}: unknown kind {kind!r}")
        if record.get("status") == "ok":
            n_ok += 1
        else:
            n_failed += 1
            warnings.append(f"{name}: status {record.get('status')!r} (error block recorded)")
        if record.get("contended"):
            n_contended += 1
            reasons = "; ".join(str(item) for item in (record.get("contention_reasons") or []))
            warnings.append(f"{name}: contended ({reasons})")
        if not record.get("command"):
            warnings.append(f"{name}: empty command")
    return {
        "n_records": len(loaded),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_contended": n_contended,
        "heads": {kind: sorted(sha for sha in shas if sha) for kind, shas in sorted(heads.items())},
        "warnings": warnings,
        "baseline": common.BASELINE_COMMIT,
        "baseline_sha_expected": BASELINE_SHA,
        "baseline_env_diff_empty_all": diff_empty_all,
        "schema": common.SCHEMA,
        "schemas_seen": {str(key): value for key, value in sorted(schema_counts.items(), key=lambda item: str(item[0]))},
        "kinds": {str(key): value for key, value in sorted(kinds.items(), key=lambda item: str(item[0]))},
    }


# ---------------------------------------------------------------------------------------------
# per-record extraction
# ---------------------------------------------------------------------------------------------
def _signature(record):
    """Stable workload signature: what makes two records the same configuration."""
    workload = record.get("workload") or {}
    parts = [
        record.get("task"), workload.get("pop_size"), workload.get("rows"), workload.get("horizon"),
        workload.get("residency"), workload.get("backend"), workload.get("device"),
        workload.get("physics_backend"), workload.get("variant"), workload.get("tell_backend"),
        workload.get("probe"),
    ]  # fmt: skip
    return "|".join("" if part is None else str(part) for part in parts)


def _head(name, record):
    """The provenance fields every extracted entry carries."""
    return {
        "record": _stem(name), "task": record.get("task"), "seed": record.get("seed"),
        "repeat": record.get("repeat"), "process_role": record.get("process_role"),
        "status": record.get("status"), "contended": record.get("contended"),
        "created_utc": record.get("created_utc"), "head": _pick(record, "git", "head"),
        "workload": _workload_view(record.get("workload") or {}),
    }  # fmt: skip


def _generations(measurements):
    """Per-generation headline values (the warm-up generation is flagged, never dropped silently)."""
    keys = (
        "generation", "warmup", "steps", "steps_per_s", "env_steps_per_s", "wall_time_s",
        "best_fitness", "median_fitness", "evaluate_s", "tell_s", "generation_s",
        "cache_hits", "cache_misses", "cache_hits_delta", "cache_misses_delta",
    )  # fmt: skip
    return [{key: entry.get(key) for key in keys} for entry in measurements.get("generations") or []]


def _time_split(measurements):
    """Per-step cost and share of the timed generations, from the record's own sums.

    ``per_step_means[*].seconds`` are sums over the timed generations, so dividing by the timed
    steps and wall clock gives an exact mean and share without re-reading raw samples.
    """
    timed = [entry for entry in measurements.get("generations") or [] if not entry.get("warmup")]
    wall = sum(_num(entry.get("wall_time_s")) or 0.0 for entry in timed)
    steps = sum(_num(entry.get("steps")) or 0.0 for entry in timed)
    per_step_ms, shares = {}, {}
    for name, entry in sorted((measurements.get("per_step_means") or {}).items()):
        seconds = _num(entry.get("seconds")) or 0.0
        per_step_ms[name] = 1000.0 * seconds / steps if steps else None
        shares[name] = seconds / wall if wall else None
    residual = sum(_num(entry.get("residual_s")) or 0.0 for entry in timed)
    return {
        "timed_generations": [entry.get("generation") for entry in timed],
        "wall_s": wall or None,
        "steps": steps or None,
        "per_step_ms": per_step_ms,
        "shares": shares,
        "residual_s": residual or None,
        "residual_share": residual / wall if wall else None,
    }


def _phases(phases, keys):
    """Project the phase keys a family needs."""
    return {key: phases.get(key) for key in keys}


def _host_counters(source):
    """The three per-step host-traffic counters of a measurement or a residency side."""
    return {
        "host_calls_per_step": source.get("host_calls_per_step"),
        "host_bytes_per_step": source.get("host_bytes_per_step"),
        "sync_calls_per_step": source.get("sync_calls_per_step"),
    }


def _resident_view(block):
    """What stays on the device and what still costs a host touch (the prose source line is dropped)."""
    if not isinstance(block, dict):
        return None
    return {key: block.get(key) for key in ("items", "host_touches_per_step")}


def _stats_view(block):
    """Keep the distribution a record reports (median, range, spread, count) and drop mean/stdev."""
    if not isinstance(block, dict):
        return block
    return {key: block.get(key) for key in ("median", "min", "max", "spread", "n")}


def _throughput_block(block):
    """One throughput block: the two rates and the wall time as medians, plus the workload it covers."""
    block = block or {}
    view = {key: _stats_view(block.get(key)) for key in ("steps_per_s", "env_steps_per_s", "wall_time_s")}
    view.update({key: block.get(key) for key in ("rows", "steps_total", "timed_generations")})
    return view


def _graph_builds(builds):
    """Per-graph compile cost: one build per graph, then cache lookups (the compile-once proof)."""
    return {name: {key: entry.get(key) for key in ("build_s", "builds", "calls")} for name, entry in (builds or {}).items()}


def _throughput_entry(name, record):
    """Extraction of one ``env_throughput`` record."""
    measurements, phases = record.get("measurements") or {}, record.get("phases") or {}
    return {
        **_head(name, record),
        "generations": _generations(measurements),
        "throughput": _throughput_block(measurements.get("throughput")),
        "split": _time_split(measurements),
        "phases": _phases(phases, ("build_s", "graph_build_s", "graph_build_total_s", "first_generation_s", "steady_generation_median_s", "close_s", "total_s")),
        "host_counters": _host_counters(measurements),
        "cache": measurements.get("cache"),
        "checks": measurements.get("checks"),
        "state_read_parity": measurements.get("state_read_parity"),
        "device_resident_intermediates": measurements.get("device_resident_intermediates"),
    }


def _replay_entry(name, record):
    """Extraction of one ``env_replay`` record (oracle parity)."""
    measurements = record.get("measurements") or {}
    parity = measurements.get("parity") or {}
    records = {key: parity.get(key) for key in ("n_steps_compared", "n_steps_dropped", "rows", "observation", "reward", "returns", "termination", "reset_observation", "commands", "yaw_deg")}
    return {
        **_head(name, record),
        "tolerances": _pick(record, "workload", "tolerances"),
        "generations": _generations(measurements),
        "fingerprint": _pick(measurements, "fingerprint", "sha256"),
        "fingerprints": _pick(measurements, "fingerprint", "per_generation"),
        "parity": records,
        "phases": _phases(record.get("phases") or {}, ("build_s", "first_generation_s", "steady_generation_median_s", "total_s")),
    }


def _residency_side(side, phase_block, name=""):
    """Extraction of one residency side (``numpy`` or ``device``) of an ``env_residency`` record."""
    builds = side.get("graph_build_s") or {}
    return {
        "backend": side.get("backend"),
        "device_name": side.get("device_name"),
        "build_s": side.get("build_s"),
        "graph_build_s": _graph_builds(builds),
        "graph_build_total_s": _num(side.get("graph_build_total_s")) or sum(_num(_pick(item, "build_s")) or 0.0 for item in builds.values()),
        "first_generation_s": _pick(phase_block, "first_generation_s"),
        "steady_generation_median_s": _pick(phase_block, "steady_generation_median_s"),
        "throughput": _throughput_block(side.get("throughput")),
        "cache": side.get("cache"),
        "checks": side.get("checks"),
        "device_resident_intermediates": _resident_view(side.get("device_resident_intermediates")),
        "state_read_parity": side.get("state_read_parity") if name == "numpy" else None,
    }


def _parity_arrays(arrays):
    """Keep only the numbers of one parity comparison array (shape equality is asserted, not stored)."""
    return {
        name: {"max_abs_diff": entry.get("max_abs_diff"), "bit_identical": entry.get("bit_identical"), "pass": entry.get("pass")}
        for name, entry in (arrays or {}).items()
    }  # fmt: skip


def _parity_scalars(scalars):
    """Keep only the numbers of one parity comparison scalar (``end_reasons`` keeps both count maps)."""
    return {name: {"observed": entry.get("observed"), "reference": entry.get("reference"), "pass": entry.get("pass")} for name, entry in (scalars or {}).items()}  # fmt: skip


def _residency_entry(name, record):
    """Extraction of one ``env_residency`` record (NumPy vs device residency, same physics)."""
    measurements, phases = record.get("measurements") or {}, record.get("phases") or {}
    parity, sides = measurements.get("parity") or {}, measurements.get("sides") or {}
    return {
        **_head(name, record),
        "comparison": measurements.get("comparison"),
        "parity": {
            "generation": parity.get("generation"),
            "all_pass": parity.get("all_pass"),
            "arrays": _parity_arrays(parity.get("arrays")),
            "scalars": _parity_scalars(parity.get("scalars")),
            "fingerprint": parity.get("fingerprint"),
        },
        "sides": {key: _residency_side(sides.get(key, {}), phases.get(key) or {}, key) for key in ("numpy", "device")},
        "phases": {key: _phases(phases.get(key) or {}, ("build_s", "graph_build_total_s", "first_generation_s", "steady_generation_median_s", "total_s")) for key in ("numpy", "device")},
    }


def _neat_entry(name, record):
    """Extraction of one ``neat_train`` record (one full TensorNEAT run)."""
    measurements, phases = record.get("measurements") or {}, record.get("phases") or {}
    totals = measurements.get("totals") or {}
    holdout = measurements.get("holdout") or {}
    curve = measurements.get("learning_curve") or {}
    best = holdout.get("best_policy") or {}
    totals_keys = ("generations", "steps", "env_steps", "evaluate_s", "tell_s", "generation_s", "loop_s", "best_fitness", "final_best_fitness")
    return {
        **_head(name, record),
        "residency": measurements.get("residency"),
        "backend": measurements.get("backend"),
        "device_name": measurements.get("device_name"),
        "tell": measurements.get("tell"),
        "generations": _generations(measurements),
        "learning_curve": {key: curve.get(key) for key in ("generations", "best_fitness", "median_fitness", "cumulative_env_steps", "cumulative_loop_s")},
        "totals": {
            **{key: totals.get(key) for key in totals_keys},
            "steps_per_s": _stats_view(_pick(totals, "steps_per_s", "stats")),
            "env_steps_per_s": _stats_view(_pick(totals, "env_steps_per_s", "stats")),
        },
        "holdout": {
            "source": holdout.get("source"),
            "disjoint_from_training": holdout.get("disjoint_from_training"),
            "initial_median": _pick(holdout, "initial", "median_fitness"),
            "initial_best": _pick(holdout, "initial", "best_fitness"),
            "trained_median": _pick(holdout, "trained", "median_fitness"),
            "trained_best": _pick(holdout, "trained", "best_fitness"),
            "median_delta": holdout.get("median_delta"),
            "separates_initial_from_trained": holdout.get("separates_initial_from_trained"),
            "best_policy": {key: best.get(key) for key in ("index", "train_fitness", "heldout_fitness", "delta")},
        },
        "phases": _phases(phases, ("import_s", "build_s", "setup_s", "trainer_build_s", "tell_build_s", "graph_build_total_s", "first_generation_s", "steady_generation_median_s", "holdout_s", "close_s", "totals_s")),
        "checks": {key: (measurements.get("checks") or {}).get(key) for key in NEAT_CHECKS},
    }


def _determinism_entry(name, record):
    """Extraction of the ``neat_determinism`` cross-process check."""
    measurements = record.get("measurements") or {}
    return {
        **_head(name, record),
        "records": measurements.get("records"),
        "checks": measurements.get("checks"),
        "verdict": measurements.get("verdict"),
    }


def _probe_case(case):
    """One subprocess probe case: its verdict, its failure and its cost, never its captured output."""
    case = case or {}
    return {
        "verdict": case.get("verdict"),
        "status": case.get("status"),
        "returncode": case.get("returncode"),
        "elapsed_s": case.get("elapsed_s"),
        "timed_out": case.get("timed_out"),
        "timeout_s": case.get("timeout_s"),
        "exception": case.get("exception") or case.get("error"),
        "extra_pythonpath": case.get("extra_pythonpath"),
    }


def _probe_interface(mismatch):
    """The interface facts (not the source listings) of one unimportable baseline."""
    return {
        "attempted": mismatch.get("attempted"),
        "blocked_at": mismatch.get("blocked_at"),
        "has_evaluate": mismatch.get("has_evaluate"),
        "verdict": mismatch.get("verdict"),
        "sources": mismatch.get("sources"),
        "step": _pick(mismatch, "step", "qualified"),
        "step_line": _pick(mismatch, "step", "line"),
        "step_returns": _pick(mismatch, "step", "returns"),
        "auto_reset": _pick(mismatch, "auto_reset", "qualified"),
        "config_source": _pick(mismatch, "config_source", "qualified"),
        "config_source_returns": _pick(mismatch, "config_source", "returns"),
        "reasons": mismatch.get("reasons"),
        "required_for_adapter": mismatch.get("required_for_adapter"),
        "observed_import_error": mismatch.get("observed_import_error"),
    }


def _ast_summary(summary):
    """The scan summary as counts plus the module-scope torch files (the rest is in the per-file block)."""
    summary = summary or {}
    keys = ("scanned", "rule", "torch_at_import_time", "neuroevolution_torch_at_import_time")
    counts = {f"{key}_n": len(summary[key]) for key in ("no_torch", "tensordict_or_genesis") if key in summary}
    return {**{key: summary.get(key) for key in keys}, **counts}


def _probe_ast(files):
    """One scanned source: whether torch is imported at module scope and how often it is deferred."""
    facts = files or {}
    torch_imports = facts.get("torch") or []
    return {
        "exists": facts.get("exists"),
        "syntax_error": facts.get("syntax_error"),
        "torch_module": any(item.get("scope_kind") == "module" for item in torch_imports),
        "torch_deferred": sum(1 for item in torch_imports if item.get("scope_kind") != "module"),
        "genesis": len(facts.get("genesis") or []),
        "tensordict": len(facts.get("tensordict") or []),
        "rsl_rl": len(facts.get("rsl_rl") or []),
    }


def _probe_boundary(block):
    """One cited boundary block: where it was measured, at which head, and the numbers themselves."""
    block = block or {}
    return {
        "source": block.get("source"),
        "git_head": block.get("git_head"),
        "status": block.get("status"),
        "values": block.get("values"),
    }


def _probe_entry(name, record):
    """Extraction of one Phase-4 probe record: verdicts, failures and cited numbers, no captured output."""
    kind, measurements = record.get("kind"), record.get("measurements") or {}
    if kind == "probe_unmatched_baselines":
        return {
            **_head(name, record),
            "verdict": measurements.get("verdict"),
            "imports": {key: _probe_case(case) for key, case in (measurements.get("imports") or {}).items()},
            "adapter_attempts": {key: _probe_case(case) for key, case in (measurements.get("adapter_attempts") or {}).items()},
            "adapter_verdict": measurements.get("adapter_verdict"),
            "install_attempt": measurements.get("install_attempt"),
            "interface_mismatch": {key: _probe_interface(case) for key, case in (measurements.get("interface_mismatch") or {}).items()},
            "ppo": measurements.get("ppo"),
        }
    if kind == "probe_tell_backend":
        entries = {}
        for key, case in (measurements.get("entries") or {}).items():
            result = case.get("result") or {}
            entries[key] = {
                **_probe_case(case),
                "backend": case.get("backend"),
                "device": case.get("device"),
                "last_phase": case.get("last_phase"),
                "phases_reached": case.get("phases_reached"),
                "plugin_path": case.get("plugin_path"),
                "result": {field: result.get(field) for field in ("status", "build_s", "run_s", "total_s", "resolved_backend", "resolved_device")} if result else None,
            }
        return {**_head(name, record), "verdict": measurements.get("verdict"), "entries": entries, "gpu_after": measurements.get("gpu_after")}
    return {
        **_head(name, record),
        "ast_scan": {
            "summary": _ast_summary(_pick(measurements, "ast_scan", "summary")),
            "files": {key: _probe_ast(facts) for key, facts in (_pick(measurements, "ast_scan", "files") or {}).items()},
        },
        "runtime_probe": {
            "elapsed_s": _pick(measurements, "runtime_probe", "elapsed_s"),
            "returncode": _pick(measurements, "runtime_probe", "returncode"),
            "report": _pick(measurements, "runtime_probe", "report"),
        },
        "measured_boundary": {
            task: {family: _probe_boundary(block) for family, block in families.items() if isinstance(block, dict)}
            for task, families in (_pick(measurements, "measured_boundary") or {}).items()
            if isinstance(families, dict)
        },
    }


EXTRACTORS = {
    "env_replay": _replay_entry,
    "env_throughput": _throughput_entry,
    "env_residency": _residency_entry,
    "neat_train": _neat_entry,
    "neat_determinism": _determinism_entry,
    "probe_unmatched_baselines": _probe_entry,
    "probe_tell_backend": _probe_entry,
    "probe_torch_cpu_boundary": _probe_entry,
}


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------
def _timed(entries, key):
    """Every timed-generation value of ``key`` across the entries (warm-up generation excluded)."""
    return [
        _num(generation.get(key))
        for entry in entries
        for generation in entry["generations"]
        if not generation["warmup"]
    ]


def _aggregate(kind, entries):
    """Per-group aggregate over the repeats/seeds of one (kind, task, workload signature)."""
    aggregate = {"n_records": len(entries)}
    if kind in ("env_throughput", "env_replay"):
        for key in ("steps_per_s", "env_steps_per_s", "wall_time_s"):
            aggregate[key] = _stats_view(common.stats(_timed(entries, key)))
        phases = [entry["phases"] for entry in entries]
        for key in ("build_s", "first_generation_s", "steady_generation_median_s"):
            aggregate[key] = _stats_view(common.stats([_num(block.get(key)) for block in phases]))
        if kind == "env_throughput":
            aggregate["graph_build_total_s"] = _stats_view(common.stats([_num(block.get("graph_build_total_s")) for block in phases]))
    elif kind == "env_residency":
        aggregate["speedup"] = _stats_view(common.stats([_num(_pick(entry, "comparison", "speedup")) for entry in entries]))
        for side in ("numpy", "device"):
            aggregate[f"{side}_steps_per_s"] = _stats_view(common.stats([_num(_pick(entry, "sides", side, "throughput", "steps_per_s", "median")) for entry in entries]))
            aggregate[f"{side}_env_steps_per_s"] = _stats_view(common.stats([_num(_pick(entry, "sides", side, "throughput", "env_steps_per_s", "median")) for entry in entries]))
            for key in ("build_s", "first_generation_s", "graph_build_total_s"):
                aggregate[f"{side}_{key}"] = _stats_view(common.stats([_num(_pick(entry, "sides", side, key)) for entry in entries]))
        aggregate["parity_all_pass"] = all(bool(_pick(entry, "parity", "all_pass")) for entry in entries)
        aggregate["fingerprints_equal"] = all(bool(_pick(entry, "parity", "fingerprint", "equal")) for entry in entries)
    elif kind == "neat_train":
        for key in ("generation_s", "evaluate_s", "tell_s", "best_fitness", "final_best_fitness", "holdout_trained_median", "holdout_initial_median"):
            source = "totals" if key in ("generation_s", "evaluate_s", "tell_s", "best_fitness", "final_best_fitness") else "holdout"
            aggregate[key] = _stats_view(common.stats([_num(_pick(entry, source, key)) for entry in entries]))
        for key in ("steps_per_s", "env_steps_per_s"):
            aggregate[key] = _stats_view(common.stats([_num(_pick(entry, "totals", key, "median")) for entry in entries]))
        for key in ("steady_generation_median_s", "first_generation_s", "tell_build_s", "build_s"):
            aggregate[key] = _stats_view(common.stats([_num(_pick(entry, "phases", key)) for entry in entries]))
        aggregate["seeds"] = sorted({entry["seed"] for entry in entries})
        aggregate["repeats"] = sorted({entry["repeat"] for entry in entries})
        aggregate["fingerprints"] = sorted({_pick(entry, "checks", "determinism_fingerprint") for entry in entries} - {None})
    elif kind == "neat_determinism":
        aggregate["all_ok"] = all(bool(_pick(entry, "verdict", "ok")) for entry in entries)
    return aggregate


def group_records(loaded):
    """Group the extracted records by ``(kind, task, workload signature)`` in report order."""
    groups = {}
    for name, record in loaded:
        extractor = EXTRACTORS.get(record.get("kind"))
        if extractor is None:
            continue
        signature = _signature(record)
        groups.setdefault(record.get("kind"), {}).setdefault(record.get("task"), {}).setdefault(
            signature,
            {"signature": signature, "workload": _workload_view(record.get("workload") or {}), "entries": []},
        )["entries"].append(extractor(name, record))
    ordered = {}
    for kind in KIND_ORDER:
        if kind not in groups:
            continue
        ordered[kind] = {}
        for task in sorted(groups[kind]):
            ordered[kind][task] = {}
            for signature in sorted(groups[kind][task]):
                group = groups[kind][task][signature]
                group["records"] = [entry["record"] for entry in group["entries"]]
                group["process_roles"] = sorted({entry["process_role"] for entry in group["entries"]})
                group["aggregate"] = _aggregate(kind, group["entries"])
                ordered[kind][task][signature] = group
    return ordered


def environment_block(loaded):
    """Host/GPU/package provenance common to all records, plus the load-average range."""
    gpus, packages, loadavgs = set(), {}, []
    cpus, pythons, drivers = set(), set(), set()
    for _, record in loaded:
        gpu = record.get("gpu") or {}
        gpus.add((gpu.get("name"), gpu.get("hip"), gpu.get("torch"), gpu.get("device_count")))
        for name, version in (record.get("packages") or {}).items():
            packages.setdefault(name, set()).add(version)
        loadavgs.append(_num(_pick(record, "host", "loadavg_1m")))
        cpus.add(_pick(record, "host", "cpu_count"))
        pythons.add(_pick(record, "host", "python"))
        for line in _pick(record, "host", "rocm_driver") or []:
            if str(line).startswith("Driver version:"):
                drivers.add(str(line).split(":", 1)[1].strip())
    finite = [value for value in loadavgs if value is not None]
    return {
        "gpu": [{"name": name, "hip": hip, "torch": torch, "device_count": count} for name, hip, torch, count in sorted(gpus, key=str)],
        "gpu_names": sorted({item[0] for item in gpus if item[0]}),
        "rocm_driver": sorted(drivers),
        "packages": {name: sorted(str(v) for v in values) for name, values in sorted(packages.items())},
        "loadavg_1m": {"min": min(finite, default=None), "max": max(finite, default=None), "n": len(loadavgs)},
        "cpu_count": sorted(cpus, key=str),
        "python": sorted(pythons, key=str),
        "contention_threshold_loadavg": common.CONTENTION_LOADAVG,
    }


def build_summary(loaded):
    """Build the whole ``summary.json`` payload from the loaded records."""
    return {
        "schema": "neuroevolution-benchmark-summary/1",
        "generated_by": "examples/neuroevolution/benchmarks/summarize.py",
        "record_schema": common.SCHEMA,
        "validation": validate_records(loaded),
        "environment": environment_block(loaded),
        "families": group_records(loaded),
        "sources": [name for name, _ in loaded],
    }


# ---------------------------------------------------------------------------------------------
# printing, self-check and CLI
# ---------------------------------------------------------------------------------------------
def serialize_summary(summary):
    """The compact JSON payload written to ``summary.json``.

    Compact (no indentation) on purpose: the aggregates of 37 records only fit under the size cap
    that way. ``results/tables.md`` is the readable rendering of the same payload.
    """
    return json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def print_summary(summary, summary_path, tables_path):
    """Print the concise console summary: counts, warnings and the headline medians."""
    validation = summary["validation"]
    print(f"records: {validation['n_records']} (ok {validation['n_ok']}, failed {validation['n_failed']}, contended {validation['n_contended']})")
    print(f"baseline {validation['baseline_sha_expected']}; baseline_env_diff_empty_all={validation['baseline_env_diff_empty_all']}")
    print("heads: " + ", ".join(f"{kind}={len(shas)}" for kind, shas in validation["heads"].items()))
    if validation["warnings"]:
        print(f"warnings ({len(validation['warnings'])}):")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    else:
        print("warnings: none")
    print("headline medians (timed generations only):")
    for kind in ("env_throughput", "env_residency", "neat_train"):
        for task in sorted(summary["families"].get(kind) or {}):
            for signature in sorted(summary["families"][kind][task]):
                group = summary["families"][kind][task][signature]
                aggregate, workload = group["aggregate"], group["workload"]
                label = f"{kind}/{task}/pop {workload.get('pop_size')}/residency {workload.get('residency')}"
                if kind == "env_throughput":
                    print(f"  {label}: {_mean_spread(aggregate['steps_per_s'])} steps/s, {_mean_spread(aggregate['env_steps_per_s'])} env-steps/s")
                elif kind == "env_residency":
                    print(
                        f"  {label}: speedup {_f(_pick(aggregate, 'speedup', 'median'), 3)}x "
                        f"(numpy {_f(_pick(aggregate, 'numpy_steps_per_s', 'median'), 2)} -> "
                        f"device {_f(_pick(aggregate, 'device_steps_per_s', 'median'), 2)} steps/s)"
                    )
                else:
                    print(f"  {label}: steady gen {_mean_spread(aggregate['steady_generation_median_s'])} s, best fitness {_mean_spread(aggregate['best_fitness'])}")
    print(f"wrote {summary_path}")
    print(f"wrote {tables_path}")


def _selfcheck(loaded, summary):
    """Assert the summary/tables invariants without a simulator; returns the pinned spot checks."""
    validation = summary["validation"]
    assert validation["n_records"] == len(loaded) == len(summary["sources"]), "record count mismatch"
    assert validation["n_ok"] + validation["n_failed"] == validation["n_records"], "status counts mismatch"
    assert validation["baseline_env_diff_empty_all"], "a record reports a non-empty baseline env diff"
    assert all(len(shas) >= 1 for shas in validation["heads"].values()), "a kind has no head sha"
    grouped = {name for tasks in summary["families"].values() for groups in tasks.values() for group in groups.values() for name in group["records"]}
    assert grouped == {_stem(name) for name in summary["sources"]}, "a record is missing from its family group"
    for tasks in summary["families"].values():
        for groups in tasks.values():
            for group in groups.values():
                assert group["entries"], "empty group"
                assert group["aggregate"]["n_records"] == len(group["entries"]), "aggregate count mismatch"
    payload = serialize_summary(summary)
    assert len(payload) <= MAX_SUMMARY_BYTES, f"summary is {len(payload)} bytes > {MAX_SUMMARY_BYTES}"
    assert json.loads(payload) == summary, "summary does not round-trip through JSON"
    assert serialize_summary(build_summary(loaded)) == payload, "summary is not deterministic"
    tables = markdown_document(summary)
    assert tables.count("\n### T") >= 18, "fewer tables than expected"
    assert tables.count("Sources:") >= 18, "a table has no source footnote"
    for name in summary["sources"]:
        assert f"`{_stem(name)}.json`" in tables, f"{_stem(name)} is not cited by any table"
    go2_replay = next(entry for entry in _entries(summary, "env_replay") if entry["task"] == "go2" and entry["repeat"] == 0)
    go2_residency = next(entry for entry in _entries(summary, "env_residency") if entry["task"] == "go2" and entry["repeat"] == 0)
    go2_neat = next(entry for entry in _entries(summary, "neat_train") if entry["record"] == "neat_go2_seed0")
    checks = [
        ("go2 replay observation max abs diff", _pick(go2_replay, "parity", "observation", "max_abs_diff")),
        ("go2 residency speedup (device / numpy)", _pick(go2_residency, "comparison", "speedup")),
        ("go2 seed 0 final best fitness", _pick(go2_neat, "totals", "final_best_fitness")),
    ]
    assert abs(checks[0][1] - 4.76837158203e-07) < 1e-15, checks[0]
    assert 1.30 < checks[1][1] < 1.36, checks[1]
    assert abs(checks[2][1] + 6.03994607925415) < 1e-12, checks[2]
    return checks


def main(argv=None):
    """Parse ``argv``, aggregate, write the two artifacts and print the summary."""
    parser = argparse.ArgumentParser(
        prog="summarize.py",
        description="Aggregate the committed Stage-3 benchmark records into results/summary.json and results/tables.md.",
    )
    parser.add_argument("command", nargs="?", choices=("build", "check"), default="build", help="build (default) or run the self-check only")
    parser.add_argument("--results", default=str(RESULTS_DIR), help="directory of raw records")
    parser.add_argument("--out", default=str(SUMMARY_PATH), help="summary JSON path")
    parser.add_argument("--markdown", default=str(TABLES_PATH), help="markdown tables path")
    parser.add_argument("--print-tables", action="store_true", help="also print the markdown tables to stdout")
    parser.add_argument("--quiet", action="store_true", help="only print the written paths")
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    loaded = load_records(args.results)
    summary = build_summary(loaded)
    tables = markdown_document(summary)
    if args.command == "check":
        print(f"SELFCHECK OK: {summary['validation']['n_records']} records, {tables.count(chr(10) + '### T')} tables")
        for label, value in _selfcheck(loaded, summary):
            print(f"  pinned: {label} = {value}")
        return 0
    summary_path, tables_path = Path(args.out), Path(args.markdown)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(serialize_summary(summary), encoding="utf-8")
    tables_path.write_text(tables, encoding="utf-8")
    if args.print_tables:
        print(tables)
    if args.quiet:
        print(f"wrote {summary_path}")
        print(f"wrote {tables_path}")
    else:
        print_summary(summary, summary_path, tables_path)
    _selfcheck(loaded, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
