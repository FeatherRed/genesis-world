"""Markdown rendering of the aggregated Stage-3 benchmark summary.

This is the presentation half of :mod:`summarize`: it turns the ``summary.json`` payload into the
GitHub-flavoured tables that ``RESULTS.md`` embeds verbatim, so no number in the report is
hand-transcribed. It owns the number formatting and one builder per table family; it never reads a
raw record and never imports ``genesis``, ``torch``, ``tensordict`` or ``evoxc``.

Table ids are stable (``T1``.. ``T21``) and referenced from ``RESULTS.md``
"""

from __future__ import annotations

#: Timer groups that get their own column in the per-step time-split tables.
CHUNK_COLUMNS = (
    "physics",
    "graph_policy",
    "graph_advance",
    "state_read",
    "write_targets",
    "write_reset",
    "host_rng",
)

#: Timer groups folded into the ``other`` column (they are per-generation or tiny per step).
OTHER_GROUPS = ("graph_obs", "graph_controls", "graph_fitness", "download", "stage_upload", "sync")

#: Column order of the compact per-run startup rows stored in ``summary.json`` (T21).
STARTUP_RUN_FIELDS = (
    "case",
    "repeat",
    "wall_s",
    "import_s",
    "init_s",
    "module_to_init_s",
    "reexec_done",
    "opt_out",
)  # fmt: skip


def _num(value):
    """Return ``float(value)`` or ``None`` (records hold ``null`` for unmeasured values)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick(node, *keys):
    """Walk ``keys`` through nested dicts, returning ``None`` when a level is missing."""
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _at(values, index):
    """Return ``values[index]`` for a list, else ``None``."""
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return None


def _f(value, digits=3, unit=""):
    """Format a number for markdown (``n/a`` for ``None``, scientific for tiny magnitudes)."""
    number = _num(value)
    if number is None:
        return "n/a"
    if number == 0:
        return f"0{unit}"
    if abs(number) < 10.0 ** (-digits):
        return f"{number:.{digits}e}{unit}"
    return f"{number:.{digits}f}{unit}"


def _ms(seconds):
    """Format a per-step duration in milliseconds."""
    return "n/a" if seconds is None else _f(_num(seconds) * 1000.0, 4)


def _pct(value):
    """Format a fraction as a percentage."""
    return "n/a" if value is None else _f(_num(value) * 100.0, 2, "%")


def _mean_spread(block):
    """Render a ``common.stats`` block as ``median ± spread``."""
    if not isinstance(block, dict):
        return "n/a"
    return f"{_f(block.get('median'))} ± {_f(block.get('spread'))}"


def _table(columns, rows):
    """Render one GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(str(c) for c in columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _entries(summary, kind):
    """Every extracted per-record entry of ``kind``, across tasks and workload groups."""
    out = []
    for task in sorted(summary["families"].get(kind) or {}):
        for signature in sorted(summary["families"][kind][task]):
            out.extend(summary["families"][kind][task][signature]["entries"])
    return out


def _table_block(table_id, title, columns, rows, sources, note=None):
    """One markdown section: heading, optional note, the table, then its source footnote."""
    parts = [f"### {table_id} {title}", ""]
    if note:
        parts += [note, ""]
    parts += [_table(columns, rows), "", "Sources: " + ", ".join(f"`{name}.json`" for name in sorted(set(sources))), ""]
    return "\n".join(parts)


def _inventory_table(summary):
    """T1: one row per committed record with its provenance."""
    rows = []
    for kind in summary["families"]:
        for entry in _entries(summary, kind):
            rows.append(
                [
                    entry["record"],
                    kind,
                    entry["task"],
                    _f(entry["seed"], 0),
                    _f(entry["repeat"], 0),
                    entry["process_role"],
                    (entry["head"] or "")[:8],
                    entry["status"],
                    str(bool(entry.get("contended"))).lower(),
                ]
            )
    return [
        _table_block(
            "T1",
            "Record inventory",
            ["record", "kind", "task", "seed", "repeat", "process", "head", "status", "contended"],
            rows,
            [row[0] for row in rows],
            note="Head shas differ per record family because the harness evolved across commits; the pilot sources are byte-identical to the baseline in every record (`git.baseline_env_diff_empty`).",
        )
    ]


def _replay_tables(summary):
    """T2 (oracle parity) and T3 (replay throughput) of the pre-migration replay runs."""
    entries = _entries(summary, "env_replay")
    go2 = next((entry for entry in entries if entry["task"] == "go2"), None)
    hover = next((entry for entry in entries if entry["task"] == "hover"), None)
    if not go2 or not hover:
        return []
    rows = [
        ["steps compared", _f(_pick(go2, "parity", "n_steps_compared"), 0), _f(_pick(hover, "parity", "n_steps_compared"), 0)],
        ["observation max abs diff", _f(_pick(go2, "parity", "observation", "max_abs_diff"), 6), _f(_pick(hover, "parity", "observation", "max_abs_diff"), 6)],
        ["observation mismatches", _f(_pick(go2, "parity", "observation", "mismatch_count"), 0), _f(_pick(hover, "parity", "observation", "mismatch_count"), 0)],
        ["reward per-term max abs diff", "not measured (aggregate only)", "n/a"],
        ["reward total max abs diff", _f(_pick(go2, "parity", "reward", "total_max_abs_diff"), 6), _f(_pick(hover, "parity", "reward", "total_max_abs_diff"), 6)],
        ["reward total mismatches", _f(_pick(go2, "parity", "reward", "total_mismatch_count"), 0), _f(_pick(hover, "parity", "reward", "total_mismatch_count"), 0)],
        ["step return max abs diff", _f(_pick(go2, "parity", "returns", "step_max_abs_diff"), 6), _f(_pick(hover, "parity", "returns", "step_max_abs_diff"), 6)],
        ["accumulated return max abs diff", _f(_pick(go2, "parity", "returns", "accumulated_max_abs_diff"), 6), _f(_pick(hover, "parity", "returns", "accumulated_max_abs_diff"), 6)],
        ["fitness (from oracle) max abs diff", _f(_pick(go2, "parity", "returns", "fitness_from_oracle_max_abs_diff"), 6), _f(_pick(hover, "parity", "returns", "fitness_from_oracle_max_abs_diff"), 6)],
        ["done mismatch count", _f(_pick(go2, "parity", "termination", "done_mismatch_count"), 0), _f(_pick(hover, "parity", "termination", "mask_mismatch_count", "done"), 0)],
        ["tilted / crash mismatch count", _f(_pick(go2, "parity", "termination", "tilted_mismatch_count"), 0), _f(_pick(hover, "parity", "termination", "mask_mismatch_count", "crash"), 0)],
        ["timeout mismatch count", _f(_pick(go2, "parity", "termination", "timeout_mismatch_count"), 0), _f(_pick(hover, "parity", "termination", "mask_mismatch_count", "timeout"), 0)],
        ["episode-length mismatch count", _f(_pick(go2, "parity", "termination", "episode_length_mismatch_count"), 0), "n/a (mask-based)"],
        ["reset observation max abs diff", "n/a (Go2 resets from a host qpos)", _f(_pick(hover, "parity", "reset_observation", "max_abs_diff"), 6)],
        ["command max abs diff", "n/a (Go2 samples commands inside the graph)", _f(_pick(hover, "parity", "commands", "max_abs_diff"), 6)],
        ["yaw max abs diff (deg)", "n/a", _f(_pick(hover, "parity", "yaw_deg", "max_abs_diff"), 9)],
        ["fingerprint (generation 0)", (_pick(go2, "fingerprint") or "")[:12], (_pick(hover, "fingerprint") or "")[:12]],
    ]
    tables = [
        _table_block(
            "T2",
            "Environment replay parity against the pre-migration oracle",
            ["metric", "Go2 (pop 8, horizon 100)", "hover (pop 8, horizon 100)"],
            rows,
            [entry["record"] for entry in entries],
            note=(
                "Tolerances (record `workload.tolerances`): observation 1e-06, reward term and total 1e-05, "
                "step return 1e-04, accumulated return 1e-03, masks exact, hover yaw 1e-03 deg. "
                "Go2 reward is aggregate-only because `run_advance` returns the sum, not the six terms."
            ),
        )
    ]
    rows = [
        [
            entry["task"],
            entry["record"],
            _f(generation["generation"], 0),
            _f(generation["steps"], 0),
            _f(generation["steps_per_s"], 2),
            _f(generation["env_steps_per_s"], 1),
            _f(generation["wall_time_s"], 4),
        ]
        for entry in entries
        for generation in entry["generations"]
    ]
    tables.append(
        _table_block(
            "T3",
            "Environment replay throughput per generation",
            ["task", "record", "gen", "steps", "steps/s", "env-steps/s", "wall s"],
            rows,
            [entry["record"] for entry in entries],
            note="`gen 0` is the first generation in a fresh process; `gen 1` is steady state for this 2-generation replay.",
        )
    )
    return tables


def _throughput_tables(summary):
    """T4-T7: CPU throughput, per-step time split and the pop 8 / pop 32 comparison."""
    entries = _entries(summary, "env_throughput")
    rows = []
    for entry in entries:
        workload, block = entry["workload"], entry["throughput"] or {}
        rows.append(
            [
                entry["task"],
                _f(workload.get("pop_size"), 0),
                _f(workload.get("rows"), 0),
                entry["process_role"],
                _f(entry["repeat"], 0),
                _mean_spread(block.get("steps_per_s")),
                _mean_spread(block.get("env_steps_per_s")),
                _mean_spread(block.get("wall_time_s")),
                _f(_pick(entry, "phases", "first_generation_s"), 3),
                _pct(_pick(entry, "split", "residual_share")),
                entry["record"],
            ]
        )
    tables = [
        _table_block(
            "T4",
            "CPU NumPy-residency environment throughput (timed generations 1-3, horizon 200)",
            ["task", "pop", "rows", "process", "repeat", "steps/s (median ± spread)", "env-steps/s", "generation wall s", "first gen s", "residual", "record"],
            rows,
            [entry["record"] for entry in entries],
            note="Generation 0 is excluded everywhere (first-step JIT, empty caches). `generation wall s` is the per timed generation wall clock, median ± spread over generations 1-3. `residual` is the unmeasured share of the generation wall clock.",
        )
    ]
    columns = ["task", "pop", "record"] + list(CHUNK_COLUMNS) + ["other", "residual"]
    rows_ms, rows_share = [], []
    for entry in entries:
        split, workload = entry["split"], entry["workload"]
        head = [entry["task"], _f(workload.get("pop_size"), 0), entry["record"]]
        steps = split["steps"] or 1
        rows_ms.append(
            head
            + [_f(split["per_step_ms"].get(name), 6) for name in CHUNK_COLUMNS]
            + [_f(sum(split["per_step_ms"].get(name) or 0.0 for name in OTHER_GROUPS), 4), _ms((split["residual_s"] or 0.0) / steps)]
        )
        rows_share.append(
            head
            + [_pct(split["shares"].get(name)) for name in CHUNK_COLUMNS]
            + [_pct(sum(split["shares"].get(name) or 0.0 for name in OTHER_GROUPS)), _pct(split["residual_share"])]
        )
    tables.append(
        _table_block(
            "T5",
            "Per-step time split, milliseconds per step (timed generations)",
            columns,
            rows_ms,
            [entry["record"] for entry in entries],
            note="`other` = graph_obs + graph_controls + graph_fitness + download + stage_upload + sync.",
        )
    )
    tables.append(
        _table_block(
            "T6",
            "Per-step time share of the measured generation wall clock",
            columns,
            rows_share,
            [entry["record"] for entry in entries],
            note="Shares are exact sums over the three timed generations (`per_step_means[*].seconds` / timed wall clock); they add to 100% minus the residual.",
        )
    )
    rows = [
        [
            entry["task"],
            _f(entry["workload"].get("pop_size"), 0),
            _f(entry["workload"].get("rows"), 0),
            _f(_pick(entry, "throughput", "steps_per_s", "median"), 2),
            _f(_pick(entry, "throughput", "env_steps_per_s", "median"), 1),
            _f(_pick(entry, "phases", "steady_generation_median_s"), 4),
            _f(_pick(entry, "phases", "first_generation_s"), 3),
            entry["record"],
        ]
        for entry in entries
    ]
    tables.append(
        _table_block(
            "T7",
            "Batch scaling: pop 8 (24 rows) vs pop 32 (96 rows), horizon 200",
            ["task", "pop", "rows", "steps/s median", "env-steps/s median", "generation wall s", "first gen s", "record"],
            rows,
            [entry["record"] for entry in entries],
            note="Rows are policy-major (3 episodes per policy), so `rows = pop_size * 3`. Throughput does not scale linearly with the batch.",
        )
    )
    return tables


def _residency_tables(summary):
    """T8-T11: NumPy vs device residency comparison, parity, build and per-graph compile times."""
    entries = _entries(summary, "env_residency")
    tables = []
    rows = []
    for entry in entries:
        comparison = entry["comparison"] or {}
        rows.append(
            [
                entry["task"],
                entry["record"],
                _f(_pick(comparison, "steps_per_s_median", "numpy"), 2),
                _f(_pick(comparison, "steps_per_s_median", "device"), 2),
                _f(comparison.get("speedup"), 3) + "x",
                _f(_pick(comparison, "env_steps_per_s_median", "numpy"), 1),
                _f(_pick(comparison, "env_steps_per_s_median", "device"), 1),
                _f(_pick(comparison, "host_bytes_per_step", "numpy"), 1),
                _f(_pick(comparison, "host_bytes_per_step", "device"), 1),
                _f(_pick(comparison, "host_calls_per_step", "numpy"), 3),
                _f(_pick(comparison, "host_calls_per_step", "device"), 3),
                _f(_pick(comparison, "sync_calls_per_step", "numpy"), 3),
                _f(_pick(comparison, "sync_calls_per_step", "device"), 3),
            ]
        )
    tables.append(
        _table_block(
            "T8",
            "Residency comparison: NumPy residency vs device residency (same physics backend, same seed)",
            ["task", "record", "numpy steps/s", "device steps/s", "speedup", "numpy env-steps/s", "device env-steps/s", "numpy B/step", "device B/step", "numpy calls/step", "device calls/step", "numpy sync/step", "device sync/step"],
            rows,
            [entry["record"] for entry in entries],
            note="`residency_hover_device_cpu*` is the `iree@cpu:0` device run: both sides are CPU, so its ratio is noise-dominated and must not be quoted as a win.",
        )
    )
    rows = [
        [
            entry["task"],
            entry["record"],
            _f(_pick(entry, "parity", "arrays", "seeds", "max_abs_diff"), 6),
            _f(_pick(entry, "parity", "arrays", "init_qpos", "max_abs_diff"), 6),
            _f(_pick(entry, "parity", "arrays", "returns", "max_abs_diff"), 6),
            _f(_pick(entry, "parity", "arrays", "episode_returns", "max_abs_diff"), 6),
            _f(_pick(entry, "parity", "arrays", "fitness", "max_abs_diff"), 6),
            _f(_pick(entry, "parity", "arrays", "commands", "max_abs_diff"), 6),
            str(bool(_pick(entry, "parity", "fingerprint", "equal"))),
            _f(_pick(entry, "sides", "numpy", "state_read_parity", "base_quat", "max_abs_diff"), 6),
            str(bool(_pick(entry, "parity", "all_pass"))),
        ]
        for entry in entries
    ]
    tables.append(
        _table_block(
            "T9",
            "Residency parity (generation 0, identical seed and perturbation)",
            ["task", "record", "seeds", "init_qpos", "returns", "episode_returns", "fitness", "commands", "fingerprint equal", "base_quat (getter vs buffer)", "all pass"],
            rows,
            [entry["record"] for entry in entries],
            note="Every column is a max abs diff. `base_quat` is `state_read_parity`: NumPy residency reads the entity getter, device residency the live solver buffer, which is the only arithmetic difference between the two paths.",
        )
    )
    rows = []
    for entry in entries:
        for side in ("numpy", "device"):
            block = entry["sides"][side]
            checks = block.get("checks")
            rows.append(
                [
                    entry["task"],
                    entry["record"],
                    side,
                    _f(block["build_s"], 2),
                    _f(block["graph_build_total_s"], 4),
                    _f(block["first_generation_s"], 3),
                    _f(block["steady_generation_median_s"], 4),
                    _f(_pick(block, "throughput", "steps_per_s", "median"), 2),
                    _f(_pick(block, "throughput", "env_steps_per_s", "median"), 1),
                    f"{_f(_pick(block, 'cache', 'hits'), 0)}/{_f(_pick(block, 'cache', 'misses'), 0)}",
                    "n/a" if not checks else str(bool(checks.get("pass"))),
                ]
            )
    tables.append(
        _table_block(
            "T10",
            "Residency build, first generation and steady state per side",
            ["task", "record", "side", "build s", "graph build total s", "first gen s", "steady gen s", "steps/s median", "env-steps/s median", "cache hits/misses", "checks pass"],
            rows,
            [entry["record"] for entry in entries],
            note="`build s` is the whole side's construction (scene + every graph); `graph build total s` is the sum of the per-graph compile times only.",
        )
    )
    rows = []
    for entry in entries:
        for side in ("numpy", "device"):
            builds = entry["sides"][side].get("graph_build_s") or {}
            rows.append(
                [entry["task"], entry["record"], side]
                + [_f(_pick(builds, name, "build_s"), 4) for name in ("advance", "policy", "obs", "fitness", "controls", "reset_obs", "tell")]
                + [_f(entry["sides"][side]["graph_build_total_s"], 4)]
            )
    tables.append(
        _table_block(
            "T11",
            "Per-graph compile times per residency side (seconds)",
            ["task", "record", "side", "advance", "policy", "obs", "fitness", "controls", "reset_obs", "tell", "total"],
            rows,
            [entry["record"] for entry in entries],
            note="`n/a` means the task has no such graph (Go2 has no controls; no residency side builds a tell graph).",
        )
    )
    return tables


def _neat_tables(summary):
    """T12-T15: per-generation curves, held-out score, run configurations and determinism."""
    entries = _entries(summary, "neat_train")
    tables = []
    rows = []
    for entry in entries:
        curve = entry["learning_curve"]
        for index, generation in enumerate(entry["generations"]):
            rows.append(
                [
                    entry["task"],
                    _f(entry["seed"], 0),
                    _f(generation["generation"], 0),
                    _f(generation["best_fitness"], 5),
                    _f(generation["median_fitness"], 5),
                    _f(_at(curve.get("cumulative_env_steps"), index), 0),
                    _f(_at(curve.get("cumulative_loop_s"), index), 3),
                    _f(generation["evaluate_s"], 4),
                    _f(generation["tell_s"], 5),
                    _f(generation["generation_s"], 4),
                    _f(generation["steps_per_s"], 2),
                    entry["record"],
                ]
            )
    tables.append(
        _table_block(
            "T12",
            "TensorNEAT trainer: per-generation learning curves (pop 8, 3 generations, horizon 200)",
            ["task", "seed", "gen", "best fitness", "median fitness", "cumulative env steps", "cumulative loop s", "evaluate s", "tell s", "generation s", "steps/s", "record"],
            rows,
            [entry["record"] for entry in entries],
            note="Rows include the CPU NumPy-residency and the GPU runs; T14 states which is which. `evaluate_s` is `env.evaluate` (seed draw + reset + steps + fitness graph), `generation_s` the whole generation including `tell_s`.",
        )
    )
    rows = [
        [
            entry["task"],
            _f(entry["seed"], 0),
            _f(_pick(entry, "holdout", "initial_median"), 5),
            _f(_pick(entry, "holdout", "trained_median"), 5),
            _f(_pick(entry, "holdout", "median_delta"), 5),
            _f(_pick(entry, "holdout", "best_policy", "train_fitness"), 5),
            _f(_pick(entry, "holdout", "best_policy", "heldout_fitness"), 5),
            _f(_pick(entry, "holdout", "best_policy", "delta"), 5),
            str(bool(_pick(entry, "holdout", "separates_initial_from_trained"))),
            entry["record"],
        ]
        for entry in entries
    ]
    tables.append(
        _table_block(
            "T13",
            "Held-out score: generation-0 population vs last-generation population on one unseen seed triple",
            ["task", "seed", "initial median", "trained median", "median delta", "best policy train", "best policy held-out", "best policy held-out improvement", "separates", "record"],
            rows,
            [entry["record"] for entry in entries],
            note="Held-out seeds are `seeding.generation_seed_table(master_seed + 1000, 1)`, never used by a training generation. `median delta` is the median over policies of (trained − initial) on that triple; a negative value means the trained population scored worse. `best policy train`/`held-out` are that policy's fitness on the training seeds and on the unseen triple, and the last column is its held-out improvement, not a train-vs-held-out gap.",
        )
    )
    rows = [
        [
            entry["task"],
            entry["record"],
            entry.get("residency") or _pick(entry, "workload", "residency"),
            _pick(entry, "tell", "backend"),
            entry["process_role"],
            _f(entry["repeat"], 0),
            _f(_pick(entry, "phases", "steady_generation_median_s"), 4),
            _f(_pick(entry, "totals", "steps_per_s", "median"), 2),
            _f(_pick(entry, "totals", "env_steps_per_s", "median"), 1),
            _f(_pick(entry, "phases", "tell_build_s"), 3),
            _f(_pick(entry, "phases", "first_generation_s"), 3),
            _f(_pick(entry, "totals", "generation_s"), 4),
            _f(_pick(entry, "holdout", "trained_median"), 5),
            (_pick(entry, "checks", "determinism_fingerprint") or "")[:12],
        ]
        for entry in entries
    ]
    tables.append(
        _table_block(
            "T14",
            "Trainer configurations: NumPy vs device residency, NumPy vs XLA tell, CPU vs GPU physics",
            ["task", "record", "residency", "tell backend", "process", "repeat", "steady gen s", "steps/s median", "env-steps/s median", "tell build s", "first gen s", "generation_s total", "held-out trained median", "fingerprint"],
            rows,
            [entry["record"] for entry in entries],
            note="`residency` is where the rollout operands live; `tell backend` is where the NEAT generation transition runs. `neat_*_seed1/2` are master seeds 1 and 2; every other run is seed 0.",
        )
    )
    determinism = _entries(summary, "neat_determinism")
    rows = [
        [
            entry["record"],
            _pick(entry, "checks", "fingerprint", "a"),
            _pick(entry, "checks", "fingerprint", "b"),
            str(bool(_pick(entry, "checks", "fingerprint", "match"))),
            _f(_pick(entry, "checks", "seed_table", "max_abs_diff"), 6),
            _f(_pick(entry, "checks", "fitness", "max_abs_diff"), 6),
            _f(_pick(entry, "checks", "holdout_initial", "max_abs_diff"), 6),
            _f(_pick(entry, "checks", "holdout_trained", "max_abs_diff"), 6),
            _f(len(_pick(entry, "checks", "generation_seeds", "mismatched_generations") or []), 0),
            str(bool(_pick(entry, "verdict", "ok"))),
        ]
        for entry in determinism
    ]
    tables.append(
        _table_block(
            "T15",
            "Determinism cross-check: two processes of the same command must agree bit-for-bit",
            ["record", "fingerprint A", "fingerprint B", "match", "seed table max diff", "fitness max diff", "held-out initial max diff", "held-out trained max diff", "mismatched generations", "verdict ok"],
            rows,
            [entry["record"] for entry in determinism],
            note="The compared pair is `neat_go2_seed0.json` (process `only`) and `neat_go2_seed0_r1.json` (process `warm`).",
        )
    )
    return tables


def _probe_tables(summary):
    """T16-T20: unmatched baselines, the tell-backend matrix and the torch/CPU boundary."""
    tables = []
    unmatched = _entries(summary, "probe_unmatched_baselines")
    if unmatched:
        measurements = unmatched[0]
        rows = [
            ["import", name, case.get("verdict"), _f(case.get("elapsed_s"), 3), f"{_pick(case, 'exception', 'type')}: {_pick(case, 'exception', 'message')}", _f(case.get("returncode"), 0)]
            for name, case in sorted((measurements.get("imports") or {}).items())
        ]
        rows += [
            ["thin adapter", name, case.get("verdict"), _f(case.get("elapsed_s"), 3), f"{_pick(case, 'exception', 'type')}: {_pick(case, 'exception', 'message')}", _f(case.get("returncode"), 0)]
            for name, case in sorted((measurements.get("adapter_attempts") or {}).items())
        ]
        install = measurements.get("install_attempt") or {}
        rows.append(["offline install", "pip --no-deps --target $TMPDIR", install.get("status"), _f(_pick(install, "pip", "elapsed_s"), 3), "No module named pip; uv not on PATH", _f(_pick(install, "pip", "returncode"), 0)])
        ppo = measurements.get("ppo") or {}
        rows.append(["PPO baseline", "rsl-rl PPO", ppo.get("status"), "n/a", f"{_pick(ppo, 'error', 'type')}: {_pick(ppo, 'error', 'message')}", "n/a"])
        tables.append(
            _table_block(
                "T16",
                "Unmatched baselines: every import, adapter and install attempt",
                ["case kind", "case", "verdict", "elapsed s", "exception", "returncode"],
                rows,
                [unmatched[0]["record"]],
                note="Each case ran as its own subprocess with a hard timeout. Neither torch environment could be imported, so no torch-vs-migrated environment comparison exists.",
            )
        )
    for entry in _entries(summary, "probe_tell_backend"):
        rows = []
        for name, case in sorted((entry.get("entries") or {}).items()):
            result = case.get("result") or {}
            rows.append(
                [
                    name,
                    case.get("status"),
                    _f(case.get("elapsed_s"), 3),
                    case.get("last_phase") or "n/a",
                    _f(result.get("build_s"), 3),
                    _f(result.get("run_s"), 3),
                    _f(result.get("total_s"), 3),
                    case.get("timeout_s") if case.get("timed_out") else "n/a",
                    case.get("plugin_path") or "n/a",
                ]
            )
        tables.append(
            _table_block(
                "T17",
                "tell-graph backend matrix on ROCm (`_tell_worker.py`, pop 4, genome 45 -> 12)",
                ["backend@device", "status", "elapsed s", "last phase", "build s", "run s", "total s", "timeout s", "PJRT plugin"],
                rows,
                [entry["record"]],
                note="`iree@rocm:0` is killed at its hard timeout inside `build_tell` (`returncode -9`); the phases it reached are in the raw record.",
            )
        )
    for entry in _entries(summary, "probe_torch_cpu_boundary"):
        measurements = entry
        scan_summary = _pick(measurements, "ast_scan", "summary") or {}
        rows = []
        for path, facts in sorted((_pick(measurements, "ast_scan", "files") or {}).items()):
            rows.append(
                [
                    path,
                    "yes" if facts.get("torch_module") else "no",
                    _f(facts.get("torch_deferred"), 0),
                    _f(facts.get("genesis"), 0),
                    _f(facts.get("tensordict"), 0),
                    _f(facts.get("rsl_rl"), 0),
                ]
            )
        tables.append(
            _table_block(
                "T18",
                "Torch/CPU boundary: AST import scan",
                ["file", "torch at import time", "torch inside functions", "genesis mentions", "tensordict mentions", "rsl_rl mentions"],
                rows,
                [entry["record"]],
                note="`torch_at_import_time` = "
                + ", ".join(scan_summary.get("torch_at_import_time") or ["none"])
                + "; `no_torch` = "
                + _f(scan_summary.get("no_torch_n"), 0)
                + " of "
                + _f(scan_summary.get("scanned"), 0)
                + " scanned files.",
            )
        )
        report = _pick(measurements, "runtime_probe", "report") or {}
        finds = report.get("find_spec") or {}
        rows = [
            ["genesis import time (fresh process)", _f(report.get("genesis_import_s"), 3) + " s"],
            ["torch in sys.modules before `import genesis`", str(bool(report.get("torch_in_sys_modules_before_genesis")))],
            ["torch in sys.modules after `import genesis`", str(bool(report.get("torch_in_sys_modules_after_import_genesis")))],
            ["torch version", str(report.get("torch_version"))],
            ["torch HIP version", str(report.get("torch_hip_version"))],
            ["HIP device", str(report.get("hip_device_name"))],
            ["`find_spec` tensordict / rsl_rl", f"{finds.get('tensordict')} / {finds.get('rsl_rl')}"],
            ["`find_spec` evoxc / tensorneat / torch", f"{finds.get('evoxc')} / {finds.get('tensorneat')} / {finds.get('torch')}"],
        ]
        tables.append(
            _table_block(
                "T19",
                "Torch/CPU boundary: fresh-subprocess `import genesis` probe",
                ["fact", "value"],
                rows,
                [entry["record"]],
                note="`import genesis` pulls `torch` into the process even though the neuroevolution layer does not import it.",
            )
        )
        rows = []
        for task, families in sorted((_pick(measurements, "measured_boundary") or {}).items()):
            if not isinstance(families, dict):
                continue
            for family, block in sorted(families.items()):
                values = block.get("values") or {}
                rows.append(
                    [
                        task,
                        family,
                        block.get("source"),
                        (block.get("git_head") or "")[:8],
                        block.get("status"),
                        _f(values.get("state_read_ms_per_step"), 4),
                        _pct(values.get("state_read_share_of_generation")),
                        _f(_pick(values, "host_bytes_per_step", "numpy"), 1),
                        _f(_pick(values, "host_bytes_per_step", "device"), 1),
                        _f(_pick(values, "sync_calls_per_step", "device"), 3),
                    ]
                )
        tables.append(
            _table_block(
                "T20",
                "Torch/CPU boundary: cited committed numbers (never re-measured by the probe)",
                ["task", "family", "source record", "head", "status", "state_read ms/step", "state_read share", "numpy B/step", "device B/step", "device sync/step"],
                rows,
                [entry["record"]],
                note="The probe cites these values from the records named in the `source record` column; the head column is the head that produced the cited record.",
            )
        )
    return tables


def _startup_sentinel(run):
    """How the ``GS_HSA_REEXEC_DONE`` sentinel and the opt-out identify one run's case."""
    if run.get("reexec_done") == "1":
        return "`GS_HSA_REEXEC_DONE=1` (re-executed)"
    if run.get("opt_out") == "0":
        return "none (`GS_HSA_REEXEC=0`)"
    return "none (no HSA check)"


def _startup_tables(summary):
    """T21: process start up to ``gs.init``, with and without the ROCm HSA re-exec repair."""
    tables = []
    for entry in _entries(summary, "probe_startup"):
        runs = [dict(zip(STARTUP_RUN_FIELDS, row)) for row in entry.get("runs") or []]
        rows = [
            [
                run["case"],
                _f(run["repeat"], 0),
                _f(run["wall_s"]),
                _f(run["import_s"]),
                _f(run["init_s"]),
                _f(run["module_to_init_s"]),
                _startup_sentinel(run),
                entry["record"],
            ]
            for run in runs
        ]
        cases, derived = entry.get("cases") or {}, entry.get("derived") or {}
        medians = "; ".join(
            f"`{case}` {_mean_spread(_pick(cases, case, 'wall_s'))} s wall / "
            f"{_mean_spread(_pick(cases, case, 'module_to_init_s'))} s module→init"
            for case in ("repair", "no_repair", "cpu")
            if _pick(cases, case, "wall_s")
        )
        note = (
            "`wall` is spawn→exit of the whole `python` process (so it includes the re-exec when one happens); "
            "`module→init` is `import genesis` + `gs.init` in the process that printed the payload (the "
            "re-executed child when the repair applies). Both are process start only: they end before "
            "`phases.build_s` begins, so no build, compile or first-generation number in this report is "
            f"inflated by them. Median ± spread over the repeats: {medians}. Derived re-exec cost: "
            f"{_f(derived.get('reexec_cost_s'))} s of wall (`repair` − `no_repair`), which is the import and "
            f"partial init the repair performs twice, against {_f(derived.get('reexec_cost_module_s'))} s on "
            "module→init, i.e. the repaired child is not slower to import — the whole cost is the second "
            "process start."
        )
        tables.append(
            _table_block(
                "T21",
                "GPU process start: with and without the HSA re-exec repair",
                ["case", "repeat", "wall s", "import s", "init s", "module→init s", "re-exec sentinel", "record"],
                rows,
                [entry["record"]],
                note=note,
            )
        )
    return tables


def markdown_document(summary):
    """Render the whole ``tables.md`` document from a ``summarize.build_summary`` payload."""
    validation = summary["validation"]
    environment = summary["environment"]
    lines = [
        "# Stage-3 benchmark tables",
        "",
        "Generated from `results/summary.json` by `summarize.py`; do not edit by hand.",
        "Every table names the record files it was derived from, and every record is validated for schema, baseline provenance and status before it is aggregated.",
        "",
        f"Records: {validation['n_records']} ({validation['n_ok']} ok, {validation['n_failed']} failed, {validation['n_contended']} contended). "
        f"Baseline `{validation['baseline_sha_expected']}`; `git.baseline_env_diff_empty` in every record: {str(validation['baseline_env_diff_empty_all']).lower()}.",
        f"Host: {', '.join(environment['gpu_names']) or 'no GPU'}, ROCm driver {', '.join(environment['rocm_driver']) or 'n/a'}, "
        f"1-minute load average {_f(_pick(environment, 'loadavg_1m', 'min'), 2)}-{_f(_pick(environment, 'loadavg_1m', 'max'), 2)} over {_f(_pick(environment, 'loadavg_1m', 'n'), 0)} records.",
        "",
    ]
    lines += (
        ["Validation warnings:", ""] + [f"- {warning}" for warning in validation["warnings"]] + [""]
        if validation["warnings"]
        else ["Validation warnings: none.", ""]
    )
    sections = (
        _inventory_table(summary)
        + _replay_tables(summary)
        + _throughput_tables(summary)
        + _residency_tables(summary)
        + _neat_tables(summary)
        + _probe_tables(summary)
        + _startup_tables(summary)
    )
    return "\n".join(lines + sections).rstrip() + "\n"
