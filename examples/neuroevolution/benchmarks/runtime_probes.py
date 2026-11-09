"""Timing and traffic probes of the Phase-2 runtime benchmarks.

This module holds the record's measurement vocabulary -- the timer groups, the traffic tags, the
description of what each region covers and the list of device-resident intermediates -- together
with the monkey-patches that produce those numbers: the per-graph compile timer
(:func:`_wrap_executable_cache`), the host/device transfer and rollout-entry wrappers of both
residencies, the ``RigidOptions`` runtime override and the getter-vs-live-buffer state probe.

Nothing here builds an environment or writes a record; :mod:`runtime_measure` runs the
environments with these probes installed and :mod:`bench_runtime` owns the CLI.  Every probe
returns a ``restore`` callable so no patch outlives the environment it was installed for.
"""

from __future__ import annotations

import time

import numpy as np

import bench_envs
from bench_envs import _instrument, _r
from common import wrap_method, wrap_traffic

RESULTS_DIR = bench_envs.RESULTS_DIR

#: Generation index used as the warm-up (compile/JIT dominated); never in the headline median.
WARMUP_GENERATION = 0
#: Timed repeats after the warm-up generation.
TIMED_GENERATIONS = 3
DEFAULT_GENERATIONS = 1 + TIMED_GENERATIONS

#: The graph kind the executable cache key starts with, and the only values seen today.
GRAPH_NAMES = ("obs", "policy", "advance", "fitness", "controls", "reset_obs", "tell")

#: Timer accumulators: the Phase-1 groups plus the device-rollout boundary of each residency.
TIMER_GROUPS = (
    *bench_envs.TIMER_GROUPS,
    # Device-resident rollout of one generation (its ``run`` / ``evaluate`` entry point).
    "rollout_total",
    # hover device: the inline ``evoxc.run`` calls (policy / controls / advance), which do not go
    # through a named method and are therefore timed through ``evoxc.run`` itself.
    "graph_all",
    # go2 device: the explicit torch <-> IREE fences (``_sync``).
    "sync",
    # Host -> device staging (go2 ``_stage``, hover ``_host_operand``).
    "stage_upload",
    # Device -> host readback (go2 ``_host``, hover ``_download``).
    "download",
    # hover device: graph output -> genesis export (contains the module's only synchronisation).
    "export",
    # hover device: the solver-error latch query (a host query by contract).
    "host_query",
)

#: Timer groups that are not summed into ``timings.measured_s`` (they contain other groups).
_SCOPE_TAGS = ("generation_total", "rollout_total")

#: Timer group descriptions added on top of :data:`bench_envs.TIMER_DESCRIPTIONS`.
RUNTIME_TIMER_DESCRIPTIONS = {
    "rollout_total": "go2_device_rollout._DeviceRollout.run / hover_device_rollout.DeviceRollout.evaluate",
    "graph_all": "evoxc.run inside the hover device rollout (policy + controls + advance inline)",
    "sync": "go2 device _sync: torch.cuda.synchronize() fencing torch against IREE's queue",
    "stage_upload": "host -> device staging of one operand",
    "download": "device -> host readback of one graph output",
    "export": "hover device _export: graph output -> genesis, including the single GPUSync",
    "host_query": "hover device _error_mask: scene.get_error_envs_mask as a host array",
}

#: Traffic counter names (the host <-> device boundaries a residency crosses).
TRAFFIC_NAMES = (
    "read_state",
    "write_actuators",
    "write_reset",
    "host_rng",
    "graph_policy",
    "graph_obs",
    "graph_advance",
    "graph_controls",
    "graph_reset_obs",
    "graph_fitness",
    "stage_upload",
    "download",
    "export",
    "import_state",
    "sync",
    "host_query",
    "classify_endings",
)

#: What the device residencies keep on the accelerator (from the device modules' docstrings).
DEVICE_RESIDENT_INTERMEDIATES = {
    "go2": {
        "source": "examples/locomotion/go2_device_rollout.py, section 'What stays on the device'",
        "items": [
            "observations (the obs output is handed straight to the policy graph)",
            "raw and clipped actions",
            "last_actions",
            "returns / alive / episode_length (fed back into the advance graph)",
            "PD joint targets (built from last_actions on the device)",
            "the per-generation init-qpos tile and the reset rows gathered from it",
            "the zero-valued reset operands (base_ang_vel_world / base_quat / dof_vel / actions)",
        ],
        "host_touches_per_step": [
            "the (rows,) done mask, for np.flatnonzero and the end_reasons classification",
            "the solver-error test, plus the base_rpy_deg / error_mask rows on a step that ended",
        ],
    },
    "hover": {
        "source": "examples/drone/hover_device_rollout.py, sections 'Host round-trips left' and '_ensure_state'",
        "items": [
            "the staged base-state buffers (pos / quat / vel / ang) imported over the live solver buffers",
            "observation, clipped actions and propeller RPM",
            "returns / alive / episode_length / end_code / commands",
            "last_actions and last_base_pos",
            "the four genome payload leaves (staged once per generation)",
        ],
        "host_touches_per_step": [
            "scene.get_error_envs_mask() (a host query by contract)",
            "the two host command draws (x, y uniforms, plus the pinned z), uploaded per step",
        ],
    },
}


# --- byte estimates ----------------------------------------------------------------


def _nbytes(value) -> int:
    """Estimate the bytes of one array-like value (numpy, torch, evoxc, container, else 0)."""
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        return sum(_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(item) for item in value)
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            element = getattr(value, "element_size", None)
            width = int(element()) if callable(element) else 4
            return int(numel()) * width
        except Exception:  # noqa: BLE001 - an unmeasurable tensor counts as zero bytes
            return 0
    return 0


def _arg_bytes(index: int):
    """A :func:`common.wrap_traffic` size function over one positional argument."""

    def size(args, kwargs, result):
        return _nbytes(args[index]) if len(args) > index else 0

    return size


def _result_bytes(args, kwargs, result):
    """A :func:`common.wrap_traffic` size function over the call's result."""
    return _nbytes(result)


def _call_bytes(args, kwargs, result):
    """Estimate every operand and the result of a graph call (host <-> graph traffic)."""
    return sum(_nbytes(arg) for arg in args[1:]) + sum(_nbytes(value) for value in kwargs.values()) + _nbytes(result)


# --- executable-cache instrumentation ----------------------------------------------


def _wrap_executable_cache(builds: dict):
    """Patch ``evolution_graphs.ExecutableCache.get`` at class level and time every build.

    The cache key's first element names the graph (``"obs"``, ``"policy"``, ``"advance"``,
    ``"fitness"``, ``"controls"``, ``"reset_obs"``, ``"tell"``), so one timer per graph separates
    the compile calls of the cold process from the cache lookups of the warm ones.  A call is a
    build exactly when the cache's ``misses`` counter moved during it.
    """
    from evolution_graphs import ExecutableCache

    original = ExecutableCache.get

    def wrapper(self, key, builder):
        name = str(tuple(key)[0]) if key else "unknown"
        misses_before = self.misses
        start = time.perf_counter()
        try:
            return original(self, key, builder)
        finally:
            elapsed = time.perf_counter() - start
            entry = builds.setdefault(name, {"calls": 0, "builds": 0, "build_s": 0.0, "lookup_s": 0.0})
            entry["calls"] += 1
            if self.misses > misses_before:
                entry["builds"] += 1
                entry["build_s"] += elapsed
            else:
                entry["lookup_s"] += elapsed

    wrapper.__name__ = "get"
    wrapper.__wrapped__ = original
    ExecutableCache.get = wrapper

    def restore():
        ExecutableCache.get = original

    return restore


def _patch_collision_cap(module, cap: int):
    """Raise ``gs.options.RigidOptions.max_collision_pairs`` from the harness, without an edit.

    ``examples/locomotion/go2_neat_env.py`` hardcodes ``max_collision_pairs=20`` (a memory-saving
    cap that raises when the narrow phase exceeds it) outside its ``CFG``, so a larger batch can
    only be measured by overriding the constructor at runtime.  The patch wraps
    ``gs.options.RigidOptions`` for the lifetime of one environment build and raises the cap to at
    least ``cap`` (the env passes its own value explicitly, so a plain ``setdefault`` would keep
    it); the sibling source file is never modified and the original class is restored afterwards.

    Returns:
        callable: Restore function putting the original ``RigidOptions`` back.
    """
    import genesis as gs

    original = gs.options.RigidOptions

    def RigidOptions(**kwargs):
        current = int(kwargs.get("max_collision_pairs") or 0)
        kwargs["max_collision_pairs"] = max(current, int(cap))
        return original(**kwargs)

    RigidOptions.__name__ = "RigidOptions"
    RigidOptions.__doc__ = getattr(original, "__doc__", None)
    gs.options.RigidOptions = RigidOptions

    def restore():
        gs.options.RigidOptions = original

    return restore


def _state_read_parity(env) -> dict:
    """Compare the env's state getters with the live solver buffers the device path reads.

    Go2's device rollout reads ``rigid_info.qpos`` / ``dyn_state.dofs`` zero-copy, while the NumPy
    residency reads the entity getters; this block measures the two against each other, which is
    what a residency parity difference can be attributed to (it is independent of the graphs).
    """
    import torch
    from genesis.utils.misc import qd_to_torch

    solver = env.robot._solver
    qpos = qd_to_torch(solver.rigid_info.qpos, transpose=True, copy=False)
    dofs_pos = qd_to_torch(solver.dyn_state.dofs.pos, transpose=True, copy=False)[:, env._control_dof_slice]
    slot = torch.as_tensor(env._qpos_dof_slot, device=dofs_pos.device)
    pairs = {
        "base_pos": (env.robot.get_pos(), qpos[:, :3]),
        "base_quat": (env.robot.get_quat(), qpos[:, 3:7]),
        "dof_pos": (env.robot.get_dofs_position(env.motors_dof_idx), dofs_pos.index_select(1, slot)),
    }
    out = {}
    for name, (getter, buffer) in pairs.items():
        left = getter.detach().float().cpu().numpy()
        right = buffer.detach().float().cpu().numpy()
        out[name] = {
            "max_abs_diff": _r(float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))),
            "bit_identical": bool(np.array_equal(left, right)),
        }
    out["note"] = (
        "getter vs live solver buffer, read after the last generation; a nonzero base_quat entry "
        "means the two residencies feed the graphs a last-bit different orientation, which is the "
        "only arithmetic difference between them"
    )
    return out


# --- instrumentation ---------------------------------------------------------------


def _instrument_numpy_path(task, env, timer, traffic):
    """Time and count the host-staged (NumPy residency) path; no step capture.

    The timers are :func:`bench_envs._instrument` with ``None`` captures (nothing is kept, so a
    200-step generation stays cheap) and the traffic counters wrap the same boundaries again to
    measure how many bytes cross them.
    """
    restores = _instrument(task, env, timer, None, None)
    if task == "go2":
        import go2_neat_env as module

        restores += [
            wrap_traffic(module, "run_policy", traffic, "graph_policy", size=_call_bytes),
            wrap_traffic(module, "run_obs", traffic, "graph_obs", size=_call_bytes),
            wrap_traffic(module, "run_advance", traffic, "graph_advance", size=_call_bytes),
            wrap_traffic(module, "run_fitness", traffic, "graph_fitness", size=_call_bytes),
            wrap_traffic(type(env), "read_state", traffic, "read_state", size=_result_bytes),
            wrap_traffic(type(env), "_write_targets", traffic, "write_actuators", size=_arg_bytes(1)),
            wrap_traffic(type(env), "_write_reset", traffic, "write_reset", size=_arg_bytes(1)),
            # The end-of-row classification both residencies delegate to the env.
            wrap_traffic(type(env), "_classify_endings", traffic, "classify_endings"),
        ]
    else:
        import hover_neat_env as module

        restores += [
            wrap_traffic(module, "run_policy", traffic, "graph_policy", size=_call_bytes),
            wrap_traffic(module, "run_controls", traffic, "graph_controls", size=_call_bytes),
            wrap_traffic(module, "run_reset_obs", traffic, "graph_reset_obs", size=_call_bytes),
            wrap_traffic(module, "run_advance", traffic, "graph_advance", size=_call_bytes),
            wrap_traffic(module, "run_fitness", traffic, "graph_fitness", size=_call_bytes),
            wrap_traffic(type(env), "read_state", traffic, "read_state", size=_result_bytes),
            wrap_traffic(type(env), "reset_generation", traffic, "write_reset", size=_reset_bytes),
            wrap_traffic(type(env.drone), "set_propellers_rpm", traffic, "write_actuators", size=_arg_bytes(1)),
        ]
    return restores


def _reset_bytes(args, kwargs, result):
    """Bytes of one hover generation reset: the init pose it writes plus its commands and obs."""
    env = args[0]
    written = _nbytes(getattr(env, "_reset_pos", None)) + _nbytes(getattr(env, "_reset_quat", None))
    return written + _nbytes(result)


def _instrument_device_path(task, env, timer, traffic):
    """Time and count the device-resident rollout (class level, before the first ``evaluate``).

    The rollout object is created lazily inside ``env.evaluate``, so patching the class is what
    patches every instance.  Only the task's own rollout class is patched; the NumPy path of the
    same task never calls into it.

    Timers are installed only on boundaries that do not contain each other: ``_import`` runs
    inside ``_ensure_state`` inside ``_read_state`` and ``_policy_payloads`` wraps four
    ``_host_operand`` calls, so those inner hooks are counted for *traffic* only (calls, and bytes
    on the primitive that actually moves them).  Without that split one timing would be attributed
    to two groups and ``residual_s`` would go negative.
    """
    restores = []
    if task == "go2":
        import go2_device_rollout as rollout

        cls = rollout._DeviceRollout
        for name, tag in (
            ("run", "rollout_total"),
            ("_run_policy", "graph_policy"),
            ("_run_advance", "graph_advance"),
            ("_reset_obs", "graph_obs"),
            ("_fitness", "graph_fitness"),
            ("_read_state", "state_read"),
            ("_sync", "sync"),
            ("_stage", "stage_upload"),
            ("_host", "download"),
        ):
            restores.append(wrap_method(cls, name, timer, tag))
        restores += [
            wrap_traffic(cls, "_stage", traffic, "stage_upload", size=_arg_bytes(1)),
            wrap_traffic(cls, "_host", traffic, "download", size=_result_bytes),
            wrap_traffic(cls, "_sync", traffic, "sync"),
            wrap_traffic(cls, "_read_state", traffic, "read_state"),
        ]
    else:
        import hover_device_rollout as rollout

        cls = rollout.DeviceRollout
        for name, tag in (
            ("evaluate", "rollout_total"),
            ("_host_operand", "stage_upload"),
            ("_download", "download"),
            ("_export", "export"),
            ("_read_state", "state_read"),
            ("_error_mask", "host_query"),
        ):
            restores.append(wrap_method(cls, name, timer, tag))
        restores += [
            # Bytes are counted on the primitive that moves them: ``_host_operand`` uploads,
            # ``_download`` downloads.  The wrappers around them only count calls.
            wrap_traffic(cls, "_policy_payloads", traffic, "stage_upload"),
            wrap_traffic(cls, "_host_operand", traffic, "stage_upload", size=_arg_bytes(1)),
            wrap_traffic(cls, "_download", traffic, "download", size=_result_bytes),
            wrap_traffic(cls, "_export", traffic, "export"),
            wrap_traffic(cls, "_export", traffic, "sync"),
            wrap_traffic(cls, "_import", traffic, "import_state"),
            wrap_traffic(cls, "_ensure_state", traffic, "state_setup"),
            wrap_traffic(cls, "_error_mask", traffic, "host_query"),
            wrap_traffic(cls, "_read_state", traffic, "read_state"),
        ]
        # The inline policy / controls / advance graphs are timed through ``evoxc.run`` itself;
        # no other timer of this path tags them, so nothing double counts.
        import evoxc

        restores.append(wrap_method(evoxc, "run", timer, "graph_all"))
    return restores


