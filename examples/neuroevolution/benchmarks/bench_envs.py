"""Stage-3 benchmark CLI for the PyTorch -> EvoX Compiler migration: the ``replay`` subcommand.

The first phase of the harness answers one question per pilot: *does the migrated evaluation
environment, replayed on a fixed genome payload, still compute what the pre-migration maths
computed, and how fast is it?* ``replay`` builds a pilot environment with NumPy residency on
CPU, runs the same generation twice in one process (generation 0 is compile/cache warm-up
dominated, generation 1 is steady state), and records:

* **parity** -- every step's ``run_advance`` operands are captured by monkey-patching the
  read-only sibling modules (nothing on disk is touched; the patches are restored), then the same
  operands are replayed through the independent oracles of :mod:`oracle`. The record carries the
  per-field maximum absolute deviation and mismatch counts: the observation, every reward term and
  the total, the termination decision (``done`` / ``tilted`` / ``timeout`` or ``crash``), the
  accumulated episode returns and, for hover, the analytic reset observation;
* **timings** -- named accumulators (``physics``, ``graph_policy``, ``graph_obs``,
  ``graph_advance``, ``graph_controls``, ``graph_fitness``, ``state_read``, ``write_targets``,
  ``write_reset``, ``host_rng``) plus ``generation_total``, reported per generation and per step,
  with the unmeasured ``residual`` stated explicitly;
* **fingerprint** -- a sha256 over seeds, returns and fitness rounded to 6 decimals, so a second
  process running the same command must reproduce it exactly.

Usage (from the repository root)::

    PYTHONPATH=$PWD MPLCONFIGDIR=$TMPDIR/mpl /root/evox-robotics/.venv/bin/python \
        examples/neuroevolution/benchmarks/bench_envs.py replay --task go2 \
        --pop-size 8 --horizon 100 --master-seed 0 --repeat 0 --process-role only

The sibling paths are added by the CLI itself (see :func:`common.setup_paths`), so ``PYTHONPATH``
is only needed for the repository root import of ``common``/``oracle``.

Phase 2 (``throughput``, ``residency-parity`` and the batch sweep) lives in the companion
``bench_runtime.py`` CLI, which imports the instrumentation, the parity oracles and the record
helpers from here and from :mod:`common` instead of copying them; this module stays the ``replay``
measurement and the shared pilot-env plumbing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

import common
import oracle
from common import Timer, run_guarded, setup_paths, sync, wrap_function, wrap_method, write_record

#: Tolerance of the replayed observation against :func:`oracle.go2_observations`, per element.
OBS_ATOL = 1e-6
#: Tolerance of one reward term (hover exposes all five; Go2 exposes only the total).
REWARD_ATOL = 1e-5
#: Tolerance of the ``dt``-scaled reward total.
TOTAL_ATOL = 1e-5
#: Tolerance of the per-step ``returns`` increment (one reward, so the total tolerance applies).
RETURNS_ATOL = 1e-4
#: Tolerance of the fully accumulated returns (one generation of per-step error drifts/sums).
ACCUMULATED_RETURNS_ATOL = 1e-3
#: Tolerance of the hover yaw angle in degrees.
HOVER_YAW_ATOL = 1e-3

#: Generations per process: 0 is warm-up (compile + empty caches), 1 is steady state.
DEFAULT_GENERATIONS = 2

#: Steps whose operands and outputs are kept for the parity replay (memory/file-size guard).
MAX_CAPTURED_STEPS = 512

#: Every accumulator the harness fills, plus the group each wrapper feeds.
TIMER_GROUPS = (
    "physics",
    "graph_policy",
    "graph_obs",
    "graph_advance",
    "graph_controls",
    "graph_fitness",
    "state_read",
    "write_targets",
    "write_reset",
    "host_rng",
    "generation_total",
)

#: Human description of each accumulator, stored in the record so a reader knows what was timed.
TIMER_DESCRIPTIONS = {
    "physics": "genesis.Scene.step (class-level patch): rigid solver + state/material update",
    "graph_policy": "run_policy: genome payload x observation -> action",
    "graph_obs": "run_obs (Go2) / run_obs + run_reset_obs (hover): observation graph",
    "graph_advance": "run_advance: clip -> observation -> reward -> termination -> bookkeeping",
    "graph_controls": "run_controls (hover only): action -> clipped action + propeller RPM",
    "graph_fitness": "run_fitness: episode returns -> policy fitness",
    "state_read": "env.read_state: the device -> host boundary (torch reads, rpy, error mask)",
    "write_targets": "actuator write: env._write_targets (Go2) / drone.set_propellers_rpm (hover)",
    "write_reset": "env._write_reset (Go2) / exclusive share of reset_generation (hover)",
    "host_rng": "host draws: perturbed_init_qpos (Go2) / _draw_commands (hover)",
    "generation_total": "env.evaluate, wall clock, synchronised before and after",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --- small helpers ------------------------------------------------------------------


def _copy(value):
    """Deep-copy the numpy payload of a captured operand/output (dicts and lists included)."""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _r(value, digits: int = 12):
    """Round a float to ``digits`` significant digits for a compact, stable record."""
    if value is None:
        return None
    return float(f"{float(value):.{digits}g}")


def _max_abs_diff(reference, observed) -> float:
    """Return the maximum absolute element-wise difference of two arrays (0.0 when empty)."""
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if reference.shape != observed.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape} vs observed {observed.shape}")
    if reference.size == 0:
        return 0.0
    return float(np.max(np.abs(reference - observed)))


def _mismatches(expected, observed) -> int:
    """Return the number of element-wise disagreements of two boolean masks."""
    expected = np.asarray(expected, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    return int(np.count_nonzero(expected != observed))


def _fingerprint(result: dict) -> tuple[str, str]:
    """Return ``(sha256, source_text)`` over seeds, returns and fitness rounded to 6 decimals."""
    parts = []
    for key in ("seeds", "returns", "fitness"):
        values = np.round(np.asarray(result[key], dtype=np.float64), 6)
        parts.append(",".join(f"{value:.6f}" for value in values.reshape(-1)))
    text = "|".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), text


class StepCapture:
    """Collect the operands and outputs of every wrapped ``run_advance`` call.

    Used as the ``on_call`` hook of :func:`common.wrap_function`, i.e. it runs *outside* the timed
    region. Only ``MAX_CAPTURED_STEPS`` steps are kept: a parity replay of the first 512 steps of a
    generation is representative, and the cap keeps the process memory bounded for a longer horizon
    that a later phase may use.
    """

    def __init__(self, limit: int = MAX_CAPTURED_STEPS):
        self.limit = int(limit)
        self.steps = []
        self.dropped = 0

    def clear(self) -> None:
        """Forget the captured generation."""
        self.steps = []
        self.dropped = 0

    def __call__(self, args, kwargs, result) -> None:
        if len(self.steps) >= self.limit:
            self.dropped += 1
            return
        self.steps.append({"operands": _copy(kwargs), "outputs": _copy(result)})


class CallCapture:
    """Collect the keyword operands and one output field of a wrapped call (hover reset obs)."""

    def __init__(self, field: str = "obs"):
        self.field = field
        self.calls = []

    def clear(self) -> None:
        """Forget the captured generation."""
        self.calls = []

    def __call__(self, args, kwargs, result) -> None:
        self.calls.append({"operands": _copy(kwargs), "output": _copy(result[self.field])})


# --- task adapters ------------------------------------------------------------------


def _load_task(task: str):
    """Import the pilot environment module and return ``(module, env_cls, payload_getter)``."""
    setup_paths()
    if task == "go2":
        import go2_neat_env

        return go2_neat_env, go2_neat_env.Go2NeatEnv, lambda env: env._trace_pop
    if task == "hover":
        import hover_neat_env

        return hover_neat_env, hover_neat_env.HoverNeatEnv, lambda env: env.initial_pop_arrays()
    raise ValueError(f"unknown task {task!r}; expected 'go2' or 'hover'")


def _build_env(
    env_cls,
    pop_size: int,
    master_seed: int,
    horizon: int,
    *,
    residency: str = "numpy",
    backend: str = "numpy",
    device=None,
    stop_when_done: bool = False,
    perturbation=None,
):
    """Build one pilot environment in the reference configuration (NumPy residency, CPU).

    ``stop_when_done=False`` keeps every generation the same length, which is what makes the
    per-step timing means comparable between generations; the perturbation stays the default
    all-zero one, so the reset is the fixed reset of the pre-migration environment.

    The keyword arguments are the residency/latency overrides ``bench_runtime.py`` needs (device
    residency, an IREE graph target, a non-zero perturbation); every default reproduces the
    reference configuration of ``replay`` exactly, so the replay record is unchanged.
    """
    kwargs = {
        "horizon": int(horizon),
        "backend": backend,
        "device": device,
        "show_viewer": False,
        "stop_when_done": bool(stop_when_done),
        "residency": residency,
    }
    if perturbation is not None:
        kwargs["perturbation"] = perturbation
    return env_cls(int(pop_size), int(master_seed), **kwargs)


def _instrument(task: str, env, timer: Timer, steps, resets) -> list:
    """Monkey-patch the read-only sibling modules and ``genesis.Scene.step``; return restores.

    Every patch only wraps an existing callable (see :func:`common.wrap_function` /
    :func:`common.wrap_method`): the sibling files are never modified, and the returned restore
    callables put the originals back. Instrumentation runs after the environment is built, so the
    compile phase is not part of any accumulator (it is reported as ``phases.build_s``).

    Args:
        task (str): ``"go2"`` or ``"hover"``.
        env: The built environment (needed to reach the hover drone's actuator write).
        timer (Timer): Accumulators to fill.
        steps (StepCapture | None): Capture of the advance-graph operands/outputs; ``None`` times
            the generation without capturing anything (the ``throughput`` measurement, whose
            generations are long and whose record carries no per-step trace).
        resets (CallCapture | None): Capture of hover's analytic reset-observation graph; ``None``
            disables it for the same reason.

    Returns:
        list: Restore callables, to be called once the generations are done.
    """
    import genesis as gs

    module, env_cls, _payload = _load_task(task)
    restores = [wrap_method(gs.Scene, "step", timer, "physics")]
    if task == "go2":
        restores += [
            wrap_function(module, "run_policy", timer, "graph_policy"),
            wrap_function(module, "run_obs", timer, "graph_obs"),
            wrap_function(module, "run_advance", timer, "graph_advance", on_call=steps),
            wrap_function(module, "run_fitness", timer, "graph_fitness"),
            wrap_function(module, "perturbed_init_qpos", timer, "host_rng"),
            wrap_method(env_cls, "read_state", timer, "state_read"),
            wrap_method(env_cls, "_write_targets", timer, "write_targets"),
            wrap_method(env_cls, "_write_reset", timer, "write_reset"),
        ]
    else:
        restores += [
            wrap_function(module, "run_policy", timer, "graph_policy"),
            wrap_function(module, "run_controls", timer, "graph_controls"),
            wrap_function(module, "run_reset_obs", timer, "graph_obs", on_call=resets),
            wrap_function(module, "run_advance", timer, "graph_advance", on_call=steps),
            wrap_function(module, "run_fitness", timer, "graph_fitness"),
            wrap_method(env_cls, "read_state", timer, "state_read"),
            wrap_method(env_cls, "_draw_commands", timer, "host_rng"),
            # The whole reset (pose setters + analytic reset observation + first command draw);
            # ``exclusive`` removes the nested graph/host time, leaving the setter cost.
            wrap_method(env_cls, "reset_generation", timer, "write_reset", exclusive=True),
            # The actuator write of the hover loop. ``DroneEntity.set_propellers_rpm`` is the only
            # actuator write of the drone, and this process owns the one drone it patches.
            wrap_method(type(env.drone), "set_propellers_rpm", timer, "write_targets"),
        ]
    return restores


# --- parity: Go2 --------------------------------------------------------------------


def _go2_parity(steps: StepCapture, result: dict, env, module) -> dict:
    """Replay every captured Go2 step through :mod:`oracle` and summarise the deviations.

    ``go2_neat_kernels.run_advance`` returns only the aggregate ``reward`` (no per-term dict,
    unlike the hover graph), so the six terms are recorded as reference *values* -- their weighted
    sum is exactly the total that is compared -- and a per-term deviation cannot be reported.
    """
    max_episode_length = int(env.max_episode_length)
    rows = int(env.rows)
    obs_diff = 0.0
    obs_mismatch = 0
    total_diff = 0.0
    total_mismatch = 0
    done_mismatch = 0
    tilted_mismatch = 0
    timeout_mismatch = 0
    alive_mismatch = 0
    length_mismatch = 0
    returns_step_diff = 0.0
    returns_step_mismatch = 0
    term_values = {name: [] for name in (*oracle.GO2_REWARD_TERMS, "total")}
    accumulated = np.zeros(rows, dtype=np.float64)

    for step in steps.steps:
        operands = step["operands"]
        outputs = step["outputs"]
        obs_ref = oracle.go2_observations(
            operands["base_ang_vel_world"],
            operands["base_quat"],
            operands["dof_pos"],
            operands["dof_vel"],
            operands["actions"],
        )
        diff = np.abs(obs_ref - np.asarray(outputs["obs"], dtype=np.float32))
        obs_diff = max(obs_diff, float(diff.max()) if diff.size else 0.0)
        obs_mismatch += int(np.count_nonzero(diff > OBS_ATOL))

        clipped = np.clip(operands["actions"], -module.CLIP_ACTIONS, module.CLIP_ACTIONS).astype(np.float32)
        terms = oracle.go2_reward(
            operands["base_pos"],
            operands["base_quat"],
            operands["base_lin_vel_world"],
            operands["base_ang_vel_world"],
            operands["dof_pos"],
            clipped - operands["last_actions"],
        )
        for name in term_values:
            term_values[name].append(float(np.mean(np.abs(terms[name]))))
        reward_diff = np.abs(terms["total"] - np.asarray(outputs["reward"], dtype=np.float64))
        total_diff = max(total_diff, float(reward_diff.max()) if reward_diff.size else 0.0)
        total_mismatch += int(np.count_nonzero(reward_diff > TOTAL_ATOL))

        done_ref, tilted_ref, timeout_ref = oracle.go2_done(
            operands["base_quat"], operands["error_mask"], operands["episode_length"], max_episode_length
        )
        alive_pre = np.asarray(operands["alive"], dtype=bool)
        done_expected = done_ref & alive_pre
        done_mismatch += _mismatches(done_expected, outputs["done"])
        tilted_mismatch += int(np.count_nonzero(tilted_ref & alive_pre & ~np.asarray(outputs["done"], dtype=bool)))
        timeout_mismatch += int(np.count_nonzero(timeout_ref & alive_pre & ~np.asarray(outputs["done"], dtype=bool)))
        alive_mismatch += _mismatches(alive_pre & ~done_ref, outputs["alive"])
        # A row that ended is reset to counter 0 by the graph; everything else increments.
        length_expected = np.where(np.asarray(outputs["done"], dtype=bool), 0, operands["episode_length"] + 1)
        length_mismatch += _mismatches(length_expected, outputs["episode_length"])

        returns_expected = np.asarray(operands["returns"], dtype=np.float64) + terms["total"] * alive_pre
        diff = np.abs(returns_expected - np.asarray(outputs["returns"], dtype=np.float64))
        returns_step_diff = max(returns_step_diff, float(diff.max()) if diff.size else 0.0)
        returns_step_mismatch += int(np.count_nonzero(diff > RETURNS_ATOL))

        accumulated += terms["total"] * alive_pre

    returns_accumulated_diff = _max_abs_diff(accumulated, result["returns"])
    fitness_from_oracle = np.nan_to_num(accumulated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fitness_from_oracle = fitness_from_oracle.reshape(int(env.pop_size), int(env.episodes_per_policy))
    fitness_from_oracle = fitness_from_oracle.mean(axis=1, dtype=np.float32)
    fitness_diff = _max_abs_diff(fitness_from_oracle, result["fitness"])
    parity = {
        "n_steps_compared": len(steps.steps),
        "n_steps_dropped": int(steps.dropped),
        "rows": rows,
        "observation": {
            "max_abs_diff": _r(obs_diff),
            "mismatch_count": obs_mismatch,
            "atol": OBS_ATOL,
            "pass": bool(obs_diff <= OBS_ATOL and obs_mismatch == 0),
        },
        "reward": {
            "total_max_abs_diff": _r(total_diff),
            "total_mismatch_count": total_mismatch,
            "total_atol": TOTAL_ATOL,
            "per_term_available": False,
            "per_term_reason": (
                "go2_neat_kernels.run_advance returns only the aggregate reward; the six oracle "
                "terms are recorded as their mean absolute value, and their dt-scaled sum is the "
                "compared total"
            ),
            "term_mean_abs_values": {
                name: _r(float(np.mean(values))) if values else None for name, values in term_values.items()
            },
            "pass": bool(total_diff <= TOTAL_ATOL and total_mismatch == 0),
        },
        "termination": {
            "done_mismatch_count": done_mismatch,
            "tilted_mismatch_count": tilted_mismatch,
            "timeout_mismatch_count": timeout_mismatch,
            "alive_mismatch_count": alive_mismatch,
            "episode_length_mismatch_count": length_mismatch,
            "pass": bool(done_mismatch == 0 and tilted_mismatch == 0 and timeout_mismatch == 0 and alive_mismatch == 0),
        },
        "returns": {
            "step_max_abs_diff": _r(returns_step_diff),
            "step_mismatch_count": returns_step_mismatch,
            "step_atol": RETURNS_ATOL,
            "accumulated_max_abs_diff": _r(returns_accumulated_diff),
            "accumulated_atol": ACCUMULATED_RETURNS_ATOL,
            "fitness_from_oracle_max_abs_diff": _r(fitness_diff),
            "fitness_atol": ACCUMULATED_RETURNS_ATOL,
            "fitness_note": (
                "the oracle fitness is the float32 mean of the oracle-accumulated returns, so it "
                "inherits the accumulated returns drift (and only that); the env's own float32 vs "
                "float64 fitness rule is pinned by examples/neuroevolution/selfcheck.py"
            ),
            "pass": bool(
                returns_step_diff <= RETURNS_ATOL
                and returns_step_mismatch == 0
                and returns_accumulated_diff <= ACCUMULATED_RETURNS_ATOL
                and fitness_diff <= ACCUMULATED_RETURNS_ATOL
            ),
        },
    }
    return parity


# --- parity: hover ------------------------------------------------------------------


def _hover_parity(steps: StepCapture, resets: CallCapture, result: dict, env) -> dict:
    """Replay every captured hover step through ``hover_legacy_reference.step_maths``.

    Unlike the Go2 advance graph, the hover graph returns all five reward terms, its ``yaw_deg``
    and its next ``commands``, so every field can be compared element-wise; the analytic reset
    observation is compared separately, once per generation.
    """
    rows = int(env.rows)
    obs_diff = 0.0
    obs_mismatch = 0
    term_diff = {name: 0.0 for name in oracle.HOVER_REWARD_TERMS}
    term_mismatch = {name: 0 for name in oracle.HOVER_REWARD_TERMS}
    total_diff = 0.0
    total_mismatch = 0
    yaw_diff = 0.0
    mask_mismatch = {"crash": 0, "timeout": 0, "done": 0, "at_target": 0, "alive": 0}
    commands_diff = 0.0
    returns_step_diff = 0.0
    returns_step_mismatch = 0
    accumulated = np.zeros(rows, dtype=np.float64)

    for step in steps.steps:
        operands = step["operands"]
        outputs = step["outputs"]
        reference = oracle.hover_oracle_step(
            base_pos=operands["base_pos"],
            last_base_pos=operands["last_base_pos"],
            base_quat=operands["base_quat"],
            base_lin_vel_world=operands["base_lin_vel_world"],
            base_ang_vel_world=operands["base_ang_vel_world"],
            commands=operands["commands"],
            actions=operands["actions"],
            last_actions=operands["last_actions"],
            episode_length=operands["episode_length"],
            resampled_commands=operands["resampled_commands"],
            error_mask=operands["error_mask"],
        )
        diff = np.abs(np.asarray(reference["obs"], dtype=np.float32) - np.asarray(outputs["obs"], dtype=np.float32))
        obs_diff = max(obs_diff, float(diff.max()) if diff.size else 0.0)
        obs_mismatch += int(np.count_nonzero(diff > OBS_ATOL))

        # The legacy oracle reports the five terms pre-scale; the graph reports them already
        # multiplied by dt * scale, so the oracle is rescaled to the graph's convention here
        # (``total`` agrees on both sides and is passed through unchanged).
        scaled = oracle.hover_graph_terms(reference["reward_terms"])
        for name in oracle.HOVER_REWARD_TERMS:
            diff = np.abs(
                np.asarray(scaled[name], dtype=np.float64)
                - np.asarray(outputs["reward_terms"][name], dtype=np.float64)
            )
            term_diff[name] = max(term_diff[name], float(diff.max()) if diff.size else 0.0)
            term_mismatch[name] += int(np.count_nonzero(diff > REWARD_ATOL))
        diff = np.abs(
            np.asarray(scaled["total"], dtype=np.float64)
            - np.asarray(outputs["reward_terms"]["total"], dtype=np.float64)
        )
        total_diff = max(total_diff, float(diff.max()) if diff.size else 0.0)
        total_mismatch += int(np.count_nonzero(diff > TOTAL_ATOL))

        diff = np.abs(np.asarray(reference["yaw_deg"], dtype=np.float64) - np.asarray(outputs["yaw_deg"], dtype=np.float64))
        yaw_diff = max(yaw_diff, float(diff.max()) if diff.size else 0.0)

        for name in ("crash", "timeout", "done", "at_target"):
            mask_mismatch[name] += _mismatches(reference[name], outputs[name])
        alive_pre = np.asarray(operands["alive"], dtype=bool)
        mask_mismatch["alive"] += _mismatches(alive_pre & ~np.asarray(reference["done"], dtype=bool), outputs["alive"])
        commands_diff = max(commands_diff, _max_abs_diff(reference["commands"], outputs["commands"]))

        credit = alive_pre & ~np.asarray(operands["error_mask"], dtype=bool)
        returns_expected = np.asarray(operands["returns"], dtype=np.float64) + np.where(
            credit, np.asarray(reference["reward_terms"]["total"], dtype=np.float64), 0.0
        )
        diff = np.abs(returns_expected - np.asarray(outputs["returns"], dtype=np.float64))
        returns_step_diff = max(returns_step_diff, float(diff.max()) if diff.size else 0.0)
        returns_step_mismatch += int(np.count_nonzero(diff > RETURNS_ATOL))
        accumulated += np.where(credit, np.asarray(reference["reward_terms"]["total"], dtype=np.float64), 0.0)

    reset_diff = 0.0
    reset_mismatch = 0
    for call in resets.calls:
        diff = np.abs(
            np.asarray(oracle.hover_reset_obs(call["operands"]["commands"]), dtype=np.float32)
            - np.asarray(call["output"], dtype=np.float32)
        )
        reset_diff = max(reset_diff, float(diff.max()) if diff.size else 0.0)
        reset_mismatch += int(np.count_nonzero(diff > OBS_ATOL))

    returns_accumulated_diff = _max_abs_diff(accumulated, result["returns"])
    fitness_from_oracle = np.nan_to_num(accumulated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fitness_from_oracle = fitness_from_oracle.reshape(int(env.pop_size), int(env.episodes_per_policy))
    fitness_from_oracle = fitness_from_oracle.mean(axis=1, dtype=np.float32)
    fitness_diff = _max_abs_diff(fitness_from_oracle, result["fitness"])
    parity = {
        "n_steps_compared": len(steps.steps),
        "n_steps_dropped": int(steps.dropped),
        "rows": rows,
        "observation": {
            "max_abs_diff": _r(obs_diff),
            "mismatch_count": obs_mismatch,
            "atol": OBS_ATOL,
            "pass": bool(obs_diff <= OBS_ATOL and obs_mismatch == 0),
        },
        "reset_observation": {
            "n_compared": len(resets.calls),
            "max_abs_diff": _r(reset_diff),
            "mismatch_count": reset_mismatch,
            "atol": OBS_ATOL,
            "pass": bool(reset_diff <= OBS_ATOL and reset_mismatch == 0),
        },
        "reward": {
            "term_max_abs_diff": {name: _r(value) for name, value in term_diff.items()},
            "term_mismatch_count": term_mismatch,
            "term_atol": REWARD_ATOL,
            "total_max_abs_diff": _r(total_diff),
            "total_mismatch_count": total_mismatch,
            "total_atol": TOTAL_ATOL,
            "pass": bool(
                max(term_diff.values()) <= REWARD_ATOL
                and sum(term_mismatch.values()) == 0
                and total_diff <= TOTAL_ATOL
                and total_mismatch == 0
            ),
        },
        "termination": {
            "mask_mismatch_count": mask_mismatch,
            "pass": bool(all(count == 0 for count in mask_mismatch.values())),
        },
        "yaw_deg": {
            "max_abs_diff": _r(yaw_diff),
            "atol": HOVER_YAW_ATOL,
            "pass": bool(yaw_diff <= HOVER_YAW_ATOL),
        },
        "commands": {"max_abs_diff": _r(commands_diff), "pass": bool(commands_diff == 0.0)},
        "returns": {
            "step_max_abs_diff": _r(returns_step_diff),
            "step_mismatch_count": returns_step_mismatch,
            "step_atol": RETURNS_ATOL,
            "accumulated_max_abs_diff": _r(returns_accumulated_diff),
            "accumulated_atol": ACCUMULATED_RETURNS_ATOL,
            "fitness_from_oracle_max_abs_diff": _r(fitness_diff),
            "fitness_atol": ACCUMULATED_RETURNS_ATOL,
            "fitness_note": (
                "the oracle fitness is the float32 mean of the oracle-accumulated returns, so it "
                "inherits the accumulated returns drift (and only that); the env's own float32 vs "
                "float64 fitness rule is pinned by examples/neuroevolution/selfcheck.py"
            ),
            "pass": bool(
                returns_step_diff <= RETURNS_ATOL
                and returns_step_mismatch == 0
                and returns_accumulated_diff <= ACCUMULATED_RETURNS_ATOL
                and fitness_diff <= ACCUMULATED_RETURNS_ATOL
            ),
        },
    }
    return parity


# --- the replay measurement ---------------------------------------------------------


def _generation_entry(index: int, result: dict, delta: dict, parity: dict, capture: StepCapture) -> dict:
    """Assemble the per-generation record: throughput, cache, endings, timings and parity."""
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
    measured = sum(seconds for tag, (seconds, _c) in delta.items() if tag != "generation_total")
    fingerprint, source = _fingerprint(result)
    return {
        "generation": int(index),
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
        "episode_returns": [[_r(value) for value in row] for row in np.asarray(result["episode_returns"])],
        "returns": [_r(value) for value in np.asarray(result["returns"]).reshape(-1)],
        "captured_steps": len(capture.steps),
        "captured_steps_dropped": int(capture.dropped),
        "timings": {"groups": groups, "measured_s": _r(measured)},
        "residual_s": _r(total_seconds - measured),
        "residual_fraction": _r((total_seconds - measured) / total_seconds) if total_seconds else None,
        "fingerprint": {"sha256": fingerprint, "source": source},
        "parity": parity,
    }


def _merge_parity(parities: list) -> dict:
    """Reduce per-generation parity dicts to a worst-case aggregate.

    Numeric leaves (deviations, mismatch counts) take the worst value over the generations and a
    ``pass`` flag is ``True`` only when every generation passed, so the aggregate answers "did any
    generation disagree?" in one place.
    """

    def walk(items):
        first = items[0]
        if isinstance(first, dict):
            out = {}
            for key in first:
                if key.startswith("n_") and isinstance(first[key], int) and key != "n_steps_dropped":
                    out[key] = max(item[key] for item in items)
                else:
                    out[key] = walk([item[key] for item in items])
            if "pass" in out and not isinstance(out["pass"], dict):
                out["pass"] = all(item.get("pass", True) for item in items)
            return out
        if isinstance(first, bool):
            return all(items)
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            return max(items)
        return first

    return walk(list(parities))


def _replay(args) -> dict:
    """Run the replay measurement and return the finished record."""
    task = str(args.task)
    workload = {
        "task": task,
        "pop_size": int(args.pop_size),
        "horizon": int(args.horizon),
        "generations": int(args.generations),
        "residency": "numpy",
        "backend": "numpy",
        "device": "cpu",
        "stop_when_done": False,
        "perturbation": "zero (default seeding.Perturbation)",
        "payload": "env._trace_pop (go2) / env.initial_pop_arrays() (hover)",
        "timer_groups": TIMER_DESCRIPTIONS,
        "capture_limit": MAX_CAPTURED_STEPS,
        "tolerances": {
            "observation": OBS_ATOL,
            "reward_term": REWARD_ATOL,
            "reward_total": TOTAL_ATOL,
            "returns_step": RETURNS_ATOL,
            "returns_accumulated": ACCUMULATED_RETURNS_ATOL,
            "hover_yaw_deg": HOVER_YAW_ATOL,
            "masks": "exact (0 mismatches)",
        },
    }
    record = common.new_record(
        kind="env_replay",
        command=sys.argv,
        task=task,
        seed=args.master_seed,
        repeat=args.repeat,
        process_role=args.process_role,
        workload=workload,
    )
    out_path = Path(args.out) if args.out else _default_out_path(args)
    run_guarded(lambda rec: _run_replay(args, rec), record)
    write_record(out_path, record)
    _print_summary(record, out_path)
    return record


def _default_out_path(args) -> Path:
    """``results/env_<task>_replay_seed<seed>[_r<repeat>].json`` for an unspecified ``--out``."""
    suffix = "" if int(args.repeat) == 0 else f"_r{int(args.repeat)}"
    return RESULTS_DIR / f"env_{args.task}_replay_seed{int(args.master_seed)}{suffix}.json"


def _run_replay(args, record: dict) -> None:
    """The body of :func:`_replay`: build, instrument, run the generations, fill the record."""
    task = str(args.task)
    started = time.perf_counter()
    module, env_cls, payload_of = _load_task(task)
    timer = Timer(TIMER_GROUPS)
    steps = StepCapture()
    resets = CallCapture()

    build_start = time.perf_counter()
    env = _build_env(env_cls, args.pop_size, args.master_seed, args.horizon)
    build_s = time.perf_counter() - build_start
    payload = payload_of(env)
    record["workload"].update(
        {
            "rows": int(env.rows),
            "episodes_per_policy": int(env.episodes_per_policy),
            "max_episode_length": int(env.max_episode_length),
            "device": f"{env.device.kind}:{env.device.index}",
        }
    )

    restores = _instrument(task, env, timer, steps, resets)
    synchronized = False
    entries = []
    parities = []
    try:
        for generation in range(int(args.generations)):
            steps.clear()
            resets.clear()
            sync()
            before = timer.snapshot()
            start = time.perf_counter()
            result = env.evaluate(payload, generation)
            elapsed = time.perf_counter() - start
            synchronized = sync() or synchronized
            timer.add("generation_total", elapsed)
            after = timer.snapshot()
            if task == "go2":
                parity = _go2_parity(steps, result, env, module)
            else:
                parity = _hover_parity(steps, resets, result, env)
            entries.append(_generation_entry(generation, result, timer.delta(before, after), parity, steps))
            parities.append(parity)
    finally:
        for restore in restores:
            restore()
        close_start = time.perf_counter()
        env.close()
        close_s = time.perf_counter() - close_start

    record["phases"] = {
        "build_s": _r(build_s),
        "generations_s": [_r(entry["wall_time_s"]) for entry in entries],
        "close_s": _r(close_s),
        "total_s": _r(time.perf_counter() - started),
        "synchronized_before_and_after_each_generation": bool(synchronized),
        "generation_total_scope": (
            "generation_total is env.evaluate end to end (host seed draw + reset + steps + "
            "run_fitness); measurements.generations[*].wall_time_s is the environment's own "
            "counter, which starts after the host seed draw and stops before run_fitness"
        ),
        "sync_note": (
            "common.sync() calls torch.cuda.synchronize() whenever a HIP/CUDA device is available "
            f"(it reported {synchronized} here) and is a no-op otherwise; this benchmark runs "
            "entirely on the CPU, so every timed region is host wall clock (time.perf_counter) and "
            "the synchronisation only proves no accelerator work was pending around each generation"
        ),
    }
    run_fingerprint = hashlib.sha256(
        "|".join(entry["fingerprint"]["sha256"] for entry in entries).encode("utf-8")
    ).hexdigest()
    record["timings"] = {
        "groups": timer.as_dict(),
        "note": "per-generation and per-step breakdowns live in measurements.generations[*].timings",
    }
    record["measurements"] = {
        "task": task,
        "generations": entries,
        "parity": _merge_parity(parities),
        "fingerprint": {
            "sha256": run_fingerprint,
            "per_generation": [entry["fingerprint"]["sha256"] for entry in entries],
            "source": "|".join(entry["fingerprint"]["source"] for entry in entries),
            "rule": "sha256 over seeds + returns + fitness rounded to 6 decimals, per generation, then over the generation digests",
        },
    }


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
    parity = record["measurements"]["parity"]
    obs = parity["observation"]
    print(
        f"[bench] obs max|d|={obs['max_abs_diff']:.3e} (tol {obs['atol']:.0e}, "
        f"{obs['mismatch_count']} mismatches) pass={obs['pass']}"
    )
    reward = parity["reward"]
    if "term_max_abs_diff" in reward:
        terms = ", ".join(f"{name}={value:.3e}" for name, value in reward["term_max_abs_diff"].items())
        print(f"[bench] reward terms: {terms} (tol {reward['term_atol']:.0e})")
    print(f"[bench] reward total max|d|={reward['total_max_abs_diff']:.3e} pass={reward['pass']}")
    termination = parity["termination"]
    if "mask_mismatch_count" in termination:
        print(f"[bench] masks: {termination['mask_mismatch_count']} pass={termination['pass']}")
    else:
        print(
            f"[bench] done mismatches: {termination['done_mismatch_count']} "
            f"(tilted {termination['tilted_mismatch_count']}, timeout {termination['timeout_mismatch_count']}, "
            f"alive {termination['alive_mismatch_count']}) pass={termination['pass']}"
        )
    returns = parity["returns"]
    print(
        f"[bench] returns: step max|d|={returns['step_max_abs_diff']:.3e} "
        f"accumulated max|d|={returns['accumulated_max_abs_diff']:.3e} pass={returns['pass']}"
    )
    if "reset_observation" in parity:
        reset = parity["reset_observation"]
        print(f"[bench] reset obs max|d|={reset['max_abs_diff']:.3e} n={reset['n_compared']} pass={reset['pass']}")
    for entry in record["measurements"]["generations"]:
        residual = entry["residual_fraction"]
        share = "n/a" if residual is None else f"{100.0 * residual:.1f}%"
        print(
            f"[bench] gen {entry['generation']}: steps={entry['steps']} "
            f"{entry['steps_per_s']:.2f} steps/s {entry['env_steps_per_s']:.1f} env-steps/s "
            f"wall={entry['wall_time_s']:.3f}s cache={entry['cache_hits']}h/{entry['cache_misses']}m "
            f"ends={entry['end_reasons']} residual={share}"
        )
        groups = entry["timings"]["groups"]
        total = groups.get("generation_total", {}).get("seconds") or 0.0
        split = " ".join(
            f"{tag}={100.0 * groups[tag]['seconds'] / total:.0f}%"
            for tag in ("physics", "graph_policy", "graph_advance", "state_read", "write_targets")
            if tag in groups and total
        )
        print(f"[bench] gen {entry['generation']} split: {split}")
    print(f"[bench] fingerprint={record['measurements']['fingerprint']['sha256']}")


# --- CLI ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the benchmark CLI (``replay`` implemented, others planned)."""
    parser = argparse.ArgumentParser(
        prog="bench_envs.py",
        description="Stage-3 neuroevolution benchmark harness (PyTorch -> EvoX Compiler migration).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser(
        "replay",
        help="replay one fixed genome payload through a pilot env and check per-step parity",
        description=(
            "Build a pilot environment (NumPy residency, CPU), instrument its rollout by runtime "
            "monkey-patching, run the same generation twice and record parity, timing and a "
            "determinism fingerprint."
        ),
    )
    replay.add_argument("--task", required=True, choices=("go2", "hover"), help="pilot task to replay")
    replay.add_argument("--pop-size", type=int, default=8, help="policies per generation (rows = 3 x this)")
    replay.add_argument("--horizon", type=int, default=100, help="scene.step() calls per generation")
    replay.add_argument("--master-seed", type=int, default=0, help="run seed (episode seeds derive from it)")
    replay.add_argument("--repeat", type=int, default=0, help="replica index; nonzero proves cross-process determinism")
    replay.add_argument(
        "--process-role",
        default="only",
        choices=("cold", "warm", "only"),
        help="where this process sits in the warm-up chain (cold/warm compile cache)",
    )
    replay.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="generations per process")
    replay.add_argument("--out", default=None, help="record path (default: results/env_<task>_replay_seed<seed>[_r<repeat>].json)")
    # Planned subcommand of a later phase (not implemented here):
    #   batch -- pop-size / horizon sweep on one host
    # ``throughput`` and ``residency-parity`` are implemented by ``bench_runtime.py``, which reuses
    # ``_load_task`` / ``_build_env`` / ``_instrument`` / ``_generation_entry`` from this module.
    return parser


def main(argv=None) -> int:
    """Parse ``argv`` (``sys.argv[1:]`` by default) and run the requested subcommand.

    A bare invocation prints the usage and exits 0: ``tests/test_examples.py`` runs every
    ``examples/**/*.py`` as a subprocess, and a command-line tool without a subcommand is not a
    failing example. Anything that does name a subcommand is parsed strictly (exit 2 on misuse).
    """
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command == "replay":
        record = _replay(args)
        return 0 if record["status"] == "ok" else 1
    raise SystemExit(f"unsupported subcommand {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
