"""Smoke and parity CLI for the batched TensorNEAT drone hover evaluation environment.

Four things this script is for, all of them cheap enough for the example smoke suite:

* **Default mode** builds :class:`hover_neat_env.HoverNeatEnv`, scores the untouched population
  of the trace pipeline for ``--generations`` generations and prints the per-generation
  bookkeeping (seed triple, fitness, end reasons, steps, throughput, executable-cache
  hits/misses, graph backend@device).  It defaults to a small CPU/numpy run so
  ``tests/test_examples.py`` stays quick: no GPU, no viewer, and a horizon of a few steps.
* **``--parity``** replays a fixed action trace through ``hover_legacy_reference`` (the numpy
  oracle) and through the ``hover_kernels`` graphs, and prints every disagreement as a stable
  ``name<TAB>max_abs_diff`` line: the five ``dt``-scaled reward terms, the observation, the RPM
  affine map, the clipped actions, yaw in degrees, the termination/dead masks and the carried
  returns.  It then checks the analytic reset observation against the observation of the actual
  reset state of a live scene, the policy's policy-major row tiling, and prints a deterministic
  episode fingerprint (``fingerprint<TAB>...``) that two independent process invocations must
  reproduce byte for byte.
* **``--perturb``** re-runs either mode with a nonzero reset perturbation (three amplitudes,
  ``base_pos`` / ``base_rpy`` / ``joint_pos``; bare flag: ``0.05 0.05 0.0``): the same
  fingerprint determinism, the same numeric tolerances, and the analytic reset observation must
  still match the live reset state at a perturbed pose.
* **``--imports``** statically proves that the migrated stack stays free of ``torch``: an AST
  scan of ``hover_kernels.py``, ``hover_device_kernels.py``, ``hover_device_rollout.py``,
  ``hover_neat_env.py``, ``hover_neat_train.py`` (skipped with a note while a file does not exist
  yet) and the six modules of the shared ``examples/neuroevolution`` layer.  The device rollout is
  inside that list: it exports to Genesis through the torch-free DLPack bridge of
  ``genesis.utils.device_buffer``, not through ``torch.from_dlpack``.
* **``--residency``** selects the rollout residency of every env this script builds:
  ``numpy`` (the default, host-staged) or ``device`` (the zero-copy device-resident rollout of
  ``hover_device_rollout.py``, which needs a non-numpy graph backend, e.g.
  ``--evoxc-backend iree --evoxc-device cpu:0``).  It applies to the default mode and to
  ``--parity``.
* **``--torch-free``** is the Torch claim of the device residency, mirroring
  ``examples/locomotion/go2_neat_train.py --torch-free``: it implies ``--residency device``, defaults
  the graphs to ``iree@rocm:0`` and refuses the flags that would import Torch (a host rollout, a numpy
  graph backend).  It applies to every mode above and, at the end of the run, prints
  ``torch_free<TAB>1|0`` for the fact that ``"torch"`` never reached the process -- the exit code is 1
  when it did.  ``--imports`` is a static scan, so it proves its part of the claim without it.
* **``--residency-parity``** scores generation 0 twice -- once with a numpy-residency env, once
  with a device-residency env built from the same seed, population, horizon and perturbation --
  and prints the disagreement as ``name<TAB>value`` lines: ``returns_max_abs_diff``,
  ``fitness_max_abs_diff``, ``end_reasons_equal``, ``live_buffer_vs_getter`` (the live solver
  buffers against ``drone.get_pos/get_quat/get_vel/get_ang``, which must be exactly 0.0) and
  ``seeds_equal``, then ``residency_parity_ok<TAB>0|1``.  Both envs are closed before the next
  one is built.

The graph-vs-oracle part needs no simulator, so it prints before the scene is built.  The exit
code is 0 when every number is within its tolerance, 1 otherwise (the tolerances are the
targets of the migration: ~1e-5 per reward term, ~1e-6 on the observation, exact masks).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np

#: Rows and ``scene.step()`` calls of the fixed oracle replay (``rows * steps`` states compared).
PARITY_ROWS = 6
PARITY_STEPS = 200
#: Seed of the fixed action/state trace, so every invocation replays the same episode.
FIXED_TRACE_SEED = 20241109

#: Amplitudes the bare ``--perturb`` flag uses: ``base_pos`` (m), ``base_rpy`` (rad), ``joint_pos``.
DEFAULT_PERTURBATION = (0.05, 0.05, 0.0)

#: Largest accepted ``|graph - oracle|`` per reported quantity; masks and the repeated
#: in-process episode must agree exactly.
TOLERANCES = {
    "reward_target": 1e-5,
    "reward_smooth": 1e-5,
    "reward_yaw": 1e-5,
    "reward_angular": 1e-5,
    "reward_crash": 1e-5,
    "reward_total": 1e-5,
    "returns": 1e-4,
    "observation": 1e-6,
    "rpm": 1e-2,
    "clipped_actions": 1e-7,
    # The graph's ``atan2`` is a degree-15 float32 minimax polynomial and it composes the init
    # quaternion on the other side of the product than ``transform_quat_by_quat`` does (the init
    # quaternion is the identity, so this is pure float32 rounding): measured <= 1e-6 rad, i.e.
    # ~4 ulp at 180 degrees.  The yaw *reward* it feeds agrees to ~1e-10.
    "yaw_deg": 5e-4,
    "mask_crash": 0.0,
    "mask_timeout": 0.0,
    "mask_done": 0.0,
    "mask_at_target": 0.0,
    "mask_alive": 0.0,
    "reset_obs_vs_obs": 1e-6,
    "policy_row_identity": 1e-6,
    "fingerprint_repeat": 0.0,
}

#: Largest accepted disagreement between the two rollout residencies (``--residency-parity``);
#: the equality flags are exact (0/1) and the live-buffer read must be bit-identical.
RESIDENCY_TOLERANCES = {
    "returns_max_abs_diff": 1e-6,
    "fitness_max_abs_diff": 1e-6,
    "end_reasons_equal": 1,
    "live_buffer_vs_getter": 0.0,
    "seeds_equal": 1,
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pop-size", type=int, default=4, help="policies per generation (default 4)")
    parser.add_argument("--generations", type=int, default=1, help="generations to score (default 1)")
    parser.add_argument("--master-seed", type=int, default=0, help="seed of the run (default 0)")
    parser.add_argument("--horizon", type=int, default=20, help="scene.step() calls per generation (default 20)")
    parser.add_argument(
        "--episode-length-s", type=float, default=15.0, help="episode cap in seconds (default 15.0)"
    )
    parser.add_argument(
        "--evoxc-backend", default=None, help="evoxc backend of the graphs (default numpy, or iree with -g)"
    )
    parser.add_argument(
        "--evoxc-device", default=None, help="evoxc device of the graphs (default cpu:0, or rocm:0 with -g)"
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="initialize Genesis on the GPU and default the graphs to iree@rocm:0"
    )
    parser.add_argument("--parity", action="store_true", help="replay the fixed oracle trace instead of scoring")
    parser.add_argument(
        "--residency",
        choices=("numpy", "device"),
        default=None,
        help="rollout residency of every env built here: numpy (default) or device "
        "(the zero-copy rollout of hover_device_rollout.py, needs a non-numpy graph backend)",
    )
    parser.add_argument(
        "--torch-free",
        action="store_true",
        help="run with the Torch claim of the device residency: implies --residency device and "
        "iree graphs, and prints torch_free<TAB>0|1 for `\"torch\" in sys.modules` at the end",
    )
    parser.add_argument(
        "--residency-parity",
        action="store_true",
        help="score generation 0 with both residencies and compare the results",
    )
    parser.add_argument(
        "--perturb",
        nargs="*",
        type=float,
        default=None,
        metavar="AMPLITUDE",
        help="enable the reset perturbation, optionally as three amplitudes "
        "base_pos/base_rpy/joint_pos (bare flag: "
        f"{DEFAULT_PERTURBATION[0]} {DEFAULT_PERTURBATION[1]} {DEFAULT_PERTURBATION[2]})",
    )
    parser.add_argument(
        "--imports", action="store_true", help="AST-scan the migrated stack for `import torch` and exit"
    )
    parser.add_argument("--quiet", action="store_true", help="print only the summary lines")
    return parser.parse_args(argv)


def _resolve_torch_free(args):
    """Resolve ``--torch-free``: imply the device residency and refuse what would import Torch.

    The claim is the device residency's (no host array at a graph boundary), so it is the only
    residency this flag can run, and the graphs have to be a device backend.  Everything else is
    inherited from the flags, so a refusal always names the flag that has to go.
    """
    if not args.torch_free:
        if args.residency is None:
            args.residency = "numpy"
        return
    if args.residency == "numpy":
        raise SystemExit(
            "--torch-free runs the device-resident rollout, which stages no host array at a graph "
            "boundary, so --residency numpy is not available here; drop --residency numpy (device "
            "residency is implied) or drop --torch-free"
        )
    if args.evoxc_backend == "numpy":
        raise SystemExit(
            "--torch-free needs a device evoxc backend (device residency is IREE-backed), got "
            "--evoxc-backend numpy; drop --evoxc-backend numpy or drop --torch-free"
        )
    args.residency = "device"


def _torch_in_process():
    """Whether ``torch`` has been imported in this process (the torch-free claim, reported only)."""
    return "torch" in sys.modules


def _report_torch_free():
    """Print the ``--torch-free`` verdict; returns the exit-code contribution (0 ok, 1 imported)."""
    imported = _torch_in_process()
    print(f"torch_free\t{int(not imported)}")
    if imported:
        print(
            "# torch reached the process through an imported module; the torch-free paths (the device "
            "rollout of hover_device_rollout.py, the device kernels) must not import it"
        )
    return 1 if imported else 0


def _resolve_targets(args):
    """Resolve the graph backend/device: explicit flags win, then the GPU defaults, then CPU."""
    backend = args.evoxc_backend or ("iree" if (args.gpu or args.torch_free) else "numpy")
    device = args.evoxc_device or ("rocm:0" if args.gpu else "cpu:0")
    return backend, device


def _perturbation(args):
    """Return the ``seeding.Perturbation`` of ``--perturb``, or None for the fixed reset."""
    if args.perturb is None:
        return None
    from seeding import Perturbation  # ``hover_kernels`` puts the shared layer on ``sys.path``

    amplitudes = list(args.perturb) or list(DEFAULT_PERTURBATION)
    if len(amplitudes) != 3:
        raise SystemExit(f"--perturb takes three amplitudes (base_pos base_rpy joint_pos), got {len(amplitudes)}")
    return Perturbation(*amplitudes)


def _refuse_cpu_genesis_over_device_graphs(args):
    """Refuse device-resident graphs over a CPU Genesis.

    The device residency exports the graphs' output as a DLPack source on the graph's device and
    writes it straight into the scene, so a CPU Genesis refuses it deep inside the simulator
    (``Only a host DLPack source can be copied into a CPU buffer``); ``-g/--gpu`` puts Genesis on the
    graph's device and is the way out.  Every mode that builds a scene reaches that path --
    ``--residency device`` in the default scoring or ``--parity``, and ``--residency-parity``'s
    second half, which is always the device residency -- while the numpy residency is host-staged
    at both graph boundaries and therefore runs over a CPU Genesis in any mode.  ``--imports``
    builds neither graph nor scene, so it is exempt.
    """
    if args.imports or not (args.residency_parity or args.residency == "device"):
        return
    backend, device = _resolve_targets(args)
    kind = str(device).partition(":")[0].strip()
    if args.gpu or not kind or kind == "cpu":
        return
    mode = "--residency-parity" if args.residency_parity else "--residency device"
    raise SystemExit(
        f"{mode} with graphs on {device} ({backend}) needs Genesis on the same device, but -g/--gpu was "
        f"not given, so Genesis would stay on the CPU and refuse the device DLPack source the rollout "
        f'exports ("Only a host DLPack source can be copied into a CPU buffer"); add -g/--gpu to put '
        f"Genesis on the graphs' device, or drop --evoxc-device {device} to keep both on the CPU"
    )


def _print_rows(values):
    """Print ``name<TAB>max_abs_diff`` lines; return the names that exceed their tolerance."""
    for name, value in values.items():
        print(f"{name}\t{value:.3e}")
    return [name for name, value in values.items() if not value <= TOLERANCES[name]]


def _leaf(result, name):
    """One training leaf of an evaluation result as a host array.

    ``residency="numpy"`` hands the four training leaves back as numpy arrays, while
    ``residency="device"`` hands device tensors back.  A device tensor is materialized through its
    payload's explicit readback (``data.to_host()``, the same call the device rollout's ``_host``
    makes), because ``Tensor.numpy()`` needs host memory and raises ``DeviceError`` on a tensor that
    lives on a GPU (``iree@rocm:0``); the ``numpy``/``np.asarray`` fallbacks stay for a value that
    carries no such readback, so nothing else in the script has to know which residency produced
    the result.
    """
    value = result[name]
    if isinstance(value, np.ndarray):
        return value
    to_host = getattr(getattr(value, "data", None), "to_host", None)
    if callable(to_host):
        return np.asarray(to_host())
    numpy = getattr(value, "numpy", None)
    if callable(numpy) and not isinstance(value, (list, tuple)):
        return np.asarray(numpy())
    return np.asarray(value)


# --- Fixed-trace graph parity -------------------------------------------------------


def _plausible_states(rng, commands, rows, step):
    """Draw one plausible measured state per row, around the row's current commands.

    A fifth of the rows sit within the at-target radius (``norm(rel_pos) < 0.1``), a tenth fly
    far away (``|rel_pos_x| > 3``, a crash) and a twentieth of the steps report a low base (a
    crash); a twentieth of the rows also report an external solver error.  All the counts are
    properties of the fixed trace, so the same states are replayed every run.
    """
    offset = np.stack(
        [
            rng.uniform(-0.4, 0.4, rows),
            rng.uniform(-0.4, 0.4, rows),
            rng.uniform(-0.5, 0.5, rows),
        ],
        axis=-1,
    ).astype(np.float32)
    near_target = rng.random(rows) < 0.2
    if near_target.any():
        offset[near_target] = rng.uniform(-0.03, 0.03, (int(near_target.sum()), 3)).astype(np.float32)
    base_pos = (commands + offset).astype(np.float32)
    far = rng.random(rows) < 0.1
    base_pos[far, 0] += 4.0
    low = rng.random(rows) < 0.05
    base_pos[low, 2] = 0.02

    small_tilt = rng.random(rows) < 0.9
    roll = np.where(small_tilt, rng.uniform(-0.3, 0.3, rows), rng.uniform(-1.5, 1.5, rows))
    pitch = np.where(small_tilt, rng.uniform(-0.3, 0.3, rows), rng.uniform(-1.4, 1.4, rows))
    yaw = rng.uniform(-3.1, 3.1, rows)
    error_mask = rng.random(rows) < 0.02
    if step == 0:
        error_mask[rows // 2] = True  # one guaranteed solver-error row
    return {
        "base_pos": base_pos,
        "base_quat": _rpy_quat(roll, pitch, yaw),
        "base_lin_vel_world": rng.uniform(-1.0, 1.0, (rows, 3)).astype(np.float32),
        "base_ang_vel_world": rng.uniform(-2.0, 2.0, (rows, 3)).astype(np.float32),
        "error_mask": error_mask,
    }


def _rpy_quat(roll, pitch, yaw):
    """Unit quaternions ``(n, 4)`` (w-first) from intrinsic x-y-z Euler angles."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    ).astype(np.float32)


def _graph_parity(backend, device):
    """Replay a fixed trace through the oracle and the graphs; returns the max abs differences."""
    import hover_legacy_reference as ref

    from hover_kernels import (
        BASE_INIT_POS,
        DT,
        MAX_EPISODE_LENGTH,
        NUM_ACTIONS,
        NUM_COMMANDS,
        REWARD_SCALES,
        REWARD_TERM_NAMES,
        build_advance,
        build_controls,
        run_advance,
        run_controls,
    )

    rows, n_steps = PARITY_ROWS, PARITY_STEPS
    rng = np.random.default_rng(FIXED_TRACE_SEED)
    controls_executable = build_controls(rows, backend=backend, device=device)
    advance_executable = build_advance(rows, backend=backend, device=device)
    raw_actions = rng.uniform(-1.5, 1.5, (n_steps, rows, NUM_ACTIONS)).astype(np.float32)
    resampled = rng.uniform(-1.0, 1.0, (n_steps, rows, NUM_COMMANDS)).astype(np.float32)
    resampled[:, :, 2] = 1.0

    # Carried buffers, all driven by the oracle so both sides see identical inputs every step.
    commands = np.stack(
        [rng.uniform(-1.0, 1.0, rows), rng.uniform(-1.0, 1.0, rows), np.ones(rows)], axis=-1
    ).astype(np.float32)
    last_base_pos = np.ascontiguousarray(np.tile(BASE_INIT_POS, (rows, 1)))
    last_actions = np.zeros((rows, NUM_ACTIONS), dtype=np.float32)
    episode_length = np.zeros(rows, dtype=np.int32)
    episode_length[[0, 2, 4]] = MAX_EPISODE_LENGTH - 2  # three rows time out inside the trace
    returns = np.zeros(rows, dtype=np.float32)
    alive = np.ones(rows, dtype=bool)

    error = dict.fromkeys(
        (
            "reward_target",
            "reward_smooth",
            "reward_yaw",
            "reward_angular",
            "reward_crash",
            "reward_total",
            "returns",
            "observation",
            "rpm",
            "clipped_actions",
            "yaw_deg",
            "mask_crash",
            "mask_timeout",
            "mask_done",
            "mask_at_target",
            "mask_alive",
        ),
        0.0,
    )
    for step in range(n_steps):
        state = _plausible_states(rng, commands, rows, step)
        control = run_controls(controls_executable, raw_actions[step])
        actions = control["actions"]
        error["rpm"] = max(error["rpm"], float(np.abs(control["rpm"] - ref.propeller_rpm(raw_actions[step])).max()))
        error["clipped_actions"] = max(
            error["clipped_actions"], float(np.abs(actions - np.clip(raw_actions[step], -1.0, 1.0)).max())
        )

        reference = ref.step_maths(
            base_pos=state["base_pos"],
            last_base_pos=last_base_pos,
            base_quat=state["base_quat"],
            base_lin_vel_world=state["base_lin_vel_world"],
            base_ang_vel_world=state["base_ang_vel_world"],
            commands=commands,
            actions=actions,
            last_actions=last_actions,
            episode_length=episode_length,
            resampled_commands=resampled[step],
            error_mask=state["error_mask"],
        )
        stepped = run_advance(
            advance_executable,
            commands=commands,
            base_pos=state["base_pos"],
            last_base_pos=last_base_pos,
            base_quat=state["base_quat"],
            base_lin_vel_world=state["base_lin_vel_world"],
            base_ang_vel_world=state["base_ang_vel_world"],
            actions=actions,
            last_actions=last_actions,
            episode_length=episode_length,
            resampled_commands=resampled[step],
            error_mask=state["error_mask"],
            alive=alive,
            returns=returns,
        )

        # The oracle's terms are pre-scale; the graph's are ``dt``-scaled, so undoing the scale is
        # what makes them comparable.
        for name in REWARD_TERM_NAMES:
            expected = DT * REWARD_SCALES[name] * reference["reward_terms"][name]
            error[f"reward_{name}"] = max(
                error[f"reward_{name}"], float(np.abs(stepped["reward_terms"][name] - expected).max())
            )
        error["reward_total"] = max(
            error["reward_total"],
            float(np.abs(stepped["reward_terms"]["total"] - reference["reward_terms"]["total"]).max()),
        )
        error["observation"] = max(error["observation"], float(np.abs(stepped["obs"] - reference["obs"]).max()))
        error["yaw_deg"] = max(error["yaw_deg"], float(np.abs(stepped["yaw_deg"] - reference["yaw_deg"]).max()))
        for name in ("crash", "timeout", "done", "at_target"):
            error[f"mask_{name}"] = max(
                error[f"mask_{name}"],
                float(np.abs(stepped[name].astype(np.int8) - reference[name].astype(np.int8)).max()),
            )

        # The oracle's own bookkeeping, computed here so the graph's is checked, not mirrored.
        credit = alive & ~state["error_mask"]
        returns_next = returns + np.where(credit, reference["reward_terms"]["total"], 0.0).astype(np.float32)
        alive_next = alive & ~reference["done"]
        error["returns"] = max(error["returns"], float(np.abs(stepped["returns"] - returns_next).max()))
        error["mask_alive"] = max(
            error["mask_alive"], float(np.abs(stepped["alive"].astype(np.int8) - alive_next.astype(np.int8)).max())
        )

        # The next step reads the oracle chain, so a difference cannot compound into a mask flip.
        commands = reference["commands"]
        last_base_pos = state["base_pos"]
        last_actions = actions
        episode_length = episode_length + np.int32(1)
        returns = np.ascontiguousarray(returns_next, dtype=np.float32)
        alive = alive_next
    return error


# --- Scene parity: analytic reset observation, policy tiling, fingerprint ------------


def _scene_parity(args, backend, device):
    """Compare the reset observation with a live scene's, check the policy tiling, fingerprint."""
    from hover_kernels import NUM_ACTIONS, NUM_OBS, build_obs, run_obs
    from hover_neat_env import HoverNeatEnv
    from policy_graphs import run_policy

    env = HoverNeatEnv(
        args.pop_size,
        args.master_seed,
        horizon=args.horizon,
        episode_length_s=args.episode_length_s,
        backend=backend,
        device=device,
        perturbation=_perturbation(args),
        residency=args.residency,
    )
    try:
        # (a) The analytic reset observation vs the observation of the same reset state.
        commands, reset_obs = env.reset_generation(0)
        state = env.read_state()
        live_obs = run_obs(
            build_obs(env.rows, backend=backend, device=device),
            commands=commands,
            base_pos=state["base_pos"],
            base_quat=state["base_quat"],
            base_lin_vel_world=state["base_lin_vel_world"],
            base_ang_vel_world=state["base_ang_vel_world"],
            last_actions=np.zeros((env.rows, NUM_ACTIONS), dtype=np.float32),
        )["obs"]
        values = {"reset_obs_vs_obs": float(np.abs(reset_obs - live_obs).max())}

        # The three rows of one policy must share the genome statics but keep their own input row.
        pop_arrays = env.initial_pop_arrays()
        flat_obs = np.zeros((env.rows, NUM_OBS), dtype=np.float32)
        actions = run_policy(env.policy_executable, pop_arrays, flat_obs)
        triples = actions.reshape(env.pop_size, env.episodes_per_policy, NUM_ACTIONS)
        values["policy_row_identity"] = float(np.abs(triples - triples[:, :1]).max())

        # (c) One deterministic episode, twice in process: identical returns and a fingerprint.
        first = env.evaluate(pop_arrays, 0)
        second = env.evaluate(pop_arrays, 0)
        values["fingerprint_repeat"] = float(np.abs(_leaf(first, "returns") - _leaf(second, "returns")).max())
        fingerprint = first
    finally:
        env.close()
    return values, fingerprint


def _print_fingerprint(result):
    """Print the cross-process fingerprint: seed triple, per-row returns, their sum."""
    seeds = ",".join(str(int(seed)) for seed in _leaf(result, "seeds"))
    returns = ",".join(f"{value:.9f}" for value in _leaf(result, "returns"))
    total = f"{float(np.sum(_leaf(result, 'returns'))):.9f}"
    print(f"fingerprint\tseeds={seeds}\treturns={returns}\tsum={total}")


def run_parity(args):
    """Run every parity check; returns the process exit code."""
    backend, device = _resolve_targets(args)
    if not args.quiet:
        print(f"# parity: rows={PARITY_ROWS} steps={PARITY_STEPS} evoxc={backend}@{device}", flush=True)
    failed = _print_rows(_graph_parity(backend, device))

    if args.gpu:
        import genesis as gs

        gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    values, fingerprint = _scene_parity(args, backend, device)
    failed += _print_rows(values)
    _print_fingerprint(fingerprint)
    print(f"parity_ok\t{int(not failed)}")
    for name in failed:
        print(f"parity_failed\t{name}")
    return 0 if not failed else 1


# --- Residency parity: numpy staging vs the device-resident rollout -----------------


def _score_generation(args, backend, device, residency):
    """Score generation 0 in one residency; returns ``(result, live_buffer_vs_getter)``.

    The env owns the scene, so it is closed (and the scene destroyed) before the caller builds
    the next one; ``live_buffer_vs_getter`` is only measured for the device residency, where the
    live solver buffers are the rollout's own state source.
    """
    from hover_neat_env import HoverNeatEnv

    env = HoverNeatEnv(
        args.pop_size,
        args.master_seed,
        horizon=args.horizon,
        episode_length_s=args.episode_length_s,
        backend=backend,
        device=device,
        perturbation=_perturbation(args),
        residency=residency,
    )
    live = None
    try:
        pop_arrays = env.initial_pop_arrays()
        result = env.evaluate(pop_arrays, 0)
        if residency == "device":
            from hover_device_rollout import DeviceRollout

            rollout = DeviceRollout(env)
            try:
                live = rollout.state_parity()
            finally:
                rollout.close()
    finally:
        env.close()
    return result, live


def run_residency_parity(args):
    """Score one generation in both residencies and compare them; returns the exit code."""
    backend, device = _resolve_targets(args)
    if not args.quiet:
        print(f"# residency parity: generation 0, evoxc={backend}@{device}", flush=True)

    if args.gpu:
        import genesis as gs

        gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    staged_result, _ = _score_generation(args, backend, device, "numpy")
    device_result, live = _score_generation(args, backend, device, "device")

    values = {
        "returns_max_abs_diff": float(np.abs(_leaf(staged_result, "returns") - _leaf(device_result, "returns")).max()),
        "fitness_max_abs_diff": float(np.abs(_leaf(staged_result, "fitness") - _leaf(device_result, "fitness")).max()),
        "end_reasons_equal": int(staged_result["end_reasons"] == device_result["end_reasons"]),
        "live_buffer_vs_getter": float(live),
        "seeds_equal": int(np.array_equal(_leaf(staged_result, "seeds"), _leaf(device_result, "seeds"))),
    }
    failed = []
    for name, value in values.items():
        if name in ("end_reasons_equal", "seeds_equal"):
            print(f"{name}\t{int(value)}")
            failed += [] if int(value) == RESIDENCY_TOLERANCES[name] else [name]
        else:
            print(f"{name}\t{value:.3e}")
            failed += [] if value <= RESIDENCY_TOLERANCES[name] else [name]
    if not args.quiet:
        print(
            f"# steps numpy={staged_result['steps']} device={device_result['steps']} "
            f"end_reasons numpy={staged_result['end_reasons']} device={device_result['end_reasons']}"
        )
    print(f"residency_parity_ok\t{int(not failed)}")
    for name in failed:
        print(f"residency_parity_failed\t{name}")
    return 0 if not failed else 1


# --- Torch-free import scan ---------------------------------------------------------

#: ``examples/`` root, so the printed module labels stay relative to the example tree.
_EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Drone modules of the migrated stack; a file that does not exist yet is skipped with a note.
_HOVER_MODULES = (
    "hover_kernels.py",
    "hover_device_kernels.py",
    "hover_device_rollout.py",
    "hover_neat_env.py",
    "hover_neat_train.py",
)
#: The shared layer's modules, which are torch-free by contract (see its CONTEXT.md).
_SHARED_MODULES = (
    "seeding.py",
    "env_base.py",
    "evox_runtime.py",
    "evolution_graphs.py",
    "policy_graphs.py",
    "trainer.py",
)


def _torch_free_paths():
    """The ``(label, path)`` pairs the import scan covers, in report order."""
    paths = [(name, os.path.join(_EXAMPLES_DIR, "drone", name)) for name in _HOVER_MODULES]
    paths += [
        (os.path.join("neuroevolution", name), os.path.join(_EXAMPLES_DIR, "neuroevolution", name))
        for name in _SHARED_MODULES
    ]
    return paths


def _imports_torch(path):
    """Whether ``path`` imports torch, in any form (``import torch``, ``from torch.x import y``)."""
    with open(path, encoding="utf-8") as source:
        tree = ast.parse(source.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            # A relative import (``level > 0``) can never reach the installed torch package.
            if node.level == 0 and node.module and (node.module == "torch" or node.module.startswith("torch.")):
                return True
    return False


def run_imports():
    """Print one ``module<TAB>torch_free|torch_import`` line per file; returns the exit code."""
    failed = False
    for label, path in _torch_free_paths():
        if not os.path.exists(path):
            print(f"# skipped\t{label}\tdoes not exist yet")
            continue
        imported = _imports_torch(path)
        failed = failed or imported
        print(f"{label}\t{'torch_import' if imported else 'torch_free'}")
    print(f"imports_ok\t{int(not failed)}")
    return 0 if not failed else 1


# --- Default mode: score the trace population ---------------------------------------


def run_default(args):
    """Score the trace pipeline's population for ``--generations`` generations; returns the code."""
    backend, device = _resolve_targets(args)
    if args.gpu:
        import genesis as gs

        gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    from hover_neat_env import HoverNeatEnv

    env = HoverNeatEnv(
        args.pop_size,
        args.master_seed,
        horizon=args.horizon,
        episode_length_s=args.episode_length_s,
        backend=backend,
        device=device,
        perturbation=_perturbation(args),
        residency=args.residency,
    )
    try:
        # No trainer here: every generation scores the same untouched population, so the seed
        # triple is the only thing that changes between generations.
        pop_arrays = env.initial_pop_arrays()
        results = []
        for generation in range(args.generations):
            result = env.evaluate(pop_arrays, generation)
            results.append(result)
            if not args.quiet:
                seeds = ",".join(str(int(seed)) for seed in _leaf(result, "seeds"))
                fitness = ", ".join(f"{value:.6f}" for value in _leaf(result, "fitness"))
                print(
                    f"[gen {generation}] seeds=[{seeds}] fitness=[{fitness}] "
                    f"end_reasons={result['end_reasons']} steps={result['steps']} "
                    f"wall_time_s={result['wall_time_s']:.3f} steps/s={result['steps_per_s']:.1f} "
                    f"env-steps/s={result['env_steps_per_s']:.1f} cache={result['cache_hits']}/"
                    f"{result['cache_misses']} evoxc={backend}@{device}"
                )
    finally:
        env.close()

    env_steps = sum(result["env_steps"] for result in results)
    wall_time_s = sum(result["wall_time_s"] for result in results)
    best = max((float(np.max(_leaf(result, "fitness"))) for result in results), default=float("nan"))
    print(
        f"SUMMARY\tgenerations={len(results)}\tpop_size={args.pop_size}\trows={_leaf(results[0], 'returns').size} "
        f"backend={backend}\tdevice={device}\tenv_steps={env_steps} wall_time_s={wall_time_s:.3f} "
        f"steps_per_s={sum(result['steps'] for result in results) / wall_time_s:.1f} "
        f"env_steps_per_s={env_steps / wall_time_s:.1f} max_fitness={best:.6f}"
    )
    return 0


def main(argv=None):
    args = _parse_args(argv)
    _resolve_torch_free(args)
    _refuse_cpu_genesis_over_device_graphs(args)
    if args.imports:
        code = run_imports()
    elif args.residency_parity:
        code = run_residency_parity(args)
    else:
        code = run_parity(args) if args.parity else run_default(args)
    if args.torch_free:
        code = max(code, _report_torch_free())
    return code


if __name__ == "__main__":
    sys.exit(main())
