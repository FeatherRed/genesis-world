"""TensorNEAT training entry point of the drone *hover* task on top of the shared neuroevolution layer.

This is the CLI of the TensorNEAT/``evoxc`` hover stack.  Genesis owns the scene and the physics,
the compiled graphs of ``hover_kernels.py`` own the task math (controls, observation, reward,
termination), ``hover_neat_env.HoverNeatEnv`` is the batched seeded rollout, and
``examples/neuroevolution/trainer.NeuroevolutionTrainer`` owns the ask -> evaluate -> fitness ->
tell loop, the seed streams and the artifacts.  This module is only the wiring: CLI, Genesis
initialisation, the task-specific summary fields and the exit path.

What is where
-------------
``examples/drone/hover_env.py``      the behaviour being reproduced (env semantics); the PPO
                                    baseline, never imported here (it raises without ``rsl-rl-lib``).
``examples/drone/hover_neat_env.py`` ``HoverNeatEnv``: one scene, ``3 * pop_size`` policy-major rows,
                                    one episode per row per generation.
``examples/drone/hover_kernels.py``  the task graphs and the genome widths (17 -> 4).
``examples/neuroevolution/``         the trainer, the env contract, the seeding rules, the fitness /
                                    policy / tell graphs and the executable cache.

Seeds -- two independent streams from one master seed
----------------------------------------------------
``np.random.SeedSequence(master_seed).spawn(2)``: the **evolution** stream
(:func:`seeding.evolution_seed`) seeds the TensorNEAT pipeline, the **environment** stream
(:func:`seeding.generation_seed_table`) gives one ordered triple of episode seeds per generation.
Every policy of a generation is scored on the same triple, and row ``3p + e`` draws its per-step
command stream from the episode seed of slot ``e``, so two runs of the same configuration are
identical.  This script adds no RNG of its own.

Residencies
-----------
``--residency numpy`` (default) is the correctness-first host-staged rollout: every Genesis read and
every actuator write crosses the device boundary once, as a numpy array, and the graphs run on
``--evoxc-backend`` / ``--evoxc-device``.
``--residency device`` selects the opt-in zero-copy device-resident rollout, which lives in the
companion ``hover_device_rollout.py`` and is imported lazily by ``HoverNeatEnv.evaluate``; it needs
a device ``evoxc`` backend (``iree``/``xla``) and is not implemented in this file.

``--torch-free`` is the device residency with the Torch claim enforced: it refuses any configuration
that would import Torch (a viewer, a host rollout, an XLA ``tell``) and, because device residency is
single-backend, leaves ``tell()`` on the environment's own backend/device (``iree@rocm:0``) instead of
forwarding ``--tell-backend`` / ``--pjrt-plugin``.  It mirrors
``examples/locomotion/go2_neat_train.py --torch-free`` flag for flag; the guard after the scene build,
after the evolution-graph build and after the last generation proves the claim instead of reporting it.

Artifacts (git-ignored, never committed)
----------------------------------------
``<output-dir>/history.jsonl``  one JSON object per generation (seeds, returns, fitness, end reasons).
``<output-dir>/summary.json``   the trainer's resolved config plus every generation record and the
                                totals, with the drone's task fields (horizon, episode length,
                                perturbation, genesis/evoxc/tell targets, residency) added.
``-o`` defaults to ``out/hover_neat``; ``-e/--exp-name`` appends a subdirectory.

HSA runtime
-----------
Unlike ``examples/locomotion/go2_neat_train.py`` this script carries **no** ``LD_PRELOAD`` re-exec
helper: Genesis repairs the torch-bundled ROCm HSA runtime itself inside ``gs.init``
(``genesis.utils.hsa.ensure_system_hsa_runtime``), so the GPU path starts once.  The only leftover
of that recipe is the ``os._exit(0)`` guard at the end, which keeps a live PJRT client from aborting
at interpreter exit on ROCm after the artifacts are already on disk.

Under ``PYTEST_VERSION`` the run shrinks to one generation of 20 steps unless the matching flags are
given explicitly, so the example smoke test stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

import genesis as gs

# Importing the env also puts the shared ``examples/neuroevolution`` layer on ``sys.path``.
from hover_neat_env import MAX_CONNS, MAX_NODES, SPECIES_SIZE, HoverNeatEnv  # noqa: E402
from hover_kernels import NUM_ACTIONS, NUM_OBS  # noqa: E402
from seeding import Perturbation, evolution_seed  # noqa: E402
from trainer import NeuroevolutionTrainer  # noqa: E402  (path set up by hover_neat_env)

logger = logging.getLogger(__name__)


# --- CLI ----------------------------------------------------------------------------


def _parse_args(argv=None):
    """Parse the command line; every default that pytest shrinks keeps a ``None`` sentinel."""
    parser = argparse.ArgumentParser(
        description="Train a TensorNEAT drone hover population with Genesis and EvoX Compiler graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  python examples/drone/hover_neat_train.py\n"
            "  python examples/drone/hover_neat_train.py --pop-size 8 --generations 2\n"
            "  python examples/drone/hover_neat_train.py --pop-size 8 --generations 2 -g\n"
            "  python examples/drone/hover_neat_train.py -g --evoxc-device rocm:0 -v  # AMD recipe\n"
            "  python examples/drone/hover_neat_train.py --torch-free --pop-size 8 --generations 2\n"
        ),
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Policies per generation (rows = 3 * pop_size); default 8 (2 under PYTEST_VERSION).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Generations to run; default 2 (1 under PYTEST_VERSION).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "scene.step() calls per generation; default the full episode (episode_length_s / dt), "
            "20 under PYTEST_VERSION."
        ),
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=15.0,
        help="Episode cap in seconds; it fixes max_episode_length = ceil(episode_length_s / 0.01).",
    )
    parser.add_argument("--master-seed", type=int, default=0, help="Seed of both RNG streams.")
    parser.add_argument("--perturb-base-pos", type=float, default=0.05, help="Max |base position| offset [m].")
    parser.add_argument(
        "--perturb-base-rpy", type=float, default=0.05, help="Max |roll/pitch/yaw| reset offset [rad]."
    )
    parser.add_argument(
        "--perturb-joint-pos",
        type=float,
        default=0.0,
        help="Max |joint| reset offset [rad]; inert here, the drone is a body with no joints.",
    )
    parser.add_argument(
        "--no-perturbation",
        action="store_true",
        help="Zero every reset offset, reproducing the fixed reset of hover_env.",
    )
    parser.add_argument("-e", "--exp-name", type=str, default=None, help="Experiment name, a subdirectory of -o.")
    parser.add_argument("-o", "--output-dir", type=str, default="out/hover_neat", help="Directory of the artifacts.")
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Init Genesis on gs.gpu and default the EvoX Compiler graphs to IREE on rocm:0.",
    )
    parser.add_argument("--evoxc-backend", type=str, default=None, help="Rollout backend override (numpy or iree).")
    parser.add_argument(
        "--evoxc-device", type=str, default=None, help="Rollout device override, e.g. cpu:0, cuda:0 or rocm:0."
    )
    parser.add_argument(
        "--tell-backend",
        choices=("numpy", "iree", "xla"),
        default=None,
        help=(
            "Evolution (tell) backend override. Defaults to the rollout backend and device, except on "
            "ROCm where it defaults to xla because IREE/ROCm tell compilation does not terminate. "
            "Under --torch-free evolution always runs on the rollout backend/device, so an explicit "
            "numpy or xla is rejected."
        ),
    )
    parser.add_argument(
        "--pjrt-plugin",
        type=str,
        default=None,
        help=(
            "Path to the XLA/PJRT plugin .so used by the xla tell backend; also exported as "
            "EVOXC_PJRT_PLUGIN (when that variable is unset) so device staging uses it. When omitted "
            "evoxc's own discovery applies."
        ),
    )
    parser.add_argument(
        "--residency",
        choices=("numpy", "device"),
        default=argparse.SUPPRESS,
        help=(
            "Rollout residency: numpy = host staging (the default), device = the zero-copy device "
            "rollout of hover_device_rollout.py, which needs a device evoxc backend. --torch-free "
            "implies device, so an explicit numpy is refused there."
        ),
    )
    parser.add_argument(
        "--torch-free",
        action="store_true",
        help=(
            "Run the whole training loop without Torch: implies --residency device and defaults the "
            "rollout graphs to IREE on rocm:0 (CUDA: cuda:0), with evolution on the same "
            "backend/device. Refuses a configuration that would import torch."
        ),
    )
    parser.add_argument(
        "--stop-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a generation as soon as every row has ended (disable to force the full horizon).",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Open the Genesis viewer.")
    parser.add_argument("--quiet", action="store_true", help="Write the artifacts without the header/summary lines.")
    return parser.parse_args(argv)


def _torch_in_process():
    """Whether ``torch`` has been imported in this process (the torch-free claim, reported only)."""
    return "torch" in sys.modules


def _refuse_torch_import(torch_free, stage):
    """Fail a ``--torch-free`` run as soon as Torch reaches the process.

    A configuration that imports Torch is not the configuration the flag promises (the
    device-resident loop creates no Torch tensor itself, so an import means an imported module
    pulled it in), so it is refused here instead of being reported after the work is done.
    """
    if torch_free and _torch_in_process():
        raise RuntimeError(
            f"--torch-free was requested, but 'torch' reached the process {stage}; the device-resident "
            "loop never imports it, so an imported module did -- refusing to continue with a claim that "
            "is no longer true (report the module that pulled it in, or drop --torch-free)"
        )


# --- Artifacts ----------------------------------------------------------------------


def _to_json(value):
    """Fallback JSON encoder for the numpy scalars the trainer may leave in the summary."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_summary(path, summary):
    """Rewrite ``summary.json`` with the task-specific config fields added to the trainer's own."""
    path.write_text(json.dumps(summary, indent=2, default=_to_json) + "\n", encoding="utf-8")


def _task_config(args, env, perturbation, show_viewer, torch_free):
    """Return the drone-specific ``summary.json`` config fields the shared trainer does not know."""
    return {
        "horizon": int(env.horizon),
        "episode_length_s": float(args.episode_length_s),
        "perturbation": {
            "base_pos": float(perturbation.base_pos),
            "base_rpy": float(perturbation.base_rpy),
            "joint_pos": float(perturbation.joint_pos),
            "is_zero": bool(perturbation.is_zero),
        },
        "genesis_backend": "gpu" if (args.gpu or torch_free) else "cpu",
        "evoxc_backend": env.backend,
        "evoxc_device": env.device_name,
        "residency": env.residency,
        "genome_num_inputs": NUM_OBS,
        "genome_num_outputs": NUM_ACTIONS,
        "show_viewer": bool(show_viewer),
        "stop_when_done": bool(args.stop_when_done),
        "torch_free": bool(torch_free),
        "torch_in_process": _torch_in_process(),
    }


def _perturbation_label(perturbation):
    """One header token describing the reset perturbation in force."""
    if perturbation.is_zero:
        return "zero"
    return (
        f"base_pos {perturbation.base_pos:.3g} base_rpy {perturbation.base_rpy:.3g} "
        f"joint_pos {perturbation.joint_pos:.3g} (inert)"
    )


def _format_header(args, env, trainer, pop_size, generations, perturbation, show_viewer, torch_free):
    """One line with every resolved choice of the run: sizes, targets, seed and perturbation."""
    return (
        f"Hover NEAT: pop_size={pop_size} rows={env.rows} generations={generations} "
        f"horizon={env.horizon} max_episode_length={env.max_episode_length} "
        f"genesis={'gpu' if (args.gpu or torch_free) else 'cpu'} evoxc={env.backend}@{env.device_name} "
        f"tell={trainer.tell_backend}@{trainer.tell_device_name} "
        f"master_seed={trainer.master_seed} evolution_seed={trainer.evolution_seed} "
        f"perturbation={_perturbation_label(perturbation)} "
        f"residency={env.residency} show_viewer={show_viewer} "
        f"torch_free={torch_free} torch_in_process={_torch_in_process()}"
    )


# --- Main ---------------------------------------------------------------------------


def main(argv=None):
    """Run the configured number of generations and write the artifacts; returns the exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pytest_mode = "PYTEST_VERSION" in os.environ
    pop_size = args.pop_size if args.pop_size is not None else (2 if pytest_mode else 8)
    generations = args.generations if args.generations is not None else (1 if pytest_mode else 2)
    horizon = args.horizon if args.horizon is not None else (20 if pytest_mode else None)
    if pop_size < 1:
        raise ValueError(f"--pop-size must be positive, got {pop_size}")
    if generations < 1:
        raise ValueError(f"--generations must be positive, got {generations}")
    if horizon is not None and horizon < 1:
        raise ValueError(f"--horizon must be positive, got {horizon}")
    if args.episode_length_s <= 0.0:
        raise ValueError(f"--episode-length-s must be positive, got {args.episode_length_s}")

    perturbation = (
        Perturbation()
        if args.no_perturbation
        else Perturbation(
            base_pos=args.perturb_base_pos,
            base_rpy=args.perturb_base_rpy,
            joint_pos=args.perturb_joint_pos,
        )
    )

    torch_free = bool(args.torch_free)
    # ``None`` means --residency was not given at all, so the default (numpy) cannot be confused
    # with an explicit request for it, which is what --torch-free has to refuse.
    residency = getattr(args, "residency", None)
    if torch_free and residency == "numpy":
        # Device residency is the only mode in which no host/Torch tensor is created at a graph
        # boundary, so it is the only one a Torch-free process can run.
        raise ValueError(
            "--torch-free runs the device-resident rollout, which stages no host array at a graph "
            f"boundary, so --residency {residency!r} is not available here; drop --residency "
            "numpy (device residency is implied) or drop --torch-free"
        )
    args.residency = "device" if torch_free else (residency or "numpy")
    backend = args.evoxc_backend or ("iree" if (args.gpu or torch_free) else "numpy")
    if args.evoxc_device:
        device = args.evoxc_device
    elif (args.gpu or torch_free) and backend != "numpy":
        device = "rocm:0"  # the documented GPU target of this stack
    else:
        device = None
    if torch_free and backend == "numpy":
        raise ValueError(
            "--torch-free needs a non-CPU rollout backend (device residency is IREE-backed), got "
            f"--evoxc-backend {backend!r}; drop --evoxc-backend numpy or drop --torch-free"
        )
    if torch_free and args.vis:
        raise ValueError(
            "--torch-free runs without Torch, but the built-in viewer is a Torch consumer; drop "
            "-v/--vis or drop --torch-free"
        )
    if torch_free and args.tell_backend == "xla":
        raise ValueError(
            "--torch-free keeps the rollout and the evolution on one backend/device, and an XLA "
            "payload cannot enter the IREE executables that run the rollout (there is no "
            "device-to-device transfer between the two runtimes), so --tell-backend xla is not "
            "available here; drop --tell-backend xla, evolution then runs on the env's own "
            "backend/device (iree on the rollout device), or drop --torch-free"
        )
    if torch_free and args.tell_backend == "numpy":
        raise ValueError(
            "device residency runs every graph on one backend/device, so --torch-free cannot use "
            f"--tell-backend {args.tell_backend!r}; evolution runs on the rollout backend/device "
            "(iree on the rollout device), or drop --torch-free"
        )
    if torch_free and args.pjrt_plugin:
        raise ValueError(
            "--pjrt-plugin is an XLA compile option and --torch-free never compiles an XLA graph "
            "(evolution runs on the IREE rollout backend); drop --pjrt-plugin or drop --torch-free"
        )
    # ``--torch-free`` is single-backend: ``tell_backend=None`` resolves to the env's own
    # backend/device inside ``NeuroevolutionTrainer`` (iree on the rollout device), the trainer
    # rejects any other pair, and no PJRT plugin is forwarded.
    tell_backend = None if torch_free else args.tell_backend
    plugin_path = None if torch_free else args.pjrt_plugin
    show_viewer = bool(args.vis) and not pytest_mode

    if plugin_path:
        # ``Tensor.to`` passes no options, so the option-less PJRT discovery used for device
        # staging must see the same plugin as the xla tell build.
        os.environ.setdefault("EVOXC_PJRT_PLUGIN", str(plugin_path))

    if not gs._initialized:
        if args.gpu or torch_free:
            # ``performance_mode=True`` hands the state to Quadrants ndarrays whose staged writes the
            # torch-free setters cannot consume yet (and its build fails torch-free), so the
            # torch-free path keeps the array mode off.
            gs.init(
                backend=gs.gpu,
                precision="32",
                logging_level="warning",
                performance_mode=not torch_free,
            )
        else:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    output_dir = Path(args.output_dir)
    if args.exp_name:
        output_dir = output_dir / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env = HoverNeatEnv(
        pop_size,
        args.master_seed,
        horizon=horizon,
        episode_length_s=args.episode_length_s,
        backend=backend,
        device=device,
        show_viewer=show_viewer,
        stop_when_done=args.stop_when_done,
        perturbation=perturbation,
        residency=args.residency,
    )
    _refuse_torch_import(torch_free, "while building the scene and the compiled graphs")
    trainer = NeuroevolutionTrainer(
        env,
        pop_size,
        generations,
        args.master_seed,
        num_inputs=NUM_OBS,
        num_outputs=NUM_ACTIONS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        species_size=SPECIES_SIZE,
        # The tell target is inherited: numpy rollout -> numpy tell on cpu:0, ROCm -> xla on the
        # env device; ``--tell-backend`` overrides the backend only.
        tell_backend=tell_backend,
        plugin_path=plugin_path,
        output_dir=output_dir,
        logger=logger,
    )
    try:
        trainer.setup()
        _refuse_torch_import(torch_free, "while building the evolution graph")
        if not args.quiet:
            print(
                _format_header(args, env, trainer, pop_size, generations, perturbation, show_viewer, torch_free),
                flush=True,
            )
        summary = trainer.run()
        summary["config"].update(_task_config(args, env, perturbation, show_viewer, torch_free))
        _write_summary(trainer.summary_path, summary)
    finally:
        trainer.close()
        env.close()

    # The loop belongs to the trainer, so the last point at which the claim can be checked once
    # every generation has run is here; the artifacts are already on disk.
    _refuse_torch_import(torch_free, "after the generations completed")

    if not args.quiet:
        best = summary["totals"]["best_fitness"]
        print(
            f"done: {generations} generations, {summary['totals']['steps']} steps, "
            f"best fitness {'n/a' if best is None else f'{best:+.4f}'}; "
            f"torch_in_process={_torch_in_process()}; "
            f"wrote {trainer.history_path} and {trainer.summary_path}"
        )
    if trainer.tell_backend == "xla" and trainer.tell_device.kind != "cpu":
        # A live PJRT client makes the ROCm plugin abort at interpreter exit, after the work is
        # done and the artifacts are on disk; skip the destructor.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
