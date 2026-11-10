"""TensorNEAT trainer of the Go2 *walking* task on top of the compiled EvoX Compiler graphs.

This is the entry point of the TensorNEAT/``evoxc`` walking stack.  Every tensor op that
matters -- observation assembly, the genome forward pass, the reward/termination step, the
episode-mean fitness and the NEAT ``tell`` transition -- runs as a compiled graph from
``go2_neat_kernels.py``.  Genesis owns the scene and the physics; this module owns only the
CLI, the wiring and the artifacts.

What is where
-------------
``examples/locomotion/go2_env.py``   the behaviour being reproduced (env semantics).
``examples/locomotion/go2_train.py`` the untouched PPO control path (``rsl-rl``); read its
                                    ``get_cfgs()`` for the task constants, never import it --
                                    it raises at import time without ``rsl-rl-lib``.
``go2_neat_env.py``                 the batched seeded evaluation env (``Go2NeatEnv``).
``go2_neat_kernels.py``             the five compiled graphs and ``ExecutableCache``.
``examples/neuroevolution/``        the shared layer: seeding, the env protocol, the graphs
                                    and ``NeuroevolutionTrainer``, which runs the loop.

Loop
----
``NeuroevolutionTrainer`` owns the ask/tell loop: it traces the TensorNEAT pipeline only for
its statics (the stub ``TraceOnlyProblem`` never evaluates a problem, the fitness always comes
from the simulation) and per generation runs ``Go2NeatEnv.evaluate`` -> fitness -> ``run_tell``
-> record.  ``Go2NeatEnv.__init__`` builds the four env graphs and ``trainer.setup()`` the
``tell`` graph, all through the same ``ExecutableCache`` (``env.cache``), so every generation
reuses all five executables and the per-generation ``cache`` counters stay at misses 0.

Seeds -- two independent streams from one master seed
----------------------------------------------------
``np.random.SeedSequence(master_seed).spawn(2)``:
* the **evolution** stream -> ``evolution_seed(master_seed)`` -> ``Pipeline(seed=...)``;
* the **environment** stream -> ``generation_seed_table(master_seed, ...)``, one row of
  ``EPISODES_PER_POLICY`` distinct reset seeds per generation, prefix-stable.  Every policy
  of a generation shares the generation's seed triple (its three rows use the three seeds),
  so a generation compares policies on the same three initial states.

The table is built here and handed to both ``Go2NeatEnv`` (``seed_table=``) and the trainer,
so the episodes the env runs are provably the ones the trainer records.

Backends
--------
CPU/``numpy`` is the default.  ``-g/--gpu`` runs Genesis on ``gs.gpu`` and the rollout graphs
with IREE on ``cuda:0``.  ``--tell-backend numpy`` explicitly runs evolution on CPU.  On ROCm
(``--evoxc-device rocm:0``) the rollout runs on IREE and the evolution defaults to XLA on
``rocm:0``, with an explicit ``--pjrt-plugin`` (or the ``EVOXC_PJRT_PLUGIN`` environment
variable) selecting the PJRT plugin, because IREE's ROCm compile of the ``tell`` graph does
not terminate.  That resolution lives in ``NeuroevolutionTrainer``; this script only forwards
the flags and reads back the resolved backend/device for the header and the teardown.

On a ROCm host whose torch wheel bundles its own ``libhsa-runtime64`` that differs from the
system ROCm one, long GPU rollouts abort (torch's bundled HSA build corrupts AQL packets,
``HSA_STATUS_ERROR_INVALID_PACKET_FORMAT``).  That repair lives in ``gs.init``
(``genesis/utils/hsa.py``), which re-executes the process with the system runtime preloaded;
``GS_HSA_REEXEC=0`` opts out and ``GS_HSA_REEXEC_DONE=1`` is its one-shot sentinel.  Nothing
trainer-specific is needed here.

Artifacts (git-ignored, never committed)
----------------------------------------
``<output-dir>/history.jsonl``  one JSON object per generation.
``<output-dir>/summary.json``   resolved config + every generation record + totals.
``-o`` defaults to ``out/go2_neat``; ``-e/--exp-name`` appends a subdirectory.

Residency
---------
``--residency numpy`` (the default, the reference) stages every graph operand through host arrays.
``--residency device`` keeps them all on the ``evoxc`` device (zero-copy DLPack bridge, see
``go2_device_rollout.py``) and therefore needs a non-CPU ``--evoxc-device``; both residencies run the
same graphs on the same values and are expected to agree bit-for-bit.

``--torch-free`` is the device residency with the Torch claim enforced: it refuses any configuration
that would import Torch (a viewer, a host rollout, an XLA ``tell``) and, because device residency is
single-backend, leaves ``tell()`` on the environment's own backend/device (``iree@rocm:0``) instead
of forwarding ``--tell-backend`` / ``--pjrt-plugin``.

Under ``PYTEST_VERSION`` the run shrinks to one generation of 20 steps unless the matching
flags are given explicitly, so a smoke test cannot hang.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import genesis as gs

from go2_neat_env import (
    MAX_CONNS,
    MAX_NODES,
    SPECIES_SIZE,
    Go2NeatEnv,
    generation_seed_table,
)
from go2_neat_kernels import NUM_ACTIONS, NUM_OBS, Perturbation

# ``go2_neat_env`` imports ``go2_neat_kernels`` first, whose import-time shim puts
# ``examples/neuroevolution`` on ``sys.path``, so the shared layer imports by bare name.
from trainer import NeuroevolutionTrainer


# --- CLI ----------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a TensorNEAT Go2 walking population with Genesis and EvoX Compiler graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  python examples/locomotion/go2_neat_train.py --pop-size 8 --generations 3 --horizon 25\n"
            "  python examples/locomotion/go2_neat_train.py --pop-size 8 --generations 3\n"
            "  python examples/locomotion/go2_neat_train.py --pop-size 8 --generations 3 -g\n"
            "  python examples/locomotion/go2_neat_train.py -g --evoxc-device rocm:0  # AMD recipe\n"
        ),
    )
    parser.add_argument("--pop-size", type=int, default=8, help="Policies per generation (rows = 3 * pop_size).")
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
        help="scene.step() calls per generation; default the full episode (episode_length_s / dt).",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=20.0,
        help="Episode cap in seconds; it fixes the max_episode_length of the advance graph.",
    )
    parser.add_argument("--master-seed", type=int, default=0, help="Seed of both RNG streams.")
    parser.add_argument("--perturb-base-pos", type=float, default=0.05, help="Max |base position| reset offset [m].")
    parser.add_argument("--perturb-base-rpy", type=float, default=0.05, help="Max |roll/pitch/yaw| reset offset [rad].")
    parser.add_argument("--perturb-joint-pos", type=float, default=0.05, help="Max |joint| reset offset [rad].")
    parser.add_argument(
        "--no-perturbation",
        action="store_true",
        help="Zero all three reset offsets, reproducing the fixed reset of go2_env.",
    )
    parser.add_argument(
        "--stop-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a generation as soon as every row has ended (disable to force the full horizon).",
    )
    parser.add_argument(
        "--residency",
        choices=("numpy", "device"),
        default="numpy",
        help=(
            "Where the values that cross a graph boundary live: numpy (host arrays, the reference) "
            "or device (zero-copy, every operand stays on the evoxc device; needs a non-CPU device)."
        ),
    )
    parser.add_argument("-e", "--exp-name", type=str, default=None, help="Experiment name, a subdirectory of -o.")
    parser.add_argument("-o", "--output-dir", type=str, default="out/go2_neat", help="Directory of the artifacts.")
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Init Genesis on gs.gpu and default the EvoX Compiler graphs to IREE on cuda:0.",
    )
    parser.add_argument("--evoxc-backend", type=str, default=None, help="Rollout backend override (numpy or iree).")
    parser.add_argument("--evoxc-device", type=str, default=None, help="Rollout device override, e.g. cuda:0 or rocm:0.")
    parser.add_argument(
        "--tell-backend",
        choices=("numpy", "iree", "xla"),
        help=(
            "Evolution backend override. Defaults to the rollout backend and device, except on "
            "ROCm where it defaults to xla. numpy = CPU; iree = IREE; xla = PJRT/XLA. Under "
            "--torch-free evolution always runs on the rollout backend/device, so xla is rejected."
        ),
    )
    parser.add_argument(
        "--pjrt-plugin",
        type=str,
        default=None,
        help=(
            "Path to the XLA/PJRT plugin .so used by the xla tell backend; also exported as "
            "EVOXC_PJRT_PLUGIN (when that variable is unset) so device-device staging uses it. "
            "When omitted evoxc's own discovery applies (EVOXC_PJRT_PLUGIN or the installed "
            "jax_plugins/*/*.so)."
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
    parser.add_argument("-v", "--vis", action="store_true", help="Open the Genesis viewer.")
    parser.add_argument("--quiet", action="store_true", help="Write the artifacts without per-generation lines.")
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


# --- Main ---------------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pytest_mode = "PYTEST_VERSION" in os.environ
    generations = args.generations if args.generations is not None else (1 if pytest_mode else 2)
    if generations < 1:
        raise ValueError(f"--generations must be positive, got {generations}")
    horizon = args.horizon if args.horizon is not None else (20 if pytest_mode else None)
    if horizon is not None and horizon < 1:
        raise ValueError(f"--horizon must be positive, got {horizon}")
    if args.pop_size < 1:
        raise ValueError(f"--pop-size must be positive, got {args.pop_size}")

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
    if torch_free and args.residency != "device":
        # Device residency is the only mode in which no host/Torch tensor is created at a graph
        # boundary, so it is the only one a Torch-free process can run.
        args.residency = "device"
    backend = args.evoxc_backend or ("iree" if (args.gpu or torch_free) else "numpy")
    device = args.evoxc_device or ("cuda:0" if args.gpu and backend != "numpy" else None)
    if torch_free and device is None:
        # The MI300X host this path was built on; a CUDA host passes --evoxc-device cuda:0.
        device = "rocm:0"
    if torch_free and backend == "numpy":
        raise ValueError(
            "--torch-free needs a non-CPU rollout backend (device residency is IREE-backed), got "
            f"--evoxc-backend {backend!r}"
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
        # `Tensor.to` passes no options, so the option-less PJRT discovery used
        # for device staging must see the same plugin as the xla tell build.
        os.environ.setdefault("EVOXC_PJRT_PLUGIN", str(plugin_path))

    if not gs._initialized:
        if args.gpu or torch_free:
            # ``performance_mode=True`` selects Quadrants' static-array kernels on both residencies.
            gs.init(
                backend=gs.gpu,
                precision="32",
                logging_level="warning",
                performance_mode=True,
            )
        else:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    output_dir = Path(args.output_dir)
    if args.exp_name:
        output_dir = output_dir / args.exp_name

    seed_table = generation_seed_table(args.master_seed, generations)
    env = Go2NeatEnv(
        pop_size=args.pop_size,
        master_seed=args.master_seed,
        perturbation=perturbation,
        horizon=horizon,
        backend=backend,
        device=device,
        show_viewer=show_viewer,
        episode_length_s=args.episode_length_s,
        stop_when_done=args.stop_when_done,
        seed_table=seed_table,
        residency=args.residency,
    )
    _refuse_torch_import(torch_free, "while building the scene and the compiled graphs")
    trainer = NeuroevolutionTrainer(
        env,
        pop_size=args.pop_size,
        generations=generations,
        master_seed=args.master_seed,
        num_inputs=NUM_OBS,
        num_outputs=NUM_ACTIONS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        species_size=SPECIES_SIZE,
        tell_backend=tell_backend,
        plugin_path=plugin_path,
        output_dir=output_dir,
        seed_table=seed_table,
    )

    if not args.quiet:
        print(
            f"Go2 NEAT walking: pop_size={args.pop_size} rows={env.rows} generations={generations} "
            f"horizon={env.horizon} max_episode_length={env.max_episode_length} "
            f"genesis={'gpu' if args.gpu or torch_free else 'cpu'} evoxc={env.backend}@{env.device_name} "
            f"tell={trainer.tell_backend}@{trainer.tell_device_name} "
            f"master_seed={args.master_seed} evolution_seed={trainer.evolution_seed} "
            f"perturbation={'zero' if perturbation.is_zero else 'nonzero'} residency={env.residency} "
            f"torch_free={torch_free} torch_in_process={_torch_in_process()}"
        )

    trainer.setup()
    _refuse_torch_import(torch_free, "while building the evolution graph")
    summary = trainer.run()

    # The trainer owns the task-agnostic config keys; the Go2 run detail is merged into
    # ``config`` here without removing or renaming anything the trainer wrote.
    summary["config"].update(
        {
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
            "show_viewer": show_viewer,
            "stop_when_done": bool(args.stop_when_done),
            "residency": env.residency,
            "torch_free": torch_free,
            "torch_in_process": _torch_in_process(),
        }
    )
    trainer.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    env.close()
    if not args.quiet:
        print(
            f"done: {generations} generations, {summary['totals']['steps']} steps, "
            f"{summary['totals']['wall_time_s']:.1f} s total, "
            f"best fitness {summary['totals']['best_fitness']:+.4f}; "
            f"torch_in_process={_torch_in_process()}; "
            f"wrote {trainer.history_path} and {trainer.summary_path}"
        )
    if torch_free and _torch_in_process():
        print(
            "warning: --torch-free was requested but 'torch' is in sys.modules; some imported module "
            "pulled it in (report it, the loop itself never imports torch)",
            file=sys.stderr,
        )
    if trainer.tell_backend == "xla" and trainer.tell_device.kind != "cpu":
        # A live PJRT client makes the ROCm plugin abort at interpreter exit (after the work
        # is done, so the artifacts above are already on disk); skip the destructor.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
