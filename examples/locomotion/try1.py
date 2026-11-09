"""Go2 scripted-trot demo driven by EvoX Compiler (``evoxc``) graphs.

Genesis World still owns the simulation: the URDF, the plane, scene/physics
stepping, contacts and the viewer.  The migration boundary is narrow and lives
in :func:`read_state` / :func:`apply_targets` below:

* read the robot state out of Genesis (torch tensors on ``gs.device``) and hand
  it to the compiled graphs as host NumPy ``float32``;
* convert the gait graph's NumPy joint targets back to a ``gs.device`` tensor
  and push them through ``control_dofs_position``.

Both the scripted trot and the Go2 Walking reward are computed by the compiled
``evoxc`` executables from :mod:`go2_evox_kernels` — built *once* before the
step loop and reused every step.  No torch arithmetic happens in control or
reward; only this boundary touches tensors.

Usage::

    python examples/locomotion/try1.py                 # viewer, runs until closed
    python examples/locomotion/try1.py --max-steps 60  # viewer, bounded
    python examples/locomotion/try1.py --headless      # 5 s run, saves reward curve
    python examples/locomotion/try1.py --headless --evox-backend xla --evox-device cuda:0
"""

import argparse
import time
from itertools import count
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from go2_evox_kernels import (
    DT,
    STANDING,
    build_gait,
    build_reward,
    resolve_backend_device,
    run_gait,
    run_reward,
)


# --- Simulation boundary ------------------------------------------------------------
# These two helpers are the ONLY place where the compiled kernels meet Genesis.
# Everything they do is: read torch state -> NumPy, and NumPy targets -> torch.


def read_state(robot, dofs):
    """Read Genesis state (torch on ``gs.device``) as host NumPy ``float32``.

    Returns ``(state, elapsed_s)`` where ``elapsed_s`` is the measured boundary
    cost of the torch -> NumPy handover for this step.
    """
    start = time.perf_counter()
    with torch.no_grad():
        state = (
            robot.get_pos().detach().cpu().numpy().astype(np.float32),
            robot.get_quat().detach().cpu().numpy().astype(np.float32),
            robot.get_vel().detach().cpu().numpy().astype(np.float32),
            robot.get_ang().detach().cpu().numpy().astype(np.float32),
            robot.get_dofs_position(dofs).detach().cpu().numpy().astype(np.float32),
        )
    return state, time.perf_counter() - start


def apply_targets(robot, dofs, targets):
    """Send NumPy joint targets back to Genesis via ``control_dofs_position``.

    Returns the measured boundary cost of the NumPy -> torch write.
    """
    start = time.perf_counter()
    target_tensor = torch.as_tensor(targets, dtype=gs.tc_float, device=gs.device)
    robot.control_dofs_position(target_tensor, dofs)
    return time.perf_counter() - start


# --- Demo ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Preview a Go2 in Genesis World, driven by EvoX Compiler graphs.")
    parser.add_argument("--headless", action="store_true", help="Simulate five seconds and save the reward curve.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Bound the viewer run to N steps (headless always runs 250 steps).",
    )
    parser.add_argument(
        "--evox-backend",
        default=None,
        help="EvoX Compiler backend name (default: numpy; honors EVOXC_BACKEND).",
    )
    parser.add_argument(
        "--evox-device",
        default=None,
        help="EvoX Compiler device spec, e.g. 'cpu:0' or 'cuda:0' (default: cpu:0; honors EVOXC_DEVICE).",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", theme="raw")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, -2.2, 1.1),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
        ),
        show_viewer=not args.headless,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.42)))
    scene.build()

    legs = ("FR", "FL", "RR", "RL")
    joints = ("hip", "thigh", "calf")
    dofs = [robot.get_joint(f"{leg}_{joint}_joint").dofs_idx_local[0] for leg in legs for joint in joints]
    robot.set_dofs_position(STANDING.ravel(), dofs)
    robot.set_dofs_kp(np.full(12, 40.0), dofs)
    robot.set_dofs_kv(np.full(12, 1.0), dofs)
    if not args.headless:
        scene.viewer.follow_entity(robot, smoothing=0.15)

    # Compile both EvoX Compiler graphs ONCE, before the loop, and reuse them every step.
    resolved_backend, resolved_device = resolve_backend_device(args.evox_backend, args.evox_device)
    compile_start = time.perf_counter()
    gait_executable = build_gait(backend=args.evox_backend, device=args.evox_device)
    reward_executable = build_reward(backend=args.evox_backend, device=args.evox_device)
    compile_ms = (time.perf_counter() - compile_start) * 1e3
    print(
        f"EvoX Compiler backend: {resolved_backend}, device: {resolved_device.kind}:{resolved_device.index} "
        f"(compiled in {compile_ms:.1f} ms)"
    )

    previous_action = np.zeros(12, dtype=np.float32)
    rewards = []
    read_s = write_s = 0.0
    position = None

    # Headless always runs the five-second scripted trot; the viewer loops until
    # it is closed, optionally bounded by --max-steps.
    num_steps = 250 if args.headless else args.max_steps
    for step in range(num_steps) if num_steps is not None else count():
        if not args.headless and not scene.viewer.is_alive():
            break

        # Control comes entirely from the compiled gait graph.
        gait = run_gait(gait_executable, step, previous_action)
        write_s += apply_targets(robot, dofs, gait["targets"])
        scene.step()

        if args.headless:
            # Reward comes entirely from the compiled reward graph; the state read
            # and target write are the only tensor handovers in the loop.
            (base_pos, base_quat, base_lin_vel_world, base_ang_vel_world, dof_pos), read_elapsed = read_state(
                robot, dofs
            )
            read_s += read_elapsed
            terms = run_reward(
                reward_executable,
                base_pos,
                base_quat,
                base_lin_vel_world,
                base_ang_vel_world,
                dof_pos,
                gait["action_change"],
            )
            rewards.append(float(terms["total"]))
            position = base_pos

        previous_action = gait["action"]

    if args.headless:
        import matplotlib.pyplot as plt

        assert np.isfinite(position).all() and position[2] > 0.2, position
        assert np.isfinite(rewards).all(), "Non-finite walking reward"
        steps = len(rewards)
        print(f"Go2 final position after 5 seconds: {position}")
        print(f"Walking reward: mean={np.mean(rewards):.6f}, total={sum(rewards):.6f}")
        print(
            f"Boundary overhead: {1e3 * (read_s + write_s) / steps:.3f} ms/step "
            f"(read {1e3 * read_s / steps:.3f} + write {1e3 * write_s / steps:.3f}), "
            f"total {1e3 * (read_s + write_s):.3f} ms"
        )

        figure, axes = plt.subplots(figsize=(9, 4), layout="constrained")
        axes.plot(DT * np.arange(1, len(rewards) + 1), rewards, color="#136f86", linewidth=1.5)
        axes.set(title="Go2 scripted trot | Walking reward", xlabel="Simulation time (s)", ylabel="Reward per step")
        axes.grid(alpha=0.2)
        output = Path(__file__).with_name("go2_reward.png")
        figure.savefig(output, dpi=160)
        plt.close(figure)
        print(f"Reward curve: {output}")


if __name__ == "__main__":
    main()
