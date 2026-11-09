import argparse
import math
from itertools import count
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat


def walking_reward(robot, dofs, standing, action_change, dt):
    """Score the demo using the default Go2 Walking reward terms."""
    base_inv_quat = inv_quat(robot.get_quat())
    velocity = transform_by_quat(robot.get_vel(), base_inv_quat)
    yaw_rate = transform_by_quat(robot.get_ang(), base_inv_quat)[2]
    velocity_error = (velocity[0] - 0.5).square() + velocity[1].square()
    height_error = robot.get_pos()[2] - 0.3
    pose_error = (robot.get_dofs_position(dofs) - standing).abs().sum()
    return dt * (
        (-velocity_error / 0.25).exp()
        + 0.2 * (-yaw_rate.square() / 0.25).exp()
        - velocity[2].square()
        - 50.0 * height_error.square()
        - 0.005 * np.square(action_change).sum().item()
        - 0.1 * pose_error
    )


def main():
    parser = argparse.ArgumentParser(description="Preview a Go2 in Genesis World.")
    parser.add_argument("--headless", action="store_true", help="Simulate five seconds and save the reward curve.")
    args = parser.parse_args()

    dt = 0.02
    gs.init(backend=gs.gpu, precision="32", theme="raw")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
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
    standing = np.array([[0.0, 0.8, -1.5], [0.0, 0.8, -1.5], [0.0, 1.0, -1.5], [0.0, 1.0, -1.5]])
    robot.set_dofs_position(standing.ravel(), dofs)
    robot.set_dofs_kp(np.full(12, 40.0), dofs)
    robot.set_dofs_kv(np.full(12, 1.0), dofs)
    if not args.headless:
        scene.viewer.follow_entity(robot, smoothing=0.15)
    else:
        standing_tensor = torch.as_tensor(standing.ravel(), dtype=gs.tc_float, device=gs.device)
        previous_action = np.zeros_like(standing)
        rewards = []

    # The diagonal pairs alternate swing and stance.
    phases = np.array([0.0, math.pi, math.pi, 0.0])
    for step in range(250) if args.headless else count():
        if not args.headless and not scene.viewer.is_alive():
            break
        elapsed = max(0.0, step * dt - 0.5)
        phase = 2.0 * math.pi * 2.0 * elapsed + phases
        ramp = min(1.0, 2.0 * elapsed)
        lift = np.maximum(0.0, -np.cos(phase))
        target = standing.copy()
        target[:, 1] += ramp * (-0.20 * np.sin(phase) + 0.20 * lift)
        target[:, 2] -= ramp * 0.45 * lift
        robot.control_dofs_position(target.ravel(), dofs)
        scene.step()
        if args.headless:
            action = (target - standing) / 0.25  # Recover Go2 Walking's normalized action units.
            rewards.append(walking_reward(robot, dofs, standing_tensor, action - previous_action, dt).item())
            previous_action = action

    if args.headless:
        import matplotlib.pyplot as plt

        position = robot.get_pos().cpu().numpy()
        assert np.isfinite(position).all() and position[2] > 0.2, position
        assert np.isfinite(rewards).all(), "Non-finite walking reward"
        print(f"Go2 final position after 5 seconds: {position}")
        print(f"Walking reward: mean={np.mean(rewards):.6f}, total={sum(rewards):.6f}")

        figure, axes = plt.subplots(figsize=(9, 4), layout="constrained")
        axes.plot(dt * np.arange(1, len(rewards) + 1), rewards, color="#136f86", linewidth=1.5)
        axes.set(title="Go2 scripted trot | Walking reward", xlabel="Simulation time (s)", ylabel="Reward per step")
        axes.grid(alpha=0.2)
        output = Path(__file__).with_name("go2_reward.png")
        figure.savefig(output, dpi=160)
        plt.close(figure)
        print(f"Reward curve: {output}")


if __name__ == "__main__":
    main()
