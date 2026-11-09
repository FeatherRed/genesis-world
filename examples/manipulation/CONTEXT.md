## Intent
Robotic grasping example: a vectorized Franka Emika Panda arm that grasps and lifts a box in parallel environments.
Training is a two-stage pipeline — first a privileged reinforcement-learning (PPO) teacher policy via rsl-rl, then a vision-based behavior-cloning student distilled from stereo RGB plus end-effector pose.

## API Surface
- `grasp_env.py` — `GraspEnv` (vectorized env) and `Manipulator` (Panda arm wrapper around an MJCF entity).
- `grasp_env.py` `GraspEnv` exposes `reset()`, `step(actions)`, `get_observations()`, `get_stereo_rgb_images(normalize=True)`, `rescale_action()`, `grasp_and_lift_demo()`, and reward helpers.
- `grasp_env.py` `Manipulator` exposes `reset()`, `apply_action()`, `go_to_goal()`, `set_pd_gains()`, and pose properties (`base_pos`, `ee_pose`, `left/right/center_finger_pose`).
- `grasp_train.py` — training entry point: `python examples/manipulation/grasp_train.py --stage=rl|bc` (options `--exp-name`, `--vis`, `--num-envs`, `--max-iterations`, `--seed`).
- `grasp_eval.py` — evaluation entry point: `python examples/manipulation/grasp_eval.py --stage=rl|bc [--record [--video-path ...]]`.
- `behavior_cloning.py` — `BehaviorCloning` (trainer), `ExperienceBuffer` (FIFO replay), and `Policy` (shared stereo CNN encoder with an action head and a pose head).

## Constraints
- These files are run as scripts from within `examples/manipulation/` (they import each other relatively, e.g. `from grasp_env import GraspEnv`).
- None of them run as pytest examples: `tests/test_examples.py` `examples` marker ignores `*_train.py`, `*_eval.py`, `*_env.py`, and `behavior_cloning.py` — they are documentation/training entry points only.
- The directory has no child subdirectories (flat, four files).

## Dependencies
- `torch` — tensors, neural networks, tensorboard writer.
- `tensordict` (dev extra) — `TensorDict` observation container.
- `rsl-rl-lib >= 5.0.0` — PPO `OnPolicyRunner` used by train/eval (version checked via `importlib.metadata`).
- `gs-madrona` (optional, Linux-only native batch renderer) — enables `BatchRendererCameraOptions(use_rasterizer=True)`; guarded by an ImportError fallback.
- `numpy` — `behavior_cloning.py` reward logging.

## Gotchas
- Image resolution is configured as `(width, height)`; `grasp_env.py` stores `rgb_image_shape = (3, height, width)`.
- Camera sensors return `(B, H, W, 3)`; `get_stereo_rgb_images()` permutes to `(B, 3, H, W)`, divides by 255 when normalized, and concatenates left+right to `(B, 6, H, W)` — the BC policy consumes 6 channels.
- The batch-renderer camera path is only active when `gs_madrona` is importable AND `gs.backend == gs.cuda`; otherwise `RasterizerCameraOptions` is used.
- RL training requires a GPU: `grasp_train.py` inits `gs.gpu` with 2048 parallel envs by default; the BC stage drops to 10 envs.
- Sensor camera caches are manually invalidated (`left_cam._stale = True`, `right_cam._stale = True`) after object resets.
- DLS inverse kinematics moves the linear-algebra solve to CPU when the device is MPS (~300x faster there).
- PD gains in `Manipulator.set_pd_gains()` are tuned specifically for the Franka arm.
- Checkpoints: RL uses `logs/{exp_name}_rl/model_*.pt`; BC uses `logs/{exp_name}_bc/checkpoint_*.pt`; all configs are pickled as `cfgs.pkl` in the stage log dir.
- Evaluation loads `cfgs.pkl` and sets `box_fixed=False` plus `visualize_camera=True`; `--record` enables per-camera video recorders via `env_cfg["record_video"]`.

## See Also
- `../../genesis/options/sensors.py` — `BatchRendererCameraOptions` / `RasterizerCameraOptions` used for the stereo cameras.
- `../../genesis/vis/camera.py` — `Camera` class for the debug scene camera.
- `../../tests/test_examples.py` — the pytest `examples` marker that excludes this directory's scripts.
