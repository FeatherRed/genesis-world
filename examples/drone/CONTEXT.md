## Intent
Demonstrates Crazyflie 2.X quadcopter simulation (`urdf/drones/cf2x.urdf`) across four interaction styles: keyboard teleop, scripted trajectory playback, PID point-to-point flight, and RL hover training.
All four motors are driven via `Drone.set_propellers_rpm(...)` around a hover base RPM of `14468.429183500699`.

## API Surface
- `interactive_drone.py` — keyboard-teleop demo; `DroneController` maps held arrow keys to differential rotor RPMs via Genesis keybindings.
- `fly.py` — pre-programmed flight; steps through a hardcoded `(N,4)` rotor-thrust trajectory array with `-v/--vis` flag.
- `fly_route.py` — point-to-point flight using `DronePIDController`; visits waypoints and records to `out/fly_route.mp4`.
- `quadcopter_controller.py` — `PIDController` (single-axis PID) and `DronePIDController` (cascaded position→velocity→attitude PID plus a 4-motor mixer).
- `hover_env.py` — `HoverEnv`, a vectorized rsl-rl environment; `step`/`reset`/`get_observations` return a `tensordict.TensorDict` observation.
- `hover_train.py` — PPO training via `rsl_rl.runners.OnPolicyRunner` on the GPU backend.
- `hover_eval.py` — loads a saved checkpoint, runs one inference rollout, optionally records to `out/hover.mp4`.
- `README.md` — run commands, control mappings, and installation notes.

## HoverEnv Internals
- Sizes: 4 actions, 3 command components (world-frame target `(x, y, z)`), 17-dim observation = `rel_pos*1/3` clipped ±1 (3) + raw `base_quat` (4) + body-frame `lin_vel*1/3` clipped ±1 (3) + body-frame `ang_vel/3.14159` clipped ±1 (3) + `last_actions` (4); current `actions` are not observed.
- Reward terms and scales (scale × `dt` at construction): `target` +10, `smooth` -1e-4, `yaw` +0.01, `angular` -2e-4, `crash` -10; `_reward_yaw` is `exp(-10*|yaw_rad|)`.
- Motor command each step is `(1 + 0.8*clip(action, ±1.0)) * 14468.429183500699` passed to `drone.set_propellers_rpm`, which raises if called twice in one sim step.
- `step()` order is fixed: clip action -> set RPM -> `scene.step()` -> read base pos/quat/vel/ang -> resample commands -> termination -> `reset_idx` -> reward -> observation -> `last_actions = actions`.
- Termination thresholds: roll/pitch 180°, `rel_pos` x/y/z 3/3/2 m, `base_pos.z < 0.1`; episode timeout at `ceil(15/0.01) = 1500` steps.
- Commands are resampled only when `|rel_pos| < 0.1` (uniform x,y in ±1, z fixed 1.0), never on a timer.
- `episode_sums[name]` is zeroed inside `reset_idx`, then the reset step's reward is added on top, so one reset-step reward leaks into the new episode's sum.
- `last_actions` is zeroed for reset rows before the reward pass, so `_reward_smooth` compares the new action against 0 on reset steps.
- `reset_buf` is allocated as `gs.tc_int` but `step()` rebinds it to the bool result of `(episode_length_buf > max) | crash_condition`.
- All base state reads (`get_pos`, `get_quat`, `get_vel`, `get_ang`) return `(n_envs, 3|4)` device tensors; velocities are world-frame and are rotated into the body frame via `transform_by_quat(v, inv_quat(base_quat))`.

## Constraints
- Drone model is `urdf/drones/cf2x.urdf`; base hover RPM `14468.429183500699`; RPMs are clamped to safe ranges (0–25000 for teleop, 0.9×–1.5× base for PID).
- `interactive_drone.py`, `fly.py`, `fly_route.py`, and `hover_eval.py` run on `gs.cpu`; `hover_train.py` initializes `gs.gpu` and expects a GPU for its 8192 parallel environments.
- `hover_env.py` simulates at `dt=0.01` (100 Hz) with rigid substeps at `dt/2` and Newton constraint solver.
- `hover_train.py` and `hover_eval.py` enforce `rsl-rl-lib>=5.0.0` at import time and raise `ImportError` otherwise.
- Scripts use the `gs.morphs.Drone` morph and expect the URDF/mesh assets to be resolvable relative to the repo root.

## Dependencies
- RL stack: `rsl-rl-lib>=5.0.0`, `tensorboard` (training logger), and `tensordict` (observations).
- Training throughput depends on a CUDA GPU; eval and interactive scripts run on CPU.
- RL upstream reference: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv); reward design inspired by Champion-level drone racing (Nature 2023).

## Gotchas
- README says `fly.py` "follows a predefined trajectory stored in `fly_traj.pkl`", but that file is not tracked and `fly.py` actually hardcodes the trajectory inline.
- Teleop keys: arrow Up/Down/Left/Right = forward/back/left/right; Space = increase thrust; Left Shift = decrease thrust; Esc = quit.
- `interactive_drone.py` breaks its run loop immediately when `PYTEST_VERSION` is set, so it exits after one step under pytest.
- `hover_eval.py --record` captures video but is notably slow; the README and docstring recommend dropping `--record` for performance.
- Dead config/state in `hover_env.py`: `simulate_action_latency` and `resampling_time_s` are read but never used; `_reward_yaw`'s `yaw > 180` wrap can never fire (Euler yaw ∈ [-180, 180]); the roll/pitch 180° termination terms are unreachable because Euler roll ≤ 180 and pitch ≤ 90.
- `hover_env.py` never calls `scene.get_error_envs_mask()` nor `scene.clear_errno()`, so a latched rigid-solver error re-raises on the next `scene.step()` instead of being absorbed by a reset.
- PID gains in `fly_route.py` are "tuned such that the drone can fly, not optimized".
- `hover_train.py` deletes any existing `logs/<exp-name>` directory before training.
- `fly_route.py` and `hover_eval.py` write recordings under `out/`, which must exist or be creatable.

## Test Strategy
- `tests/test_examples.py` (pytest `examples` marker) runs only `interactive_drone.py`, `fly.py`, and `fly_route.py`; `*_train.py`, `*_eval.py`, `*_env.py`, and `quadcopter_controller.py` are ignored by the example test filter.
- RL scripts are exercised manually, not by the example suite.

## See Also
- `urdf/drones/cf2x.urdf` — Crazyflie 2.X drone model asset used by all scripts here.
- `../` — sibling example groups and the shared `genesis` framework entry points.
