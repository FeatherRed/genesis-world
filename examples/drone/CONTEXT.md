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
- PID gains in `fly_route.py` are "tuned such that the drone can fly, not optimized".
- `hover_train.py` deletes any existing `logs/<exp-name>` directory before training.
- `fly_route.py` and `hover_eval.py` write recordings under `out/`, which must exist or be creatable.

## Test Strategy
- `tests/test_examples.py` (pytest `examples` marker) runs only `interactive_drone.py`, `fly.py`, and `fly_route.py`; `*_train.py`, `*_eval.py`, `*_env.py`, and `quadcopter_controller.py` are ignored by the example test filter.
- RL scripts are exercised manually, not by the example suite.

## See Also
- `urdf/drones/cf2x.urdf` — Crazyflie 2.X drone model asset used by all scripts here.
- `../` — sibling example groups and the shared `genesis` framework entry points.
