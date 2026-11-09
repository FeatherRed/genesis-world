# examples/locomotion

## Intent
RL locomotion stack for the Unitree Go2 quadruped in Genesis.
Provides a vectorized training environment, rsl-rl PPO training/eval entry points, and playback of pre-trained backflip policies.

## API Surface
- `go2_env.py` → `Go2Env` class: vectorized (`num_envs`) Go2 environment using `tensordict.TensorDict` for observations.
  Control runs at 50 Hz (`dt=0.02`, `substeps=2`), driving 12 motors via PD position control (`control_dofs_position`).
  `step(actions)` returns `(obs TensorDict, rew_buf, reset_buf, extras)`; `reset()` returns the initial `TensorDict`.
  Observation is a single `"policy"` tensor (45 dims): 3 base ang-vel + 3 projected gravity + 3 commands + 12 dof-pos + 12 dof-vel + 12 actions.
  Reward functions are auto-discovered from `reward_cfg["reward_scales"]` keys via `_reward_<name>` methods.
- `go2_train.py` → PPO training entry point via `rsl_rl.runners.OnPolicyRunner`; default 4096 parallel envs on GPU, logs to `logs/<exp_name>/` (pickles `cfgs.pkl`, saves `model_<iter>.pt`).
- `go2_eval.py` → loads a trained checkpoint (`logs/<exp_name>/model_<ckpt>.pt`) and runs the inference policy in a single viewer env on CPU; clears reward scales before building the env.
- `go2_backflip.py` → defines `BackflipEnv(Go2Env)` (60-dim phase-augmented observation, no commands/rewards) and plays `single.pt`/`double.pt` checkpoints from `backflip/`.
- `go2_evox_kernels.py` → EvoX Compiler (`evoxc`) tensor graphs for the scripted-trot gait (`GAIT`) and Go2 Walking reward (`REWARD`), built once via `build_gait`/`build_reward` and stepped via `run_gait`/`run_reward`; pure evoxc, no torch or genesis. `resolve_backend_device` honors explicit args plus `EVOXC_BACKEND`/`EVOXC_DEVICE` (default numpy/cpu).
- `go2_legacy_reference.py` → the pre-migration `try1.py` math transcribed into pure functions (`gait_step`, `walking_reward_terms`); the parity oracle for the evoxc port, with no `genesis` import (numpy gait, torch reward).

## Constraints
- Scripts are plain modules, not a package; run from the repo root, e.g. `python examples/locomotion/go2_train.py`.
- `go2_train.py` / `go2_eval.py` enforce `rsl-rl-lib>=5.0.0` via `importlib.metadata` at import time.
- `go2_env.py` imports `tensordict.TensorDict` and helpers from `genesis.utils.geom` (quat transforms).

## Design Decisions
- `SimOptions(dt=0.02, substeps=2)` matches the real robot's 50 Hz control rate.
- `simulate_action_latency=True` executes the previous step's actions (`last_actions`) to mirror the real robot's 1-step latency.
- `RigidOptions(enable_self_collision=False, max_collision_pairs=20, tolerance=1e-5)` is tuned for a locomotion policy with few contact pairs (saves memory).
- Backflip policies use a phase clock (`sin`/`cos` of `episode_length_buf`) in the observation instead of velocity commands.

## Dependencies
- `tensordict` (installed via the `dev` extra of `pyproject.toml`).
- `rsl-rl-lib>=5.0.0` (installed separately; not declared in `pyproject.toml`).
- `torch` and `genesis` (project core).

## Gotchas
- `max_collision_pairs=20`: exceeding ~20 collision pairs raises an exception; keep policy contact sparse.
- `base_init_quat` differs between walking (`[1,0,0,0]`) and backflip (`[0,0,0,1]`) configs; the backflip robot starts inverted.
- Backflip `.pt` checkpoints are data (torch `state_dict`s), not code; they are loaded with `torch.load(..., weights_only=True)`.
- `go2_backflip.py` is the only file run by `tests/test_examples.py` (pytest `examples` marker ignores `*_train.py`, `*_eval.py`, `*_env.py`); it shortens playback to 5 steps when `PYTEST_VERSION` is set.
- `go2_train.py` expects a GPU for the 4096-env throughput; `go2_eval.py`/`go2_backflip.py` run on CPU.

## Test Strategy
- Smoke-tested by `tests/test_examples.py` via `go2_backflip.py` (short, deterministic loop under pytest); the other scripts are deliberately excluded from the test run.

## Routing Table
- `backflip/` → Pre-trained single/double backflip policy checkpoints + readme (data, not code).
- `../tests/test_examples.py` → Example smoke-test runner (sibling — read-only, escalate writes to parent).
