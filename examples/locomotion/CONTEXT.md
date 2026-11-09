# examples/locomotion

## Intent
RL locomotion stack for the Unitree Go2 in Genesis: a scripted-trot demo whose control and reward math run as compiled EvoX Compiler (`evoxc`) graphs, plus an rsl-rl PPO training/eval stack and playback of pre-trained backflip policies.

## API Surface

### Scripted trot on EvoX Compiler graphs
- `try1.py` → the demo. Default = viewer until the window is closed; `--max-steps N` bounds the viewer run; `--headless` runs 250 steps (5 s), keeps per-step rewards, writes `go2_reward.png` next to the script and prints cumulative total + final pose.
  `--evox-backend` / `--evox-device` select the compiled-graph backend/device (default `numpy` / `cpu:0`); the resolved backend, device and compile time are printed.
- `go2_evox_kernels.py` → the two `evoxc` graphs (pure evoxc ops, no torch, no genesis) plus build/run helpers; both graphs are compiled ONCE before the step loop and reused every step.
  `GAIT` runtime inputs: `step` (0-d int32), `previous_action` (float32, 12) → outputs `targets`, `action`, `action_change` (float32, 12 each).
  `GAIT` static specializations: `standing_flat` (12), `phases` (4, `[0, pi, pi, 0]` per leg), `dt`, `inv_action_scale` (= 1/0.25).
  `REWARD` runtime inputs: `base_pos`(3), `base_quat`(4), `base_lin_vel_world`(3), `base_ang_vel_world`(3), `dof_pos`(12), `action_change`(12), all float32 → dict of the six Go2 Walking terms (`tracking_lin_vel`, `tracking_ang_vel`, `lin_vel_z`, `base_height`, `action_rate`, `similar_to_default`) plus `total`.
  `REWARD` static specializations: `standing_flat` (12), `dt`.
  Helpers: `DT`, `ACTION_SCALE`, `STANDING` (4×3) / `STANDING_FLAT` (12), `REWARD_TERM_NAMES`, `resolve_backend_device(backend, device)`, `build_gait()`, `build_reward()`, `run_gait(executable, step, previous_action)`, `run_reward(executable, ...)`.
- `go2_legacy_reference.py` → the pre-migration math as importable pure functions and the parity oracle: `gait_step(step, previous_action, ...)` (numpy float64) and `walking_reward_terms(...)` (torch float32 state terms, numpy float64 action rate). No genesis import; quaternion helpers are transcribed from `genesis.utils.geom`.

### RL stack
- `go2_env.py` → `Go2Env` class: vectorized (`num_envs`) Go2 environment using `tensordict.TensorDict` for observations.
  Control runs at 50 Hz (`dt=0.02`, `substeps=2`), driving 12 motors via PD position control (`control_dofs_position`).
  `step(actions)` returns `(obs TensorDict, rew_buf, reset_buf, extras)`; `reset()` returns the initial `TensorDict`.
  Observation is a single `"policy"` tensor (45 dims): 3 base ang-vel + 3 projected gravity + 3 commands + 12 dof-pos + 12 dof-vel + 12 actions.
  Reward functions are auto-discovered from `reward_cfg["reward_scales"]` keys via `_reward_<name>` methods.
- `go2_train.py` → PPO training entry point via `rsl_rl.runners.OnPolicyRunner`; default 4096 parallel envs on GPU, logs to `logs/<exp_name>/` (pickles `cfgs.pkl`, saves `model_<iter>.pt`).
- `go2_eval.py` → loads a trained checkpoint (`logs/<exp_name>/model_<ckpt>.pt`) and runs the inference policy in a single viewer env on CPU; clears reward scales before building the env.
- `go2_backflip.py` → defines `BackflipEnv(Go2Env)` (60-dim phase-augmented observation, no commands/rewards) and plays `single.pt`/`double.pt` checkpoints from `backflip/`.

## Simulation boundary
`try1.py` is split in three: Genesis owns the URDF/plane/scene/physics stepping/contacts/viewer; the compiled graphs own control and reward; `read_state()` and `apply_targets()` are the only contact points between them.
- Read: `robot.get_pos()/get_quat()/get_vel()/get_ang()/get_dofs_position(dofs)` return torch tensors on `gs.device` (CUDA with `gs.init(backend=gs.gpu)`); each is converted with `.detach().cpu().numpy().astype(np.float32)`.
- Write: the gait graph's numpy `targets` go back through `torch.as_tensor(targets, dtype=gs.tc_float, device=gs.device)` → `robot.control_dofs_position(...)`.
- torch appears only in that boundary; there is no torch arithmetic in control or reward.
- Measured handover cost on the numpy/cpu path: ~1.1–1.5 ms/step (read ~0.6–0.8 ms + write ~0.5–0.7 ms), dominated by the CUDA→host copies, not by the graphs.

## Backend / device configuration
- Precedence: explicit `--evox-backend`/`--evox-device` → `EVOXC_BACKEND`/`EVOXC_DEVICE` → `numpy` / `cpu:0`.
- The `numpy` reference backend is CPU-only (`BackendError: the numpy backend v1 supports CPU devices only, got device kind 'cuda'`); it is the default and the fastest path here (compile ~26–29 ms for both graphs; ~0.31 ms/step isolated).
- CUDA (measured on the RTX 5070): reachable with the `iree` adapter after `uv pip install iree-base-compiler iree-base-runtime`; compile ~1.0 s, ~18 ms/step end to end because every 3–12 element tensor is copied host↔device per graph call.
- Other adapters: `xla` fails without a PJRT C-API plugin; `tvm` fails without `jaxlib`; `iree` on `cpu:0` fails on Windows (`no linker tool discovered ... vcvarsall`).
- ROCm recipe (config-only, never exercised here — no AMD GPU): `--evox-backend iree --evox-device rocm:0`, or `EVOXC_BACKEND=iree EVOXC_DEVICE=rocm:0`, with the same iree extras installed.
- Nothing falls back silently: an unusable backend/device raises the adapter's error.

## Parity (measured)
- Gait vs `go2_legacy_reference.py`, 250 chained steps: max abs diff targets 2.6e-6, action 1.0e-5, action_change 1.3e-5 (float32 phase argument up to ~56 rad).
- Reward on 500 randomized plausible states: each term ≤ 9.6e-7 (`lin_vel_z`, `base_height` exactly 0), total 7.5e-9.
- 250-step headless run (numpy backend): total `-0.343842`, mean `-0.001375`, final position `[-2.32599, -0.7139309, 0.29976612]`.
  The pre-migration float64 baseline is total `-0.343841`, position `[-2.3259852, -0.71393394, 0.29976726]`, so deltas are ~5e-6 and the trajectory is not bit-identical (float32 accumulation, including the chained `previous_action`).
- `iree`/`cuda:0` reproduces the same 250-step totals to 6 decimals; isolated 250-step reward sums are identical between numpy and iree.

## Constraints
- Scripts are plain modules, not a package; run from the repo root, e.g. `python examples/locomotion/try1.py --headless`.
- `try1.py` imports `go2_evox_kernels` by bare module name (the script directory is on `sys.path`); both need `evoxc` installed, which is NOT declared in `pyproject.toml`.
- Control/reward math lives exclusively in `go2_evox_kernels.py`; `try1.py` must not re-derive the standing pose, `dt`, or the reward terms.
- `go2_train.py` / `go2_eval.py` enforce `rsl-rl-lib>=5.0.0` via `importlib.metadata` at import time.
- `go2_env.py` imports `tensordict.TensorDict` and helpers from `genesis.utils.geom` (quat transforms).
- `go2_reward.png` is a run artifact next to the script: not git-ignored, never committed.

## Design Decisions
- Static graph specializations (standing pose, phases, dt, 1/action_scale) keep the runtime signature to genuinely per-step values only.
- The `run_*` wrappers do input staging (`evoxc.from_numpy` when the executable is not on CPU) and output materialization, so the demo only performs boundary conversion.
- `SimOptions(dt=0.02, substeps=2)` matches the real robot's 50 Hz control rate.
- `simulate_action_latency=True` executes the previous step's actions (`last_actions`) to mirror the real robot's 1-step latency.
- `RigidOptions(enable_self_collision=False, max_collision_pairs=20, tolerance=1e-5)` is tuned for a locomotion policy with few contact pairs (saves memory).
- Backflip policies use a phase clock (`sin`/`cos` of `episode_length_buf`) in the observation instead of velocity commands.

## Dependencies
- `evoxc` (EvoX Compiler, installed in site-packages here; not declared in `pyproject.toml`), with optional `iree-base-compiler` / `iree-base-runtime` adapters for CUDA/ROCm targets.
- `tensordict` (installed via the `dev` extra of `pyproject.toml`).
- `rsl-rl-lib>=5.0.0` (installed separately; not declared in `pyproject.toml`).
- `torch` and `genesis` (project core).

## Gotchas
- The gait `step` input must be a 0-d `int32` ndarray; `np.ascontiguousarray` promotes 0-d to shape `(1,)` and the executable rejects it.
- Static ndarray leaves specialize the graph but must still be passed to `evoxc.run` on every call (validated with `array_equal`).
- Graph-time constants need `evoxc.constant(evoxc.from_numpy(arr))`; `evoxc.zeros` is a concrete creator, not a graph op.
- `max_collision_pairs=20`: exceeding ~20 collision pairs raises an exception; keep policy contact sparse.
- `base_init_quat` differs between walking (`[1,0,0,0]`) and backflip (`[0,0,0,1]`) configs; the backflip robot starts inverted.
- Backflip `.pt` checkpoints are data (torch `state_dict`s), not code; they are loaded with `torch.load(..., weights_only=True)`.
- `go2_train.py` expects a GPU for the 4096-env throughput; `go2_eval.py`/`go2_backflip.py`/`try1.py` run on CPU or GPU.

## Test Strategy
- `tests/test_examples.py` executes every `examples/**/*.py` as a subprocess, so `try1.py`, `go2_evox_kernels.py` and `go2_legacy_reference.py` all must import cleanly there (the last two exit immediately with no side effects).
- That harness still ignores `*_train.py`, `*_eval.py`, `*_env.py`, `behavior_cloning.py`, `quadcopter_controller.py`; `go2_backflip.py` is the other smoke test here and shortens playback to 5 steps when `PYTEST_VERSION` is set.
- Numeric parity is asserted against `go2_legacy_reference.py` with an ad-hoc script (not committed); no permanent unit test exists for it.

## Routing Table
- `backflip/` → Pre-trained single/double backflip policy checkpoints + readme (data, not code).
- `../tests/test_examples.py` → Example smoke-test runner (sibling — read-only, escalate writes to parent).
