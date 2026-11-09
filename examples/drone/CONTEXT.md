# Intent
Demonstrates Crazyflie 2.X quadcopter simulation (`urdf/drones/cf2x.urdf`) across four interaction styles: keyboard teleop, scripted trajectory playback, PID point-to-point flight, and RL hover training.
All four motors are driven via `Drone.set_propellers_rpm(...)` around a hover base RPM of `14468.429183500699`.
The RL hover task also exists in an EvoX Compiler (`evoxc`) form used by the shared neuroevolution trainer: a pure-`evoxc` kernel module, a batched seeded evaluation environment, a NumPy parity oracle, and a smoke/parity CLI.
The torch/PPO path (`hover_env.py`, `hover_train.py`, `hover_eval.py`) stays untouched as the reference baseline the EvoX Compiler port is validated against.

# API Surface
- `interactive_drone.py` — keyboard-teleop demo; `DroneController` maps held arrow keys to differential rotor RPMs via Genesis keybindings.
- `fly.py` — pre-programmed flight; steps through a hardcoded `(N,4)` rotor-thrust trajectory array with `-v/--vis` flag.
- `fly_route.py` — point-to-point flight using `DronePIDController`; visits waypoints and records to `out/fly_route.mp4`.
- `quadcopter_controller.py` — `PIDController` (single-axis PID) and `DronePIDController` (cascaded position→velocity→attitude PID plus a 4-motor mixer).
- `hover_env.py` — `HoverEnv`, a vectorized rsl-rl environment; `step`/`reset`/`get_observations` return a `tensordict.TensorDict` observation; the reference baseline, not migrated.
- `hover_train.py` — PPO training via `rsl_rl.runners.OnPolicyRunner` on the GPU backend; the reference baseline, not migrated.
- `hover_eval.py` — loads a saved checkpoint, runs one inference rollout, optionally records to `out/hover.mp4`.
- `hover_legacy_reference.py` — the parity oracle: the pre-migration maths of `hover_env.py` as pure NumPy functions (`quat_to_xyz` incl. the gimbal-lock branch, `inv_quat`/`quat_mul`/`transform_quat_by_quat`/`transform_by_quat`, `propeller_rpm`, `body_frame_velocities`, `observation`, `reset_observation`, `reward_terms`, `crash_mask`, `timeout_mask`, `resample_commands`, `step_maths`); imports neither genesis, torch, tensordict nor evoxc.
- `hover_kernels.py` — pure `evoxc` task graphs (controls, obs, reset obs, advance) on top of the shared `examples/neuroevolution` runtime helpers; imports neither torch nor genesis.
- `hover_neat_env.py` — `HoverNeatEnv`, the batched seeded evaluation environment implementing the shared neuroevolution-env contract (`rows`, `episodes_per_policy`, `max_episode_length`, `evaluate(pop_arrays, generation)`, `close()`).
- `hover_neat_check.py` — smoke/parity/`--imports` CLI for the migrated stack; also the collected example smoke for it.
- `README.md` — run commands, control mappings, and installation notes.

# HoverEnv Internals
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

# EvoX Compiler Migration (hover)
## Graph contract (`hover_kernels.py`)
- Constants mirror `hover_train.get_cfgs()`: `NUM_OBS=17`, `NUM_ACTIONS=4`, `NUM_COMMANDS=3`, `DT=0.01`, `EPISODES_PER_POLICY=3`, `MAX_EPISODE_LENGTH=1500`, `CLIP_ACTIONS=1.0`, `HOVER_RPM=14468.429183500699`, `ACTION_GAIN=0.8`, `AT_TARGET_THRESHOLD=0.1`, `OBS_SCALES`, `REWARD_SCALES` (pre-`dt`), `REWARD_TERM_NAMES`, `YAW_LAMBDA=-10`, `TERMINATION`, `BASE_INIT_POS`, `BASE_INIT_QUAT`, `QUAT_EPS=1.1920928955078125e-07` (the float32 `gs.EPS`), `END_REASONS`.
- `hover_controls` / `build_controls(rows)` / `run_controls` — runtime raw `actions (rows,4)` -> clipped actions and `rpm (rows,4)`; separate from `hover_advance` because the RPM must be written before `scene.step()` while every other quantity is computed after it.
- `hover_obs` / `build_obs` / `run_obs` — `commands`, `base_pos`, absolute `base_quat`, world-frame lin/ang velocity, `last_actions` -> `(rows,17)` observation; the body-frame rotation is `_rotate_rows(v, _inv_quat_rows(base_quat))` on the raw, unnormalized quaternion.
- `hover_reset_obs` / `build_reset_obs(rows)` / `run_reset_obs(executable, *, commands, base_init_pos, base_init_quat)` — `commands` plus the per-row init pose (`(rows,3)` / `(rows,4)`, runtime operands, so a perturbed reset is supported) -> the analytic post-reset observation, with no `scene.step()`.
- `hover_advance` / `build_advance(rows, max_episode_length)` / `run_advance` — one post-step graph: episode counter, `rel_pos`/`last_rel_pos`, body velocities, yaw in degrees, `at_target`, next `commands`, the five `dt`-scaled reward terms, `crash`/`timeout`/`done`, next `obs`, `returns`, `alive`, the latched `end_code` (int8) and the consumed `base_pos` (`(rows,3)` float32 snapshot).
- `run_advance` takes `end_code` (`(rows,)` int8, the last entry of `_ADVANCE_OPERANDS`, default all-unfinished) and latches the first ending as error > timeout > crash, exactly like the host-side `HoverNeatEnv._classify_endings`.
- Fitness, policy and seeding are no longer task graphs: `evolution_graphs.build_fitness`/`run_fitness`, `policy_graphs.build_policy`/`run_policy` (width-generic genome forward, 17 -> 4) and `seeding.generation_seed_table`/`evolution_seed`/`EPISODES_PER_POLICY` are imported from `examples/neuroevolution`.
- The module reaches that shared layer through a documented, idempotent `sys.path` shim (`_SHARED` = sibling `neuroevolution` directory); it no longer imports anything from `examples/locomotion`.
- Static specializations: scales, thresholds, `dt`, `max_episode_length` and row count are baked into each graph; only the genuinely per-step values appear in a `run_*` signature, and the reset pose is a runtime operand.
- `hover_advance` mirrors the baseline order, including that the target term uses the command resampled on arrival (against both `rel_pos` and `last_rel_pos`) and that the observation's last slot is the previous step's action.
- Yaw is the `qd_atan2_fast` polynomial of `genesis/utils/geom.py` transcribed with `abs`/`min`/`max`/`sqrt`/`select` only (there is no `atan2` in `evoxc`), plus the `quat_to_xyz` gimbal-lock select; measured error is 1 float32 ulp (1.526e-05 deg near ±180°).
- Dropped as unreachable, and documented as such: the roll/pitch 180° termination terms (roll ≤ 180°, pitch ≤ 90°) and `_reward_yaw`'s `yaw > 180` unwrap.

## Environment contract (`hover_neat_env.py`)
- `HoverNeatEnv(pop_size, master_seed, *, horizon=None, episode_length_s=None, backend="numpy", device=None, show_viewer=False, seed_table=None, stop_when_done=True)`.
- Attributes: `rows = n_envs = 3 * pop_size` (policy-major: rows `3p, 3p+1, 3p+2` are the three episodes of policy `p`), `episodes_per_policy = 3`, `max_episode_length = ceil(episode_length_s / 0.01) = 1500`, `horizon` (default `max_episode_length`), `cache`, `executables()`.
- `evaluate(pop_arrays, generation)` returns `seeds (3,)`, `episode_returns (pop_size,3)`, `returns (rows,)`, `fitness (pop_size,)`, `steps`, `env_steps`, `wall_time_s`, `steps_per_s`, `env_steps_per_s`, `cache_hits`, `cache_misses`, `end_reasons` (`timeout`/`crash`/`error`/`unfinished`), `commands` (final targets).
- One episode per row per generation: a dead row is never reset inside a generation, so the baseline's reset-step reward pass (episode sums zeroed before the reset-step reward, reward computed on the post-reset state with the pre-reset yaw) is not reproduced; a truncated PPO episode consequently carries one extra post-reset step that this evaluator does not.
- Scene is the baseline's physics: `SimOptions(dt=0.01)`, `RigidOptions(dt=0.005, enable_collision=True, enable_joint_limit=True, constraint_solver=Newton)`, a `Plane` and `gs.morphs.Drone(file="urdf/drones/cf2x.urdf")`, built with `n_envs=rows`.
- `gs.init(backend=gs.cpu, precision="32", logging_level="warning")` runs only when genesis is not initialized, so a caller that already did `gs.init(backend=gs.gpu, ...)` keeps its backend.
- Command resampling: one `np.random.Generator` per row seeded from `[episode-slot seed, generation]`, and every row burns exactly three uniforms per step (`x~U(-1,1)`, `y~U(-1,1)`, `z~U(1,1)` with `z` pinned to 1.0); the distribution matches the baseline's conditional draws while the stream stays row-order and step-order independent.
- `scene.clear_errno()` runs once per generation reset and after any `get_error_envs_mask()` row, because a latched solver error re-raises on the next `scene.step()`.
- The module never imports torch: reads go through `genesis.utils.misc.tensor_to_array`, and the Genesis setters accept NumPy `array_like` directly (including `envs_idx`), so no host->device helper is needed.
- Remaining Genesis boundary per step: `get_pos`, `get_quat`, `get_vel`, `get_ang`, `get_error_envs_mask` (device -> host) and `set_propellers_rpm` (host -> device, NumPy accepted); per generation: `set_pos`/`set_quat`/`zero_all_dofs_velocity`.
- NumPy staging is the correctness-first boundary; zero-copy DLPack is the next optimisation round.

## Parity (`hover_neat_check.py --parity`)
- `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --parity` prints one `name<TAB>max_abs_diff` line per gate and `parity_ok 1`.
- Measured maxima (200 randomized plausible states plus a 200-step driven trace, `numpy`/`cpu:0`): `reward_target`, `reward_smooth`, `reward_angular`, `reward_crash` exactly 0; `reward_yaw` 1.673e-10; `reward_total` 3.576e-07; `returns` 5.960e-08; `observation`, `rpm`, `clipped_actions` exactly 0; `yaw_deg` 6.104e-05 deg; `crash`/`timeout`/`done`/`at_target`/`alive` masks exactly 0 difference.
- `reset_obs_vs_obs` (analytic reset observation vs the observation from the same live reset state) is exactly 0, and `policy_row_identity` (rows `3p..3p+2` answering to policy `p`) is exactly 0.
- Deterministic replay: `--parity --pop-size 2 --horizon 30` prints a `fingerprint` line; two independent processes give the identical line `seeds=1454269175,757617120,1421746521 returns=0.045082949,-0.005940037,-0.015909065,-0.062878124,-0.002508280,0.017566836 sum=-0.024585718`.
- `hover_kernels.py` also self-checks standalone (`PYTHONPATH=$PWD python examples/drone/hover_kernels.py`): yaw 1.526e-05 deg max, obs 0, fixed-state reward 7.451e-09, randomized reward 1.490e-08, masks agree on 200 states, the latched `end_code` matches a numpy replication of `_classify_endings`, the `base_pos` passthrough is the consumed position, and the reset observation follows the runtime init pose.
- `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --imports` AST-scans `hover_kernels.py`, `hover_neat_env.py`, `hover_neat_train.py` (skipped while absent) and the six `examples/neuroevolution` modules, prints one `module<TAB>torch_free|torch_import` line each plus `imports_ok 1`; `hover_device_rollout.py` is deliberately excluded because it exports to Genesis via `torch.from_dlpack`.

## Reproduce
- Smoke (defaults are small so the example harness stays fast): `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py`.
- Full-length CPU run: `... examples/drone/hover_neat_check.py --pop-size 8 --horizon 1500 --generations 2` (add `--no-stop-when-done` to keep every row stepping to the horizon).
- GPU run: `... examples/drone/hover_neat_check.py -g --pop-size 8 --horizon 1500 --generations 2` (Genesis on `gs.gpu`, graphs on `iree`@`rocm:0`; `--evoxc-backend`/`--evoxc-device` override).
- Check `rocm-smi --showpids` first: the MI300X is shared, and a contended run must not be quoted as throughput.

## Measured throughput
- CPU (pop 8 = 24 rows): 211.1 and 231.8 `scene.step()`/s per generation (221.6 steps/s, 5318 env-steps/s aggregate); with the run held to the horizon it sustains 234.0 / 233.6 steps/s and 5615 / 5607 env-steps/s (6.41 / 6.42 s per generation).
- Full-length runs end early because an untrained random genome leaves the 3/3/2 m box: 126 and 142 steps, every row ending in `crash`.
- MI300X / `iree`@`rocm:0` (pop 8 = 24 rows, contended by another agent's process): 67.8 and 107.4 steps/s (84.3 steps/s, 2022 env-steps/s aggregate), fitness matching the CPU run to ≤7e-6 and identical `end_reasons`.
- A 24-row batch is far too small to profit from the GPU (and the first generation pays the IREE compile), so CPU stays ~2.7x faster here — the same conclusion as the Go2 pilot; an uncontended 12-row/40-step smoke ran at 31.4 steps/s.

# Constraints
- Drone model is `urdf/drones/cf2x.urdf`; base hover RPM `14468.429183500699`; RPMs are clamped to safe ranges (0–25000 for teleop, 0.9×–1.5× base for PID).
- `interactive_drone.py`, `fly.py`, `fly_route.py`, and `hover_eval.py` run on `gs.cpu`; `hover_train.py` initializes `gs.gpu` and expects a GPU for its 8192 parallel environments.
- `hover_env.py` simulates at `dt=0.01` (100 Hz) with rigid substeps at `dt/2` and Newton constraint solver.
- `hover_env.py`, `hover_train.py` and `hover_eval.py` are the PPO/reference baseline and stay on torch + `tensordict`; the EvoX Compiler stack is additive and validated against them, not against a live import of them (the module cannot be imported where `tensordict` is absent, so `hover_legacy_reference.py` transcribes its maths).
- `hover_kernels.py` must stay free of torch and genesis; every graph is one `evoxc.defn` with a `build_*` that bakes statics and a `run_*` that stages operands, and the shared build/run helpers come from `examples/neuroevolution/evox_runtime.py`.
- `hover_neat_env.py` must not import torch; only NumPy bookkeeping and the Genesis boundary are allowed outside the graphs.
- `hover_train.py` and `hover_eval.py` enforce `rsl-rl-lib>=5.0.0` at import time and raise `ImportError` otherwise.
- Scripts use the `gs.morphs.Drone` morph and expect the URDF/mesh assets to be resolvable relative to the repo root.

# Dependencies
- RL stack of the baseline: `rsl-rl-lib>=5.0.0`, `tensorboard` (training logger), and `tensordict` (observations); not installed in the EvoX Compiler work environment and not required by the migrated stack.
- Migrated stack: `evoxc` 0.1.0 (VCS install, not declared in `pyproject.toml`) and, for the policy graph, `tensorneat` 0.2.0; optional `iree-base-compiler`/`iree-base-runtime` for non-CPU `evoxc` targets.
- Training throughput depends on a CUDA GPU; eval and interactive scripts run on CPU.
- RL upstream reference: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv); reward design inspired by Champion-level drone racing (Nature 2023).

# Gotchas
- README says `fly.py` "follows a predefined trajectory stored in `fly_traj.pkl`", but that file is not tracked and `fly.py` actually hardcodes the trajectory inline.
- Teleop keys: arrow Up/Down/Left/Right = forward/back/left/right; Space = increase thrust; Left Shift = decrease thrust; Esc = quit.
- `interactive_drone.py` breaks its run loop immediately when `PYTEST_VERSION` is set, so it exits after one step under pytest.
- `hover_eval.py --record` captures video but is notably slow; the README and docstring recommend dropping `--record` for performance.
- Dead config/state in `hover_env.py`: `simulate_action_latency` and `resampling_time_s` are read but never used; `_reward_yaw`'s `yaw > 180` wrap can never fire (Euler yaw ∈ [-180, 180]); the roll/pitch 180° termination terms are unreachable because Euler roll ≤ 180 and pitch ≤ 90.
- `hover_env.py` never calls `scene.get_error_envs_mask()` nor `scene.clear_errno()`, so a latched rigid-solver error re-raises on the next `scene.step()` instead of being absorbed by a reset.
- PID gains in `fly_route.py` are "tuned such that the drone can fly, not optimized".
- `hover_train.py` deletes any existing `logs/<exp-name>` directory before training.
- `fly_route.py` and `hover_eval.py` write recordings under `out/`, which must exist or be creatable.
- The policy graph is the shared width-generic `policy_graphs.build_policy`/`run_policy`, so the env passes `num_inputs=NUM_OBS` / `num_outputs=NUM_ACTIONS` explicitly; `go2_neat_kernels.build_policy`'s baked Go2 widths no longer apply here.
- The parity table's yaw tolerance is 5e-4 deg: the residual is the float32 minimax `atan2` plus the opposite quaternion-composition side (`quat_mul(inv(q_init), q)` vs `quat_mul(q, inv(q_init))`, identity here), while `reward_yaw` itself agrees to 1.7e-10.
- One ROCm run aborted after generation 0 with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` / `hipErrorLaunchFailure` (exit 134), a pre-existing HIP/XLA coexistence limit on that host; it did not reproduce in the validation runs and must be reported rather than papered over if it returns.
- `hover_kernels.py` and `hover_legacy_reference.py` are helper modules, but `tests/test_examples.py`'s `IGNORE_SCRIPT_PATTERNS` does not list them (it does list the three Go2 helpers), so the harness executes their `__main__` self-checks; adding them there is a sibling edit for the parent node.

# Test Strategy
- `tests/test_examples.py` (pytest `examples` marker) runs `interactive_drone.py`, `fly.py` and `fly_route.py`; `*_train.py`, `*_eval.py`, `*_env.py`, and `quadcopter_controller.py` are ignored by the example test filter.
- `hover_neat_check.py` is collected by the same harness and therefore defaults to a small CPU run (`--pop-size 4 --generations 1 --horizon 20`, ~19 s including the genesis import) that exits 0 headless.
- `hover_kernels.py` and `hover_legacy_reference.py` are collected too; both `__main__` blocks are fast self-checks (they need `evoxc` / nothing respectively).
- Iteration loop for the migrated stack: `hover_kernels.py` self-check, then `hover_neat_check.py --imports` for the torch-free proof, then `hover_neat_check.py --parity` for the numerical gates, then the deterministic fingerprint twice in separate processes, then a full-length CPU run.
- RL scripts of the baseline are exercised manually, not by the example suite.

# See Also
- `urdf/drones/cf2x.urdf` — Crazyflie 2.X drone model asset used by all scripts here.
- `../neuroevolution/` — the shared layer: `evox_runtime.py` (staging helpers, imported by `hover_kernels.py`), `seeding.py`/`policy_graphs.py`/`evolution_graphs.py` (imported by `hover_neat_env.py` and the self-check).
- `../locomotion/go2_neat_kernels.py` — `ExecutableCache` and width-fixed policy graph imported by `hover_neat_env.py` (sibling, read-only).
- `../` — sibling example groups and the shared `genesis` framework entry points.
