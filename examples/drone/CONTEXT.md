# Intent
Demonstrates Crazyflie 2.X quadcopter simulation (`urdf/drones/cf2x.urdf`) across four interaction styles: keyboard teleop, scripted trajectory playback, PID point-to-point flight, and RL hover training.
Motors turn around a hover base RPM of `14468.429183500699`, always through `Drone.set_propellers_rpm(...)`: the numpy residency hands over host arrays, the device residency the graph's rpm tensor, which on a CPU-kind IREE build travels via host memory and on every other build straight from the device.
The RL hover task also exists in an EvoX Compiler (`evoxc`) form on the shared `examples/neuroevolution` layer: pure-`evoxc` task kernels, one device-only graph, a batched seeded evaluation environment and a training entry point.
The evaluation environment has two residencies: the default NumPy-staged one and an opt-in, zero-copy device-resident rollout (`hover_device_rollout.py`).
Both residencies are Torch-free, and `--torch-free` on the trainer and on the check CLI runs a whole training loop in a process where `import torch` fails.
The torch/PPO baseline (`hover_env.py`, `hover_train.py`, `hover_eval.py`) stays untouched and unmigrated; the EvoX Compiler stack is validated against its maths (transcribed in `hover_legacy_reference.py`), not against a live import of it.

# API Surface
- `interactive_drone.py` — keyboard-teleop demo; `DroneController` maps held arrow keys to differential rotor RPMs via Genesis keybindings.
- `fly.py` — pre-programmed flight; steps through a hardcoded `(N,4)` rotor-thrust trajectory array with `-v/--vis` flag.
- `fly_route.py` — point-to-point flight using `DronePIDController`; visits waypoints and records to `out/fly_route.mp4`.
- `quadcopter_controller.py` — `PIDController` (single-axis PID) and `DronePIDController` (cascaded position→velocity→attitude PID plus a 4-motor mixer).
- `hover_env.py` — `HoverEnv`, a vectorized rsl-rl environment; `step`/`reset`/`get_observations` return a `tensordict.TensorDict` observation; the reference baseline, not migrated.
- `hover_train.py` — PPO training via `rsl_rl.runners.OnPolicyRunner` on the GPU backend; the reference baseline, not migrated.
- `hover_eval.py` — loads a saved checkpoint, runs one inference rollout, optionally records to `out/hover.mp4`.
- `hover_legacy_reference.py` — the parity oracle: the hover maths of `hover_env.py` as pure NumPy functions (`quat_to_xyz` incl. the gimbal-lock branch, `inv_quat`/`quat_mul`/`transform_quat_by_quat`/`transform_by_quat`, `propeller_rpm`, `body_frame_velocities`, `observation`, `reset_observation`, `reward_terms`, `crash_mask`, `timeout_mask`, `resample_commands`, `step_maths`); imports neither genesis, torch, tensordict nor evoxc.
- `hover_kernels.py` — the four drone task graphs (`hover_controls`, `hover_obs`, `hover_reset_obs`, `hover_advance`) plus the live `qd_atan2_fast` transcription and the `__main__` parity self-check; everything generic (seeding, policy graph, fitness, cache, staging helpers) is imported from `examples/neuroevolution`; imports neither torch nor genesis.
- `hover_device_kernels.py` — `hover_device_episode_returns` / `build_episode_returns(n_policies, *, episodes_per_policy=EPISODES_PER_POLICY, backend="numpy", device=None)` / `run_episode_returns(executable, returns)`: the `(rows,)` float32 returns → the policy-major `(pop_size, 3)` matrix as a real `evoxc.reshape` device op; imports neither genesis nor torch and has no `__main__`.
- `hover_neat_env.py` — `HoverNeatEnv`, the batched seeded evaluation environment implementing `env_base.BatchedEpisodeEnv` (`rows`, `episodes_per_policy`, `max_episode_length`, `cache`, `evaluate(pop_arrays, generation)`, `close()`), with `perturbation=seeding.Perturbation(...)` and `residency="numpy"|"device"`.
- `hover_device_rollout.py` — `DeviceRollout`, the `residency="device"` rollout: Torch-free, owns the per-step device tensors, the DLPack import of the live getters and of the solver-error mask, the `DroneEntity.set_propellers_rpm` write, and the `(rows,)` → `(pop,3)` returns graph call.
- `hover_neat_train.py` — the drone's training entry point on `trainer.NeuroevolutionTrainer` (CLI, `history.jsonl` + `summary.json`, one log line per generation, `--torch-free`).
- `hover_neat_check.py` — the environment-level smoke/parity CLI (`--parity`, `--residency-parity`, `--imports`, `--perturb`, `--residency`, `--torch-free`), which reads a device leaf back through its payload's `to_host()` (`.numpy()` needs host memory and raises on `iree@rocm:0`); also the collected example smoke for the migrated stack.
- `README.md` — run commands, control mappings, installation notes and the Torch-free training commands.

# HoverEnv Internals (the baseline the graphs reproduce)
- Sizes: 4 actions, 3 command components (world-frame target `(x, y, z)`), 17-dim observation = `rel_pos*1/3` clipped ±1 (3) + raw `base_quat` (4) + body-frame `lin_vel*1/3` clipped ±1 (3) + body-frame `ang_vel/3.14159` clipped ±1 (3) + `last_actions` (4); current `actions` are not observed.
- Reward terms and scales (scale × `dt` at construction): `target` +10, `smooth` -1e-4, `yaw` +0.01, `angular` -2e-4, `crash` -10; `_reward_yaw` is `exp(-10*|yaw_rad|)`.
- Motor command each step is `(1 + 0.8*clip(action, ±1.0)) * 14468.429183500699` passed to `drone.set_propellers_rpm`, which raises if called twice in one sim step.
- `step()` order is fixed: clip action -> set RPM -> `scene.step()` -> read base pos/quat/vel/ang -> resample commands -> termination -> `reset_idx` -> reward -> observation -> `last_actions = actions`.
- Termination thresholds: roll/pitch 180°, `rel_pos` x/y/z 3/3/2 m, `base_pos.z < 0.1`; episode timeout at `ceil(15/0.01) = 1500` steps.
- Commands are resampled only when `|rel_pos| < 0.1` (uniform x,y in ±1, z fixed 1.0), never on a timer.
- `episode_sums[name]` is zeroed inside `reset_idx`, then the reset step's reward is added on top, so one reset-step reward leaks into the new episode's sum.
- `last_actions` is zeroed for reset rows before the reward pass, so `_reward_smooth` compares the new action against 0 on reset steps.
- `reset_buf` is allocated as `gs.tc_int` but `step()` rebinds it to the bool result of `(episode_length_buf > max) | crash_condition`.
- All base state reads (`get_pos`, `get_quat`, `get_vel`, `get_ang`) return `(n_envs, 3|4)` device tensors; velocities are world-frame and are rotated into the body frame via `transform_quat_by_quat(v, inv_quat(base_quat))`.

# Task graphs
- Constants mirror `hover_train.get_cfgs()`: `NUM_OBS=17`, `NUM_ACTIONS=4`, `NUM_COMMANDS=3`, `DT=0.01`, `MAX_EPISODE_LENGTH=1500`, `CLIP_ACTIONS=1.0`, `HOVER_RPM=14468.429183500699`, `ACTION_GAIN=0.8`, `AT_TARGET_THRESHOLD=0.1`, `OBS_SCALES`, `REWARD_SCALES` (pre-`dt`), `REWARD_TERM_NAMES`, `YAW_LAMBDA=-10`, `TERMINATION`, `BASE_INIT_POS`, `BASE_INIT_QUAT`, `QUAT_EPS` (the float32 `gs.EPS`), `END_REASONS`.
- `hover_controls` / `build_controls(rows)` / `run_controls` — runtime raw `actions (rows,4)` -> clipped actions and `rpm (rows,4)`; separate from `hover_advance` because the RPM must be written before `scene.step()` while every other quantity is computed after it.
- `hover_obs` / `build_obs` / `run_obs` — `commands`, `base_pos`, absolute `base_quat`, world-frame lin/ang velocity, `last_actions` -> `(rows,17)`; the body-frame rotation is `_rotate_rows(v, _inv_quat_rows(base_quat))` on the raw, unnormalized quaternion; the environment does not drive this graph (its observation comes out of `hover_advance`), only the parity CLI and the graph tests do.
- `hover_reset_obs` / `build_reset_obs(rows)` / `run_reset_obs(executable, *, commands, base_init_pos, base_init_quat)` — the analytic post-reset observation, with NO `scene.step()`; the init pose is a runtime operand `(rows,3)`/`(rows,4)` so a perturbed reset is supported.
- `hover_advance` / `build_advance(rows, max_episode_length)` / `run_advance` — one post-step graph: episode counter, `rel_pos`/`last_rel_pos`, body velocities, yaw in degrees, `at_target`, next `commands`, the five `dt`-scaled reward terms, `crash`/`timeout`/`done`, next `obs`, `returns`, `alive`, `yaw_deg`, the latched `end_code`, and a `base_pos` snapshot (a fresh buffer for the next `last_base_pos`).
- `end_code` (int8, 0 unfinished / 1 timeout / 2 crash / 3 error) is a stateful input+output: the graph latches the first ending with the rule error > timeout > crash, once, never re-armed; the env derives `end_reasons` from it instead of re-classifying on the host, so the rule has exactly one owner.
- Static specializations: scales, thresholds, `dt`, `max_episode_length` and row count are baked into each graph; only genuinely per-step values appear in a `run_*` signature.
- `hover_advance` mirrors the baseline order, including that the target term uses the command resampled on arrival (against both `rel_pos` and `last_rel_pos`) and that the observation's last slot is the previous step's action.
- Yaw is the `qd_atan2_fast` polynomial of `genesis/utils/geom.py` transcribed with `abs`/`min`/`max`/`sqrt`/`select` only (there is no `atan2` in `evoxc`), plus the `quat_to_xyz` gimbal-lock select; measured error is 1 float32 ulp (1.526e-05 deg near ±180°).
- Dropped as unreachable, and documented as such: the roll/pitch 180° termination terms (roll ≤ 180°, pitch ≤ 90°) and `_reward_yaw`'s `yaw > 180` unwrap.
- `hover_device_kernels.hover_device_episode_returns` is the one graph a device-resident step needs beyond those four: the `(rows,)` returns → the policy-major `(pop,3)` matrix the trainer scores, because `DeviceArray.reshape` and every numpy call on a device array dispatch through numpy and return a host copy, so a shape change of a device buffer must be a graph.
- It owns exactly one graph where `go2_device_kernels` owns three: `hover_controls` already is the clip → RPM graph, and the Torch-free getters hand the advance operands over directly, so there is no control or state-composition graph to add.
- `build_episode_returns` stashes `n_policies`/`episodes_per_policy`/`rows` on the executable and `run_episode_returns` re-passes them, because `evoxc` validates static operands on every call; the rows are policy-major, so the reshape is the whole mapping.
- `hover_kernels.py` puts the sibling `examples/neuroevolution` directory on `sys.path` once (documented shim), then imports `EPISODES_PER_POLICY` from `seeding` and `POP_OPERANDS`/`_attach`/`_constant`/`_device_name`/`_outputs`/`_positive_int`/`_staged`/`_to_device_array`/`_to_numpy`/`resolve_backend_device` from `evox_runtime`; `hover_device_kernels.py` imports `_attach`/`_device_name`/`_outputs`/`_positive_int`/`_rows_of`/`_to_device_array`/`resolve_backend_device` the same way.
- `hover_neat_env.py` imports `build_policy`/`run_policy` from `policy_graphs`, `ExecutableCache`/`build_fitness`/`run_fitness` from `evolution_graphs`, `Perturbation`/`EPISODES_PER_POLICY`/`generation_seed_table`/`perturbation_from_seed` from `seeding`, `BatchedEpisodeEnv`/`validate_evaluation`/`validate_evaluation_device` from `env_base`, `TraceOnlyProblem` from `trainer` and `build_episode_returns` from `hover_device_kernels`.
- Nothing drone-specific duplicates the shared layer: no local seed table, evolution seed, policy graph, fitness graph, executable cache or hand-rolled generation loop, and `examples/locomotion` is not imported by this directory.

# Environment contract (`hover_neat_env.py`)
- `HoverNeatEnv(pop_size, master_seed, *, horizon=None, episode_length_s=None, backend="numpy", device=None, show_viewer=False, seed_table=None, stop_when_done=True, perturbation=None, residency="numpy")`.
- Attributes: `rows = n_envs = 3 * pop_size` (policy-major: rows `3p, 3p+1, 3p+2` are the three episodes of policy `p`), `episodes_per_policy = 3`, `max_episode_length = ceil(episode_length_s / 0.01) = 1500`, `horizon` (default `max_episode_length`), `cache`, `residency`, `executables()`.
- `evaluate(pop_arrays, generation)` returns `seeds (3,)`, `episode_returns (pop_size,3)`, `returns (rows,)`, `fitness (pop_size,)`, `steps`, `env_steps`, `wall_time_s`, `steps_per_s`, `env_steps_per_s`, `cache_hits`, `cache_misses`, `end_reasons`, `commands`.
- The numpy residency checks the mapping with `env_base.validate_evaluation`; the device residency checks it with `env_base.validate_evaluation_device` (the four training leaves are device tensors).
- `__init__` asserts `isinstance(self, BatchedEpisodeEnv)` and names any missing contract attribute, which is the declaration that the env satisfies the shared protocol.
- One episode per row per generation: a dead row is never reset inside a generation, so the baseline's reset-step reward pass is not reproduced; a truncated PPO episode consequently carries one extra post-reset step that this evaluator does not.
- Scene is the baseline's physics: `SimOptions(dt=0.01)`, `RigidOptions(dt=0.005, enable_collision=True, enable_joint_limit=True, constraint_solver=Newton)`, a `Plane` and `gs.morphs.Drone(file="urdf/drones/cf2x.urdf")`, built with `n_envs=rows`.
- `gs.init(backend=gs.cpu, precision="32", logging_level="warning")` runs only when genesis is not initialized, so a caller that already did `gs.init(backend=gs.gpu, ...)` keeps its backend.
- `reset_generation(generation, rngs)` writes the generation's pose with `drone.set_qpos(np.concatenate((pos, quat), axis=1), zero_velocity=True, envs_idx=None)`: the base pose of a free link *is* its qpos, measured bit-identical to `drone.set_pos`/`set_quat`, which work Torch-free too (verified before and after a `scene.step()`).
- `reset_generation` then calls `zero_all_dofs_velocity()` and `scene.clear_errno()`, and returns the first command row plus the analytic reset observation from `run_reset_obs` (no step needed).
- Command resampling: one `np.random.Generator` per row seeded from `[episode-slot seed, generation]`, and every row burns exactly three uniforms per step (`x~U(-1,1)`, `y~U(-1,1)`, `z~U(1,1)` with `z` pinned to 1.0); the distribution matches the baseline's conditional draws while the stream stays row-order and step-order independent.
- Seed table: `rows = 3 * pop_size`, `episodes_per_policy = 3`, one ordered seed triple per generation shared by every policy and a fresh triple next generation — generation 0 `[1454269175, 757617120, 1421746521]`, generation 1 `[1313752747, 908599966, 53557616]`.
- `perturbation` defaults to an all-zero `seeding.Perturbation()`, which reproduces the fixed reset bit-for-bit (the `is_zero` short-circuit returns the exact tiled constants).
- A nonzero perturbation recomputes `_reset_pos`/`_reset_quat` per generation as `BASE_INIT_POS + offset` and `normalize(q_pert * q_init)`; row `r` uses episode slot `r % 3`, so every policy of a generation starts from the SAME three perturbed states (a paired comparison).
- NumPy staging is the correctness-first boundary for `residency="numpy"`: `get_pos`/`get_quat`/`get_vel`/`get_ang`/`get_error_envs_mask` go through `genesis.utils.misc.tensor_to_array`, and the setters (`set_propellers_rpm`, `set_qpos`) accept NumPy `array_like` directly, so this residency runs Torch-free too.
- `executables()` returns the five graphs of `hover_kernels` (controls, reset observation, policy, advance, fitness) and, under `residency="device"`, the episode-returns graph of `hover_device_kernels` as a sixth.
- `scene.clear_errno()` runs once per generation reset (the numpy residency), and the device rollout clears every step right after its advance graph consumed the mask, because a latched solver error re-raises on the next `scene.step()`.

# Device residency (`hover_device_rollout.py`)
- `DeviceRollout(env, executables=None)` with `evaluate(pop_arrays, generation)`, `state_parity()` and an idempotent `close()`; it returns exactly the env's result mapping.
- It rejects an env whose `backend` is not `iree`, and `HoverNeatEnv` rejects `residency="device"` on the numpy backend, so the zero-copy interop is IREE's DLPack bridge.
- Per step: policy graph -> `hover_controls` (clipped actions + rpm) -> `drone.set_propellers_rpm(rpm)` -> `scene.step()` -> `qd.sync()` -> `hover_advance`; the state carried between steps (`commands`, `returns`, `episode_length`, `alive`, `end_code`, `last_actions`, `last_base_pos`, `obs`) never leaves the device.
- The generation reset stays the env's `reset_generation` (one host round trip: state write, commands, seeds), and the reset OBSERVATION is recomputed on device by `hover_reset_obs` from the same command and init-pose operands; the env's host observation is discarded.
- Results: `episode_returns` from `run_episode_returns`, `returns`, `fitness` from the compiled fitness graph and `seeds` as a device int64 tensor; `steps`, `env_steps`, `wall_time_s`, both rates, `cache_hits`/`cache_misses`, `end_reasons` and `commands` stay host logging values.
- `state_parity()` reads the four imported state operands back once and proves them bit-identical to `drone.get_pos/get_quat/get_vel/get_ang` (0.0 max abs diff).
- Solver reads are the live Torch-free getters: `get_pos`/`get_quat`/`get_ang` hand out a strided `DeviceView` of the batch-last solver field, compacted on device with `DeviceView.contiguous()` (IREE refuses a strided DLTensor), and `get_vel` already carries the `cd_vel + cd_ang × (pos − root_COM)` transport term the entity getter composes by hand.
- Zero-copy import: `capsule = view.__dlpack__()` -> `hal.from_dlpack_capsule(capsule)` -> `iree.runtime.DeviceArray(hal, view)` -> `evoxc.core.Tensor(IreeDevicePayload(...))`, the call form of `evoxc.backends.adapters.iree._device_array_from_dlpack`; the HAL device is taken from the advance executable's `backend_executable.runtime_device` with a fallback that remembers the first staged payload's.
- Graph output -> graph input passes through with no staging (`IreeExecutable.run` blocks on its own queue), which is how the raw action, the clipped action, the RPMs, the observation, `returns`/`alive`/`episode_length`/`end_code` and the returns handed to the fitness and episode-returns graphs travel.
- `_write_rpm` calls `DroneEntity.set_propellers_rpm` directly: its torch-free branch stages the graph's rpm tensor with `torch_free.stage_kernel_values` (device to device on the streaming Quadrants stream, a fresh buffer per call) and advances the device-resident `_propellers_revs` accumulator that `update_propeller_vgeoms` (visualizer only, never called by a headless rollout) animates with.
- On a CPU-kind IREE device (`iree@cpu:0`) the payload exports no DLPack (its read path is `data.to_host()`), so `_rpm_source` hands the setter that host array — the only host hop of the write path, and only for that configuration.
- `qd.sync()` fences once per step — after `scene.step()` and after the error-mask kernels, before the advance graph imports the buffers Genesis just wrote — and once per generation after the reset write, before the reset observation and the first step read the state; the rotor write needs no fence of its own, because the setter stages on that same stream.
- Remaining host touches per step, exhaustive: the `(rows,3)` host RNG command draw (by contract, staged as `resampled_commands`), and — while `stop_when_done` is on — the `(rows,)` int8 `end_code` readback that reproduces the numpy loop bound; the ending decision itself is in-graph.
- The solver-error mask is a device operand, not a host touch: the torch-free scene hands out the live `(rows,)` bool buffer, which is imported zero-copy as the `error_mask` operand (IREE takes the `kDLBool` DLPack) and then cleared with a bare `clear_errno()`; passing the mask back would read it to the host in `mask_indices`, so the unconditional kernel-only clear is the one that stays on the device.
- Remaining host touches per generation: the reset write plus seed/command staging, the four genome payload leaves (pass-through when the trainer hands device tensors), and at the end `end_code` (when `stop_when_done` is off), `commands` and the bookkeeping.
- Bool graph outputs cannot be DLPack-exported, so every exported leaf stays f32/i32; the int8 `end_code` only ever travels host-ward.
- Nothing in this module reshapes, slices or `np.*`-es a device value; the one shape change goes through `run_episode_returns`.

# Torch boundary
- `hover_kernels.py`, `hover_device_kernels.py`, `hover_device_rollout.py`, `hover_neat_env.py` and `hover_neat_train.py`, plus the six modules of `examples/neuroevolution`, contain no `import torch` / `from torch ...`; `hover_neat_check.py --imports` is the AST-scan proof over exactly those eleven files and prints `imports_ok 1` (all `torch_free`).
- `hover_device_rollout.py` imports no Torch at module scope or inside a helper, and calls no Torch operation: the bridges are `genesis.utils.device_buffer` on the way in (a torch-free getter's `DeviceView` and the scene's bool error mask are their own DLPack producers) and `DroneEntity.set_propellers_rpm` on the way out, so no Torch tensor exists anywhere on the path.
- `import genesis` is itself Torch-free on this path (`genesis.utils.torch_compat`), so the whole trainer runs with `import torch` failing; `--torch-free` proves it at runtime by refusing the run as soon as `"torch" in sys.modules`.
- Every torch-blocked run keeps `torch` out of `sys.modules`: `hover_neat_train.py` prints `torch_in_process=False` in its header and `summary.json`, `hover_neat_check.py` prints `torch_free 1` (it has no `torch_in_process` line).
- `genesis.utils.misc.tensor_to_array` on the read path (numpy residency) is a device -> host copy required because the graphs are host-staged; `residency="device"` removes it.
- NumPy arrays passed to `set_propellers_rpm`/`set_qpos` (numpy residency) are one host -> device copy per write.
- `performance_mode=True` selects the static-array mode (`_use_ndarray=False`, which also turns off deterministic algorithms) and builds and runs Torch-free.
- It is not bit-identical to the deterministic ndarray mode the rollout is validated in: the same fixed population through the device residency moves `returns` by up to 1.1176e-08 (1-10 float32 ulp over 30 steps) while `fitness` `[0.007744627, -0.015939852]` and `seeds` stay exact.
- The torch-free path of `hover_neat_train.py` therefore keeps `performance_mode=False`, and the comment above its `gs.init` call records that reason.
- The viewer is a Torch consumer, which is why `--torch-free` refuses `-v/--vis`.
- `tests/test_torch_free_training.py` passes 6/6; its scan's only exemptions are `TORCH_IMPORT_ALLOWLIST` = `genesis/utils/torch_compat.py` and `genesis/grad/bridge.py`, and the drone needs no allowance; its `TRAINING_PATH` globs match `hover_device_kernels.py` and `hover_device_rollout.py`.
- `examples/neuroevolution` imports `tensorneat`/`evoxc` only; the trainer adds no torch import of ours.
- Environment variables: `EVOXC_BACKEND`/`EVOXC_DEVICE` are honoured by `evox_runtime.resolve_backend_device`, `EVOXC_PJRT_PLUGIN` by the xla tell backend.

# Training entry point (`hover_neat_train.py`)
- Built on `NeuroevolutionTrainer(env, pop_size, generations, master_seed, num_inputs=17, num_outputs=4, ...)`; the trainer owns the ask/evaluate/fitness/tell loop, the seed streams, the float32 fitness aggregation and the artifacts.
- CLI: `--pop-size` (8), `--generations` (2), `--horizon` (full episode), `--episode-length-s` (15.0), `--master-seed` (0), `--perturb-base-pos`/`--perturb-base-rpy`/`--perturb-joint-pos` (0.05/0.05/0.0, joint inert for a body-only task) and `--no-perturbation`, `-e/--exp-name`, `-o/--output-dir` (`out/hover_neat`), `-g/--gpu`, `--evoxc-backend`, `--evoxc-device`, `--tell-backend {numpy,iree,xla}`, `--pjrt-plugin`, `--residency {numpy,device}`, `--stop-when-done/--no-stop-when-done`, `--torch-free`, `-v/--vis`, `--quiet`.
- `-g` initializes `gs.gpu` (performance mode) and defaults the graphs to `iree@rocm:0`; the CPU path is `numpy@cpu:0`.
- `--torch-free` implies `--residency device`, defaults the graphs to `iree@rocm:0` (an explicit `--evoxc-device` wins), initializes `gs.gpu` with `performance_mode=False`, and keeps the evolution on the rollout backend/device (`tell_backend=None` -> the env's own backend and device).
- `--torch-free` refuses `--residency numpy`, `--evoxc-backend numpy`, `--tell-backend numpy`, `--tell-backend xla`, `--pjrt-plugin` and `-v/--vis`, each with the reason in the message.
- `--torch-free` re-checks `"torch" in sys.modules` after scene and graph construction, after `trainer.setup()` and after the generations have run, and fails loud instead of reporting a claim that no longer holds.
- `--tell-backend` defaults are inherited from the shared trainer: a numpy env stays `numpy@cpu:0`, a ROCm/HIP env becomes `xla@rocm:0` (needs `--pjrt-plugin` or `EVOXC_PJRT_PLUGIN`), CUDA+iree keeps the reduction-distribution flag.
- Under `PYTEST_VERSION`, with no explicit flags, the run shrinks to `--pop-size 2 --generations 1 --horizon 20`; explicit values always win.
- Artifacts per run: `history.jsonl` (one JSON object per generation, flushed) and `summary.json` (resolved config with `torch_free: true`, `torch_in_process: false`, `residency`, tell backend/device, seed table, plus every record and the totals); the header line carries `torch_free=… torch_in_process=…`.
- The script carries no HSA re-exec helper: Genesis `gs.init` performs the torch/ROCm HSA runtime repair (`genesis.utils.hsa.ensure_system_hsa_runtime`).
- The `os._exit(0)` guard after the artifacts are written survives for an `xla` tell on a non-CPU device (a live PJRT client aborts at interpreter exit).

# Parity (measured)
- `hover_neat_check.py --parity` (graph-vs-oracle, 6 rows x 200 steps, `numpy@cpu:0`) prints `reward_target`, `reward_smooth`, `reward_angular`, `reward_crash`, `observation`, `rpm`, `clipped_actions` exactly 0; `reward_yaw` 1.673e-10; `reward_total` 3.576e-07; `returns` 5.960e-08; `yaw_deg` 6.104e-05 deg; every mask exactly 0 difference.
- `reset_obs_vs_obs` (analytic reset observation vs the live reset state) is exactly 0, `policy_row_identity` (rows `3p..3p+2` answering to policy `p`) is exactly 0, and `fingerprint_repeat` (the same generation evaluated twice in one process) is exactly 0.
- Deterministic replay: `--parity --pop-size 2 --horizon 30` prints `fingerprint	seeds=1454269175,757617120,1421746521	returns=0.045082949,-0.005940037,-0.015909065,-0.062878124,-0.002508280,0.017566836	sum=-0.024585718`; two independent processes give the identical line.
- Same-population device-vs-numpy: `returns`, `episode_returns` and `fitness` are EXACTLY 0.0 on `iree@cpu:0` (horizon 30) and on `iree@rocm:0` (horizon 30, with and without `--perturb`); `live_buffer_vs_getter` is 0.000e+00.
- Per-step 22-quantity contract (pop 8, 24 rows, horizon 25, `iree@cpu:0` vs `numpy@cpu:0`): `observation` 0, `reward_target`/`reward_smooth`/`reward_angular`/`reward_crash` 0, `reward_yaw` 7.276e-12, `reward_total` 1.164e-10, `commands`/`yaw_deg`/`base_pos` 0, and `crash`/`timeout`/`done`/`at_target`/`alive`/`episode_length`/`end_code` exactly equal; a 24/24-timeout scenario gives the same picture (`reward_total` 2.910e-11).
- After one population update from one shared generation-0 result on those same backends the four genome leaves are bitwise equal (matching NaN padding), randkey `-5246925798323494445` and generation exact, `returns` 1.118e-08 and `fitness` 3.725e-09.
- Trainer-level, each residency scoring its own population (pop 8, full episodes, `iree@rocm:0` vs `numpy@cpu:0`, torch blocked): generation 0 `returns` max |diff| 4.1282e-04 (4 of 24 rows > 1e-5, 2 > 1e-4, none > 1e-3, 18 of 24 <= 1e-6) and `fitness` 1.235e-04; generation 1 `returns` 3.5763e-07 and `fitness` 1.341e-07.
- `seeds`, `steps`, `env_steps` and `end_reasons` are exact in both generations, and `max_fitness` agrees to 6 significant digits (device `-0.14863242208957672`, numpy `-0.14863240718841553`).
- The generation-0 residual is a single-sample chaotic amplification of the 2.384e-07 setup drift by a 160-step crashing flight, so it is 2.5e-04-4.1e-04 across runs of the same setting; the same population scores exactly 0.
- `device_setup` vs `Pipeline.setup` at evolution_seed=3757552657: on `numpy@cpu:0` all ten float leaves are bitwise equal (0.0) and randkey 5923423680489327984 / generation / 4 integer leaves are exact; on `iree@rocm:0` the float leaves differ by 2.384e-07 (one float32 ulp at magnitude 2-4) on `pop.node_attrs`, `pop.conn_attrs`, `species[6]` and `species[8]`, while the other six float leaves, all integer leaves, randkey and the NaN padding masks are exact.
- The tell graph's first population update from one shared generation-0 result drifts by at most 2.384e-07 on those same four float leaves (randkey, generation, the 4 integer leaves and the NaN padding masks exact).
- `--residency-parity` (pop 2, horizon 30, steps 30/30) prints `returns_max_abs_diff 0.000e+00`, `fitness_max_abs_diff 0.000e+00`, `live_buffer_vs_getter 0.000e+00`, `end_reasons_equal 1`, `seeds_equal 1`, `residency_parity_ok 1` on `iree@cpu:0` and on `iree@rocm:0`, with and without `--perturb`.
- A device bool mask holding true rows gives the same `crash` / `alive` / `end_code` / `returns` leaves as a host bool mask with the same values.
- `hover_kernels.py` also self-checks standalone (`PYTHONPATH=$PWD python examples/drone/hover_kernels.py`): yaw 1.526e-05 deg max, obs 0, fixed-state reward 7.451e-09, obs-only 0, reset 0, randomized obs 0, randomized reward 1.490e-08, masks and the latched `end_code` agree on 200 states.

# Measured throughput
- Conditions of the record below: torch blocked (`PYTHONPATH=/tmp/notorch:$PWD`), MI300X VF / ROCm 7.1, `gs.gpu`, `iree@rocm:0` rollout and tell, master seed 0, full episodes (`max_episode_length=1500`, `stop_when_done`), default perturbation (base_pos 0.05 / base_rpy 0.05, joint inert), `performance_mode=False`.
- pop 8 (24 rows): setup and compile 8.951 s (tell build 7.331, rule 0.264, verification 0.324; 6 cache hits / 9 misses); generation 0 took 160 steps, 1.813 s, 88.3 steps/s, 2118 env-steps/s, best -0.148632 -> median -0.399830; generation 1 took 58 steps, 0.554 s, 104.7 steps/s, 2512 env-steps/s, best -0.169136 -> median -0.197512; crash 24/24; run totals 218 steps / 2.6 s / 5232 env-steps.
- pop 32 (96 rows): setup 8.362 s (tell build 6.757, rule 0.269, verification 0.334; 6 hits / 9 misses); generation 0 took 160 steps, 1.751 s, 91.4 steps/s, 8773 env-steps/s, best -0.148037 / median -0.293192; generation 1 took 149 steps, 1.183 s, 125.9 steps/s, 12089 env-steps/s, best -0.146592 / median -0.197399; crash 96/96; totals 309 steps / 3.3 s / 29664 env-steps.
- Quote these as ranges, not single figures: over cold-cache runs on this shared VF, setup 8.4-10.2 s, of which 6.8-8.5 s is IREE compiling the tell graph with no on-disk cache, and 88-126 steps/s at both sizes; a ~20 % run-to-run spread comes from host/GPU contention (the same command twice gave 88.3/104.7 then 110.2/122.7 steps/s).
- Env-steps scale about linearly with rows — 2.1k-2.9k env-steps/s at 24 rows and 8.8k-12.1k at 96 rows — while steps/s stay flat, so the per-step cost is dominated by the per-step host work, not by the row count.
- The host-staged baseline of the same runs is the numpy residency (`-g --residency numpy --tell-backend numpy`, 24 rows, full episodes): 40.3 / 78.9 steps/s and 968 / 1894 env-steps/s, so the device residency is ~2.2-2.8x the numpy rate; the CPU horizon-20 pop-2 check runs at ~22 steps/s.
- Determinism: two independent processes with the same master seed produce a field-identical `history.jsonl` (0 differing fields over `generation`, `seeds`, `episode_returns`, `returns`, `fitness`, `best_fitness`, `median_fitness`, `steps`, `env_steps`, `end_reasons`, cache counters and diagnostics); only `wall_time_s`/`steps_per_s`/`env_steps_per_s` differ.
- The trainer's record metrics are read back only AFTER the tell graph consumed the device tensors, and `self.pop_arrays = tell["pop"]` is the raw device output.

# Device placement (measured)
- Probed with the `examples/neuroevolution/device_runtime.py` helpers, torch blocked, `gs.gpu`, `iree@rocm:0`, pop 2, 2 generations of 84 then 87 steps.
- Every boundary is fully device-resident with `off_device: 0`: `initialize.pop` 4 leaves, `initialize.species` 10, `initialize.key` 2 (randkey int64 + generation), `policy.obs` 1, `policy.raw_action` 1, `evaluate.g0` 4 (returns/episode_returns/fitness/seeds), `evaluate.g1` 4, `advance.inputs` 14, `advance.outputs` 6, and `tell.g0`/`tell.g1` pop 4 / species 10 / key 2.
- Counters: `non_device_leaves 0` (empty name list), `payload_entering_gen1_is_tell_output 1`, `species_entering_gen1_is_tell_output 1`, `final_pop_is_last_tell_output 1`, `final_species_is_last_tell_output 1`, `tell_matches_env_device 1`.
- `"torch" in sys.modules` is false after the whole run.
- After `device_setup` the trainer's pop/species/randkey/generation already sit on `rocm:0`, and after the run all four result leaves do too; the generation-2 population and species are pointer-identical to the tell graph's own outputs.
- Per-step copies by design: 1 device-to-device staging copy of the rotor RPM through `DroneEntity.set_propellers_rpm` (fresh buffer per call, stream-ordered, no host wait), 1 host-to-device stage (`resampled_commands`), the int8 `end_code` device-to-host readback while `stop_when_done` is on, and the kernel-only `clear_errno()` per step; everything else is graph-to-graph pass-through or a zero-copy device import.

# Reproduce
- Every command needs `PYTHONPATH=$PWD`, or the editable `genesis` of the main checkout is imported and `--residency device` dies with `ModuleNotFoundError: No module named 'genesis.utils.device_buffer'`.
- Kernel self-check: `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_kernels.py`.
- Torch-free proof: `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --imports`.
- Numerical parity + fingerprint: `... examples/drone/hover_neat_check.py --parity --pop-size 2 --horizon 30` (add `--perturb` for the nonzero-perturbation replay).
- Residency parity: `... examples/drone/hover_neat_check.py --residency-parity --evoxc-backend iree --evoxc-device cpu:0 --pop-size 2 --horizon 30` (GPU: the same with `-g --evoxc-device rocm:0`, with and without `--perturb`).
- Environment smoke (collected by the example suite): `... examples/drone/hover_neat_check.py` (defaults `--pop-size 4 --generations 1 --horizon 20`).
- Torch-free check run: `... examples/drone/hover_neat_check.py --torch-free --pop-size 2 --horizon 5` (`iree@cpu:0` without `-g`, `iree@rocm:0` with it; prints `torch_free 1`).
- Torch-free numpy residency: `PYTHONPATH=/tmp/notorch:$PWD ... examples/drone/hover_neat_check.py --residency numpy --pop-size 2 --horizon 20` (`numpy@cpu:0`, `max_fitness=0.006757`) and `PYTHONPATH=/tmp/notorch:$PWD ... examples/drone/hover_neat_train.py --residency numpy --pop-size 8 --generations 2 -g --tell-backend numpy` (prints `torch_in_process=False`).
- Trainer smoke: `PYTHONPATH=$PWD PYTEST_VERSION=1 /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py`.
- Full-length CPU run: `... examples/drone/hover_neat_train.py --pop-size 8 --generations 2`.
- GPU runs: `... examples/drone/hover_neat_train.py -g --tell-backend numpy --pop-size 8 --generations 2`, the same with `--residency device`, and the same with `--residency numpy` for the host-staged baseline.
- Torch-free GPU run, torch present: `... examples/drone/hover_neat_train.py --torch-free --pop-size 8 --generations 2 -o out/hover_neat_pop8`; the same with `PYTHONPATH=/tmp/notorch:$PWD` proves the claim.
- GPU with an xla tell: add `--pjrt-plugin /path/to/<name>_plugin.so`; without one the ROCm default tell backend raises the xla adapter's "no PJRT C API plugin" error (fail-loud, see Gotchas).
- Check `rocm-smi --showpids` first: the MI300X is shared.

# Constraints
- Drone model is `urdf/drones/cf2x.urdf`; base hover RPM `14468.429183500699`; RPMs are clamped to safe ranges (0–25000 for teleop, 0.9×–1.5× base for PID).
- `interactive_drone.py`, `fly.py`, `fly_route.py`, and `hover_eval.py` run on `gs.cpu`; `hover_train.py` initializes `gs.gpu` and expects a GPU for its 8192 parallel environments.
- `hover_env.py` simulates at `dt=0.01` (100 Hz) with rigid substeps at `dt/2` and Newton constraint solver.
- `hover_env.py`, `hover_train.py` and `hover_eval.py` are the PPO/reference baseline and stay on torch + `tensordict`; the EvoX Compiler stack is validated against `hover_legacy_reference.py`, not against a live import of them (that module cannot be imported where `tensordict` is absent).
- `hover_kernels.py` and `hover_device_kernels.py` must stay free of torch and genesis; one graph per task concern, a `build_*` that bakes statics and a `run_*` that stages operands.
- `hover_neat_env.py` must not import torch; only NumPy bookkeeping and the Genesis boundary are allowed outside the graphs, and the device rollout is imported lazily inside the `residency == "device"` branch.
- `hover_device_rollout.py` must not import torch at all; its only bridges are `genesis.utils.device_buffer` (in) and `DroneEntity.set_propellers_rpm` (out), and the graph compute path uses no device-side staging.
- `hover_device_kernels.py` must never reshape or `np.*` a device value outside a graph; the returns reshape is the graph's whole purpose.
- `hover_train.py` and `hover_eval.py` enforce `rsl-rl-lib>=5.0.0` at import time and raise `ImportError` otherwise.
- Scripts use the `gs.morphs.Drone` morph and expect the URDF/mesh assets to be resolvable relative to the repo root.
- Files stay well under ~1000 lines, one module per concern; `hover_kernels.py` is the largest (963) because it carries four task graphs plus the parity self-check and must not absorb any more generic logic, and `hover_neat_check.py` is next (769) because it carries four modes under one CLI.

# Dependencies
- RL stack of the baseline: `rsl-rl-lib>=5.0.0`, `tensorboard` (training logger), and `tensordict` (observations); not installed in the EvoX Compiler work environment and not required by the migrated stack.
- Migrated stack: `evoxc` 0.1.0 (VCS install, not declared in `pyproject.toml`) and `tensorneat` 0.2.0; optional `iree-base-compiler`/`iree-base-runtime` for non-CPU `evoxc` targets, plus a PJRT plugin (`jax-rocm7-pjrt`) only for an `xla` tell on ROCm.
- `torch` is required by the PPO baseline and by the viewer; the Torch-free training path needs neither.
- Training throughput depends on a CUDA/ROCm GPU; eval, the smoke checks and the trainer's numpy path run on CPU.
- RL upstream reference: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv); reward design inspired by Champion-level drone racing (Nature 2023).

# Gotchas
- In this venv `import quadrants` itself puts `torch` in `sys.modules`, so an unblocked `hover_neat_check.py --torch-free` reports `torch_free 0` and, with the HSA teardown flake, exits 139 after printing; the claim is proven with `PYTHONPATH=/tmp/notorch:$PWD` (then `torch_free 1`, and the same fitness).
- Known flake: an occasional segfault at interpreter teardown AFTER all output printed (Genesis' torch/ROCm HSA re-exec on a contended card).
- On ROCm the trainer's default tell backend is `xla`, which needs a PJRT C API plugin; this host has none installed, so `-g` without `--pjrt-plugin` fails loudly at tell build time — use `--tell-backend numpy` (or `--torch-free`, which keeps the tell on IREE) to keep the evolution off XLA.
- The xla tell leaves a live PJRT client that aborts at interpreter exit, hence the `os._exit(0)` after the artifacts are written.
- `--perturb-joint-pos` is inert: the drone is a body with no joints, and the env draws the perturbation with `num_dof=0`.
- Teleop keys: arrow Up/Down/Left/Right = forward/back/left/right; Space = increase thrust; Left Shift = decrease thrust; Esc = quit.
- `interactive_drone.py` breaks its run loop immediately when `PYTEST_VERSION` is set, so it exits after one step under pytest.
- `hover_eval.py --record` captures video but is notably slow; the README and docstring recommend dropping `--record` for performance.
- Dead config/state in `hover_env.py`: `simulate_action_latency` is read from the config and never used; `_reward_yaw`'s `yaw > 180` wrap can never fire (Euler yaw ∈ [-180, 180]); the roll/pitch 180° termination terms are unreachable because Euler roll ≤ 180 and pitch ≤ 90.
- `hover_env.py` never calls `scene.get_error_envs_mask()` nor `scene.clear_errno()`, so a latched rigid-solver error re-raises on the next `scene.step()` instead of being absorbed by a reset.
- PID gains in `fly_route.py` are "tuned such that the drone can fly, not optimized".
- `hover_train.py` deletes any existing `logs/<exp-name>` directory before training.
- `fly_route.py` and `hover_eval.py` write recordings under `out/`, which must exist or be creatable.
- The parity table's yaw tolerance is 5e-4 deg: the residual is the float32 minimax `atan2` plus the opposite quaternion-composition side (`quat_mul(inv(q_init), q)` vs `quat_mul(q, inv(q_init))`, identity here), while `reward_yaw` itself agrees to 1.7e-10.
- `examples/neuroevolution/device_runtime.py`'s `device_setup` docstring claims its CPU path is bitwise identical to `Pipeline.setup()`; the `iree@rocm:0` drift is 2.384e-07 (see Parity), which a 160-step crashing flight amplifies.
- Device residency over a CPU Genesis is refused: the rollout exports a DLPack source on the graph's device and a CPU Genesis rejects it (`Only a host DLPack source can be copied into a CPU buffer`), so `--residency device`/`--residency-parity` with `--evoxc-device rocm:0`/`cuda:0` needs `-g`.

# Test Strategy
- `tests/test_examples.py` (pytest `examples` marker) runs `interactive_drone.py`, `fly.py`, `fly_route.py` and `hover_neat_check.py`; `*_train.py`, `*_eval.py`, `*_env.py`, `quadcopter_controller.py`, `hover_kernels.py`, `hover_legacy_reference.py` and `hover_device_rollout.py` are filtered out.
- `hover_neat_check.py` is collected by the same harness with an `evoxc`/`tensorneat` dependency skip and defaults to a small CPU run (`--pop-size 4 --generations 1 --horizon 20`) that exits 0 headless.
- `hover_device_kernels.py` is collected (no matching ignore pattern) and has no `EXAMPLE_DEPENDENCIES` entry, so that collection fails where `evoxc` is missing; it imports `evoxc` at module scope and has no `__main__`, so it exits 0 as a script where `evoxc` is present.
- `hover_neat_train.py` is ignored by the `*_train.py` pattern, so its pytest shortening only matters when it is invoked explicitly under `PYTEST_VERSION`.
- `tests/test_neuroevolution_*` cover the shared layer, and `tests/test_neuroevolution_device_parity.py` pins the Go2 device parity.
- `tests/test_hover_evox_parity.py` passes 7/7 (the four graphs and the oracle on the numpy backend, including the latched `end_code`) and `tests/test_hover_neat_seeding.py` 6/6 (seed triples, the supplied table, the zero-perturbation fixed reset, the policy-major slot mapping and the per-row command streams, without a simulator).
- `tests/test_hover_device_parity.py` passes 5/5 under `LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1.18.70100` (180.20 s); without the preload the three in-process cases skip on the HSA mapping and the two child-process cases pass (2 passed / 3 skipped).
- Its five cases: per-step observation and reward parity plus the termination/timeout masks against `hover_legacy_reference.step_maths`, the two-generation end-to-end reference numbers (seed table, `steps [160, 58]`, `end_reasons`, the documented `returns`/`fitness` drift), device-resident training with `torch` blocked in a child, and the real `hover_neat_train.py --torch-free` CLI with its `history.jsonl`/`summary.json` artifacts.
- `tests/test_torch_free_training.py` passes 6/6, and the three files together run 19 passed (264.55 s).
- A whole numpy-residency generation runs Torch-free: `--residency numpy --pop-size 2 --horizon 20` under `PYTHONPATH=/tmp/notorch:$PWD` scores `fitness=[0.006757, 0.000416]`, `max_fitness=0.006757`, byte-identical to the baseline, and a full-episode pop-8 GPU run under the blocked path gives the fitness history `[-0.14863241, -0.16913588]` with `torch_in_process=False`.
- The CLI refuses `--torch-free` together with `--residency numpy` by design, so the numpy proof is the trainer's `torch_in_process=False`, not a `torch_free 1` line.
- `hover_neat_check.py --torch-free --pop-size 2 --horizon 5` is green on `iree@cpu:0` and on `iree@rocm:0` (`fitness=[0.000281, 0.000389]`, `torch_free 1`), and `--imports` still prints `imports_ok 1`.
- pytest is not installed in the project venv (no pip, read-only root FS); the suites run as `PYTHONPATH=$PWD:/root/.local/share/pytest-target python -m pytest <targets> -n 0 -p no:cacheprovider` (pytest 9.2.0 + xdist + timeout from a uv target dir).
- Iteration loop for the migrated stack: kernel self-check, `--imports`, `--torch-free`, `--parity`, `--residency-parity`, the fingerprint twice in separate processes, a full-length CPU trainer run, then the Torch-free GPU run with and without torch blocked.
- RL scripts of the baseline are exercised manually, not by the example suite.

# See Also
- `urdf/drones/cf2x.urdf` — Crazyflie 2.X drone model asset used by all scripts here.
- `../neuroevolution/CONTEXT.md` — the shared layer this directory builds on (contract, seeding, graphs, trainer, artifacts).
- `../locomotion/CONTEXT.md` — the Go2 pilot, whose three device graphs are the template this directory reduced to one.
- `../../tests/test_examples.py` — the example harness and its ignore/dependency lists (sibling, read-only).
- `../../tests/test_hover_evox_parity.py`, `../../tests/test_hover_neat_seeding.py`, `../../tests/test_hover_device_parity.py`, `../../tests/test_torch_free_training.py` — the drone graph, seed, device-parity and Torch-free-scan tests (sibling, read-only).
- `../` — sibling example groups and the shared `genesis` framework entry points.
