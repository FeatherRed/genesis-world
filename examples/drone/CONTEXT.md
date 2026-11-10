# Intent
Demonstrates Crazyflie 2.X quadcopter simulation (`urdf/drones/cf2x.urdf`) across four interaction styles: keyboard teleop, scripted trajectory playback, PID point-to-point flight, and RL hover training.
Motors turn around a hover base RPM of `14468.429183500699`: through `Drone.set_propellers_rpm(...)` everywhere except the device-resident rollout, which calls the solver actuator the setter forwards to.
The RL hover task also exists in an EvoX Compiler (`evoxc`) form on the shared `examples/neuroevolution` layer: pure-`evoxc` task kernels, one device-only graph, a batched seeded evaluation environment, a training entry point, a NumPy parity oracle, and a smoke/parity CLI.
The evaluation environment has two residencies: the default NumPy-staged one and an opt-in, zero-copy device-resident rollout (`hover_device_rollout.py`).
The device residency is Torch-free, and `--torch-free` on the trainer and on the check CLI runs a whole training loop in a process where `import torch` fails.
The torch/PPO baseline (`hover_env.py`, `hover_train.py`, `hover_eval.py`) stays untouched and unmigrated; the EvoX Compiler stack is validated against its maths (transcribed in `hover_legacy_reference.py`), not against a live import of it.

# API Surface
- `interactive_drone.py` — keyboard-teleop demo; `DroneController` maps held arrow keys to differential rotor RPMs via Genesis keybindings.
- `fly.py` — pre-programmed flight; steps through a hardcoded `(N,4)` rotor-thrust trajectory array with `-v/--vis` flag.
- `fly_route.py` — point-to-point flight using `DronePIDController`; visits waypoints and records to `out/fly_route.mp4`.
- `quadcopter_controller.py` — `PIDController` (single-axis PID) and `DronePIDController` (cascaded position→velocity→attitude PID plus a 4-motor mixer).
- `hover_env.py` — `HoverEnv`, a vectorized rsl-rl environment; `step`/`reset`/`get_observations` return a `tensordict.TensorDict` observation; the reference baseline, not migrated.
- `hover_train.py` — PPO training via `rsl_rl.runners.OnPolicyRunner` on the GPU backend; the reference baseline, not migrated.
- `hover_eval.py` — loads a saved checkpoint, runs one inference rollout, optionally records to `out/hover.mp4`.
- `hover_legacy_reference.py` — the parity oracle: the pre-migration maths of `hover_env.py` as pure NumPy functions (`quat_to_xyz` incl. the gimbal-lock branch, `inv_quat`/`quat_mul`/`transform_quat_by_quat`/`transform_by_quat`, `propeller_rpm`, `body_frame_velocities`, `observation`, `reset_observation`, `reward_terms`, `crash_mask`, `timeout_mask`, `resample_commands`, `step_maths`); imports neither genesis, torch, tensordict nor evoxc.
- `hover_kernels.py` — the four drone task graphs (`hover_controls`, `hover_obs`, `hover_reset_obs`, `hover_advance`) plus the live `qd_atan2_fast` transcription and the `__main__` parity self-check; everything generic (seeding, policy graph, fitness, cache, staging helpers) is imported from `examples/neuroevolution`; imports neither torch nor genesis.
- `hover_device_kernels.py` — `hover_device_episode_returns` / `build_episode_returns(n_policies, *, episodes_per_policy=EPISODES_PER_POLICY, backend="numpy", device=None)` / `run_episode_returns(executable, returns)`: the `(rows,)` float32 returns → the policy-major `(pop_size, 3)` matrix as a real `evoxc.reshape` device op; imports neither genesis nor torch and has no `__main__`.
- `hover_neat_env.py` — `HoverNeatEnv`, the batched seeded evaluation environment implementing `env_base.BatchedEpisodeEnv` (`rows`, `episodes_per_policy`, `max_episode_length`, `cache`, `evaluate(pop_arrays, generation)`, `close()`), with `perturbation=seeding.Perturbation(...)` and `residency="numpy"|"device"`.
- `hover_device_rollout.py` — `DeviceRollout`, the `residency="device"` rollout: Torch-free, owns the per-step device tensors, the DLPack import of the live getters, the actuator bridge and the `(rows,)` → `(pop,3)` returns graph call.
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
- `reset_generation(generation, rngs)` writes the generation's pose with `drone.set_qpos(np.concatenate((pos, quat), axis=1), zero_velocity=True, envs_idx=None)`: the base pose of a free link *is* its qpos (measured bit-identical to `set_pos`/`set_quat` before and after stepping), and those two hit a Torch-free Genesis guard bug (see Gotchas).
- `reset_generation` then calls `zero_all_dofs_velocity()` and `scene.clear_errno()`, and returns the first command row plus the analytic reset observation from `run_reset_obs` (no step needed).
- Command resampling: one `np.random.Generator` per row seeded from `[episode-slot seed, generation]`, and every row burns exactly three uniforms per step (`x~U(-1,1)`, `y~U(-1,1)`, `z~U(1,1)` with `z` pinned to 1.0); the distribution matches the baseline's conditional draws while the stream stays row-order and step-order independent.
- `perturbation` defaults to an all-zero `seeding.Perturbation()`, which reproduces the fixed reset bit-for-bit (the `is_zero` short-circuit returns the exact tiled constants).
- A nonzero perturbation recomputes `_reset_pos`/`_reset_quat` per generation as `BASE_INIT_POS + offset` and `normalize(q_pert * q_init)`; row `r` uses episode slot `r % 3`, so every policy of a generation starts from the SAME three perturbed states (a paired comparison).
- NumPy staging is the correctness-first boundary for `residency="numpy"`: `get_pos`/`get_quat`/`get_vel`/`get_ang`/`get_error_envs_mask` go through `genesis.utils.misc.tensor_to_array`, and the setters (`set_propellers_rpm`, `set_qpos`) accept NumPy `array_like` directly.
- `executables()` returns the five graphs of `hover_kernels` (controls, reset observation, policy, advance, fitness) and, under `residency="device"`, the episode-returns graph of `hover_device_kernels` as a sixth.
- `scene.clear_errno()` runs once per generation reset and after any `get_error_envs_mask()` row, because a latched solver error re-raises on the next `scene.step()`.

# Device residency (`hover_device_rollout.py`)
- `DeviceRollout(env, executables=None)` with `evaluate(pop_arrays, generation)`, `state_parity()` and an idempotent `close()`; it returns exactly the env's result mapping.
- It rejects an env whose `backend` is not `iree`, and `HoverNeatEnv` rejects `residency="device"` on the numpy backend, so the zero-copy interop is IREE's DLPack bridge.
- Per step: policy graph -> `hover_controls` (clipped actions + rpm) -> actuator bridge -> `scene.step()` -> `qd.sync()` -> `hover_advance`; the state carried between steps (`commands`, `returns`, `episode_length`, `alive`, `end_code`, `last_actions`, `last_base_pos`, `obs`) never leaves the device.
- The generation reset stays the env's `reset_generation` (one host round trip: state write, commands, seeds), and the reset OBSERVATION is recomputed on device by `hover_reset_obs` from the same command and init-pose operands; the env's host observation is discarded.
- Results: `episode_returns` from `run_episode_returns`, `returns`, `fitness` from the compiled fitness graph and `seeds` as a device int64 tensor; `steps`, `env_steps`, `wall_time_s`, both rates, `cache_hits`/`cache_misses`, `end_reasons` and `commands` stay host logging values.
- `state_parity()` reads the four imported state operands back once and proves them bit-identical to `drone.get_pos/get_quat/get_vel/get_ang` (0.0 max abs diff).
- Solver reads are the live Torch-free getters: `get_pos`/`get_quat`/`get_ang` hand out a strided `DeviceView` of the batch-last solver field, compacted on device with `DeviceView.contiguous()` (IREE refuses a strided DLTensor), and `get_vel` already carries the `cd_vel + cd_ang × (pos − root_COM)` transport term the entity getter composes by hand.
- Zero-copy import: `capsule = view.__dlpack__()` -> `hal.from_dlpack_capsule(capsule)` -> `iree.runtime.DeviceArray(hal, view)` -> `evoxc.core.Tensor(IreeDevicePayload(...))`, the call form of `evoxc.backends.adapters.iree._device_array_from_dlpack`; the HAL device is taken from the advance executable's `backend_executable.runtime_device` with a fallback that remembers the first staged payload's.
- Graph output -> graph input passes through with no staging (`IreeExecutable.run` blocks on its own queue), which is how the raw action, the clipped action, the RPMs, the observation, `returns`/`alive`/`episode_length`/`end_code` and the returns handed to the fitness and episode-returns graphs travel.
- The **actuator bridge** bypasses `DroneEntity.set_propellers_rpm` because that setter cannot take any input without Torch (see Gotchas): `_write_rpm` fills a persistent batch-first `qd.ndarray(qd.f32, shape=(rows, drone.n_propellers))` device-to-device with `genesis.utils.device_buffer.copy_from_dlpack(self._rpm, source, transpose=False)`, then calls `drone.solver.set_drone_rpm(drone.propellers_idx, drone.KF, drone.KM, self._rpm, drone.propellers_spin, drone.model == "RACE")` — exactly what the entity setter forwards to.
- The source of that copy is the graph's rpm tensor, or its host mapping (`payload.to_host()`) on a CPU-kind IREE device (`iree@cpu:0`), which exports no DLPack and whose device *is* the host; that is the only host hop of the write path and only for that configuration.
- The bridge drops two things the entity setter does on top of that call, neither of which touches the physics: `drone._propellers_revs` is not advanced, so the propeller visual animation is frozen in device residency, and the once-per-step guard is bypassed because the loop writes once per `scene.step()` by construction.
- The physics of the bridge is bit-identical to the entity path: `get_pos`/`get_quat` agree to 0.0 after 5 steps of both.
- `qd.sync()` fences once per step — after `scene.step()`, before the advance graph imports the buffers Genesis just wrote — and once per generation after the reset write, before the reset observation and the first step read the state; `copy_from_dlpack` synchronizes around the rotor write itself because producer (IREE) and consumer (the Quadrants kernel of `scene.step()`) are on different streams.
- Remaining host touches per step, exhaustive: the `(rows,3)` host RNG command draw (by contract, staged as `resampled_commands`), `scene.get_error_envs_mask()` (Genesis exposes the solver-error latch as a host query only, re-staged as the `error_mask` operand, with `clear_errno()` whenever a row is flagged), and — while `stop_when_done` is on — the `(rows,)` int8 `end_code` readback that reproduces the numpy loop bound; the ending decision itself is in-graph.
- Remaining host touches per generation: the reset write plus seed/command staging, the four genome payload leaves (pass-through when the trainer hands device tensors), and at the end `end_code` (when `stop_when_done` is off), `commands` and the bookkeeping.
- Bool graph outputs cannot be DLPack-exported, so every exported leaf stays f32/i32; the int8 `end_code` only ever travels host-ward.
- Nothing in this module reshapes, slices or `np.*`-es a device value; the one shape change goes through `run_episode_returns`.

# Torch boundary
- `hover_kernels.py`, `hover_device_kernels.py`, `hover_device_rollout.py`, `hover_neat_env.py` and `hover_neat_train.py`, plus the six modules of `examples/neuroevolution`, contain no `import torch` / `from torch ...`; `hover_neat_check.py --imports` is the AST-scan proof over exactly those eleven files and prints `imports_ok 1`.
- `hover_device_rollout.py` imports no Torch at module scope or inside a helper, and calls no Torch operation: the two bridges are `genesis.utils.device_buffer` on the way in (a torch-free getter's `DeviceView` is its own DLPack producer) and `copy_from_dlpack` on the way out, so no Torch tensor exists anywhere on the path.
- `import genesis` is itself Torch-free on this path (`genesis.utils.torch_compat`), so the whole trainer runs with `import torch` failing; `--torch-free` proves it at runtime by refusing the run as soon as `"torch" in sys.modules`.
- `genesis.utils.misc.tensor_to_array` on the read path (numpy residency) is a device -> host copy required because the graphs are host-staged; `residency="device"` removes it.
- NumPy arrays passed to `set_propellers_rpm`/`set_qpos` (numpy residency) are one host -> device copy per write; `residency="device"` replaces the rotor write with the DLPack copy.
- Viewer and `performance_mode=True` are Torch consumers (the Quadrants ndarray staging), which is why `--torch-free` refuses `-v/--vis` and keeps the array mode off.
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
- Artifacts: `history.jsonl` (one JSON object per generation, flushed) + `summary.json` (resolved config, every record, totals), where the config carries `torch_free` and `torch_in_process` and the header line carries `torch_free=… torch_in_process=…`.
- The script carries no HSA re-exec helper: Genesis `gs.init` performs the torch/ROCm HSA runtime repair (`genesis.utils.hsa.ensure_system_hsa_runtime`).
- The `os._exit(0)` guard after the artifacts are written survives for an `xla` tell on a non-CPU device (a live PJRT client aborts at interpreter exit).

# Parity (measured)
- `hover_neat_check.py --parity` (graph-vs-oracle, 6 rows x 200 steps, `numpy`/`cpu:0`) prints `reward_target`, `reward_smooth`, `reward_angular`, `reward_crash`, `observation`, `rpm`, `clipped_actions` exactly 0; `reward_yaw` 1.673e-10; `reward_total` 3.576e-07; `returns` 5.960e-08; `yaw_deg` 6.104e-05 deg; every mask exactly 0 difference.
- `reset_obs_vs_obs` (analytic reset observation vs the live reset state) is exactly 0, `policy_row_identity` (rows `3p..3p+2` answering to policy `p`) is exactly 0, and `fingerprint_repeat` (the same generation evaluated twice in one process) is exactly 0.
- Deterministic replay: `--parity --pop-size 2 --horizon 30` prints `fingerprint	seeds=1454269175,757617120,1421746521	returns=0.045082949,-0.005940037,-0.015909065,-0.062878124,-0.002508280,0.017566836	sum=-0.024585718`; two independent processes give the identical line.
- `hover_neat_check.py --residency-parity` (pop 2, horizon 30: numpy env generation 0, then a fresh device env on the same seeds/pop/horizon/perturbation): `returns_max_abs_diff 0.000e+00`, `fitness_max_abs_diff 0.000e+00`, `live_buffer_vs_getter 0.000e+00`, `end_reasons_equal 1`, `seeds_equal 1` on `iree@cpu:0` and — with `-g` — on `iree@rocm:0`, with and without `--perturb`.
- Step-level parity (device rollout vs the numpy residency, both on `iree@cpu:0`, fixed seeds, generations 0 and 1): 22 of 22 quantities bit-exact (0.0) — observation (17), the five reward terms and the total, crash/timeout/done/at_target, commands (epoch resampling), episode_length, returns, alive, yaw_deg, end_code, base_pos, plus the operands (error_mask, clipped actions, last_base_pos).
- Pilot-level device-vs-numpy with the same seed and torch present, CPU Genesis both sides: when both residencies score the SAME population the rollout agrees exactly (`returns`/`episode_returns` 0.0, same `steps`, same `end_reasons`).
- End to end with the trainer, `seeds`/`steps`/`env_steps`/`end_reasons` are exact but `returns` differ by 2.022e-03 in generation 0 (two of 24 rows; the other 22 <= 3e-05) and 2.980e-07 in generation 1, fitness 6.785e-04 then 1.788e-07; the difference is upstream of the drone and is recorded in Gotchas.
- First population update from the same generation-0 result: `randkey` (5923423680489327984), `generation`, every integer leaf and the NaN padding mask are exact; float leaves agree to <= 2.980e-07 (`node_attrs` 2.384e-07, `conn_attrs` 2.980e-07, species leaf [1] 1.490e-08), i.e. the same tell graph executed by two float32 implementations.
- `hover_kernels.py` also self-checks standalone (`PYTHONPATH=$PWD python examples/drone/hover_kernels.py`): yaw 1.526e-05 deg max, obs 0, fixed-state reward 7.451e-09, obs-only 0, reset 0, randomized obs 0, randomized reward 1.490e-08, masks and the latched `end_code` agree on 200 states.

# Measured throughput
- Conditions of the record below: Torch blocked (`PYTHONPATH=/tmp/notorch:$PWD`), MI300X VF / ROCm 7.1, `gs.gpu`, `iree@rocm:0` rollout and tell, master seed 0, full episodes (`max_episode_length=1500`, `stop_when_done`), default perturbation (base_pos 0.05 / base_rpy 0.05, joint inert).
- pop 8 (24 rows): setup and compile 9.204 s (tell build 7.515, rule 0.287, verification 0.332; 6 cache hits / 9 misses); generation 0 took 160 steps, 1.598 s, 100.1 steps/s, 2403 env-steps/s; generation 1 took 58 steps, 0.394 s, 147.1 steps/s, 3531 env-steps/s; best fitness -0.148632 -> -0.169136, median -0.399830 -> -0.197512; `end_reasons` crash 24/24.
- pop 32 (96 rows): setup 8.762 s; generation 0 took 160 steps, 1.755 s, 91.2 steps/s, 8751 env-steps/s; generation 1 took 149 steps, 1.285 s, 115.9 steps/s, 11129 env-steps/s; best fitness -0.148037 -> -0.146592, median -0.293192 -> -0.197399; crash 96/96.
- Quote these as ranges, not single figures (host noise is ±10-15 % on wall time and rates): setup 8.4-9.2 s, of which 6.6-7.5 s is IREE compiling the tell graph with no on-disk cache, and ~90-115 steps/s at both sizes.
- Env-steps scale about linearly with rows — ~2.4k-3.5k env-steps/s at 24 rows and ~8.8k-11.1k at 96 rows — while steps/s stay flat, so the per-step cost is dominated by the per-step host work, not by the row count.
- Determinism: two independent processes with the same master seed produce a field-identical `history.jsonl` (0 differing fields over `generation`, `seeds`, `episode_returns`, `returns`, `fitness`, `best_fitness`, `median_fitness`, `steps`, `env_steps`, `end_reasons`, cache counters and diagnostics); only `wall_time_s`/`steps_per_s`/`env_steps_per_s` differ.
- Seeding: `rows = 3 * pop_size`, `episodes_per_policy = 3`, one ordered seed triple per generation shared by every policy and a fresh triple next generation — generation 0 `[1454269175, 757617120, 1421746521]`, generation 1 `[1313752747, 908599966, 53557616]`.
- Placement (probed with the `examples/neuroevolution/device_runtime.py` helpers, Torch-free, `non_device_leaves 0`, `payload_entering_gen1_is_tell_output 1`, `tell_matches_env_device 1`): every genome leaf, the species pytree (10 leaves), `randkey` and `generation` after `device_setup` are on `rocm:0`; the `(rows,17)` observation into the policy graph and the `(rows,4)` raw action out are on `rocm:0`; `returns`/`episode_returns`/`fitness`/`seeds`/`end_code`/`episode_length` are on `rocm:0`; the tell graph's payload and species outputs are on `rocm:0`; generation 1 receives the tell graph's own device leaves (pointer-identical).
- The trainer's record metrics are read back only AFTER the tell graph consumed the device tensors (`examples/neuroevolution/trainer.py`: `run_tell` at 579, `_build_record_device` at 588, `_to_numpy` at 713-716), and `self.pop_arrays = tell["pop"]` (604) is the raw device output.

# Reproduce
- Every command needs `PYTHONPATH=$PWD`, or the editable `genesis` of the main checkout is imported and `--residency device` dies on the missing `genesis.utils.device_buffer`.
- Kernel self-check: `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_kernels.py`.
- Torch-free proof: `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --imports`.
- Numerical parity + fingerprint: `... examples/drone/hover_neat_check.py --parity --pop-size 2 --horizon 30` (add `--perturb` for the nonzero-perturbation replay).
- Residency parity: `... examples/drone/hover_neat_check.py --residency-parity --evoxc-backend iree --evoxc-device cpu:0 --pop-size 2 --horizon 30` (GPU: the same with `-g --evoxc-device rocm:0`).
- Environment smoke (collected by the example suite): `... examples/drone/hover_neat_check.py` (defaults `--pop-size 4 --generations 1 --horizon 20`).
- Torch-free check run: `... examples/drone/hover_neat_check.py --torch-free --pop-size 2 --horizon 5` (`iree@cpu:0` without `-g`, `iree@rocm:0` with it; prints `torch_free 1`).
- Trainer smoke: `PYTHONPATH=$PWD PYTEST_VERSION=1 /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py`.
- Full-length CPU run: `... examples/drone/hover_neat_train.py --pop-size 8 --generations 2`.
- GPU runs: `... examples/drone/hover_neat_train.py -g --tell-backend numpy --pop-size 8 --generations 2` and the same with `--residency device`.
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
- `hover_device_rollout.py` must not import torch at all; its only bridges are `genesis.utils.device_buffer` (in) and `copy_from_dlpack` (out), and the graph compute path uses no device-side staging.
- `hover_device_kernels.py` must never reshape or `np.*` a device value outside a graph; the returns reshape is the graph's whole purpose.
- `hover_train.py` and `hover_eval.py` enforce `rsl-rl-lib>=5.0.0` at import time and raise `ImportError` otherwise.
- Scripts use the `gs.morphs.Drone` morph and expect the URDF/mesh assets to be resolvable relative to the repo root.
- Files stay well under ~1000 lines, one module per concern; `hover_kernels.py` is the largest (963) because it carries four task graphs plus the parity self-check and must not absorb any more generic logic, and `hover_neat_check.py` is next (769) because it carries four modes under one CLI.

# Dependencies
- RL stack of the baseline: `rsl-rl-lib>=5.0.0`, `tensorboard` (training logger), and `tensordict` (observations); not installed in the EvoX Compiler work environment and not required by the migrated stack.
- Migrated stack: `evoxc` 0.1.0 (VCS install, not declared in `pyproject.toml`) and `tensorneat` 0.2.0; optional `iree-base-compiler`/`iree-base-runtime` for non-CPU `evoxc` targets, plus a PJRT plugin (`jax-rocm7-pjrt`) only for an `xla` tell on ROCm.
- `torch` is required by the PPO baseline, by the viewer, and by the `performance_mode=True` staging; the Torch-free training path needs none of them.
- Training throughput depends on a CUDA/ROCm GPU; eval, the smoke checks and the trainer's numpy path run on CPU.
- RL upstream reference: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv); reward design inspired by Champion-level drone racing (Nature 2023).

# Gotchas
- `DroneEntity.set_propellers_rpm` is unusable without Torch for ANY input: a numpy array raises `AttributeError: 'ScalarNdarray' object has no attribute 'T'` (`genesis/engine/entities/rigid_entity/drone_entity.py:84`, `propellers_rpm.T`), and a device producer is rejected by the Torch-free sanitizer with `TypeError: float() argument must be a string or a real number, not 'Tensor'` (`_sanitize_io_variables` has no `is_dlpack_source` branch, and the setter has none either).
- Reproducer for that gap: build a `gs.morphs.Plane()` + `gs.morphs.Drone(file="urdf/drones/cf2x.urdf")` scene with 2 envs under `PYTHONPATH=/tmp/notorch:$PWD` and call `drone.set_propellers_rpm(np.full((2,4), 14468.43, np.float32))`; genesis needs an `is_dlpack_source` staged branch plus a Torch-free `_propellers_revs` update.
- That is why the numpy residency of this pilot cannot run Torch-free, and why the device residency owns the actuator bridge instead.
- Torch-free `RigidSolver.set_base_links_pos`/`set_base_links_quat` (`genesis/engine/solvers/rigid/rigid_solver.py`, the `has_fixed_verts` guard around 1982-1991 and 2019-2026) iterate the sanitized `links_idx` — a device `ScalarNdarray` — in Python and read past its end, so ANY env-specific pose write raises `Specifying env-specific pos for fixed links with at least one geometry requires setting morph option 'batch_fixed_verts=True'`.
- The drone owns no fixed link, so that morph option does not help; the workaround here is `set_qpos`, and genesis needs the guard evaluated on host indices.
- `examples/neuroevolution/device_runtime.py`'s `device_setup` docstring claims the CPU path is "bitwise identical to `Pipeline.setup()`"; measured at this HEAD the float leaves differ by up to 3.576e-07 (integer leaves exact, `node_attrs` 2.682e-07, `conn_attrs` 3.576e-07 at `evolution_seed=3757552657`), and a 160-step crashing flight amplifies that.
- That seed difference is the whole explanation of the trainer-level device-vs-numpy gap; when both residencies score the same population, and on the same returns in the fitness graph, the agreement is exact.
- `tests/test_examples.py` ignores `hover_kernels.py`, `hover_legacy_reference.py` and `hover_device_rollout.py`, but not `hover_device_kernels.py`: it imports `evoxc` and deliberately has no `__main__`, so the harness runs it and exits 0 — yet it has no `EXAMPLE_DEPENDENCIES` entry, so the run fails where `evoxc` is missing.
- `tests/test_torch_free_training.py`'s `TRAINING_PATH` glob `drone/hover_neat_*.py` does not cover `hover_device_kernels.py` or `hover_device_rollout.py`, so the static Torch-free scan misses the two files that matter most; those entries belong to the tests node.
- The drone graph tests are `tests/test_hover_evox_parity.py` (graph-vs-oracle, including the latched `end_code`) and `tests/test_hover_neat_seeding.py` (seed rules, slot mapping, command streams); no test covers the device residency, the state the rollout carries across a generation, or `hover_device_kernels.py`, and the Go2 model for those is `tests/test_go2_neat_kernels.py`/`test_go2_neat_seeding.py`.
- `tests/test_neuroevolution_device_parity.py` covers Go2 only.
- Every drone command needs `PYTHONPATH=$PWD`, otherwise the editable `genesis` of the main checkout is imported and `--residency device` dies with `ModuleNotFoundError: No module named 'genesis.utils.device_buffer'`.
- Known flake: an occasional segfault at interpreter teardown AFTER all output printed (Genesis' torch/ROCm HSA re-exec on a contended card).
- On ROCm the trainer's default tell backend is `xla`, which needs a PJRT C API plugin; this host has none installed, so `-g` without `--pjrt-plugin` fails loudly at tell build time — use `--tell-backend numpy` (or `--torch-free`, which keeps the tell on IREE) to keep the evolution off XLA.
- The xla tell leaves a live PJRT client that aborts at interpreter exit, hence the `os._exit(0)` after the artifacts are written.
- `--perturb-joint-pos` is inert: the drone is a body with no joints, and the env draws the perturbation with `num_dof=0`.
- README says `fly.py` "follows a predefined trajectory stored in `fly_traj.pkl`", but that file is not tracked and `fly.py` actually hardcodes the trajectory inline.
- Teleop keys: arrow Up/Down/Left/Right = forward/back/left/right; Space = increase thrust; Left Shift = decrease thrust; Esc = quit.
- `interactive_drone.py` breaks its run loop immediately when `PYTEST_VERSION` is set, so it exits after one step under pytest.
- `hover_eval.py --record` captures video but is notably slow; the README and docstring recommend dropping `--record` for performance.
- Dead config/state in `hover_env.py`: `simulate_action_latency` and `resampling_time_s` are read but never used; `_reward_yaw`'s `yaw > 180` wrap can never fire (Euler yaw ∈ [-180, 180]); the roll/pitch 180° termination terms are unreachable because Euler roll ≤ 180 and pitch ≤ 90.
- `hover_env.py` never calls `scene.get_error_envs_mask()` nor `scene.clear_errno()`, so a latched rigid-solver error re-raises on the next `scene.step()` instead of being absorbed by a reset.
- PID gains in `fly_route.py` are "tuned such that the drone can fly, not optimized".
- `hover_train.py` deletes any existing `logs/<exp-name>` directory before training.
- `fly_route.py` and `hover_eval.py` write recordings under `out/`, which must exist or be creatable.
- The parity table's yaw tolerance is 5e-4 deg: the residual is the float32 minimax `atan2` plus the opposite quaternion-composition side (`quat_mul(inv(q_init), q)` vs `quat_mul(q, inv(q_init))`, identity here), while `reward_yaw` itself agrees to 1.7e-10.
- Device residency over a CPU Genesis is refused: the rollout exports a DLPack source on the graph's device and a CPU Genesis rejects it (`Only a host DLPack source can be copied into a CPU buffer`), so `--residency device`/`--residency-parity` with `--evoxc-device rocm:0`/`cuda:0` needs `-g`.

# Test Strategy
- `tests/test_examples.py` (pytest `examples` marker) runs `interactive_drone.py`, `fly.py`, `fly_route.py` and `hover_neat_check.py`; `*_train.py`, `*_eval.py`, `*_env.py`, `quadcopter_controller.py`, `hover_kernels.py`, `hover_legacy_reference.py` and `hover_device_rollout.py` are filtered out.
- `hover_neat_check.py` is collected by the same harness with an `evoxc`/`tensorneat` dependency skip and defaults to a small CPU run (`--pop-size 4 --generations 1 --horizon 20`) that exits 0 headless.
- `hover_device_kernels.py` is collected (no matching ignore pattern and no dependency entry), imports `evoxc` at module scope and has no `__main__`, so it exits 0 as a script.
- `hover_neat_train.py` is ignored by the `*_train.py` pattern, so its pytest shortening only matters when it is invoked explicitly under `PYTEST_VERSION`.
- `tests/test_hover_evox_parity.py` drives the four graphs and the oracle on the numpy backend; `tests/test_hover_neat_seeding.py` pins the seed triples, the supplied table, the zero-perturbation fixed reset, the policy-major slot mapping and the per-row command streams without a simulator.
- `tests/test_torch_free_training.py` is the static Torch-free scan of the training path, and `tests/test_neuroevolution_*` cover the shared layer and the Go2 device parity.
- Iteration loop for the migrated stack: kernel self-check, `--imports`, `--torch-free`, `--parity`, `--residency-parity`, the fingerprint twice in separate processes, a full-length CPU trainer run, then the Torch-free GPU run with and without torch blocked.
- RL scripts of the baseline are exercised manually, not by the example suite.

# See Also
- `urdf/drones/cf2x.urdf` — Crazyflie 2.X drone model asset used by all scripts here.
- `../neuroevolution/CONTEXT.md` — the shared layer this directory builds on (contract, seeding, graphs, trainer, artifacts).
- `../locomotion/CONTEXT.md` — the Go2 pilot, whose three device graphs are the template this directory reduced to one.
- `../../tests/test_examples.py` — the example harness and its ignore/dependency lists (sibling, read-only).
- `../../tests/test_hover_evox_parity.py`, `../../tests/test_hover_neat_seeding.py` — the drone graph and seed tests (sibling, read-only).
- `../` — sibling example groups and the shared `genesis` framework entry points.
