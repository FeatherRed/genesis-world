# Intent
Demonstrates Crazyflie 2.X quadcopter simulation (`urdf/drones/cf2x.urdf`) across four interaction styles: keyboard teleop, scripted trajectory playback, PID point-to-point flight, and RL hover training.
All four motors are driven via `Drone.set_propellers_rpm(...)` around a hover base RPM of `14468.429183500699`.
The RL hover task also exists in an EvoX Compiler (`evoxc`) form on the shared `examples/neuroevolution` layer: a pure-`evoxc` task-kernel module, a batched seeded evaluation environment, a training entry point, a NumPy parity oracle, and a smoke/parity CLI.
The evaluation environment has two residencies: the default NumPy-staged one and an opt-in, zero-copy device-resident rollout (`hover_device_rollout.py`).
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
- `hover_neat_env.py` — `HoverNeatEnv`, the batched seeded evaluation environment implementing `env_base.BatchedEpisodeEnv` (`rows`, `episodes_per_policy`, `max_episode_length`, `cache`, `evaluate(pop_arrays, generation)`, `close()`), with `perturbation=seeding.Perturbation(...)` and `residency="numpy"|"device"`.
- `hover_device_rollout.py` — `DeviceRollout`, the `residency="device"` rollout: owns the per-step device tensors and the DLPack bridges in both directions; imports torch lazily inside its staging/export helpers only.
- `hover_neat_train.py` — the drone's training entry point on `trainer.NeuroevolutionTrainer` (CLI, `history.jsonl` + `summary.json`, one log line per generation).
- `hover_neat_check.py` — the environment-level smoke/parity CLI (`--parity`, `--residency-parity`, `--imports`, `--perturb`, `--residency`); also the collected example smoke for the migrated stack.
- `README.md` — run commands, control mappings, and installation notes.

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
- All base state reads (`get_pos`, `get_quat`, `get_vel`, `get_ang`) return `(n_envs, 3|4)` device tensors; velocities are world-frame and are rotated into the body frame via `transform_by_quat(v, inv_quat(base_quat))`.

# Task graphs (`hover_kernels.py`)
- Constants mirror `hover_train.get_cfgs()`: `NUM_OBS=17`, `NUM_ACTIONS=4`, `NUM_COMMANDS=3`, `DT=0.01`, `MAX_EPISODE_LENGTH=1500`, `CLIP_ACTIONS=1.0`, `HOVER_RPM=14468.429183500699`, `ACTION_GAIN=0.8`, `AT_TARGET_THRESHOLD=0.1`, `OBS_SCALES`, `REWARD_SCALES` (pre-`dt`), `REWARD_TERM_NAMES`, `YAW_LAMBDA=-10`, `TERMINATION`, `BASE_INIT_POS`, `BASE_INIT_QUAT`, `QUAT_EPS` (the float32 `gs.EPS`), `END_REASONS`.
- `hover_controls` / `build_controls(rows)` / `run_controls` — runtime raw `actions (rows,4)` -> clipped actions and `rpm (rows,4)`; separate from `hover_advance` because the RPM must be written before `scene.step()` while every other quantity is computed after it.
- `hover_obs` / `build_obs` / `run_obs` — `commands`, `base_pos`, absolute `base_quat`, world-frame lin/ang velocity, `last_actions` -> `(rows,17)`; the body-frame rotation is `_rotate_rows(v, _inv_quat_rows(base_quat))` on the raw, unnormalized quaternion.
- `hover_reset_obs` / `build_reset_obs(rows)` / `run_reset_obs(executable, *, commands, base_init_pos, base_init_quat)` — the analytic post-reset observation, with NO `scene.step()`; the init pose is a runtime operand `(rows,3)`/`(rows,4)` so a perturbed reset is supported.
- `hover_advance` / `build_advance(rows, max_episode_length)` / `run_advance` — one post-step graph: episode counter, `rel_pos`/`last_rel_pos`, body velocities, yaw in degrees, `at_target`, next `commands`, the five `dt`-scaled reward terms, `crash`/`timeout`/`done`, next `obs`, `returns`, `alive`, `yaw_deg`, the latched `end_code`, and a `base_pos` snapshot (a fresh buffer for the next `last_base_pos`).
- `end_code` (int8, 0 unfinished / 1 timeout / 2 crash / 3 error) is a stateful input+output: the graph latches the first ending with the rule error > timeout > crash, once, never re-armed; the env derives `end_reasons` from it instead of re-classifying on the host, so the rule has exactly one owner.
- Static specializations: scales, thresholds, `dt`, `max_episode_length` and row count are baked into each graph; only genuinely per-step values appear in a `run_*` signature.
- `hover_advance` mirrors the baseline order, including that the target term uses the command resampled on arrival (against both `rel_pos` and `last_rel_pos`) and that the observation's last slot is the previous step's action.
- Yaw is the `qd_atan2_fast` polynomial of `genesis/utils/geom.py` transcribed with `abs`/`min`/`max`/`sqrt`/`select` only (there is no `atan2` in `evoxc`), plus the `quat_to_xyz` gimbal-lock select; measured error is 1 float32 ulp (1.526e-05 deg near ±180°).
- Dropped as unreachable, and documented as such: the roll/pitch 180° termination terms (roll ≤ 180°, pitch ≤ 90°) and `_reward_yaw`'s `yaw > 180` unwrap.

# Shared-layer adoption
- `hover_kernels.py` puts the sibling `examples/neuroevolution` directory on `sys.path` once (documented shim; the example stack resolves bare module names this way), then imports `EPISODES_PER_POLICY` from `seeding` and `POP_OPERANDS`/`_attach`/`_constant`/`_device_name`/`_outputs`/`_positive_int`/`_staged`/`_to_device_array`/`_to_numpy`/`resolve_backend_device` from `evox_runtime`.
- `hover_neat_env.py` imports `build_policy`/`run_policy` from `policy_graphs`, `ExecutableCache`/`build_fitness`/`run_fitness` from `evolution_graphs`, `Perturbation`/`EPISODES_PER_POLICY`/`generation_seed_table`/`perturbation_from_seed` from `seeding`, `BatchedEpisodeEnv`/`validate_evaluation` from `env_base`, and `TraceOnlyProblem` from `trainer`.
- Nothing drone-specific duplicates the shared layer any more: no local `generation_seed_table`/`evolution_seed`/`EPISODES_PER_POLICY`/`build_policy`/`run_policy`/`build_fitness`/`run_fitness`/`ExecutableCache`, no second policy graph, no hand-rolled generation loop.
- `examples/locomotion` is no longer imported by this directory.

# Environment contract (`hover_neat_env.py`)
- `HoverNeatEnv(pop_size, master_seed, *, horizon=None, episode_length_s=None, backend="numpy", device=None, show_viewer=False, seed_table=None, stop_when_done=True, perturbation=None, residency="numpy")`.
- Attributes: `rows = n_envs = 3 * pop_size` (policy-major: rows `3p, 3p+1, 3p+2` are the three episodes of policy `p`), `episodes_per_policy = 3`, `max_episode_length = ceil(episode_length_s / 0.01) = 1500`, `horizon` (default `max_episode_length`), `cache`, `residency`, `executables()`.
- `evaluate(pop_arrays, generation)` returns `seeds (3,)`, `episode_returns (pop_size,3)`, `returns (rows,)`, `fitness (pop_size,)`, `steps`, `env_steps`, `wall_time_s`, `steps_per_s`, `env_steps_per_s`, `cache_hits`, `cache_misses`, `end_reasons`, `commands`, and validates the mapping with `env_base.validate_evaluation` before returning.
- `__init__` asserts `isinstance(self, BatchedEpisodeEnv)` and names any missing contract attribute, which is the declaration that the env satisfies the shared protocol.
- One episode per row per generation: a dead row is never reset inside a generation, so the baseline's reset-step reward pass is not reproduced; a truncated PPO episode consequently carries one extra post-reset step that this evaluator does not.
- Scene is the baseline's physics: `SimOptions(dt=0.01)`, `RigidOptions(dt=0.005, enable_collision=True, enable_joint_limit=True, constraint_solver=Newton)`, a `Plane` and `gs.morphs.Drone(file="urdf/drones/cf2x.urdf")`, built with `n_envs=rows`.
- `gs.init(backend=gs.cpu, precision="32", logging_level="warning")` runs only when genesis is not initialized, so a caller that already did `gs.init(backend=gs.gpu, ...)` keeps its backend.
- Command resampling: one `np.random.Generator` per row seeded from `[episode-slot seed, generation]`, and every row burns exactly three uniforms per step (`x~U(-1,1)`, `y~U(-1,1)`, `z~U(1,1)` with `z` pinned to 1.0); the distribution matches the baseline's conditional draws while the stream stays row-order and step-order independent.
- `perturbation` defaults to an all-zero `seeding.Perturbation()`, which reproduces the fixed reset bit-for-bit (the `is_zero` short-circuit returns the exact tiled constants).
- A nonzero perturbation recomputes `_reset_pos`/`_reset_quat` per generation as `BASE_INIT_POS + offset` and `normalize(q_pert * q_init)`; row `r` uses episode slot `r % 3`, so every policy of a generation starts from the SAME three perturbed states (a paired comparison).
- NumPy staging is the correctness-first boundary for `residency="numpy"`: `get_pos`/`get_quat`/`get_vel`/`get_ang`/`get_error_envs_mask` go through `genesis.utils.misc.tensor_to_array`, and the setters (`set_propellers_rpm`, `set_pos`, `set_quat`) accept NumPy `array_like` directly.
- `scene.clear_errno()` runs once per generation reset and after any `get_error_envs_mask()` row, because a latched solver error re-raises on the next `scene.step()`.

# Device residency (`hover_device_rollout.py`)
- `DeviceRollout(env)` with `evaluate(pop_arrays, generation)` and an idempotent `close()`; it returns exactly the env's result mapping and runs the same `validate_evaluation` check.
- Per generation: one host round-trip reuses `env.reset_generation(...)`, uploads the `(rows,17)` observation and the four genome payload leaves; per step all operands stay device tensors, and the state carried between steps (`commands`, `returns`, `episode_length`, `alive`, `end_code`, `last_actions`, `last_base_pos`, `obs`) never leaves the device.
- The only per-generation host materialization is `returns`, `fitness` (from the compiled fitness graph on the device `returns`), `end_code` and `commands`.
- Per-step host work is limited to the host RNG command draws (by contract), the `scene.get_error_envs_mask()` latch (a host-only query in Genesis), and — only while `stop_when_done` is on — the `(rows,)` `end_code` read that reproduces the numpy loop bound.
- Solver reads are the live `scene.sim.rigid_solver.dyn_state.links` fields at `env.drone.base_link_idx`: `pos`, `quat`, `cd_ang`, and `cd_vel + cd_ang × (pos − root_COM)` (the transport term of `RigidSolver.get_links_vel`, zero for cf2x because its propellers are massless but computed anyway); `state_parity()` proves them bit-identical to `drone.get_pos/get_quat/get_vel/get_ang` (0.0 max abs diff).
- Zero-copy quadrants -> IREE: `qd_to_torch(field, copy=False)` -> `torch.utils.dlpack.to_dlpack` -> `runtime_device.from_dlpack_capsule(cap)` -> `iree.runtime.DeviceArray(runtime_device, view)` -> `evoxc.Tensor(IreeDevicePayload(arr, evoxc.Device(kind, index)))`; the torch view is kept alive and the imported tensors are cached by field identity, invalidated on every generation reset.
- DLPack import requires 64-byte (`local-task`) / 256-byte (HIP) alignment while a link slice is misaligned (`pos` of the base link starts 12 bytes into the field on CPU), so `_read_state` `copy_`s each field into a persistent aligned device staging buffer once per step and imports that buffer.
- Zero-copy IREE -> Genesis: `IreeDevicePayload.__dlpack__` exports the SAME HAL buffer, so `torch.from_dlpack(rpm)` aliases IREE memory and is passed straight to `drone.set_propellers_rpm`; `IreeExecutable.run` blocks on its queue, and exactly one `torch.cuda.synchronize()` guards the export -> setter hand-off (the Genesis/qd kernel runs on torch's stream).
- Bool graph outputs cannot be DLPack-exported, so every exported leaf stays f32/i32; the int8 `end_code` only ever travels host-ward via `.to(cpu)`/`.numpy()`.
- `residency="device"` requires a device `evoxc` backend (`iree`); `DeviceRollout` rejects anything else, and the env never imports the module in numpy mode.

# Torch boundary (the migration contract)
- `hover_kernels.py`, `hover_neat_env.py`, `hover_neat_train.py` and all six `examples/neuroevolution` modules contain no `import torch` / `from torch ...`; `hover_neat_check.py --imports` is the AST-scan proof and prints `imports_ok 1`.
- `import genesis` still pulls torch into the process through `quadrants`, which is why the pid stack is only "torch-free by our code, not by the process"; nothing can remove that short of a torch-free quadrants build.
- `genesis.utils.misc.tensor_to_array` on the read path (numpy residency) is a device -> host copy required because the graphs are host-staged; `residency="device"` removes it.
- NumPy arrays passed to `set_propellers_rpm`/`set_pos`/`set_quat` (numpy residency) are one host -> device copy per write; `residency="device"` replaces it with the `torch.from_dlpack` alias.
- `hover_device_rollout.py` may `import torch` only inside its staging/export helpers, for two reasons: producing a DLPack capsule from a quadrants buffer (`qd_to_torch` + `torch.utils.dlpack.to_dlpack`, because quadrants vec3 fields have no `to_dlpack`) and consuming IREE's exported capsule (`torch.from_dlpack`) so Genesis's setters can write through it. A quadrants-side capsule producer and a Genesis setter that accepts a foreign device pointer would remove both.
- `examples/neuroevolution` imports `tensorneat`/`evoxc` only; the trainer adds no torch import of ours.
- Environment variables: `EVOXC_BACKEND`/`EVOXC_DEVICE` are honoured by `evox_runtime.resolve_backend_device`, `EVOXC_PJRT_PLUGIN` by the xla tell backend.

# Training entry point (`hover_neat_train.py`)
- Built on `NeuroevolutionTrainer(env, pop_size, generations, master_seed, num_inputs=17, num_outputs=4, ...)`; the trainer owns the ask/evaluate/fitness/tell loop, the seed streams, the float32 fitness aggregation and the artifacts.
- CLI: `--pop-size` (8), `--generations` (2), `--horizon` (full episode), `--episode-length-s` (15.0), `--master-seed` (0), `--perturb-base-pos`/`--perturb-base-rpy`/`--perturb-joint-pos` (0.05/0.05/0.0, joint inert for a body-only task) and `--no-perturbation`, `-e/--exp-name`, `-o/--output-dir` (`out/hover_neat`), `-g/--gpu`, `--evoxc-backend`, `--evoxc-device`, `--tell-backend {numpy,iree,xla}`, `--pjrt-plugin`, `--residency {numpy,device}`, `--stop-when-done/--no-stop-when-done`, `-v/--vis`, `--quiet`.
- `-g` initializes `gs.gpu` (performance mode) and defaults the graphs to `iree@rocm:0`; the CPU path is `numpy@cpu:0`.
- `--tell-backend` defaults are inherited from the shared trainer: a numpy env stays `numpy@cpu:0`, a ROCm/HIP env becomes `xla@rocm:0` (needs `--pjrt-plugin` or `EVOXC_PJRT_PLUGIN`), CUDA+iree keeps the reduction-distribution flag.
- Under `PYTEST_VERSION`, with no explicit flags, the run shrinks to `--pop-size 2 --generations 1 --horizon 20`; explicit values always win.
- Artifacts: `history.jsonl` (one JSON object per generation, flushed) + `summary.json` (resolved config, every record, totals); the trainer logs one line per generation.
- The script carries no HSA re-exec helper: Genesis `gs.init` now performs the torch/ROCm HSA runtime repair (`genesis.utils.hsa.ensure_system_hsa_runtime`).
- The `os._exit(0)` guard after the artifacts are written survives for an `xla` tell on a non-CPU device (a live PJRT client aborts at interpreter exit).

# Parity (measured)
- `hover_neat_check.py --parity` (graph-vs-oracle, 6 rows x 200 steps, `numpy`/`cpu:0`) prints, after the shared-layer extraction: `reward_target`, `reward_smooth`, `reward_angular`, `reward_crash`, `observation`, `rpm`, `clipped_actions` exactly 0; `reward_yaw` 1.673e-10; `reward_total` 3.576e-07; `returns` 5.960e-08; `yaw_deg` 6.104e-05 deg; every mask exactly 0 difference.
- `reset_obs_vs_obs` (analytic reset observation vs the live reset state) is exactly 0, `policy_row_identity` (rows `3p..3p+2` answering to policy `p`) is exactly 0, and `fingerprint_repeat` (the same generation evaluated twice in one process) is exactly 0.
- Deterministic replay: `--parity --pop-size 2 --horizon 30` prints `fingerprint	seeds=1454269175,757617120,1421746521	returns=0.045082949,-0.005940037,-0.015909065,-0.062878124,-0.002508280,0.017566836	sum=-0.024585718`; two independent processes give the identical line.
- `hover_neat_check.py --residency-parity` (numpy env generation 0, then a fresh device env on the same seeds/pop/horizon/perturbation): `returns_max_abs_diff 0.000e+00`, `fitness_max_abs_diff 0.000e+00`, `end_reasons_equal 1`, `live_buffer_vs_getter 0.000e+00`, `seeds_equal 1` on `iree@cpu:0` and on `iree@rocm:0`.
- Full-length trainer runs (`--pop-size 8 --generations 2`, 24 rows, `max_episode_length=1500`): `numpy` residency and `residency="device"` agree to exactly 0.0 on every `returns` and `fitness` value, with identical `steps` (160 then 58 for the untrained population) and identical `end_reasons`; CPU vs GPU physics differ by 1.1e-3 max abs return (a different physics backend, not the residency) with the same `end_reasons`.
- Two independent processes with the same `--master-seed` produce identical `history.jsonl` `seeds`/`returns`/`fitness`/`end_reasons` records.
- `hover_kernels.py` also self-checks standalone (`PYTHONPATH=$PWD python examples/drone/hover_kernels.py`): yaw 1.526e-05 deg max, obs 0, fixed-state reward 7.451e-09, obs-only 0, reset 0, randomized obs 0, randomized reward 1.490e-08, masks and the latched `end_code` agree on 200 states.

# Reproduce
- Kernel self-check: `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_kernels.py`.
- Torch-free proof: `PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --imports`.
- Numerical parity + fingerprint: `... examples/drone/hover_neat_check.py --parity --pop-size 2 --horizon 30` (add `--perturb` for the nonzero-perturbation replay).
- Residency parity: `... examples/drone/hover_neat_check.py --residency-parity --evoxc-backend iree --evoxc-device cpu:0 --pop-size 2 --horizon 30` (GPU: the same with `-g --evoxc-device rocm:0`).
- Environment smoke (collected by the example suite): `... examples/drone/hover_neat_check.py` (defaults `--pop-size 4 --generations 1 --horizon 20`).
- Trainer smoke: `PYTHONPATH=$PWD PYTEST_VERSION=1 /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py`.
- Full-length CPU run: `... examples/drone/hover_neat_train.py --pop-size 8 --generations 2`.
- GPU runs: `... examples/drone/hover_neat_train.py -g --tell-backend numpy --pop-size 8 --generations 2` and the same with `--residency device` (both reproduce each other exactly).
- GPU with an xla tell: add `--pjrt-plugin /path/to/<name>_plugin.so`; without one the ROCm default tell backend raises the xla adapter's "no PJRT C API plugin" error (fail-loud, see Gotchas).
- Check `rocm-smi --showpids` first: the MI300X is shared.

# Measured throughput
- Deliberately not quoted here: the host is shared and the numbers recorded before the shared-layer migration and the device rollout no longer describe this stack.
- A central measurement round owns the throughput numbers for both residencies and both tell backends; re-measure before quoting any steps/s.

# Constraints
- Drone model is `urdf/drones/cf2x.urdf`; base hover RPM `14468.429183500699`; RPMs are clamped to safe ranges (0–25000 for teleop, 0.9×–1.5× base for PID).
- `interactive_drone.py`, `fly.py`, `fly_route.py`, and `hover_eval.py` run on `gs.cpu`; `hover_train.py` initializes `gs.gpu` and expects a GPU for its 8192 parallel environments.
- `hover_env.py` simulates at `dt=0.01` (100 Hz) with rigid substeps at `dt/2` and Newton constraint solver.
- `hover_env.py`, `hover_train.py` and `hover_eval.py` are the PPO/reference baseline and stay on torch + `tensordict`; the EvoX Compiler stack is validated against `hover_legacy_reference.py`, not against a live import of them (that module cannot be imported where `tensordict` is absent).
- `hover_kernels.py` must stay free of torch and genesis; one graph per task concern, a `build_*` that bakes statics and a `run_*` that stages operands.
- `hover_neat_env.py` must not import torch; only NumPy bookkeeping and the Genesis boundary are allowed outside the graphs, and the device rollout is imported lazily inside the `residency == "device"` branch.
- `hover_device_rollout.py` keeps `import torch` inside functions, only for the two DLPack bridges; the graph compute path never uses torch.
- `hover_train.py` and `hover_eval.py` enforce `rsl-rl-lib>=5.0.0` at import time and raise `ImportError` otherwise.
- Scripts use the `gs.morphs.Drone` morph and expect the URDF/mesh assets to be resolvable relative to the repo root.
- Files stay well under ~1000 lines, one module per concern; `hover_kernels.py` is the largest (~960) because it carries four task graphs plus the parity self-check and must not absorb any more generic logic.

# Dependencies
- RL stack of the baseline: `rsl-rl-lib>=5.0.0`, `tensorboard` (training logger), and `tensordict` (observations); not installed in the EvoX Compiler work environment and not required by the migrated stack.
- Migrated stack: `evoxc` 0.1.0 (VCS install, not declared in `pyproject.toml`) and `tensorneat` 0.2.0; optional `iree-base-compiler`/`iree-base-runtime` for non-CPU `evoxc` targets, plus a PJRT plugin (`jax-rocm7-pjrt`) only for an `xla` tell on ROCm.
- `torch` is required by Genesis itself and by the device rollout's DLPack bridges; the drone graph modules do not import it.
- Training throughput depends on a CUDA/ROCm GPU; eval, the smoke checks and the trainer's numpy path run on CPU.
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
- The parity table's yaw tolerance is 5e-4 deg: the residual is the float32 minimax `atan2` plus the opposite quaternion-composition side (`quat_mul(inv(q_init), q)` vs `quat_mul(q, inv(q_init))`, identity here), while `reward_yaw` itself agrees to 1.7e-10.
- `tests/test_examples.py`'s `IGNORE_SCRIPT_PATTERNS` lists none of `hover_kernels.py`, `hover_legacy_reference.py`, `hover_device_rollout.py`, so the harness runs the kernel `__main__` self-check and merely imports the other two; those entries and an `EXAMPLE_DEPENDENCIES` entry for `evoxc` belong to the tests node.
- On ROCm the trainer's default tell backend is `xla`, which needs a PJRT C API plugin; this host has none installed, so `-g` without `--pjrt-plugin` fails loudly at tell build time — use `--tell-backend numpy` to keep the evolution on the host while the rollout runs on GPU.
- The xla tell leaves a live PJRT client that aborts at interpreter exit, hence the `os._exit(0)` after the artifacts are written.
- `--perturb-joint-pos` is inert: the drone is a body with no joints, and the env draws the perturbation with `num_dof=0`.

# Test Strategy
- `tests/test_examples.py` (pytest `examples` marker) runs `interactive_drone.py`, `fly.py` and `fly_route.py`; `*_train.py`, `*_eval.py`, `*_env.py`, and `quadcopter_controller.py` are ignored by its filter.
- `hover_neat_check.py` is collected by the same harness and therefore defaults to a small CPU run (`--pop-size 4 --generations 1 --horizon 20`) that exits 0 headless.
- `hover_neat_train.py` is ignored by the `*_train.py` pattern, so its pytest shortening only matters when it is invoked explicitly under `PYTEST_VERSION`.
- No `tests/` file covers this directory yet; `tests/test_go2_neat_kernels.py`/`test_go2_neat_seeding.py` are the model for `tests/test_hover_neat_kernels.py`/`test_hover_neat_seeding.py` (drone graph parity, the seed rules, the `end_code` latch, the residency equivalence).
- Iteration loop for the migrated stack: kernel self-check, `--imports`, `--parity`, `--residency-parity`, the fingerprint twice in separate processes, a full-length CPU trainer run, then the GPU pair.
- RL scripts of the baseline are exercised manually, not by the example suite.

# See Also
- `urdf/drones/cf2x.urdf` — Crazyflie 2.X drone model asset used by all scripts here.
- `../neuroevolution/CONTEXT.md` — the shared layer this directory builds on (contract, seeding, graphs, trainer, artifacts).
- `../locomotion/CONTEXT.md` — the Go2 pilot and the MI300X/HSA facts (sibling, read-only).
- `../../tests/test_examples.py` — the example harness and its ignore/dependency lists (sibling, read-only).
- `../` — sibling example groups and the shared `genesis` framework entry points.
