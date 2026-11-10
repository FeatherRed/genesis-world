# examples/neuroevolution

## Intent
The shared, task-agnostic neuroevolution layer of the example tree: one trainer, one environment contract, one seeding rule set and one copy of each compiled graph, used by every TensorNEAT/`evoxc` task instead of per-task copies.
A task contributes only its simulator env (`examples/locomotion/go2_neat_env.py`, `examples/drone/hover_neat_env.py`) and its own task graphs; the ask -> rollout -> fitness -> tell loop, the artifact format, the reproducibility rules and the measurement harness live here.
Nothing in this directory imports `torch`, `genesis` or `tensordict`, so the whole layer is testable without a simulator or a GPU.

## API Surface

### `seeding.py`
`EPISODES_PER_POLICY = 3`; `Perturbation(base_pos=0.0, base_rpy=0.0, joint_pos=0.0)` frozen dataclass with `is_zero`; `generation_seed_table(master_seed, num_generations, episodes=EPISODES_PER_POLICY) -> (G, episodes) int64`; `evolution_seed(master_seed) -> int`; `perturbation_from_seed(seed, perturbation, num_dof) -> {"base_pos" (3,), "base_quat" (4,), "joint_pos" (num_dof,)}` all float32, `num_dof` mandatory.
`perturbation_from_seed` transcribes `genesis.utils.geom._np_xyz_to_quat` locally as `_rpy_to_quat` (float64, then cast) because genesis may not be imported; the draw order is 3 base-position, 3 base-rpy, `num_dof` joint offsets, each uniform in `[-p, +p]`.

### `env_base.py`
`BatchedEpisodeEnv` — a `runtime_checkable` `typing.Protocol` with `rows`, `episodes_per_policy`, `max_episode_length`, `cache` and `evaluate(pop_arrays, generation) -> dict` / `close()`.
`REQUIRED_KEYS`, `OPTIONAL_KEYS`, `rows_for(pop_size, episodes=EPISODES_PER_POLICY)`, `validate_evaluation(result, pop_size, episodes=EPISODES_PER_POLICY) -> None` (raises one `ValueError` listing every problem).
`_fitness_from_episode_returns(episode_returns) -> (pop_size,) float32` — the layer's ONE fitness rule (a non-finite episode counts as 0, mean taken in float32); `validate_evaluation` applies it and `trainer._aggregate` imports it, so the two cannot drift.

### `evox_runtime.py`
The plain-`evoxc` plumbing shared by the two graph modules: `resolve_backend_device(backend=None, device=None)` (explicit args > `EVOXC_BACKEND`/`EVOXC_DEVICE` > `numpy`/`cpu:0`), `POP_OPERANDS`, and the private `_parse_device`, `_constant`, `_to_device_array`, `_to_numpy`, `_attach`, `_device_name`, `_positive_int`, `_rows_of`, `_staged`, `_host`, `_outputs`, `_contiguous`, `_spec_for`, `_pop_specs`.
`_to_device_array` re-selects `evoxc.backends.get(backend)` before staging on a non-CPU device (mixed IREE/XLA processes), which must not be removed.

### `evolution_graphs.py`
`build_fitness(n_policies, *, episodes_per_policy=EPISODES_PER_POLICY, backend="numpy", device=None)` / `run_fitness(executable, returns)` — the `fitness_graph` (`evoxc.reduce_mean` of `(NP*episodes,) -> (NP, episodes)` over `axes=1`).
`build_tell(algo, config, pop_arrays, species_leaves, n_policies, *, backend="numpy", device=None, plugin_path=None)` / `run_tell(executable, *, randkey, pop_arrays, generation, fitness, species_leaves)` — one generation transition, returning `{"randkey", "generation", "pop", "species"}`.
`ExecutableCache` — shape-keyed `get(key, builder)` with `hits`/`misses` and `clear()`.

### `policy_graphs.py`
`build_policy(algo, config, pop_arrays, rows, *, num_inputs, num_outputs, episodes=EPISODES_PER_POLICY, backend="numpy", device=None)` / `run_policy(executable, pop_arrays, obs)` — the single width-generic genome -> action graph.
`algo`/`config` are required because `algo.transform`, `algo.prepare_forward(samples=episodes)` and `algo.forward_prepared` are trace-time operations; `num_inputs`/`num_outputs` are explicit and validated against `algo.genome` so a task cannot trace the wrong width.
Both pilots reach this graph: `../locomotion/go2_neat_kernels.py` wraps `build_policy` as `_shared_build_policy` and imports `run_policy`, and `../drone/hover_neat_env.py` imports both directly.

### `trainer.py`
`NeuroevolutionTrainer(env, pop_size, generations, master_seed=0, *, num_inputs, num_outputs, max_nodes=80, max_conns=640, species_size=3, tell_backend=None, tell_device=None, plugin_path=None, output_dir=None, logger=None, problem=None, seed_table=None)` with `setup() -> self`, `run() -> summary`, `close()`.
`TraceOnlyProblem(num_inputs, num_outputs)` (a `problem.BaseProblem` exposing only `input_shape`/`output_shape`; `evaluate` raises), `pop_arrays_of(state)`, `species_leaves_of(species)`, `_jsonable`, `_format_generation`.
Attributes: `algo`, `config`, `pop_arrays`, `species_leaves`, `randkey`, `generation_counter`, `seed_table`, `cache`, `tell_backend`, `tell_device`, `history_path`, `summary_path`, `records`, `fitness_history`, `summary`.

### `selfcheck.py`
The node's own runner: seeding values, `validate_evaluation` accept/reject, both policy widths, tell/fitness on the numpy backend, trainer determinism over 3 generations twice, the large-return fitness regression, and an AST import-hygiene scan.

### `benchmarks/`
The Stage-3 measurement harness (env replay parity, throughput, residency, trainer runs, backend/startup probes), its committed JSON records under `benchmarks/results/` and the published `benchmarks/RESULTS.md`; `benchmarks/CONTEXT.md` documents it and nothing here imports it.

## Interface contract (`BatchedEpisodeEnv.evaluate`)
`evaluate(pop_arrays, generation)` simulates ONE generation and returns at least `seeds (3,) int64`, `episode_returns (NP,3) float32`, `returns (rows,) float32`, `fitness (NP,) float32`, `steps`, `env_steps`, `wall_time_s`, and optionally `steps_per_s`, `env_steps_per_s`, `cache_hits`, `cache_misses`, `end_reasons` plus task diagnostics.
`rows = pop_size * episodes_per_policy` and rows are policy-major: policy `p` owns rows `episodes*p .. episodes*p+episodes-1`, which is the tiling `prepare_forward(samples=episodes)` declares.
Every policy of a generation is scored on the SAME ordered triple of episode seeds and every generation gets a fresh triple; a row ends once (first `done` inclusive) and `alive` never re-arms.
`pop_arrays` is the 4-leaf TensorNEAT genome payload (`node_keys`, `node_attrs`, `conn_keys`, `conn_attrs`) as a tuple in `GenomeTensors` order; a mapping is accepted by the contract for envs that name the leaves.
`env.cache` is optional: when `None` the trainer uses its private `ExecutableCache`, so a task env can share its own cache or have none.
The env owns its scene and `close()`; the trainer never closes the env.

## Seed rules
`np.random.SeedSequence(master_seed).spawn(2)` yields `(evolution_seq, env_seq)`; the evolution stream (`evolution_seed`, first child) seeds `Pipeline`, the env stream (second child) drives `generation_seed_table`, and the two are independent.
`evolution_seed(0) == 3757552657`, and `generation_seed_table(0, 3)` is `[[1454269175, 757617120, 1421746521], [1313752747, 908599966, 53557616], [1683657442, 1375977993, 1654802006]]`; `selfcheck.py` pins both.
The table is drawn one row at a time with `rng.choice(2**31, size=episodes, replace=False)`, which keeps a row's seeds distinct and the table prefix-stable (`table(seed, N)[i]` is the same for every `N > i`).
A zero `Perturbation` draws nothing and returns exact zeros plus `base_quat == [1,0,0,0]`, so the unperturbed config reproduces a fixed reset bit-for-bit.
The trainer adds no RNG of its own: `master_seed` alone reproduces a whole run (two runs of one configuration give identical `fitness_history`, records and seed table).

## Fitness rule
Fitness is the mean of a policy's `episodes_per_policy` episode returns, and an episode that was never scored (`NaN`, `+inf` or `-inf`) counts as 0.
`env_base._fitness_from_episode_returns` is the single owner of that rule: it zeroes every non-finite entry and takes the mean in float32, and both `env_base.validate_evaluation` and `trainer._aggregate` call it instead of restating it.
The trainer raises `ValueError` when the env-reported `fitness` disagrees with that rule beyond `atol=1e-6`.
The mean must be taken in float32 (the arithmetic of `evolution_graphs.build_fitness`/`evoxc.reduce_mean`): a float64 mean narrows back to float32 up to 3.8e-6 away at episode-return magnitudes of ~100 (the Go2 pilot), i.e. beyond the tolerance.
`validate_evaluation` checks the dtype and the `(pop_size, episodes)` shape first and only then compares the fitness, so a wrongly shaped or wrongly typed episode array is still rejected instead of being coerced by the rule.

## Constraints
No module here may `import torch`, `import genesis`, `import tensordict`, or import a task module (`go2_*`, `hover_*`); allowed are `numpy`, `evoxc`, `tensorneat` and the sibling modules of this directory.
Imports between these modules are bare-name (`from seeding import ...`), matching the rest of `examples/`; put this directory on `sys.path` (`PYTHONPATH=$PWD/examples/neuroevolution`) or extend `sys.path` as `tests/test_go2_neat_kernels.py` does. There is no `__init__.py`.
Every file stays well under ~1000 lines, one module per concern; the numerics, dtypes, reduction axes and compile options copied from the task modules must not change.
`tell()` compile options stay exactly as extracted: the IREE/CUDA reduction-distribution workaround (`--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`, gated on `backend == "iree" and device.kind == "cuda"`) and the `plugin_path` option for the `xla` backend.
Tell backend default: `env.backend` unless it is `numpy`; `rocm`/`hip` devices force `xla` because IREE/ROCm tell compilation does not terminate with evoxc's default flags (it does terminate once the reduction-distribution flag is passed, see below); the numpy tell runs on `cpu:0`.

## Device backends (measured on AMD Instinct MI300X VF, gfx942, ROCm 7.1)
`evoxc.devices()` -> `[cpu:0, cuda:0, rocm:0]` and `evoxc.devices("rocm")` -> `[rocm:0]`; the first enumeration costs 6.9 s because it probes the runtimes.
`evoxc.backends.get(name)` activates the `iree`/`xla` adapter and also fixes the preferred device-transfer provider for the device kind, so the last `get()` decides what `Tensor.to(rocm:0)` produces.
The xla adapter finds its PJRT plugin through an options-less chain only: `EVOXC_PJRT_PLUGIN=/tmp/xla_rocm_whl/x/jax_plugins/xla_rocm7/xla_rocm_plugin.so`, or a `jax_plugins/*/*.so` under any `sys.path` entry (both verified; a non-file `EVOXC_PJRT_PLUGIN` value raises `BackendError` naming the path).
So `build_tell(..., backend="xla", plugin_path=P)` builds and loads but the first `run_tell` raises `BackendError: the xla adapter requires a PJRT C API plugin library ... none was found`, because the placement provider behind `Tensor.to(rocm:0)` never sees the options; `build_fitness` has no `plugin_path` parameter, so an xla fitness graph needs the env var or the `sys.path` plugin.
Per-backend option surface: xla compile `plugin_path`/`xla_compile_options`/`opt_level`, load `plugin_path`, lower `split_reduction`/`rng_bit_generator`/`dot_precision`/`eigh_algorithm`, run none; iree compile `target_backends`/`iree_compile_args`/`opt_level`/`vulkan_target`/`dispatch_formation`, load `iree_runtime_args`; `target_backends` is iree-only and `plugin_path` is xla-only (each backend rejects the other's key with a `BackendError` that lists its own options).
IREE/ROCm cannot compile the tell graph with evoxc's defaults: `iree-compile` spins at ~140% CPU and 3.4 GB RSS and never returns (killed at 300 s, stack parked in `iree.compiler.tools.binaries.invoke_immediate`), which is the origin of the "IREE/ROCm tell does not terminate" rule.
Adding the CUDA-era workaround flag fixes ROCm too: `EVOXC_IREE_COMPILE_ARGS=--iree-codegen-llvmgpu-use-reduction-vector-distribution=false` (equivalently `iree_compile_args=[...]`) builds the tell in 7.6 s at pop 8 and 7.6 s at pop 32, and matches numpy (randkey and every integer leaf identical, float leaves <= 2 ulp); `build_tell` currently passes that flag only for `backend == "iree" and device.kind == "cuda"`.
Tell, cold build + steady run per generation, pop 8 / pop 32: numpy `cpu:0` 0.29 s + ~6 ms / 0.30 s + ~9.8 ms; iree `llvm-cpu` 6.1 s + ~13 ms; xla `rocm:0` 4.8 s + 6-8 ms / 5.1 s + 6.0 ms; iree `rocm:0` with the flag 7.6 s + ~90 ms / 7.6 s + ~159 ms.
Fitness graph steady run: numpy 0.03 ms, iree 0.21 ms, xla 0.46 ms at pop 8 (0.04 / 0.20 / 0.42 ms at pop 32); a first xla run in a fresh process costs ~6.2 s (BFC allocator reserves 143.77 GiB plus executable init), a first iree run 62-67 ms.
iree fitness is bit-identical to `env_base._fitness_from_episode_returns`; xla fitness differs by up to 7.6e-6 at return magnitudes ~75, i.e. past the trainer's `atol=1e-6` check, so an env that computes fitness on an xla graph can fail `_aggregate`.
Device payloads are not interchangeable: an `IreeDevicePayload` input to an xla executable raises `DeviceError ... Got a tensor carrying IreeDevicePayload` and an `XlaDevicePayload` input to iree raises the mirror error, because each adapter accepts only its own payload type on a device-resident run.
`Tensor.to(rocm:0)` returns `self` when the device already matches, so a tensor keeps whichever provider placed it; staging after the wrong `backends.get()` yields the other backend's payload and the run then fails at the adapter — the re-select in `evox_runtime._to_device_array` is what prevents that (reproduced by staging right after `backends.get("xla")`).
There is no device->device transfer: `np.from_dlpack(iree_tensor)` -> `BufferError: Unsupported device in DLTensor`, `evoxc.from_dlpack` rejects both device tensors and raw capsules, `XlaDevicePayload` exposes no `__dlpack__`, and `Tensor.to` refuses a payload source for a non-cpu target.
The working zero-copy bridges are torch-mediated: `torch.from_dlpack(iree_tensor)` (6.5 us, contiguous) and `hal.from_dlpack_capsule(t.__dlpack__()) -> IreeDevicePayload(rt.DeviceArray(hal, view), device)` (the `../locomotion/go2_device_rollout.py` pattern); an xla output leaves only through `payload.to_host()`/`t.to(cpu)`.
Host bridges (medians): `(8,)` float32 fitness iree->host->xla 205 us; the four pop leaves (74,240 B) xla->host->iree 775 us; all 15 tell operands (102,196 B) iree D2H 2.7 ms + H2D 1.1 ms, xla D2H 1.6 ms + H2D 2.3 ms.
A same-backend device-resident loop is possible: `evoxc.run(tell_exe, ..., fitness_tensor, ...)` accepts the fitness graph's own device output, pop 8 fitness+tell per generation = xla 1.88 ms synchronized (1.60 ms enqueue-only, device runs are async) and iree 85.9 ms.
The same xla tell through `run_tell`'s host staging costs 5.6 ms per generation, so at 24-32 rows the staging roundtrip, not the graph, is most of the wall time.
The mixed design (rollout iree + tell xla) therefore cannot stay on the device: the fitness vector and every pop/species leaf cross the host each generation.

## Routing Table
- `seeding.py` -> seed streams, seed table, reset perturbation draws (no simulator, no evoxc).
- `env_base.py` -> the environment contract, its validator and the single fitness rule.
- `evox_runtime.py` -> backend/device resolution, operand staging, payload specs.
- `evolution_graphs.py` -> fitness + tell graphs, `ExecutableCache`.
- `policy_graphs.py` -> the one width-generic genome -> action graph.
- `trainer.py` -> ask/rollout/fitness/tell loop, artifacts, logging.
- `selfcheck.py` -> the node's own executable test suite.
- `benchmarks/` -> the Stage-3 measurement harness (`replay`/`throughput`/`residency-parity`/`train`/probes), its committed raw records under `results/` and `RESULTS.md`.
- `../locomotion/go2_neat_env.py`, `../locomotion/go2_neat_kernels.py`, `../locomotion/go2_neat_train.py`, `../locomotion/go2_device_rollout.py` -> the Go2 walking pilot (siblings — read-only, escalate writes to parent).
- `../drone/hover_neat_env.py`, `../drone/hover_kernels.py`, `../drone/hover_neat_train.py`, `../drone/hover_device_rollout.py`, `../drone/hover_neat_check.py` -> the drone hover pilot plus its smoke/parity CLI (siblings — read-only, escalate writes to parent).
- `../../tests/test_neuroevolution_trainer.py`, `../../tests/test_go2_neat_kernels.py`, `../../tests/test_go2_neat_seeding.py`, `../../tests/test_hover_neat_seeding.py` -> the pytest modules that pin this layer and its adopters (siblings — read-only, escalate writes to parent).

## Design Decisions
`evox_runtime.py` is a separate module because both `evolution_graphs.py` and `policy_graphs.py` need the same staging helpers; keeping them in either graph module would make the policy graph depend on the tell graph.
There is exactly ONE policy graph (`policy_graphs.py`) instead of one per task; it takes the widths as parameters at 45 -> 12 and 17 -> 4, which keeps the two pilots' policy numerics identical by construction.
`TraceOnlyProblem` stores the widths in `_input_shape`/`_output_shape` and re-exposes them as properties because `BaseProblem.input_shape` is a read-only property, so the class cannot be parametrised by plain attributes.
`NeuroevolutionTrainer._static_config` is a staticmethod taking `(pipeline, state)` and raising `RuntimeError` if TensorNEAT's private `Pipeline._static_config` disappears, so the private-API dependency fails loudly instead of silently tracing with missing statics.
`output_dir=None` keeps every artifact in memory (`records`/`summary`) so a trainer check needs no filesystem; a path creates `history.jsonl` (one JSON object per generation, flushed) and `summary.json`.
Optional task hooks: an env exposing `executables()` is warmed once in `setup()` so its graphs compile once, and any result key the trainer does not consume is carried through as the record's `diagnostics` (e.g. `init_qpos`, `commands`).
The species pytree is captured at trace time with symbolic leaves, so `executable.species_tree` must not be compared for equality after a run; compare root type/leaf count and rebuild via the returned leaves.
The fitness rule lives in `env_base` rather than in `trainer` because the validator is the seam every task crosses first; one function behind both call sites is what makes "the trainer and the validator cannot disagree" checkable rather than aspirational.

## Migration Map

### Migrated pilots
- **Go2 walking** (`../locomotion/go2_neat_env.py`, `go2_neat_train.py`, `go2_neat_kernels.py`, `go2_device_rollout.py`): obs 45 -> actions 12, `dt=0.02`, `max_episode_length=1000` (20 s), pop 8, rows 24; entry point `go2_neat_train.py`, whose `--residency numpy` (default) is the host-staged reference and `--residency device` routes `Go2NeatEnv.evaluate` to `go2_device_rollout.evaluate_device`.
- **Drone hovering** (`../drone/hover_neat_env.py`, `hover_kernels.py`, `hover_neat_train.py`, `hover_device_rollout.py`, `hover_neat_check.py`): obs 17 -> actions 4, `dt=0.01`, `max_episode_length=1500` (15 s), pop 4-8; entry point `hover_neat_train.py` with the same `--residency numpy`/`device` split, and `hover_neat_check.py` is the cheap smoke/parity CLI (default mode, `--parity`, `--perturb`).

### Not migrated
- **Manipulation grasping** (`../manipulation/grasp_env.py`, `grasp_train.py`, `grasp_eval.py`, `behavior_cloning.py`): 6 actions, a 7-dim end-effector pose plus stereo RGB camera observations, learned with rsl-rl PPO and behaviour cloning (torch + tensordict throughout). Blocked: the observation is multi-modal RGB, so it needs a vision encoder inside the compiled policy graph, and the whole task still lives in the torch/rsl-rl world — do not migrate until a vision policy graph exists.
- **PPO baselines** (`../locomotion/go2_env.py`, `../drone/hover_env.py`): rsl-rl reference implementations kept as the behavioural oracle for their neuroevolution twins; they stay on torch/tensordict and are not migrated.
- **Legacy math oracles** (`../locomotion/go2_legacy_reference.py`, `../drone/hover_legacy_reference.py`): dependency-free numpy transcriptions of the pre-migration torch maths, used only by the parity tests; they are references, never imported by this layer.
- **Speed benchmarks** (`../speed_benchmark/franka.py`, `anymal_c.py`, `timers.py`): throughput/timing only, no policy, no episode returns; torch appears solely to build initial states. Not a migration target — no fitness to evolve.
- Everything else under `examples/` (rigid, coupling, deformable, sensors, rendering, usd, ipc, kinematic, gui, viewer_plugin, tutorials) is a demo without an episode/return loop; only a future task that scores rollouts needs this layer.

### Shared engine code (`genesis/`)
- `import genesis` pulls torch into the process (`genesis.utils.misc` imports it, and the quadrants runtime `genesis` is built on depends on torch), so an `evoxc`-only pilot still has torch resident even though no module here imports it.
- Entity getters clone: `RigidEntity.get_pos/get_quat/get_vel/get_ang/get_dofs_position/get_dofs_velocity` reach `BaseSolver._read_data(..., copy=True)` -> `qd_to_torch(..., copy=True)`, i.e. a device-to-device copy per read.
- Entity setters write through zero-copy torch views: `RigidSolver.set_qpos` under `gs.use_zerocopy` writes into `qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)`, so the value handed to `set_qpos`/`set_pos`/`set_quat` must be a torch tensor on the engine device.
- `scene.get_error_envs_mask()` is a host-query torch tensor, and `scene.clear_errno()` must run before the next `scene.step()` or a latched solver error re-raises.
- The numpy residency of both pilots still touches all four: the Go2 env reads the getters (`_to_host_float`, `.detach().cpu().numpy()`), converts writes itself (`torch.as_tensor` in `_to_device_float`) and reads the error mask per generation; the drone env reads through `tensor_to_array` and hands NumPy `array_like` straight to its setters, and reads the mask plus `clear_errno()` per generation.
- The device residency of both pilots (`go2_device_rollout.py`, `hover_device_rollout.py`) reads the LIVE solver buffers with `qd_to_torch(..., copy=False)`, exports them with `torch.utils.dlpack.to_dlpack`, imports the graph's IREE capsule with `torch.from_dlpack`, and still calls `scene.get_error_envs_mask()`/`clear_errno()`.
- What would remove each: a quadrants-side DLPack capsule producer drops the `qd_to_torch` + `torch.utils.dlpack.to_dlpack` bridge; a Genesis setter accepting a foreign device pointer/capsule drops `torch.from_dlpack` on the write path; an error mask returned as a host bool array (or kept device-resident) drops the per-generation host read; a torch-free quadrants build drops torch from `import genesis`, and with it the getters' torch return type.

## Current Adopters
- `../locomotion/go2_neat_env.py` -> `env_base.{BatchedEpisodeEnv, validate_evaluation}`, `evox_runtime._device_name`, `seeding.{EPISODES_PER_POLICY, Perturbation, evolution_seed, generation_seed_table, perturbation_from_seed}`, `trainer.{TraceOnlyProblem, pop_arrays_of}`.
- `../locomotion/go2_neat_kernels.py` -> `evolution_graphs.{ExecutableCache, build_fitness, build_tell, run_fitness, run_tell}`, `evox_runtime.{_attach, _device_name, _outputs, _positive_int, _rows_of, _staged}`, `policy_graphs.{build_policy as _shared_build_policy, run_policy}`, `seeding.{EPISODES_PER_POLICY, Perturbation}`.
- `../locomotion/go2_neat_train.py` -> `trainer.NeuroevolutionTrainer`.
- `../locomotion/go2_device_rollout.py` -> nothing directly; it consumes the device operands through `go2_neat_kernels`.
- `../drone/hover_neat_env.py` -> `env_base.{BatchedEpisodeEnv, validate_evaluation}`, `evolution_graphs.{ExecutableCache, build_fitness, run_fitness}`, `evox_runtime.resolve_backend_device`, `policy_graphs.{build_policy, run_policy}`, `seeding.{EPISODES_PER_POLICY, Perturbation, generation_seed_table, perturbation_from_seed}`, `trainer.TraceOnlyProblem`.
- `../drone/hover_kernels.py` -> `evox_runtime.{POP_OPERANDS, _attach, _constant, _device_name, _outputs, _positive_int, _staged, _to_device_array, _to_numpy, resolve_backend_device}`, `seeding.EPISODES_PER_POLICY`; its `__main__` self-check additionally imports `evolution_graphs.{build_fitness, run_fitness}` and `seeding.{evolution_seed, generation_seed_table}`.
- `../drone/hover_neat_train.py` -> `seeding.{Perturbation, evolution_seed}`, `trainer.NeuroevolutionTrainer`.
- `../drone/hover_device_rollout.py` -> `env_base.validate_evaluation`, `evox_runtime.{POP_OPERANDS, _outputs}`.
- `../drone/hover_neat_check.py` -> `seeding.Perturbation`, and `policy_graphs.run_policy` in `--parity` mode.
- `../../tests/test_neuroevolution_trainer.py` -> `env_base`, `policy_graphs`, `seeding`, `trainer`; `../../tests/test_go2_neat_kernels.py` -> `evolution_graphs`, `policy_graphs`, `seeding`, `trainer`; `../../tests/test_go2_neat_seeding.py` and `../../tests/test_hover_neat_seeding.py` -> `seeding`.

## Test Strategy
`selfcheck.py` is the node's executable suite: `PYTHONPATH=$PWD/examples/neuroevolution /root/evox-robotics/.venv/bin/python examples/neuroevolution/selfcheck.py` (~3 s, numpy backend, exit 0 and `ALL CHECKS PASSED`).
It pins the documented seed values and prefix stability, the `validate_evaluation` accept/reject matrix (including a non-finite episode counted as 0 and the rejection of a fitness that keeps an infinity), the fitness rule at Go2 magnitudes (via the real `build_fitness`/`run_fitness`), both policy widths, a tell round-trip and `ExecutableCache` hit/miss counts, trainer determinism over 3 generations twice with the same master seed, and the absence of `torch`/`genesis`/`tensordict` imports in all six modules.
`../../tests/test_neuroevolution_trainer.py` is the pytest mirror of this layer (`@pytest.mark.required`, no scene): the env contract accept/reject matrix, the seed constants and stream independence, the width-generic policy graph at both widths, and trainer reproducibility plus the float32 aggregation tolerance.
`../../tests/test_go2_neat_kernels.py` drives the Go2 task graphs and the shared `ExecutableCache`/`policy_graphs`/`TraceOnlyProblem` they consume; `../../tests/test_go2_neat_seeding.py` and `../../tests/test_hover_neat_seeding.py` pin `seeding` through each env's pure reset helpers, scene-free.
`../../tests/test_go2_evox_parity.py` and `../../tests/test_hover_evox_parity.py` compare the task kernels against `go2_legacy_reference.py`/`hover_legacy_reference.py`; they do not import this directory, but they are the numeric guard on the graphs whose staging helpers it owns.
Every one of those modules `pytest.importorskip`s `evoxc`/`tensorneat`, since neither is a declared dependency of `genesis-world`.
`../../tests/test_examples.py` ignores `neuroevolution/*.py` (so `selfcheck.py` is not run as an example) and registers `neuroevolution -> ["evoxc", "tensorneat"]`; the `benchmarks/` CLIs are still discovered by it and exit 0 on bare invocation.
The measurement record is committed, not re-derived: `benchmarks/results/` (38 records), `benchmarks/results/summary.json` and `tables.md` (regenerated by `benchmarks/summarize.py`), and `benchmarks/RESULTS.md`; `summarize.py check` and `bench_neat.py check-determinism` are their executable checks.
The graph layer's numeric identity is guarded from both sides: the tests above pin the graphs against the task math and the legacy oracles, and the committed `benchmarks/results/` records pin environment replay parity, residency differences and trainer determinism per pilot.

## Notes for Agents
`evoxc` emits a `UserWarning: Embedding a tensor of 2851288 bytes as a graph constant` while building the 45->12 tell graph (its padded genome constant exceeds the 1 MiB embedding threshold); that is expected library noise, not a defect, and the self-check leaves it visible.
The numpy backend is the default and the fastest choice at these row counts (24-32 rows); a GPU tell run needs `--tell-backend iree` (CUDA) or `xla` (ROCm, with the PJRT plugin visible to the options-less discovery chain) plus the ROCm HSA-runtime handling documented in `../locomotion/CONTEXT.md`, and an iree/ROCm tell additionally needs `EVOXC_IREE_COMPILE_ARGS=--iree-codegen-llvmgpu-use-reduction-vector-distribution=false` or it hangs in `iree-compile` — see "Device backends".
`NeuroevolutionTrainer.run()` returns the summary and never writes outside `output_dir`; long runs should pass `output_dir` so a crash keeps the flushed `history.jsonl`.
The two pilots put this directory on `sys.path` themselves (`_ensure_shared_layer_path` in the Go2 helpers, the `_SHARED` insert in `hover_kernels.py`), so a new task module must do the same before its first bare-name import.
`../drone/hover_neat_check.py` and the two trainers are the only adopters meant to be run by hand; every other adopter is imported by them or by `tests/`.
