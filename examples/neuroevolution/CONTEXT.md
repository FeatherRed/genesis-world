# examples/neuroevolution

## Intent
The shared, task-agnostic neuroevolution layer of the example tree: one trainer, one environment contract, one seeding rule set and one copy of each compiled graph, used by every TensorNEAT/`evoxc` task instead of per-task copies.
A task contributes only its simulator env (`examples/locomotion/go2_neat_env.py`, `examples/drone/hover_neat_env.py`) and its task graphs; the ask -> rollout -> fitness -> tell loop, the artifact format and the reproducibility rules live here.
Nothing in this directory imports `torch`, `genesis` or `tensordict`, so the whole layer is testable without a simulator or a GPU.

## API Surface

### `seeding.py`
`EPISODES_PER_POLICY = 3`; `Perturbation(base_pos=0.0, base_rpy=0.0, joint_pos=0.0)` frozen dataclass with `is_zero`; `generation_seed_table(master_seed, num_generations, episodes=EPISODES_PER_POLICY) -> (G, episodes) int64`; `evolution_seed(master_seed) -> int`; `perturbation_from_seed(seed, perturbation, num_dof) -> {"base_pos" (3,), "base_quat" (4,), "joint_pos" (num_dof,)}` all float32, `num_dof` mandatory.
`perturbation_from_seed` transcribes `genesis.utils.geom._np_xyz_to_quat` locally as `_rpy_to_quat` (float64, then cast) because genesis may not be imported; the draw order is 3 base-position, 3 base-rpy, `num_dof` joint offsets, each uniform in `[-p, +p]`.

### `env_base.py`
`BatchedEpisodeEnv` — a `runtime_checkable` `typing.Protocol` with `rows`, `episodes_per_policy`, `max_episode_length`, `cache` and `evaluate(pop_arrays, generation) -> dict` / `close()`.
`REQUIRED_KEYS`, `OPTIONAL_KEYS`, `rows_for(pop_size, episodes=EPISODES_PER_POLICY)`, `validate_evaluation(result, pop_size, episodes=EPISODES_PER_POLICY) -> None` (raises one `ValueError` listing every problem).

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

### `trainer.py`
`NeuroevolutionTrainer(env, pop_size, generations, master_seed=0, *, num_inputs, num_outputs, max_nodes=80, max_conns=640, species_size=3, tell_backend=None, tell_device=None, plugin_path=None, output_dir=None, logger=None, problem=None, seed_table=None)` with `setup() -> self`, `run() -> summary`, `close()`.
`TraceOnlyProblem(num_inputs, num_outputs)` (a `problem.BaseProblem` exposing only `input_shape`/`output_shape`; `evaluate` raises), `pop_arrays_of(state)`, `species_leaves_of(species)`, `_jsonable`, `_format_generation`.
Attributes: `algo`, `config`, `pop_arrays`, `species_leaves`, `randkey`, `generation_counter`, `seed_table`, `cache`, `tell_backend`, `tell_device`, `history_path`, `summary_path`, `records`, `fitness_history`, `summary`.

### `selfcheck.py`
The node's own runner: seeding values, `validate_evaluation` accept/reject, both policy widths, tell/fitness on the numpy backend, trainer determinism over 3 generations twice, the large-return fitness regression, and an AST import-hygiene scan.

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
Fitness is the mean of a policy's `episodes_per_policy` episode returns, and an episode that was never scored (non-finite return) counts as 0.
The trainer is the single owner of that rule: it recomputes `np.nan_to_num(episode_returns, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=1)` in float32 and raises `ValueError` when the env-reported `fitness` disagrees beyond `atol=1e-6`.
The mean must be taken in float32 (the arithmetic of `evolution_graphs.build_fitness`/`evoxc.reduce_mean`): a float64 mean narrows back to float32 up to 3.8e-6 away at episode-return magnitudes of ~100 (the Go2 pilot), i.e. beyond the tolerance, and `env_base.validate_evaluation` applies the same float32 rule.

## Constraints
No module here may `import torch`, `import genesis`, `import tensordict`, or import a task module (`go2_*`, `hover_*`); allowed are `numpy`, `evoxc`, `tensorneat` and the sibling modules of this directory.
Imports between these modules are bare-name (`from seeding import ...`), matching the rest of `examples/`; put this directory on `sys.path` (`PYTHONPATH=$PWD/examples/neuroevolution`) or extend `sys.path` as `tests/test_go2_neat_kernels.py` does. There is no `__init__.py`.
Every file stays well under ~1000 lines, one module per concern; the numerics, dtypes, reduction axes and compile options copied from the task modules must not change.
`tell()` compile options stay exactly as extracted: the IREE/CUDA reduction-distribution workaround (`--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`, gated on `backend == "iree" and device.kind == "cuda"`) and the `plugin_path` option for the `xla` backend.
Tell backend default: `env.backend` unless it is `numpy`; `rocm`/`hip` devices force `xla` because IREE/ROCm tell compilation does not terminate; the numpy tell runs on `cpu:0`.

## Routing Table
- `seeding.py` -> seed streams, seed table, reset perturbation draws (no simulator, no evoxc).
- `env_base.py` -> the environment contract and its validator.
- `evox_runtime.py` -> backend/device resolution, operand staging, payload specs.
- `evolution_graphs.py` -> fitness + tell graphs, `ExecutableCache`.
- `policy_graphs.py` -> the one width-generic genome -> action graph.
- `trainer.py` -> ask/rollout/fitness/tell loop, artifacts, logging.
- `selfcheck.py` -> the node's own executable test suite.
- `../locomotion/go2_neat_env.py` -> the Go2 task env implementing this contract (sibling — read-only, escalate writes to parent).
- `../drone/hover_neat_env.py` -> the drone task env implementing this contract (sibling — read-only, escalate writes to parent).
- `../../tests/test_go2_neat_kernels.py`, `../../tests/test_go2_neat_seeding.py` -> graph/seeding behaviour tests of the migration targets (sibling — read-only, escalate writes to parent).

## Design Decisions
`evox_runtime.py` is a separate module because both `evolution_graphs.py` and `policy_graphs.py` need the same staging helpers; keeping them in either graph module would make the policy graph depend on the tell graph.
There is exactly ONE policy graph (`policy_graphs.py`) instead of one per task; it takes the widths as parameters and was proven bit-identical to both former copies (0.0 max abs diff against `go2_neat_kernels.build_policy` at 45->12 and `hover_kernels.build_policy` at 17->4).
`TraceOnlyProblem` stores the widths in `_input_shape`/`_output_shape` and re-exposes them as properties because `BaseProblem.input_shape` is a read-only property, so the class cannot be parametrised by plain attributes.
`NeuroevolutionTrainer._static_config` is a staticmethod taking `(pipeline, state)` and raising `RuntimeError` if TensorNEAT's private `Pipeline._static_config` disappears, so the private-API dependency fails loudly instead of silently tracing with missing statics.
`output_dir=None` keeps every artifact in memory (`records`/`summary`) so a trainer check needs no filesystem; a path creates `history.jsonl` (one JSON object per generation, flushed) and `summary.json`.
Optional task hooks: an env exposing `executables()` is warmed once in `setup()` so its graphs compile once, and any result key the trainer does not consume is carried through as the record's `diagnostics` (e.g. `init_qpos`, `commands`).
The species pytree is captured at trace time with symbolic leaves, so `executable.species_tree` must not be compared for equality after a run; compare root type/leaf count and rebuild via the returned leaves.

## Migration Map
- **Go2 walking** (`../locomotion/go2_neat_env.py`, `go2_neat_train.py`, `go2_neat_kernels.py`): obs 45 -> actions 12, `dt=0.02`, `max_episode_length=1000` (20 s), pop 8, rows 24; torch surface is the Genesis write boundary only (`torch.as_tensor` in `_to_device_float`; reads go through `detach/cpu/numpy`). Blocks it: nothing — it is the reference implementation, and adopting this layer means importing `Perturbation`/`EPISODES_PER_POLICY`/`generation_seed_table`/`evolution_seed`/`perturbation_from_seed` from `seeding`, the four graph builders from `evolution_graphs`/`policy_graphs`, and replacing `build_genome_pipeline` + the main loop with `NeuroevolutionTrainer`.
- **Drone hovering** (`../drone/hover_neat_env.py`, `hover_kernels.py`, `hover_neat_check.py`): obs 17 -> actions 4, `dt=0.01`, `max_episode_length=1500` (15 s), pop 4-8; already torch-free (numpy boundary via `genesis.utils.misc.tensor_to_array`). Blocks it: nothing — it already implements this contract; it needs the same `seeding`/graph imports and a `NeuroevolutionTrainer` entry point, after which `hover_neat_check.py` can become a thin env check.
- **Manipulation grasping** (`../manipulation/grasp_env.py`, `grasp_train.py`, `grasp_eval.py`, `behavior_cloning.py`): 6 actions, a 7-dim end-effector pose plus stereo RGB camera observations, learned with rsl-rl PPO and behaviour cloning (torch + tensordict throughout). Blocks it: image observations have no compiled-graph encoder path, the policy input is multi-modal, and there is no episode-return boundary yet — do not migrate until a vision policy graph exists.
- **Speed benchmarks** (`../speed_benchmark/franka.py`, `anymal_c.py`, `timers.py`): throughput/timing only, no policy, no episode returns; torch appears solely to build initial states. Not a migration target — no fitness to evolve.
- **PPO baselines** (`../locomotion/go2_env.py`, `../drone/hover_env.py`): rsl-rl reference implementations kept as the behavioural oracle for their neuroevolution twins; they stay on torch/tensordict and are not migrated.
- Everything else under `examples/` (rigid, coupling, deformable, sensors, rendering, usd, ipc, kinematic, gui, viewer_plugin, tutorials) is a demo without an episode/return loop; only a future task that scores rollouts needs this layer.

## Test Strategy
`selfcheck.py` is the node's executable suite: `PYTHONPATH=$PWD/examples/neuroevolution /root/evox-robotics/.venv/bin/python examples/neuroevolution/selfcheck.py` (~3 s, numpy backend, exit 0 and `ALL CHECKS PASSED`).
It pins the documented seed values and prefix stability, the `validate_evaluation` accept/reject matrix, the fitness rule at Go2 magnitudes (via the real `build_fitness`/`run_fitness`), both policy widths, a tell round-trip and `ExecutableCache` hit/miss counts, trainer determinism over 3 generations twice with the same master seed, and the absence of `torch`/`genesis`/`tensordict` imports in all six modules.
`../../tests/test_examples.py` discovers every `examples/**/*.py` except `*_train.py`, `*_eval.py`, `*_env.py`, `behavior_cloning.py` and the Go2 helper modules, so `selfcheck.py` IS collected as an example subprocess; it needs an `EXAMPLE_DEPENDENCIES` entry for `evoxc`/`tensorneat` (and the other modules here should be added to `IGNORE_SCRIPT_PATTERNS` as shared helpers) — that file belongs to the tests node and is not editable from here.
The graph layer's numeric identity is pinned by construction: the extracted policy graph was compared to both former copies at 0.0 max abs diff, the fitness graph to the float32 episode mean at 0.0, and the tell graph to a numpy-backend round-trip.
No test files live under `tests/` for this node yet; a `tests/test_neuroevolution_*.py` mirroring `test_go2_neat_kernels.py`/`test_go2_neat_seeding.py` is the natural follow-up owned by the tests node.

## Notes for Agents
Delete these symbols from the siblings once their tasks adopt this layer (this node must not edit them): `../locomotion/go2_neat_kernels.py` -> `build_tell`, `run_tell`, `build_fitness`, `run_fitness`, `ExecutableCache`, `_species_specs`, `_pop_specs`, `_host`, `_outputs`, `_contiguous`, `_spec_for`, `_attach`, `_device_name`, `_positive_int`, `build_policy`, `run_policy`.
Also from `../locomotion/go2_neat_env.py` -> `Perturbation`, `EPISODES_PER_POLICY`, `generation_seed_table`, `evolution_seed`, `perturbation_from_seed` (import from `seeding`); `TraceOnlyProblem` (use `trainer.TraceOnlyProblem(num_inputs, num_outputs)` — the legacy copy takes no arguments); and from `../locomotion/go2_neat_train.py` -> `build_genome_pipeline`, `pop_arrays_of`, `species_leaves_of`, `_jsonable`, `_format_generation` and the hand-rolled generation loop.
From `../drone/hover_kernels.py` -> `build_policy`, `run_policy`, `POP_OPERANDS`, `_pop_specs`, `_host`, `generation_seed_table`, `evolution_seed`, `EPISODES_PER_POLICY`, `hover_fitness`, `build_fitness`, `run_fitness`; and `../locomotion/go2_evox_kernels.py` keeps only its Go2 gait/reward task math and should re-import `resolve_backend_device`/`_constant`/`_to_device_array`/`_to_numpy` from `evox_runtime`.
`evoxc` emits a `UserWarning: Embedding a tensor of 2851288 bytes as a graph constant` while building the 45->12 tell graph (its padded genome constant exceeds the 1 MiB embedding threshold); that is expected library noise, not a defect, and the self-check leaves it visible.
The numpy backend is the default and the fastest choice at these row counts (24-32 rows); a GPU tell run needs `--tell-backend iree` (CUDA) or `xla` (ROCm) plus the ROCm HSA-runtime handling documented in `../locomotion/CONTEXT.md`.
`NeuroevolutionTrainer.run()` returns the summary and never writes outside `output_dir`; long runs should pass `output_dir` so a crash keeps the flushed `history.jsonl`.
