# examples/locomotion

## Intent
RL locomotion stack for the Unitree Go2 in Genesis: a scripted-trot demo whose control and reward math run as compiled EvoX Compiler (`evoxc`) graphs, an independent TensorNEAT neuroevolution training entry for walking that sits on the same compiled-graph layer, plus an rsl-rl PPO training/eval stack and playback of pre-trained backflip policies.

## API Surface

### Scripted trot on EvoX Compiler graphs
- `try1.py` → the demo. Default = viewer until the window is closed (under pytest: no viewer, 5 steps); `--max-steps N` bounds the viewer run; `--headless` runs 250 steps (5 s), keeps per-step rewards, writes `go2_reward.png` next to the script and prints cumulative total + final pose.
  `--evox-backend` / `--evox-device` select the compiled-graph backend/device (default `numpy` / `cpu:0`); the resolved backend, device and compile time are printed.
- `go2_evox_kernels.py` → the two `evoxc` graphs (pure evoxc ops, no torch, no genesis) plus build/run helpers; both graphs are compiled ONCE before the step loop and reused every step.
  `GAIT` runtime inputs: `step` (0-d int32), `previous_action` (float32, 12) → outputs `targets`, `action`, `action_change` (float32, 12 each).
  `GAIT` static specializations: `standing_flat` (12), `phases` (4, `[0, pi, pi, 0]` per leg), `dt`, `inv_action_scale` (= 1/0.25).
  `REWARD` runtime inputs: `base_pos`(3), `base_quat`(4), `base_lin_vel_world`(3), `base_ang_vel_world`(3), `dof_pos`(12), `action_change`(12), all float32 → dict of the six Go2 Walking terms (`tracking_lin_vel`, `tracking_ang_vel`, `lin_vel_z`, `base_height`, `action_rate`, `similar_to_default`) plus `total`.
  `REWARD` static specializations: `standing_flat` (12), `dt`.
  Helpers: `DT`, `ACTION_SCALE`, `STANDING` (4×3) / `STANDING_FLAT` (12), `REWARD_TERM_NAMES`, `resolve_backend_device(backend, device)`, `build_gait()`, `build_reward()`, `run_gait(executable, step, previous_action)`, `run_reward(executable, ...)`.
- `go2_legacy_reference.py` → the pre-migration math as importable pure functions and the parity oracle: `gait_step(step, previous_action, ...)` (numpy float64) and `walking_reward_terms(...)` (torch float32 state terms, numpy float64 action rate). No genesis import; quaternion helpers are transcribed from `genesis.utils.geom`.

### TensorNEAT Go2 walking graphs (`go2_neat_kernels.py`)
- Pure-`evoxc` module (no `genesis`, no torch, no simulation): every compiled graph the TensorNEAT Go2-walking trainer needs.
- Constants: `NUM_OBS=45`, `NUM_ACTIONS=12`, `EPISODES_PER_POLICY=3`, `CLIP_ACTIONS=100.0`, `TERMINATION_LIMIT_DEG=10.0`, `OBS_SCALES`, `REWARD_SCALES`, `DEFAULT_COMMANDS`.
- `DEFAULT_COMMANDS = [0.5, 0, 0]` is exact (not an approximation): `go2_train.get_cfgs()` uses singleton ranges, so `go2_env`'s command resampling is a no-op.
- `Perturbation(base_pos, base_rpy, joint_pos)` — frozen reset-amplitude dataclass with `is_zero`; zero = the original fixed reset. The RNG-side `perturbed_init_qpos` belongs to the env module, not here.
- `build_obs(rows, *, standing_flat, commands, backend, device)` / `run_obs(exe, *, base_ang_vel_world, base_quat, dof_pos, dof_vel, actions, commands)` → `(rows,45)`, same layout/scaling as `go2_env._update_observation`.
- `build_policy(algo, config, pop_arrays, rows, ...)` / `run_policy(exe, pop_arrays, obs)` → raw `(rows,12)` actions; one trace runs `algo.transform` + `algo.prepare_forward(samples=EPISODES_PER_POLICY)` + `algo.forward_prepared`; `rows = pop_size * EPISODES_PER_POLICY`, policy-major rows.
- `build_advance(rows, *, max_episode_length, ...)` / `run_advance(...)` → `{"obs","reward","returns","alive","done","episode_length"}`: clips raw actions to ±100, the six dt-scaled `go2_env` reward terms, `alive`/`done`/`returns` masking, int32 episode counter, arctan2-free roll/pitch termination.
- `build_fitness(n_policies, ...)` / `run_fitness(exe, returns)` → per-policy episode mean `(NP,)`.
- `build_tell(algo, config, pop_arrays, species_leaves, n_policies, ...)` / `run_tell(...)` → `{"randkey","generation","pop","species"}`, species rebuilt with the pytree captured during the trace.
- `ExecutableCache.get(key, builder)` / `hits` / `misses` / `clear()`: shape-keyed reuse, so generation N+1 runs on generation N's executables (each call logged as `hit`/`miss` + key).

### TensorNEAT Go2 walking trainer (`go2_neat_env.py` + `go2_neat_train.py`)
- Two modules layered on `go2_neat_kernels.py`: `go2_neat_env.py` is the batched, seeded evaluation env (no CLI); `go2_neat_train.py` is the CLI entry that drives NEAT generations through it.
- `go2_neat_env.py` builds ONE `gs.Scene` with `n_envs = rows = EPISODES_PER_POLICY * pop_size` (policy `p` owns rows `3p, 3p+1, 3p+2`), with the same plane+Go2 morphs, `SimOptions(dt=0.02, substeps=2)` and `RigidOptions(enable_self_collision=False, max_collision_pairs=20, tolerance=1e-5)` as `go2_env.py`.
- `CFG` is a literal copy of the authoritative `go2_train.get_cfgs()` numbers (joint names, default dof angles, kp/kd, base init pos/quat, `clip_actions`, `action_scale`, termination thresholds, `tracking_sigma`, `base_height_target`, `episode_length_s`, command ranges), because `go2_train` cannot even be imported without `rsl-rl-lib`.
- Public API of `go2_neat_env.py`: `generation_seed_table(master_seed, num_generations, episodes=EPISODES_PER_POLICY) -> (G,3) int64`; `evolution_seed(master_seed) -> int`; `perturbation_from_seed(seed, perturbation, num_dof=NUM_ACTIONS) -> {"base_pos","base_quat","joint_pos"}`; `perturbed_init_qpos(init_qpos, seeds, perturbation) -> (3,19) float32`; `TraceOnlyProblem`; `class Go2NeatEnv(pop_size, master_seed, *, perturbation=Perturbation(), horizon=None, episode_length_s=None, backend="numpy", device=None, show_viewer=False, seed_table=None, stop_when_done=True)`.
- `Go2NeatEnv.evaluate(pop_arrays, generation)` returns `{"seeds" (3,), "episode_returns" (NP,3), "returns" (rows,), "fitness" (NP,) float32, "steps", "env_steps", "wall_time_s", "steps_per_s", "env_steps_per_s", "cache_hits", "cache_misses", "end_reasons", "init_qpos"}`; `steps` counts simulated steps and `env_steps = steps * rows`.
- `Go2NeatEnv` also exposes `.rows`, `.max_episode_length`, `.cache`, `.executables()` (the four built graph executables), `.read_state()`, `._write_targets()`, `._write_reset()`, `.close()`.
- `horizon` = steps per episode; default = the full episode (`episode_length_s / dt` = 1000) and it is the `max_episode_length` baked into `build_advance`.
- `end_reasons` buckets every row at the end of the generation: `timeout`, `tilted`, `error` (genesis `scene.get_error_envs_mask()`), `stopped` (early stop with `alive` rows still live), `running`.
- `stop_when_done=True` (default) ends the generation as soon as every row's `alive` is False; `stop_when_done=False` always simulates the full horizon (needed for throughput measurements and to exercise auto-reset/timeout).
- `go2_neat_train.py` CLI: `--pop-size` (8), `--generations` (2; 1 under pytest), `--horizon` (full 1000; 20 under pytest), `--episode-length-s` (20.0), `--master-seed` (0), `--perturb-base-pos` / `--perturb-base-rpy` / `--perturb-joint-pos` (0.05 each), `--no-perturbation`, `--stop-when-done` / `--no-stop-when-done`, `-e/--exp-name`, `-o/--output-dir` (`out/go2_neat`), `-g/--gpu`, `--evoxc-backend`, `--evoxc-device`, `-v/--vis`, `--quiet`.
- Trainer loop: `DefaultGenome(num_inputs=45, num_outputs=12, max_nodes=80, max_conns=640)` -> `NEAT(gen, pop_size=NP, species_size=3)` -> `Pipeline(algo, TraceOnlyProblem(), seed=evolution_seed(master_seed), backend="numpy", is_detailed=False)` -> `state = pipe.setup()` -> `config = pipe._static_config(state)`; then per generation ask (`pop` leaves) -> `env.evaluate` -> `run_fitness` -> `run_tell`.
- `TraceOnlyProblem(tensorneat.problem.BaseProblem)` only declares `input_shape=(45,)` / `output_shape=(12,)`; its `evaluate` raises `NotImplementedError` because all real evaluation happens in the host loop.
- All five graphs (obs, policy, advance, fitness, tell) are built once through `ExecutableCache` and reused for every generation; each generation logs `+N hits/+0 misses` and the trainer prints the total `built 5 executables in ... (cache 0 hits / 5 misses)` line.
- Outputs: one compact line per generation on stdout, `history.jsonl` (one JSON object per generation) and `summary.json` (config + records + totals) under `--output-dir`; both files are git-ignored run artifacts.
- `go2_train.py` stays the untouched PPO control path (rsl-rl); neither new module imports it.

### Seed rules
- `np.random.SeedSequence(master_seed).spawn(2)` gives `(evolution_seq, env_seq)`; `evolution_seed(master_seed)` is the first child (`evolution_seed(0) = 3757552657`) and feeds `Pipeline(seed=...)`, while the second child drives `generation_seed_table` — the evolution and environment streams are independent.
- The seed table holds one row per generation with 3 distinct seeds per row; a generation's ordered triple is shared by every policy (and by all three rows of a policy), so all policies are compared on identical reset perturbations.
- `perturbation_from_seed` draws the whole reset (base position offset, a small w-first `base_quat` rotation, 12 joint offsets, each component uniform in `[-p, p]`) from `default_rng(seed)` alone and returns exact zeros when `Perturbation.is_zero`, so the zero-perturbation config reproduces the original fixed reset bit-for-bit.
- A perturbed episode keeps its perturbation across auto-reset (post-reset episodes are discarded by the `alive`/`returns` masking), so a fresh seed per episode is unnecessary.
- Fitness = mean of the policy's 3 episode returns, where an episode return includes the step that ended it (first done inclusive).
- Same `master_seed` + same flags implies a byte-identical seed table, per-generation seeds, returns and fitness arrays.

### Measured numbers (TensorNEAT trainer, RTX 5070, CPU/numpy path)
- Compile cost: `built 5 executables in 7.4–9.7 s (cache 0 hits / 5 misses)`, then 6 hits / 0 misses per generation for every generation.
- NP=8 (24 rows), full horizon 1000 steps, `--no-stop-when-done`: 5.61 / 5.85 / 5.94 s per generation (178.3 / 170.9 / 168.4 steps/s, 4279 / 4101 / 4042 env-steps/s); best fitness −7.3785 / −7.3180 / −6.0399; ends timeout=0 tilted=24 every generation; 3000 steps in 17.5 s.
- NP=16 (48 rows), full horizon 1000 steps, `--no-stop-when-done`: 7.04 / 8.09 / 8.20 s per generation (142.0 / 123.5 / 122.0 steps/s, 6817 / 5930 / 5857 env-steps/s); best fitness −6.6796 / −5.6710 / −3.6158; tilted=48; 3000 steps in 23.4 s.
- NP=8, `--horizon 25` (early stop): 21 / 10 / 13 steps, warm-up generation 12.7 steps/s then 164–167 steps/s (~3945–4017 env-steps/s); best fitness −7.3785 / −7.3180 / −6.0399; 44 steps in 4.4 s including first-generation warm-up.
- Timeout probe (`--generations 1 --episode-length-s 0.1 --horizon 40`, so `max_episode_length=5`): ends timeout=12 tilted=12, 40 steps in 0.475 s (84.2 steps/s, 2022 env-steps/s).
- Reproducibility (NP=8, horizon 25, `--master-seed 0`, run twice): identical seed table `[[1454269175, 757617120, 1421746521], [1313752747, 908599966, 53557616], [1683657442, 1375977993, 1654802006]]`, `fitness` arrays byte-identical for all generations, identical returns/episode_returns/steps.
- Zero perturbation (`--no-perturbation`, NP=8, horizon 25, 2 generations): initial observation bitwise identical across all rows (max peak-to-peak 0.0), fitness −7.4132 / −7.4132; the seed table still changes per generation while `init_qpos` does not; perturbed runs deviate from the base init qpos by at most 0.0492 per component (0.0802 in joint space).
- CUDA/`iree` graph compile (measured, `iree-base-compiler`/`iree-base-runtime` 3.13.0): `build_obs`, `build_advance`, `build_fitness` and `build_policy` all compile on evoxc `iree`/`cuda:0`; only `build_tell` fails — `iree-compile` reports `error: 'func.func' op failed to distribute` on a `stablehlo.reduce` of the tell graph, i.e. an IREE GPU-codegen limit on TensorNEAT's own reduction, not on our kernels (`--iree-cuda-target=sm_60`/`sm_120` fail identically).
  End-to-end `-g` training therefore still stops at the first tell; CPU/numpy remains the only fully measured trainer path.
- Cross-device parity, CPU/numpy reference vs `iree`/`cuda:0` (measured, RTX 5070): `build_obs` output bitwise identical (0/1080 elements differ); `build_advance` obs/returns/alive/done/episode_length identical, `reward` max abs diff 2.8e-09; a full generation-0 `Go2NeatEnv.evaluate` (NP=8, horizon 25) gives the identical seed triple and `init_qpos`, 11 steps on both, `tilted=24` on both, bitwise-identical `fitness`, and `episode_returns` within 4.8e-07 (1 ulp).
- ROCm (`--evoxc-device rocm:0`, or `EVOXC_DEVICE=rocm:0`, with the same iree extras) stays UNMEASURED — no AMD GPU here — and would hit the same `build_tell` codegen failure.

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
- `go2_neat_kernels.py` is deliberately one module (~1000 lines, mostly per-graph docstrings): it is the single import surface the env/train modules build against, so do not split it.
- `go2_reward.png` is a git-ignored run artifact next to the script, never tracked.

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
- Under pytest (`PYTEST_VERSION` in the environment) `try1.py` creates no viewer and runs 5 steps, mirroring the shortened `go2_backflip.py` playback, so the example smoke test cannot hang on the viewer loop; explicit `--headless` (250 steps) and `--max-steps N` still win over that cap.
- `scene.viewer` is `None` whenever the viewer is not shown, so `try1.py` only touches `scene.viewer` (follow / `is_alive`) when it actually created one.
- The gait `step` input must be a 0-d `int32` ndarray; `np.ascontiguousarray` promotes 0-d to shape `(1,)` and the executable rejects it.
- Static ndarray leaves specialize the graph but must still be passed to `evoxc.run` on every call (validated with `array_equal`).
- Graph-time constants need `evoxc.constant(evoxc.from_numpy(arr))`; `evoxc.zeros` is a concrete creator, not a graph op.
- `scene.clear_errno()` must be called for every row the graphs report in `error_mask` (and once per generation reset): a latched solver error otherwise re-raises `GenesisException: Invalid constraint forces causing 'nan'` on the next `scene.step()` although the graph already marked those rows dead.
- The trainer's per-generation `end_reasons` count comes from the graphs' `done`/`alive` state plus the genesis error mask; `tilted` dominates for untrained policies at the full horizon.
- `max_collision_pairs=20`: exceeding ~20 collision pairs raises an exception; keep policy contact sparse.
- `base_init_quat` differs between walking (`[1,0,0,0]`) and backflip (`[0,0,0,1]`) configs; the backflip robot starts inverted.
- `go2_neat_kernels.py` bakes its static operands (tiled `standing_flat`/`commands`, `dt`, thresholds, `max_episode_length`) into each graph and re-passes them on every call; use the `run_*` wrappers (they read them off the executable) instead of calling `evoxc.run` directly.
- `run_advance` takes `base_quat` already relative to the init pose (identity for the Go2 config) and world-frame `get_vel()`/`get_ang()`; the graph rotates by `inv(base_quat)`.
- In `run_advance` the timeout fires when `episode_length + 1 > max_episode_length`; the dying step's reward is counted exactly once, `alive` never re-arms, and `episode_length` restarts at 0 on that step.
- Roll/pitch termination is `|sinp| > tanθ·cosp` and `|sinrcosp| > tanθ·cosrcosp`, with the `cosrcosp·cosp <= 0` quadrant guard; do not add a second `cosp` factor (that shrinks the roll threshold with pitch and fires false positives).
- `build_policy` logs an `evoxc` `UserWarning` about embedding the pop batch as a >1 MiB graph constant (raise `EVOXC_LARGE_CONSTANT_BYTES` if `pop_size` grows).
- Pass a positive `axis` to `evoxc.concatenate`: evoxc copies a negative axis verbatim into StableHLO (`dimension = -1 : i64`) and `iree-compile` rejects it as a non-negative-integer attribute violation; `evoxc.stack` normalizes its own axis (`stack(..., axis=-1)` emits `stablehlo.reshape` + `dimension = 2`).
- Backflip `.pt` checkpoints are data (torch `state_dict`s), not code; they are loaded with `torch.load(..., weights_only=True)`.
- `go2_train.py` expects a GPU for the 4096-env throughput; `go2_eval.py`/`go2_backflip.py`/`try1.py` run on CPU or GPU.

## Test Strategy
- `tests/test_examples.py` smoke-tests `try1.py` as a subprocess: no viewer, 5 steps under `PYTEST_VERSION`, and skipped where `evoxc` is missing via the `EXAMPLE_DEPENDENCIES` entry for `try1.py`.
- That harness still ignores `*_train.py`, `*_eval.py`, `*_env.py`, `behavior_cloning.py`, `quadcopter_controller.py`; `go2_backflip.py` is the other smoke test here and shortens playback to 5 steps when `PYTEST_VERSION` is set.
- `go2_evox_kernels.py` and `go2_legacy_reference.py` are excluded from that smoke run as shared helper modules, not runnable demos.
- `tests/test_go2_evox_parity.py` (`@pytest.mark.required`, `importorskip("evoxc")`) covers the graph math without a simulator: 250 chained gait steps (5e-5 tolerance), exact gait semantics at steps 0/25/26, and 200 randomized states (1e-5 per reward term, 1e-6 on the total) against the `go2_legacy_reference.py` oracle.
- No committed test covers `go2_neat_kernels.py`; verify a change with throwaway scripts against `go2_legacy_reference.walking_reward_terms` (reward), an independent numpy `go2_env._update_observation` (obs, 1e-6) and `genesis.utils.geom.quat_to_xyz(..., rpy=True, degrees=True)` (10k quats incl. near-threshold and pitch≈±90°, exact agreement on the roll-or-pitch decision).

## Routing Table
- `backflip/` → Pre-trained single/double backflip policy checkpoints + readme (data, not code).
- `../tests/test_examples.py` → Example smoke-test runner (sibling — read-only, escalate writes to parent).
- `../tests/test_go2_evox_parity.py` → Numeric parity tests of `go2_evox_kernels.py` against `go2_legacy_reference.py` (sibling — read-only, escalate writes to parent).
