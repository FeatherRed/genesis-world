# tests — Genesis World test suite

## Intent
Pytest suite for the `genesis-world` physics engine (v1.4.6).
Covers rigid-body solvers (with MuJoCo parity), deformables (FEM/muscle), particles (MPM/PBD/SPH), sensors, rendering (snapshot-based), asset parsers, IPC, gradients, benchmarks, and end-to-end scenarios.
Tests run on multiple backends (cpu/gpu/cuda/amdgpu/metal) and both field (`GS_ENABLE_NDARRAY=0`) and ndarray (`GS_ENABLE_NDARRAY=1`) data layouts.

## How to run
- Default addopts (from `pyproject.toml` `[tool.pytest.ini_options]`): `--verbose -r EsxXw --import-mode=importlib --numprocesses=auto --dist=worksteal --durations-min=100.0 --timeout=1200 -m "not (benchmarks or examples)"`.
- `slow` tests are **included** by default (only `benchmarks`/`examples` are excluded); they are reordered to run first.
- Minimal merge set: `pytest -m required`.
- Skip slow: `pytest -m "not slow"`.
- Benchmarks (exclusive, GPU-only): `pytest -m benchmarks`.
- Examples (runs `examples/*.py` as subprocesses): `pytest -m examples`.
- Backend: `pytest --backend {cpu,gpu,...}` (default cpu); field vs ndarray is selected by `GS_ENABLE_NDARRAY` env, not a CLI flag.
- Snapshot regeneration: `--snapshot-update` (forced serial; writes rendering/trajectory snapshots).
- `--failure-artifacts-dir DIR` writes the image a mismatching snapshot test rendered.
- `--mem-monitoring-filepath FILE` launches `monitor_test_mem.py` (Linux + GPU management library only).
- EvoX Compiler work environment: `/root/evox-robotics/.venv` has no pytest and a read-only `site-packages`, so the suite is invoked with a scratch plugin tree: `PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest <paths>`.
- Recreate the scratch tree with `python -m pip install --target /tmp/gs-pytest-libs pytest pytest-xdist pytest-timeout pytest-print syrupy pytest-forked` if it is missing (the venv supplies the other conftest imports).
- Purely CPU/numpy modules (`test_hsa_runtime.py`, `test_hover_*.py`, `test_neuroevolution_trainer.py`) put `pytest.mark.parametrize("backend", [None])` in `pytestmark`, which makes the autouse `initialize_genesis` fixture early-return and skips the per-test `gs.init`/`gs.destroy` (~2.5 s/test); `test_examples.py` uses the same switch per test.

## API Surface (top-level files)
- `conftest.py` (root, ~1300 lines) — the suite's shared brain; see Design Decisions below.
  - Session fixtures: `backend`, `show_viewer`, `tol`, `asset_tmp_path`, `cube_verts_and_faces`, `box_obj_path`.
  - Autouse `initialize_genesis` fixture: per-test `gs.init(backend, precision, seed=0, deterministic, performance_mode, ...)` and `gs.destroy()`, with backend/dtype skip logic.
  - Marker-driven option fixtures: `precision`, `friction_cone`, `adjacent_collision`, `gjk_collision`, `enable_collision`, `disable_constraint`, `friction_torsional`, `friction_rolling`, `merge_fixed_links`, `multi_contact`, `dof_damping`, `cache`, `performance_mode`, `debug`, `use_deterministic_algorithms`, `mujoco_compatibility`.
  - MuJoCo parity fixtures: `xml_path`, `xml_paths`, `mj_sim`, `gs_sim` (wrap `tests/utils/simulators.py`).
  - Snapshot fixtures: `png_snapshot` (custom syrupy `PixelMatchSnapshotExtension`) and `load_trajectory_snapshot`.
  - Skip-reason registry (`SKIP_NO_GPU`, `SKIP_NO_MADRONA`, `SKIP_NO_LUISA`, `SKIP_NO_VIEWER`, `SKIP_NO_IPC`, `SKIP_METAL_64BIT`, …) with skip-location normalization for grouped summaries.
- `gpu_info.py` — cross-vendor GPU discovery (`NvidiaBackend`/`AmdBackend` via NVML/AMD SMI, imported lazily) and `detect_gpu_backend()`.
- `monitor_test_mem.py` — standalone per-process VRAM monitor; parses pytest process titles, writes pipe-delimited `key=value` results.
- `torch_block.py` — the "PyTorch unavailable" harness (helper module, not collected): `TORCH_BLOCK_SOURCE` (a `sys.meta_path` finder raising `ImportError` for `torch`/`torch.*`, prepended before any import), `run_python`/`run_python_checked` (child `python -c` with the blocker, a leak-check epilogue that fails the child if `torch` reached `sys.modules`, and a writable `MPLCONFIGDIR`), `torch_import_violations`/`assert_no_torch_imports` (AST scan of files, directories and globs for `import torch`/`from torch ...`, with `allowlist` and `exclude`; missing paths are ignored but an empty scan is an error), plus `REPO_ROOT`/`EXAMPLES_DIR`/`GENESIS_DIR` and `has_gpu()` (reuses `gpu_info.detect_gpu_backend()`, falling back to the `/dev/kfd`/`/dev/nvidiactl` device nodes, because `amdsmi`/`pynvml` are not installed).
- `test_examples.py` — discovers and runs `examples/*.py` as subprocesses (marker `examples`; optional-dependency gates, including `evoxc` for `locomotion/try1.py`, `evoxc`+`tensorneat` for `drone/hover_neat_check.py` and the whole `neuroevolution/` tree, and `tensordict` for `locomotion/go2_backflip.py`).
  `IGNORE_SCRIPT_PATTERNS` excludes `*_env.py`/`*_train.py`/`*_eval.py`, the `go2_*` kernel/oracle/device helpers, `hover_kernels.py`, `hover_legacy_reference.py`, `hover_device_rollout.py`, and every `neuroevolution/*.py` helper (including `selfcheck.py` and everything under `neuroevolution/benchmarks/`); `drone/hover_neat_check.py` stays collected as the drone smoke.
  `_discover_examples()` matches those patterns with `fnmatch` against each script's POSIX path relative to `examples/` (plus its bare filename), so `*` crosses `/` and one subtree pattern covers nested directories, which `Path.match` cannot do.
- `test_go2_evox_parity.py` — `required`; loads `examples/locomotion/go2_evox_kernels.py` and `go2_legacy_reference.py` by file path, skips when `evoxc` is absent, and compares 250 chained gait steps plus 200 randomized reward states against the legacy oracle.
- `test_go2_neat_kernels.py` — `required`; loads the task graphs from `examples/locomotion/go2_neat_kernels.py` and the generic objects from the shared `examples/neuroevolution` layer (`policy_graphs.build_policy/run_policy`, `evolution_graphs.build_fitness/run_fitness/ExecutableCache`, `seeding.EPISODES_PER_POLICY`, `trainer.TraceOnlyProblem`), skips when `evoxc`/`tensorneat` are absent, and checks the TensorNEAT graph surfaces: policy-major `3p..3p+2` row tiling, first-done truncation (including the `max_episode_length` timeout and the never-re-arming `alive` mask), `run_fitness` = per-policy episode mean, observation/reward parity against `go2_env` and `go2_legacy_reference`, roll/pitch termination parity against `quat_to_xyz(rpy=True, degrees=True)`, and `ExecutableCache` hit/miss/clear behavior.
- `test_go2_neat_seeding.py` — `required`; covers the seed plumbing of `examples/locomotion/go2_neat_env.py` against the canonical `seeding` module (`EPISODES_PER_POLICY`, `Perturbation`, `evolution_seed`, `generation_seed_table`, and `perturbation_from_seed(..., num_dof=NUM_ACTIONS)` through a thin `num_dof` adapter): seed-table shape/reproducibility/distinctness, independence from `evolution_seed`, perturbation determinism/bounds/draw-order, `perturbed_init_qpos` row layout, seed indexing and world-axis rotation composition (`R_pert @ R_init`), and the zero-perturbation fixed reset.
- `test_hsa_runtime.py` — `required`, GPU-free and re-exec-free; 62 pure unit tests of `genesis/utils/hsa.py` driven by a fake torch install and a fake ROCm library directory in `tmp_path` (never the host's `/opt/rocm`), covering the detection status, `system_hsa_runtime_path`, `LD_PRELOAD` matching by path and sha256, the sentinel/opt-out gates, `_warn_only_context`, and every `ensure_system_hsa_runtime` gate up to a recorded (never real) `os.execve`.
- `test_hover_evox_parity.py` — `required`, CPU/numpy only; mirrors `test_go2_evox_parity.py` for `examples/drone`: 200 randomized states for the `(rows,17)` observation and the five reward terms, the RPM affine map/clipping, the `end_code`/crash/timeout/done/at_target masks and next commands, the analytic reset observation, the `returns` credit rule, and the yaw transcription over ~20k random + 200 near-gimbal quaternions.
- `test_hover_neat_seeding.py` — `required`; scene-free (`HoverNeatEnv.__new__`) checks that the drone env's episode seed triples match `seeding.generation_seed_table`, that `HoverNeatEnv` re-exports the canonical `seeding` objects, that rows are policy-major (`r % 3` slot mapping, rows of a policy share the three perturbations), that a zero perturbation reproduces the tiled init pose bit-for-bit, and that per-row command streams are independent of row/step order.
- `test_neuroevolution_trainer.py` — `required`, CPU/numpy only and under ~10 s; pins the shared contract: `env_base.validate_evaluation` accept/reject matrix, the documented seed constants (`evolution_seed(0) == 3757552657` and `generation_seed_table(0, 3)`), prefix stability and evolution/environment stream independence, the width-generic `policy_graphs` graph at `45->12` and `17->4` (plus bit-identical agreement with the Go2 adapter at `45->12` when it exists), `NeuroevolutionTrainer` determinism over 3 generations at one master seed vs a different master seed, the float32 episode-mean fitness rule, and the in-memory artifact behavior.
- `test_neuroevolution_device.py` — `required`; the placement contract of `examples/neuroevolution`'s `residency="device"` path on the first reachable non-CPU `evoxc` device (`iree` by default, `EVOXC_BACKEND`/`EVOXC_DEVICE` override): device-first initialization of the payload/species/key/counter, the 45→12 policy graph's device output (with a host `run_policy` control), the four device `evaluate` metrics plus `validate_evaluation_device` rejections, `run_tell` outputs with the counter advanced by one, a two-generation run whose population stays resident, the negative placement predicates, the accelerator-free `numpy@cpu:0` plumbing mode, and pop 32/64 at 1 generation (`slow`). Accelerator tests skip cleanly without a device; one shared `ExecutableCache` means only the first device test pays the cold compiles (~40 s total on an MI300X VF, `iree@rocm:0`).
- `upload_benchmarks_table_to_wandb.py` — standalone W&B uploader for benchmark result files (memory/perf).

## Markers (declared in pyproject.toml, resolved in conftest.py)
- `required` — minimal merge set; `required("pairwise")` / `required("sparse")` keep a per-backend subset of a parametrized test (resolved before `-m` selection).
- `slow` — baseline runtime `>= 150s`; optional backend args, e.g. `slow("gpu")`, restrict the flag to those backends.
- `benchmarks` — exclusive (may not combine with other markers) and GPU-only; `--backend cpu` is rejected/coerced to gpu.
- `examples` — example scripts.
- Option markers (resolved to fixture values): `precision("32"|"64")`, `friction_cone`, `adjacent_collision`, `gjk_collision`, `enable_collision`, `disable_constraint`, `friction_torsional`, `friction_rolling`, `merge_fixed_links`, `multi_contact`, `dof_damping`, `cache`, `performance_mode`, `debug`, `use_deterministic_algorithms`, `mujoco_compatibility`, `split_entities`.
- `xfail_strict = true` is set project-wide.

## Routing Table
- `rigid/` → Largest rigid-body suite (API, asset loading, collision, nonconvex, constraints, control, dynamics, friction, heterogeneous, islands, kinematics, MuJoCo parity, narrowphase, serialization, sparse, terrain); its `conftest.py` is a 1500-line inline MJCF/URDF model-fixture library.
- `parsers/` → Mesh and USD parsing (GLB/OBJ/STL/MJCF/URDF/USD), equivalence and tolerance checks; conftest holds ~30 asset-generator fixtures.
- `rendering/` → Offscreen/batch/debug-draw/dynamic-mesh/ImGui/interactive-viewer tests with syrupy snapshots from HuggingFace `Genesis-Intelligence/snapshots`.
- `particles/` → MPM/PBD/SPH solver tests (materials, sampling, coupling, stability, pressure).
- `deformable/` → FEM and muscle simulation (implicit/explicit, couplers, material models, vertex constraints).
- `sensors/` → Camera/contact/IMU/joint-torque/raycaster/tactile/temperature and SensorManager API, mostly analytic ground truth.
- `grad/` → Differentiable simulation: analytic-vs-finite-difference gradients, torch autograd semantics, optimizer convergence.
- `core/` → Engine internals: backend selection, BVH, Quadrants, recorders, surface, misc utils.
- `coupling/` → Hybrid (CPIC) particle-rigid coupling, SAP coupler, SPH-rigid buoyancy/hibernation.
- `integration/` → End-to-end scenarios (pick-and-place, hanging cable, Franka grasping a FEM entity).
- `ipc/` → uipc-backed Incremental Potential Contact coupler (api/deformable/rigid).
- `benchmarks/` → GPU perf/memory benchmark sweep feeding W&B tables.
- `utils/` → Shared helpers: assertions, MuJoCo parity simulators, HuggingFace assets, collision ground truth (read by most groups).

## Constraints
- **No eager heavyweight imports at conftest module scope**: `genesis`, `numba`, `numpy`, `quadrants`, `torch` must only be imported via `lazy_imports()` or after `pytest_sessionstart` configures the worker (a `pytest_sessionstart` hook raises if they are imported early).
- MuJoCo OpenGL (`MUJOCO_GL=0`) and tqdm (`TQDM_DISABLE=1`) are force-disabled; headless pyglet is configured when no display is available.
- Assets and snapshots are NOT committed — they are downloaded at runtime from HuggingFace `Genesis-Intelligence/{assets,snapshots}` at pinned revisions (see `utils/assets.py`).
- Tests must destroy Genesis themselves or rely on the autouse `initialize_genesis` fixture (which calls `gs.destroy()` in a `finally`).

## Design Decisions
- A single autouse fixture centralizes `gs.init`/`gs.destroy` and backend/dtype skipping instead of each test initializing Genesis.
- Option surfaces (friction cone, GJK, adjacent collision, …) are expressed as markers so a sweep can be requested per test without hardcoding in fixtures.
- MuJoCo parity is centralized in `utils/simulators.py` + `utils/mujoco_parity.py`; the root `mj_sim`/`gs_sim` fixtures build both engines from the same MJCF/URDF bytes and step them in lockstep.
- xdist workers are sized from RAM/VRAM/cores (`pytest_xdist_auto_num_workers`), and each worker gets GPU affinity (`CUDA_VISIBLE_DEVICES`/`QD_VISIBLE_DEVICE`) plus thread-count caps; EGL device is matched to the CUDA device per test.
- Canonical skip reasons normalize their report locations so identical skips stack into a single summary line.
- `benchmarks` is opt-in and exclusive; `--snapshot-update` forces serial execution because syrupy writes snapshots in the main process.

## Known Issues
- The root `mj_sim` fixture forwards its `gjk_collision` flag into `build_mujoco_sim`'s `native_ccd` positional parameter (naming/semantic mismatch in `tests/utils/simulators.py`); verify intent before changing either side.
- `mj_sim`/`gs_sim`/`xml_path` depend on `model_name`/`gs_solver`/`gs_integrator`, which are **parametrize arguments** defined only in directories like `rigid/` (not fixtures) — requesting these parity fixtures in a directory that does not parametrize those names fails collection.
- Cross-sibling fixture dependency: `deformable/test_fem.py` uses the `xml_path` fixture defined only in `parsers/conftest.py`, so running `tests/deformable/` in isolation fails collection (see `deformable/CONTEXT.md`).
- GPU management libraries (`nvidia-ml-py`, `amdsmi`) are optional; without them multi-GPU affinity degrades to single-GPU (a warning is emitted).
- `env_base.validate_evaluation` and `trainer._aggregate` share the same rule: every non-finite episode return (NaN and ±inf) counts as 0, aggregated in float32; `test_neuroevolution_trainer.py` pins that rule (a task reporting the non-finite-counted-as-0 mean validates, a NaN-only or offset fitness is rejected); `examples/neuroevolution` is read-only for this node.
- `examples/locomotion/go2_backflip.py` imports `go2_env.py`, which imports `tensordict`; `tensordict` is undeclared in `pyproject.toml` and absent from the EvoX Compiler venv, so `EXAMPLE_DEPENDENCIES` gates that example on `tensordict` (it skips there and runs where the dev extra is installed).
- The device-resident DLPack rollouts (`examples/locomotion/go2_device_rollout.py`, `examples/drone/hover_device_rollout.py`) need a non-CPU `evoxc` backend (IREE) and are GPU-only; no CPU test exercises the bridge — they are excluded from the example harness and only their `--residency-parity` smoke covers them on a GPU host.
- `examples/neuroevolution/benchmarks/` has no stable CLI and is untested (and not collected by the example harness).

## Test Strategy
- Fast correctness tests are the default; `slow` (>=150s) run in the default suite but are scheduled first so they overlap with the rest.
- `-m required` is the merge gate; `benchmarks`/`examples` are opt-in CI jobs.
- Parity is validated both statically (model constants) and dynamically (per-step qpos/qvel/forces) against MuJoCo.
- The Go2 EvoX Compiler graphs are checked against the legacy oracle with float32 tolerances: 5e-5 on gait outputs, 1e-5 per reward term, 1e-6 on the total.
- The TensorNEAT Go2 stack is tested at the helper/kernel level only (no rollouts, no viewer, no training run): 1e-6 on observation assembly against a numpy reimplementation of `go2_env._update_observation`, 1e-5 per reward term against `go2_legacy_reference.walking_reward_terms`, and exact agreement on the roll/pitch termination decision over randomized, near-threshold and pitch≈±90° quaternions.
- Rendering correctness is snapshot-based (blurred std/ratio thresholds) rather than exact pixel equality.
- Coverage runs across backend × dtype (field/ndarray) and, where supported, cpu/gpu.
- The four neuroevolution/drone/HSA modules are CPU-only, network-free and fast: `80 passed in ~14 s` under `-n 2`, and the whole 99-test touched set passes under the default 20-worker xdist in ~25 s.
- Drone graph parity: observation, RPM, masks, next commands and the reset observation are compared exactly; reward terms use 1e-5 per term and 1e-6 on the total (measured 0..8.4e-11 and 1.5e-8); yaw uses 5e-4 deg (measured 1.5e-5 deg).
- HSA tests isolate completely from the host (fake torch `lib/`, fake ROCm dir, monkeypatched `_candidate_lib_dirs`, recorded `os.execve`): no GPU, no re-exec, no `/opt/rocm` read.
- The shared trainer test uses a tiny simulator-free `FakeEnv` from `selfcheck.py` (copied, not imported) with the numpy backend and minimal genome budgets, so determinism is checked without a scene.

## Dependencies (beyond the package manager)
- HuggingFace network access for the `Genesis-Intelligence/assets` and `Genesis-Intelligence/snapshots` datasets.
- `evoxc` (+ `tensorneat`) for the Go2, drone and shared-neuroevolution tests — both optional and not declared in `pyproject.toml`, so those modules `importorskip` and skip silently when absent.
- `tensordict` is needed by `examples/locomotion/go2_env.py` (and therefore `go2_backflip.py`) and by the PPO baselines; it is not declared in `pyproject.toml` and is absent from the EvoX Compiler venv, so the example harness cannot run the Go2 RL demos there.
- Optional renderers/backends: `gs-madrona` (BatchRenderer), `LuisaRenderPy` (RayTracer), `imgui-bundle` (overlay), `uipc` (IPC coupler), `pxr`/usd-core + omniverse-kit (USD), NVML/AMD SMI for GPU introspection.
- pytest plugins: `pytest-xdist`, `pytest-forked`, `pytest-timeout`, `pytest-print`, `syrupy` (see `pyproject.toml` `dev` extra).

## See Also
- `../pyproject.toml` — `[tool.pytest.ini_options]` (addopts, markers, filterwarnings) and `dev`/`render`/`usd` extras.
- `../genesis/` — the engine under test (source, not modified by this suite).
- `../examples/` — run by `test_examples.py`.
