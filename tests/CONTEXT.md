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
- `test_examples.py` — discovers and runs `examples/*.py` as subprocesses (marker `examples`; optional-dependency gates, including `evoxc` for `locomotion/try1.py`; shared helper modules — `*_env.py`, `*_train.py`, `*_eval.py` and the explicit `go2_*` kernel/oracle files — are ignored).
- `test_go2_evox_parity.py` — `required`; loads `examples/locomotion/go2_evox_kernels.py` and `go2_legacy_reference.py` by file path, skips when `evoxc` is absent, and compares 250 chained gait steps plus 200 randomized reward states against the legacy oracle.
- `test_go2_neat_kernels.py` — `required`; loads `examples/locomotion/go2_neat_kernels.py` by file path, skips when `evoxc` is absent, and checks the TensorNEAT graph surfaces: policy-major `3p..3p+2` row tiling, first-done truncation (including the `max_episode_length` timeout and the never-re-arming `alive` mask), `run_fitness` = per-policy episode mean, observation/reward parity against `go2_env` and `go2_legacy_reference`, roll/pitch termination parity against `quat_to_xyz(rpy=True, degrees=True)`, and `ExecutableCache` hit/miss/clear behavior.
- `test_go2_neat_seeding.py` — `required`; covers the seed plumbing of `examples/locomotion/go2_neat_env.py`: `generation_seed_table` shape/reproducibility/distinctness, independence from `evolution_seed`, `perturbation_from_seed` determinism/bounds/draw-order, `perturbed_init_qpos` row layout, seed indexing and world-axis rotation composition (`R_pert @ R_init`), and the zero-perturbation fixed reset.
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

## Test Strategy
- Fast correctness tests are the default; `slow` (>=150s) run in the default suite but are scheduled first so they overlap with the rest.
- `-m required` is the merge gate; `benchmarks`/`examples` are opt-in CI jobs.
- Parity is validated both statically (model constants) and dynamically (per-step qpos/qvel/forces) against MuJoCo.
- The Go2 EvoX Compiler graphs are checked against the legacy oracle with float32 tolerances: 5e-5 on gait outputs, 1e-5 per reward term, 1e-6 on the total.
- The TensorNEAT Go2 stack is tested at the helper/kernel level only (no rollouts, no viewer, no training run): 1e-6 on observation assembly against a numpy reimplementation of `go2_env._update_observation`, 1e-5 per reward term against `go2_legacy_reference.walking_reward_terms`, and exact agreement on the roll/pitch termination decision over randomized, near-threshold and pitch≈±90° quaternions.
- Rendering correctness is snapshot-based (blurred std/ratio thresholds) rather than exact pixel equality.
- Coverage runs across backend × dtype (field/ndarray) and, where supported, cpu/gpu.

## Dependencies (beyond the package manager)
- HuggingFace network access for the `Genesis-Intelligence/assets` and `Genesis-Intelligence/snapshots` datasets.
- `evoxc` (+ `tensorneat`) for the Go2 compiler/NEAT tests — both optional and not declared in `pyproject.toml`, so those modules `importorskip` and skip silently when absent.
- Optional renderers/backends: `gs-madrona` (BatchRenderer), `LuisaRenderPy` (RayTracer), `imgui-bundle` (overlay), `uipc` (IPC coupler), `pxr`/usd-core + omniverse-kit (USD), NVML/AMD SMI for GPU introspection.
- pytest plugins: `pytest-xdist`, `pytest-forked`, `pytest-timeout`, `pytest-print`, `syrupy` (see `pyproject.toml` `dev` extra).

## See Also
- `../pyproject.toml` — `[tool.pytest.ini_options]` (addopts, markers, filterwarnings) and `dev`/`render`/`usd` extras.
- `../genesis/` — the engine under test (source, not modified by this suite).
- `../examples/` — run by `test_examples.py`.
