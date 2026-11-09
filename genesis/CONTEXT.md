# genesis/ — The `genesis` Package (user-facing API)

## Intent
The Python package behind Genesis World (imported as `import genesis as gs`, package name `genesis-world` v1.4.6).
It exposes the user-facing simulation interface and assembles the three backend layers — physics (`engine/`), render (`vis/` + `ext/`), compiler (external `quadrants`) — behind one `gs.Scene` object.
This node documents the package root (top-level modules + `logging/` + `assets/`); the heavy subsystems live in the child directories below.

## API Surface (top-level modules)
- `__init__.py` — the public namespace and lifecycle: `gs.init(...)` / `gs.destroy()`, `gs.register_external_module(init_fun, destroy_fun)` / `unregister_external_module`, `gs.GenesisException`.
  It configures Quadrants + torch, sets backend-dependent dtypes (`gs.qd_float`/`gs.np_float`/`gs.tc_float`, `gs.qd_vec3`, `gs.qd_mat3`, int/bool equivalents), zero-copy mode, and precision (32/64).
  It re-exports: `gs.Scene`, `gs.Mesh`, `gs.states`, `gs.materials`, `gs.force_fields`, `gs.morphs`, `gs.sensors`, `gs.renderers`, `gs.surfaces`, `gs.textures`, `gs.recorders`, `gs.tools`, `gs.UID`, `gs.List`, `gs.utils.geom.*`, misc helpers (`raise_exception`, `assert_built`, ...), and the constants enums (`gs.JOINT_TYPE`, `gs.GEOM_TYPE`, `gs.CTRL_MODE`, `gs.integrator`, `gs.constraint_solver`, `gs.friction_cone`, `gs.contact_resolution`, `gs.broadphase_traversal`, `gs.link_ref_frame`, `gs.backend` with members `cpu/gpu/cuda/amdgpu/metal`).
- `_main.py` — the `gs` CLI (pyproject `[project.scripts] gs = "genesis._main:main"`): `gs launch` / `gs view` (deprecated alias) to inspect an asset in the viewer, `gs play` (ImGui joint controls), `gs replay` (`.gstraj` trajectory), `gs animate` (images → video).
- `constants.py` — file-format constants (`URDF_FORMAT`, `MJCF_FORMAT`, `MESH_FORMATS`, `USD_FORMATS`, `XACRO_FORMAT`, `GLTF_FORMATS`) and the enums above (rigid solver geometry/joint/equality/control types, integrator/constraint/friction/contact-resolution/broadphase options, `backend`, `IMAGE_TYPE`, `PARA_LEVEL`, `ACTIVE`/`INACTIVE`).
- `datatypes.py` — `gs.List[T]`: a pydantic-compatible, richly-repr'd list used everywhere for entity/link/geom collections.
- `repr_base.py` — `RBC` (REPR Base Class): every Genesis class inherits it to get the colorized, property-listing `__repr__` and `__repr_name__` (`<gs.engine.entities.RigidEntity>` style).
- `styles.py` — log theming: `THEME` (`dark`/`light`/`raw`), the `~kind<...>~` log markup parser, `COLORS`/`FORMATS`, and the module-level `style`/`colors`/`formats` singletons.
- `typing.py` — pydantic `Annotated` type aliases used by `options/` (`Vec3FType`, `UnitVec3FType`, `RotationMatrixType`, `FrozenDictType`, ...) with strict validators (vector normalization, int coercion rejecting bool/float).
- `version.py` — `__version__ = "1.4.6"`.
- `logging/` → `Logger`, `GenesisFormatter`, `TimeElapser` (see its CONTEXT.md).

## Routing Table
- `engine/` → core physics engine: Scene/Simulator, entities, solvers, materials, sensors, couplers, boundaries, states.
- `utils/` → shared helpers: geometry math, mesh/asset parsers (URDF/MJCF/SDF/USD/GLTF), array/cache/tools; `misc.py`/`geom.py` are the canonical helpers every module must use.
- `vis/` → visualization: Visualizer, Viewer, Camera, Rasterizer, Raytracer, BatchRenderer, viewer plugins.
- `ext/` → vendored third-party code: pyrender fork, urdfpy fork, isaacgym terrain utils, `_trimesh_patch`, and the `LuisaRender/` gitlink submodule (render backends).
- `options/` → pydantic option classes (SceneOptions, solver/coupler/sensor/renderer/morph/surface/texture options) consumed by `engine/` at build time.
- `recorders/` → recording/export: video/CSV/NPZ writers, `.gstraj` trajectories, plotters, RecorderManager.
- `grad/` → differentiable-simulation bridge: RolloutGraph, LaunchRecorder, kernel-backward decorators.
- `logging/` → logger + elapsed-time helper (tiny, documented here).
- `assets/` → bundled static data: meshes, textures, URDF/MJCF robot models, sample XML scenes (data only, no Python).

## Design Decisions
Scenes are authored declaratively (`Scene` + morph/mesh/material/surface/sensor/recorder options) and resolved into solver buffers once at `Scene.build` — descriptions (options) are plain pydantic dataclasses, live entities only index into solver state.
One `Simulator` owns exactly one solver of each kind plus one coupler, so rigid/FEM/MPM/PBD/SPH share one scene and one state; substeps are reconciled to a single rate.
All hot-path computation runs as Quadrants kernels (Python lowered to CUDA/ROCm/Metal/Vulkan/x86/ARM); the `gs.backend` enum abstracts the device and `gs.use_ndarray` toggles static (performance) vs dynamic (fast-compile) array mode.
Autodiff replays a recorded rollout (`grad/`) rather than in-place autograd through the C++ solver.
`RBC`/`styles` give Genesis its distinctive colorized, markup-tagged console output (`~<value>~`, `~~<name>~~`, `~~~<uid>~~~`, `~~~~<title>~~~~`).

## Constraints
Coding conventions are enforced project-wide (see `../CLAUDE.md`, `../CODING_GUIDELINES.md`, `../.github/contributing/`): the rigid solver (`engine/solvers/rigid/`) is the reference implementation for naming and kernel structure.
No `getattr`/`hasattr`, no on-the-fly attributes, no plain dicts for attribute packing, no commented-out prints, no `detach().cpu().numpy()` (use `utils.misc.tensor_to_array`), no local imports unless required to break cycles.
All state-mutating engine methods must be tagged `@mutates` so sensors and the rollout graph see mutations.
External code in `ext/` does not follow these conventions (vendored).

## Dependencies
Beyond pip (see `../pyproject.toml`): `quadrants==1.4.0` (compiler/kernels), `torch` (install separately, per-backend), `trimesh>=5.0.0`, `mujoco>=3.2.5`, `numpy`.
Optional backends: `pyuipc`/`libuipc` (IPC coupler), `usd-core`/`pxr` (USD), LuisaRenderPy (ray tracer, built from `ext/LuisaRender` submodule), `gs-madrona` (Linux batch renderer), `rsl-rl-lib` (RL examples).

## Test Strategy
Package behavior is covered by the sibling `../tests/` tree, not inside this node (`tests/rigid/`, `tests/particles/`, `tests/deformable/`, `tests/sensors/`, `tests/grad/`, `tests/parsers/`, `tests/rendering/`, ...).
Run `pytest -m required` (minimal merge set); default addopts exclude `benchmarks`/`examples` markers and run `--numprocesses=auto --dist=worksteal --import-mode=importlib`.

## Notes for Agents
`genesis/ext/LuisaRender` and `doc/` are git submodules that arrive as EMPTY placeholders in worktrees; populate with `git submodule update --init` if their content is needed (never delete the placeholder or commit inside them).
`assets/` is ~hundreds of binary files (meshes/textures/robot models) — treat as static data; do not refactor or split it.
Each child directory's CONTEXT.md holds the detailed per-module API; drill down before re-investigating.

### Device-memory interop (Quadrants ↔ torch)
Solver state lives in Quadrants device buffers and reaches torch only through DLPack; there is no `__cuda_array_interface__` and no Python-level `data_ptr` on `qd.Tensor`.
`genesis.utils.misc.qd_to_torch(field, copy=False)` returns a torch tensor that aliases the Quadrants buffer and caches it on the qd impl as `_tc`/`_T_tc`; entity/scene getters call it with `copy=True` (a fresh device clone — never a host copy).
`Field`/`Ndarray` export DLPack via `to_dlpack()` / `to_torch(copy=False)`; the `qd.Tensor` wrapper does not itself implement `__dlpack__`, so call `to_dlpack()` or use the cached torch view.
On ROCm (`qd.init(arch=qd.amdgpu)`) Quadrants memory appears to torch as `torch.device("cuda", 0)` and its DLPack device is `(kDLROCM=10, 0)`; `gs.device` is `torch.device("cuda", 0)` and `gs.use_zerocopy.is_supported` is True.
`gs.tc_float`/`gs.tc_int`/`gs.tc_bool` are float32/int32/bool (Quadrants f32/i32/bool), while `gs.use_ndarray` defaults to True; recording (autodiff) disables zero-copy, so `copy=False` views are unavailable in a recording scene.

## See Also
- `../README.md`, `../pyproject.toml`, `../CLAUDE.md`, `../CODING_GUIDELINES.md` — project overview and conventions.
- `../tests/` — test suites (sibling, read-only).
- `../examples/` — runnable demos (sibling, read-only).

## Status
Complete: package root and all child subpackages (`engine/`, `utils/`, `vis/`, `ext/`, `options/`, `recorders/`, `grad/`, `logging/`, `assets/`) are documented.
