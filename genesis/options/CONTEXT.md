# Intent
This directory is the configuration/options layer of Genesis World: pydantic `BaseModel` dataclasses that describe every configurable component of a scene (simulator, solvers, coupler, renderer, viewer, visualization, profiler, recorders, morphs, surfaces, textures, sensors).
Options are validated, self-documenting, and serializable bundles of parameters; engine objects (solvers, sensors, visualizer) are constructed from them rather than from raw constructor kwargs.
Options do not hold engine objects or state — they are pure configuration consumed at build time.

# API Surface
The package is a flat set of modules plus one subpackage, re-exported under the `gs` namespace in `genesis/__init__.py`.

## `__init__.py`
Re-exports the top-level option classes and defines `__all__`: `KinematicOptions`, `BaseCouplerOptions`, `CoacdOptions`, `FEMOptions`, `FoamOptions`, `IPCCouplerOptions`, `LegacyCouplerOptions`, `MPMOptions`, `PBDOptions`, `ProfilingOptions`, `RigidOptions`, `SAPCouplerOptions`, `SceneOptions`, `SimOptions`, `SPHOptions`, `ViewerOptions`, `VisOptions`.
`genesis/__init__.py` also exposes whole modules directly: `gs.renderers`, `gs.surfaces`, `gs.textures`, `gs.morphs`, `gs.sensors` (each is a `genesis/options/*` module), while `gs.options.*` reaches the individual classes (e.g. `gs.options.SceneOptions`, `gs.options.VisOptions`).

## `options.py`
Defines the shared base class `Options(RBC, BaseModel)` used by every option class.
Key behavior: `model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)`.
`__init__`/`__setattr__` intercept `ValidationError` and reformat it via `_validation_error` into a concise, aggregated `GenesisException` (groups invalid-type and fixed-length-sequence errors).
`model_copy_from(other, override=False)` merges two option objects field-by-field without re-validating (used for SimOptions inheritance).
`__repr__colorized__` prints a pretty, colorized attribute table.

## `scene.py`
`SceneOptions(Options)` — one object bundling every option a `Scene` is created with; fields are `sim`, `rigid`, `kinematic`, `mpm`, `sph`, `fem`, `pbd`, `coupler`, `vis`, `viewer`, `profiling`, `renderer`.
`_given_or_declared` treats explicit `None` as "use default", so `Scene.__init__` can pass every field unconditionally.
`model_post_init` cross-validates `rigid` against `sim.requires_grad` and then resolves inheritance: each sub-option pulls values it declares but leaves unset from `sim` via `model_copy_from`.

## `solvers.py`
`SimOptions(Options)` — global simulator config (`dt`, `substeps`, `gravity`, `requires_grad`).
Coupler options: `BaseCouplerOptions` base, then `LegacyCouplerOptions`, `SAPCouplerOptions` (Drake semi-analytic primal), `IPCCouplerOptions` (Incremental Potential Contact; many `None` = libuipc default fields).
Solver mixins: `TimeBasedMixin` (`dt`), `GravityMixin` (`gravity`) — both default `None` meaning "inherit from SimOptions".
Solver options: `KinematicOptions`, `RigidOptions(GravityMixin, TimeBasedMixin)`, `MPMOptions`, `SPHOptions`, `PBDOptions`, `FEMOptions` (the particle/solid solvers also use the two mixins).
Grid-based solvers (MPM/SPH/PBD) compute `_hash_grid_res`/`_support_radius` private attrs and validate bounds in `model_post_init`.

## `renderers.py`
`RendererOptions` base; `Rasterizer` (no params; viewer always rasterizes), `RayTracer` (LuisaRender path tracer: device, tracing/roulette, environment sphere, `lights`), `BatchRenderer` (Madrona; `use_rasterizer`).
`SphereLight` (pos/color/intensity/radius) for the ray tracer.
`RayTracer` resolves `env_euler` → `env_quat` (mutually exclusive) in a before-validator.

## `vis.py`
`ViewerOptions` (resolution, thread, refresh rate, realtime factor, camera pose, GUI overlay) and `VisOptions` (world/link frames, shadows, plane reflection, `split_envs`, background/ambient, segmentation level, particle rendering, lights).
Light classes `DirectionalLight`/`PointLight`/`AmbientLight` use a pydantic discriminator on the `type` field.
Deprecated aliases (`max_FPS` → `refresh_rate`, `n_rendered_envs` → `rendered_envs_idx`, `env_separate_rigid` → `split_envs`) are migrated in validators with warnings.

## `morphs.py`
Morphs are a solver-agnostic "geometry + pose" concept: `Morph` base (pos/euler/quat + offset pose, `visualization`/`collision` flags) and its tree:
`Primitive` (adds `fixed`, `contype`/`conaffinity`) → `Box`, `Cylinder`, `Sphere`, `Plane` (Box/Cylinder/Sphere mix in `TetGenMixin` for FEM tetrahedralization).
`FileMorph` (file-loaded assets) → `Mesh` (also `TetGenMixin`), `MeshSet`, `MJCF`, `URDF`, `Drone`, `USD`.
`Terrain` (procedural terrain), `Nowhere` (internal emitter placeholder).
`TetGenMixin` carries tetrahedralization params (`order`, `minratio`, `maxvolume`, ...) shared by FEM-capable morphs.
`euler`/`quat` and `offset_euler`/`offset_quat` are mutually exclusive; euler is resolved to a quaternion (wxyz, scipy extrinsic convention).

## `surfaces.py`
`Surface` base encapsulates the visual information of an entity/link/geom as `gs.textures.Texture` fields, with shortcut scalars (`color`, `opacity`, `roughness`, `metallic`, `emissive`) resolved to `ColorTexture`s in a before-validator.
Direct subclasses: `Glass` (→ `Water`), `Metal` (→ `Iron`, `Aluminium`, `Copper`, `Gold`), `Plastic` (→ `Rough`, `Smooth`, `Reflective`, `Collision`), `BSDF` (→ `Default`), and `Emission`.
`FoamOptions` (from `misc.py`) configures foam generation on particle surfaces.

## `textures.py`
`Texture` base with `ColorTexture`, `ImageTexture` (path/array + encoding srgb/linear, `SerializationMixin`), `BatchTexture` (per-environment texture atlas; built from `image_paths`/`image_folder`/`image_arrays` or colors).
`Texture` restores identity hashing (`__hash__ = object.__hash__`) because pydantic disables it and textures are shared by reference and cached.

## `misc.py`
`FoamOptions` and `CoacdOptions` (CoACD convex-decomposition parameters).

## `profiling.py`
`ProfilingOptions` (`show_FPS`, `FPS_tracker_alpha`).

## `recorders.py`
`RecorderOptions` (hz/buffer) → `BaseFileWriterOptions` (filename) → `VideoFile` (PyAV), `CSVFile`, `NPZFile`, `TrajectoryFile`.
`BasePlotterOptions` → `PyQtLinePlot`, `MPLLinePlot`, `MPLImagePlot`, `MPLVectorFieldPlot`; `LinePlotterMixinOptions` shared by line plotters.

## `sensors/` (subpackage)
Sensor-option classes (`SensorOptions` base + mixins + concrete classes for cameras, raycaster, tactile, IMU, contacts, temperature, surface distance).
See `sensors/CONTEXT.md` for the full tree.

# How options flow into the engine
`gs.Scene(...)` accepts either a single `options=SceneOptions` or individual `*_options`/`renderer` keyword arguments and assembles them into `SceneOptions` (the two forms are mutually exclusive).
`Scene.__init__` stores the resolved `SceneOptions` as `scene.options`, then constructs `Simulator(scene, options)`.
`Simulator.__init__` keeps `options.sim` for itself (`_dt`, `_substeps`, `_requires_grad`) and hands each solver its sub-options: `options.rigid/kinematic/mpm/sph/fem/pbd`, and dispatches the coupler on `options.coupler` type (SAP/Legacy/IPC).
`Visualizer` receives `options.vis`, `options.viewer`, and `options.renderer`.
`scene.build()` calls `self._sim.build()`, then builds the visualizer and recorder manager — options are resolved/validated at construction, before build.
Sensors are configured with `*Options` objects passed to `scene.add_sensor(...)` / entity sensors; concrete sensor classes bind to their option class via `Sensor.__init_subclass__` (see `sensors/CONTEXT.md`).

# Constraints
- Every option class must inherit from `Options` (strict, forbid-extra pydantic model); do not instantiate base classes like `Options`, `Morph`, `Surface`, `Texture`, `RendererOptions`, `BaseCouplerOptions` directly.
- Keep validation in `model_validator`/`model_post_init`/`field_validator`; raise via `gs.raise_exception` so the friendly error path and Genesis exception type are used.
- Use the strict type aliases from `genesis.typing` (`PositiveFloat`, `Vec3FType`, `StrictBool`, ...) for field types — they carry pydantic constraints.
- Do not store engine objects or per-scene runtime state inside options; options are configuration and must stay serializable/rebuildable.
- Shortcut fields that duplicate a resolved field (`Surface.color/roughness/...`, `Morph.euler`, `RayTracer.env_euler`) must be `Field(exclude=True, repr=False)` so serialization keeps only the canonical field.
- Deprecated parameters are kept as constructor kwargs (or excluded fields) that warn and map to the new field, not silently dropped.

# Design Decisions
Options exist as validated pydantic models instead of raw constructor args to get construction-time validation with aggregated friendly errors, a single documented source of defaults, and serializable scene descriptions (`scene.save`/`load`, trajectory files, in-memory URDF).
All scene options are bundled into one `SceneOptions` so cross-field rules (rigid vs `requires_grad`) and SimOptions inheritance resolve in exactly one place.
Solver options inherit only the SimOptions fields they declare (via `GravityMixin`/`TimeBasedMixin` and `model_copy_from`), so `dt`/`gravity`/`requires_grad` are shared unless a solver overrides its own.
Morphs are deliberately solver-agnostic (one `Mesh` can become a rigid/MPM/FEM entity); the solver is chosen by the entity's material class, not the morph.
Coupler choice is encoded as an options subclass, dispatched by type in `Simulator`.
Renderers are options classes too (not engine objects), selectable per camera while the viewer stays on `Rasterizer`.

# Known Issues / Gotchas
`Options` is `strict=True` + `extra="forbid"`: typos or wrong numeric types (int vs float) raise instead of coercing; the aggregated message lists all errors at once.
`SceneOptions.model_post_init` raises for `box_box_detection=True` + `requires_grad`, `use_gjk_collision=False` + `requires_grad`, or `enable_mujoco_compatibility=True` + `requires_grad`.
A solver's own `dt` must divide `sim.dt` evenly (enforced in the solver base, not in the options classes); setting both a solver `dt` and `sim.substeps` that disagree raises.
Raycaster patterns (`GridPattern`, `SphericalPattern`, `DepthCameraPattern`) are plain `@dataclass`s, not pydantic, and eagerly allocate torch tensors on `gs.device`.
`gs.options.sensors.types` maps option-class *names* to int tags; `types.Lidar` (alias of `Raycaster`) yields a distinct unused tag — use `types.Raycaster` (see `sensors/CONTEXT.md`).
Many deprecated aliases still warn rather than error (`max_FPS`, `n_rendered_envs`, `env_separate_rigid`, `contact_resolve_time`, `use_sparse_grid`, `openvdb`, `is_free`, `requires_jac_and_IK`); treat them as removed in new code.

# Dependencies
Pydantic v2 (`BaseModel`, `ConfigDict`, validators) — the core of this layer.
`genesis.typing` for strict type aliases, `genesis.repr_base.RBC` for repr/colorized output, `genesis.datatypes.List` for torch-backed strict lists, `genesis.constants` for format/backend enums.
External heavy deps only where needed: `av` (PyAV, `recorders.py`), `PIL` (`textures.py`), `numpy`, `torch`; `urdfpy` and LuisaRender are used by `morphs.py`/`renderers.py` via `genesis.ext`.

# Test Strategy
No tests live in this directory; option behavior is covered indirectly through the engine test suite under `../../tests/` (e.g. `tests/core/test_surface.py`, `tests/core/test_recorders.py`, `tests/parsers/`, `tests/sensors/`).
Unit-test new validators via direct `Options` construction (validation, cross-field raises, deprecated-alias migration) without needing a built scene.

# Routing Table
- `sensors/` → Sensor option classes and their `Sensor`-class registration (`gs.options.sensors`, aliased `gs.sensors`); has its own CONTEXT.md.
- `solvers.py` → SimOptions, coupler options, and per-solver options (rigid/MPM/SPH/PBD/FEM/kinematic).
- `scene.py` → `SceneOptions` bundle and SimOptions inheritance/cross-validation.
- `morphs.py` → Entity geometry+pose morphs (primitives, mesh/URDF/MJCF/USD, terrain); FEM tetrahedralization mixin.
- `surfaces.py` + `textures.py` → Visual appearance (`gs.surfaces`, `gs.textures`).
- `renderers.py` → Renderer selection/parameters (`gs.renderers`).
- `vis.py` → Viewer and scene visualization options.
- `recorders.py` → Recorder/plotter options.
- `profiling.py` + `misc.py` → Small option bundles (profiler, foam, CoACD).
- `../engine/` → Consumers (Scene, Simulator, solvers, couplers, sensors, Visualizer) — sibling, read-only.
- `../typing.py`, `../repr_base.py`, `../datatypes.py`, `../constants.py` → Shared foundations — siblings, read-only.
- `../../tests/` → Engine/integration test suite exercising this layer — sibling, read-only.
