# Intent
Pydantic option dataclasses for every sensor type in the Genesis World engine (package `genesis-world` v1.4.6).
Each concrete sensor in `genesis.engine.sensors` has a matching options class here that the user passes to `scene.add_sensor(...)`.
The package also exposes the lazy `types` namespace used as the integer-tag key into `scene.read_sensors()` / `entity.read_sensors()`.

# API Surface
- `__init__.py` — re-exports all symbols from the four modules, aliases `Raycaster` as `Lidar`, and defines `types` (see below).
- `options.py` — base hierarchy plus the non-camera, non-tactile concrete options.
- `camera.py` — camera options for the Rasterizer / Raytracer / Batch Renderer backends.
- `raycaster.py` — `RaycastPattern` base and concrete ray patterns (`GridPattern`, `SphericalPattern`, `DepthCameraPattern`).
- `tactile.py` — tactile mixins and concrete contact/taxel options.

## Base / mixin / concrete hierarchy (`options.py`)
```
SensorOptions(Options)
├── KinematicSensorOptionsMixin(SensorOptions)
│   ├── RigidSensorOptionsMixin(KinematicSensorOptionsMixin)
│   │   ├── RigidEntitySensorOptionsMixin(RigidSensorOptionsMixin)   → JointTorque
│   │   ├── ContactFilterOptionsMixin(RigidSensorOptionsMixin)       → Contact, ContactForce (options.py); ContactProbe, ContactDepthProbe, KinematicTaxel (tactile.py)
│   │   ├── TemperatureGrid(RigidSensorOptionsMixin, SimpleSensorOptions)
│   │   ├── IMU(RigidSensorOptionsMixin, SimpleSensorOptions)
│   │   └── SurfaceDistanceProbe(RigidSensorOptionsMixin, SimpleSensorOptions, ProbeSensorOptionsMixin)
│   └── Raycaster(KinematicSensorOptionsMixin, SimpleSensorOptions)
│       └── DepthCamera(Raycaster)
├── SimpleSensorOptions(SensorOptions)  — hardware imperfections, mixed into concrete classes
├── ProbeSensorOptionsMixin(SensorOptions)
│   └── ProbesWithNormalSensorOptionsMixin(ProbeSensorOptionsMixin)
└── (tactile.py mixins descend from SensorOptions / ProbeSensorOptionsMixin)
```
`SensorOptions` fields: `history_length` (NonNegativeInt, default 0), `delay` (NonNegativeFloat, 0.0), `jitter` (NonNegativeFloat, 0.0), `draw_debug` (StrictBool, False), `entity_idx` (StrictInt, default -1, `ge=-1`).
`entity_idx` accepts `None` and is normalized to `-1` by a `field_validator(mode="before")` so the SensorManager can sort uniformly (`-1` = static / not link-attached).
`KinematicSensorOptionsMixin` adds `link_idx_local` (NonNegativeInt, 0), `pos_offset` (Vec3FType, (0,0,0)), `euler_offset` (Vec3FType, (0,0,0)).
`RigidSensorOptionsMixin` narrows `validate_scene` to `RigidEntity`; `RigidEntitySensorOptionsMixin` further requires `entity_idx >= 0`.
`ContactFilterOptionsMixin` adds `filter_link_idx` (OptionalIArrayType, default `()`) to ignore contacts with chosen counterpart links.
`SimpleSensorOptions` adds `resolution`, `bias`, `noise`, `random_walk` (each `FArrayType | float`, default 0.0) — the embedded-sampler hardware imperfections.
`ProbeSensorOptionsMixin` adds `probe_local_pos`, `probe_radius`, `probe_radius_noise`, and three debug-probe visual fields.
`ProbesWithNormalSensorOptionsMixin` adds `probe_local_normal` and a `_is_probe_local_normal_required` property subclasses may override to skip normal shape validation.

## Concrete options (`options.py`)
- `JointTorque` — actuator output effort (torque/force at gearbox output); `dofs_idx_local` (None = all DOFs).
- `Contact` — boolean contact; `threshold`, `debug_sphere_radius`, `debug_color`.
- `ContactForce` — total contact force in local frame; `resolution` (LaxVec3FType), `min_force`/`max_force` (unbounded vec3), `debug_color`/`debug_scale`.
- `TemperatureProperties` (NamedTuple helper) — material thermal props (`base_temperature`, `conductivity`, `density`, `specific_heat`, `emissivity`).
- `TemperatureGrid` — link temperature grid; `properties_dict`, `ambient_temperature`, `convection_coefficient`, `simulate_all_link_temperatures`, `grid_size`, `heat_generation`, `sensor_time_constant`, `contact_depth_weight`, `debug_temperature_range`; shared/merged across all Temperature sensors.
- `IMU` — acc/gyro/mag channels with per-axis `resolution`, `cross_axis_coupling`, `bias`, `noise`, `random_walk`, plus `magnetic_field` and debug arrow colors/scales.
- `SurfaceDistanceProbe` — nearest distances from probes to tracked meshes; `probe_radius` (default 0.5), `track_link_idx`.
- `Raycaster` — ray casting / point clouds; `pattern` (a `RaycastPattern`), `min_range`, `max_range`, `no_hit_value`, `return_world_frame`, `return_points`, debug sphere fields.
- `DepthCamera(Raycaster)` — ray-cast depth image; `pattern` narrowed to `DepthCameraPattern`.

## Camera options (`camera.py`)
- `BaseCameraOptions(KinematicSensorOptionsMixin)` — `res`, `pos`, `lookat`, `up`, `fov` (`gt=0, lt=180`), `lights`, `offset_T` (overrides `pos_offset`/`euler_offset` when set).
- `RasterizerCameraOptions` — adds `near`, `far` (OpenGL).
- `RaytracerCameraOptions` — adds `model` ("pinhole"/"thinlens"), `spp`, `denoise`, `aperture`, `focal_len`, `focus_dist`, and env-map fields (`env_surface`, `env_radius`, `env_pos`, `env_quat`).
- `BatchRendererCameraOptions` — adds `model` ("pinhole"/"thinlens"/"fisheye"), `near`, `far`, `use_rasterizer`; all batch cameras must share the same resolution.

## Ray patterns (`raycaster.py`)
Plain `@dataclass` classes, NOT `Options`: they eagerly allocate `_ray_dirs` / `_ray_starts` torch tensors on `gs.device` in `__init__` and are passed as `pattern=` into `Raycaster`/`DepthCamera`.
- `RaycastPattern` — abstract base; `return_shape`, `ray_dirs`, `ray_starts` read-only properties; subclasses implement `_get_return_shape`, `compute_ray_dirs`, `compute_ray_starts`.
- `GridPattern` — uniform 2D grid of rays (`resolution`, `size`, `direction`).
- `SphericalPattern` — spherical scan (`fov`, `n_points`, `angular_resolution`, `angles`); helper `_generate_uniform_angles`.
- `DepthCameraPattern` — pinhole camera rays; computes `fx/fy/cx/cy` from `res` + `fov_horizontal`/`fov_vertical` (or direct `fx/fy`), converts to the robotics camera frame.

## Tactile options (`tactile.py`)
- `ViscoelasticHysteresisOptionsMixin(SensorOptions)` — `hysteresis_strength`, `hysteresis_tau` (single-Maxwell; `tau > 0` required when `strength > 0`).
- `SpatialCrosstalkOptionsMixin(SensorOptions)` — `crosstalk_strength`/`crosstalk_sigma` (Gaussian) or explicit `crosstalk_kernel` (mutually exclusive); `is_crosstalk_enabled` property.
- `TactileProbeSensorOptionsMixin(ProbeSensorOptionsMixin)` — `debug_contact_color`, `probe_gain`, `probe_gain_resample_range`, `dead_taxel_probability`, `dead_taxel_value_range`, `contact_depth_query` ("sdf"/"raycast"/None).
- `PointCloudTactileSensorMixin(TactileProbeSensorOptionsMixin)` — `track_link_idx`, `n_sample_points`, `use_visual_mesh`, point-cloud debug visuals.
- `ContactHysteresisOptionsMixin(SensorOptions)` — Schmitt trigger `contact_threshold` / `release_threshold` (None = same as contact threshold; may be negative).
- Concrete: `ContactProbe`, `ContactDepthProbe`, `KinematicTaxel` (solver-gated depth/force per probe), `ElastomerTaxel` (HydroShear marker displacement), `ProximityTaxel` (spherical point-cloud force/torque).

## `types` namespace (`__init__.py`)
`types = _SensorTypesNamespace()` lazily maps sensor-options class NAMES to opaque integer tags on first attribute access.
`SensorManager.read_sensors()`/`entity.read_sensors()` return `dict[tag, tensor]`; look up the key as `gs.options.sensors.types.<Name>` (e.g. `types.Raycaster`).

# Relationship to `genesis.engine.sensors`
Concrete sensor classes declare their options via Generic parameters, e.g. `class RaycasterSensor(..., SimpleSensor[Raycaster, RaycastContext, ...])`.
`Sensor.__init_subclass__` (in `genesis/engine/sensors/base_sensor.py`) inspects `__orig_bases__` and records the first type argument as `_options_cls` (plus shared-context, metadata, and return-data classes).
`SensorManager.SENSOR_TYPES_MAP` then maps options class → sensor class for runtime dispatch in `scene.add_sensor`.
Options classes do NOT import their sensor class at runtime — the `SensorT` parameters are forward-ref strings and the engine imports appear only under `TYPE_CHECKING`.

# Validation / post-init gotchas
- `entity_idx`: user may pass `None`, normalized to `-1` before validation (`ge=-1`).
- `SensorOptions.model_post_init` raises if `jitter > delay`; `validate_scene` only WARNS when `delay` is not a multiple of `scene.sim.dt`.
- Camera sensors do NOT support `history_length`: `BaseCameraOptions.model_post_init` raises when it is `> 0` (lazy render-on-read bypasses the shared cache/ring that backs history).
- Cameras do NOT inherit `SimpleSensorOptions`, so they have no `resolution`/`bias`/`noise`/`random_walk` fields.
- `Raycaster.model_post_init` defaults `no_hit_value` to `max_range` and raises when `max_range <= min_range`.
- `IMU.model_post_init` sums the acc/gyro/mag channel values into the inherited `SimpleSensorOptions` `resolution`/`bias`/`noise`/`random_walk` fields (see in-code FIXME: these should be private).
- `ContactForce.model_post_init` raises when any axis has `max_force <= min_force`.
- `ProbeSensorOptionsMixin` validates that array-like `probe_radius` matches the probe count from `probe_local_pos`.
- `ProbesWithNormalSensorOptionsMixin` validates `probe_local_normal` unless the subclass sets `_is_probe_local_normal_required` to `False`.
- `SpatialCrosstalkOptionsMixin` requires exactly one of `crosstalk_strength` / `crosstalk_kernel`; `crosstalk_kernel` must be 2D/3D with odd spatial dims and finite entries.
- Tactile `contact_depth_query`: `"sdf"` | `"raycast"` | `None`; all sensors of the same class must agree, resolved to `"sdf"` when none sets it.
- Tactile `probe_radius` entries of `0` are inactive filler probes (grid padding); `_validate_filler_probe_radius` requires at least one positive radius.
- `ContactHysteresisOptionsMixin` raises when `release_threshold > contact_threshold`; `ElastomerTaxel` defaults to `contact_threshold=1e-5`, `release_threshold=-1e-4`.
- Ray patterns allocate torch tensors at construction on `gs.device`; they are validated by their `__init__` (e.g. `GridPattern` min resolution `1e-3`).

# Known Issues
- `types` tags are keyed by attribute NAME, not class identity: `types.Lidar` (alias of `Raycaster`) yields a separate, unused integer tag; always use `types.Raycaster` (the SensorManager looks up `options_cls.__name__`).
- Docstrings reference `gs.sensors.types.<Name>`, which is correct: `genesis/__init__.py` re-exports this package as `gs.sensors` (`from .options import sensors`), so `gs.sensors.types` and `gs.options.sensors.types` are the same object.
- `tactile.py`'s `TYPE_CHECKING` block imports all five tactile sensor classes from `.kinematic_tactile`, but `ElastomerTaxelSensor` and `ProximityTaxelSensor` are actually defined in `point_cloud_tactile.py` (harmless — never executed, only affects IDE/type-check resolution).

# Constraints
- Options are pydantic models: `Options` base sets `strict=True`, `extra="forbid"`, `validate_default=True` (see `genesis/options/options.py`).
- This package must not import sensor classes at runtime (keep engine imports under `TYPE_CHECKING`) to avoid circular imports.
- `raycaster.py` patterns are intentionally not `Options` — do not convert them to pydantic models; they hold tensors and are composed inside `Raycaster.pattern`.

# Dependencies
- pydantic, numpy, torch; `genesis` (`gs`) for `raise_exception`/`EPS`/device; `genesis.typing` annotated aliases; `genesis.utils.geom.spherical_to_cartesian`.
- Concrete sensor classes in `genesis/engine/sensors/` must exist (and be importable) for `scene.add_sensor` to resolve; options only reference them as strings at runtime.

# Test Strategy
No tests live inside this package; option behavior is exercised through the engine in `tests/sensors/` (read-only sibling: `test_camera.py`, `test_contact.py`, `test_imu.py`, `test_joint_torque.py`, `test_raycaster.py`, `test_tactile.py`, `test_temperature.py`, `test_api.py`).
Validation and post-init rules are best covered by constructing options (some with a scene) rather than unit-testing models in isolation.

# Routing Table
- `options.py` — base/mixin hierarchy and non-camera/non-tactile sensor options.
- `camera.py` — camera options for the three rendering backends.
- `raycaster.py` — ray-cast patterns (`RaycastPattern`, `GridPattern`, `SphericalPattern`, `DepthCameraPattern`).
- `tactile.py` — tactile mixins and concrete taxel/contact-probe options.
- `__init__.py` — re-exports, `Lidar` alias, and the `types` tag namespace.
- `../options.py` — `Options` base class (sibling, read-only).
- `../../engine/sensors/` — concrete sensor classes that register against these options (sibling, read-only).
- `../../../tests/sensors/` — sensor test suite (sibling, read-only).

# See Also
- `genesis/engine/sensors/base_sensor.py` — `Sensor.__init_subclass__` registration and the `SimpleSensor`/`RigidSensorMixin` hierarchy.
- `genesis/engine/sensors/sensor_manager.py` — consumes `genesis.options.sensors.types` and `SENSOR_TYPES_MAP`.
- `genesis/typing.py` — the annotated field types used across these models.

# Status
Complete — all four modules and the `types` namespace are documented against the current codebase.
