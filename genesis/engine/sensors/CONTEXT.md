# genesis/engine/sensors

## Intent
Read-only observation layer of the Genesis World physics engine: sensors extract data (images, depth, ray hits, IMU, contact/joint forces, tactile maps, temperature) from the simulation without modifying physics.
One `SensorManager` per `Sim` owns all sensors, their shared caches, and the per-step update pipeline.
Users create sensors with `scene.add_sensor(options)` and read them with `scene.read_sensors()`, `entity.read_sensors()`, or `sensor.read()` / `sensor.read_ground_truth()`.

## API Surface (flat directory — one module per sensor family or concern)

- `base_sensor.py` — core contracts: `Sensor`, `SimpleSensor`, `SharedSensorContext`, metadata dataclasses, link-attach mixins, ring-pipeline hooks.
- `sensor_manager.py` — `SensorManager`: class registry (`SENSOR_TYPES_MAP`), per-type/per-dtype caches, timeline rings, `read_sensors`, `get_sensors_by_entity`, `_resolve_sensor_cls` lazy discovery.
- `camera.py` — `CameraReturnType`, `BaseCameraWrapper`/`RasterizerCameraWrapper`/`BatchRendererCameraWrapper`, `BaseCameraSensor`, `RasterizerCameraSensor`, `RaytracerCameraSensor`, `BatchRendererCameraSensor`, per-class `*CameraSharedMetadata`.
- `depth_camera.py` — `DepthCameraSensor` (thin reshape over `RaycasterSensor`).
- `raycaster.py` — `RaycasterSensor` (Lidar), `RaycasterReturnType`, `RaycasterSharedMetadata`, `BVHContext`, `RaycastContext` (the cross-type shared context).
- `imu.py` — `IMUSensor`, `IMUReturnType`, `IMUSharedMetadata`, `_kernel_update_imu_raw_data`.
- `contact_force.py` — `ContactSensor`, `ContactForceSensor`, `ContactFilterMetadataMixin`, `ContactSensorMetadata`, `ContactForceSensorMetadata`, contact-aggregation kernels.
- `joint_torque.py` — `JointTorqueSensor`, `JointTorqueSensorMetadata`.
- `probe.py` — `ProbeSensorMetadataMixin`, `ProbeSensorMixin`, `ProbesWithNormalSensorMetadataMixin`, `ProbesWithNormalSensorMixin`, `get_measured_bufs`, generic probe debug renderer.
- `surface_distance_probe.py` — `SurfaceDistanceProbeSensor`, `SurfaceDistanceProbeMetadata`, `SurfaceDistanceProbeSensorMetadataMixin`, `TriangleMeshBVH`.
- `tactile_shared.py` — shared tactile machinery: static-chunk BVH builders, `BVHMetadata`/`ChunkedBVHData`, grid-FFT convolution registration, contact prefilter, contact-depth query mixins, viscoelastic hysteresis, spatial crosstalk.
- `kinematic_tactile.py` — `KinematicTactileSensorMixin`, `ContactDepthProbeSensor`, `ContactProbeSensor`, `KinematicTaxelSensor`, `KinematicTaxelReturnType`, SDF/BVH contact-depth kernels.
- `point_cloud_tactile.py` — `PointCloudTactileSensorMixin`, `PointCloudTactileSharedMetadata`, `ProximityTaxelSensor`, `ElastomerTaxelSensor`, `PointCloudBVH`, elastomer FFT-dilation kernels.
- `temperature.py` — `TemperatureGridSensor`, `TemperatureGridSensorMetadata`, FFT diffusion kernels.
- `__init__.py` — imports the modules and re-exports `SensorManager`.

## Core Contracts

### Sensor generic parameters
Each concrete sensor declares `class MySensor(Sensor[Options, Context, Metadata, Data])`.
`Sensor.__init_subclass__` extracts these via `__orig_bases__`/`get_args` into `_options_cls`, `_shared_context_cls`, `_metadata_cls`, `_return_data_cls`, and auto-registers `options_cls -> sensor_cls` in `SensorManager.SENSOR_TYPES_MAP`.
`DataT` defaults to `tuple`; the second parameter (`Context`) is the cross-type `SharedSensorContext` (use `None` when there is none — `SensorManager` then passes `None` to per-step hooks).

### Instance lifecycle
`create_sensor(options)` validates options against the scene, resolves the class (lazy module discovery if unregistered), lazily creates the per-class metadata and the shared context, then instantiates `Sensor(options, idx, context, metadata, manager)`.
`sensor.build()` runs during scene build and pushes per-sensor state into the shared metadata tensors.
The public API is `read(envs_idx)` (measured, post-imperfections) and `read_ground_truth(envs_idx)` (raw simulated phenomenon), both pure views into per-class return caches plus `get_cloned_from_cache`.

### Return-format / dtype contract
`_get_return_format()` (instance method, may depend on options) returns a single `(N,)` or a tuple-of-shapes for multi-tensor returns (e.g. IMU's NamedTuple).
`_get_cache_dtype()` (classmethod) is class-uniform so each class maps to one contiguous per-dtype slice.
`_get_intermediate_format` / `_get_intermediate_dtype` default to return space; overrides are mandatory when `_post_process` is overridden (the intermediate buffer must be distinct, because timeline rings live in intermediate space).

### Ring pipeline (SimpleSensor)
`SimpleSensor.uses_ring_pipeline = True`; per branch: GT = `raw -> _apply_transform(is_measured=False) -> _post_process`; measured = `raw -> _apply_physics_imperfections -> _apply_transform(is_measured=True) -> _apply_hardware_imperfections -> _post_process -> delay sampling`.
`_update_raw_data` + `_apply_physics_imperfections` are packaged in `_update_current_timestep_data` (override to fuse in one kernel launch).
Concrete sensors override hooks, not `_update_shared_cache`.
Timeline rings store post-transform, PRE-hardware data so `_apply_transform` recurrence reads clean slots; hardware imperfections mutate only the per-step working buffer.
`_post_process` projects intermediate→return space once per branch per step; delay sampling then reads stale slots of the per-class return-space ring.
History reads gather post-everything snapshots from the return-space ring (final measured values observed at each past step).

### Hardware imperfections (SimpleSensor)
`noise`/`bias`/`random_walk`/`resolution` live in `SimpleSensorMetadata`; each contribution is gated by a precomputed Python bool (`has_any_*`) so zero-valued classes pay no GPU work.
Setters (`set_noise`, `set_bias`, `set_random_walk`, `set_resolution`, `set_jitter`) write metadata and recompute the gate flags (one GPU→CPU sync per setter, not hot path).

### Shared sensors vs simple sensors
Cameras derive from `Sensor` directly (via `BaseCameraSensor`) and use `SharedSensorMetadata` (time state only); they set `uses_ring_pipeline = False` and render lazily on `read()` into a per-sensor `image_cache`, so delay/jitter/history are rejected at construction.
All other sensors derive from `SimpleSensor` and use the batched ring pipeline with a per-class shared metadata that fuses per-entity sensor columns into flat tensors so one kernel launch updates the whole class.

### Shared contexts (cross-type)
`SharedSensorContext` is a single resource shared by *several* sensor types, owned by `SensorManager` (keyed by context class).
`RaycastContext` is the prime example: it owns the rigid solver's collision-BVH entries and is shared by `RaycasterSensor`, `DepthCameraSensor`, and the raycast-mode tactile sensors.
A context is an optimization only — a sensor must produce identical results shared or not.
Lifecycle: `activate` (idempotent, called from a consumer's `build`, constructs the resource once scene geometry exists), `update` (manager, once/step), `reset`/`destroy` (manager-driven); querying an inactive context must raise.

## Design Decisions

- Per-class batched metadata: sensors of one type are concatenated into flat per-class tensors so kernels are launched once per class, not once per sensor; `SensorManager` keeps `_entity_slice_in_class` and per-sensor `_cache_offset`/`_cache_size` to slice reads back per entity.
- Link attachment: `_LinkAttachedSensorMixin` holds the Python `_link` and appends pos/euler offsets; `RigidSensorMixin` (single solver + `links_idx` tensor) vs `KinematicSensorMixin` (per-solver `_SolverLinkGroup` buckets so the per-step gather is one bulk read per solver).
- Cameras wrap renderers rather than rendering themselves: `RasterizerCameraSensor` reuses the viewer's GL context when a viewer exists (fragile to have a second headless context) or builds a standalone offscreen `Rasterizer`; `RaytracerCameraSensor` targets the scene raytracer; `BatchRendererCameraSensor` uses `MinimalVisualizerWrapper` (a minimal visualizer-like shim, since `BatchRenderer` expects one). Camera instances are aggregated per class in `*CameraSharedMetadata` (sensors/lights/renderer/image_cache).
- Static-geometry BVH reuse in `raycaster.py`: `BVHContext.maybe_static` entries skip the per-step rebuild (dominant raycast cost) and rebuild only when a lazy `GEOMETRY` `Subscriber` flags `set_pos`/`set_quat`/`set_vverts`; static collision entries can also group identical envs into shared trees (`is_grouping_fixed`).
- Tactile subfamilies share `probe.py` (probe registration/gains/dead-taxels) and `tactile_shared.py` (BVH chunking, grid FFT, crosstalk, hysteresis): kinematic tactile queries the sensor link's own SDF/geometry; point-cloud tactile samples a point cloud on tracked links (`ProximityTaxel`) or adds an elastomer model (`ElastomerTaxel`) with viscoelastic hysteresis and an FFT convolution for dilation/shear.
- Elastomer `HydroShear` uses a precomputed real-FFT kernel (`_precompute_hydroshear_dilate_kernel_fft`) with a `compressibility` blend (1 = local Gaussian, 0 = incompressible 1/r or bonded-layer transfer) and a `dilation_reg` epsilon; `register_grid_fft_sensor` in `tactile_shared.py` handles per-sensor FFT sizing/padding.
- `TemperatureGridSensor` uses batched FFT semi-implicit diffusion with mirror padding (Neumann BC, no wrap-around) and explicit heat generation.

## Host/Device Traffic (per step)

- `SensorManager.step` (sensor_manager.py:324) performs no host↔device transfer itself: ring rotation advances a host-resident 0-D index (utils/ring_buffer.py:42,113-117) and every cache/ring write is device-to-device.
- The default `SimpleSensor` pipeline is fully device-resident; per class per step it costs roughly 6-8 full-cache-sized GPU copies (GT and measured timeline slots, intermediate working buffer, return-space ring slot, return cache) plus `randn` of the same size when noise/random-walk is enabled.
- Per-step D2H syncs exist in a few sensor classes: `TemperatureGridSensor` (`.item()` per sensor in `_apply_radiation_convection`, temperature.py:437,439,440, and `_apply_T_measured_filter`, temperature.py:468); `ContactSensor`/`ContactForceSensor` via `Collider.get_contacts` (`n_contacts.max().item()`, solvers/rigid/collider/collider.py:969/976, once per solver step thanks to `_contact_data_cache`); `ElastomerTaxelSensor` on the BVH path via `_build_shear_active_pc_index` (`.tolist()` x3, point_cloud_tactile.py:1781-1783).
- Syncs that only occur on first step/reset/geometry change: `ProbeSensorMixin._maybe_build_cache_col_probe_idx` (probe.py:304-305), `RaycastContext` static regroup (`bool(device_tensor.all())`, raycaster.py:140 and 200/221).
- `read()` / `read_ground_truth()` / `read_sensors()` never synchronize and never return host data: they return fresh device tensors (advanced indexing copy of the per-class cache, base_sensor.py:532, sensor_manager.py:473-478); history reads allocate `(B, H, ...)` on device.
- Camera `read()` is the outlier: each stale read re-renders and uploads a host-produced image to the device (camera.py:555,713,813) and then returns views of the per-sensor `image_cache` without cloning (camera.py:367-390).

## Constraints

- Do NOT write/modify source code in this directory from context-extraction tasks — only CONTEXT.md.
- Directory is flat by design; kernels and metadata are tightly coupled to their sensor module. Do not split files without reason.
- Sensors must never modify physics state.
- `read()`/`read_ground_truth()` return pure views/clones; the caller may mutate the result (manager clones on read).

## Gotchas / Known Issues

- `rg` returns only one match per file in this sandbox; use `git grep -o -n -E` or the `read` tool for reliable definition listings.
- Metadata is `@dataclass` (not pydantic) because pydantic does not handle `torch.Tensor` fields; `make_tensor_field` builds empty tensor fields lazily on `gs.device`.
- `gs.tc_bool` maps to `torch.int32` on Apple Metal for quadrants interop — masks consumed by `torch.where` must use `torch.bool` tensors on the torch side (see `ProbeSensorMetadataMixin`).
- Overriding `_post_process` without overriding `_get_intermediate_format`/`_get_intermediate_dtype` raises at class definition; the intermediate buffer must be distinct even when shape/dtype coincide (timeline rings are in intermediate space).
- `jitter <= delay` and `jitter < dt` are enforced; delay sampling uses zero-order-hold (dtype-safe for bool/int/uint8/quantized returns).
- `contact_force.py`: `ContactSensor` returns bool from a float intermediate (`_post_process` thresholds); `ContactForceSensor` clamps to `min_force`/`max_force`; `ContactFilterMetadataMixin` uses `-1` padding in `filter_links_idx` so kernels scan columns unconditionally.
- `kinematic_tactile.py` caps `_MAX_CONTACTS_PER_SENSOR = 1024` and `_MAX_GEOMS_PER_SENSOR = 64`; overflow silently truncates (documented, not an error).
- `temperature.py`: MPS silently ignores the `device` arg on `fftfreq`/`rfftfreq`, so `_compute_K2_rfft3` moves the result explicitly; `STEFAN_BOLTZMANN`/`KELVIN_OFFSET`/`MAX_TEMP` are module constants.
- FP precision is configurable via `gs.tc_float` / `gs.qd_float` / `gs.np_float`; kernels must not hardcode float32/float64.
- `scene.add_sensor` is wrapped in `with namespace(self._namespace)`; sensor options carry `entity_idx`/`link_idx_local` to attach to entities (or `entity_idx=-1` for static sensors).

- `read_sensors()` also covers camera classes, but their per-class cache is never written (`Camera._update_shared_cache` is a no-op), so it returns zero-filled image-shaped tensors for them; use `Camera.read()` instead.
- `read()` / `read_ground_truth()` copy the whole per-class cache (fancy-indexing with the full `envs_idx` tensor) even for a single sensor, so many per-step reads of different sensors of one class duplicate that copy.

## Dependencies

- `quadrants` (`qd`) DSL for compiled kernels (`@qd.kernel`, `@qd.func`, `fastcache=True`), `gs.qd_float`/`gs.qd_int`.
- `trimesh` (temperature debug mesh), `numpy`, `torch`, `typing_extensions.TypeVarWithDefault`, `pydantic` (options layer, in `genesis/options/sensors/`, not this directory).
- Internal peers: `genesis.engine.solvers` (rigid/kinematic state, collider, BVH), `genesis.engine.entities`, `genesis.utils` (`geom`, `misc`, `raycast_qd`, `sdf`, `array_class`, `ring_buffer`), `genesis.vis` (rasterizer/raytracer/batch renderer + `RasterizerContext`).

## Test Strategy

No tests live in this directory. Sensor behavior is validated indirectly through options validation, scene build, and downstream integration; kernel correctness rests on quadrants compilation plus the per-class metadata invariants. When changing the pipeline, re-run the project's sensor/demo scripts and confirm `read()` vs `read_ground_truth()` agreement, delay/jitter/history behavior, and the `uses_ring_pipeline=False` camera path.

## See Also

- `genesis/options/sensors/` — `SensorOptions` and concrete option classes (validation + `entity_idx`/`link_idx_local`), plus `gs.sensors.types.<Name>` integer tags used as keys into `scene.read_sensors()`.
- `genesis/engine/scene.py` — `Scene.add_sensor` (line 582) and `Scene.read_sensors` (line 597) thin wrappers over `SensorManager`.
- `genesis/utils/ring_buffer.py` — `TensorRingBuffer` backing the timeline/return rings.
- `genesis/engine/bvh.py` and `genesis/utils/raycast_qd.py` — shared BVH config/data and raycast kernels used by `raycaster.py`.
