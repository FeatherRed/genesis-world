## Intent
Sensor test suite verifying every registered `gs.sensors.*` class against analytic ground truth and live simulation state.
Tests exercise the sensor measurement pipeline (`_update_raw_data` → imperfections → transforms → `_post_process`), not just returned values.
Camera rendering is the one snapshot-based exception (PNG images); all other sensors assert against closed-form physics.

## API Surface
- `__init__.py` — empty package marker (do not add logic).
- `conftest.py` — session-scoped `free_box` fixture: builds a MJCF free box (freejoint + box geom) to load with `align=False`.
- `test_api.py` — SensorManager lazy plugin discovery, pipeline-contract TypeError, fake `SimpleSensor` pipeline variants, `test_add_and_read_all_registered_sensors`, history length, shared `RaycastContext`, `scene.read_sensors()` bulk API.
- `test_camera.py` — rasterizer, Madrona batched renderer, LuisaRender raytracer: PNG snapshots, destroy/weakref lifecycle, `look_at` entity.
- `test_contact.py` — `Contact` (bool) and `ContactForce`: gravity support force, bias/noise/random_walk/delay/jitter, `filter_link_idx`.
- `test_imu.py` — `IMU`: freefall lin_acc/ang_vel/mag, offsets, cross-axis coupling, bias/noise, resolution quantization, reset-cache behavior.
- `test_joint_torque.py` — `JointTorque`: Newton/Euler torque identity and torque-through-contact; session fixtures `joint_torque_pendulums`, `joint_torque_contact_pendulum`.
- `test_raycaster.py` — `Raycaster`/`Lidar`/`DepthCamera` raycast core: patterns, graze rays, visual-vs-collision meshes, static/dynamic BVH split-merge, cache offsets.
- `test_tactile.py` — `SurfaceDistanceProbe`, `ContactProbe`, `ContactDepthProbe`, `KinematicTaxel`, `ProximityTaxel`, `ElastomerTaxel`: hysteresis, gain/dead resample, crosstalk, SDF-vs-raycast parity, filler probes.
- `test_temperature.py` — `TemperatureGrid` + `TemperatureProperties`: contact heating/cooling, `simulate_all_link_temperatures`, reset to base temp.

## Constraints
- Only `CONTEXT.md` may be written in this node; never modify test or source files.
- Depends on shared root infra documented in `tests/CONTEXT.md` (autouse `initialize_genesis`, `backend`/`tol`/`show_viewer`, `required`/`slow` markers); do not re-document here.
- `required` marks CI-gating cases; `slow` marks multi-minute tests (`test_add_and_read_all_registered_sensors`, `test_rasterizer_attached_batched`, ~200s each).
- Most tests parametrize `n_envs` as `[0, pytest.param(2, marks=pytest.mark.required)]`: plain CI runs the fast `n_envs=0` single-scene path, merge gates the batched path.

## Design Decisions
- Assert against analytic ground truth (freefall gravity, penetration×stiffness force, joint-torque Newton identity with `effective_inertia = INERTIA + damping * DT` under Euler) rather than regression snapshots.
- Raycaster/tactile correctness is validated through `_shared_context` BVH internals (`n_trees`, `maybe_static`, `is_env_uniform`, `rebuild_subscriber.pending`) because static/dynamic tree regrouping and env→tree routing are the core invariants.
- `contact_depth_query` is class-wide, so SDF-vs-raycast parity builds two separate scenes instead of mixing modes in one.
- Fake-sensor plugins (`options.py`/`sensor.py`/`broken.py`/`optional.py`) in `tmp_path` verify lazy discovery without touching the real plugin registry.
- Imperfection tests always pair `read()` against `read_ground_truth()` to prove the GT branch is untouched.

## Known Issues
- Raycaster/Lidar kernels require a GPU (`gs.cuda`); raytracer needs `LuisaRenderPy`, batched renderer needs `gs_madrona` — camera tests gate on `find_spec` and import `SKIP_NO_LUISA`/`SKIP_NO_MADRONA` from `..conftest`.
- macOS Apple Software Renderer misrasterizes out-of-view/edge geometry: camera tests use finite-plane layouts and platform-specific thresholds.
- `test_batch_renderer_destroy` carries a FIXME: it fails when the scene has no entities.
- `test_contact.py::test_gravity_force` carries a FIXME: a CoM offset on the box disturbs contact force on the second box.
- Raytracer snapshots use a blur kernel of 3 to absorb hardware nondeterminism.
- Some tactile tests add a dummy IMU (same dtype) to force a discontinuous ground-truth memory layout and catch per-sensor buffer aliasing.

## Test Strategy
- Run single files with `pytest tests/sensors/test_<sensor>.py`; `-m required` restricts to CI-gating cases; camera raytracer/batch tests auto-skip without optional backends.
- `n_envs=0` cases are fast and CPU-friendly; `n_envs=2` (and `n_envs>2` BVH/heterogeneous tests) cover batching and require GPU backends.
- Coverage gap: no dedicated test for `SensorManager` cross-backend field-vs-ndarray parity here — that lives in the root backend parametrization.

## See Also
- `../conftest.py` — shared fixtures, markers, backend parametrization (read-only sibling).
- `../utils/assertions.py` — `assert_allclose`/`assert_equal` helpers.
- `../utils/assets.py` — `get_hf_dataset` asset downloads (not used by this suite).
