# Intent
Reference examples demonstrating the Genesis sensor APIs (`scene.add_sensor(...)` and the `gs.sensors.*` options classes).
Each script shows one sensor family end-to-end: sensor creation, `scene.step()`, and reading/visualizing data.
Scripts follow the shared Genesis example pattern: `import genesis as gs` -> `gs.init` -> `gs.Scene` -> `add_entity` -> `build` -> `step`.

# API Surface
- `camera_as_sensor.py` — RGB camera sensors via `scene.add_sensor` across three rendering backends (`RasterizerCameraOptions`, `RaytracerCameraOptions`, `BatchRendererCameraOptions`); tests camera attachment to an entity link, per-camera lights, and batch rendering; dumps PNGs to `out/camera_as_sensor/`.
- `depth_camera_custom_vverts.py` — `gs.sensors.DepthCamera` (with `DepthCameraPattern`) attached to a rigid Go2 base link and a kinematic plane link; per-frame `set_vverts` on a deforming kinematic sphere; saves depth PNGs with `FrameImageExporter`.
- `lidar_teleop.py` — `gs.sensors.Lidar` (`SphericalPattern`/`GridPattern`) or a `DepthCamera` (`--pattern depth`) on a Go2/Box robot; keyboard teleop through a ring of obstacles.
- `imu_franka.py` — `gs.sensors.IMU` on a Franka end-effector with accelerometer/gyroscope noise, random walk, delay, and jitter; plots measured vs ground truth and records `out/imu_data.npz`.
- `joint_torque_franka.py` — `gs.sensors.JointTorque` on a Franka pressing into a wall; plots control torque vs sensor torque vs difference for J0/J1/J2.
- `tactile_franka.py` — tactile sensors on Franka gripper fingers, selectable via `--sensor`: `ElastomerTaxel`, `ContactDepthProbe`, `KinematicTaxel`, `ProximityTaxel`; headless auto-pickup or GUI teleop with matplotlib vector-field/line plots.
- `tactile_sandbox.py` — tactile sensors on a fixed taxel pad (box or dome), selectable via `--sensor`: `ContactDepthProbe`, `ContactProbe`, `ElastomerTaxel`, `KinematicTaxel`, `ProximityTaxel`; 4 parallel envs each pressing a different object (torus/sphere/duck/dragon) into the pad; `--noise` enables hysteresis, probe-radius noise, gain, and spatial crosstalk.
- `contact_force_go2.py` — `gs.sensors.ContactForce` (or boolean `gs.sensors.Contact` via `--no-force`) on each Go2 foot; pyqtgraph/matplotlib live line plots.
- `surface_distance_shadowhand.py` — `gs.sensors.SurfaceDistanceProbe` on Shadow Hand palm and fingertips, tracking a duck mesh and a box; keyboard teleop via IK.
- `temperature_grid.py` — `gs.sensors.TemperatureGrid` on a platform top surface; contact-based temperature blending via `TemperatureProperties` (hot pusher, warm objects, cool ground); keyboard teleop and optional `--simulate-all-links`.

# Constraints
- All sensors are registered with `scene.add_sensor(gs.sensors.<Sensor>(...))`; camera sensors take renderer option classes (`gs.options.sensors.*CameraOptions`) instead of a bare `gs.sensors.Camera`.
- CLI follows the `examples/` convention (see `.github/contributing/EXAMPLES.md`): `-v/--vis` (default off), `-g/--gpu` (default CPU), `-t/--seconds` or `-s/--steps`, `-b/--num-envs`, `-o/--output-dir`; parser named `parser` at the top of `main()`.
- Backend is spelled out: `gs.gpu if args.gpu else gs.cpu` except `camera_as_sensor.py`, which hardcodes `gs.init(backend=gs.gpu)` because it demonstrates the CUDA-only BatchRenderer.
- Outputs go under `out/`, resolved against the launch directory.
- Long-running loops break after one step when `"PYTEST_VERSION" in os.environ` (a `.github/contributing/EXAMPLES.md` convention; the in-repo harness does not set it at HEAD) so CI stays cheap.

# Dependencies
- Core Python: `genesis`, `numpy`, `torch` (most scripts), `tqdm` (`contact_force_go2`, `imu_franka`, `joint_torque_franka`), `trimesh` (`depth_camera_custom_vverts`, `tactile_sandbox`).
- Optional plotting: `pyqtgraph` preferred, `matplotlib` fallback, both checked via `genesis.recorders.plotters` (`IS_PYQTGRAPH_AVAILABLE` / `IS_MATPLOTLIB_AVAILABLE`); `camera_as_sensor.py` requires `matplotlib` unconditionally.
- Optional render backends: `LuisaRenderPy` enables the RayTracer; `gs_madrona` enables the BatchRenderer (Linux-only and CUDA-only).
- Simulation assets (repo-local): `urdf/go2/urdf/go2.urdf`, `xml/franka_emika_panda/panda.xml`, `urdf/shadow_hand/shadow_hand.urdf`, `meshes/duck.obj`, `meshes/dragon/dragon.obj`.

# Gotchas
- `camera_as_sensor.py` needs a GPU (`gs.gpu`); it silently skips the RayTracer backend without `LuisaRenderPy` and the BatchRenderer backend without `gs_madrona` + `gs.backend == gs.cuda`, falling back to the `Rasterizer` scene renderer.
- Raytracer camera options require `model`, `spp`, and (for unattached cameras) `env_surface`/`env_radius`; BatchRenderer options set `use_rasterizer=True` and reject directional lights.
- Tactile sensor readings in `tactile_franka.py` and `tactile_sandbox.py` are uncalibrated units, intended only for visualization.
- `lidar_teleop.py` always opens the viewer (`show_viewer=True`) and relies on an interactive/EGL-capable display even under test; pass `--pattern depth` to swap the LiDAR for a DepthCamera.
- `depth_camera_custom_vverts.py` uses a `Kinematic` material with `use_visual_raycasting=True` and `MeshSet(..., enable_custom_vverts=True)` so the raycast depth camera sees per-frame vertex deformation.
- `tactile_sandbox.py` creates a torus mesh in the OS temp directory (`tactile_sandbox_torus.obj`) on first run.

# Test Strategy
Covered by `tests/test_examples.py`, which discovers every `examples/**/*.py`, runs each as a subprocess (marker `pytest.mark.examples`, `PYNPUT_BACKEND=dummy`, 600s timeout), and asserts exit code 0.
Script-level optional-dependency skips live in `tests/test_examples.py`'s `EXAMPLE_DEPENDENCIES` map; `camera_as_sensor.py` and `lidar_teleop.py` are not listed there because they degrade gracefully instead of skipping.
Targeted sensor unit tests live under `tests/sensors/` (e.g. `test_camera.py`, `test_imu.py`, `test_joint_torque.py`, `test_tactile.py`, `test_temperature.py`, `test_raycaster.py`).

# See Also
- `../` — parent `examples/` tree and its per-area folders.
- `../../tests/test_examples.py` — examples CI harness (sibling, read-only).
- `../../tests/sensors/` — focused sensor API unit tests (sibling, read-only).
- `../../.github/contributing/EXAMPLES.md` — examples CLI/backend/output conventions (sibling, read-only).
