## Intent

End-to-end integration tests that validate the full Genesis pipeline across scene assembly, physics stepping, and entity/solver coupling, rather than any single module in isolation.
Each test builds a real `gs.Scene`, adds entities, steps it many times, and asserts on the resulting simulation state.

## API Surface

- `test_integration.py` — the only test module in this directory.
- `__init__.py` — empty package marker.

### Tests

- `test_pick_and_place(mode, show_viewer)` — full Franka pick-and-place of a rigid cube.
  Parametrized over `mode` [0, 1, 2] (integrator presets) and `backend` [cpu, gpu].
  Exercises `gs.Scene` + `RigidOptions`, `Plane`/`Box`/`MJCF` morphs, `surfaces.Plastic`, DoF armature, PD gains (`set_dofs_kp`/`kv`/`force_range`), `inverse_kinematics`, `plan_path`, position/force control, and `scene.step`.
  Final assertions: cube velocity decays to 0 and the released cube settles at z=0.075.
- `test_hanging_rigid_cable(show_viewer, tol)` — a resting cable (MJCF `xml/cable.xml`) must stay at rest over 100 steps.
  Reads internal solver state via `scene.rigid_solver.dyn_state.links.pos/quat.to_numpy()` and asserts position/quaternion error within `tol=1e-3`.
- `test_franka_panda_grasp_fem_entity(primitive_type, show_viewer)` — rigid-soft coupling: Franka grasps a deformable FEM object (Sphere or HF-dataset `meshes/cube8.obj`).
  Parametrized over `primitive_type` [box, sphere] and `precision` ["64"].
  Exercises `FEMOptions` implicit solver and `SAPCouplerOptions` with tight tolerances, per-geom contact filtering (`geom.desc.contype`/`conaffinity`), and asserts the object moves by the expected lift delta without slipping, then stays stationary.

## Constraints

- Shared fixtures and markers live in `tests/conftest.py` and are documented at the `tests/` root; do not duplicate that documentation here.
- Uses `assert_allclose` from `tests/utils/assertions.py` (tensor-to-numpy aware) and `get_hf_dataset` from `tests/utils/assets.py`.
- These are physics-accuracy tests: tolerances are tight and intentionally load-sensitive; do not loosen them casually.

## Known Issues

- The mode 0 integrator (`approximate_implicitfast`, substeps=1, armature=0) is numerically unstable for interacting entities; the in-file comment explains that first-order implicit correction stabilizes each entity independently and needs DoF armature (or a higher-order integrator like RK4) once entities interact.
- `test_hanging_rigid_cable` carries a `FIXME` comment asking why better than 1e-3 accuracy cannot be achieved for the resting cable.
- `test_franka_panda_grasp_fem_entity` self-skips when `gs.use_ndarray` is set: `SAPCoupler` does not support the ndarray backend yet.
- `test_franka_panda_grasp_fem_entity` has no explicit `backend` parametrization; it runs on the backend chosen by the autouse fixture and fixes `precision="64"`.

## Test Strategy

- Markers: `test_pick_and_place` and `test_hanging_rigid_cable` are `@pytest.mark.slow("gpu")` (gpu runs ~250s each); `test_hanging_rigid_cable` is additionally `@pytest.mark.required`.
- `slow` is NOT deselected by the project default addopts (only `benchmarks` and `examples` are); these tests run by default and are expensive, so CI/dev runs typically add `-m "not slow"`.
- `test_franka_panda_grasp_fem_entity` requires the HuggingFace asset `meshes/cube8.obj` (downloaded via `get_hf_dataset`), so it needs network on first run.
- Visuals (`show_viewer`) are controlled by the session `show_viewer` fixture; `show_FPS` is disabled in all three scenes.

## See Also

- `../conftest.py` — shared fixtures (`initialize_genesis`, `backend`, `tol`, `show_viewer`) and markers (sibling — read-only).
- `../utils/assertions.py` — `assert_allclose` helper (sibling — read-only).
- `../utils/assets.py` — `get_hf_dataset` asset downloader (sibling — read-only).
- `../../genesis/options/` — `SimOptions`, `RigidOptions`, `FEMOptions`, `SAPCouplerOptions` used by these tests.
- `../../genesis/engine/` — solver internals (`rigid_solver.dyn_state`) exercised by `test_hanging_rigid_cable`.
