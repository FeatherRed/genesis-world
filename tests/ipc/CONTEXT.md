# tests/ipc — IPC Coupler Integration Tests

## Intent
Integration tests for the Genesis IPC (Incremental Potential Contact) coupler, which couples Genesis rigid/FEM/cloth solvers with the optional `uipc` library.
They validate coupling-type auto-detection, rigid-solver contact delegation, and physical correctness (freefall, contact, friction, momentum, cloth stretching, robot grasping).
These are white-box tests: they assert on the coupler's private state as well as on simulated behavior.

## API Surface
- `test_api.py` — coupler bookkeeping and entity/link mapping: `needs_coup`, coup_type auto-detection from DOFs, `coup_links` filtering, fixed-link merge target resolution, rigid-vs-IPC collision delegation, `coup_collision_links`.
- `test_deformable.py` — FEM/cloth ↔ rigid interaction: MJCF Franka grasping an FEM box, cloth corner drag, cloth biaxial stretching with force/material checks.
- `test_rigid.py` — rigid + FEM/cloth IPC dynamics: freefall, ground clearance/sliding, contact-pair friction/resistance, object collision, momentum conservation, single-joint articulation, base-link force application.
- `utils.py` — `uipc` `SceneVisitor` helpers that read IPC-scene geometry via meta attributes: `collect_ipc_geometry_entries`, `find_ipc_geometries`, `get_ipc_merged_geometry`, `get_ipc_positions`, `get_ipc_rigid_links_idx`.
- `__init__.py` — empty package marker.

## Constraints
Every test module and `utils.py` skips at module level when `uipc` is unavailable: `if find_spec("uipc") is None: pytest.skip(SKIP_NO_IPC, allow_module_level=True)`.
The skip check runs before importing `genesis`/`uipc`, so a missing optional dependency fails cleanly without import errors.
The coupler is obtained via `cast("IPCCoupler", scene.sim.coupler)`.
Multi-env cases are parametrized as `n_envs in {0, 2}`; `n_envs=0` means a single env with no batch dimension, iterated uniformly as `range(max(scene.n_envs, 1))`.

## Design Decisions
IPC-scene geometries are tagged with uipc meta attributes (`solver_type`, `env_idx`, and `link_idx` for rigid / `entity_idx` for FEM+cloth); `utils.py` reads those tags through `SceneVisitor` so tests validate exactly what the coupler hands to uipc.
Coupler bookkeeping is pinned through private attributes (`_coup_type_by_entity`, `_coup_links`, `_abd_slots_by_link`, `_abd_data_by_link`, `_articulation_data_by_entity`, `_coupling_collision_settings`, `_ipc_scene`, `_ipc_ground_contacts`, `_ipc_abd_contacts`, `_ipc_fem_contacts`).
Physical tests assert analytic invariants (Euler freefall, Green–Lagrange biaxial stress, momentum conservation) rather than golden outputs.
COUPLING_TYPE auto-detection under test: fixed zero-DOF entity → `IPC_ONLY`, free rigid → `TWO_WAY_SOFT_CONSTRAINT`, fixed articulated robot with DOFs → `EXTERNAL_ARTICULATION`.

## Known Issues
`test_objects_freefall` skips per-vertex displacement validation for the sphere FEM entity (`# FIXME: This test does not pass for sphere entity...`).
`test_cloth_uniform_biaxial_stretching` uses a coarse analytic force estimate (`# FIXME: The estimated force is not very accurate`, tolerance `1e4 / E`).
`test_single_joint` uses much looser transform tolerances for the non-fixed robot than the fixed one (`# FIXME: Why the tolerance is must so large if no fixed ?!`), and the robot may dip slightly below the ground (`dist_min > -0.1`).
`test_momentum_conservation` documents a small z-momentum leak from TetGen's asymmetric Steiner-point insertion (NOTE comment in the test body).

## Dependencies
Requires the optional `uipc` Python module (and its backend); without it every module here skips with `SKIP_NO_IPC`.
Cloth/FEM meshes come from `get_hf_dataset(pattern="IPC/grid20x20.obj")` (network download on first use).
Robot assets: `urdf/simple/two_cube_{revolute,prismatic}.urdf`, `urdf/panda_bullet/panda_nohand.urdf`, `urdf/go2/urdf/go2.urdf`, `xml/franka_emika_panda/panda_non_overlap.xml`.

## Test Strategy
Markers: `required` (default run) and `slow` (each carries an approximate runtime comment, e.g. `# ~250s`); `show_viewer` session fixture toggles the viewer per test.
`slow` tests are not excluded by the repo's default addopts (which only excludes `benchmarks`/`examples`); the 1200 s per-test timeout accommodates them.
Run a single module with `pytest tests/ipc/test_rigid.py`; without `uipc` the whole package is skipped, so run on a machine with `uipc` installed.
Tolerances follow the shared `TOL_SINGLE` (single-precision) constant from `tests/conftest.py`.

## See Also
- `../../genesis/engine/couplers/ipc_coupler/` → coupler implementation under test (`coupler.py`, `data.py` with `COUPLING_TYPE`, `utils.py`) (sibling — read-only, escalate writes to parent).
- `../conftest.py` → shared fixtures and `SKIP_NO_IPC`/`TOL_SINGLE` (documented at tests root).
- `../utils/` → `assert_allclose` assertions and `get_hf_dataset` assets (sibling — read-only, escalate writes to parent).
