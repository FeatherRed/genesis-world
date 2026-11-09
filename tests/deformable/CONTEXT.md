# tests/deformable

## Intent
- Integration tests for Genesis FEM (finite-element) deformable bodies and muscle actuation.
- Verifies tetrahedralization, vertex constraints, explicit/implicit FEM solvers, rigid-FEM coupler contact, and muscle group contraction against analytical references computed in-test.
- Every test builds a fresh `gs.Scene` inside the shared autouse `initialize_genesis` fixture.

## API Surface
- `conftest.py` — session fixture `coupled_sliders` returning an MJCF `xml.etree.ElementTree.Element` of two vertical slide joints tied by a joint equality (masses 1.0 and 0.01, polycoef `0 2 0 0 0`).
- `test_fem.py` — exercises `gs.materials.FEM.Muscle`, `gs.materials.FEM.Elastic` (models `linear`, `linear_corotated`, `stable_neohookean`), and `gs.materials.FEM.Cloth`.
- `test_fem.py` — exercises both FEM solvers (`FEMOptions.use_implicit_solver` True/False), `SAPCouplerOptions` and `LegacyCouplerOptions`, and the vertex-constraint API (`set_vertex_constraints`, `remove_vertex_constraints`, `update_constraint_targets`, incl. `link=` and `envs_idx`).
- `test_muscle.py` — exercises `gs.materials.MPM.Muscle` (explicit) and `gs.materials.FEM.Muscle` (explicit + implicit) with `set_muscle`/`set_actuation` over 4 muscle groups.

## Test Inventory
- `test_interior_tetrahedralized_vertex` (required) — `maxvolume` refinement keeps surface vertices on the input faces and the visualizer tracks simulated vertices through `vgeom.sim_verts_idx`.
- `test_multi_submesh_render_decoupling` (required, `n_envs` 0/2) — Cloth on a 2-submesh GLB welds shared boundary vertices and maps render verts per submesh/environment.
- `test_maxvolume` (required) — `maxvolume=0.01` yields more elements than the unconstrained tetrahedralization.
- `test_offset_pos` (required) — morph `offset_pos` shifts state positions exactly.
- `test_halt_on_nan` (required, `precision="32"`) — stiff sphere diverges to NaN; `get_error_envs_mask` tracks it and `scene.reset()` clears it.
- `test_implicit_falling_sphere_box` (required, `precision="64"`) — implicit solver mass conservation and analytical bar sag across SAP/legacy couplers and two material models.
- `test_legacy_coupler_rigid_velocity_correction` (required, `precision="64"`, `backend=[gs.cpu, gs.gpu]`) — legacy coupler ground-contact velocity correction matches the analytical per-triangle scaling.
- `test_hard_constraint` (required, `use_implicit_solver` True/False, `precision="64"`) — vertex constraints, energy-gradient vs solver forces, bitwise free-fall replay, per-environment constraint targets.
- `test_explicit_legacy_coupler_soft_constraint_box` (xfail) — soft-constraint spring dynamics; currently inconsistent with the analytical formula.
- `test_implicit_sap_coupler_collide_sphere_box` (not required) — SAP coupler sphere-box collision; ~250s on CPU.
- `test_implicit_sap_coupler_hard_constraint_and_collision` (not required) — SAP coupler, rigid slider joint equality, FEM vertex constraints and contact compliance; ~400s on CPU.

## Constraints
- Only `CONTEXT.md` may be modified in this directory; the test code belongs to the source tree and must not be edited here.
- Fixtures `cube_verts_and_faces` and `box_obj_path` come from `../conftest.py`; session fixtures `show_viewer`, `tol`, `backend`, and the `precision` marker are also defined there.
- Assertions use `assert_allclose`/`assert_equal` from `../utils/assertions.py` and mesh assets via `get_hf_dataset` from `../utils/assets.py`.
- `test_fem.py` additionally imports `igl` (libigl) and `torch`, which must be installed alongside the Python test dependencies.

## Design Decisions
- Analytical results (bar sag, discrete free-fall, contact velocity scaling, energy-gradient forces) are recomputed inside the tests to pin exact solver semantics instead of comparing snapshots.
- Parametrizing the `precision` fixture (e.g. `["64"]`) overrides the autouse backend-dependent default for numerically sensitive or float32-divergence scenarios.
- CPU-heavy SAP/legacy collision scenarios are kept out of the `required` set so the minimal CI gate stays fast.

## Known Issues
- `test_implicit_sap_coupler_hard_constraint_and_collision` uses the `xml_path` fixture, which is defined only in `../parsers/conftest.py` (a sibling), so running `pytest tests/deformable/` in isolation fails at collection with "fixture 'xml_path' not found"; only `coupled_sliders` (the MJCF element) is local.
- `test_explicit_legacy_coupler_soft_constraint_box` is marked `xfail` because soft-constraint dynamics do not yet match the analytical spring formula.
- `test_halt_on_nan` forces `precision="32"` specifically to trigger float32 divergence; do not "fix" it to 64-bit.
- `test_legacy_coupler_rigid_velocity_correction` overrides the session `--backend` by parametrizing `backend=[gs.cpu, gs.gpu]` itself, so it always runs on both backends.

## Test Strategy
- The `required` marker gates the minimal CI set; `test_muscle` marks only the `n_envs=2` case required via `pytest.param(..., marks=pytest.mark.required)`.
- The two unmarked collision tests (~250s/~400s CPU) are neither `required` nor `slow`; they run only in the full suite.
- Run with `--backend` cpu/gpu and field/ndarray backends as parametrized by the shared harness.

## See Also
- `../conftest.py` — autouse `initialize_genesis` fixture, session fixtures, and marker resolution (read-only).
- `../parsers/conftest.py` — defines `xml_path` and `model_name` fixtures relied upon by one test here (sibling; escalate any write to the parent).
- `../utils/assets.py`, `../utils/assertions.py` — HuggingFace asset download and shared numeric assertions.
