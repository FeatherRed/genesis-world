# tests/rigid

## Intent
The largest rigid-body test group for the Genesis rigid solver.
Covers dynamics, collision (convex and nonconvex), constraints/equalities, friction, control, kinematics, island partitioning, asset parsing (MJCF/URDF/XACRO), scene/trajectory serialization, terrain, sparse solve, and MuJoCo parity.
Ships its own inline MJCF/URDF model-generator fixture library in `conftest.py` so tests are self-contained and the same model bytes feed both Genesis and MuJoCo.

## API Surface

### conftest.py — inline model fixture library (62 fixtures, no pytest hooks)
Defines only model-generator fixtures and module-level helpers; no `pytest_configure`, no marker registration, no option fixtures.
Marker registration, autouse `initialize_genesis`, and all option fixtures live in `../conftest.py` (tests/ root) and are inherited here.

Module-level helpers (not fixtures):
- `_build_plane_contact_model(model_name, condim, friction, plane_size)` — shared skeleton: one contact default applied to every geom plus a plane floor; free bodies appended by `_add_free_body`.
- `_add_free_body(...)`, `_build_fixed_base_dual_arm(...)`, `_build_chain_capsule_hinge(...)`, `_build_multi_pendulum(...)`, `_add_sphere_link(...)`, `_build_primitive_pair_mjcf(...)`.
- `ellipsoid_mjcf(semi_axes, ...)` — plain function imported directly by `test_collision.py` (not only a fixture).

Session-scoped model generators (each returns an `ET.Element` MJCF/URDF, a file path, a tuple, or a builder closure), grouped by consumer:
- Contact/parity primitives: `box_box`, `box_plan`, `free_boxes_and_slider`, `sphere_plane_roll`, `sphere_plane_spin`, `humanoid_ball_floor`, `tet_meshball`, `collision_edge_cases(mode)`, `decompose_fusion_groups`, `ellipsoid`, `side_by_side_capsules`, `collinear_capsules`, `side_by_side_cylinders`, `collinear_cylinders`, `collinear_spheres`.
- Joints/equalities/mimic: `mimic_hinges`, `scaled_mjcf_joint_equalities`, `scaled_urdf_mimic`, `connected_boxes`, `swinging_spindle`, `two_aligned_hinges`, `offset_hinge_box`, `compound_joint`, `hinge_slide`, `double_ball_pendulum`.
- Inertial edge cases: `undefined_inertia`, `undefined_inertia_arm`, `implicit_inertial_origin`, `implicit_inertial_origin_chain`, `degenerate_inertials`, `zero_density_marker_mjcf`, `aligned_link_frame_urdfs`, `sliding_ball_pair` (returns a `build(base_mass, tip_mass)` closure), `free_bodies_in_one_model`.
- Kinematic trees / islands: `depth_first_tree_mjcf`, `depth_first_tree_urdf`, `long_chain`, `two_fixed_branches`, `box_freejoint_offset`, `freeflyer_mjcf`, `freeflyer_urdf`, `trees_and_slider_mjcf`, `chain_capsule_hinge_mesh`, `chain_capsule_hinge_capsule`.
- Dynamics/energy: `pendulum`, `double_pendulum`, `pendulum_with_joint_dynamics(joint_damping, joint_friction)`, `joint_with_partial_dynamics(joint_damping, joint_friction)`, `spring_double_pendulum`, `damped_flap`, `resting_light_stick`, `damped_pendulum_and_tilted_light_capsule`.
- Actuator: `general_actuator`.
- Asset loading: `urdf_with_external_assets(asset_tmp_path)`, `urdf_with_legacy_parsed_primitives(asset_tmp_path)`, `mjcf_include_default_and_asset_mesh(asset_tmp_path)`, `xacro_robot(tmp_path)`.
- Entity merge: `merged_arm_hand_models`, `merged_overlapping_models`, `fixed_base_dual_arm`, `fixed_base_dual_arm_high_damping`, `fixed_base_dual_arm_chained`.

These model fixtures are resolved by the parent `xml_path` fixture through `request.getfixturevalue(model_name)`: `model_name` is a parametrize argument naming one of the fixtures above, or an asset-relative path (containing `/`) that passes straight through to the asset resolver.

Function-scoped model fixtures require additional parameters: `collision_edge_cases(asset_tmp_path, mode)`, `xacro_robot(tmp_path)`, `pendulum_with_joint_dynamics(joint_damping, joint_friction)`, `joint_with_partial_dynamics(joint_damping, joint_friction)`.

### Test files (file-by-file purpose)
- `test_api.py` (1417 lines) — data accessors and state get/set APIs across `n_envs` and batched/ndarray/field layouts.
- `test_asset_loading.py` (2151 lines) — MJCF/URDF/XACRO parsing: depth-first link ordering, external mesh/texture assets, legacy-parser fallback, `<include>`/default asset-path rewriting, Hugging Face dataset assets.
- `test_collision.py` (2261 lines) — convex collision detection and response, GJK vs analytic contacts, contact force directions, collision edge cases (`mode` 0-8), ellipsoid, self-collision masking across the entity merge boundary.
- `test_collision_nonconvex.py` (1018 lines) — nonconvex mesh decomposition/convexification and decomposition fusion groups over Hugging Face mesh datasets.
- `test_constraints.py` (581 lines) — joint/equality constraint scaling, mimic, equality-link (weld/connect) handling.
- `test_control.py` (309 lines) — PD/general/motor actuators, position control vs native PD emulation.
- `test_dynamics.py` (1298 lines) — gravity, all-fixed scenes, box-box dynamics sweeps, many-boxes dynamics, merged-entity DOF-layout equivalence, GPU cholesky tiling beyond shared memory.
- `test_friction.py` (1055 lines) — friction cones (pyramidal/elliptic), `frictionloss`, torsional/rolling friction, sphere-plane roll/spin.
- `test_heterogeneous.py` (537 lines) — heterogeneous environments in one scene (fixed-child mesh objects), per-env parity against homogeneous references, broadphase traversal.
- `test_islands.py` (1256 lines) — island partitioning of multi-free-body entities: kinematic edges vs shared static links.
- `test_kinematics.py` (1159 lines) — forward kinematics, link/joint velocities, sweeps over `mujoco_compatibility`.
- `test_mujoco_parity.py` (438 lines) — the MuJoCo parity driver (see below).
- `test_narrowphase.py` (1309 lines) — analytic capsule/sphere vs GJK narrowphase, using a monkey-patched `narrowphase.py` forced to GJK and `errno` bit abuse for call tracing.
- `test_serialization.py` (946 lines) — scene and trajectory save/load (zip manifest, pickle, JSON), description digest.
- `test_sparse.py` (111 lines) — sparse-solve no-slip resting stability.
- `test_terrain.py` (447 lines) — terrain generation (flat/fractal/sloped/pyramid/stairs/stepping-stones/wave) and terrain collision.

## Solver / integrator / option parametrization
`model_name`, `gs_solver`, and `gs_integrator` are NOT fixtures defined here — they are `@pytest.mark.parametrize` arguments.
`gs_solver` ∈ `{gs.constraint_solver.CG, gs.constraint_solver.Newton}`; `gs_integrator` ∈ `{gs.integrator.implicitfast, gs.integrator.Euler}`.
They are consumed by the parent `mj_sim`/`gs_sim` parity fixtures, so every test requesting those fixtures must parametrize both.
Backend selection is via `@pytest.mark.parametrize("backend", [...])`, resolved by the autouse `initialize_genesis` fixture.
Marker-driven option fixtures live in `../conftest.py` and are set by markers on the test, e.g. `@pytest.mark.friction_cone(gs.friction_cone.elliptic)`, `@pytest.mark.enable_collision(False)`, `@pytest.mark.split_entities`, `@pytest.mark.merge_fixed_links(...)`, `@pytest.mark.multi_contact(...)`, `@pytest.mark.gjk_collision(...)`, `@pytest.mark.adjacent_collision(...)`, `@pytest.mark.friction_torsional/rolling(...)`, `@pytest.mark.mujoco_compatibility(...)`, `@pytest.mark.disable_constraint(...)`.
`split_entities` makes the parent `xml_paths` fixture split a model into one entity per worldbody subtree, so Genesis holds one entity per subtree while MuJoCo holds one model.

## MuJoCo parity (test_mujoco_parity.py)
Drives `gs_sim` + `mj_sim` through `tests/utils/mujoco_parity.py`: `simulate_and_check_mujoco_consistency`, `check_mujoco_data_consistency`, `check_mujoco_model_consistency`, `init_paired_simulators`, `set_paired_inertial_properties`.
`tests/utils/simulators.py` (`build_mujoco_sim`/`build_genesis_sim`) builds both simulators from the same `xml_path`/`xml_paths`, so the same model bytes feed both engines.
Tests step both engines, compare qpos/qvel over time, and cross-check model constants (inertias, constraint weights, armatures).

## Constraints
- Tests only; never modify Genesis source from this directory.
- Models are generated inline (via `xml.etree.ElementTree`), not checked-in files, so both engines consume identical geometry and parameters.
- `conftest.py` must stay a pure fixture/helper library: no pytest hooks, no marker registration (those belong to `../conftest.py`).

## Design Decisions
- Inline ET-generated MJCF/URDF instead of asset files: identical bytes to both engines, self-contained, and parameterizable per test.
- Session scope for most model generators (cheap strings/ET elements); function scope only where a fixture needs `tmp_path`/`asset_tmp_path` or parametrize args.
- `_build_plane_contact_model` centralizes the condim/friction contact defaults shared by the plane-contact models.
- MuJoCo is the reference engine; models are authored to pair its contact conventions (e.g. per-box yaw spreads corner contacts, `humanoid_ball_floor` swaps the floor plane for a sphere to avoid a degenerate plane-capsule second contact).
- Solver/integrator sweeps (CG/Newton × implicitfast/Euler) run through parametrize to cover both constraint-solve paths.

## Known Issues / Gotchas
- Weld/connect equality world frames differ between MuJoCo and Genesis for sites (MuJoCo uses site 1, Genesis the parent link frame); `test_equality_link` sets `ignore_constraints` for site-based welds.
- Closed kinematic chains need `@pytest.mark.enable_collision(False)` because adjacent-link filtering alone does not disable the self-collision.
- Several models are deliberately extreme to expose solver paths: 1e-6 kg m^2 inertias (`damped_flap`, `resting_light_stick`, `damped_pendulum_and_tilted_light_capsule`) stress the implicit damping pass; `long_chain` (128 DOFs) exceeds GPU shared memory to exercise the tiled mass assembly.
- `@pytest.mark.required("pairwise"|"sparse")` prunes a parametrize product to a per-backend coverage subset in `../conftest.py` (pairwise keeps pairs of option values greedily; sparse keeps the diagonal of the axes grid); adding/removing a param axis changes the kept subset.
- `@pytest.mark.slow` with no argument means slow on every backend; `@pytest.mark.slow("gpu")` means slow only on the named backends (marker is dropped on other backends before `-m` selection).
- `@pytest.mark.precision("32"|"64")` overrides the default precision (64 on cpu, 32 on gpu and benchmarks).
- `TOL_SINGLE`/`TOL_DOUBLE` live in `../conftest.py`; `test_narrowphase.py` imports `TOL_SINGLE` directly from `..conftest`.
- `collision_edge_cases` is function-scoped and requires the `mode` parametrize argument (0-8) alongside `asset_tmp_path`.
- GPU-only tests skip when no CUDA device or when the device lacks opt-in shared memory beyond the default 48 kB (`test_cholesky_tiling_large_shared_memory`).
- `test_terrain.py` is ~1000 s on CPU; its GPU case carries the `required` marker instead.

## Test Strategy
- Run with `pytest tests/rigid`; root addopts already exclude `benchmarks`/`examples`, use xdist worksteal, a 1200 s timeout, and `xfail_strict`.
- `-m required` selects the fast smoke subset (further pruned per backend by the `required("pairwise"|"sparse")` logic).
- `slow` tests take >=150 s (some ~200 s, some ~250 s on GPU); run them explicitly with `-m slow` (and optional backend args).
- Parity tests require MuJoCo; `MUJOCO_GL=0` is forced in `../conftest.py`.
- Backend coverage is parametrized cpu/gpu/cuda/amdgpu/metal × field-vs-ndarray through the `initialize_genesis` fixture.

## See Also
- `../conftest.py` — session fixtures (`backend`, `tol`, `show_viewer`, `xml_path`, `xml_paths`, `mj_sim`, `gs_sim`), marker registration, and the autouse `initialize_genesis` fixture (parent; read-only).
- `../utils/mujoco_parity.py`, `../utils/simulators.py`, `../utils/assertions.py`, `../utils/assets.py`, `../utils/collision.py` — shared helpers used throughout.
- `../grad/test_rigid_*.py` — differentiable rigid tests (dynamics, collision, constraints, optimization).
- `../parsers/` — mesh/USD parser tests.
