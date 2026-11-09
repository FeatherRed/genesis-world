# tests/grad — differentiable / autodiff simulation tests

## Intent
This directory validates the reverse-mode (backward pass) gradients of Genesis's differentiable simulation.
Gradients are checked three ways: analytic reverse-mode vs central finite differences, torch autograd semantics, and end-to-end optimization loss curves.
Coverage spans rigid forward kinematics/dynamics, contact and collision, constraints (joint limits, frictionloss, equality rows), rollout/state semantics, and MPM/FEM deformable coupling.

## API Surface
- `conftest.py` — local fixtures and an autouse gradient-recording invariant checker.
  - `step_overwrites` (autouse, CPU-only) monkeypatches `genesis.grad.bridge.LaunchRecorder.insert/entry/reverse/__exit__` to detect any array a recorded kernel overwrites but the backward pass did not save; fails the test on the first miss.
  - `_add_hinge_arm(...)` builds a 1-DOF hinge-arm MJCF subtree (capsule geom, explicit inertia).
  - Session-scoped MJCF XML fixtures (each returns an XML string, no asset files): `grad_free`, `grad_revolute`, `grad_revolute_frictionloss`, `grad_prismatic`, `grad_spherical`, `grad_capsule`, `grad_capsule_ground`, `grad_free_with_revolute`, `grad_free_with_hinge_ball_slide`, `grad_chain3`, `grad_slider_limit`, `grad_hinge_pair_joint_eq_linear`, `grad_hinge_pair_joint_eq_quadratic`, `grad_hinge_joint_eq_constant`, `grad_connect_loop`, `grad_weld_pair`, `grad_all_eq_fric`, `grad_cartpole`, `grad_hopper`.
- `utils.py` — the FD-vs-analytic comparison harness.
  - `DiffScenePair` NamedTuple: `scene_ana`/`entity_ana` (diff mode, `backward()` runs here) and `scene_fd`/`entity_fd` (production mode, FD perturbs here).
  - `make_diff_scene_pair(mjcf, *, n_envs, substeps, dt, gravity, enable_joint_limit, disable_constraint, modes, show_viewer, ...)` builds both sides of the pair from one MJCF with identical config.
  - `assert_grad_matches_fd(pair, inputs, apply_fn, loss_fn, *, rtol, atol, eps, n_steps, setup_fn, step_fn)` runs the analytic rollout, backprops, then re-runs the full trajectory with +/- eps perturbations per input entry (O(n_steps * input size)).
- Test modules (routing table):
  - `test_rollout.py` → rollout/horizon semantics, state getters vs FD, closed-loop torch semantics, rejection of non-differentiable solvers.
  - `test_rigid_dynamics.py` → forward-kinematics setters and multistep forces vs FD, PD-target gradients, diff-vs-non-diff forward parity.
  - `test_rigid_collision.py` → contact-force gradients, diff collision detection Jacobian, constraint-solver backward, unsupported-configuration rejection.
  - `test_rigid_constraints.py` → joint-limit, per-step force-into-limit, frictionloss, equality (joint/connect/weld/quadratic) and all-groups constraint gradients.
  - `test_rigid_optim.py` → Adam/SGD trajectory-recovery and goal-pose optimization convergence thresholds.
  - `test_hybrid_push.py` → MPM-rigid coupling, MPM/FEM muscle actuation and coupling-wrench gradients.

## Constraints
Heavy imports (`torch`, `quadrants`, `genesis`) must go through `lazy_imports()` from `../conftest.py` (the root import guard forbids import-time side effects).
Scenario-specific helpers belong here (or in `utils.py`); shared numeric assertions and simulation builders live in `../utils/`.
Gradient comparisons must be anchored to central finite differences, never analytic-vs-analytic alone.
Every FD comparison must select `eps`/`rtol`/`atol` per precision: fp32 uses a large eps to clear the state-noise floor, fp64 uses a small eps to bound truncation.
Tests that cannot pass at fp32 must be pinned with `@pytest.mark.precision("64")`.

## Design Decisions
The dual-scene pair (`scene_ana` vs `scene_fd`) is the core pattern: both scenes share config and produce bit-identical forward states, so FD on the production scene is a valid reference for the analytic gradient without running FD through the autodiff graph.
`assert_grad_matches_fd` applies input i before step i, gives a None input no adjoint, and lets extra steps run without input; `setup_fn`/`step_fn` carry non-differentiated setup and per-step scenario commands.
MJCF topologies are inlined as session-scoped XML strings so each test's physics scenario is self-contained and the reason a topology exists is documented next to it (e.g. skew-axis welds to keep a nonzero velocity-product bias).
Contact FD references must also run in diff mode (`requires_grad=True`): diff scenes route convex-convex contacts through `diff_gjk`, whose contact set differs slightly from production narrowphase, so a production reference would measure that forward gap instead of the gradient.
FD tolerances are pinned per topology/per precision in dictionaries to the measured finite-difference floor, and each FD check first asserts the gradient magnitude clears the noise floor so the comparison is not vacuous.

## Known Issues
Non-differentiable features raise `GenesisException` at `scene.build()` under `requires_grad=True`: PBD liquid, SPH liquid, `LegacyCoupler` (FEM-rigid/MPM coupling), and `SAPCoupler` (see `test_reject_non_differentiable_solvers`).
`RigidOptions(enable_torsional_friction=True)` and `box_box_detection=True` are unsupported in diff sim and rejected at build; torsional/rolling friction rows have no reverse.
`entity.attach(...)` merges kinematic trees across entities and is rejected because the per-entity backward kernels cannot reverse it.
Sphere/ellipsoid contact is rejected at build (diff_gjk EPA never converges on the everywhere-curved boundary); sphere/sphere is exempt via analytic detection.
Screw constraints are not differentiable: `add_screw_constraint` raises `Screw constraints are not supported yet`.
Contact-force gradients are tiny (stiff contact barely moves), so at fp32 the FD reference mostly rounds to zero and the absolute tolerance carries the check.
The FEM implicit solver's fp32 FD is noisy (substep displacement rounds against positions) while its analytic gradient matches fp64, so it gets a looser standalone tolerance.
The quaternion setter is the most-curved input (normalization), so its FD eps is bounded by truncation rather than rounding.

## Test Strategy
Gradient correctness is validated by central-FD agreement, torch-autograd behavioral checks (retain_graph, mutation detection, last-occurrence-wins for repeated indexed writes), and optimizer convergence to per-target thresholds.
`@pytest.mark.required` selects the CI-critical subset; `@pytest.mark.slow` marks long tests (MPM-rigid push ~75s, goal-pose optimization); `@pytest.mark.debug(False)` disables dev/debug mode for numerically heavy tests.
Precision is fixture-driven ("64" on CPU, "32" on GPU) and some tests parametrize `backend` directly with `[gs.cpu, gs.gpu]`.
Forward-parity assertions (diff vs non-diff states, entity-state vs solver-state getters) are used alongside gradient checks to isolate state-query regressions from gradient regressions.
Run a single test with `-k <name> --backend cpu`; FD-heavy tests are slow, so the required subset is what CI gates on.

## See Also
- `../conftest.py` — root fixtures (`backend`, `precision`, `tol`, `show_viewer`, autouse `initialize_genesis` with deterministic algorithms/seed 0).
- `../utils/assertions.py` — `assert_allclose`/`assert_equal` used for every numeric comparison.
- `../utils/simulators.py` / `../utils/mujoco_parity.py` — shared parity builders are NOT used here; grad tests use the local dual-scene FD harness (`utils.py`) instead.
- `../../genesis/grad/` — the implementation under test (`bridge.py` `LaunchRecorder`, `kernel_writes.py`).
- `../../genesis/engine/solvers/rigid/` — rigid solver / constraint solver / collider backends exercised directly by `test_rigid_collision.py`.
