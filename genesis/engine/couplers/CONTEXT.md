# Intent
This directory holds the couplers of the Genesis World physics engine.
A coupler couples two or more active solvers (rigid, FEM, MPM, SPH, PBD) that share one scene and state within each simulation substep.
The `Simulator` owns exactly one coupler and selects it from the class of `SimOptions.coupler` (`genesis/engine/simulator.py`): `SAPCoupler` (default, Semi-Analytic Primal contact), `LegacyCoupler` (deprecated, SDF-based coupling), or `IPCCoupler` (external libuipc Incremental Potential Contact).
`__init__.py` re-exports all three: `LegacyCoupler`, `SAPCoupler`, `IPCCoupler`.

# Coupler Interface (called by Simulator)
`Simulator.substep(f)` runs, in order: `self._coupler.preprocess(f)` -> `substep_pre_coupling(f)` (every active solver) -> `self._coupler.couple(f)` -> `substep_post_coupling(f)` (every active solver).
Every coupler exposes:
- `__init__(simulator, options)`: stores solver proxies and reads options.
- `build()`: allocates fields/state once all solvers have been built.
- `reset(envs_idx=None)`: clears per-env coupling state on reset.
- `preprocess(f)`: prepares data before the solvers integrate the substep.
- `couple(f)`: resolves the coupling (contacts/constraints) after the solvers integrate the substep.
- `is_differentiable`: whether the coupler supports a `requires_grad` scene.
- `active_solvers` (property): the scene's active solvers (used for gradient/timing).

# API Surface
- `sap_coupler.py` (~4060 lines): `SAPCoupler` plus contact/constraint handler classes, `FEMFloorContactType`, `RigidFloorContactType`, `RigidRigidContactType`, and `qd.func` helpers `func_filter_fem_surface_tets`, `func_filter_rigid_tets`, `tri_barycentric`, `tet_barycentric`.
- `legacy_coupler.py` (~990 lines): `LegacyCoupler`, deprecated in favor of SAP.
- `ipc_coupler/`: `IPCCoupler` subpackage (see `ipc_coupler/CONTEXT.md`).
- Coupler option classes live in `genesis/options/solvers.py`: `SAPCouplerOptions`, `LegacyCouplerOptions`, `IPCCouplerOptions`.

# SAPCoupler (SAP)
Port of Drake's Semi-Analytic Primal (SAP) contact solver (arxiv 2110.10107) that simultaneously solves rigid-FEM contact, self-contact, floor contact, and rigid joint equality constraints as one primal optimization.
It requires `precision='64'` (raises for qd.f32) and requires the FEM solver to use its implicit solver.
Per substep `preprocess` -> `precompute` (FEM pressure gradients, rigid FK + rigid volume verts) -> `update_bvh` -> `update_contact` (broad-phase BVH + per-handler `detection`/`compute_jacobian`) -> `compute_regularization`; `couple` then runs `sap_solve` (SAP outer iterations of unconstrained gradient + contact Hessian + PCG + exact line search) and `update_vel` writes solver velocities back.
`is_differentiable = False` (class attribute).
Contact handlers are chosen in `build()` by the contact-type options and appended to `self.contact_handlers`:
- `FEMFloorTetContactHandler` / `FEMFloorVertContactHandler` (FEM <-> floor, hydroelastic tet vs point vert, driven by `fem_floor_contact_type`).
- `FEMSelfTetContactHandler` (FEM hydroelastic self-contact, `enable_fem_self_tet_contact`, BVH-based).
- `RigidFloorVertContactHandler` / `RigidFloorTetContactHandler` (rigid <-> floor, `rigid_floor_contact_type`).
- `RigidRigidTetContactHandler` (rigid <-> rigid hydroelastic, `rigid_rigid_contact_type`).
- `RigidFemTriTetContactHandler` (rigid triangle <-> FEM tet, `enable_rigid_fem_contact`).
`RigidConstraintHandler` (built when `n_equalities > 0` and constraints are not disabled) handles joint equality constraints as SAP constraints alongside the contact handlers.

## Handler class hierarchy
- `BaseConstraintHandler`: SAP constraint base (regularization, energy, gamma/G); subclass `RigidConstraintHandler` for joint equalities.
- `BaseContactHandler`: SAP contact base (contact-mode energy/gamma/G, friction regularization); `ContactMode` enum `STICK`/`SLIDE`/`NO_CONTACT`.
- `RigidContactHandler(BaseContactHandler)` and `FEMContactHandler(BaseContactHandler)`: rigid- and FEM-side contact-frame Jacobian/velocity plumbing.
- Concrete handlers subclass `RigidContactHandler`, `FEMContactHandler`, `RigidFEMContactHandler`, or `RigidRigidContactHandler` as noted above.
- `RigidFEMContactHandler(RigidContactHandler)` and `RigidRigidContactHandler(RigidContactHandler)` are intermediate bases for two-body contacts.

# LegacyCoupler
Deprecated SDF/velocity-based coupler for MPM, SPH, PBD, FEM <-> rigid coupling (particle/vertex vs rigid SDF), plus MPM<->SPH/PBD and FEM<->MPM/SPH grids.
Enabled pairs are decided in `build()` from `LegacyCouplerOptions` flags (`rigid_mpm`, `rigid_sph`, `rigid_pbd`, `rigid_fem`, `mpm_sph`, `mpm_pbd`, `fem_mpm`, `fem_sph`) and solver activity.
`preprocess` handles MPM CPIC surface-to-particle; `couple` runs `mpm_grid_op`, `sph_rigid`, `kernel_pbd_rigid_collide`/animate, and `fem_surface_force`.
`is_differentiable` is a property that is `False` whenever FEM couples with rigid/MPM/SPH (those couplings overwrite surface vertex velocities and carry no gradient).

# IPCCoupler
Delegates all contact/FEM/rigid simulation to the external C++ `libuipc` (`uipc`/`pyuipc`) instead of running Genesis `qd` kernels; see `ipc_coupler/CONTEXT.md` for its full API, coupling types, and gotchas.

# Design Decisions
SAP is primal (optimizes impulses/velocities directly) and near-rigid-regularized via `sap_beta`/`sap_sigma`/`sap_taud`, in contrast to the legacy SDF velocity-projection coupling and IPC's barrier/penalty contact.
The hardcoded floor of SAP/FEM was replaced by the rigid ground plane: `build()` finds a single fixed upward-facing PLANE geom (`self._ground_geom_idx`) and forbids TERRAIN or non-fixed/rotated ground; a scene without one disables floor contact.
SAP contact handlers are selected declaratively from option strings ("tet"/"vert"/"none") mapped to IntEnums in `__init__`, then instantiated once in `build()`.
All three couplers slot into the same per-substep `preprocess`/`couple` seams, so the Simulator is coupler-agnostic.

# Known Issues / Gotchas
- SAP only supports joint equality constraints with a `joint2`; `build()` raises for `EQUALITY_TYPE.JOINT` without `joint2`, and dynamically added constraints are not supported (TODO in `_init_equality_constraint`).
- SAP assumes identical constraints across all batches (TODO in `RigidConstraintHandler.build_constraints`).
- SAP contact query overflow raises with a per-handler "max contact pairs" report; `MAX_N_QUERY_RESULT_PER_AABB = 32` is a heuristic bound.
- SAP supports a single fixed upward plane as ground only; terrain and any other ground configuration raise at build.
- IPC scene setup must not run Genesis kernels at build: neutral-config poses come from pure-Python `compute_forward_kinematics` (see `ipc_coupler/CONTEXT.md`).
- IPC requires a CUDA GPU and `uipc`/`pyuipc`; `polyscope` is optional (GUI only).
- `LegacyCoupler.reset` clears MPM/SPH-rigid normals; SAP `reset` is a no-op; IPC `reset` recovers its world to frame 0.

# Dependencies
SAP uses `igl` (libigl) for tet tables at build and `quadrants` (`qd`) for kernels; the solver-side math it couples into is owned by `genesis/engine/solvers/`.
IPC additionally needs `uipc`/`pyuipc`, `polyscope` (GUI), `numba` (coupling forces), and a CUDA runtime.

# Test Strategy
No tests live under this directory; coupler behavior is exercised through scene-level Genesis tests/examples that build a `Simulator` with one of the three coupler options (SAP tests exist under `tests/`, e.g. fragile SAP unit tests fixed in 2b0b2aa).
Smoke-test SAP contact types separately (fem/rigid floor tet vs vert, fem self-contact, rigid-rigid, rigid-FEM) and the IPC coupling types (two_way_soft_constraint, external_articulation, ipc_only).

# Routing Table
- `sap_coupler.py` -> SAP primal contact/constraint coupler and all its handler classes.
- `legacy_coupler.py` -> Deprecated SDF-based multi-solver coupler.
- `ipc_coupler/` -> IPC coupler subpackage (IPCCoupler, data classes, helpers).
- `../../options/solvers.py` -> `SAPCouplerOptions`/`LegacyCouplerOptions`/`IPCCouplerOptions` (outside this node; read-only, escalate writes to the parent).
- `../simulator.py` -> `Simulator` coupler selection and the per-substep `preprocess`/`couple` invocation.
- `../solvers/` -> The rigid/FEM/MPM/SPH/PBD solvers and state these couplers read and write.
