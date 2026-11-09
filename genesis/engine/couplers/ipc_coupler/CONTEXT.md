# Intent
This subpackage implements `IPCCoupler`, the Incremental Potential Contact (IPC) coupler of the Genesis World physics engine.
It is one of three couplers the `Simulator` can select (`SAPCoupler`, `LegacyCoupler`, `IPCCoupler`); the selection is made from the `options.coupler` class in `genesis/engine/simulator.py`.
The coupler is invoked per substep by `Simulator.substep(f)`: `preprocess(f)` before the rigid solver pre-coupling, `couple(f)` after it.
Unlike the other couplers, it does not use the Genesis Quadrants (`qd`) kernel; it drives the external C++ library `libuipc` (Python binding `pyuipc`, imported as `uipc`) for IPC contact and FEM/rigid simulation, and exchanges state between Genesis solvers and the IPC `World`.
It supports rigid bodies (as AffineBody/ABD objects), volumetric FEM, and cloth in one unified contact framework, with per-entity coupling types.

# API Surface
`__init__.py` re-exports `IPCCoupler` (from `coupler.py`).

`coupler.py` defines:
- `IPCCoupler(RBC)`: the coupler class; see the public method list below.
- `_animate_rigid_link(coupler_ref, link, env_idx, info)`: module-level animator callback that enables the IPC soft-constraint and sets its `aim_transform` from the stored Genesis link transform; holds a `weakref` to the coupler.
- Module constants `ABD_KAPPA = 100.0` (MPa), `STIFFNESS_DEFAULT = 1e4`, `JOINT_STRENGTH_RATIO = 100.0`.

`data.py` defines:
- `COUPLING_TYPE(IntEnum)`: `TWO_WAY_SOFT_CONSTRAINT=0`, `EXTERNAL_ARTICULATION=1`, `IPC_ONLY=2`, `NONE=3`.
- `ABDLinkEntry(NamedTuple)`: per-link, per-env ABD state retrieved from IPC (`transform`, `velocity`).
- `ArticulatedEntityData(dataclass)`: per-entity external-articulation coupling buffers (joint child links, local q indices, articulation slots, qpos/delta/ref buffers, previous link transforms).
- `IPCCouplingData`: pre-allocated arrays for coupling-force computation (links, ABD body indices, per-link mass/inertia, `ipc_transforms`/`aim_transforms`/`out_forces`/`out_torques`).

`utils.py` defines stateless helpers:
- `has_articulation_dofs(entity)`: true if `entity.n_dofs > 0` (uses n_dofs, not n_joints, so fixed Planes/Boxes are not articulated).
- `default_coup_type(entity)`: auto-picks `external_articulation` / `two_way_soft_constraint` / `ipc_only` when `material.coup_type` is None.
- `find_target_link_for_fixed_merge(link)`: walks up the kinematic tree to the merge target, skipping FIXED joints.
- `compute_link_to_link_transform(from_link, to_link)`: relative (pos, quat) between two links of one entity.
- `build_ipc_scene_config(options, sim_options, dt)`: builds the `Scene` config dict from `IPCCouplerOptions` and `SimOptions` (gravity, dt, newton/line-search/linear-system/contact/collision/CFL/sanity/diff-sim settings).
- `read_ipc_geometry_metadata(geo)`: reads `(solver_type, env_idx, idx)` from a geometry's metadata, or None if not Genesis-created.
- `update_coupling_forces(...)`: numba-jitted computation of coupling forces/torques from current vs aim transforms.
- `_set_if_not_none(config, keys, value)`: sets a nested config key only if value is not None, casting to native Python types for pybind11.

# IPCCoupler public methods
- `is_differentiable = False` (class attribute): IPC does not support `requires_grad`.
- `__init__(simulator, options)`: raises `ImportError` if `uipc` is missing; stores solver proxies and initializes all internal state.
- `build()`: validates options, then `_init_ipc` -> `_setup_coupling_config` -> `_add_objects_to_ipc` -> `_finalize_ipc` -> `_init_accessors` (plus optional `_init_ipc_gui`).
- `preprocess(f)`: no-op (`pass`).
- `couple(f)`: runs one coupling step (store rigid states -> pre-advance articulation -> `world.advance()`/`retrieve()` -> retrieve FEM/rigid -> apply forces -> post-advance articulation/ipc-only -> tick GUI).
- `reset(envs_idx=None)`: requires `envs_idx is None`; recovers the IPC world to frame 0 and retrieves.
- `is_active` (property): the IPC world is built.
- `has_any_rigid_coupling` (property): any rigid entity has a coupling type.

# Coupling types (per rigid entity)
`_setup_coupling_config` reads `material.coup_type` (defaulted by `default_coup_type`) and maps it to `COUPLING_TYPE`.
- `two_way_soft_constraint`: link-level Animator + `SoftTransformConstraint`; Genesis pose is the aim, IPC constraint force is applied back via `_apply_abd_coupling_forces`.
- `external_articulation`: joint-level `ExternalArticulationConstraint`; requires a fixed base and articulation DOFs; reads Genesis mass matrix and qpos, writes back joint deltas.
- `ipc_only`: one-way; IPC fully drives the rigid transforms (base link pose copied back; velocity zeroed to avoid double integration).
Collision and link filters come from `material.enable_coup_collision` / `coup_collision_links` / `coup_links`.

# Constraints
This coupler is not a `qd` kernel coupler; all IPC state lives in the `uipc` C++ engine, never in Genesis solver buffers.
The IPC scene is a single scene shared across all environments, so the rigid topology (links, joints, geoms) must be identical across envs.
Plane geoms are only supported with `ipc_only` coupling.
Only REVOLUTE and PRISMATIC joints are supported for `external_articulation`; all other joint types raise.

# Design Decisions
IPC scene setup at build time runs no Genesis `qd` kernels: `_add_objects_to_ipc` obtains neutral-config link poses/joint anchors from the pure-Python `compute_forward_kinematics(...)` instead of reading `rigid_solver.dyn_state` via `qd_to_numpy` (commit f70d8fd4).
Soft-constraint strengths are scaled by `1 / substep_dt^2` at build time to reach the units the IPC world expects.
Fixed-joint link geoms are merged into the parent's mesh (mirroring `mjcf.py` `_merge_target_id`) before being handed to IPC.
One IPC workspace dir is created per scene under the system temp dir (`genesis_ipc_<scene_uid>`), and frame 0 is checkpointed (`world.dump()`) so `reset()` can `recover(0)`.
Coupling forces use the Genesis-side masses/inertias so the reaction force exactly matches the IPC constraint force (`G_ipc = M * (q_ipc - q_genesis)`).

# Known Issues
`Solver.set_gravity` is silently ignored: it writes a solver buffer the IPC world never reads, and the world is built before the write, so nothing detects it (see FIXME in `build()`).
The coupler refuses per-env topology (raises when `batch_links_info`/`batch_dofs_info`/`batch_joints_info` are set) because it builds one shared IPC scene.
For `ipc_only`, base velocity is deliberately zeroed (not copied from IPC) to avoid double time integration by the rigid solver; the commented-out `_apply_base_link_velocity_from_ipc` call marks this.
The `gpu_device` option cannot select a CUDA device in the `Engine` constructor; device selection must go through `CUDA_VISIBLE_DEVICES` before Genesis initializes.
`_init_ipc_gui` silently degrades to no GUI on failure (logs a warning and continues).

# Dependencies
`uipc`/`pyuipc` (Python binding of `libuipc`) is required; the import is guarded by `UIPC_AVAILABLE` and `IPCCoupler.__init__` raises `ImportError` if missing.
`polyscope` is imported alongside `uipc` and used only when `_show_ipc_gui` is enabled.
The IPC `Engine` is created with the `"cuda"` backend, so a CUDA-capable GPU/runtime is required.
`numba` is used to JIT `update_coupling_forces`.

# Test Strategy
No tests live in this subpackage; exercises come through Genesis examples/tests that build a `Simulator` with an `IPCCouplerOptions`-derived coupler and a CUDA device.
Smoke-test the three coupling types (soft constraint, external articulation, ipc_only) separately, plus `reset()` behavior.

# See Also
Parent directory `../` (`genesis/engine/couplers/`) holds the sibling couplers `sap_coupler.py` and `legacy_coupler.py` and re-exports all three (read-only siblings).
`genesis/options/solvers.py` defines `IPCCouplerOptions` (all tunables this coupler consumes).
`genesis/engine/simulator.py` selects and drives the coupler per substep.
`genesis/engine/entities/rigid_entity/rigid_entity.py` provides `compute_forward_kinematics` used for neutral-config poses.
