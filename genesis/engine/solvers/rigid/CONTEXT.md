## Intent
`rigid/` is the rigid-body (articulated-body dynamics, ABD) solver of the Genesis engine.
`rigid_solver.py` (3545 lines) defines the module-level `_sanitize_sol_params` and the `RigidSolver` class, the reference implementation for solver naming and kernel structure.
`data.py` (717 lines) declares every struct the solver allocates (`RigidData`, `RigidInfo`, `DynInfo`, `DynState` and their members) — the state layout any accessor question is answered from.
`RigidSolver` subclasses `GravityMixin, TimeBasedMixin, KinematicSolver` (from `../base_solver.py` and `../kinematic_solver.py`).
Three child subpackages hold the compute kernels: `abd/` (dynamics numerics), `collider/` (collision detection), `constraint/` (constraint solve).
All kernel code is written against the Quadrants DSL (`import quadrants as qd`), not raw Taichi.

## Class & Lifecycle
`RigidSolver` has `material_cls = Rigid` and `_entity_classes = ((Drone, DroneEntity), (Terrain, TerrainEntity), (Morph, RigidEntity))`.
`__init__(scene, sim, options)` copies solver options (collision/multi-contact/mujoco-compat/joint-limit/self/neutral/adjacent-collision, `disable_constraint`, `max_collision_pairs`, `integrator`, `box_box_detection`, `requires_grad`, `use_hibernation`, `friction_cone`, `constraint_solver`, `contact_resolution`, `impratio`) and resolves coupled defaults.
`build()` (line 292) floors solver time constants against `_substep_dt`, resolves the tolerance default (`1e-5` for f32, `1e-8` for f64, x0.1 unless MuJoCo-compat), inits vert/geom/equality/dof fields, computes the neutral-pose cartesian space, builds `collider` + `constraint_solver`, registers `ErrorCode` messages, refreshes inverse weights/mean inertia, fills default rotor armatures, warns on links too light for the constraint solve, and computes morph pose offsets.
`_create_data_manager` (line 784) builds the `RigidData` and binds `rigid_info`, `dyn_info`, `dyn_state`, `kinematics_scratch`, `errno`, then writes gravity into `rigid_info.gravity`.

## Substep Pipeline (ABD ordering)
`simulator.py::substep(f)` drives coupling as `coupler.preprocess(f)` -> `substep_pre_coupling(f)` (each active solver) -> `coupler.couple(f)` -> `substep_post_coupling(f)` (each active solver).
`RigidSolver.substep_pre_coupling(f)` (1597) runs `self.substep(f)`, except under `IPCCoupler` with rigid coupling, which defers the rigid step to post-coupling.
`RigidSolver.substep_post_coupling(f)` (1621) handles `SAPCoupler` (`update_qacc_from_qvel_delta` + `kernel_step_2`) and `IPCCoupler` with rigid coupling (`substep(f)`).
`RigidSolver.substep(f)` (1290) has three paths:
- Default fused path: `kernel_substep_dynamics` (forward kinematics if needed -> forward velocity -> forward dynamics, then equality-constraint assembly) -> `collider.detection()` -> `kernel_substep_inequality` (inequality/contact assembly) -> `func_solve_body` (constraint solve, `_n_iterations`) -> `kernel_substep_post` (acceleration/contact-force write-back + optional noslip, then integrate + copy-next-to-curr + refresh cartesian/velocity unless MuJoCo-compat).
- SAP-coupler / non-recording-grad path: `kernel_step_1` (FK + velocity + dynamics) -> `update_qvel`; post-coupling finishes with `update_qacc_from_qvel_delta` + `kernel_step_2`.
- Recording (differentiable) path: FK -> `kernel_forward_dynamics_without_qacc` -> `kernel_compute_smooth_qacc` + `_func_constraint_force` (equality + collision + inequality + resolve) under `custom_backward(_constraint_force_grad)` -> `kernel_update_acc` -> `kernel_integrate` -> `kernel_copy_next_to_curr_clear_coupling` -> FK again (unless MuJoCo-compat).
The fused-pipeline entry kernels (`kernel_step_1`, `kernel_step_2`, `kernel_substep_dynamics`, `kernel_substep_inequality`, `kernel_substep_post`) and their `func_step_1`/`func_step_2` bodies are defined module-level at the BOTTOM of `rigid_solver.py` (lines 3403-3545), after the class.

## Public API Surface
Stepping: `substep`, `substep_pre_coupling`, `substep_post_coupling`, `reset_grad` (1611, zeroes grads of `dyn_state.links/dofs/joints/geoms/entities` and `rigid_info` since backward accumulates via `atomic_add`), `update_forward_pos` (1268), `_update_forward_kinematics` (1278), `detect_collision` (1414, clears+re-detects; returns geom pair indices for `env_idx=0` only).
State: `get_state` (1640, returns a `RigidSolverState`), `set_state` (1664, `@mutates(GEOMETRY, DYNAMICS)`, supports `partial` and zero-copy fast path), `data` property (1801, yields solver + collider + constraint-solver data items).
Setters: `set_links_pos`, `set_base_links_pos`, `set_links_quat`, `set_base_links_quat`, `_set_links_info`, `set_links_mass`, `set_links_inertia`, `set_links_COM`, `set_qpos`, `set_global_sol_params`, `set_sol_params`, `_set_dofs_info`, `set_dofs_kp/kv/act_gain/act_bias/force_range/stiffness/armature/damping/frictionloss/limit/position/velocity`, `control_dofs_force/velocity/position/position_velocity`, `set_geoms_friction_ratio`, `set_geom_friction(_torsional/_rolling)`, `set_geoms_friction(_torsional/_rolling)`, `set_drone_rpm`, `add/delete_weld_constraint`, `add/delete_screw_constraint`, `clear_external_force`, `apply_links_external_wrench`.
Getters: `get_links_pos/vel/acc/acc_ang/root_COM/COM/mass/inertia/invweight`, `get_geoms_pos/quat/friction/friction_torsional/friction_rolling/friction_ratio`, `get_dofs_control_force/actuator_force/force/acc/kp/kv/act_gain/act_bias/force_range/stiffness/invweight/armature/damping/frictionloss`, `get_sol_params`, `get_mass_mat`, `get_kinetic_energy`, `get_potential_energy`, `get_total_energy`, `get_AABB`, `get_weld_constraints`, `get_equality_constraints`, `is_links_info_batched`.
Properties: `n_geoms`, `n_verts`, `n_free_verts`, `n_fixed_verts`, `n_faces`, `n_edges`, `max_collision_pairs`, `n_equalities`, `equalities` (built values cached after `build`).
Inherited, not overridden here (all from `../kinematic_solver.py`): `get_links_quat` (1172), `get_links_ang` (1241), `get_qpos` (1245), `get_dofs_velocity` (1249), `get_dofs_position` (1253), `set_dofs_velocity` (1040, reached through `super()`), plus the `get_state`/`set_state` pair.
`set_links_pos`/`set_links_quat` are removed stubs raising `DeprecationError` (1823, 1959); `set_base_links_pos`/`set_base_links_quat` are the base-pose setters.

## Scene / State Model
`RigidData` (`data.py`) allocates four structs per solver: `rigid_info` (`RigidInfo`), `dyn_info` (`DynInfo`), `dyn_state` (`DynState`), `kinematics_scratch`, plus the `errno` per-env int buffer.
`RigidInfo` (constant model params) holds `gravity`, `meaninertia`, `n_awake_dofs`, `qpos0`/`qpos`/`qpos_next`, `geoms_init_AABB`, the mass matrix and its LDL^T factor (`mass_mat`, `mass_mat_L`, `mass_mat_D_inv`), kinematic root tables (`roots_link_idx`, `links_root_end`), tree tables (`trees_*`, `links_tree_idx`, level-sweep tables), child tables, dof mass-block tables, and runtime constants (`substep_dt`, `iterations`, `tolerance`, `ls_*`, `noslip_*`, `impratio`, `n_equalities`, `hibernation_thresh_vel`, `EPS`).
`DynInfo` holds per-element Info structs (`entities/links/joints/dofs/geoms/verts/faces/edges/vverts/vfaces/vgeoms/equalities`); `DynState` holds the corresponding State structs plus `free_verts`/`fixed_verts`/`vverts`/`vgeoms`.
Bodies are laid out as flat batched arrays: one slot per link/joint/dof/geom/vert across all entities, one slot per environment in the batch (batch dim LAST).
`ColliderState` and `ConstraintState` live in `collider/data.py` and `constraint/data.py`, owned by those subpackages.
Structs are `genesis.utils.data.Data` subclasses: class annotations with `field(...)`, allocated as `qd` fields/ndarrays on the current backend with `gs.qd_float`/`gs.qd_int`/`gs.qd_bool` (allocation in `genesis/utils/data.py:687-750`).

## `qd` Conventions & Memory Layout
qd fields are field-major with the batch (env) dim LAST: e.g. `dyn_state.dofs.vel[i_d, i_b]`, `mass_mat[DOFS, DOFS, ENVS]`; torch arrays at the API boundary are batch-first (`qpos[i_b, i_q]`).
Conversion happens only at the boundary via `qd_to_torch(..., transpose=True)`, `qd_from_torch`, `qd_to_numpy(..., transpose=...)` in `genesis/utils/misc.py`; zero-copy is used when `gs.use_zerocopy` (set_state/get_state fast path).
`func_*` are `@qd.func` device subroutines inlined into kernels; `kernel_*` are `@qd.kernel(fastcache=True)` launched entrypoints (some `graph=True`); there are no `@gs.jit` pipelines inside `rigid/` — the composition lives in `RigidSolver.substep`.
Getter data flow: state getters (`get_links_*`, `get_dofs_*`, `get_qpos`) return device torch tensors through `Solver._read_data(..., copy=True)` -> `qd_to_torch(..., transpose=True, copy=True)`, i.e. a contiguous DLPack view of the qd field cloned on the device (no host copy) and then masked with basic slicing where possible; `get_links_vel`/`get_links_acc` fall back to a `@qd.kernel` gather into a freshly allocated `torch.empty` when `gs.use_zerocopy` is false.
Without PyTorch the same getters return a zero-copy `DeviceView` from `torch_free.read_view(field, envs_idx, idx)` (the `Solver._read_data` torch-free branch) and `get_links_vel`/`get_links_acc` gather into a qd ndarray through the same kernels; the getters that need device arithmetic, the COM frame or a non-identity morph offset raise `require_torch('<feature>')` instead.
Setter data flow splits in two: `set_qpos`/`set_base_links_pos`/`set_base_links_quat`/`set_dofs_velocity`/`control_dofs_*` write the buffer in place through cached `qd_to_torch(..., copy=False)` views under `gs.use_zerocopy`; `set_dofs_position` and every non-zero-copy branch write via a `kernel_set_*`/`kernel_control_*` launch, and `_sanitize_io_variables` only converts the input when its dtype/device/shape differs (`broadcast_tensor` -> `torch.as_tensor`).
Torch-free the numpy-source branch keeps those `kernel_set_*`/`kernel_control_*` launches, fed by the numpy branch of `Solver._sanitize_io_variables` (qd values plus int32 index arrays), and a DLPack producer takes the staged write instead: `_stage_values` (which rejects any producer but `(n_envs, n_values)`) -> `torch_free.DlpackStager.stage` (device-to-device into a cached batch-last buffer) -> `kernel_set_qpos_staged`/`kernel_control_dofs_staged`/`kernel_control_dofs_position_velocity_staged` of `abd/torch_free_kernels.py`, which also write `ctrl_mode` and zero `ctrl_vel` for a position control.
A write taking two producers at once (`control_dofs_position_velocity`) stages the second one through a second `DlpackStager` (`_stager_secondary`), because a stager caches one buffer per shape and would otherwise hand the same buffer, holding the values staged last, to both kernel arguments.
`gs.use_zerocopy` is a `ZeroCopyMode` object (`genesis/utils/misc.py:151`): truthy when the backend shares memory with the torch device and not inside `disabled()`, which `RolloutGraph` enters to force the kernel path while recording (`genesis/grad/bridge.py:175`).

## GPU Hot Path (per-substep device residency)
The default non-differentiable, non-SAP substep (`RigidSolver.substep` else-branch, line 1352) launches device kernels only: `kernel_substep_dynamics` (graph) -> `collider.detection` (`kernel_detection`, graph) -> `kernel_substep_inequality` (graph) -> `func_solve_body` -> `kernel_substep_post` (graph); none reads a device value back to the host.
No per-substep Python decision reads device values: hibernation (`_use_hibernation`), `len(sim.active_solvers)`, `_is_forward_pos_updated`/`_is_forward_vel_updated`, `_enable_collision`, `_disable_constraint`, `noslip_iterations`, and collider `_n_possible_pairs`/`_use_split_narrowphase`/`_use_coop_dedup` are Python-side build-time booleans/ints.
Hibernation and island gating is device-side inside `_kernel_solve_monolith` (`n_constraints[i_b] > 0`, `n_awake_dofs[i_b] > 0`, constraint/solver.py:5898).
`func_solve_body` is a `@qd.perf_dispatch` keyed on `hash((jac.shape, rigid_config, _n_iterations))` (host metadata only); its timing/autotune warmup is the sole framework-level sync caveat.
The only CUDA host synchronization in a step is the non-blocking `check_errno` at `Simulator.step` start: `_errno_event.synchronize()` plus a pinned `_errno_host` read of the copy enqueued one step earlier (base_solver.py:363-373); the per-step merge `Solver.update_errno` is a device in-place `|=` (base_solver.py:321).
Control writes (`control_dofs_position/_velocity/_force`) are pure torch in-place writes to zero-copy views when `gs.use_zerocopy` (CUDA default), else a device-only fallback kernel; a CUDA input tensor never round-trips the host.
A torch-free control write is either the same device-only fallback kernel (numpy source) or the staged scatter (DLPack source), so it also never round-trips the host.
`detect_collision` (rigid_solver.py:1416) and `Collider.get_contacts` (collider.py:969,1024,1036) are the host readback paths, but neither runs during a substep.
Metal-only `torch.mps.synchronize()` flushes in the write/read helpers have no CUDA equivalent (CUDA shares the default stream).
Those flushes go through `torch_free.mps_synchronize()`, a no-op without PyTorch.

## Known Issues / Gotchas
`_sanitize_sol_params` (line 166) treats solver params as a 7-tuple (`timeconst, dampratio, dmin, dmax, width, mid, power`); negative `timeconst` (direct stiffness/damping parameterization) is unsupported and falls back to defaults.
`dampratio` must be strictly positive — despite its name it is the spring stiffness; `dmin/dmax/mid` clamp to `[1e-4, 0.9999]`, `width >= 0`, `power >= 1`.
The elliptic friction cone + `contact_resolution == signorini` requires `friction_cone == elliptic`, `constraint_solver == Newton`, and no MuJoCo compatibility; `box_box_detection` and the elliptic cone are unsupported under `requires_grad`.
Hibernation is disabled under `requires_grad`; MuJoCo-compat uses its own hibernation velocity tolerance (1e-4 vs 2e-3 default).
`FIXME` at line 476: two `func_vel_at_point`/`func_apply_coupling_force` aliases to be removed after migration.
`FIXME quadrants#946` (line 1338): a graph refuses ndarrays that own a gradient, so gradient-tracking scenes launch substep kernels one by one.
`detect_collision` has `TODO: support batching` (returns only `env_idx=0` pairs).
Constraints with unsupported contact pairs / NaN contacts / NaN forces / NaN acc raise per-env `ErrorCode`s registered in `build()`.

## Routing Table
- `abd/` -> articulated-body-dynamics numerics: forward kinematics/dynamics, mass matrix factor/solve, accessors, inverse kinematics, manual backward, misc/coupling helpers.
- `collider/` -> broadphase + narrowphase collision detection and contact assembly (GJK/EPA/MPR, box/capsule specializations, multi-contact).
- `constraint/` -> constraint row assembly, island partition, Newton/CG solve with line search, noslip, and the differentiable backward.
- `../base_solver.py` -> `Solver`, `GravityMixin`, `TimeBasedMixin` (sibling, read-only).
- `../kinematic_solver.py` -> `KinematicSolver` and geom/vert helpers (`_fill_base_link_geom_offsets` etc.) (sibling, read-only).
- `../../entities/` -> `RigidEntity`, `DroneEntity`, `TerrainEntity` and link/joint descriptions consumed here (sibling, read-only).
- `data.py` -> this solver's struct declarations: `RigidInfo`, `RigidData`, `DynInfo`, `DynState`, the `Links*/Dofs*/Joints*/Geoms*/Verts*/VVerts*/VGeoms*/Entities*/Equalities*` members, `RigidSimStaticConfig`, `RigidErrorCode`.
- `collider/data.py`, `constraint/data.py` -> `ColliderState` and `ConstraintState` (owned by those subpackages).
- `../base_solver.py` -> `Solver` (`_sanitize_io_variables`, `_read_data`, `record_read`, checkpointing) and `ErrorCode`; `../../../utils/data.py` -> `Data`/`DataKind`/`field` and the struct allocation.
- `../../../utils/data.py` + `../../../utils/misc.py` -> `qd_to_torch`/`qd_from_torch`/`qd_to_numpy` transpose + zero-copy conventions (read-only reference).

## Notes for Agents
This sandbox's `rg` returns only ONE match per file and plain `git grep -n` returns empty; use `git grep -o -n -E "<pattern>"` to list definitions with line numbers, and the `read` tool with `max_bytes` up to 131072 for large files.
Do not split or rewrite the fused-pipeline kernels at the bottom of `rigid_solver.py`; they are compiled once and shared by the fused/SAP paths.
Treat `abd/`, `collider/`, `constraint/` CONTEXT.md files as authoritative for their subtrees; keep this file a routing table plus the cross-cutting lifecycle/layout/convention notes.

## Test Strategy
No unit tests live under `rigid/`; physics and differentiable-simulation correctness is exercised by end-to-end tests elsewhere in the repo.
Relevant historical regression themes: batch dim moved to the last index, zero gradient through joints at rest in diff sim, fp64 vs fp32 tolerance and mass-floor stability.

## See Also
`genesis/engine/simulator.py` (`substep` coupling order), `genesis/engine/couplers/` (SAP/IPC couplers), `genesis/options/solvers.py` (`RigidOptions` user surface), `genesis/utils/geom.py` (solver-param defaults and quaternion helpers).
