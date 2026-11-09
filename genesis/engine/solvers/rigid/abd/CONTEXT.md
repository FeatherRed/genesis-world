# Intent

`abd/` holds the articulated-body-dynamics (ABD) compute kernels and device functions of the rigid-body solver.
It implements the physics numerics of the articulated rigid-body simulator: forward kinematics, velocity/acceleration propagation, mass-matrix assembly/factorization/solve, torque/passive/bias forces, integration, state accessors/setters, inverse kinematics, and the manual reverse-mode kernels of the differentiable backward pass.
Everything here is written against the Quadrants DSL (`import quadrants as qd`), not raw Taichi, and exposes no Python classes other than two `IntEnum`s (`ConstraintType`, `LINK_SWEEP_PASS`).
State lives in Quadrants structs (`DynState`, `DynInfo`, `RigidInfo`, `ConstraintState`, `LinksState`, ...) from `genesis.utils.array_class`; this directory only manipulates those structs and caller-owned tensors.

# API Surface

`__init__.py` star-imports `forward_kinematics`, `inverse_kinematics`, `forward_dynamics`, `accessor`, `diff`, and `misc`.
`manual_bw` is NOT star-exported: it is imported by name from `forward_kinematics` and `forward_dynamics`.

## accessor.py — state get/set/control API consumed by `RigidSolver`
- `ConstraintType` (`IntEnum`): selects which sol-params table a solver-parameter setter targets — `GEOM = 0`, `JOINT = 1`, `EQUALITY = 2`.
- State get/set: `kernel_get_kinematic_state`, `kernel_set_kinematic_state`, `kernel_set_state` (full state + friction ratio, resets ctrl force/mode), `kernel_set_qpos`, `kernel_set_links_pos`, `kernel_set_links_quat`, `kernel_set_links_mass`, `kernel_set_links_COM`, `kernel_set_links_inertia`, `kernel_set_vverts`.
- Solver/PD params: `kernel_set_global_sol_params`, `kernel_set_sol_params` (dispatch by `ConstraintType`), `kernel_set_dofs_kp`, `kernel_set_dofs_kv`, `kernel_set_dofs_act_gain`, `kernel_set_dofs_act_bias`, `kernel_set_dofs_force_range`, `kernel_set_dofs_stiffness`, `kernel_set_dofs_armature`, `kernel_set_dofs_damping`, `kernel_set_dofs_frictionloss`, `kernel_set_dofs_limit`.
- DOF control/velocity/position: `kernel_set_dofs_velocity`, `kernel_set_dofs_zero_velocity`, `kernel_set_dofs_position`, `kernel_control_dofs_force`, `kernel_control_dofs_velocity`, `kernel_control_dofs_position`, `kernel_control_dofs_position_velocity` (set `ctrl_mode`/`ctrl_force`/`ctrl_vel`/`ctrl_pos`).
- Wake-up (hibernation): `kernel_wake_up_entities_by_links`, `kernel_wake_up_entities_held_by_links`, `kernel_wake_up_entities_by_dofs`, `kernel_wake_up_entities_by_qs`, plus `func_wakeup_links_island` (wakes the whole constraint island of each hibernated link).
- Read-out: `kernel_get_links_vel`, `kernel_get_links_acc`, `kernel_get_terrain_height`.
- Friction/geometry: `kernel_set_geom_friction(_torsional/_rolling)`, `kernel_set_geoms_friction(_torsional/_rolling)`, `kernel_set_geoms_friction_ratio`.
- Drone: `kernel_set_drone_rpm`, `kernel_update_drone_propeller_vgeoms`.
- Helpers: `func_set_link_mass` (optionally scales inertia by the mass ratio), `func_link_offset_shift`.

## forward_kinematics.py — pose/velocity/geom propagation over kinematic trees
- `LINK_SWEEP_PASS` (`IntEnum`): selects the per-link pass of `func_sweep_links_by_level` — `FORWARD_KINEMATICS = 0`, `FORWARD_VELOCITY = 1`, `ACCELERATION = 2`, `INERTIA_FORCE_FOLD = 3`.
- Entry kernels: `kernel_forward_kinematics_links_geoms` (+ `_masked`), `kernel_forward_kinematics` (+ `_masked`), `kernel_forward_velocity` (+ `_masked`).
- Cartesian/geom/vert updates: `func_update_cartesian_space` / `kernel_update_cartesian_space`, `kernel_update_verts_for_geoms`, `kernel_update_vgeoms`, `kernel_update_vverts_for_vgeoms`, `kernel_update_all_verts`, `func_update_geom_aabbs`, `func_update_geoms`.
- Per-index funcs: `func_update_kinematics_root`, `func_forward_kinematics_link`/`_root`, `func_forward_velocity_link`/`_root`, `func_update_acc_link`, `func_COM_link`/`_root`/`_links`/`_root_sum`, `func_update_geoms_link`/`_root`, `func_update_verts_for_geom`, `func_update_all_verts`, `func_sweep_links_by_level`.
- Replay kernels used only by the manual backward (`kernel_forward_kinematics_replay`, `kernel_update_geoms_replay`, `kernel_COM_links_replay`).
- Custom-backward entry points: `reverse_cartesian_space`, `reverse_forward_velocity` (re-run sub-stages forward, then reverse via `manual_bw`).

## forward_dynamics.py — the core dynamics pipeline (largest module)
- Composer: `func_forward_dynamics` chains torque/passive force → `func_update_acc` → mass matrix + link force → `func_factor_mass` → `func_bias_force` → `func_compute_qacc`.
- Mass matrix: `func_compute_mass_matrix_and_link_force` (+ `_masked`), `func_crb_initialize`/`func_crb_fold`, `func_link_force_initialize`, `func_mass_mat_force`, `func_mass_mat_add_diag`, `func_mass_mat_assemble_tree_cooperative`/`_dof`.
- Factorization (selectable via `rigid_config`): `func_factor_mass` (dispatcher), `func_factor_mass_tree`, `func_factor_mass_tree_global`, `func_factor_mass_tree_tiled`, `func_factor_mass_tree_shared`, `func_factor_mass_tiled`, `func_factor_mass_tiled_masked`, `func_factor_mass_masked`, `func_has_implicit_damping_tree`, `func_enter_neutral_configuration`/`func_exit_neutral_configuration`.
- Solve: `func_solve_mass_block`, `func_solve_mass_batch`.
- Forces: `func_torque_and_passive_force` (ctrl-mode actuation + passive spring/damper/friction, clamps to `force_range`), `func_bias_force` (Coriolis/velocity bias + gravity into `qf_bias`/`force`), `func_compute_qacc` (solve smooth acceleration per mass block).
- Inverse-dynamics weights: `func_init_meaninertia`, `func_init_link_invweight`, `func_init_dofs_invweight`, `func_refresh_links_invweight_and_meaninertia` / `kernel_refresh_invweight_and_meaninertia`.
- Integration: `func_integrate`, `func_implicit_damping`, `func_midpoint_eligible`/`_has_fixed_children`/`_is_aligned`/`_free_body`/`_free_bodies`, `func_count_settled_step`.
- Standalone launches: `update_qacc_from_qvel_delta`, `update_qvel`, `kernel_update_acc`, `kernel_compute_smooth_qacc` (factor + smooth qacc, launched separately in differentiable scenes), `kernel_integrate`, `kernel_forward_dynamics_without_qacc` (forward dynamics minus factor/qacc for differentiable scenes).
- Misc: `func_vel_at_point`, `func_clear_coupling_wrench`, `func_bias_force`.
- File ends with a deprecated-module shim aliasing `genesis.engine.solvers.rigid.rigid_solver_dynamics_decomp` to this module.

## inverse_kinematics.py — damped-least-squares IK on caller-owned scratch
- `func_inverse_kinematics`: multi-restart Gauss-Newton (damped least squares) over stacked multi-target pose error, writing the best config into `ik_state.qpos_best`; reads live solver state only, evaluates on the `IKScratchFK` scratch.
- `func_forward_kinematics_scratch` (entity FK into caller-owned buffers), `func_jacobian_scratch`/`func_get_jacobian` (+ `kernel_get_jacobian`, `kernel_get_jacobian_zero`), `func_integrate_dq_scratch`.
- `kernel_inverse_kinematics_entity` (entity-facing launch wrapper), `kernel_set_ik_targets` (copy targets into `IKTargets`, non-zero-copy fallback), `kernel_forward_kinematics_query`.

## diff.py — state commit and finiteness guard
- `func_copy_next_to_curr`: copy `vel_next`/`qpos_next` into `vel`/`qpos`; any NaN/inf raises `ErrorCode.INVALID_ACC_NAN` for that env and leaves its state unchanged (one thread per velocity/position).
- `kernel_copy_next_to_curr_clear_coupling`: the above plus `func_clear_coupling_wrench`.

## manual_bw.py — manual reverse-mode kernels for the differentiable backward
- Reverses forward stages that Quadrants autodiff silently drops or cannot express: `func_forward_kinematics_link_bw` / `kernel_manual_forward_kinematics_bw`, `func_cd_contraction_bw`, `func_forward_velocity_link_bw` / `kernel_manual_forward_velocity_bw`.
- `kernel_manual_compute_qacc_bw`: reverses the mass solve via the implicit function theorem, seeding `mass_mat.grad` directly and skipping the LDL^T factor chain.
- Chain-rule building blocks (`*_grad_*` quaternion/rotation/motion-cross adjoints) live in `genesis/utils/geom.py`, next to their forwards.

## misc.py — coupling, init, wake/hibernation, atomics, helpers
- `linear_to_lower_tri`: linear index → (row, col) of a (strict) lower-triangular matrix.
- Wake/sleep: `func_wakeup_island`, `func_wakeup_link`, `func_is_awake_link`, `func_is_awake_tree`, `func_hibernate_link`, `func_hibernate_island_if_settled`, `kernel_reset_hibernation`, `kernel_wakeup_coupled_links`.
- Field init: `kernel_init_dof_fields`, `kernel_init_link_fields`, `kernel_init_link_dynamics`, `kernel_init_joint_fields`, `kernel_init_vert_fields`, `kernel_init_vvert_fields`, `kernel_init_geom_fields`, `kernel_init_vgeom_fields`, `kernel_init_entity_fields`, `kernel_init_equality_fields`, `kernel_update_heterogeneous_links_vgeom`, `kernel_update_heterogeneous_link_info`.
- External wrench/coupling: `kernel_apply_links_external_wrench`, `kernel_apply_links_external_wrench_at_pos`, `func_apply_coupling_force`, `func_apply_link_external_wrench`, `func_clear_external_force`, `kernel_clear_external_force`.
- Atomics/conditional field access (autodiff-safe): `func_atomic_add_if`, `func_add_safe_backward`, `func_read_field_if`, `func_write_field_if`, `func_write_and_read_field_if`.
- List helpers: `func_list_range_start`, `func_list_item`, `func_equality_links`.
- File ends with a deprecated-module shim aliasing `genesis.engine.solvers.rigid.rigid_solver_util_decomp` to this module.

# func_* vs kernel_* split

`kernel_*` are launched kernels decorated `@qd.kernel(fastcache=True)` (grid entry points).
`func_*` are Quadrants device functions decorated `@qd.func` — they compose loops and are inlined into the kernels that call them; they do not launch on their own.
There are NO `@gs.jit` decorators in this directory; the `@gs.jit`-level end-to-end compilation lives one level up in `rigid/rigid_solver.py`, which composes these `func_*`/`kernel_*` into each step.
Two autodiff decorators from `genesis.grad.bridge` attach to kernels:
- `@reversed_with(is_backward=True)` declares a kernel is reversed by its own autodiff with a variant argument overwritten (used where the forward variant syncs lanes of a block, which autodiff cannot reverse).
- `@register_custom_backward(fn)` declares a manual backward for a kernel whose gradient autodiff cannot derive (forward kinematics, forward velocity).

# Memory layout / transpose convention

Quadrants tensors are field-major with the environment (batch) as the LAST index, e.g. `dyn_state.dofs.vel[i_d, i_b]`, `rigid_info.qpos[i_q, i_b]`.
Torch-facing array arguments passed into kernels are batch-first, e.g. `qpos[i_b, i_q]`, `mass[i_b_, i_l_]`.
The transpose happens only at the torch↔qd boundary in `genesis/utils/misc.py` via `qd_to_torch(..., transpose=True)` / `qd_from_torch` / `qd_to_numpy`; `transpose=True` moves the first non-batch dimension to the front.
This directory contains no `qd_to_torch`/`qd_from_torch` calls — it indexes qd tensors directly and receives already-transposed torch arrays.
`genesis.utils.data.eoa(tensor)` returns loop axes `(1, 0)` on CPU when the batch (last) dimension is memory-outermost so each thread walks contiguous per-env memory; it is passed as `axes=qd.static(eoa(...))` and, besides ordering, communicates loop bounds to autodiff.

# Constraints

- Source is not modified here; this directory only documents itself.
- All kernels are Quadrants DSL; Taichi syntax is not interchangeable.
- State is only ever reached through the `array_class.py` structs (`DynState`, `DynInfo`, `RigidInfo`, `ConstraintState`, `LinksState`, ...); do not invent new arrays inside kernels.
- `qd.template()` parameters are compile-time switches (`rigid_config`, `is_backward`, `implicit_damping`, ...): changing them changes the kernel cache key.
- Batch/field index order and the `batch_links_info`/`batch_joints_info`/`batch_dofs_info` flags determine whether `dyn_info` arrays are indexed `[i, i_b]` or `[i]`.

# Design Decisions

- `func_*` device functions + `@qd.kernel` launches give the higher-level composer (in `rigid_solver.py`) the granularity to fuse decomposed kernels into one.
- Forward kinematics and forward velocity gradients are hand-written because Quadrants autodiff silently drops their gradients; replay kernels re-run the forward sub-stages to rebuild intermediates before the manual reverses.
- The mass solve's backward seeds `mass_mat.grad` via the implicit function theorem (`kernel_manual_compute_qacc_bw`), avoiding a backward through the LDL^T factor chain; differentiable scenes therefore launch `kernel_forward_dynamics_without_qacc` + `kernel_compute_smooth_qacc` instead of `func_forward_dynamics`' fused path.
- Multiple mass-matrix factorization/solve variants exist (tree vs dof ordering; tiled/cooperative/shared-memory vs global; masked for env subsets) selected by `rigid_config` at compile time.
- Hibernation (`use_hibernation`) skips sleeping bodies and wakes whole constraint islands when setters or forces touch a sleeping link; setters reweigh against the neutral configuration.
- `linear_to_lower_tri` uses a float sqrt with integer post-correction to survive GPUs (Apple Metal) whose sqrt is not correctly rounded for perfect squares.

# Known Issues

- `forward_dynamics.py:876` and `:954` (`FIXME`): diagonal coefficients of the LDL^T `L` are ignored (forced to `1.0`) because they are never read.
- `misc.py:545` (`FIXME`): a commented-out edge-length computation "does not work"; the `.norm()` line below is used instead.
- `misc.py:608` (`TODO`): `GeomsInitAABB` parameter should move to rigid global info.
- `forward_kinematics.py:1243` (`TODO`): once every cartesian-space sub-stage reverses correctly under autodiff, drop the replay kernels and manual reverses and differentiate `kernel_update_cartesian_space` whole.
- `genesis/utils/data.py:eoa` documents upstream Quadrants issues (quadrants#934, #1081, #1082) that this directory works around with `qd.tensor`-allocated ndarrays and `axes=qd.static(eoa(...))`.

# Notes for Agents

- This sandbox's `rg` returns only one match per file and plain `git grep -n` returns empty; use `git grep -o -n -E "<pattern>" -- genesis/engine/solvers/rigid/abd/*.py` to list `def`/`class`/decorators with line numbers.
- `func_*` are `@qd.func`, not `@gs.jit`; do not assume the `func_*`/`kernel_*` naming maps to `gs.jit` pipelines.
- Cross-module import graph is tight: `forward_dynamics` imports `forward_kinematics` + `misc`; `forward_kinematics` imports `manual_bw` + `misc`; `accessor` imports `forward_dynamics` + `misc`; `diff` imports `forward_dynamics`; `inverse_kinematics` imports `forward_kinematics`.
- The two trailing `create_virtual_deprecated_module(...)` calls alias legacy module names (`rigid_solver_dynamics_decomp`, `rigid_solver_util_decomp`); keep them in place.
- File sizes are large (forward_dynamics ~2457 lines, accessor ~1215, forward_kinematics ~1275, misc ~1035); long lines are clipped by the read tool — raise `max_bytes` as needed.

# Dependencies

- `quadrants` (`qd`) — the compute DSL.
- `torch`, `numpy`.
- `genesis` core (`gs`), `genesis.utils.geom` (`gu`), `genesis.utils.linalg` (`lu`), `genesis.utils.simt` (`su`).
- `genesis.utils.data.eoa`, `genesis.utils.misc.qd_isfinite`, `genesis.utils.array_class` state structs.
- `genesis.grad.bridge` (`reversed_with`, `register_custom_backward`).
- No system packages or external services.

# Test Strategy

No unit tests live in this directory; physics correctness is exercised through end-to-end simulation and differentiable-simulation tests elsewhere in the repo.
Precision/autodiff regressions historically relevant here: "Fix zero gradient through joints at rest in diff sim", "Move batch dim in all solver data as last index" (see git log).

# See Also

- `../rigid_solver.py` — orchestrator that composes and launches these kernels (parent node).
- `../constraint/` — constraint solver, island, backward, linesearch, noslip (sibling, read-only).
- `../collider/` — broadphase/narrowphase collision detection (sibling, read-only).
- `genesis/utils/geom.py` — quaternion/rotation adjoint building blocks used by `manual_bw.py`.
- `genesis/grad/bridge.py` — `reversed_with` / `register_custom_backward` autodiff tags.
- `genesis/utils/array_class.py` — the solver state structs this directory operates on.
