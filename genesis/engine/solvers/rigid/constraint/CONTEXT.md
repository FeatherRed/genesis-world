# Intent

Constraint-solver subsystem of the rigid-body solver (RigidSolver).
It assembles equality and inequality constraint rows into a sparse Jacobian/diag/aref system, partitions the dof-carrying kinematic trees into decoupled constraint islands, iterates Newton / conjugate-gradient (CG) with a MuJoCo-style line search to find the constraint forces, then post-processes with optional no-slip friction sweeps and writes back accelerations and contact forces.
It also implements the differentiable backward pass (adjoint solve + manual reverses) for `requires_grad` simulations.

# Module Map

No child directories — the subsystem is six modules plus `__init__.py`, listed below with their responsibilities.

- `__init__.py` → exports `ConstraintSolver`, then imports `solver` (declares the `func_solve_body` perf-dispatch) and finally `solver_breakdown` (registers the decomposed arm against `func_solve_body`). Import order is load-bearing.
- `solver.py` (~6072 lines) → `ConstraintSolver` class, all forward row-assembly funcs, the solve pipeline, and the forward kernel entry points.
- `backward.py` (~1442 lines) → differentiable backward: adjoint solve, gradient computation, and hand-written reverses of the constraint assembly.
- `island.py` (~996 lines) → union-find island partition, constraint grouping by island, contact sorting, and RCM dof reorder for the CPU skyline path.
- `linesearch.py` (~1634 lines) → per-island Newton/CG iteration: MuJoCo bracketing line search as a state machine, exit decision, warm-start certificate.
- `noslip.py` (~620 lines) → matrix-free no-slip friction-force refinement sweeps (Gauss-Seidel with row coloring).
- `solver_breakdown.py` (~375 lines) → decomposed (per-island parallel) solve arm and its CUDA-graph / host-side iteration loop.

# API Surface

## solver.py — `ConstraintSolver`

Constructor `ConstraintSolver(rigid_solver)` builds `self.constraint_state` (a `ConstraintState` from `genesis/utils/array_class.py`) sized from the rigid solver's links/joints/dofs, the `max_contacts` budget, `rows_per_contact`, and the solver options.

Public methods:
- `reset(envs_idx=None)` → clears caches, warmstart flags, and `qacc_ws` (zerocopy fast path or `constraint_solver_kernel_reset`).
- `clear(envs_idx=None)` → `reset` plus zero constraint counts/equality counts (masked-clear or plain-clear kernels).
- `add_equality_constraints()` → launches `kernel_add_equality_constraints` (JOINT/CONNECT/WELD/SCREW rows first).
- `add_inequality_constraints()` → launches `kernel_add_inequality_constraints` (sort+build islands, frictionloss, collision/contact, joint limits).
- `resolve()` → runs `func_solve_body` (perf-dispatch) then `kernel_resolve_post` (update qacc/qfrc_constraint, optional noslip, contact forces).
- `get_equality_constraints(as_tensor, to_torch)` / `get_weld_constraints(...)` → read back equality/weld data (kernel_get_equality_constraints); cached in `_eq_const_info_cache`.
- `add_dynamic_constraint(...)` / `delete_dynamic_constraint(...)` → runtime SCREW/CONNECT/WELD add/remove via `kernel_add_dynamic_constraint` / `kernel_delete_dynamic_constraint`.
- `backward()` → adjoint solve (`kernel_solve_adjoint_u` + `kernel_compute_gradients`); requires `requires_grad=True`.
- `data` property → yields every array of `constraint_state` tagged by kind (for the composable data system).

Instance attributes: `constraint_state`, `_solver`, `_collider`, `_B`, `_para_level`, `_solver_type` (`gs.constraint_solver`), `_n_iterations`, `tolerance`, `ls_iterations`, `ls_tolerance`, `sparse_solve`, `len_constraints`, `_eq_const_info_cache`.
It also aliases many `ConstraintState` fields (e.g. `jac`, `diag`, `aref`, `efc_D`, `efc_force`, `active`, `qacc`, `qacc_ws`, `qfrc_constraint`, `grad`, `Mgrad`, `search`, `Jaref`, `Ma`, `Ma_ws`, `mv`, `jv`, plus `cg_prev_grad`/`cg_prev_Mgrad` for CG and `nt_H`/`nt_vec` for Newton) for compact kernel-argument passing.

Key forward entry kernels (`@qd.kernel(fastcache=True)`): `kernel_get_equality_constraints`, `constraint_solver_kernel_reset`/`_clear`/`_masked_clear`, `kernel_add_equality_constraints`, `kernel_add_inequality_constraints`, `kernel_add_dynamic_constraint`, `kernel_delete_dynamic_constraint`, `func_solve_init`, `_kernel_solve_monolith`, `kernel_resolve_post`.

Key row-assembly funcs: `func_equality_jdotv`, `func_equality_links`, `func_equality_joint`, `func_screw_frame`, `func_add_relative_motion_rows`, `func_add_dynamic_constraint_rows_batch`, `_func_contact_row_direction`, `_add_friction_constraint`, `_add_collision_constraints_per_contact` / `_per_friction`, `add_collision_constraints`, `_func_joint_limit_gap`, `_func_add_joint_limit_row`, `add_limit_constraints`, `add_frictionloss_constraints`.

Key solve funcs: `func_solve_body` (a `@qd.perf_dispatch`), `func_solve_body_monolith`, `func_solve_init`, `func_solve_iter`, `func_solve_seed_batch`, the Hessian/Cholesky family (`func_hessian_*`, `func_factor_island_*`, `func_add_cone_hessian_block*`, `func_island_hessian_*`), `func_cone_update_rows`, `func_update_gradient_batch`, `func_contact_constraint_force`, `func_update_contact_force`, `func_resolve_post`, `func_update_qacc`.

## backward.py

Adjoint / differentiable path:
- `kernel_load_dL_dqacc_from_acc_grad` → loads the upstream `dL_dqacc`.
- `kernel_accumulate_constraint_solver_grads` → accumulates solver grads.
- `kernel_solve_adjoint_u` → solves `A u = g` with `A = M + J^T diag(D) J`; dispatches to CG (`func_solve_adjoint_u_cg_batch`, `func_matvec_Ap`) or per-island Cholesky.
- `kernel_compute_gradients` → computes `dL_dM`, `dL_djac`, `dL_daref`, `dL_defc_D`, `dL_dforce` from the adjoint `u`.

Manual reverses (replay forward assembly to accumulate kinematic-state grads, model params not differentiated):
- `kernel_manual_add_joint_limit_constraints_bw`, `kernel_manual_add_collision_constraints_bw`, `kernel_manual_add_frictionloss_constraints_bw`, `kernel_manual_add_equality_constraints_bw` (JOINT + CONNECT + WELD), plus helpers `func_cddb_ang_bw`, `func_equality_jdotv_bw`.

## island.py

Union-find over dof-carrying kinematic trees with edges from live contacts, equality constraints, and hibernation chains → connected components are the exactly decoupled (block-diagonal) sub-problems of the solve; components labeled by smallest tree for determinism.
`func_build_islands` (serial per-env) / `func_build_islands_coop`, `func_build_single_island` / `_coop`, `func_group_constraints_by_island` / `_coop`, `func_reorder_island_dofs` (RCM, CPU skyline only), `func_contact_island`, `func_sort_contacts` / `_coop`, `func_dof_range_start`, `func_chunk_island_rank`.

## linesearch.py

Per-island Newton/CG iteration where each island owns its own line search, convergence test, and search direction (measured against its own mass-matrix trace `IslandState.inertia`).
`func_linesearch_islands_serial` / `_coop`, `func_exit_islands_serial` / `_coop`, `func_exit_decision` (convergence test, also computes CG Hager-Zhang beta), `func_certify_decision` (warm-start certificate), per-row cost terms `func_row_p0_terms` / `func_row_alpha_terms` / `func_row_alpha_sums_by_class`, and the mv/jv products `func_mv_jv_dense` / `_islands` / `_coop`.

## noslip.py

Matrix-free no-slip post-refinement of friction forces: `func_color_rows_batch` (graph-colors rows so a color touches disjoint mass blocks), `func_solve_mass_block` / `func_apply_Minv_rows` (block LDL^T solves of `M^-1 J^T`), `func_refresh_qacc_batch`, `func_noslip_update_row`, `func_noslip_batch`, `func_noslip` (cooperative/colored variants), `func_dual_finish_batch`, `func_cost_change`.

## solver_breakdown.py

Decomposed solve arm registered on `func_solve_body`: per-island parallelism with a CUDA-graph / host-side iteration loop.
`_func_update_constraint_forces*`, `_func_update_qfrc_constraint_per_dof`, `_func_update_gradient`, `_func_islands_linesearch_and_apply`, `_func_islands_update_search_direction`, `_kernel_solve_graph`, `func_solve_decomposed`.

# Solve Pipeline

1. Equality assembly: `add_equality_constraints` → `func_add_equality_constraints` builds JOINT (1 row, optional quartic coupling), CONNECT (3 position rows), WELD (6 rows) and SCREW (coupling + friction + limit rows) as the first constraint group.
2. Inequality assembly: `add_inequality_constraints` → `func_add_inequality_constraints` sorts contacts and builds islands (`_sort_contacts_and_build_islands`), then appends dof frictionloss rows, contact/friction-cone rows, and joint-limit rows.
3. Solve: `resolve` → `func_solve_body` (perf-dispatch) then `kernel_resolve_post`.
   - `func_solve_init` groups constraints by island, seeds the Newton factor (persisting L in `nt_H`) and gradient, resets Signorini cone latches, and picks warmstart (or MuJoCo best-of-warmstart/current).
   - iteration (`func_solve_iter`, or `_kernel_solve_graph` for the decomposed arm): line search + step → update active flags/constraint forces → Newton Hessian factor (incremental rank-1 update/downdate with direct-refactor fallback) or CG direction update → gradient → per-island exit decision.
   - `func_resolve_post`: `func_update_qacc` (write acc/qfrc_constraint, warmstart qacc_ws, NaN check → `ErrorCode.INVALID_FORCE_NAN`), optional `func_noslip`, then `func_update_contact_force`.

# Solver Types & Flags (static `rigid_config` template)

- `solver_type` = `gs.constraint_solver.Newton` (default; Hessian + incremental Cholesky) or `CG` (Hager-Zhang conjugate direction). No GaussSeidel solver type exists here — Gauss-Seidel appears only as the noslip sweep method.
- `friction_cone` = `pyramidal` (default; opposing friction-mixed edge rows per tangent axis) or `elliptic` (second-order cone coupled in the Newton solver).
- `contact_resolution` = `signorini` (friction bounded by the developed normal force) or `convex` (single convex program); default resolves to signorini for elliptic+Newton, convex otherwise.
- `enable_signorini_contact`, `enable_elliptic_friction`, `enable_torsional_friction` (spin row), `enable_rolling_friction` (two extra axes, requires torsional).
- `noslip_iterations` / `noslip_tolerance`: experimental post-refinement (default 0 = off; unsupported with elliptic cone and differentiable simulation).
- `impratio`: tangential-to-normal impedance ratio (stiffens friction for resting stacks).
- Per-constraint solver params (timeconst, dampratio, dmin, dmax, width, mid, power) are sanitized by `_sanitize_sol_params` in `rigid_solver.py` and consumed by `gu.imp_aref` to produce `imp`/`aref` for diag.

# Constraints

- Quadrants (`qd`) kernels take structs (`ConstraintState`, `DynState`, `RigidInfo`, etc.) as `qd.template()` arguments; passing a struct member positionally alongside its parent struct breaks quadrants' func-argument expansion (several call sites use keyword calls to avoid it).
- Quadrants assumes distinct kernel arguments never alias: `nt_H` reuses `rigid_info.mass_mat_tiled_scratch` via `_binds`, so no kernel may access both (the factor lives only in `forward_dynamics.py`).
- Naming convention: `kernel_*` = top-level `@qd.kernel(fastcache=True)` entry points launched from Python; `func_*` = `@qd.func` helpers plus a few composed `@qd.kernel`/`@qd.perf_dispatch` pipelines (`func_solve_init`, `func_solve_body`, `func_solve_body_monolith`, `func_solve_decomposed`).
- Memory layout: `qd` tensors use batch-dim-LAST under `constraint_layout_batch_first`; `qacc`/`qacc_ws` are env-leading under the DOF-vec flip while `dofs.acc_smooth` stays canonical `(n_dofs, _B)` — ndrange axes are swapped to coalesce writes.
- Row support (`jac_dofs_idx`) is a set: shared-ancestor dofs must be deduplicated on append (`_append_relevant_dof`) or a parallel GPU assembly can spill past the row capacity into the next row.

# Design Decisions

- Island decoupling turns the solve into block-diagonal per-island sub-problems, each with its own line search, exit test, and inertia scale; a converged/sleeping island stands still while others iterate.
- `func_solve_body` is a perf-dispatch with two arms: monolith (one thread per env, 32 envs per warp) vs decomposed (per-island parallelism); each entrypoint owns its `func_solve_init` behavior so autotuning can switch arms per step.
- Newton Hessian `H = M + J^T diag(D active) J` is fixed within a step (only the active mask changes), so the factor is maintained by rank-1 update/downdate per flipped row, with degenerate downdate or excessive flips falling back to a direct refactor.
- The differentiable path solves the adjoint `A u = g` (CG over the batch or island Cholesky) rather than differentiating the iterative loop; manual reverses replay the assembly to accumulate only kinematic-state grads.

# Known Issues

- `add_limit_constraints` and `add_frictionloss_constraints` have `# TODO: sparse mode` — dense loops only.
- Plane-capsule contact tangent frame is anchored to the normal, not the capsule axis (`FIXME`), so plane-capsule friction rows differ from the reference engine; anchoring it would require hand-deriving the quaternion adjoint in `kernel_manual_add_collision_constraints_bw`.
- Apple Metal: `add_frictionloss_constraints` disables `serialize` because the frictionloss threshold condition is mis-evaluated under `serialize=True` on Metal (`FIXME`).
- MuJoCo compatibility gate in `func_exit_decision` and `func_solve_seed_batch` (`# TODO: Remove ... once on MuJoCo 3.11`).
- Joint-equality rows must populate `jac_dofs_idx`/`jac_n_dofs`; otherwise sparse-Jacobian iterations see zero support, produce zero forces, and yield NaN (see `func_equality_joint`).
- Final NaN check in `func_update_qacc` sets `ErrorCode.INVALID_FORCE_NAN` per env.
- `island.py` guards the empty dof loop of a dof-less tree because quadrants debug builds check loop-invariant accesses before the loop runs (`FIXME`).
- `noslip` is experimental, slows the solve, and is unsupported with the elliptic cone or differentiable mode.

# Notes for Agents

- Import order in `__init__.py` is load-bearing: `solver` must declare `func_solve_body` before `solver_breakdown` registers the decomposed arm.
- This sandbox's `rg` returns only one match per file and plain `git grep -n` returns empty; use `git grep -o -n -E "pattern"` to list all matches, and the `read` tool with `max_bytes` up to 131072 for large files.
- These files are hot, hand-tuned quadrants kernels with heavy layout/coalescing reasoning in comments — read the docstrings before refactoring; they encode non-obvious aliasing and parallelism invariants.

# Dependencies

- `quadrants` (`qd`) kernel DSL; `genesis.utils.array_class` (`ConstraintState`, `DynState`, `RigidInfo`, `ColliderState`, `DynInfo`, `ErrorCode`); `genesis.utils.geom` (`gu.imp_aref`, `qd_orthogonals`, quaternion/transform grads).
- `genesis.engine.solvers.rigid.abd` (`func_solve_mass_batch`, `func_vel_at_point`) and `abd.misc` (hibernation/wakeup, `linear_to_lower_tri`, list helpers).
- `genesis.engine.solvers.rigid.collider.contact` (`func_add_contacts_force_to_links`, `func_contact_order_key`, `func_promote_woken_contacts`).
- `genesis.grad.bridge.no_grad`, `genesis.utils.misc` (`assign_indexed_tensor`, `qd_to_torch`, `qd_to_numpy`).

# See Also

- `../rigid_solver.py` → the owner that configures `rigid_config`, sanitizes solver params (`_sanitize_sol_params`), and drives this solver's `resolve`/`backward` per substep (sibling — read-only).
- `../../abd/` → articulation/forward dynamics, mass-matrix factor the constraint solve reuses (sibling — read-only).
- `../collider/` → contact generation feeding `add_inequality_constraints` (sibling — read-only).
- `../../../utils/array_class.py` → `ConstraintState`/`IslandState`/`DynState` layouts and capacity definitions (read-only reference).
- `../../../../options/solvers.py` → user-facing docs for `constraint_solver`, `friction_cone`, `contact_resolution`, `noslip_*`, `impratio`.
