## Intent
Collision-detection subsystem of the Genesis rigid-body solver.
It runs broad-phase candidate generation, narrow-phase contact detection, and contact pruning/assembly, writing per-environment contact buffers (`ColliderState.contact_data`) that the inequality-constraint solver consumes.
Almost all code is written in the `quadrants` (qd) DSL, JIT-compiled to Taichi/CPU/GPU backends.

## API Surface
`__init__.py` exports `Collider` and star-imports `broadphase`, `narrowphase`, `box_contact`, `capsule_contact`, `contact`.

### collider.py — orchestration
- `class Collider`: main driver owned by `RigidSolver`.
  - Public: `clear(envs_idx=None)`, `reset(envs_idx=None, *, cache_only=True)`, `detection()`, `get_contacts(as_tensor, to_torch, keep_batch_dim, is_padded)`, `backward(dL_dposition, dL_dnormal, dL_dpenetration)`, `backward_narrowphase()`, `data` (yields solver arrays/configs).
  - Holds `collider_config` (`ColliderStaticConfig`), `collider_state` (`ColliderState`), `collider_info` (`ColliderInfo`), `sdf`, `mpr` (`MPR`), `gjk` (`GJK`), `_support_field` (`SupportField`).
  - `_contact_data_cache` is a plain Python dict (per substep cache for `get_contacts`), kept outside qd state; cleared by `detection`/`reset`/`clear`.
  - `_init_static_config` classifies pairs and resolves per-scene static flags: `has_terrain`, `has_non_box_plane_convex_convex`, `has_convex_specialization`, `has_nonconvex_nonterrain`, `has_prunable_contacts`, `ccd_algorithm` (`MPR`/`MJ_MPR`/`GJK`/`MJ_GJK`), contact budgets.
  - `_compute_collision_pair_idx` builds valid-pair indices, excludes adjacent links (fixed-joint chains), filters qpos0-active and IPC-delegated pairs.
  - `_use_split_narrowphase` = has non-box/plane convex-convex, GPU backend, and not `requires_grad`.
- `func_detection` (qd.func): the whole per-step pipeline (see Design Decisions).
- `kernel_detection` / `kernel_detection_without_graph`: the two compiled entrypoints; gradient-tracking scenes use the no-graph variant (see Known Issues).

### broadphase.py — candidate pairs
- `func_broad_phase`: collision clear → link sweep → geom pair test.
- `_func_broad_phase_links_sap` (sweep-and-prune) vs `_func_broad_phase_links_all_vs_all`, selected by `rigid_config.broadphase_traversal`.
- `func_collision_clear`: clears contacts/candidate pairs, fits link AABBs, invalidates stale contact-cache entries.
- `func_check_links_collision_valid`: drops pairs held by weld/screw equalities or by hibernation.

### narrowphase.py — narrow-phase dispatchers (largest file, ~5074 lines)
- `func_narrowphase_contact0` + `func_narrowphase_multicontact`: split convex pass (first contact then perturbed multi-contact), GPU only.
- `func_narrow_phase_convex_vs_convex`: fused convex-convex pass; per pair calls `func_convex_convex_contact`.
- `func_narrow_phase_diff_convex_vs_convex`: differentiable reverse kernel (recomputes contacts differentiably).
- `func_narrow_phase_convex_specializations`: plane-box and box-box.
- `func_narrow_phase_any_vs_terrain`: convex geom vs terrain via MPR over height-field prism.
- `func_narrow_phase_nonconvex_vs_nonterrain`: SDF vertex-scan (`func_add_polytope_vertex_contacts_sdf[_shell]`).
- Helpers: `func_analytic_contact` (closed-form sphere/capsule/box pairs), `func_prefer_gjk_refinement` (MPR→GJK fallback), `func_contact_overlap_axes`, `func_contact_mpr_terrain`, `_func_multicontact_{run_detection,detect,accept,write,gather}`.

### box_contact.py — box specializations
- `func_plane_box_contact`, `func_sphere_box_contact`, `func_box_box_contact` (MuJoCo engine_collision_box.c port; gated by `gs.box_box_detection`; not supported with `requires_grad`).

### capsule_contact.py — analytical smooth contacts
- `func_capsule_capsule_contact`, `func_sphere_sphere_contact`, `func_sphere_capsule_contact`; segment-segment closest-point via `utils.func_closest_points_on_segments`.

### contact.py — contact buffer management & pruning (~2006 lines)
- Buffer ops: `func_add_contact`, `func_reserve_contacts`, `func_set_contact`, `func_add_diff_contact_input`, `collider_kernel_get_contacts`.
- Lifecycle: `func_collider_reset_env`, `func_collider_clear_env`, `collider_kernel_reset`, `kernel_collider_clear`, `kernel_masked_collider_clear`, `func_promote_woken_contacts`, `func_add_contacts_force_to_links`, `func_is_link_pair_hibernated`.
- Tolerances: `func_compute_geom_rbound`, `func_compute_geom_pair_scale`, `func_compute_geom_pair_scale_mj`, `func_compute_mc_tolerance`.
- Contact frames: `func_contact_orthogonals`, `func_rotate_frame`, `func_perturb_geom_pair`.
- Smooth refinement: `func_refine_smooth_contact_pos`, `func_apply_smooth_refinement` (removes MPR/GJK torque bias on sphere/ellipsoid/capsule).
- Sorting/pruning: `func_contact_order_*`, `func_contact_heapsort`, `func_contact_group_by_normal`, `func_contact_support_hull`, `func_clamp_prune_contacts`, `func_clamp_prune_contacts_coop`.
- Differentiability: `func_set_upstream_grad`.

### gjk.py — Gilbert-Johnson-Keerthi distance
- `class GJK` builds `gjk_config`/`gjk_info` and `gjk_state`/`contact0_gjk_state`/`multicontact_gjk_state` scratch.
- `func_gjk` (MuJoCo-style GJK distance), `func_gjk_intersect`, `func_gjk_contact` (GJK→EPA full pipeline), `func_safe_gjk` + subdistance helpers (`func_gjk_subdistance{_3d,_2d,_1d}`), simplex validity checks (`func_is_colinear`, `func_is_coplanar`, `func_search_valid_simplex_vertex`), `func_count_support`.
- `shrink_sphere` mode treats sphere/capsule as point/segment for shallow-penetration speedups.

### epa.py — Expanding Polytope Algorithm
- `func_epa` / `func_safe_epa`: penetration depth + normal from the GJK simplex; polytope init (`func_epa_init_polytope_{2d,3d,4d}`), horizon/insert helpers, `func_epa_witness`, `func_safe_epa_init`.

### mpr.py — Minkowski Portal Refinement
- `class MPR` builds `mpr_info` and `mpr_state`/`contact0_mpr_state`/`multicontact_mpr_state`.
- `func_mpr_contact` / `func_mpr_contact_from_centers`: primary convex CCD (portal refinement, portal status via `PORTAL_STATUS`).
- Portal/point helpers: `mpr_portal_dir`, `mpr_portal_encapsules_origin`, `mpr_find_pos`, `mpr_find_penetr_*`, `guess_geoms_center`, `compute_support`.

### gjk_support.py / gjk_utils.py — shared helpers
- `func_support` (pair Minkowski support), `support_driver` (per-geom-type dispatch), `support_mesh` (exhaustive mesh vertex scan, cached previous vertex).
- `func_ray_triangle_intersection`, `func_triangle_affine_coords`, `func_point_triangle_intersection`, `func_origin_tetra_intersection`, `func_project_origin_to_plane`.

### diff_gjk.py — differentiable variant
- `func_gjk_contact`: forward multi-contact with 1 + 8 perturbed configurations, stores non-differentiable contact data + differentiable weights.
- `func_extended_epa`, `func_add_diff_contact_input`, `func_contact_orthogonals`, `func_compute_minkowski_point`, `func_differentiable_contact`, `func_plane_contact_frame`, `func_differentiable_plane_contact`, `func_triangle_affine_coords`.
- Used only under `requires_grad` (imported/called from narrowphase), differing from `gjk.py` by storing per-contact reference/weight data for the backward pass.

### multi_contact.py — polygon-clipping multi-contact
- `func_multi_contact` (Sutherland-Hodgman clipping over the EPA nearest face), `func_simplex_dim`, `func_clip_polygon`, `func_approximate_polygon_with_quad`, `func_safe_normalize`, potential-normal/edge helpers.

### support_field.py — SDF support field
- `class SupportField`: precomputed per-direction support tables (two spherical charts, resolution 120) for vertex-hull support lookups.
- `_func_support_{sphere,ellipsoid,capsule,cylinder,prism,box,world,mesh}`, `_func_direction_cell`, `_func_count_supports_*`.

### constants.py — enums
- `CCD_ALGORITHM_CODE`, `RETURN_CODE`, `GJK_RETURN_CODE`, `GJK_STAGE`, `PORTAL_STATUS`, `MPR_STAGE`, `MPR_SUPPORT_TYPE`, `CONTACT_ORDER`, `MULTICONTACT_SLOT`, `MULTICONTACT_ENTRY`, `EPA_POLY_INIT_RETURN_CODE`.

### utils.py — small geometry helpers
- `func_closest_points_on_segments` (Ericson RTCD 5.1.9), `func_det3`, `func_point_in_geom_aabb`, `func_is_geom_boxes_overlap` (AABB + oriented-box SAT), `func_is_discrete_geom(s)`.

## Detection Pipeline Order
`func_detection` runs, per substep and statically gated:
1. `func_update_geom_aabbs` (imported from `..abd.forward_kinematics`) — refresh world AABBs.
2. `func_broad_phase` — clear + link sweep + geom pair test into `broad_collision_pairs`.
3. Narrow phase:
   - split path: `func_narrowphase_contact0` then `func_narrowphase_multicontact`;
   - else `func_narrow_phase_convex_vs_convex`;
   - then `func_narrow_phase_convex_specializations` (plane-box, box-box);
   - then `func_narrow_phase_any_vs_terrain`;
   - then `func_narrow_phase_nonconvex_vs_nonterrain`.
4. `func_clamp_prune_contacts` (or `_coop` cooperative dedup) — hull prune + spatial sort of contacts.
5. `requires_grad` only: `func_fill_diff_contact_input_plane` completes differentiable plane-contact inputs.
`RigidSolver.substep` clears `collider._contact_data_cache` before calling `collider.detection()` between dynamics and inequality-constraint assembly.

## GJK / EPA / MPR Roles
MPR is the default convex CCD: portal refinement along the line between geom centres; it records a `PORTAL_STATUS` (`NONE`/`UNCONVERGED`/`APPROXIMATE`/`EXACT`).
GJK computes minimum distance; when colliding it seeds EPA, which expands the polytope for exact penetration depth + normal.
`func_prefer_gjk_refinement` re-runs GJK/EPA for MPR contacts on unconverged or provably non-minimal portals.
`requires_grad` routes through `diff_gjk`, which replays GJK/EPA over perturbed configurations and stores reference/weight data so the backward pass (`func_narrow_phase_diff_convex_vs_convex`) is differentiable.
Analytical paths (capsule-capsule, sphere-sphere, sphere-capsule, sphere-box, plane-box) bypass GJK/EPA/MPR entirely.

## Constraints
- Do not edit source here in this task; CONTEXT.md updates only.
- All kernel/`func_` code must be valid `quadrants` DSL (`@qd.func`/`@qd.kernel`, `qd.static` for compile-time branches, `qd.loop_config` for parallelism).
- `func_*` are composable subroutines; `kernel_*` are launchable entrypoints; both live at module level.
- The deprecated-module wrappers at the bottom of `collider.py`/`mpr.py` (`create_virtual_deprecated_module` for `collider_decomp`/`mpr_decomp`) must be preserved.

## Design Decisions
- Two-level broad phase: sweep link AABBs first, then test geom pairs only under overlapping link pairs.
- Geom pairs are ordered by ascending `GEOM_TYPE`; PLANE is always `i_ga`, TERRAIN always `i_gb`, which narrow-phase branches rely on.
- Batch (env) dimension is LAST in qd tensors (`[i_c, i_b]` indexing); torch-boundary views transpose it (`qd_to_torch(..., transpose=True)` / `qd_to_numpy`).
- Static config flags (`collider_config.has_*`) let scenes compile only the narrowphase passes they need; outside performance mode (`gs.use_ndarray`) the convex pass and sort are always compiled in as no-ops.
- Contact pruning caps contacts per link pair and per env, and sorts them (`CONTACT_ORDER`); cooperative warp-per-env dedup is used when envs ≤ half the GPU cores.
- Zero-copy `get_contacts` fast path returns torch views when pruning and spatial sort are statically off; otherwise it gathers through `contact_sort_idx`.

## Known Issues / Gotchas
- `collider.py:907` FIXME quadrants#946: gradient-tracking scenes launch detection without a graph.
- `gjk_utils.py:63` FIXME: on Metal, reading a fixed-size vector at a runtime index in a dynamic loop returned 0.0 (divisor → infinite contact position); the scan is manually unrolled until the backend fixes indexing.
- `narrowphase.py:4038` FIXME quadrants#925: every lane issues the claim atomic (other lanes add 0) instead of one atomic per block.
- `gjk.py:410` FIXME: an `if`-statement form of the no-contact reset triggered bugs on some backends (x86 CPU); a conditional expression is used instead.
- `mpr.py:484` FIXME: third portal vertex search can cycle when penetration < `CCD_EPS` and a centre lies outside its convex hull; it gives up after a few trials.
- `collider.py:974` FIXME: MPS gathers zeros through an expanded zero-copy index; a `.contiguous()` copy is forced.
- `box_contact.py:613` TODO and `multi_contact.py:507` TODO (mesh face-normal search would benefit from precomputed adjacency) note non-blocking optimization gaps.
- `narrowphase.py:3001` TODO: smooth refinement is not yet applied to the differentiable contact path.
- Precision: GJK constants (`FLOAT_MIN=1e-15`, `simplex_max_degeneracy_sq=1e-10`, tolerances `1e-6`) are deliberately tiny to avoid backend-dependent norm discrepancies.
- Sphere/capsule contact code keeps `dist == 0` for coincident segments to avoid the infinite gradient of `sqrt` at zero in the differentiable replay.

## Notes for Agents
- `git grep -n` returns empty in this sandbox; use `git grep -o -n -E '^def|^class ...' -- path` for definitions, or the `read` tool.
- `rg` without flags is very slow over these large files; prefer targeted `git grep`/`read`.
- This directory has no subdirectories: everything lives flat at this node, so no child CONTEXT.md files exist.
- Narrow-phase dispatch is the trickiest area to reason about; `collider._init_static_config` decides which passes compile, and `func_detection` is the authoritative ordering.
- Two deprecated modules (`collider_decomp`, `mpr_decomp`) are virtual shims; do not treat them as real code.

## Dependencies
- `quadrants` (qd) DSL + Taichi/CPU/GPU backends; `torch`, `numpy`, `trimesh`.
- `genesis.utils.array_class` structs (`ColliderState`, `ColliderInfo`, `GJKState`, `MPRState`, `DiffContactInput`, etc.).
- `genesis.utils.geom`, `genesis.utils.sdf` (`SDF`), `genesis.utils.simt`, `genesis.utils.data`, `genesis.utils.misc`.
- `genesis.engine.solvers.rigid.abd.forward_kinematics.func_update_geom_aabbs` (AABB refresh).
- `genesis.engine.entities.rigid_entity` (FK helpers used in init).

## Test Strategy
No test files live in this directory.
Integration coverage comes through the Genesis examples/scenes exercising rigid contacts (convex-convex, terrain, SDF nonconvex, box-box, differentiable/`requires_grad`).
Verify contact stability by checking `ColliderState.n_contacts`, `contact_data`, and the smooth-contact drift behaviors documented in `func_refine_smooth_contact_pos`.

## See Also
- `../rigid_solver.py` — owns the `Collider`, drives `detection()` from `substep` (sibling, read-only).
- `../constraint/` — inequality-constraint solver consuming `contact_data` (sibling, read-only).
- `../abd/forward_kinematics.py` — supplies `func_update_geom_aabbs` (sibling, read-only).
- `genesis/utils/sdf.py` — SDF queries used by the nonconvex narrowphase.
- `genesis/utils/array_class.py` — the struct/schema definitions all `*State`/`*Info` fields live in.
