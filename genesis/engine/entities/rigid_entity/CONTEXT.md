# Intent
Implements the live rigid-body entities (`KinematicEntity`, `RigidEntity`, `TerrainEntity`, `DroneEntity`) and the dataclass descriptions they are built from.
This is the reference implementation for the project's rigid naming and kernel structure (links, joints, geoms, dofs, qpos, articulated-body dynamics) — keep terminology consistent with `genesis/engine/solvers/rigid/`.
The core split is description-vs-live-object: every live link/joint/geom/equality keeps its `desc`, and `Scene.build` consumes only the fully resolved description, never the asset files again.

# API Surface
## `description.py` — dataclasses consumed by `Scene.build`
- `BaseRigidGeomDescription` — `pos`/`quat` of a geom in its link frame; base of both geom description kinds.
- `RigidVisGeomDescription` — a link's drawn geometry: pose plus `vmesh` (`gs.Mesh`).
- `SignedDistanceFieldDescription` — `n_nodes`, `cell_size`, `has_grad`; `from_geometry(mesh, geom_type, material)` resolves the SDF grid sampling at the material's resolution.
- `RigidGeomDescription` — a collision geom: `type`, `data`, `mesh`, `sdf`, `contype`, `conaffinity`, `density`, `friction`, `friction_torsional`, `friction_rolling`, `sol_params`.
- `RigidJointDescription` — `name`, `type`, `n_qs`, `n_dofs`, `pos`, `quat`, `init_qpos`, `sol_params`, `dofs_motion_ang`, `dofs_motion_vel`, `dofs_limit`, `dofs_invweight`, `dofs_frictionloss`, `dofs_stiffness`, `dofs_damping`, `dofs_armature`, `dofs_act_gain`, `dofs_act_bias`, `dofs_force_range`; `__post_init__` fills every per-dof array default.
- `KinematicLinkDescription` — `name`, `parent_idx`, `root_idx`, `pos`, `quat`, `offset_pos`, `offset_quat`, `is_robot`, `is_aligned`, `joints`, `vgeoms`.
- `RigidLinkDescription` (subclass) — adds `inertial_pos`, `inertial_quat`, `inertia`, `mass`, `invweight`, `geoms`, `contact_exclusions`.
- `RigidEqualityDescription` — `name`, `type`, `objs_name` (two link/joint names), `data`, `sol_params`.
- `KinematicVariantLinkDescription` / `RigidVariantLinkDescription` / `KinematicVariantDescription` — per-environment heterogeneous-variant data (frames, joints, vgeoms, geoms, inertial, `default_armature`).
- `KinematicAttachmentDescription` — `entity_name`/`link_name` of where the base link hangs.
- `Resolution` — transient resolve state: `options`, `is_mujoco_compatible`, `links_inertial_info`; dropped once the description is complete.
- `KinematicEntityDescription` — `morphs` (primary first, then variants), `surface`, `visualize_contact`, `name`, `links`, `variants`, `attachment`; `resolve(...)` classmethod opens/processes every asset exactly once.
- `RigidEntityDescription` (subclass) — `links` re-typed as `RigidLinkDescription` plus `equalities`.
- `TerrainEntityDescription` — adds `terrain_hf`, `terrain_scale`.
- `DroneEntityDescription` — adds `kf`, `km` read from the asset.
- Helpers: `description_from_info(cls, info)` (build a description from parsed keys), `is_link_fixed(links, idx)`, `_balanced_variant_mapping(n_variants, B)`.

## `rigid_entity.py` — the live entities (largest file)
- Module helpers: `get_variants_kinematics(entity, n_envs)` and `compute_forward_kinematics(entities, n_envs, qpos)` (host-side FK from descriptions alone, before the solver allocates).
- `KinematicEntity(Entity)` — lifecycle `_load_model`/`_build`; `attach`/`detach`; state getters `get_pos`, `get_quat`, `get_vel`, `get_ang`, `get_links_pos`, `get_links_quat`, `get_links_vel`, `get_links_ang`, `get_links_acc`, `get_links_acc_ang`, `get_vAABB`; index ranges `link_start`/`link_end`, `joint_start`/`joint_end`, `q_start`/`q_end`, `dof_start`/`dof_end`, `vgeom_start`, `vvert_start`, `vface_start`; counts `n_links`, `n_joints`, `n_dofs`, `n_qs`; accessors `links`, `joints`, `vgeoms`, `get_link`, `get_joint`.
- `RigidEntity(KinematicEntity)` — DOF state `get_dofs_position`/`get_dofs_velocity`/`get_dofs_force`/`get_dofs_acc`/`get_dofs_control_force` and `set_dofs_position`/`set_dofs_velocity` (there is no `set_dofs_force` — drive forces through `control_dofs_force`); control `control_dofs_position`/`control_dofs_velocity`/`control_dofs_force`/`control_dofs_position_velocity`; PD gains `get/set_dofs_kp/kv`; actuator `get_dofs_act_gain`/`act_bias`; `get_dofs_force_range`, `get_dofs_stiffness`, `get_dofs_invweight`, `get_dofs_armature`, `get_dofs_damping`, `get_dofs_frictionloss`; base pose `set_pos`/`set_quat`; wrench `apply_links_external_wrench`; mass/inertia `get_links_mass`, `get_links_COM`, `get_links_inertia`, `set_links_mass`, `set_links_COM`, `set_links_inertia`, `set_mass`, `get_mass`; IK `inverse_kinematics` and `inverse_kinematics_multilink`; motion planning `plan_path` (`planner="RRT"` or `"RRTConnect"`); collision `detect_collision`, `get_contacts`, `get_links_net_contact_force`; energy `get_mass_mat`, `get_kinetic_energy`, `get_potential_energy`, `get_total_energy`; friction `set_friction`, `set_friction_torsional`, `set_friction_rolling`, `set_friction_ratio`; geom/equality ranges `geom_start`/`geom_end`, `vert_start`, `face_start`, `edge_start`, `equality_start`/`equality_end`, `n_geoms`, `n_verts`, `n_faces`, `n_edges`, `n_equalities` and `geoms`/`equalities`.

## Other modules
- `rigid_geom.py` — `RigidGeom` (collision geom: `vert_start`/`face_start`/`edge_start`/`verts_state_start`, `sdf_trimesh`, `visualize_sdf`, `set_friction`/`get_friction`, `trimesh`, `active_envs_mask`, `active_envs_idx`) and `RigidVisGeom` (visual: `vvert_start`/`vface_start`, lazy low-res AABB geometry).
- `rigid_joint.py` — `RigidJoint` (link ↔ parent link): `q_start`/`dof_start`/`q_end`/`dof_end`, `n_qs`/`n_dofs`, `init_qpos`, `get_anchor_pos`/`get_anchor_axis`, `set/get_sol_params`, `link` (child link).
- `rigid_link.py` — `KinematicLink` (visual-only: `get_pos`, `get_quat`, `get_vel`, `get_ang`, `get_vAABB`, `vgeoms`, `vgeom_start`/`vgeom_end`, `n_vgeoms`, `n_vverts`, `n_vfaces`) and `RigidLink` (simulated: `geoms`, `geom_start`/`geom_end`, `get_verts`, `get_AABB`, `apply_external_wrench`/`apply_external_force`/`apply_external_torque`, `set_mass`, `set_COM`, `set_inertia`, `get_invweight`, `get_mass`, `set_friction*`).
- `rigid_equality.py` — `RigidEquality`: `eq_obj1id`/`eq_obj2id`, `eq_data`, `idx`/`idx_local`, `type`, `set/get_sol_params`.
- `terrain_entity.py` — `TerrainEntity(RigidEntity)`: `terrain_hf`, `terrain_scale`, `get_terrain_height(positions, envs_idx)`.
- `drone_entity.py` — `DroneEntity(RigidEntity)`: `set_propellers_rpm`, `update_propeller_vgeoms`, `KF`/`KM`, `n_propellers`, `propellers_idx`, `propellers_spin`, `model`, `COM_link_idx`.
- `DroneEntity.set_propellers_rpm` keeps its once-per-step guard (`_prev_prop_t` vs `sim.cur_step_global`, re-armed by `RigidSolver._restart`) in both residencies: with PyTorch it sanitizes the values and advances a `torch` accumulator with `(revs + rpm.T) % (60 / dt)` before calling `set_drone_rpm`, and without it forwards the raw producer to `RigidSolver.set_drone_propellers_rpm`, which stages, advances and applies the thrust on the device.
- `inertial.py` — `get_local_inertial_from_geom`, `compose_inertial_properties`, `compose_inertial_from_g_infos`, `finalize_inertial`, `GeomInertialInfo`, `LinkInertial`, `LinkInertialInfo`, and constants `RHO_OBJECT`/`RHO_ROBOT`/`RHO_MUJOCO`, `MASS_EPS`, `AABB_EPS`, `INERTIA_RATIO_MAX`.
- `__init__.py` re-exports `DroneEntity`, `KinematicEntity`, `RigidEntity`, `RigidGeom`, `RigidVisGeom`, `RigidJoint`, `RigidLink`, `TerrainEntity`.

# Design Decisions
The `KinematicEntityDescription.resolve()` classmethod opens every asset file (URDF/MJCF/USD/mesh/terrain) exactly once, so the entity is built from the description alone with no further file reads or geometry reprocessing.
Every live object keeps its `desc`; a `get_*` accessor returns the value the simulation currently uses (which a runtime setter may have changed).
The inverse weight of a link/joint is authored as the sentinel `-1.0`; the solver's refresh computes the real value at build, and a world-fixed link carries `0.0` (zeros).
Kinematic entities are commonly used to visualize the target reference motion a rigid entity tracks, so `_align_free_roots` is defined on the base description class and kin/rigid anchor identically — the same qpos maps to the same world pose for both.
Free-root alignment anchors a free root (and any primitive) at its fixed-subtree center of mass and principal axes, making the joint-space mass matrix exactly diagonal.
Heterogeneous variants are stored beside the one set of links (not as separate entities) and dispatched per environment via `_balanced_variant_mapping`, which balances blocks when `n_envs >= n_variants`.
MuJoCo compatibility mode drops Genesis's default rotor armature (deferring to MuJoCo's) and uses `RHO_MUJOCO` density for MJCF-loaded inertials.
The `is_robot` flag selects a different convex-decomposition error threshold (`decompose_robot_error_threshold` vs `decompose_object_error_threshold`), since convexifying robot internals is mostly wasted work.
`DroneEntity` and `TerrainEntity` are `RigidEntity` subclasses that only add data no link describes (propeller links/KF/KM for drones, height field/scales for terrain).
Collision vs visual split: collision geoms (`RigidGeom`) drive contacts and are convexified/post-processed; visual geoms (`RigidVisGeom`) only render and use a lazily built low-res AABB mesh.

# State accessor semantics (view vs copy)
Every state getter here (`get_pos`/`get_quat`/`get_vel`/`get_ang`, `get_links_*`, `get_dofs_position`/`_velocity`/`_force`, link equivalents) forwards to the solver and returns, with PyTorch, a plain `torch.Tensor` on `gs.device` with dtype `gs.tc_float` — never a `qd.Tensor`, never numpy/host (without PyTorch it returns the `DeviceView` of "Torch-free mode").
The solver getters share one path, `Solver._read_data` → `qd_to_torch(..., copy=True)` (`genesis/utils/misc.py:885`): the cached DLPack view of the qd field is cloned contiguously and then masked, so each call returns a fresh device copy, not a live alias into solver memory.
`get_links_vel` additionally composes `cd_vel + cd_ang × (pos − root_COM)` in torch (zero-copy/recording) or launches a gather kernel into a `torch.empty` output — a copy either way.
`envs_idx`/`*_idx_local` selection goes through `indices_to_mask` (`range`→`slice`, int→length-1 slice), so the default "all items" selection is a basic view of that clone while a list/tensor index gathers.
Same-device in-place writes exist only on the setter side: `set_pos`/`set_quat`/`set_qpos`/`set_dofs_velocity`/`control_dofs_*` write the cached `qd_to_torch(..., copy=False)` view under `gs.use_zerocopy`, whereas `set_dofs_position` always clones `dofs.pos` whole and launches `kernel_set_dofs_position`.
Getters clone the whole buffer before slicing, so a caller needing a zero-copy alias must call `qd_to_torch(solver.<field>, transpose=True, copy=False)` itself.

# Gotchas
`is_fixed` on a link is not "has a fixed joint": it means every joint on the link AND on every ancestor is FIXED, i.e. the link cannot move; this decides whether its vertices are computed once and shared across the batch (fixed) or stored per environment (free).
Fixed links keep mass/COM/inertia as `None`/unresolved in the description until an `attach` sets them moving or the build zeroes them — do not read a meaningful inertia off a world-fixed link.
The invweight sentinel is `-1.0`: authored as such for moving links and completed by the solver, so pre-build reads show the sentinel.
Deprecated/removed APIs: `RigidLink.is_free` and `RigidGeom.is_free`/`RigidVisGeom.is_free` (removed), `RigidJoint.get_pos`/`get_quat` (removed — use link-level getters or `get_anchor_pos`/`get_anchor_axis`), `dof_idx`→`dofs_idx`, `q_idx`→`qs_idx`, `set_links_invweight`/`set_dofs_invweight` (removed — invweight is a by-product), `get_trimesh` (deprecated, use `trimesh` property).
`RigidJoint.get_anchor_pos`/`get_anchor_axis` return the anchor point/axis in the world frame, distinct from the deprecated joint `get_pos`.
Heterogeneous entities refuse `attach`, and `get_verts` raises on them; their geoms/vgeoms carry `active_envs_mask`/`active_envs_idx` for per-environment dispatch.
`_balanced_variant_mapping` truncates to `np.arange(B)` when `B < n_variants` — a heterogeneous entity with fewer environments than variants drops the trailing variants.
Getters default to `relative=True`, which strips the morph `offset_pos`/`offset_quat` and the free-root inertial alignment so users read back the pose they authored.
`RigidEntity.get_pos`/`get_quat` are base-link accessors; use `get_links_*` for whole-entity state.
`set_friction` validates the range `[1e-2, 5.0]` for stability; per-geom `set_friction` only checks non-negativity.
`DroneEntity.set_propellers_rpm` arms its once-per-step guard before it writes, so a call that raises (a value of `None`, an RPM layout neither form accepts, a producer the sanitizer rejects) still consumes the step and a retry in the same step hits the once-per-step error.

# Dependencies / See Also
`genesis.engine.mesh` — `Mesh` and `InertialProperties` (unit-density mesh mass properties) feed the inertial composition.
`genesis.utils` — `geom`, `urdf`, `mjcf`, `usd`, `terrain`, `mesh`, `cache`, `path_planning` (the `RRT`/`RRTConnect` planners live here).
`genesis.engine.materials.rigid.Rigid` — material density/friction/gravity-compensation the geoms are resolved against.
`genesis.engine.solvers.rigid.RigidSolver` — owns the index ranges these objects index into; `kinematic_solver.KinematicSolver` handles kinematic entities.
`genesis.engine.states.entities.RigidEntityState` — the snapshot state type for rigid entities.
Sibling (read-only, escalate writes to parent): `../base_entity.py` (`Entity`, `EntityDescription`), `../` for the `entities` package layout.

# Routing Table
- `description.py` → Description dataclasses + resolution (`.resolve`, `_load_*`, `_align_free_roots`, `_describe_*`).
- `rigid_entity.py` → `KinematicEntity`/`RigidEntity` lifecycle, DOF/control API, IK, path planning, FK helpers.
- `rigid_geom.py` → Collision (`RigidGeom`) and visual (`RigidVisGeom`) geometry objects.
- `rigid_link.py` → `KinematicLink`/`RigidLink` with geoms/joints/vgeoms and inertial accessors.
- `rigid_joint.py` → `RigidJoint` (link↔parent), q/dof index ranges, anchors.
- `rigid_equality.py` → `RigidEquality` equality-constraint wrapper.
- `inertial.py` → Inertial estimation/composition helpers and density constants.
- `terrain_entity.py` → `TerrainEntity` height-field entity.
- `drone_entity.py` → `DroneEntity` propeller-thrust entity.

# Torch-free mode
Every module imports PyTorch as `from genesis.utils.torch_compat import HAS_TORCH, torch, require_torch`, so the subpackage imports with no PyTorch installed.
Pose/DOF getters (`get_pos`/`get_quat`/`get_vel`/`get_ang`, `get_links_*`, `get_qpos`, `get_dofs_*`) forward to `RigidSolver` and return a zero-copy `DeviceView` when torch is absent; the base-link row is taken with `torch_free.first_row` instead of `[..., 0, :]`.
Setters (`set_pos`/`set_quat`/`set_qpos`/`set_dofs_*`/`control_dofs_*`, link/geom friction and inertial setters) forward the user value untouched, so numpy arrays and DLPack producers both reach the device without a host roundtrip.
`DroneEntity._build` allocates `_propellers_revs`, the batch-last `(n_propellers, n_envs)` revolution accumulator, as a device buffer without PyTorch (`torch_free.device_tensor_from_numpy` of host zeros, imported locally to break the solvers↔entities cycle), where the torch path keeps a `torch.zeros` tensor.
`DroneEntity.set_propellers_rpm` is therefore device-only without PyTorch: it calls `RigidSolver.set_drone_propellers_rpm` and returns, with no host arithmetic and no `to_numpy()`, and `update_propeller_vgeoms` hands that device accumulator to `kernel_update_drone_propeller_vgeoms` in both residencies.
Entity-local vertex indices (`_free_verts_idx_local`/`_fixed_verts_idx_local`) are `int32` host arrays via the `_empty_verts_idx`/`_verts_idx_range`/`_concat_verts_idx` helpers, and `RigidVisGeom._init_pos_tc`/`_init_quat_tc`, `TerrainEntity._terrain_height_field` and the drone link/spin/vgeom tables have `HAS_TORCH` NumPy branches (numpy arrays without PyTorch, torch tensors with it), the revolution accumulator beside them staying device-resident in both residencies.
Genuine device math raises `require_torch`: `entity visual AABB aggregation`, `entity visual vertex computation`, `entity link Jacobian at a local point`, `entity multi-link inverse kinematics`, `entity motion planning`, `entity AABB aggregation`, `entity vertex readout`, `entity total energy readout`, `entity contact queries`, `entity net contact force readout`, `entity friction ratio assignment`, `entity total mass assignment`, `entity total mass readout`, plus `RigidLink`/`RigidGeom`/`RigidVisGeom` vertex and AABB readouts.

# Test Strategy
No dedicated tests live in this subpackage; behavior is exercised through the engine's scene/rigid integration tests and example scripts.
Focus coverage on description resolution (asset loads once, variant dispatch, align anchoring) and the get/set/control DOF round-trips against `RigidSolver` state.
