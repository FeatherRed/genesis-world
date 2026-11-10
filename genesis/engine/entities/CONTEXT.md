# Intent
Home of every live simulation entity in Genesis World: rigid bodies, soft/deformable bodies, particles/fluids, and the hybrid rigid+soft compound.
An entity is a runtime object that wraps solver state and exposes a per-entity API (get/set position/velocity, DOF control, muscle actuation, constraints) over global solver index ranges.
The two halves of an entity's life are kept separate: a resolved `*Description` dataclass (what `Scene.build` consumes) and the live `Entity` object it builds (which keeps the description as `desc`).
One entity is driven by exactly one solver; the solver is chosen from the entity's material (`EntityDescription.material`).

# API Surface
## `base_entity.py` — shared foundation
- `EntityDescription` — dataclass, the base of every scene description; holds `material` (the simulator picks the solver from it).
- `Entity(RBC)` — base of all entities; constructor takes `idx, scene, morph, solver, material, surface, name`.
- Properties: `uid`, `idx`, `name`, `scene`, `sim`, `solver`, `surface`, `morph`, `material`, `desc` (None by default — returning a description is what enables `Scene.export`), `is_built`, `sensors`.
- `read_sensors(envs_idx)` — one tensor per sensor class on this entity.
- `_get_global_idx(idx_local, idx_local_max, idx_global_start, unsafe, batched_idx)` — the single local→global index resolver every subclass uses; handles int/slice/range/list/tensor/boolean-mask and Python-style negative wrapping.
- `_get_morph_identifier()` — abstract; subclasses override for auto-naming.
## `particle_entity.py` — particle base
- `assert_active` — decorator rejecting calls on inactive entities (call `entity.activate()` first).
- `ParticleEntity(Entity)` — base for MPM/SPH/PBD entities; owns `particle_size`, `particle_start`, optional `vvert_start`/`vface_start` skinning, `vmesh`, `sampler`.
- Sampling: `sample()` (Mesh/MeshSet/Primitive/Nowhere) and `init_sampler()` (regular/random/pbs/pbs-<sdf_res>).
- `_add_to_solver` → `_add_particles_to_solver` (abstract) + `_add_vverts_to_solver` (chunked nearest-particle skinning supports).
- Control: `set_position` (COM-aware), `set_velocity`, `set_particles_pos/vel/active`, `get_particles_pos/vel/active`, `activate`/`deactivate`/`set_active`, `get_mass`, `find_closest_particle`.
- Properties: `n_particles`, `particle_start`/`particle_end`, `n_vverts`/`n_vfaces`, `vvert_start`/`vvert_end`, `vface_start`/`vface_end`, `particle_size`, `init_particles`.
## `mpm_entity.py` — MPM particles
- `assert_muscle` — decorator gating muscle APIs to `gs.materials.MPM.Muscle`.
- `MPMEntity(ParticleEntity)` — `need_skinning` disabled for Liquid/Sand/Snow; `get_state()` returns `MPMEntityState` (pos, vel, C, F, Jp, active).
- Muscle: `set_actuation`, `set_particles_actu`, `get_particles_actu`, `set/get_muscle_group`, `set_muscle_direction`, `set_muscle`; plus `set_free`/`get_free`.
- Constraints: `get_particles_in_bbox`, `set_particle_constraints`, `remove_particle_constraints`.
## `sph_entity.py` — SPH particles
- `SPHEntity(ParticleEntity)` — `need_skinning=False`; `get_state()` returns `SPHEntityState` (pos, vel); `get_position` aliases `get_particles_pos`; own `init_sampler`.
## `fem_entity.py` — FEM deformables
- `FEMVisGeom` — render-mesh counterpart of `RigidVisGeom`; carries `sim_verts_idx` mapping render vertices to simulated vertices.
- `assert_muscle` — gates muscle APIs to `gs.materials.FEM.Muscle`.
- `FEMEntity(Entity)` — tetrahedral/Cloth deformable; owns `v_start`, `el_start`, `s_start`, `vvert_start`, `vface_start`.
- Construction: `sample()` builds visual geoms then `instantiate(verts, elems)`; Cloth uses surface triangles as elements, volumetric uses TetGen tetrahedralization then `split_all_surface_tets`.
- `_add_to_solver`: `add_cloth` (Cloth) vs `add_elements` + `add_vverts` (volumetric).
- Control: `set_position`/`set_velocity` (COM-aware), `set_verts_pos/vel`, `set_actuation`, `set_muscle`, `set_muscle_group/direction`, `set_vertex_constraints`/`update_constraint_targets`/`remove_vertex_constraints`, `get_el2v`, `compute_pressure_field` (SAP hydroelastic), `get_state()` → `FEMEntityState`.
- Properties: `n_vertices`, `n_elements`, `n_surfaces`, `n_surface_vertices`, `surface_triangles`, `vgeoms`, `v_start`/`el_start`/`s_start`, `vvert_start`/`vface_start` (+ ends), `tet_cfg`.
## `pbd_entity.py` — position-based dynamics
- `PBDBaseEntity(ParticleEntity)` — adds `add_uvs_and_faces`, `fix_particles_to_link`, `fix_particles`, `release_particle`.
- `PBDTetEntity` — edge-based base (`n_edges`, `edge_start`, `mesh`); remeshes the vmesh to `particle_size`.
- `PBD2DEntity(PBDTetEntity)` — CLOTH material type; adds inner (bending) edges.
- `PBD3DEntity(PBDTetEntity)` — ELASTIC material type; tetrahedralizes and adds volume elements (`n_elems`, `elem_start`).
- `PBDParticleEntity(PBDBaseEntity)` — LIQUID material type (`n_fluid_particles`).
- `PBDFreeParticleEntity(PBDBaseEntity)` — non-physics PARTICLE material type (no dynamics relaxation).
## `emitter.py` — particle emission
- `Emitter(RBC)` — `set_entity`, `reset`, `emit` (droplet shapes circle/sphere/square/rectangle), `emit_omni`; recycles a particle pool and accumulates sub-pixel droplet length across substeps; properties `uid`, `entity`, `max_particles`, `solver`, `next_particle`.
## `hybrid_entity.py` — rigid + soft compound
- `HybridEntity(Entity)` — composes a rigid part and a soft (currently MPM-only) part; forwards the rigid DOF API (`get/set/control_dofs_*`), `build`, `update_soft_part`, `_kernel_update_soft_part_mpm` (bidirectional force/motion coupling).
- Properties: `n_dofs`, `fixed`, `part_rigid`, `part_soft`, `solver_rigid`, `solver_soft`.
- Helpers: `augment_link_world_coords`, `default_func_instantiate_soft_from_rigid`, `default_func_instantiate_rigid_from_soft` (mesh skeletonization → URDF), `default_func_instantiate_rigid_soft_association_from_rigid`/`_from_soft`.
## `rigid_entity/` — rigid bodies (see child CONTEXT.md)
- `KinematicEntity`, `RigidEntity`, `TerrainEntity`, `DroneEntity`, plus `RigidLink`/`KinematicLink`, `RigidJoint`, `RigidGeom`/`RigidVisGeom`, `RigidEquality`, and all `*Description` dataclasses.

# Design Decisions
Every entity is created from exactly one `Morph` (the `Entity` constructor rejects anything else); a multi-part entity like `HybridEntity` passes its primary morph and builds sub-parts via `scene.add_entity`.
Description-vs-live split: descriptions are fully resolved dataclasses (no fallbacks at build time); live entities keep the description as `desc` and expose the value the simulation currently uses.
Index ranges are owned by the solver and only referenced by the entity: each entity records `*_start` offsets (`particle_start`, `v_start`/`el_start`/`s_start`, `edge_start`/`elem_start`, `q_start`/`dof_start`, `vvert_start`/`vface_start`) and maps local indices to global ones through `_get_global_idx`.
Batched-env convention: IO methods take `envs_idx` (None = all envs) and per-item local indices with shape `([M,] [N,] ...)`, M selected envs and N items; boolean masks and negative Python indices are supported, but the torch-free read path is stricter than the setters (see "Torch-free getter/view contract").
Particle entities decouple the render mesh (`vverts`/`vfaces`, skinned by nearest-particle supports) from the simulation particles; skinning can be disabled (`need_skinning=False`) for fluids/granulars.
FEM Cloth reuses surface triangles as simulation elements (no tetrahedralization); volumetric FEM welds sub-meshes and tetrahedralizes, keeping input vertices first so `FEMVisGeom.sim_verts_idx` stays valid.
Muscle actuation is material-gated (`assert_muscle`) and grouped: MPM groups per particle, FEM groups per element, both via `material.n_groups`.

# Gotchas
`assert_active` blocks particle/muscle state mutations on inactive entities; a `Nowhere` morph yields an inactive entity that only an `Emitter` fills, and such entities are unsupported in differentiable mode.
Particle/FEM `set_position` with a single `(3,)` value moves the center of mass (each particle/vertex keeps its initial offset), not all points to that value.
MPM `need_skinning` is disabled only for Liquid/Sand/Snow; muscle MPM keeps skinning.
FEM `n_elements` means surface triangles for Cloth but tetrahedra otherwise; `el2v` maps `(n_elements, 4)`.
PBD variants map to solver material constants: `PBD2DEntity`→CLOTH, `PBD3DEntity`→ELASTIC, `PBDParticleEntity`→LIQUID, `PBDFreeParticleEntity`→PARTICLE (non-physics).
`HybridEntity` currently asserts a `gs.materials.MPM.Base` soft part (FEM/PBD are TODOs), hardcodes a `Smooth(roughness=0.4)` rigid surface (HACK), and requires rigid and soft solver `dt` to match; soft-to-rigid velocity coupling (`soft_dv_coef`) can form an unstable feedback loop.
`HybridEntity._update_soft_part_at_pre_coupling` is `False`: coupling runs after MPM g2p on the `substep_post_coupling` hook, and the kernel uses `f + 1` for the post-g2p frame.
`Emitter` accumulates sub-pixel droplet length across substeps (short droplets are skipped, not lost) and recycles its particle pool when the next index would exceed the entity's particle count.

# Known Issues
A torch-free scene with `n_envs == 0` cannot be built today: `RigidSolver.get_state` → `update_forward_pos` passes `scene._envs_idx` (a NumPy array) into `Scene._sanitize_envs_idx`, which rejects a selection on a non-parallelized scene.
Only batched scenes therefore have the torch-free device-view contract.

# Dependencies / See Also
`genesis/engine/materials` — `Material` base and `gs.materials.{MPM,FEM,PBD,SPH,Hybrid,Rigid}`; provides `idx`, `sampler`, `n_groups`, density/friction/elasticity.
`genesis/engine/solvers` — `Solver` base, `rigid.RigidSolver`, `kinematic_solver.KinematicSolver`, and particle/PBD solvers that own the index ranges and `add_*`/`set_*`/`get_*` kernels these entities call.
`genesis/engine/states/entities` — snapshot state dataclasses: `RigidEntityState`, `FEMEntityState`, `MPMEntityState`, `SPHEntityState`.
`genesis/options` — `morphs.Morph` (and `Mesh`, `MeshSet`, `Primitive`, `URDF`, `Nowhere`, `Drone`) and `surfaces.Surface`.
`genesis/utils` — `geom`, `mesh`, `particle`, `element`, `hybrid`, `misc`, `urdf`, `mjcf`, `usd`, `terrain`, `cache`.
`genesis/engine/mesh` — `Mesh` and `InertialProperties`; `genesis/engine/couplers.SAPCoupler` (FEM hydroelastic pressure fields); `genesis/repr_base.RBC`.

# Routing Table
- `base_entity.py` → `EntityDescription`, `Entity` base, index/name/sensor machinery.
- `particle_entity.py` → `ParticleEntity` base + `assert_active`.
- `mpm_entity.py` → `MPMEntity`, MPM muscle/constraint API.
- `sph_entity.py` → `SPHEntity`.
- `fem_entity.py` → `FEMEntity`, `FEMVisGeom`, FEM muscle/constraint API.
- `pbd_entity.py` → PBD entity family (tet/2D/3D/particle/free-particle).
- `emitter.py` → `Emitter` particle emission.
- `hybrid_entity.py` → `HybridEntity` rigid+soft coupling and default instantiation helpers.
- `torch_free.py` → shared torch-free helpers for the whole subtree (`first_row`, `as_device_index`, `as_index_array`, `is_device_view`, `to_numpy`, `stack_values`); see "Torch-free mode".
- `rigid_entity/` → Rigid-body entities, links, joints, geoms, equalities, and their descriptions (see its CONTEXT.md).

# Torch-free mode
PyTorch is optional for the whole subtree: every module imports it as `from genesis.utils.torch_compat import HAS_TORCH, torch, require_torch` and never as `import torch`, so `import genesis.engine.entities` works with no PyTorch installed.
Reads: getters forward to the solver, which returns either a torch tensor (torch present) or a `genesis.utils.device_buffer.DeviceView` zero-copy DLPack view (torch absent); entities never materialize a host copy on a hot path, and the view shapes and semantics are the subject of "Torch-free getter/view contract" below.
Index selection on a read (base link of a multi-link result) goes through `torch_free.first_row`, the equivalent of `value[..., 0, :]` that also works on a `DeviceView` (which has no `...` support).
Writes: entity setters forward the user value untouched to the solver setter, so a numpy array and any DLPack producer (`__dlpack__`/`__dlpack_device__`, e.g. a compiled-graph output) both reach the device without a host roundtrip.
Index bookkeeping: `Entity._get_global_idx` normalizes local index collections through `torch_free.as_device_index` (`torch.as_tensor(..., dtype=gs.tc_int, device=gs.device)` under torch, a contiguous `int32` host array otherwise) and accepts numpy/`DeviceView` boolean masks, while the solver view it feeds still rejects a non-contiguous selection on the read path.
Build-time buffers that used to be `torch.tensor`/`torch.arange`/`torch.from_numpy` (`_init_particles_offset`, `init_positions`, `_free_verts_idx_local`, `_terrain_height_field`, drone propeller tables, `RigidVisGeom._init_pos_tc`) each have a NumPy branch selected by `HAS_TORCH`.
Genuinely torch-only device math raises `require_torch("<feature>")` with a precise feature name, keeping the torch body byte-identical: entity AABB/vertex readouts, contacts and net contact force, total-mass get/set, friction-ratio assignment, multi-link IK, path planning, particle `get_mass`/`find_closest_particle`/bbox queries, FEM muscle actuation, and the `broadcast_tensor`-based COM placement (`ParticleEntity.set_position`, `FEMEntity.set_position`).
`gs.use_zerocopy` is `ZeroCopyMode(True)` without torch, so the solver reads are DLPack views; boolean Quadrants buffers read back as `DeviceView`s of `np.bool_` (`device_buffer` accepts the DLPack `kDLBool` code `6`), so a mask may stay device-resident too.
Every vec3 read depends on `view_of(..., transpose=True)`, which moves the batch axis, `grid_ndim - 1`, to the front and leaves the element axes trailing: Quadrants exports a `Vector.field(3, shape=(m, n))` as DLPack `(m, n, 3)` with `lanes=1`, so a transposed read holds `(n, m, 3)` and never `(3, m, n)`.
Torch-present behaviour is unchanged and byte-identical; a call site may therefore use these helpers unconditionally.

## Torch-free getter/view contract
Torch-free, the entity getters hand back zero-copy `DeviceView`s on the solver buffers with the batch axis first and the element axes trailing, while the solver field itself stays batch-last.
`get_pos`/`get_quat`/`get_ang` are therefore `(n_envs, 3)`/`(n_envs, 4)`/`(n_envs, 3)` with strides `(3, 1)`/`(4, 1)`/`(3, 1)`.
`get_dofs_position`/`get_dofs_velocity` are `(n_envs, n_dofs)` with strides `(1, n_envs)`, so the batch axis is the fast one and the view is not contiguous.
The `n_envs == 1` case reports `is_contiguous == False` although NumPy agrees that the single batch entry is effectively contiguous.
The link getters follow the same rule: `get_links_pos()` is `(n_envs, n_links, 3)` with strides `(3, n_envs * 3, 1)`.
`get_vel` (with `get_links_vel`/`get_links_acc`) is the exception: it returns a freshly allocated per-call device snapshot (~0.45 ms/call against 24-40 µs for the live getters), so its values do not follow a later `scene.step()`, whereas every other getter of that family is a live window on the solver buffer.
Every entity getter view is a derived view because `first_row` selects the base link, so `is_base_view == False` and its pointer is not the field base.
Read selections accept `None`, an int, a slice, a `range` and a contiguous ascending index collection; `[2, 1]`, `[0, 2]` and a non-contiguous boolean mask raise `GenesisException` on the read path, unlike the setters, which accept those forms.
The scene-level `scene.get_error_envs_mask()` that the entity layer reads follows the same contract: a `torch.bool` tensor with PyTorch and a boolean-dtype `DeviceView` (`np.bool_`) without it.

# Test Strategy
No dedicated tests live here; entity behavior is exercised through the engine's scene integration tests and example scripts.
Priority coverage: description resolution → entity build round-trips, local↔global index mapping, get/set/control state consistency against solver state, and particle/FEM skinning correctness.
