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
Batched-env convention: IO methods take `envs_idx` (None = all envs) and per-item local indices with shape `([M,] [N,] ...)`, M selected envs and N items; boolean masks and negative Python indices are supported.
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
Reads: getters forward to the solver, which returns either a torch tensor (torch present) or a `genesis.utils.device_buffer.DeviceView` zero-copy DLPack view (torch absent); entities never materialize a host copy on a hot path.
Index selection on a read (base link of a multi-link result) goes through `torch_free.first_row`, the equivalent of `value[..., 0, :]` that also works on a `DeviceView` (which has no `...` support).
Writes: entity setters forward the user value untouched to the solver setter, so a numpy array and any DLPack producer (`__dlpack__`/`__dlpack_device__`, e.g. a compiled-graph output) both reach the device without a host roundtrip.
Index bookkeeping: `Entity._get_global_idx` normalizes local index collections through `torch_free.as_device_index` (`torch.as_tensor(..., dtype=gs.tc_int, device=gs.device)` under torch, a contiguous `int32` host array otherwise) and accepts numpy/`DeviceView` boolean masks.
Build-time buffers that used to be `torch.tensor`/`torch.arange`/`torch.from_numpy` (`_init_particles_offset`, `init_positions`, `_free_verts_idx_local`, `_terrain_height_field`, drone propeller tables, `RigidVisGeom._init_pos_tc`) each have a NumPy branch selected by `HAS_TORCH`.
Genuinely torch-only device math raises `require_torch("<feature>")` with a precise feature name, keeping the torch body byte-identical: entity AABB/vertex readouts, contacts and net contact force, total-mass get/set, friction-ratio assignment, multi-link IK, path planning, particle `get_mass`/`find_closest_particle`/bbox queries, FEM muscle actuation, and the `broadcast_tensor`-based COM placement (`ParticleEntity.set_position`, `FEMEntity.set_position`).
`gs.use_zerocopy` is `ZeroCopyMode(True)` without torch, so the solver reads are DLPack views; boolean Quadrants buffers cannot be read back yet (`device_buffer` rejects the DLPack bool code), so boolean masks stay host numpy arrays.
Every vec3 read depends on `view_of(..., transpose=True)`, which currently moves the wrong axis for multi-component buffers: Quadrants exports a `Vector.field(3, shape=(m, n))` as DLPack `(m, n, 3)` with `lanes=1`, so the batch axis is `len(qd_object.shape) - 1` (here `1`), not `tensor.ndim - 1`.
Torch-present behaviour is unchanged and byte-identical; a call site may therefore use these helpers unconditionally.

# Test Strategy
No dedicated tests live here; entity behavior is exercised through the engine's scene integration tests and example scripts.
Priority coverage: description resolution → entity build round-trips, local↔global index mapping, get/set/control state consistency against solver state, and particle/FEM skinning correctness.
