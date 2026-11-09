# genesis/engine/materials

## Intent
- Defines the physical constitutive models and material parameters of the Genesis World engine.
- A material is "what" a body is made of; an entity is "how" it is discretized (mesh / tetrahedra / particles); a solver is "how" it is simulated.
- Materials are pydantic options objects (strict validation), not simulation state: the user constructs one, passes it to `scene.add_entity(material=...)`, and entities/solvers read it (mostly read-only) afterwards.

## API Surface
- `base.py` — `Material[EntityT](Options)`: root of the hierarchy; the only shared field is `use_visual_raycasting: StrictBool = False`.
- `kinematic.py` — `Kinematic[EntityT]`: visualization-only ghost material (rendered, no physics/collision).
- `rigid.py` — `Rigid(Kinematic["RigidEntity"])`: rigid-body params (rho, friction, torsional/rolling friction, SDF grid, gravity compensation, IPC coupling `coup_*` params).
- `hybrid.py` — `Hybrid(Material["HybridEntity"])`: composes required `material_rigid` + `material_soft` with soft/rigid coupling params; consumed by `HybridEntity`.
- `FEM/`, `MPM/`, `PBD/`, `SPH/` — per-solver material families (see routing table).
- `__init__.py` imports the four subpackages and re-exports `Kinematic`, `Hybrid`, `Rigid`; users access concrete materials as `gs.materials.MPM.Elastic`, `gs.materials.FEM.Muscle`, etc.

## Material -> entity -> solver dispatch
- Each solver class declares `material_cls` (the material base class it simulates); `Simulator._add_entity` routes an entity to the solver whose `material_cls` is the most-derived superclass of the material instance's type.
- `KinematicSolver.material_cls = Kinematic`; `RigidSolver.material_cls = Rigid` (Rigid derives from Kinematic, and the most-derived class wins); `MPMSolver.material_cls = MPM.Base`; `FEMSolver.material_cls = FEM.Base`; `PBDSolver.material_cls = PBD.Base`; `SPHSolver.material_cls = SPH.Base`.
- Material classes are generic in the entity they target (`Material["FEMEntity"]`, `Material["MPMEntity"]`, ...); the type parameter is used for typing/documentation, not runtime dispatch.

## Design Decisions
- Material = constitutive model, entity = discretization, solver = integrator. The same notion of "elastic" means different physics per solver, so each family (FEM/MPM/PBD/SPH) owns its own `Base` and concrete classes rather than sharing one hierarchy.
- All materials inherit `genesis.options.Options`, a pydantic `BaseModel` with `strict=True`, `extra="forbid"`, `validate_default=True` and custom `__setattr__`/`__init__` validation, so unknown or wrong-typed parameters fail at construction or assignment.
- FEM and MPM materials are also `@qd.data_oriented` (quadrants): their constitutive laws are compiled data-oriented kernels, which makes them differentiable (torch autograd through quadrants) and portable across CPU/GPU.
- FEM/MPM subclasses install "dispatch fields" (bound kernel methods such as `update_stress`, `compute_energy_gradient`, `hessian_product`) in `model_post_init`; the solver calls the field, never the concrete class, so it can compile a static per-material loop over `mat_idx`. These fields are internal (`exclude=True`, `repr=False`) and must not be set by users.
- PBD/SPH materials are plain (non-quadrants) data classes; the solver reads their scalar parameters directly.

## Constraints
- `Material` and every family `Base` must not be instantiated directly (abstract by convention, enforced only by convention/documentation).
- Do not add per-env simulation state to materials; materials are shared across all batched environments. The one known exception is `MPM.Base._default_Jp` (private, acknowledged tech debt pending refactor).
- New constitutive models belong in the matching family subpackage and must fill that family's dispatch-field contract (FEM: `build`/`pre_compute`/`pre_compute_grad`/`update_stress`/`hessian_product`/`compute_energy_gradient`/`compute_energy`; MPM: `update_F_S_Jp`/`update_stress`/`needs_svd`).

## Gotchas
- Muscle actuation is supported only by `FEM.Muscle` and `MPM.Muscle`; entity methods `set_actuation`/`set_muscle_*` are guarded by `@assert_muscle` and raise for other materials.
- `FEM.Elastic(model="stable_neohookean")` has no Hessian (implicit FEM solver unsupported) and raises in `hessian_product`/`compute_energy_gradient`.
- Numeric precision follows `gs.qd_float`/`gs.tc_float` (fp32 or fp64 by backend/precision setting); history shows fp64 inaccuracies in deformable materials were fixed.
- `Hybrid` requires both `material_rigid` and `material_soft` (no defaults).

## Test Strategy
- No unit tests live under this directory; materials are exercised indirectly through entity+solver integration and example scenes elsewhere in the repo.

## See Also
- `genesis/engine/entities/` — entities read/validate materials (`assert_muscle`, `init_sampler`, `material.sampler`/`rho`/`idx`).
- `genesis/engine/solvers/` — solvers declare `material_cls` and consume the dispatch fields.
- `genesis/options/options.py` — the `Options` pydantic base.
- `genesis/utils/` — `geom` (e.g. `gu.qd_svd`) and `typing` numeric types used by constitutive kernels.

## Routing Table
- `FEM/` — finite-element constitutive models (`Base`, `Elastic`, `Muscle`, `Cloth`).
- `MPM/` — material-point-method models (`Base`, `Elastic`, `ElastoPlastic`, `Liquid`, `Muscle`, `Sand`, `Snow`).
- `PBD/` — position-based-dynamics materials (`Base`, `Elastic`, `Cloth`, `Liquid`, `Particle`).
- `SPH/` — smoothed-particle-hydrodynamics materials (`Base`, `Liquid`).
- `../entities/` — sibling (read-only): entities that consume materials.
- `../solvers/` — sibling (read-only): solvers that consume materials.
