# genesis/engine/materials/PBD

## Intent
- Materials for the PBD (position-based dynamics) solver: cloth, volumetric elastic, liquid, and free particles.
- Unlike FEM/MPM, PBD materials are plain data (no quadrants kernels); the PBD solver reads their scalar compliance/relaxation parameters directly.

## API Surface
- `base.py` — `Base[EntityT](Material[EntityT])`: empty abstract base (no shared fields).
- `elastic.py` — `Elastic(Base["PBD3DEntity"])`: `rho`, `static_friction`, `kinetic_friction`, `stretch_compliance`, `bending_compliance`, `volume_compliance`, and their `*_relaxation` factors.
- `cloth.py` — `Cloth(Base["PBD2DEntity"])`: `rho` (areal, kg/m²), friction, stretch/bending compliance + relaxation, `air_resistance`.
- `liquid.py` — `Liquid(Base["PBDParticleEntity"])`: `rho`, `sampler`, `density_relaxation`, `viscosity_relaxation`.
- `particle.py` — `Particle(Base["PBDFreeParticleEntity"])`: `rho`, `sampler`; free particles with no inter-particle interaction.

## Design Decisions
- PBD parameterizes stiffness as compliance (inverse stiffness) rather than Young's modulus/Poisson ratio; `*_relaxation` factors scale how much of each constraint is applied per solver iteration.
- `sampler` is a pydantic `Literal["pbs", "random", "regular"]` per class (no regex needed); the shared `ParticleEntity.init_sampler` resolves plain `pbs` → `pbs-32`.
- `Cloth.rho` is areal density (kg/m²) so entity mass is `rho * surface_area`; other PBD materials use volumetric density (kg/m³).

## Constraints
- Do not instantiate `Base` directly.
- `Particle` models point particles with no interaction, hosted under PBD only to reuse the particle-solver infrastructure.

## Gotchas
- `Particle` produces entities affected only by external forces (particle animations), not a fluid.
- `Cloth`/`Elastic`/`Liquid`/`Particle` correspond to the `PBDSolver.MATERIAL` enum {CLOTH, ELASTIC, LIQUID, PARTICLE}, which the solver uses to dispatch its kernels.

## See Also
- `genesis/engine/solvers/pbd_solver.py` — `material_cls = PBD.Base`; reads compliance/relaxation parameters.
- `genesis/engine/entities/pbd_entity.py` — PBD2D/3D/Particle/Tet entity variants.
