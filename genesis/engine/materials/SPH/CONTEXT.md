# genesis/engine/materials/SPH

## Intent
- Materials for the SPH (smoothed particle hydrodynamics) fluid solver.
- Plain data classes (no quadrants kernels); the SPH solver reads the scalar parameters.

## API Surface
- `base.py` — `Base(Material["SPHEntity"])` (abstract): `sampler` (`Literal["pbs", "random", "regular"]`, default `pbs`).
- `liquid.py` — `Liquid(Base)`: `rho` (rest density), `stiffness`, `exponent`, `mu` (viscosity), `gamma` (surface tension); `sampler` default `regular`.

## Design Decisions
- SPH pressure follows a Tait-like equation of state `stiffness * ((rho/rho_rest)^exponent - 1)`; `stiffness` and `exponent` control how pressure grows with density, `mu` is dynamic viscosity, `gamma` surface tension.
- `sampler` defaults to `regular` (grid sampling) for numerical stability with SPH, overriding the shared `pbs` default.

## Constraints
- Do not instantiate `Base` directly.

## Gotchas
- `SPHEntity._add_particles_to_solver` forwards `rho`/`stiffness`/`exponent`/`mu`/`gamma` positionally to the solver; keep field order stable when editing `Liquid`.

## See Also
- `genesis/engine/solvers/sph_solver.py` — `material_cls = SPH.Base`; receives the parameters in `add_particles`.
- `genesis/engine/entities/sph_entity.py` — forwards material parameters to the solver.
