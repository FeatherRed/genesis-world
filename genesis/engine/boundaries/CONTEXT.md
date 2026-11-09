# boundaries

## Intent
Spatial domain boundary for the particle/MPM solvers: keeps simulated particles inside an axis-aligned box and reflects (or zeroes) their outward velocity at the faces.
Holds the single `CubeBoundary` class, the only boundary type in the engine.

## API Surface
- `boundaries.py` — defines `CubeBoundary`, a `@qd.data_oriented` class.
- `__init__.py` — re-exports everything from `boundaries.py` via `from .boundaries import *` (see Known Issues).

`CubeBoundary(lower, upper, restitution=0.0)`:
- `lower` / `upper` — box corners; stored twice, as numpy arrays (`self.lower` / `self.upper`, dtype `gs.np_float`) for host-side use and as `qd.Vector`s (`self.lower_qd` / `self.upper_qd`, dtype `gs.qd_float`) for kernel-side use.
- `restitution` — velocity reflection factor at the box faces; default `0.0` (inelastic: the outward velocity component is zeroed).
- `impose_pos_vel(pos, vel)` — `@qd.func`; per axis, if `pos[i] >= upper[i]` and `vel[i] >= 0` (or `pos[i] <= lower[i]` and `vel[i] <= 0`), sets `vel[i] *= -restitution`; then clamps `pos` into `[lower, upper]` and returns `(pos, vel)`.
- `impose_pos(pos)` — `@qd.func`; clamps `pos` into `[lower, upper]` only; currently unused by the engine.
- `is_inside(pos)` — host-side (numpy) strict-interior test `all(pos < upper) and all(pos > lower)`; used to validate particle positions at entity-add/emit time.
- `__repr__` — prints `lower`, `upper`, `restitution`.

Because the class is `@qd.data_oriented`, its `@qd.func` methods compile into the Quadrants kernels of any solver that holds the instance as `self.boundary`.

## Consumers (siblings — read-only, escalate writes to parent)
- `../solvers/sph_solver.py` — `setup_boundary` builds `CubeBoundary(lower, upper)`; `_kernel_advect_position` calls `impose_pos_vel` on advected positions/velocities.
- `../solvers/pbd_solver.py` — `setup_boundary` builds `CubeBoundary(lower, upper)`; `_kernel_solve_boundary_collision` (from `substep_post_coupling`) calls `impose_pos_vel` on every particle.
- `../solvers/mpm_solver.py` — `setup_boundary` builds `CubeBoundary` inset by `boundary_padding = 3 * dx`; `g2p` calls `impose_pos_vel` as a safety clamp so a diverging sim cannot index an illegal grid cell.
- `../couplers/legacy_coupler.py` — calls `mpm_solver.boundary.impose_pos_vel` on grid velocities during MPM coupling.
- `../entities/particle_entity.py` and `../entities/emitter.py` — call `is_inside` to reject particles added/emitted outside the solver boundary.

## Design Decisions
The bounds are stored in both numpy and `qd.Vector` form because Quadrants kernels read solver attributes, while host-side asserts and `is_inside` use numpy.
`impose_pos_vel` only negates the outward-moving velocity component and scales it by `restitution`, so with the default `0.0` it zeroes that component rather than inverting it.
MPM insets its boundary by `3 * dx` instead of clamping at the true box, leaving stencil room so boundary enforcement stays clear of out-of-range cell access.
`impose_pos` is retained in the API even though nothing calls it yet.

## Known Issues / Gotchas
Naming collision: `BoundaryItems` and "boundary checkpoints" in `../solvers/base_solver.py` refer to the temporal boundary between two steps (for gradient replay / reset), not to `CubeBoundary`; the two are unrelated.
`boundaries.py` defines no `__all__`, and `__init__.py` uses `from .boundaries import *`, so `genesis.engine.boundaries` leaks `np`, `qd`, `gs`, `brief`, and every public name `genesis.utils.misc` exports.
`sph_solver.setup_boundary` leaves `restitution=0.5` commented out, so all three solvers currently run with the default `restitution=0.0`.
`is_inside` is a strict (exclusive) test while the kernels clamp inclusively to the faces, so a particle sitting exactly on a face passes `impose_*` but fails `is_inside`.

## Dependencies
- `numpy`, `quadrants` (`qd`), `genesis` (`gs.np_float`, `gs.qd_float`), `genesis.utils.misc`, `genesis.utils.repr.brief`.

## Test Strategy
No dedicated tests for this module; it is exercised indirectly through SPH/PBD/MPM solver tests and the entity-add/emitter `is_inside` validation.

## See Also
- `../solvers/base_solver.py` — `Solver.data` and the boundary checkpoint save/load machinery (temporal boundary; same word, different concept).
