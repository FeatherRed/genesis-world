# tests/particles — CONTEXT

## Intent
Particle-solver integration tests covering the three Genesis particle solvers: MPM (Material Point Method), PBD (Position-Based Dynamics), and SPH (Smoothed-Particle Hydrodynamics).
Each file validates solver-specific physics, entity/material APIs, determinism, and error handling end-to-end through `gs.Scene` builds and `scene.step()`.

## API Surface
- `test_mpm.py` (3 tests) — MPM `Elastic`/`Liquid` materials, `MeshSet` member poses and particle sampling, particle↔rigid-link attachment (`set_particle_constraints`), SVD vs no-SVD dispatch, and NaN-divergence halting.
- `test_pbd.py` (4 tests) — PBD `Elastic`/`Cloth` materials, tetrahedral meshing with `maxvolume`/`nobisect`, `get_mass`, cloth `fix_particles`/`release_particle`/`fix_particles_to_link`, and cloth↔rigid-link following.
- `test_sph.py` (6 tests) — SPH `Liquid` with `WCSPH` and `DFSPH` pressure solvers, initial density/pressure fields, simulation stability, particle-size invariance, hydrostatic column pressure, and determinism/env-independence.
- `__init__.py` — empty package marker.

## Constraints
- Every test is marked `@pytest.mark.required`; there are no `slow`, `benchmarks`, or `examples` tests in this directory.
- Uses shared fixtures from `../conftest.py`: `show_viewer`, `tol`, and `box_obj_path` (session-scoped generated cube OBJ); `backend`/precision/ndarray parametrization is inherited from conftest.
- Do not use a session-scoped material fixture: PBD materials are mutated in place when a scene is built (see comment in `test_pbd.py`), so `pbd_material` is function-scoped.
- Read `../conftest.py`, `../utils/assertions.py`, `../utils/assets.py` as siblings, but never write to them (escalate writes to the tests/ root owner).

## Design Decisions
- SPH tests always use the `"regular"` sampler: irregular samplers (pbs/random) create local density fluctuations at init that produce spurious pressure forces and divergence (documented in `test_sph.py` header).
- `get_sph_density_pressure` calls private kernels `_kernel_reorder_particles`/`_kernel_compute_rho` directly to read density/pressure without stepping, mirroring `substep_pre_coupling`; reordering is mandatory because `_kernel_compute_rho` reads `particles_reordered`.
- `test_halt_on_nan` pins `precision="32"` because the MPM divergence/freeze behavior under test is fp32-specific; other tests inherit the session precision (64-bit cpu / 32-bit gpu).
- `test_cloth_attach_fixed_point` pins `backend=[gs.gpu]`, so it runs only under GPU backends.
- Determinism is asserted by `scene.reset(state)` reproducing a continuous run bit-identically and by one environment's trajectory being independent of what another environment simulates.

## Known Issues
- Non-viscous MPM liquid can penetrate the ground plane by a sub-grid-cell amount; `test_perf_dispatch` tolerates up to 2 mm.
- A stiff MPM block (E=1e9) diverges: the failing environment is flagged/frozen before its state becomes non-finite, then the next step raises `GenesisException("Invalid MPM particle velocities")` on CPU (on GPU it may be the step after).
- SPH viscosity weights each pair by the neighbor's density alone, so forces between different-density particles do not cancel; the mean-velocity check carries a `1e-4` tolerance for this momentum drift.
- The SPH column-pressure test uses wide bounds (0.4×–1.5× `weight_per_area`) to absorb residual sloshing and floor wall pressure that relieves equation-of-state compression.

## Test Strategy
- All 13 tests are fast `required` tests that run in the default CI matrix (no `--backend`-specific opt-out needed, except the gpu-pinned PBD test).
- Solver-specific tolerances are passed via the shared `tol` fixture; physical checks use loose relative bounds where dissipation or wall forces are present.
- To add DFSPH coverage, extend the existing `pressure_solver` parametrize blocks rather than writing new scene builders (see `create_sph_scene`).

## See Also
- `../conftest.py` — shared fixtures, backend/precision/ndarray parametrization, markers.
- `../utils/assertions.py` — `assert_allclose`/`assert_equal` helpers.
- `../utils/assets.py` — HuggingFace asset download (not referenced by these tests; `box_obj_path` is a locally generated temp cube OBJ from conftest).
- `../../genesis/engine/solvers/{mpm,pbd,sph}_solver.py` — the solver implementations under test.
