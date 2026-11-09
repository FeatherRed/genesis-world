# tests/coupling

## Intent
Integration tests for multi-physics coupling: rigid↔particle (MPM/SPH/PBD), rigid↔FEM, and the hybrid (soft+rigid) entity material.
Covers both the legacy particle-rigid coupling path and the newer CPIC/SAP coupling solvers.

## API Surface
- `test_hybrid.py` — hybrid particle-rigid coupling, SAP coupler, and multi-solver coexistence in one scene.
- `test_sph_rigid.py` — SPH fluid↔rigid coupling: buoyancy and rigid-body hibernation wake-up.
- `__init__.py` — empty.

## Test Inventory

### test_hybrid.py
- `test_rigid_mpm_muscle` (`slow` ~300s, `required`) — two-link URDF arm with `gs.materials.Hybrid` (Rigid with gravity compensation + `MPM.Muscle`), `enable_CPIC=True`; driven sinusoidally, a rigid ball is pushed along +y via the CPIC coupling wrench.
- `test_rigid_mpm_legacy_coupling` (`required`, `substeps` ∈ [1, 10]) — legacy (non-CPIC) coupling: an `MPM.Liquid` "sand" box pushes a rigid box; asserts substep-invariance at fixed `substep_dt` and that `reset()` clears the coupling wrench.
- `test_mesh_mpm_build` (`required`) — smoke test that a `Hybrid` material on a `Mesh` morph (`duck.obj`) builds.
- `test_deformable_parallel` (`required`, `backend` = [gs.gpu], `debug(False)`) — one scene with rigid plane (`needs_coup=True`), PBD cloth, SPH liquid, MPM elastic, and FEM elastic across 2 envs; asserts all particles fall and settle above the ground.
- `test_fem_outside_mpm_grid` (`required`) — FEM body outside the MPM grid free-falls unaffected; tolerance/CFL reasoning in comments.
- `test_fluid_emitter` — parametrized over emitter material types (SPH/PBD/MPM liquid, MPM sand/snow/elastic) × `n_envs`; emitter interacts with a fixed wheel URDF.
- `test_sap_rigid_rigid_hydroelastic_contact` (`required`, `precision=["64"]`, `batch_links_info` ∈ [False, True]) — SAP coupler rigid↔rigid hydroelastic contact; validates the "single fixed plane facing up" ground constraint and multi-env resting equilibrium.
- `test_sap_fem_vs_robot` (`required`, `precision=["64"]`) — SAP coupler rigid↔FEM contact with `enable_fem_self_tet_contact=False`; MJCF `cross.xml` robot resting on a FEM sphere.

### test_sph_rigid.py
- `test_rigid_flotation_follows_density_ratio` — parametrized (`n_envs`, `pressure_solver`) with WCSPH (`required`), DFSPH, and 2-env `slow` (~150s/~300s) variants; low/high-density rigid spheres and a tilted raft in SPH liquid verify Archimedes buoyancy and buoyancy-torque rocking via the coupling wrench.
- `test_coupling_force_wakes_hibernated_link` (`n_envs` ∈ [0, 2(required)]) — `use_hibernation=True`; teleporting SPH liquid over a hibernated ball must wake it through coupling forces alone (a dry resting bystander keeps the island partition engaged).

## Constraints
- `required` tests run in the default `-m "not (benchmarks or examples)"` selection; `slow` tests are ~150–300 s each.
- `SAP` tests pin `precision=["64"]`; on GPU the default precision is 32-bit.
- Do not re-investigate shared fixtures/markers — see `tests/conftest.py` (already documented at the `tests/` root).

## Design Decisions
- The legacy-coupling test varies `substeps` while holding `substep_dt` fixed to prove the coupling result depends only on substep time step, not on step count.
- SPH flotation is validated through a density-ratio argument (Archimedes) rather than exact trajectories, so tolerances stay backend-independent.
- The hibernation test includes a dry resting bystander because the island partition only engages with ≥2 free bodies.

## Known Issues
- `test_sap_fem_vs_robot`: FIXME in code — the SAP solve stops before converging on resting contacts, so the FEM sphere sinks/oscillates; the bound `atol=2e-3` covers that error.
- `test_fem_outside_mpm_grid`: FIXME in code — build should raise when the explicit time step exceeds the `h/c` CFL bound; the test instead relies on tight `tol` from the `tol` fixture.

## Test Strategy
- Run a single test: `pytest tests/coupling/test_sph_rigid.py::test_coupling_force_wakes_hibernated_link -m required`.
- `slow` tests are intended for CI/nightly, not local iteration.
- `test_deformable_parallel` forces `backend=[gs.gpu]` and disables debug for speed.

## See Also
- `../tests/conftest.py` — shared fixtures/markers/backend parametrization (read-only).
- `../tests/utils/assertions.py` — `assert_allclose`.
- `../tests/utils/assets.py` — `get_hf_dataset` for MJCF assets.
- `../tests/rigid/`, `../tests/particles/`, `../tests/deformable/`, `../tests/ipc/` — sibling solver test suites (read-only).
