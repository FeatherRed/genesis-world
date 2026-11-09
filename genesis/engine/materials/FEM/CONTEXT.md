# genesis/engine/materials/FEM

## Intent
- Constitutive models for the FEM solver: deformable solids on tetrahedral elements, plus IPC thin-shell cloth.
- Consumed by `FEMSolver` (explicit and implicit variants); `Cloth` is consumed by the IPC coupler.

## API Surface
- `base.py` — `Base(Material["FEMEntity"])` (abstract): `E`, `nu`, `rho`, `hydroelastic_modulus`, `friction_mu`, `contact_resistance`; derives Lamé `mu`/`lam` from E/nu; defines the dispatch fields `build`, `pre_compute`, `pre_compute_grad`, `update_stress`, `hessian_product`, `compute_energy_gradient`, `compute_energy` (no-op/raising defaults); `idx` assigned by FEMSolver.
- `elastic.py` — `Elastic(Base)` plus module-level `partialJpartialF(F)`; constitutive `model` ∈ `linear` / `stable_neohookean` / `linear_corotated`.
- `muscle.py` — `Muscle(Elastic)`: adds `n_groups` and active-stress actuation on top of an Elastic model.
- `cloth.py` — `Cloth(Base)`: IPC thin-shell params (`E`, `nu`, `rho`, `thickness`, `bending_stiffness`, `model` placeholder).

## Design Decisions
- Energy-density functions are quadrants (`@qd.func`) kernels parameterized by `(mu, lam, J, F, actu, m_dir)`; the solver differentiates them via torch/quadrants for differentiable simulation.
- Each constitutive model overrides the dispatch fields in `model_post_init`; the solver calls through those fields and loops statically over `mat_idx` (see `FEMSolver._func_element_energy_gradient`, `compute_elastic_product`).
- `mu`/`lam` are derived once from E/nu and stored (excluded) rather than recomputed per element.
- `partialJpartialF` returns ∂J/∂F as the cross products of F's columns; used in Neo-Hookean stress `lam*(J-alpha)*dJdF`.
- Linear-corotated caches the polar rotation `R = U Vᵀ` from SVD(F) in `pre_compute` (private buffer `_R`), then `hessian_product`/`compute_energy_gradient`/`compute_energy` use that rotation at the substep start.
- Muscle wraps the chosen Elastic model instead of reimplementing it: it saves the base dispatch functions, installs actuation-augmented wrappers, and adds active stress `stiffness * actu * fiber_dir ⊗ m_dir` (energy `stiffness * actu * fiber_dir · (F m_dir)`).

## Constraints
- `nu` is validated to `-1 < nu < 0.5`; volumetric locking grows near 0.5 (Cloth deliberately uses 0.49).
- `Cloth` works only with the IPC coupler on GPU and only accepts surface-mesh morphs; its "elements" are surface triangles (`FEMEntity` branches on `FEM.Cloth`).
- Do not instantiate `Base` directly.

## Gotchas
- `Elastic(model="stable_neohookean")` has no Hessian: `hessian_product` and `compute_energy_gradient` raise, so it is unusable with the implicit FEM solver (explicit-only).
- Deprecated spelling `"stable_neohooken"` is auto-corrected to `"stable_neohookean"` with a warning.
- `linear_corotated.update_stress` raises (not implemented); the implicit path relies on `hessian_product`/`compute_energy_gradient` instead.
- The implicit solver's `pre_compute_grad` for linear-corotated hand-implements the adjoint of the SVD rotation; it turns non-finite for inverted elements, halting the backward pass.
- Muscle's `_fiber_length_eps` (1e-12) guards the fiber-direction normalization.

## See Also
- `genesis/engine/solvers/fem_solver.py` — consumes the dispatch fields and `material.idx`.
- `genesis/engine/entities/fem_entity.py` — `assert_muscle`, Cloth surface-triangle branch, hydroelastic pressure field.
- `genesis/utils/geom.py` — `gu.qd_svd` used for the polar decomposition.
