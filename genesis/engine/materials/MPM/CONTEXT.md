# genesis/engine/materials/MPM

## Intent
- Constitutive models for the MPM (material point method) solver: elastic solids, elasto-plastic solids, liquid, muscle, sand, snow.

## API Surface
- `base.py` — `Base(Material["MPMEntity"])` (abstract): `E`, `nu`, `rho`, `lam`, `mu`, `sampler`; resolves Lamé params; dispatch fields `update_F_S_Jp`, `update_stress`, `needs_svd`; private `_default_Jp`; `idx` assigned by MPMSolver.
  - module-level `_validate_sampler` + `SamplerType` accept `pbs`, `pbs-<res>`, `random`, `regular` (regex-validated).
- `elastic.py` — `Elastic(Base)`: `model` ∈ `corotation` / `neohooken`.
- `elasto_plastic.py` — `ElastoPlastic(Base)`: `yield_lower`/`yield_higher` or von-Mises yield.
- `liquid.py` — `Liquid(Base)`: `viscous` flag.
- `muscle.py` — `Muscle(Elastic)`: `n_groups` plus active-stress actuation.
- `sand.py` — `Sand(Base)`: `friction_angle` → derived `alpha`, pressure-dependent plasticity projection.
- `snow.py` — `Snow(ElastoPlastic)`: compression hardening, no von-Mises.

## Design Decisions
- MPM stress is computed from `F_tmp` (deformation gradient before the material update) and the SVD `(U, S, V)` of F_tmp: the solver calls `update_F_S_Jp` (plastic flow / volume normalization, returning `F_new, S_new, Jp_new`) then `update_stress(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)`.
- `needs_svd` toggles the solver's forward SVD kernel: corotation elastic, viscous liquid, elasto-plastic, sand, and snow read U/V/S; neohooken elastic and non-viscous liquid do not, so SVD is skipped for speed.
- `_default_Jp` is the initial plastic-volume-change per material (1.0 default; 0.0 for Sand).
- Plasticity lives in `update_F_S_Jp` (it modifies the stored F via singular-value clamping), not in `update_stress`.
- Muscle augments the chosen Elastic stress with active stress `stiffness * actu * F_tmp @ (m_dir m_dirᵀ) @ F_tmpᵀ`.

## Constraints
- `nu` is validated to `-1 < nu < 0.5`.
- Sampler strings are strictly validated by regex before pydantic accepts them.
- Do not instantiate `Base` directly.

## Gotchas
- Elasto-plastic clamps `S = qd.max(S, 0.05)` before taking logs to prevent NaN under the von-Mises branch.
- `Snow` raises if `use_von_mises=True` (validator forces it False); its hardening factor `h = exp(10*(1-Jp))` stiffens the material under compression.
- `Sand` derives `alpha = sqrt(2/3)*2*sin(phi)/(3-sin(phi))` from `friction_angle` (degrees) and sets `_default_Jp = 0.0`.
- `Liquid` sets `mu = 0` when `viscous=False` (only the pressure term `lam*J*(J-1)` remains) and sets `needs_svd = viscous`.
- `Muscle` overrides Elastic defaults to `model="neohooken"` and `E=1e6` (Elastic itself defaults to `corotation`, `E=3e5`).
- `_default_Jp` is solver state kept on the material; source comments flag it as tech debt pending a larger refactor.

## See Also
- `genesis/engine/solvers/mpm_solver.py` — consumes `update_F_S_Jp`/`update_stress`/`needs_svd` and `material.idx`.
- `genesis/engine/entities/mpm_entity.py` — `assert_muscle`, disables skinning for Liquid/Sand/Snow, forwards `material.idx`/`rho`/`_default_Jp`.
- `genesis/engine/entities/particle_entity.py` — `init_sampler` resolves plain `pbs` → `pbs-32` and validates the sampler string.
