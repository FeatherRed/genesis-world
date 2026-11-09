# Intent

Demonstrates Genesis' deformable/solid and liquid solvers through four self-contained scripts.
Covers differentiable MPM, PBD elastic solids, FEM soft bodies with vertex constraints, and PBD liquid.
Each script is also a pytest example run by `tests/test_examples.py` (marker `examples`); horizons are cut short when `PYTEST_VERSION` is set.

# API Surface

- `differentiable_push.py` — Differentiable MPM push: a rigid stirrer (Rigid, `coup_friction=8.0`) pushes two `gs.materials.MPM.Elastic` objects with `SimOptions(requires_grad=True, dt=2e-3, substeps=10)`; torch autograd flows gradients back to input tensors.
- `elastic_dragon.py` — Drops a dragon mesh with `gs.materials.PBD.Elastic()`; runs on the PBD solver (not FEM despite the name).
- `fem_hard_and_soft_constraint.py` — FEM elastic sphere and box (`gs.materials.FEM.Elastic(model="linear_corotated")`) driven by `FEMOptions(use_implicit_solver=True, enable_vertex_constraints=True)`; compares a hard vertex constraint (cube) against a soft one (`is_soft_constraint=True, stiffness=1e4`, blob) while recording video.
- `pbd_liquid.py` — PBD liquid block (`gs.materials.PBD.Liquid`) inside a bounded domain via `PBDOptions(max_density_solver_iterations=10, max_viscosity_solver_iterations=1, lower_bound, upper_bound)`.

# Constraints

- Follows the examples CLI convention: `-v/--vis` (viewer off by default) and, where applicable, `-g/--gpu` (CPU default); see `.github/contributing/EXAMPLES.md`.
- No example runs at the top of `examples/`; all four live in this directory and are discovered recursively by `tests/test_examples.py`.
- Output files (e.g. the FEM script's mp4) go under `out/`, resolved against the launch directory.

# Dependencies

- `torch` — required by `differentiable_push.py` (autograd) and `fem_hard_and_soft_constraint.py` (device/dtype handling).
- `numpy`, `tqdm` — required by `fem_hard_and_soft_constraint.py`.
- No optional third-party solvers; these scripts use only core Genesis CPU backends (torch is installed on the CPU examples CI).

# Gotchas

- `differentiable_push.py` needs BOTH `SimOptions(requires_grad=True)` AND `requires_grad=True` on the input tensors (`set_pos`, `set_dofs_velocity`, `set_position`, `set_velocity`) for gradients to flow.
- `differentiable_push.py` builds with `n_envs=2` and runs two forward/backward passes, each restarting from `scene.reset()`; it computes loss from `scene.get_state().solvers_state[...]` at an intermediate step and from `obj1.get_state()` at the end.
- `elastic_dragon.py` uses `gs.materials.PBD.Elastic` (PBD solver), not FEM — its name refers to the elastic material, not the solver family.
- The FEM script must set `use_implicit_solver=True`: the `linear_corotated` model supplies only energy derivatives the implicit solver needs, so the explicit solver cannot integrate it.
- The FEM script defaults to `dt=1e-3, substeps=1` and caps recording fps at `min(100, 1/dt)` because tiny steps would otherwise produce a far denser video than needed.
- `pbd_liquid.py` builds with `n_envs=0` (no parallel environments).
- CPU examples that use MPM or PBD liquid set `precision="32"` (e.g. `differentiable_push.py`, `pbd_liquid.py`); `elastic_dragon.py` and the FEM script leave precision at default.
- `elastic_dragon.py` opts into GPU via `-g/--gpu` (`gs.gpu`); the other three hardcode `gs.cpu`.

# See Also

- `tests/test_examples.py` — runs every `examples/**/*.py` as a subprocess with the `examples` marker.
- `.github/contributing/EXAMPLES.md` — CLI/backend/output conventions all examples follow.
- `../rigid/` — rigid-body examples sharing the `gs.init` → `Scene` → `add_entity` → `build` → `step` pattern (sibling — read-only).
