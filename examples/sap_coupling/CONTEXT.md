# Intent

Reference scripts demonstrating the SAP (Semi-Analytic Primal) coupling solver — the built-in implicit contact coupler (Drake-style SAP, `genesis/options/solvers.py::SAPCouplerOptions`).
Each script couples rigid and FEM elastic bodies through `gs.Scene(coupler_options=gs.options.SAPCouplerOptions(...))`.
The coupler is a core Genesis feature; it needs no extra pip dependency.

# API Surface

- `fem_sphere_and_cube.py` — Drops a corotated FEM sphere onto a FEM cube mesh on a rigid plane; FEM↔FEM contact via SAP.
- `fem_fixed_constraint.py` — Animates a vertex constraint on a FEM cube (`FEMOptions(enable_vertex_constraints=True)` + `set_vertex_constraints`) alongside the SAP coupler.
- `franka_grasp_fem_sphere.py` — Franka Panda gripper grasps and lifts a FEM elastic sphere; rigid↔FEM SAP coupling.
- `franka_grasp_rigid_cube.py` — Franka Panda gripper grasps and lifts a rigid box; rigid-only scene still configured with SAPCouplerOptions.

All four follow the standard example CLI (`-g/--gpu` opt-in, CPU default; `-v/--vis` opt-in).

# Constraints

- All scripts run `gs.init(..., precision="64")` (double); the implicit FEM solver and SAP coupler are exercised in double precision.
- Every script is a linear `main()` with `if __name__ == "__main__": main()` at the bottom, per `../../.github/contributing/EXAMPLES.md`.
- These run as pytest examples via `tests/test_examples.py` (`pytest.mark.examples`), discovered automatically by the `rglob("*.py")` scan — a new script here is picked up with no registration.

# Dependencies

- Core `genesis` only; no optional package required (unlike `../ipc/`, which needs `pyuipc`/`uipc`).
- `fem_sphere_and_cube.py` and `fem_fixed_constraint.py` download `cube8.obj` from the `Genesis-Intelligence/assets` Hugging Face dataset via `snapshot_download` (first run needs network/cache; CI sets `HF_HUB_DOWNLOAD_TIMEOUT`).
- `franka_grasp_*` load `xml/franka_emika_panda/panda.xml` through `gs.morphs.MJCF` (bundled asset, no network).

# Design Decisions

- `precision="64"` is used instead of the default 32-bit to reduce solver drift for the tight SAP convergence targets.
- The franka grasp scripts tighten the SAP solver (`sap_convergence_atol/rtol`, `pcg_threshold`, `linesearch_ftol` all `1e-10`) well below the defaults (`1e-6`/`1e-5`/`1e-6`/`1e-6`), trading speed for accurate finger-object contact.
- `RigidOptions(enable_self_collision=False)` on the franka keeps the gripper fingers from self-colliding during grasp.

# Gotchas

- `precision="64"` is unsupported on the Apple Metal GPU backend (see `tests/conftest.py::SKIP_METAL_64BIT`); use CPU or a CUDA backend.
- `franka_grasp_rigid_cube.py` does NOT apply the `PYTEST_VERSION` horizon shortcut — it always runs the full 15+10+40 step loop (65 steps); the other three cut to ~2 steps when `PYTEST_VERSION` is in `os.environ`.
- The in-repo pytest harness (`tests/test_examples.py`) and `.github/workflows/examples.yml` do not currently set `PYTEST_VERSION`; the shortcut convention is documented in `../../.github/contributing/EXAMPLES.md`.
- The two mesh-based scripts pin an exact HF dataset revision (`4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc`) for `cube8.obj`.

# See Also

- `../coupling/` — default MPM/SPH/PBD/FEM coupling examples (`LegacyCouplerOptions`).
- `../ipc/` — external pyuipc-based IPC coupler examples (`IPCCouplerOptions`).
- `../../genesis/options/solvers.py` — `SAPCouplerOptions` definition and defaults.
- `../../tests/test_examples.py` — how these scripts are run as examples tests.
