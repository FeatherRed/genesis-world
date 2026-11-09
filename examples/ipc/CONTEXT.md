## Intent

`examples/ipc/` demonstrates the IPC (Incremental Potential Contact) coupler, which provides unified contact handling across cloth, FEM deformables, and rigid/articulated bodies.
The scripts are runnable entry points (`if __name__ == "__main__"`), each building a `gs.Scene` on the CPU backend with `coupler_options=gs.options.IPCCouplerOptions(...)`.

## API Surface

- `ipc_objects_falling.py` — cloth (`gs.materials.FEM.Cloth`, NeoHookeanShell) falls under gravity onto a plane, a rigid box, and a soft FEM ball; exercises soft-rigid and soft-soft IPC contact.
- `ipc_robot_grasp_cube.py` — Franka Panda (MJCF) grasps and lifts a deformable `stable_neohookean` cube; accepts `--no-ipc` (rigid-cube fallback) and `--coup-type {two_way_soft_constraint,external_articulation}`.
- `ipc_robot_cloth_teleop.py` — interactive keyboard teleop of a Franka manipulating two cloths over 16 fixed `ipc_only` cubes; registers viewer keybinds and runs with `show_viewer=True` (no `-v` flag).
- `ipc_momentum.py` — zero-gravity linear-momentum conservation check between a rigid cube and an FEM sphere, reading FEM state via the libuipc `SceneVisitor` API and plotting with matplotlib.
- `README.md` — documents prerequisites, run commands, and verification steps.

## Constraints

- All scripts parse argparse; `-v`/`--vis` toggles the viewer except in `ipc_robot_cloth_teleop.py`, which always shows the viewer.
- Every script inits with `gs.init(backend=gs.cpu)`; none use the GPU backend.
- Each script shortens its simulation loop (and skips viewer/plot blocking) when the `PYTEST_VERSION` env var is set.

## Dependencies

- `pyuipc` (provides the `uipc` module) — NOT listed in pyproject; install separately via `pip install pyuipc`.
- Genesis must be installed in editable/dev mode (`pip install -e .`).
- Platform is limited to Linux/Windows x86 CPU with an Nvidia GPU.
- Cloth scripts download the `IPC/grid20x20.obj` asset from Hugging Face via `huggingface_hub` (needs network).
- `ipc_momentum.py` additionally needs `matplotlib` and `numpy`.

## Gotchas

- The IPC coupler is configured through `coupler_options=gs.options.IPCCouplerOptions(...)` on `gs.Scene`; notable options include `two_way_coupling`, `contact_d_hat`, `constraint_strength_translation/rotation`, `enable_rigid_ground_contact`, and `enable_rigid_rigid_contact`.
- Per-entity coupling is set on `gs.materials.Rigid` via `coup_type`, with `coup_friction` and `coup_links` as additional kwargs.
- `coup_type` accepts `two_way_soft_constraint`, `external_articulation`, and `ipc_only`.
- `two_way_soft_constraint` requires the robot material to specify `coup_links` (e.g. `("left_finger", "right_finger")`).
- `external_articulation` does not support setting the initial configuration with `set_qpos`.
- Two-way coupling (`two_way_coupling=True`) is what makes Genesis rigid-body motion affect IPC objects and vice versa; `ipc_only` bodies are driven solely by the IPC solver.
- Dense meshes can violate IPC thickness; the README warns the built-in cloth mesh is too dense, so these examples use a coarse `grid20x20.obj`.
- `ipc_robot_cloth_teleop.py` blocks until the viewer window closes (ESC quits); it is not headless-safe.

## Test Strategy

- Covered by `tests/test_examples.py` under the pytest `examples` marker.
- The `ipc` directory maps to optional module `uipc` in `EXAMPLE_DEPENDENCIES`, so every script in this directory is skipped when pyuipc is not installed.
- Scripts detect `PYTEST_VERSION` to run shortened horizons and avoid viewer/matplotlib blocking during CI.

## See Also

- `genesis/options/solvers.py` — `IPCCouplerOptions` definition and full option list with defaults.
- `tests/test_examples.py` — example discovery and optional-dependency skip logic.
