# Intent

Onboarding progression for Genesis: 18 self-contained scripts that go from the minimal `hello_genesis.py` to parallel IK, robot control, soft-body solvers, and multi-physics hybrid entities.
Each script is a linear `main()` (or top-level) using the standard `import genesis as gs` -> `gs.init(backend=...)` -> `gs.Scene(...)` -> `scene.add_entity(...)` -> `scene.build()` -> `scene.step()` loop.
`hello_genesis.py` is the canonical first example — read it before anything else.

# API Surface

Core loop and scene basics:
- `hello_genesis.py` — Minimal hello-world: a Plane plus a Franka MJCF (`xml/franka_emika_panda/panda.xml`), 1000 rigid steps, CPU, headless (no viewer). The canonical first example.
- `entity_name.py` — Entity naming system: auto names (`{morph_type}_{uid}`), custom names, `scene.get_entity(name=...)` / `(uid=...)` lookup, and `scene.entity_names`; loads a go2 URDF; `-v/--vis` flag.
- `visualization.py` — Camera rendering and recording: `scene.add_camera`, `cam.render(rgb/depth/segmentation/normal)`, `start_recording`/`stop_recording` to `out/visualization.mp4`, `Rasterizer` renderer, `show_viewer=True`.
- `draw_debug.py` — Debug drawing primitives: `draw_debug_box/line/arrow/sphere/spheres/frame` and `clear_debug_object(s)`; `show_viewer=True`.
- `interactive_debugging.py` — Drops into `IPython.embed()` on a built Franka scene for live inspection; `show_viewer=False`.

Solver showcases (soft-body):
- `mpm.py` — MPM materials on a plane: `MPM.Elastic` box, `MPM.Liquid` sand box, `MPM.ElastoPlastic` sphere; MPM boundary viz; `show_viewer=True`.
- `sph_liquid.py` — `SPH.Liquid` box pouring onto a plane; SPH boundary viz and `get_particles_pos()`; `show_viewer=True`.
- `pbd_cloth.py` — Two `PBD.Cloth` meshes (`meshes/cloth.obj`) with corner particles fixed via `fix_particles`; `show_viewer=True`.

Robot control and IK:
- `control_your_robot.py` — Franka DOF control: `set_dofs_kp/kv/force_range`, then `control_dofs_position/velocity/force` plus `get_dofs_control_force`/`get_dofs_force`; `show_viewer=True`.
- `IK_motion_planning_grasp.py` — Franka pick-and-place: `inverse_kinematics`, `plan_path` (waypoints drawn via `draw_debug_path`), grasp with finger force control; `show_viewer=True`.
- `advanced_IK_multilink.py` — `inverse_kinematics_multilink` targeting both fingers along a path; debug frames updated via `update_debug_objects`; visualization-only `set_dofs_position` (no `scene.step()`); `show_viewer=True`.
- `batched_IK.py` — Batched `inverse_kinematics` across 16 parallel envs with `set_qpos`; headless.
- `position_control_comparison.py` — Tracks a sinusoid with `control_dofs_position` vs `control_dofs_position_velocity` and plots error with matplotlib; `-v/--vis`, `-g/--gpu`.

Parallel simulation:
- `parallel_simulation.py` — 20 envs (`scene.build(n_envs=20)`), batched `control_dofs_position` with torch tensors and per-env `envs_idx` subset control; headless.
- `selecting_rendered_envs.py` — 20 envs but only a subset rendered via `VisOptions(rendered_envs_idx=[10..14])`; `show_viewer=True`.

Advanced multi-physics:
- `advanced_hybrid_robot.py` — `gs.materials.Hybrid` (Rigid + `MPM.Muscle`) wrapping a two-link arm URDF; velocity control; `show_viewer=True`.
- `advanced_muscle.py` — `MPM.Muscle` vs `FEM.Muscle` spheres driven by `set_actuation`; `scene.build(n_envs=0)`; `show_viewer=True`.
- `advanced_worm.py` — `MPM.Muscle` worm mesh (`meshes/worm/worm.obj`) with 4 muscle groups via `set_muscle` and `set_actuation`; `scene.build(n_envs=3)`; headless.

# Constraints

- All scripts hardcode the CPU backend (`gs.init(backend=gs.cpu)`); none require a GPU.
- `advanced_IK_multilink.py`, `advanced_hybrid_robot.py`, `advanced_muscle.py`, and `advanced_worm.py` additionally pass `precision="32", logging_level="info"`.
- Long-running scripts cut their horizon under `"PYTEST_VERSION" in os.environ` (e.g. `1000 if "PYTEST_VERSION" not in os.environ else 5`), per `.github/contributing/EXAMPLES.md`.
- Tutorials are mostly linear scripts without an argparse CLI; only `entity_name.py` (`-v`) and `position_control_comparison.py` (`-v`, `-g`) expose flags.
- Scripts write outputs under `out/` (e.g. `visualization.py` -> `out/visualization.mp4`).

# Gotchas

- `show_viewer=True` (mpm, sph_liquid, pbd_cloth, visualization, draw_debug, control_your_robot, IK_motion_planning_grasp, advanced_IK_multilink, selecting_rendered_envs, advanced_hybrid_robot, advanced_muscle) opens an interactive viewer and needs a display, or a headless EGL backend; `tests/conftest.py` forces `PYGLET_HEADLESS` when no display is available, so CI stays green.
- `interactive_debugging.py` blocks in `IPython.embed()` — it is meant for manual use and has no `PYTEST_VERSION` guard; when run non-interactively it may block until EOF or the test timeout.
- `draw_debug.py` calls `time.sleep(0.01)` every step; harmless but slows the loop.
- `position_control_comparison.py` calls `plt.show()` unless `PYTEST_VERSION` is set (blocks until the window is closed).
- `advanced_IK_multilink.py` intentionally calls `set_dofs_position` without `scene.step()` (visualization-only); real control should use `control_dofs_position` + `scene.step()`.
- `advanced_worm.py` branches on `isinstance(worm.material, gs.materials.MPM.Muscle)` vs `FEM.Muscle`; only the MPM path is exercised here.

# Dependencies

- Core package only for most scripts; `pysplashsurf` backs SPH surface reconstruction (`sph_liquid.py`).
- `matplotlib` for `position_control_comparison.py`; `IPython` for `interactive_debugging.py`; `numpy`/`torch` as used.
- Asset files referenced from the repo: `xml/franka_emika_panda/panda.xml`, `urdf/go2/urdf/go2.urdf`, `urdf/simple/two_link_arm.urdf`, `meshes/cloth.obj`, `meshes/worm/*`.

# Test Strategy

- All 18 scripts are discovered and run as subprocesses by `tests/test_examples.py` (pytest `examples` marker; none match the ignore patterns `*_train.py`, `*_eval.py`, `*_env.py`).
- Under pytest the `PYTEST_VERSION` guards shrink step budgets, and `PYNPUT_BACKEND=dummy` disables keyboard control.

# See Also

- `../` — sibling example areas (rigid, coupling, deformable, sensors, etc.), each with its own CONTEXT.md.
- `../../tests/test_examples.py` — how these scripts are executed and skipped in CI.
- `../../.github/contributing/EXAMPLES.md` — example authoring conventions (CLI flags, backend, output files).
