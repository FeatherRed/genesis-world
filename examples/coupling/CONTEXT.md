# examples/coupling

## Intent
Directory of runnable example scripts demonstrating Genesis multi-solver coupling.
Each script couples at least two solvers: rigid bodies with MPM, cloth (PBD), FEM, or SPH, plus SPH-with-MPM.
These are integration examples exercised by the pytest suite, not unit tests.

## API Surface
All scripts follow the same pattern: `import genesis as gs` -> `gs.init(...)` -> `gs.Scene(sim_options=..., mpm_options=..., sph_options=..., pbd_options=..., fem_options=..., vis_options=..., viewer_options=...)` -> `add_entity`/`add_emitter` -> `build` -> loop `scene.step()`.
There are no child subdirectories; every file is a flat script in this directory.

### rigid + MPM
- `rigid_mpm_attachment.py` — attaches top particles of an MPM Elastic cube to a rigid box link via `set_particle_constraints`, then drives the box with `set_qpos`.
- `sand_wheel.py` — pours MPM Sand from an emitter over fixed URDF wheels; demonstrates `needs_coup`/`coup_friction`/`coup_softness` rigid-MPM contact.
- `water_wheel.py` — pours liquid over a fixed fancy URDF wheel; `--solver sph|mpm` picks the fluid solver and `--recon` enables SPH surface reconstruction.
- `grasp_soft_cube.py` — Franka Panda (MJCF, `coup_friction=1.0`) grasps and lifts a small MPM Elastic cube via position/force DOF control.
- `flush_cubes.py` — two MPM Liquid emitters wash MPM Elastic cube/sphere/cylinder bodies across a plane.
- `cut_dragon.py` — a cross-cutter mesh slices an MPM Elastic dragon mesh (`MPMOptions(enable_CPIC=True)`).

### rigid + cloth (PBD)
- `cloth_on_rigid.py` — PBD Cloth drapes over a fixed frictionless rigid sphere (`coup_friction=0.0`).
- `cloth_attached_to_rigid.py` — pins cloth corner particles to a rigid box link via `fix_particles_to_link`, moves/teleports the box, then `release_particle`s mid-run.

### rigid + FEM
- `fem_cube_linked_with_arm.py` — pins a FEM `linear_corotated` cube to a Franka Panda end-effector via `set_vertex_constraints`; the arm picks, carries, then drops the cube and records a video.

### SPH + rigid
- `sph_rigid.py` — drops a frictionless rigid cube into an SPH Liquid pool.

### SPH + MPM
- `sph_mpm.py` — floats an MPM Elastic duck in an SPH Liquid pool; sets both `mpm_options` and `sph_options` bounds.

## Constraints
Coupling between rigid bodies and the particle solvers is configured via rigid material flags `needs_coup`, `coup_friction`, `coup_softness` plus per-solver option bounds.
MPM and SPH each need a bounding box (`lower_bound`/`upper_bound`) sized to the scene.
Emitter-driven scenes call `scene.build(n_envs=0)`; entity-only scenes use the default `scene.build()`.
Every script is a pytest example: it shortens its run when `PYTEST_VERSION` is present in `os.environ` (typically 1-25 steps) and skips the viewer.
Manual-run CLI flags vary per script (`-v/--vis`, `-g/--gpu`, plus extras such as `--solver`, `--recon`, `--dt`, `--substeps`, `--steps`, `--num-teleports`).

## Dependencies
None beyond core Genesis and its usual deps (`torch`, `numpy`, `tqdm`).
Scripts reference shared assets resolved under `genesis/assets/` (e.g. `urdf/plane/plane.urdf`, `urdf/wheel/wheel.urdf`, `urdf/wheel/fancy_wheel.urdf`, `xml/franka_emika_panda/panda.xml`, `meshes/cloth.obj`, `meshes/cross_cutter.obj`, `meshes/dragon/dragon.obj`, `meshes/duck.obj`).
SPH surface reconstruction (`water_wheel.py --recon`, `vis_mode="recon"`) relies on pysplashsurf (called from `genesis/utils/particle.py`).

## Gotchas
Most CPU examples set `precision="32"`; `rigid_mpm_attachment.py`, `grasp_soft_cube.py`, and `fem_cube_linked_with_arm.py` rely on the default precision — keep precision consistent within a scene.
GPU is opt-in via `-g/--gpu` only in `rigid_mpm_attachment.py`, `cloth_on_rigid.py`, `cloth_attached_to_rigid.py`, and `fem_cube_linked_with_arm.py`; the other scripts hardcode `gs.cpu`.
`cut_dragon.py` is the only cutting demo (requires CPIC); `grasp_soft_cube.py` and `fem_cube_linked_with_arm.py` are the robot-control demos (Franka Panda).
`fem_cube_linked_with_arm.py` requires `FEMOptions(use_implicit_solver=True)` because the `linear_corotated` model only supplies implicit-solver energy derivatives, and it writes output to `out/cube_link_arm_*.mp4`.
`sph_mpm.py` sets both MPM and SPH bounds and the two boxes differ slightly by design.

## See Also
- `../sap_coupling/` — SAP-based rigid/FEM coupling examples (sibling — read-only, escalate writes to `../`).
