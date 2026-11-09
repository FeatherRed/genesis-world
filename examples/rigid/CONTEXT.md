# Intent

`examples/rigid/` is the largest demo group (~37 standalone scripts) exercising the genesis rigid-body solver: robot-arm control and manipulation, IK, terrain, domain randomization, multi-GPU/distributed training, mesh collision and convex decomposition, joint/constraint types, and assorted physics features.
Every script is a self-contained `main()` following the standard `gs.init` -> `gs.Scene` -> `add_entity(gs.morphs...)` -> `build` -> `step` pattern, and each is also executed as a pytest example by `tests/test_examples.py`.

# API Surface

## Control (Franka & manipulation)
- `control_franka.py` — Franka PD-control showcase cycling through position, velocity, and force control while printing `get_dofs_control_force` (CPU-only).
- `gravity_compensation.py` — Three Franka arms with `material=gs.materials.Rigid(gravity_compensation=...)` set to 0.0/0.5/1.0 to demonstrate gravity compensation.
- `single_franka.py` — Minimal single Franka scene with a GUI camera and `visualize_contact=True` (`-g` for GPU).
- `single_franka_envs.py` — Multi-env Franka rendering rgb/depth/seg/normal per step with optional video recording; `--sep` splits environments into separate islands.
- `franka_cube.py` — IK-guided grasp of a small cube using `box_box_detection` and finger position control.
- `grasp_bottle.py` — 49 parallel bottle grasps (GPU-only) using `inverse_kinematics`, `plan_path`, and position/force finger control.
- `torsional_grasp.py` — In-hand pivoting under the elliptic friction cone with torsional friction (checkered ball pinched between plates).
- `suction_cup.py` — Pick-and-place via a suction `add_weld_constraint`/`delete_weld_constraint` on the rigid solver.
- `merge_entities.py` — Assemble arm and hand from URDF/MJCF pairs and join them with `hand.attach(franka, "attachment")`.
- `control_mesh.py` — PD control of a single free mesh entity's base free-joint DOFs (zero gravity, rotates a duck).

## IK
- `ik_duck.py` — Position IK on a kinematic free mesh tracking a moving target; visualization only (no `step`).
- `ik_franka.py` — Position/quaternion IK on a kinematic Franka with `return_error` and `rot_mask`; visualization only.
- `ik_franka_batched.py` — Batched IK over 64 GPU environments with per-env angular speeds; visualization only.
- `ik_shadow_hand.py` — Multi-target IK via `inverse_kinematics_multilink` on the Shadow Hand.
- `ik_custom_chain.py` — IK restricted to a custom `dofs_idx_local` subset (wrist + index-finger chain).
- `diffik_controller.py` — Manual differentiable IK controller (Jacobian transpose + damped solve) for panda/ur5e.

## Terrain
- `terrain_from_mesh.py` — Convert `terrain_45.obj` to a height field via `genesis.utils.terrain.mesh_to_heightfield` and roll a ball down it.
- `terrain_height_field.py` — Hand-authored height field with the Newton constraint solver; draws debug spheres on the surface.
- `terrain_subterrain.py` — 2x2 subterrain grid with named terrain types (flat/random_uniform/pyramid_sloped/discrete_obstacles) across 100 environments.

## Domain randomization
- `domain_randomization.py` — Per-env friction/mass/COM/inertia randomization on a Go2 quadruped using `batch_links_info=True`.

## Multi-GPU / distributed
- `multi_gpu.py` — One simulation per GPU via `multiprocessing`, pinning `CUDA_VISIBLE_DEVICES`/`QD_VISIBLE_DEVICE`/`EGL_DEVICE_ID` per process.
- `ddp_multi_gpu.py` — torchrun DDP training of a TinyMLP, one scene per rank; NCCL (CUDA) or Gloo (CPU) backend.

## Meshes & convex decomposition
- `nonconvex_mesh.py` — Non-convex tank mesh (auto convex-decomposed) catching a falling ball.
- `convex_decomposition.py` — Downloads pre-decomposed MJCF assets (donut/mug/cup/apple) from Hugging Face and drops them into a tank.
- `authored_decomp.py` — Stacking tower of URDF parts (base/pole, rings, ball) with `max_collision_pairs=500` and optional GJK (`--gjk`) plus COM debug drawing.

## Joints & constraints
- `closed_loop.py` — Four-bar linkage closed by MJCF equality constraints (`connect` vs `weld`).
- `screw_constraint.py` — Nut screwed onto a bolt via `add_screw_constraint`; keyboard-driven in the viewer, scripted otherwise.
- `bolt_nut_self_screw.py` — Nut screwing itself down via non-convex thread contact (no convexify/decimate, stiff constraint solve).
- `rolling_coast.py` — Rolling-friction comparison: one checkered ball coasts forever, the braked one stops.

## Misc
- `accelerometer_duck.py` — `get_links_acc` debug arrows on a free-falling duck.
- `accelerometer_franka.py` — Compare `get_links_acc` against a finite difference of `get_links_vel` on a Franka.
- `apply_external_wrench.py` — Direct `rigid_solver.apply_links_external_wrench` (restoring force + torque) on a cube.
- `friction_breakaway.py` — Coulomb stiction sweep comparing pyramidal vs elliptic friction cones; documents pyramidal creep.
- `heterogeneous_simulation.py` — Per-env geometry variants via a list of morphs (assignment rules in the module docstring).
- `hibernation.py` — Islands + hibernation (`use_hibernation`, `performance_mode`) with a live step-rate/awake-bodies plot; ducks-grid or dominos scenes.
- `set_phys_attr.py` — Per-env kp/kv/force-range/armature/stiffness/damping/mass via `batch_links_info`/`batch_joints_info`/`batch_dofs_info`.
- `single_franka_batch_render.py` — BatchRenderer (gs-madrona) with multiple cameras/lights and `FrameImageExporter`.
- `wrecking_ball.py` — Inline-MJCF chain (trimesh-generated links) swinging into a cube pile; hibernation option + live step-rate plot.

# Constraints

- Scripts are flat demo entry points (no subdirectories); each keeps to one demonstrated feature.
- Most scripts take `-v/--vis` (viewer); the GPU-capable ones also take `-g/--gpu`.
- Assets are referenced relative to the repository root (`xml/`, `urdf/`, `meshes/`, `tower/`, `textures/`), so scripts must run from the repo root.
- Under pytest (`PYTEST_VERSION` in env), loops shrink to ~5 steps / tiny batches; keep that guard when editing.

# Dependencies

- `grasp_bottle.py`, `ik_franka_batched.py`, and `single_franka_batch_render.py` require a GPU backend (`gs.gpu`).
- `single_franka_batch_render.py` additionally needs the optional `gs_madrona` module (Linux-only native batch renderer) and a real CUDA torch build.
- `convex_decomposition.py` needs `huggingface_hub` and network access (downloads the `Genesis-Intelligence/assets` dataset).
- `ddp_multi_gpu.py` needs `torch` (+ `torch.distributed`); `wrecking_ball.py` needs `trimesh`.
- Several scripts use `torch`/`numpy` for tensor bookkeeping (`terrain_height_field.py`, `terrain_subterrain.py`, `domain_randomization.py`, `set_phys_attr.py`).

# Gotchas

- `gs.gpu` falls back to CPU when no GPU is present, so most GPU-flagged scripts still run on CPU CI — except `single_franka_batch_render.py`, which is CUDA-only and skipped by `tests/test_examples.py` without real CUDA.
- `domain_randomization.py` must set `RigidOptions.batch_links_info=True` or per-env mass/COM has nowhere to live (link info is shared by default).
- `set_phys_attr.py` shows `batch_links_info`/`batch_joints_info`/`batch_dofs_info` are off by default for speed; enable only when per-env attributes are needed.
- `multi_gpu.py` pins `CUDA_VISIBLE_DEVICES`, `QD_VISIBLE_DEVICE`, and `EGL_DEVICE_ID` per process; `ddp_multi_gpu.py` sets the first two but leaves `EGL_DEVICE_ID` commented out with a FIXME ("forcing rendering device is not working reliably on all machines").
- `authored_decomp.py` raises `max_collision_pairs` to 500 because stacked convex pieces exceed the default contact budget.
- `bolt_nut_self_screw.py` requires `decimate=False` + `convexify=False` to preserve the thread helix and `constraint_timeconst=4e-3` (stiff contact) or the nut sinks through the flanks; torque magnitude is capped at 5.3 N*m to avoid NaN divergence.
- `hibernation.py` and `wrecking_ball.py` use `performance_mode` (field storage), the fastest CPU layout, and tune `max_collision_pairs`/`use_hibernation`.
- `ik_*.py` and `diffik_controller.py` are visualization-only (they `set_qpos` and never `scene.step()`); for real control use `control_dofs_position` + `scene.step`.
- `screw_constraint.py` keybind callbacks run on the viewer thread, so they only set requests carried out by the main loop between steps.

# Test Strategy

- `tests/test_examples.py` runs every `.py` here as a subprocess (pytest `examples` marker) with `PYNPUT_BACKEND=dummy` and a 600s timeout.
- No script here matches the ignore patterns (`*_train.py`, `*_eval.py`, `*_env.py`); `single_franka_envs.py` ends in `_envs`, so it is executed.
- `single_franka_batch_render.py` is the only CUDA-only script (skipped without real CUDA) and the only one gated on optional `gs_madrona`.
- Budgets are shrunk under `PYTEST_VERSION`; verify a script still exercises its core path within those ~5 steps.

# See Also

- `../../tests/test_examples.py` — pytest driver for all examples (sibling — read-only).
- `../` — sibling example groups (rigid is one of several theme directories under `examples/`).
