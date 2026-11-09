# Intent

Runnable demo scripts and reference code for the `genesis-world` physics engine, organized as one themed directory per feature area.
Each script is a standalone `main()` (or top-level) following `import genesis as gs` -> `gs.init(backend=...)` -> `gs.Scene(...)` -> `add_entity(gs.morphs...)` -> `build()` -> `step()`, and almost every script is also executed as a pytest example by `tests/test_examples.py` (pytest `examples` marker).
Authoring rules live in `.github/contributing/EXAMPLES.md`; read it before adding or editing scripts here.

# API Surface

- `collision/` — rigid contact: multi-contact manifolds, contype/conaffinity filtering, box-pile/tower stress tests.
- `coupling/` — default multi-solver coupling: rigid+MPM/cloth(PBD)/FEM, SPH+rigid, SPH+MPM.
- `deformable/` — MPM/PBD/FEM solids and liquid, incl. differentiable MPM.
- `drone/` — Crazyflie 2.X quadcopter: teleop, trajectory, PID flight, RL hover train/eval.
- `gui/` — the built-in ImGui viewer overlay (`enable_gui=True`).
- `ipc/` — external IPC (Incremental Potential Contact) coupler demos via `pyuipc`.
- `kinematic/` — `KinematicEntity` as a visualization-only ghost reference motion.
- `locomotion/` — Go2 quadruped: rsl-rl PPO train/eval (`go2_train.py`/`go2_eval.py`), independent TensorNEAT neuroevolution walking trainer (`go2_neat_train.py` + `go2_neat_env.py` + `go2_neat_kernels.py`), scripted-trot EvoX demo (`try1.py`), backflip playback.
- `manipulation/` — Franka grasping RL env + train/eval + behavior cloning.
- `neuroevolution/` — task-agnostic TensorNEAT/evoxc layer shared by the Go2 and drone pilots: env contract, seed rules, evolution graphs, width-generic policy graph, `NeuroevolutionTrainer`, and `benchmarks/` (Stage-3 harness, raw records, `RESULTS.md`).
- `rendering/` — photorealistic RayTracer (LuisaRender), camera control, async render, speed test.
- `rigid/` — the largest group: Franka control, IK, terrain, domain randomization, multi-GPU, convex decomposition, constraints.
- `sap_coupling/` — built-in SAP (Semi-Analytic Primal) rigid/FEM coupler.
- `sensors/` — camera backends, LiDAR, IMU, joint torque, tactile, contact force, surface distance, temperature grid.
- `speed_benchmark/` — throughput/timing benchmarks at 128-30000 parallel envs.
- `tutorials/` — onboarding progression; `hello_genesis.py` is the canonical first example.
- `usd/` — USD asset import via `gs.morphs.USD` / `scene.add_stage`.
- `viewer_plugin/` — custom viewer plugins: keybindings, raycast point selection, mouse interaction.

# Constraints

- Nothing lives at the root of `examples/`; every script is inside a feature directory (see `.github/contributing/EXAMPLES.md`).
- CLI short flags are a closed, reserved set (one meaning per letter across the tree): `-v/--vis`, `-g/--gpu`, `-b/--num-envs`, `-s/--steps`, `-t/--seconds`, `-e/--exp-name`, `-r/--record`, `-o/--output-dir`, `-d/--debug`.
- Backend is spelled out, never implicit: CPU by default; `-g/--gpu` opt-in; a script that genuinely needs GPU hardcodes `backend=gs.gpu` and says why.
- Outputs go under `out/` (git-ignored), resolved against the launch directory; training scripts keep checkpoints under `logs/<exp-name>/`.
- Scripts conventionally shorten their run when `"PYTEST_VERSION" in os.environ` — note this variable is NOT set by `tests/test_examples.py` or `.github/workflows/examples.yml` at HEAD; it is a documented convention that must be supplied by the invoking environment.

# Test Strategy

- `tests/test_examples.py` discovers every `examples/**/*.py` and runs each as a subprocess (pytest `examples` marker, 600s timeout, `PYNPUT_BACKEND=dummy`).
- IGNORED by that harness: `*_train.py`, `*_eval.py`, `*_env.py`, `behavior_cloning.py`, `quadcopter_controller.py` (so RL train/eval/env entry points are documentation, not tests).
- Optional-dependency skips map: `usd/` -> `pxr` (usd-core), `ipc/` -> `uipc` (pyuipc), `rigid/single_franka_batch_render.py` -> `gs_madrona`, `rendering/demo.py` -> `LuisaRenderPy`; `single_franka_batch_render.py` is the only CUDA-only script.

# Dependencies

- Core package covers most examples; extras used: `usd` (`usd-core`/`pxr`, `filelock`), `render` (`imgui-bundle`, `open3d`, `pybind11`), `dev` (`tensordict`, `matplotlib`, `huggingface_hub`, etc.).
- Not in `pyproject.toml` (install separately): `pyuipc` (IPC demos), `rsl-rl-lib>=5.0.0` + `tensorboard` (drone/manipulation RL, and the locomotion PPO path `go2_train.py`/`go2_eval.py` only).
- Also not in `pyproject.toml`: `evoxc` (EvoX Compiler, 0.1.0) is required by the locomotion scripted-trot demo `locomotion/try1.py`, by both neuroevolution trainers (`locomotion/go2_neat_train.py`, `drone/hover_neat_train.py`) and by the whole `neuroevolution/` layer; `tensorneat` 0.2.0 from its `evoxc` branch is required by the two trainers and the `neuroevolution/` layer.
- Optional `iree-base-compiler`/`iree-base-runtime` adapters are what enable non-CPU `evoxc` targets; ROCm `tell()` additionally needs a `jax_rocm7_pjrt` PJRT plugin. See `locomotion/CONTEXT.md` (AMD Instinct MI300X VF / ROCm), `drone/CONTEXT.md` and `neuroevolution/CONTEXT.md` for the detail.
- `examples/neuroevolution/benchmarks/RESULTS.md` is the Stage-3 measurement record (PyTorch env vs evoxc env replay parity, environment/TensorNEAT timings, initial-port vs device-resident rollout, unmatched baselines).
- Optional backends: `LuisaRenderPy` (RayTracer; lives in the `genesis/ext/LuisaRender` git submodule), `gs_madrona` (Linux x86_64 native batch renderer, a core dep on that platform).
- `pysplashsurf` (core dep) backs SPH surface reconstruction (e.g. `coupling/water_wheel.py --recon`).
- Network-dependent scripts download assets from Hugging Face (`Genesis-Intelligence/assets`) at runtime.

# Gotchas

- Many scripts use `show_viewer=True` and need a display or headless EGL (`tests/conftest.py` forces `PYGLET_HEADLESS` when no display is available).
- `gs.gpu` falls back to CPU when no GPU is present; only `rigid/single_franka_batch_render.py` is truly CUDA-only.
- Interactive scripts (`viewer_plugin/*`, `sensors/lidar_teleop.py`, `ipc/ipc_robot_cloth_teleop.py`, `tutorials/interactive_debugging.py`) may block on the viewer/`IPython.embed()` and are meant for manual use.

# Routing Table

- `collision/` -> rigid contact / manifold / contact-filter demos
- `coupling/` -> default MPM/SPH/PBD/FEM multi-solver coupling
- `deformable/` -> MPM/PBD/FEM solids & liquid, differentiable MPM
- `drone/` -> quadcopter control + RL hover stack
- `gui/` -> ImGui viewer overlay
- `ipc/` -> external IPC coupler (`pyuipc`)
- `kinematic/` -> kinematic ghost entities
- `locomotion/` -> Go2 PPO train/eval (`go2_*`), TensorNEAT neuroevolution walking trainer (`go2_neat_*` + `go2_device_rollout.py`), scripted trot (`try1.py`), backflip (checkpoints in `locomotion/backflip/`)
- `neuroevolution/` -> shared neuroevolution env contract, seeding rules, evolution/policy graphs, `NeuroevolutionTrainer`, benchmark harness + `benchmarks/RESULTS.md`
- `manipulation/` -> grasping RL + behavior cloning
- `rendering/` -> RayTracer (LuisaRender), cameras, async render
- `rigid/` -> rigid-body control/IK/terrain/DR/multi-GPU/constraints
- `sap_coupling/` -> built-in SAP coupler
- `sensors/` -> sensor API demos
- `speed_benchmark/` -> large-batch throughput benchmarks
- `tutorials/` -> onboarding/hello-world examples
- `usd/` -> USD import
- `viewer_plugin/` -> custom viewer plugins

# See Also

- `../tests/test_examples.py` — pytest example harness (sibling — read-only, escalate writes to parent).
- `../.github/contributing/EXAMPLES.md` — example authoring conventions (sibling — read-only).
- `../pyproject.toml` — extras and core dependencies (sibling — read-only).
