# Genesis World — Repository Root

## Intent
This repository is **Genesis World** (package `genesis-world` v1.4.6, import `genesis as gs`), a simulation platform for physical AI / robotics.
It combines a unified multi-physics engine, a photo-realistic renderer (Nyx, external), and a cross-platform compiler (Quadrants, external) behind a Pythonic interface, scaling from a laptop kernel to datacenter-grade GPUs.
Previously named Genesis (academic project since Dec 2024); now developed under Genesis AI.

## Architecture (four layers)
- **Simulation Interface** — user-facing API: asset parsing (URDF/MJCF/OBJ/GLB/USD), entity accessors, controllers, sensors, parallel/heterogeneous environments, built-in GUI.
- **Physics** — unified multi-physics engine: Rigid, FEM, MPM, Particle (PBD/SPH), uipc (IPC), plus SAP coupling, sharing one scene and one state.
- **Render** — three camera-sensor render paths: Nyx (in-house, external repo), Luisa (DSL ray tracer), Pyrender (rasterizer).
- **Compiler** — Quadrants (external) lowers Python kernel code to CUDA/ROCm/Metal/Vulkan/x86/ARM and carries autodiff, GPU graphs, and fastcache.

## Repository Layout (Routing Table)
- `genesis/` → the Python package: public `gs.*` API, physics engine, utils, vis, vendored ext, options, recorders, grad bridge, assets.
- `tests/` → pytest suite (rigid, particles, deformable, sensors, grad, parsers, rendering, coupling, ipc, integration, benchmarks).
- `examples/` → runnable demo scripts (also executed as pytest `examples`).
- `.github/` → CI/CD workflows and contributor reference docs.
- `doc/` → git submodule `https://github.com/Genesis-Embodied-AI/genesis-doc` (documentation source; arrives as an EMPTY placeholder in worktrees — `git submodule update --init` to populate).
- `imgs/` → README/landing images.
- Root config files: `pyproject.toml`, `README.md`, `CLAUDE.md` (dev guidelines), `CODING_GUIDELINES.md` (naming/style conventions), `RELEASE.md`, `LICENSE`, `MANIFEST.in`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.gitignore`, `.gitattributes`.

## Key Facts
- Python `>=3.12,<3.15`; entry point `gs = "genesis._main:main"` (CLI: `gs launch` / `gs play` / `gs replay` / `gs animate`).
- Core dependency: `quadrants==1.4.0`; `torch` must be installed separately per backend.
- `genesis/ext/LuisaRender` is a git submodule (`https://github.com/Alif-01/LuisaRender`) and arrives empty in worktrees, as does `doc/`.

## Constraints
- Development conventions live in `CLAUDE.md`, `CODING_GUIDELINES.md`, and `.github/contributing/` (ARCHITECTURE, TESTING, CODING_CONVENTIONS, EXAMPLES, PULL_REQUESTS, USD_PARSER).
- The rigid solver (`genesis/engine/solvers/rigid/`) is the project's reference implementation for naming and kernel structure — imitate it.
- PR titles carry a bracket tag: `[BUG FIX]`, `[FEATURE]`, `[MISC]`, `[CHANGING]`, `[BREAKING]`; commit titles are plain sentences ending with a period.
- Lint/format via ruff (line length 120) through pre-commit.

## Dependencies
Beyond the Python package manager: `torch` (PyTorch, manual install), optional GPU backends (CUDA/ROCm/Metal), and optional extras defined in `pyproject.toml` (`dev`, `docs`, `render`, `usd`).
IPC demos need `pyuipc`/`libuipc`; RL examples need `rsl-rl-lib`; Nyx walkthroughs live in the separate `genesis-nyx` repo.
`examples/locomotion/try1.py` computes its control and reward with `evoxc` (EvoX Compiler), an optional package that is not declared in `pyproject.toml`.

## Test Strategy
pytest configured in `pyproject.toml`: `--numprocesses=auto --dist=worksteal --import-mode=importlib --timeout=1200`, default deselection of `benchmarks`/`examples` markers.
`@pytest.mark.required` = minimal set that must pass before merge; `@pytest.mark.slow` = baseline runtime ≥150s.
Tests run across backends (cpu/gpu/cuda/amdgpu/metal) and field-vs-ndarray array modes; see `tests/CONTEXT.md` for the full picture.

## See Also
- `genesis/CONTEXT.md` — package-level routing into engine/utils/vis/ext/options/recorders/grad.
- `tests/CONTEXT.md`, `examples/CONTEXT.md`, `.github/CONTEXT.md` — the other top-level trees.

## Status
Complete: every top-level tree and the `genesis/` package subtree is documented with its own CONTEXT.md.
