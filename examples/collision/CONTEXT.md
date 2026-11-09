# Intent

Runnable demos of the rigid-body collision/contact pipeline: contact filtering, multi-contact manifold recovery, and stress tests of large box stacks.
Each script is a self-contained `main()` using the standard `gs.init` -> `gs.Scene` -> `add_entity` -> `build` -> `step` loop and is executed by `tests/test_examples.py` (pytest `examples` marker).

# API Surface

- `contact_manifold.py` — Compares the two multi-contact detection schemes behind `RigidOptions.enable_contact_patch` (clipped contact patch vs. perturbed re-detection) with a scripted box-on-box sweep (`--detector both|patch|perturbation`, `-s`, `-v`). Uses `box_box_detection=False` + `use_gjk_collision=True` so GJK-based manifold code is exercised.
- `contype.py` — Demonstrates `contype`/`conaffinity` 32-bit bitmask contact filtering between a plane and red/green/blue cubes and a dragon mesh.
- `pyramid.py` — Builds a `--num-cubes` (5-10) box pyramid, `--pile-type static|falling`; `-g` selects GPU, `-s` steps.
- `tower.py` — 50-layer box tower with a dropped `--object sphere|cylinder|duck`; `performance_mode=True`, `max_collision_pairs=400`.

# Constraints

- Pure rigid-body examples; no soft-body, FEM, or coupling solvers involved.
- Standard CLI conventions: `-v/--vis` opens the viewer, `-g/--gpu` (where present) switches to `gs.gpu`.
- When `PYTEST_VERSION` is present in `os.environ` (a `.github/contributing/EXAMPLES.md` convention), step budgets shrink drastically (e.g. `tower.py` runs 20 vs. 10000 steps, `contact_manifold.py` runs 8).

# Gotchas

- `tower.py` and `pyramid.py` are collision stress tests; `tower.py` needs `max_collision_pairs=400` and `performance_mode=True` to remain fast.
- `contact_manifold.py` disables the specialized box-box detector so the GJK manifold code path is actually hit.

# Dependencies

- None beyond the core package; CPU by default, optional CUDA GPU via `-g`.

# See Also

- `../tutorials/` — gentler introduction to `gs.Scene`/`gs.morphs` basics.
- `../../tests/test_examples.py` — how these are run and skipped in CI.
