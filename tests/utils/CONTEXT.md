# tests/utils — shared test infrastructure

## Intent
Shared, importable helpers for the test suite — not tests themselves (no `test_*` functions).
Every module here is imported by test modules across `tests/`, so it must stay dependency-light and side-effect-free at import time.

## API Surface
- `simulators.py` — parity simulation builders.
  - `MjSim` dataclass (`model`, `data`).
  - `build_mujoco_sim(xml_path, gs_solver, gs_integrator, merge_fixed_links, multi_contact, adjacent_collision, native_ccd, *, friction_cone) -> MjSim`.
  - `build_genesis_sim(xml_paths, gs_solver, gs_integrator, merge_fixed_links, multi_contact, mujoco_compatibility, adjacent_collision, gjk_collision, show_viewer, mj_sim, *, enable_collision, disable_constraint, friction_cone, friction_torsional, friction_rolling) -> gs.Sim`.
  - Both apply the known MuJoCo↔Genesis normalizations (dense jacobian, `geom_condim >= 3`, islands off, aligned invweights, re-centered COM dust) so parity tests share one setup path.
- `mujoco_parity.py` — step-by-step MuJoCo/Genesis consistency checking.
  - `simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos=None, qvel=None, *, tol, num_steps, ignore_constraints=False)`.
  - `align_mujoco_invweight0(model)`.
  - Private helpers: `_get_model_mappings`, `_gs_search_by_joints_name`, `_gs_search_by_links_name`, `check_mujoco_model_consistency`, `check_mujoco_data_consistency`, `_pair_constraint_rows`, `_compute_efc_tolerances`, `init_paired_simulators`.
  - In single precision it monkeypatches `genesis.engine.solvers.rigid.rigid_solver.func_solve_body` to substitute MuJoCo's double-precision reference acceleration before each solve.
- `assertions.py` — numeric and image assertions.
  - `assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None)` (torch/array/generator aware; `tol` XOR `atol`/`rtol`).
  - `assert_equal(actual, desired, *, err_msg=None)`.
  - `assert_pixel_match(img_a, img_b, *, err_msg, verbose, std_err_threshold, ratio_err_threshold, blurred_kernel_size)` — tolerates renderer jitter; prints base64 delta PNG on mismatch.
  - `rgb_array_to_png_bytes(rgb_arr)`.
  - Constants `IMG_STD_ERR_THR`, `IMG_NUM_ERR_THR`, `IMG_BLUR_KERNEL_SIZE`.
- `assets.py` — HuggingFace dataset retrieval.
  - `get_hf_dataset(pattern, repo_name="assets"|"snapshots", local_dir=None, num_retry=4, retry_delay=30.0)`.
  - Downloads from `Genesis-Intelligence/{assets,snapshots}` at pinned revisions `HUGGINGFACE_ASSETS_REVISION` / `HUGGINGFACE_SNAPSHOT_REVISION`, validates downloaded files, retries on failure.
  - `get_git_commit_info(ref)`, `get_git_commit_timestamp(ref)` for benchmark metadata.
- `collision.py` — collision ground truth and inspection.
  - `get_genuine_interpenetration(links, cross_tol=1e-3, n_dir=40, n_bisect=9, is_exact=True) -> (max_depth, crossings)`.
  - `display_collision_pairs(pairs)` — opens `mesh_pairs_viewer.html` (the template in this directory) in a browser.
  - `Crossing` NamedTuple (`link_a`, `link_b`, `depth`).
- `misc.py` — `pprint_oneline(data, delimiter, digits=None)` used for benchmark-ID formatting.
- `mesh_pairs_viewer.html` — self-contained viewer template used by `display_collision_pairs` (holds a `__DATA__` placeholder).

## Constraints
- Import-time side effects are forbidden: conftest's `pytest_sessionstart` hook raises if `genesis`/`numba`/`numpy`/`quadrants`/`torch` are imported before the worker environment is configured.
- Signature changes ripple across many test modules — treat this as a stable internal API.

## Known Issues
- `simulators.py` carries two FIXMEs: Genesis gives every contact a sliding-friction basis (so MuJoCo `geom_condim` is forced to `>= 3`), and Genesis has no tendons (MuJoCo tendon ranges are zeroed).

## See Also
- `../conftest.py` — `mj_sim` / `gs_sim` / `xml_path` / `xml_paths` fixtures wrap `simulators.py`.
- `../rigid/` — primary consumer of the parity helpers (via `test_mujoco_parity.py`).
- `../grad/` — reuses parity helpers for differentiable tests.

## Status
Complete for its current scope.
