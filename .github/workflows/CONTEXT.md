# Intent

GitHub Actions workflows that validate, build, benchmark, and release `genesis-world`, plus the helper scripts they run inside CI.
Each workflow is documented below with its trigger, jobs, and non-obvious behaviors.

# Workflows

## `format.yml` — "Precommit Checks (Lint and Format)"

- Trigger: `pull_request` → `main`; concurrency cancels the previous run of the same PR.
- Single job on `ubuntu-24.04`, Python 3.12, runs `pre-commit/action@v3.0.1`.
- This is the fastest gate and the trigger for `nyx_plugin.yml` (via `workflow_run`).

## `generic.yml` — "Generic"

- Triggers: `pull_request` → `main` (unit tests), and `release` published on `main` (PyPI publish).
- `generic-cpu` job: matrix OS × Python × array-mode.
  Base matrix `ubuntu-24.04` + `macos-15` × Python `3.12/3.13/3.14` × CPU backend (ndarray); `include` adds `ubuntu-22.04`, `ubuntu-24.04-arm`, `windows-2025`, a CPU field-array cell per OS, and a `macos-15` GPU (Metal) cell.
  Runs `pytest -m 'required and not slow' --forked --backend <cpu|gpu>` over `./tests`; uploads failure artifacts on failure.
- `publish-pypi` job: only on `release`; builds with `pip wheel --no-deps .` and publishes via `pypa/gh-action-pypi-publish` with OIDC (`id-token: write`, environment `advance`, `skip-existing: true`).

Known issues / gotchas:

- Windows static-array (field) cell is disabled: Quadrants compilation takes ~400 min on Windows (comment in the matrix).
- macOS pins `torch<2.15` because torch>=2.15 no longer exposes the MPS device on the virtualized macOS runner GPU (Quadrants Metal interop, quadrants#1017).
- macOS forces `GS_TORCH_FORCE_CPU_DEVICE=1` (software rendering).
- `matplotlib<3.11` is pinned after Genesis install: 3.11 changed text rasterization and breaks the `MPLPlotter` PNG snapshot (`test_recorders.py::test_plotter`).
- Mesa/OpenGL drivers are installed per OS: kisak-mesa PPA on Ubuntu (with retry loop), `pal1000/mesa-dist-win` extracted to `C:/Windows/System32` on Windows.

## `production.yml` — "Production"

The heavyweight GPU validation and benchmark pipeline, run on self-hosted CoreWeave runners via Slurm.

- Triggers: `push` → `main`, `pull_request` → `main`, and `workflow_call` (reusable — downstream repos call it with `secrets: inherit`).
- `setup` job: checks out the revision holding the workflow file, validates `.github/genesis_image_ver` (regex `<major>_<minor>[<suffix>]`), and for calls from other repos verifies the originating PR is open, mints a GitHub App token (`GENESIS_CI_APP_ID`/`GENESIS_CI_APP_PRIVATE_KEY`), and waits until every older cross-repo call has released the shared runner group `genesis-world`.
- `unit-tests` job (`needs: setup`, `max-parallel: 1`): matrix over `GS_ENABLE_NDARRAY` (`1` dynamic first, then `0` field), runs `pytest --backend gpu --forked ./tests` inside the pinned sqsh container (`/mnt/data/images/genesis-v{ver}.sqsh`).
- `benchmarks` job (`needs: [setup, unit-tests]`, `max-parallel: 1`): runs `pytest -m benchmarks ./tests`, uploads `speed-test-*` and `mem-test-*` artifacts, and on push to `main` uploads results to W&B (`tests/upload_benchmarks_table_to_wandb.py`, project `genesis-benchmarks-2`).
- `workflow_call` inputs: `quadrants_run_id` (wheel artifacts to override the pinned Quadrants), `unit_tests_markexpr`, `unit_tests_modes` (JSON `["1","0"]`); outputs: `is_skipped`; optional secrets: `HF_TOKEN`, `WANDB_API_KEY`, `GENESIS_CI_APP_PRIVATE_KEY`.

Known issues / gotchas:

- Secrets are not passed to PRs from forks, so fork PRs run without `HF_TOKEN`/`WANDB_API_KEY`.
- A merged/closed PR that completed its run is treated as "closed" by the API; `setup` skips the rest and reports `is_skipped`.
- `CACHE_VERSION` env var gates fresh caches; `TIMEOUT_MINUTES=60` caps Slurm allocations; `MADRONA_DISABLE_CUDA_HEAP_SIZE=1` is set globally.
- Self-hosted runs discard caches when the runner dies, so Quadrants/Genesis/uv caches point into the workspace (never persisted across jobs).

## `alarm.yml` — "Benchmark Comparison & Alarm Regression"

- Triggers: `workflow_run` on "Production" `completed` (also `workflow_call`), only for PR events with conclusion success/neutral.
- Downloads `speed-test-*` and `mem-test-*` artifacts from the triggering run (or calling run), runs `scripts/alarm.py` against the W&B baseline, and publishes a commit status (`_Artifact: Benchmark Comparison`) plus a PR comment on regression/alert.
- Exit codes: `0` pass, `42` regression (status failure), `43` alert (warning only); any other exit code is a comparison error and fails the job.

Known issues / gotchas:

- `wandb<0.25` is pinned: 0.25.x changed `api.runs()` to no longer populate `run.config` in batch queries, breaking the `config["revision"]` lookup.
- Tolerance percentages passed to `alarm.py`: runtime FPS 8%, compile time 30%, memory 4% (with `--max-valid-revisions 5 --max-fetch-revisions 40`).
- `download-artifact` must resolve the correct run id: `github.event.workflow_run.id` for same-repo runs, `github.run_id` for calls.

## `nyx_plugin.yml` — "Nyx Plugin"

- Trigger: `workflow_run` on "Precommit Checks (Lint and Format)" `completed`, gated to PR events.
  `workflow_run` is used because checking out the internal plugin repo needs a token that fork PRs never receive; the file is always taken from the default branch.
- Runs on self-hosted `gpu-t4-4-core` (the plugin's own CI runner, so reference images stay comparable).
- Resolves the pin in `.github/nyx_plugin_commit` (must be a full 40-hex hash), checks out `Genesis-Embodied-AI/nyx-genesis-plugin` at that commit with LFS using `GENESIS_REPO_CONTENTS_READ_ONLY_ACCESS_TOKEN`, installs with `uv` (overriding `genesis-world` with the checkout under test), runs the plugin's pytest, and publishes the result as a check named "Nyx Plugin Unit Tests" on the commit.

Known issues / gotchas:

- `allow-unsafe-pr-checkout: true` is required to check out fork code; the token is consumed before any code from the revision under test runs, and `persist-credentials: false` plus a disposable runner contain the risk.
- `libglu1-mesa` must be installed: `pymeshlab` loads from a GLU-linked plugin and silently drops the remeshing filter when it is missing (fails much later as a missing attribute).
- Only dynamic arrays are tested (`GS_ENABLE_NDARRAY=1`); other backends/precisions are covered by the plugin's own CI.
- Check output is truncated to the last 60000 chars (checks API limit of 65535).

## `examples.yml` — "Examples (CPU)"

- Trigger: `pull_request` → `main`; single job `ubuntu-24.04`, Python 3.12.
- Runs `pytest -m examples tests/test_examples.py` with a Quadrants-version-keyed cache (restore/save) and headless Mesa/OpenGL drivers.
- Env: `HF_HUB_DOWNLOAD_TIMEOUT=60`, `QD_OFFLINE_CACHE=1` with `CLEANING_POLICY=never`, `OMNI_KIT_ACCEPT_EULA=yes`.

# Scripts

- `scripts/alarm.py` — benchmark regression comparator run by `alarm.yml`.
  Parses pipeline-format results files (`key=val | ...` lines) into config→metric maps, fetches the W&B baseline (project `genesis-benchmarks-2`, entity `genesis-ai-company`), builds markdown tables, writes CSVs and a check body, and exits `0`/`42`/`43`.
- `scripts/production_build.sh` — in-container bootstrap for `production.yml`.
  Installs `uv`, creates `/venv` (Python 3.12), installs `torch` (cu129), `omniverse-kit` (from `pypi.nvidia.com`), `.[dev,render,usd]` + `pyuipc==0.0.7`, and `matplotlib<3.11`.

# Dependencies

- Slurm (`salloc`/`srun`) and an Enroot sqsh container image at `/mnt/data/images/genesis-v{ver}.sqsh` on the CoreWeave runners.
- CUDA version in `production_build.sh` (cu129) must tightly match the container image (poorly packaged libs such as `libuipc` fail to import otherwise).
- Runner labels: `[self-hosted, coreweave, genesis-world]` and `gpu-t4-4-core`.

# See Also

- `../genesis_image_ver`, `../nyx_plugin_commit` → pin files consumed by these workflows.
- `../contributing/TESTING.md` → local test invocation mirroring what these workflows run.
- `../../tests/upload_benchmarks_table_to_wandb.py` → W&B upload path used by Production benchmarks.
