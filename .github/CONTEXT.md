# Intent

GitHub governance layer for the `genesis-world` Python package: CI/CD workflows, contributor-facing documentation, issue templates, PR template, code ownership, and pinned version pointers.
Nothing here is shipped in the package; it is consumed exclusively by GitHub Actions and by human contributors.

# API Surface

Top-level files:

- `CODEOWNERS` — default owners of every file: `@Genesis-Embodied-AI` (team), `@YilingQiao`, `@duburcqa`.
- `pull_request_template.md` — PR body template.
  Requires a bracketed title prefix `[BUG FIX]` / `[FEATURE]` / `[MISC]` / `[CHANGING]` / `[BREAKING]`; links to `contributing/PULL_REQUESTS.md` and `genesis-doc` for documentation changes.
- `genesis_image_ver` — pinned Slurm container image version (currently `1_26`), format `<major>_<minor>[<suffix>]`.
  Read and validated by `workflows/production.yml` (`setup` job); a PR must be rebased onto default branch before it can run if the file is missing.
- `nyx_plugin_commit` — pinned 40-character full commit hash of the `Genesis-Embodied-AI/nyx-genesis-plugin` repository.
  Read and validated by `workflows/nyx_plugin.yml`; must be a full hash (never a branch or short form) so the pin cannot be smuggled through step output or silently move.

# Routing Table

- `workflows/` → GitHub Actions workflows and their helper scripts (`alarm.py`, `production_build.sh`).
- `contributing/` → Contributor-facing reference docs (architecture, conventions, testing, examples, PR flow, USD parser).
- `ISSUE_TEMPLATE/` → GitHub issue forms (bug, feature, documentation) and issue config.

# Constraints

- Pinned versions (`genesis_image_ver`, `nyx_plugin_commit`) live in the tree, not hardcoded in workflow YAML, so a PR bumping a pin is validated by its own run before it takes effect.
- Workflows that need secrets unavailable to fork PRs use the `workflow_run` trigger (runs in base-repo context) instead of `pull_request` — see `workflows/production.yml`, `workflows/nyx_plugin.yml`, `workflows/alarm.yml`.
- All workflows target the `main` branch only.

# Design Decisions

- `production.yml` is a reusable workflow (`workflow_call`) in addition to its push/PR triggers, so downstream repositories (e.g. Quadrants) can validate against Genesis's GPU test suite and shared benchmark baseline.
- The benchmark regression comparator (`alarm.yml`) runs as a `workflow_run` on Production completion and compares against the W&B history of the default branch.
- Self-hosted GPU work is delegated to CoreWeave runners orchestrated through Slurm with a pinned container image.

# Dependencies

- GitHub Actions runners: GitHub-hosted (`ubuntu-*`, `macos-15`, `windows-2025`) plus self-hosted GPU (`[self-hosted, coreweave, genesis-world]`) and `gpu-t4-4-core`.
- External services/secrets: Weights & Biases (`WANDB_API_KEY`, entity `genesis-ai-company`), Hugging Face (`HF_TOKEN`), PyPI (OIDC publishing), GitHub App (`GENESIS_CI_APP_ID` / `GENESIS_CI_APP_PRIVATE_KEY`).
- Internal private repo `Genesis-Embodied-AI/nyx-genesis-plugin` (read via `GENESIS_REPO_CONTENTS_READ_ONLY_ACCESS_TOKEN`).
- Documentation issues are tracked here but pages live in the separate `genesis-doc` repository.

# See Also

- `../CODING_GUIDELINES.md` → repo-wide coding guidelines that point back at `contributing/`.
- `../pyproject.toml` → dev/render/usd extras, pytest markers referenced by the workflows.
- `../tests/` → test suite the workflows execute (markers `required`, `slow`, `benchmarks`, `examples`).
