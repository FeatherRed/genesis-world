# Intent

Contributor-facing reference documentation for `genesis-world`.
These are plain Markdown guides (no code execution); they are linked from `../pull_request_template.md`, `../../CODING_GUIDELINES.md`, and `../PULL_REQUESTS.md` is referenced by the PR template checklist.

# Files

- `ARCHITECTURE.md` — project structure diagram, core component flow (`gs.init() → Scene → Simulator → Solvers → Entities`), entity/morph/material/solver tables, description-based serialization (`Scene.export`/`Scene.load`), and a key-files reference.
- `CODING_CONVENTIONS.md` — import pattern, naming conventions, Pydantic options, tensor/device handling, error handling, build pattern, backend selection, and common API patterns.
- `EXAMPLES.md` — examples layout under `examples/`, the closed/reserved short-flag CLI table, backend default rules (CPU unless justified), output-file conventions (`out/`), style rules, and key example scripts.
- `PULL_REQUESTS.md` — contribution flow: reporting bugs, suggesting features, submitting changes (pre-commit, title prefix tags, sentence-style commit titles), review/merge policy (≥1 approval, Squash and Merge).
- `TESTING.md` — test suite layout, `uv`-based test invocation, pytest markers (`required`, `slow`, `examples`, `benchmarks`), key fixtures, and CI requirements.
- `USD_PARSER.md` — rules for the USD parser: scope restricted to `genesis/utils/usd`, always check `pxr-stubs` first, extract helpers to `usd_utils.py`, and pointers to the parser spec/files.

# Constraints

- These docs describe the contributor contract; keep them in sync with `../../CODING_GUIDELINES.md` (which points back here) when conventions change.
- Some diagrams in `ARCHITECTURE.md` are simplified and may lag the actual tree — verify against `../../genesis/` before treating a path as authoritative.

# See Also

- `../../CODING_GUIDELINES.md` → repo-wide guidelines that reference these docs (on conflict, that file is the authority).
- `../../tests/` → the suite described by `TESTING.md`.
- `../../examples/` → the scripts described by `EXAMPLES.md`.
- `../../genesis/utils/usd/` → the parser governed by `USD_PARSER.md`.
