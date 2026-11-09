# Intent

Loads Universal Scene Description (`.usd`) assets into Genesis via `gs.morphs.USD` / `scene.add_stage`, exercising the USD importer and its articulated-fixed/free handling.

# API Surface

- `import_stage.py` — Downloads a USD asset from Hugging Face and loads it with `scene.add_stage(gs.morphs.USD(...))`, then animates joints with a `JointAnimator`.
- `kitchen.py` — Loads the Lightwheel Kitchen dataset assets (`--asset dishwasher|bottle|all`, `--full` for the whole room, `--collision` to show collision geometry, `-v` for the interactive viewer with mouse dragging).

# Constraints

- Requires the `usd` extra (`pip install -e '.[usd]'`), which provides `usd-core` (the `pxr` module) and `filelock`.
- Both scripts fetch assets at runtime from the Hugging Face hub (`huggingface_hub` is a `dev` extra; the scripts import it directly).

# Dependencies

- `usd` extra (`pxr`/`usd-core`), `huggingface_hub`, plus network access to download `Genesis-Intelligence/assets` (pinned revision in `kitchen.py`).

# Gotchas

- `kitchen.py` needs `enable_neutral_collision=True` so articulated parts (e.g. dishwasher doors) do not clip.
- Headless runs are bounded (finite settle steps); with `-v` the viewer runs until Esc.
- The `MouseInteractionPlugin` must be attached before `scene.build()`.

# See Also

- `../../tests/test_examples.py` — maps the `usd` directory to the `pxr` optional dependency for CI skipping.
