# Intent

Throughput and constraint-solver timing benchmarks for very large batched rigid-body simulations.

# API Surface

- `franka.py` — Franka arm throughput at 30000 parallel environments (`gs.init(backend=gs.gpu, performance_mode=True)`), reporting FPS with and without DOF control.
- `anymal_c.py` — ANYmal-C quadruped at 30000 parallel environments using the Newton constraint solver.
- `timers.py` — Per-environment constraint-solver timings across 128 Go2 environments with `--mode 0|1|2` control-noise injection and optional `--profiling` (torch profiler); plots with `plotext`.

# Constraints

- These are benchmarks, not tutorials; all call `gs.init(backend=gs.gpu, performance_mode=True)` and build huge `n_envs`.
- When `PYTEST_VERSION` is present in `os.environ`, `n_envs` collapses to 2 and step counts to a handful, and `gs.gpu` falls back to CPU (these scripts are not in `CUDA_SCRIPT_NAMES` in `tests/test_examples.py`).

# Dependencies

- `torch` (already required by Genesis workflows); `plotext` only when plotting `timers.py` outside pytest.
- Full benchmarks require a CUDA GPU.

# Gotchas

- `timers.py` sets `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch.
- Commented FPS figures in `franka.py` (~43M FPS with control, ~32M in floor collision) are environment-specific and informational only.

# See Also

- `../rigid/` — smaller-scale rigid demos; `multi_gpu.py`/`ddp_multi_gpu.py` for GPU parallelism.
