# Torch-free, GPU-resident TensorNEAT training on the AMD Instinct MI300X

Deliverable report for the Torch-free / device-resident migration of the neuroevolution (TensorNEAT + EvoX Compiler)
training loop. Two pilots are complete: **Go2 Walking** (primary target) and **Drone Hover** (same trainer and
simulator interface).
Branch under review: `evogit-agent-T2-A1`.

## 1. Reproducible commands

All commands run from the repository root with the project interpreter and the Torch blocker prepended, so any
`import torch` in the child fails loudly; the runs assert `"torch" not in sys.modules` before exit.

```bash
PY=/root/evox-robotics/.venv/bin/python
export PYTHONPATH=/tmp/notorch:$PWD          # /tmp/notorch/sitecustomize.py blocks torch / torch.*

# Go2 Walking - 8 policies x 3 episodes x 2 generations, full 1000-step episodes
$PY examples/locomotion/go2_neat_train.py --torch-free --pop-size 8  --generations 2 --no-stop-when-done
# Go2 Walking - larger population (96 rows)
$PY examples/locomotion/go2_neat_train.py --torch-free --pop-size 32 --generations 2 --no-stop-when-done

# Drone Hover - 8 policies x 3 episodes x 2 generations, full 1500-step episodes
$PY examples/drone/hover_neat_train.py --torch-free --pop-size 8  --generations 2
$PY examples/drone/hover_neat_train.py --torch-free --pop-size 32 --generations 2

# Layer-level device-residency check (synthetic env, no simulator)
$PY examples/neuroevolution/verify_device.py --pops 8 32 64
```

Tests (pytest is not installed in the project venv; `/tmp/gs-pytest-libs` supplies it, and in-process GPU modules
need the system HSA runtime because the HSA repair is warn-only under pytest):

```bash
export PYTHONPATH=$PWD:/tmp/gs-pytest-libs LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1
$PY -m pytest tests/test_hsa_runtime.py tests/test_go2_evox_parity.py tests/test_go2_neat_kernels.py \
  tests/test_go2_neat_seeding.py tests/test_hover_evox_parity.py tests/test_hover_neat_seeding.py \
  tests/test_example_harness.py tests/test_neuroevolution_trainer.py tests/test_neuroevolution_device.py \
  tests/test_neuroevolution_device_parity.py tests/test_torch_free_training.py tests/test_go2_device_parity.py \
  tests/test_hover_device_parity.py tests/test_torch_free_genesis_contracts.py -q -n 2 -p no:cacheprovider
```

## 2. Test results (merged head, MI300X VF / ROCm 7.1.0)

| Run | Result |
| --- | --- |
| 14-module selection, `-n 0` | **141 passed, 0 failed, 0 errors** (726 s) |
| 14-module selection, `-n 2 --dist=worksteal` | **141 passed, 0 failed, 0 errors** (437-535 s, no worker abort) |
| `-m required --collect-only` (whole tree) | 584/913 collected, 13 collection errors - all `ModuleNotFoundError: httpcore` (asset-dependent modules, pre-existing environment gap) |
| `EVOXC_DEVICE=cpu:0` on the GPU modules | 19 passed / 20 skipped / 0 failed - the skips name the CPU-device reason |
| `tests/rigid -m required` (torch present) | 102 passed / 2 skipped / 66 errors - counts identical to the pre-change baseline; all errors environmental (`httpcore`, missing NVML/AMD SMI) |

What each new module pins:

- `tests/torch_block.py` - reusable "PyTorch unavailable" harness: a meta-path finder that raises `ImportError` for
  `torch`/`torch.*`, child-process runners with a leak epilogue, and an AST scan for `import torch` in the training path.
- `tests/test_torch_free_training.py` - `import genesis`, `gs.init`, scene build and a rollout with torch blocked;
  the real `go2_neat_train.py --torch-free` CLI behind a raising `torch` package; torch-free vs torch-present state
  boundary bit-identical (`max|diff| = 0.0`).
- `tests/test_neuroevolution_device.py` - device placement of initialization, policy forward, `evaluate` metrics,
  `tell()` and the next-generation population; negative predicates; pop 32/64.
- `tests/test_neuroevolution_device_parity.py` - CPU tier bit-exact; `iree@rocm:0` tier with measured tolerances.
- `tests/test_go2_device_parity.py`, `tests/test_hover_device_parity.py` - pilot-level parity: per-step observations,
  reward terms + total, roll/pitch termination, timeout, never-re-arming `alive`, episode returns, fitness, first
  population update, plus the documented end-to-end reference numbers.
- `tests/test_torch_free_genesis_contracts.py` - strides-less DLPack producer writes, `performance_mode=True`
  torch-free builds with masked resets, and the boolean error-mask contract (`kDLBool`).

## 3. Data flow: backend, device and copies at each boundary

Go2 device residency (drone is the same shape; the drone stages rotor RPM through `set_propellers_rpm` instead of
`control_dofs_position`):

| Boundary | Backend | Device | Copy |
| --- | --- | --- | --- |
| Initial population / species / RNG key / generation counter | evoxc graph (`device_setup`) | rocm:0 | none - created on device, kept across generations |
| Solver state (`dofs` pos/vel slices, `get_pos/get_quat/get_vel/get_ang`, error mask) -> obs graph | Quadrants -> IREE | rocm:0 | none (zero-copy DLPack view; `contiguous()` only where IREE refuses strides, device-side) |
| obs -> policy forward -> raw action -> clip/action-scale/actuator order -> PD targets | IREE graphs | rocm:0 | none between graphs; one device-to-device staging copy when the target is written to Quadrants |
| PD targets / reset qpos tile -> solver | IREE -> Quadrants (`set_qpos`, `control_dofs_position`, `set_propellers_rpm`) | rocm:0 | 1 device-to-device copy (HIP `hipMemcpy`/`hipMemcpy2D`) per write; no host hop |
| reward / termination / return accumulation / `episode_returns` reshape | IREE graphs | rocm:0 | none |
| fitness reduction (`build_fitness`) -> `tell()` | IREE graph -> IREE graph | rocm:0 | none - the device fitness tensor is the tell input |
| `tell()` -> next generation population, species, key, counter | IREE graph | rocm:0 | none - next-generation payload is the tell output object (pointer-identical) |
| Per-step `done` mask (host mirror), int8 `end_code`, `init_qpos`, seed triple | host | - | bytes per step (control-flow/logging only), one H2D seed tile per generation |
| Record metrics (`fitness`, `episode_returns`, `returns`, `seeds`) | host | - | read back **after** `tell()` consumed the device tensors, for logging only |

The training data path (observations, actions, rewards, returns, fitness, population, species, RNG) has **no NumPy or
CPU round trip**. Host traffic is limited to launch/configuration, episode seed metadata and post-hoc logging.

## 4. Compile time and steady-state throughput

Torch blocked, `iree@rocm:0`, MI300X VF, full-length episodes. Device setup covers the TensorNEAT tell graph plus the
fitness-rule and verification graphs; the task graphs are built with the scene.

| Pilot / run | Setup (compile) | Per-generation wall time | Throughput |
| --- | --- | --- | --- |
| Go2 pop 8 (24 rows), 2 x 1000 steps | 8.73 s (tell 7.11, rule 0.27, verification 0.33) | 9.66 s, 6.47 s | 103.5 -> 154.6 steps/s; 2485 -> 3711 env-steps/s |
| Go2 pop 32 (96 rows), 2 x 1000 steps | 8.38 s (tell 6.76) | 11.07 s, 7.40 s | 90.3 -> 135.2 steps/s; 8670 -> 12975 env-steps/s |
| Drone pop 8 (24 rows), 160/58 steps | 9.00 s (tell 7.37, rule 0.27, verification 0.35) | 1.69 s, 0.39 s | 94.5 -> 149.5 steps/s; 2269 -> 3589 env-steps/s |
| Drone pop 32 (96 rows), 160/149 steps | 8.40 s (tell 6.77) | 1.55 s, 1.20 s | 102.9 -> 124.1 steps/s; 9882 -> 11914 env-steps/s |
| Layer only (synthetic env, 3 episodes, 2 gens) | ~9 s | 0.09 / 0.17 / 0.27 s at pop 8 / 32 / 64 | 6360 / 14145 / 17489 env-steps/s |

Reference fitness histories reproduce the pre-migration implementation exactly: Go2 `-7.3785 / -7.3180`
(full-horizon 3-generation run: `-7.3785 / -7.3180 / -6.0399`), drone `-0.1486 / -0.1691`.
Wall-clock numbers vary with GPU sharing; the semantic leaves (seeds, steps, end reasons, fitness) are stable.

## 5. Acceptance criteria

1. **Population initialization on the device** - met. `device_setup` builds the four genome leaves, the 10-leaf species
   pytree, the RNG key and the generation counter as device tensors and keeps them there; CPU is bitwise identical to
   `Pipeline.setup()`, `iree@rocm:0` differs only on four float leaves by <= 2.384e-07 (integer leaves, key, counter
   and NaN masks exact).
2. **Full rollout on the device, Torch removed** - met. Observations, policy forward, action post-processing, reward,
   termination and return accumulation run as compiled graphs; the simulator boundary is a Torch-free device-buffer
   interface (`genesis/utils/device_buffer.py`) and the process runs with `import torch` blocked, including
   `import genesis`, `gs.init`, `scene.build` and 1000+ steps.
3. **Fitness and evolution on the device** - met. Three episodes per policy with the same ordered seed triple per
   generation and a fresh triple in the next; the three returns are reduced to fitness on the GPU and fed directly into
   `tell()`; the updated population, species and RNG state are the tell outputs consumed by the next generation.
   Backend interoperability is not needed because the whole loop runs one backend (see limitation 2).
4. **Verified on the MI300X** - met. 8 policies x 3 episodes x >= 2 generations and 32 policies x 2 generations, both
   pilots; a Torch-unavailable test harness that rejects any torch import; device placement asserted for
   initialization, policy forward, rewards, fitness, `tell()` and the next-generation population; parity against the
   existing implementation with fixed seeds and documented tolerances.

## 6. Remaining limitations and blocking APIs (with reproducers)

1. **Quadrants has no DLPack importer.** `qd.Field`/`qd.Ndarray` expose only `to_dlpack`; the pybind program exposes
   only `field_to_dlpack`/`ndarray_to_dlpack`, and `copy_from` accepts only Quadrants objects
   (`Field.copy_from` -> `TypeError: Cannot copy from a non-field object`, `Ndarray.copy_from` -> `AssertionError`).
   Graph -> simulator writes therefore use a guarded ctypes HIP DtoD shim (`genesis/utils/hip_copy.py`), which stays
   device-to-device but costs one copy per write. A native importer would remove the shim and the copy.
2. **evoxc device payloads are backend-private.** `XlaExecutable.run` rejects an `IreeDevicePayload` and vice versa, and
   `XlaDevicePayload` has no `__dlpack__`. Reproducer: `evoxc.run(build_fitness(8, backend="iree", device=ROCM), ...)`
   then `evoxc.run(build_tell(..., backend="xla", device=ROCM), ...)` ->
   `DeviceError: input ... is not a device-resident tensor on this executable's device`. Consequence: forced single
   backend; on ROCm that is IREE (IREE cannot compile the tell graph there without the reduction-distribution flag
   gate, and IREE tell is ~15x slower than XLA - the price of no host round trip).
   If a review requires IREE rollout + XLA evolution specifically, that criterion is unmet and this is the blocking API.
3. **evoxc `DeviceArray.reshape` returns a host copy** - replaced by a compiled reshape graph for `episode_returns`.
4. **IREE refuses strided DLTensors** - a batch-last solver view is compacted on device (`contiguous()`) before import.
5. **`evoxc.nan_to_num` is not IREE-compilable** and `evoxc.reduce_all` does not exist - the fitness rule uses a
   `|x| <= float32 max` select plus `reduce_mean`.
6. **An evoxc device tensor still alive when a stepped scene is torn down segfaults on ROCm** - producers must be
   dropped before scene teardown.
7. Torch-free known issues that do not affect the pilots: an opaque DLPack producer used as an *index* selection is
   still read back on the host; `KinematicSolver`'s `set_base_links_*`/`set_vverts` lack a producer branch; the Go2
   numpy residency cannot run Torch-free because `get_dofs_position(motors_dof_idx)` needs non-contiguous index
   selection (device residency is the Torch-free path; the drone numpy residency does run Torch-free).

Detailed, per-node records (measured numbers, reproducers, contracts) live in `genesis/CONTEXT.md`,
`genesis/utils/CONTEXT.md`, `genesis/engine/solvers/rigid/CONTEXT.md`, `examples/neuroevolution/CONTEXT.md`,
`examples/locomotion/CONTEXT.md`, `examples/drone/CONTEXT.md` and `tests/CONTEXT.md`.
