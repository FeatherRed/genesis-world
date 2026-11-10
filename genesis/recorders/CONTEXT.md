# genesis/recorders

## Intent
Recording/export layer of the Genesis physics engine: samples data from a 0-arg callable (a sensor `read`, a camera render, or a user `data_func`) at a configurable rate and writes it to video/CSV/NPZ files, plots it live (matplotlib/PyQtGraph), or records the full scene state as a compressed, replayable `.gstraj` trajectory.
Recorders are created through the `RecorderManager` owned by `Scene`; their whole lifecycle (build/step/record/reset/stop) is driven by the scene, never directly by users.

## Module map (API surface)
- `__init__.py` — re-exports the pydantic option classes from `genesis.options.recorders` plus `RecorderManager` and `register_recording`, so `gs.recorders.VideoFile(...)`, `gs.recorders.TrajectoryFile(...)`, etc. are the `rec_options` argument.
- `base_recorder.py` — `Recorder[T]`, the base contract every recorder implements, plus the threading/queue runtime (`step`/`record`/`sync`/`start`/`stop`).
- `recorder_manager.py` — `RecorderManager` and the `register_recording(options_cls)` decorator that binds an options class to its recorder class.
- `file_writers.py` — `BaseFileWriter` and the concrete `VideoFileWriter`, `CSVFileWriter`, `NPZFileWriter`.
- `trajectory.py` — `TrajectoryFileWriter` (`.gstraj` writer), `Trajectory` (reader/seeker/replayer), `save_checkpoint`, and the on-disk format helpers.
- `plotters.py` — `BasePlotter`, `LinePlotHelper`, `BasePyQtPlotter`/`PyQtLinePlotter`, `BaseMPLPlotter`/`MPLLinePlotter`/`MPLImagePlotter`/`MPLVectorFieldPlotter`.

## How recorders attach to a scene
- `Scene.__init__` creates one `RecorderManager(self._sim.dt)`; `Scene` is its only owner.
- `Scene.add_recorder(data_func, rec_options)` delegates to `RecorderManager.add_recorder` and returns the recorder.
- `Scene.start_recording(TrajectoryFile)` is sugar for `add_recorder` with a `data_func` returning `TrajectorySource(sim, desc, layout)`.
- `Sensor.start_recording(rec_options)` (in `genesis/engine/sensors/base_sensor.py`) wraps `sensor.read` as the data_func; entities record through their sensors or via `Scene.add_recorder` with a custom func (there is no `Entity.add_recorder`).
- Lifecycle wiring lives in `genesis/engine/scene.py`: `build()` calls `manager.build()`, each `step()` calls `manager.step(cur_step_global)` under `torch.no_grad()` inside the `recorders` timer phase, `reset()`/`__setstate__` call `manager.record(...)` then `manager.reset(...)`, `stop_recording()` calls `manager.record(cur_step_global)` then `manager.stop()`, and `destroy()` stops if still recording.
- Recorders read the state a step starts from (before `Simulator.step`), and `record()` captures the state the run stops/resets at, whatever the sampling rate.

## Base-recorder contract
- Subclasses implement `build()`, `process(data, cur_time)`, `cleanup()`, optionally `reset(envs_idx=None)`, and the `run_in_thread` property; they must not change the constructor signature (instantiation goes through the manager).
- `data_func` is a 0-arg callable returning the sampled value; `Recorder._get_frame()` converts it via `genesis.utils.misc.data_to_array` to host numpy before `process()`, so `process()` never branches on torch vs numpy.
- `run_in_thread=True` hands `(data, timestamp)` to a bounded `queue.Queue` consumed by a worker thread that calls `process()`; overflow drops the oldest sample after `buffer_full_wait_time`.
- A worker-thread exception is stored in `_error` and re-raised once on the stepping thread at the next `step()`/`sync()`/`stop()`; a failed recorder stops for good (`is_alive=False`) and `RecorderManager.reset` leaves dead recorders out.
- `hz` becomes `steps_per_sample = max(1, round(1/(hz*step_dt)))`; a non-integer multiple logs a warning and uses the rounded rate.
- `TrajectoryFileWriter` overrides `_get_frame()` and sets `run_in_thread=False` (it runs its own three-thread pipeline), so its `process()` is a no-op.

## Design decisions
- Options live in `genesis/options/recorders.py` as pydantic models; recorder classes are matched by exact `type(rec_options)` lookup in `RECORDER_TYPES_MAP`, so subclassing an options class silently fails to match.
- `VideoFileWriter.run_in_thread=False`: `VideoEncoder` already owns a bounded worker thread, and the writer only does an unconditional `data.copy()` so the caller may reuse its buffer.
- `CSVFileWriter`/`NPZFileWriter` use `run_in_thread=True`; CSV flushes per write only when `save_every_write`, NPZ buffers everything and writes once in `cleanup`.
- Plotters run on the stepping/main thread (`run_in_thread=False`) and raise in `__init__` if constructed off the main thread: Qt widgets must stay on their owning thread, and matplotlib drawing cannot overlap the interpreter lock.
- `BasePlotter` composes an internal `VideoFileWriter` recorder when `save_to_filename` is set: it renders its own frame first, then the video writer samples it.
- `.gstraj` format (magic `GSTRAJ4\n`): scene description (as `Scene.export`) + JSON manifest + chunked frames; every part carries length + CRC32, so a crash leaves every completed chunk readable and readers stop at the first untrusted record.
- Frames are stored block-sparse as XOR differences from the previous frame (256-byte blocks) and zstd-compressed at level 1; a three-thread pipeline (step→slot copy, compression, file I/O) keeps recording off the stepping thread unless compression/disk falls behind.
- `mode="exact"` stores STATE+DERIVED for bit-for-bit replay; `mode="minimal"` stores STATE only and recomputes DERIVED on seek; the default auto-selects minimal when the exact frame exceeds 4 MiB.
- Every recorder module imports PyTorch from `genesis.utils.torch_compat`, never directly, so the subtree imports without PyTorch installed.
- `RecorderManager` constructs, builds, steps, records, resets and stops torch-free when no recorder is registered, and its `record` runs under `torch_compat.no_grad()`.
- Every file writer and the trajectory writer calls `require_torch` in `build`, before opening any file, so an actual recording raises a clear GenesisException naming the recorder in torch-free mode.

## Known issues / gotchas
- The trajectory writer appends a final `LAST` record holding every array but CONSTANT (scratch included) for crash analysis; this is also the checkpoint format (`Scene.save_checkpoint` = a one-frame trajectory).
- `TrajectoryFileWriter` carries an in-code FIXME `quadrants#970`: a 0-d field has no zero-copy view, so reading one per frame runs a copy kernel and step frames leave 0-d arrays out.
- `Trajectory.play(loop=True)` requires an interactive viewer.
- Video files hold steps+1 frames (state before each step, then the state the run stopped at) — see `tests/core/test_recorders.py::test_video_writer`.
- A `.gstraj` from another version shares the `GSTRAJ` prefix; the reader detects the version and raises a specific message.

## Constraints
- Do not duplicate option documentation here: option classes, defaults and validation live in `genesis/options/recorders.py` (sibling, read-only).
- `process()` receives host numpy data by contract; only the trajectory writer bypasses this by overriding `_get_frame`.
- Add new recorder types with `@register_recording(OptionsClass)`; keep constructors instantiable only through `RecorderManager`.

## Dependencies
- In-repo: `genesis.options.recorders` (options), `genesis.utils.misc` (`data_to_array`, `has_display`), `genesis.utils.data` (`DataKind`, `ArrayItem`/`ScalarItem`/`ConfigItem`/`StaticConfig`), `genesis.utils.serialization` (scene export for the trajectory header), `genesis.utils.video_encoder` (`VideoEncoder`/PyAV), `genesis.engine.state_change` (`StateChange.MODEL`, `Subscriber`), `genesis.engine.states.solvers` (`SimulatorCheckpoint`/`SolverCheckpoint`), `genesis.engine.scene` (`TrajectorySource`, `Scene` driver).
- External: PyAV, zstandard, numba, quadrants (`qd`), Pillow; optional pyqtgraph and matplotlib>=3.7.0 for the respective plotters (checked lazily at build, not import).

## Test strategy
- No tests live in this subtree; the suite is `tests/core/test_recorders.py` (sibling, read-only), marked `@pytest.mark.required`, covering MPL line/vector-field plotters (Agg backend + png snapshots) and video writer frame counts.
- GPU/threading paths of the trajectory writer are not unit-tested here; validate end-to-end via `Scene.start_recording` → step → `stop_recording` → `Scene.load_trajectory` → `Trajectory.seek`/`play`.

## See also
- `../options/recorders.py` — recorder option definitions (read-only sibling).
- `../../tests/core/test_recorders.py` — recorder test suite (read-only sibling).
- `../engine/scene.py` — `Scene` owns and drives the `RecorderManager`.
