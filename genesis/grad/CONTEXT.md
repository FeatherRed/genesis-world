# genesis/grad

## Intent
This directory is the differentiable-simulation bridge: it turns the rollout of a differentiable scene into a `torch` autograd graph and wires that graph to the Quadrants kernel compiler so gradients flow backward through kernel launches.
It does not define any physics: the solvers (see `../engine/solvers/`) implement the differentiable protocol, and this module provides the recording/reversal machinery they plug into.
`__init__.py` is empty — the package is two modules, `bridge.py` and `kernel_writes.py`.

## API Surface

### bridge.py — the autograd graph and the launch recorder
- `Operation` — abstract node body with `forward(*inputs) -> tuple` and `backward(*outputs_grad) -> tuple`; implemented by `StepOperation` (defined in `../engine/simulator.py`), and by `SegmentOperation` / `ReadOperation` here.
- `OperationNode(torch.autograd.Function)` — the single torch node type wrapping every `Operation`.
  `forward` holds a weak reference to the graph and returns the next state handle plus the operation's outputs; `backward` is `@once_differentiable` and routes output grads into `operation.backward`.
- `RolloutGraph` — the per-scene autograd graph.
  - `__init__(requires_grad, on_pass_start, on_pass_end)`, `restart()`, `is_recording` (property).
  - `record(operation, *inputs)`, `record_step(operation)`, `run_call(owner, method, args, kwargs, *, differentiable=True)`, `record_read(solver, tensors, selection)`, `mark_state_changed()`, `invalidate(change, envs_idx)`.
- Kernel backward registration (tags go ABOVE the `@qd.kernel` decorator; each is keyed on `kernel.fn`):
  - `no_grad(kernel)` — launches carry no gradient (e.g. resets of warm start / error flags).
  - `register_custom_backward(backward)` — a hand-written backward called with the launch's arguments.
  - `reversed_with(**overwrites)` — reverse via the kernel's own autodiff with some static args overwritten (e.g. `is_backward=True`).
- `LaunchRecorder` — the tape handed to Quadrants through `impl.get_runtime().target_tape`.
  - `__enter__`/`__exit__`, `insert(kernel, args)` (the launch hook), `entry(backward, *args, restore=..., autodiff=...)` (group launches into one entry), `reverse(start, stop)`.
- `custom_backward(backward, *args)` — context manager grouping the launches of a block into one entry; a no-op outside a recorded step.
- `is_recording_launches()` — whether the currently launched kernels are recorded for a backward pass.

### kernel_writes.py — write-set analysis of Quadrants kernels
- `get_written_paths(fn, owner=None) -> frozenset[tuple[str, ...]]` — static AST analysis of a kernel/func's Python source returning the parameter paths (e.g. `("dyn_state", "dofs", "vel")`) it may write.
- `Callee` (NamedTuple) — a func reachable from a call: `fn`, `owner`, `owner_path`.
- Internal helpers: `_resolve`, `_resolve_store`, `_find_callee`.
- `_WRITTEN_PATHS_CACHE` — an `OwnedCache` keyed by `(fn, id(owner))`, entries owned by the bound object (the function itself for a free function).

## Design Decisions
Every operation on a differentiable scene's state is one node of the autograd graph; the graph is ordered by a zero-dim "state handle" tensor that each operation replaces with a fresh clone, so torch runs backward passes in reverse operation order.
A second zero-dim "model handle" is saved by every node and its version is bumped on model change / reset, so a backward through a node they invalidated raises instead of replaying stale data.
The simulator owns the graph, so `OperationNode` stores only `weakref.ref(graph)`; a strong reference would form a cycle crossing the C++ autograd graph, which the garbage collector cannot traverse (the scene would leak).
`once_differentiable` plus `set_materialize_grads(False)` means grads are `None` where the loss does not reach, and double-backward is not supported.
`LaunchRecorder` is the autodiff tape: Quadrants forwards every kernel launch before running it; the recorder groups them into entries and, on `reverse`, runs each entry's backward (kernel autodiff or custom), then restores the arrays it overwrote so earlier entries reverse at the values they left.
Save-overwrite: before each launch, the recorder copies (`qd_to_torch(copy=True)`) any state array the launch writes that an earlier entry already uses, and restores those copies after the entry's backward — derived from `get_written_paths` narrowed to the struct members the compiled kernel uses.
`get_written_paths` exists because Quadrants exposes no write set of a compiled kernel (`FIXME: quadrants#1001`); it over-approximates (both branches of a static conditional count), which costs a redundant copy, never an incorrect gradient.
A call writing the state between two steps runs with zero-copy disabled (`gs.use_zerocopy.disabled()`) so its writes go through kernel launches the recorder sees; a step likewise writes its state only through kernels while recorded (`is_recording_launches`).
Two caches bound memory by object lifetime: `_LAUNCH_ARRAYS_CACHE` (compiled-kernel key + struct identity) and `_WRITTEN_PATHS_CACHE`, both `OwnedCache`s whose entries die with the structs/objects that own them.

## Constraints
This is a bridge, not a physics module: do not add solver logic here — new solver behavior belongs in `../engine/solvers/` and uses the registration decorators above.
Kernels passed to `get_written_paths` must be importable and source-readable (`inspect.getsource`); lambdas and dynamically generated kernels break the write-set analysis.
`reversed_with` arguments must be native Python scalars (`bool`/`int`/`float`), and the overwritten variant must not write any array the original launch leaves untouched.
The kernel-backward tags (`no_grad` / `register_custom_backward` / `reversed_with`) must sit above `@qd.kernel`, not below it, because the decorator receives the `QuadrantsCallable` and indexes by `kernel.fn`.

## Known Issues / Gotchas
Backward on a scene destroyed since the rollout raises; backward after the state was set while torch recorded no gradient, after a model modification, or after a reset also raises (read the saved-tensor error to distinguish).
`FIXME: quadrants#997` — a forward launch allocates `.grad` on non-leaf tensors; the warning is silenced and grads reclaimed.
`FIXME: quadrants#1000` — the Quadrants launch hook is only reachable via the private runtime slot `target_tape`.
`FIXME: quadrants#1001` — no kernel write set exposed, hence the source-level analysis in `kernel_writes.py`.
`FIXME: torch` — `_enter_pass` relies on private `torch._C._current_graph_task_id()` and `Variable._execution_engine.queue_callback`; the public alias is missing from supported torch releases.
On Metal with zero-copy, restores must `torch.mps.synchronize()` before the next launch reads the restored arrays (`reverse` and `add_grad_from_state` both flush).
`run_call` refuses an input tensor that tracks gradients when `differentiable=False`; nested tagged calls are part of the outermost call, not separate nodes.

## Dependencies
External (via `pyproject.toml`): `quadrants==1.4.0` (the kernel compiler / autodiff frontend) and `torch` (the autograd engine).
Cross-package, read from this module: `genesis.utils.cache.OwnedCache`, `genesis.utils.data.iter_state_arrays`, `genesis.utils.misc.qd_from_torch` / `qd_to_torch` / `qd_add_grad`.
`genesis.engine` is imported only under `TYPE_CHECKING` here; the runtime integration lives on the engine side.

## Test Strategy
The dedicated suite `../../tests/grad/` (sibling of the package — read-only) exercises the differentiable rollout end to end: rollout/backward passes, rigid dynamics/constraints/collision, hybrid push, and rigid optimization.
`tests/grad/conftest.py` monkeypatches `LaunchRecorder.__exit__` to catch leaks, so keep that hook's signature/semantics stable.
The hard-to-test surface is `get_written_paths` (write-set over-approximation) and `LaunchRecorder.reverse` (restore ordering / Metal sync) — verify any change here against the backward-pass tests of all differentiable solvers.

## See Also (routing)
- `../engine/simulator.py` → constructs `RolloutGraph` (with `on_pass_start`/`on_pass_end`), defines `StepOperation`, and calls `record_step` / `mark_state_changed` (sibling — read-only, escalate writes to the parent).
- `../engine/state_change.py` → the `@mutates` decorator routes state-mutating calls through `RolloutGraph.run_call` or `mark_state_changed` (sibling — read-only).
- `../engine/solvers/base_solver.py` → the solver differentiable protocol (`requires_grad`, `is_recording`, `record_read`, `add_grad_from_state`, `reset_grad`) (sibling — read-only).
- `../engine/solvers/rigid/` → the main consumers of `no_grad`, `register_custom_backward`, `reversed_with` (sibling — read-only).
- `../utils/` → `cache.py` (`OwnedCache`), `data.py` (`iter_state_arrays`), `misc.py` (`qd_*` helpers) (sibling — read-only).
