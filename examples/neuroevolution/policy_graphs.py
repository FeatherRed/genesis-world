"""The single width-generic policy graph shared by every neuroevolution task.

This module replaces the two task-bound copies of the same graph -- the Go2
walking one with 45 observations and 12 actions, and the drone hover one with 17
observations and 4 actions -- with one parameter-only implementation.  Nothing
here knows a task: the observation and action widths, the row count and the
episodes-per-policy tiling all come in as explicit arguments, and the genome
widths are cross-checked against them up front.

``algo`` and ``config`` are build arguments because the graph cannot be traced
without them: ``algo.transform``, ``algo.prepare_forward(samples=episodes)`` and
``algo.forward_prepared`` are trace-time operations that run once, inside the
build, and cannot be called outside it.

Contract
--------
The compiled graph consumes the four ``GenomeTensors`` payload leaves and a
``(rows, num_inputs)`` float32 observation tile, and returns the raw (unclipped)
``(rows, num_outputs)`` float32 actions.

Rows are **policy-major**: rows ``episodes*p .. episodes*p+episodes-1`` are the
``episodes`` episodes of policy ``p``, exactly the tiling
``prepare_forward(samples=episodes)`` declares.  Each row carries its own
observation, so the episodes of one policy are independent.

Only ``numpy``, ``evoxc`` and ``TensorNEAT`` are imported.
"""

from __future__ import annotations

import numpy as np

import evoxc

from tensorneat.genome.utils import GenomeTensors

from evox_runtime import (
    POP_OPERANDS,
    _attach,
    _device_name,
    _host,
    _outputs,
    _pop_specs,
    _positive_int,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)


def build_policy(
    algo,
    config,
    pop_arrays,
    rows,
    *,
    num_inputs,
    num_outputs,
    episodes=3,
    backend="numpy",
    device=None,
):
    """Compile the genome-payload + observation -> raw-action graph for ``rows`` rows.

    ``rows`` must be ``algo.pop_size * episodes`` and the rows are **policy-major**:
    rows ``episodes*p .. episodes*p+episodes-1`` are the ``episodes`` episodes of
    policy ``p``, which is the tiling ``prepare_forward(samples=episodes)`` declares.
    Each row carries its own observation, so the episodes of a policy are
    independent and need not share a state.

    The widths are validated against the traced genome so a task cannot silently
    trace the wrong shape.

    Args:
        algo: a TensorNEAT ``NEAT`` algorithm instance.
        config: the tensor-free ``pipeline._static_config(state)`` captured at build time.
        pop_arrays: the four genome payload arrays (``state.pop`` as numpy).
        rows (int): ``pop_size * episodes``.
        num_inputs (int): Observation width the graph is compiled for.
        num_outputs (int): Action width the graph is compiled for.
        episodes (int): Episodes per policy, the row tiling factor.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_policy`` stages its operands.
    """
    n_policies = int(algo.pop_size)
    rows = _positive_int(rows, "rows")
    episodes = _positive_int(episodes, "episodes")
    num_inputs = _positive_int(num_inputs, "num_inputs")
    num_outputs = _positive_int(num_outputs, "num_outputs")
    if rows != n_policies * episodes:
        raise ValueError(f"rows must be pop_size * episodes = {n_policies * episodes}, got {rows}")
    if num_inputs != int(algo.genome.num_inputs):
        raise ValueError(f"num_inputs must be algo.genome.num_inputs = {int(algo.genome.num_inputs)}, got {num_inputs}")
    if num_outputs != int(algo.genome.num_outputs):
        raise ValueError(
            f"num_outputs must be algo.genome.num_outputs = {int(algo.genome.num_outputs)}, got {num_outputs}"
        )
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    specs = [*_pop_specs(pop_arrays), evoxc.TensorSpec((rows, num_inputs), evoxc.float32, name="obs")]

    def kernel(node_keys, node_attrs, conn_keys, conn_attrs, obs):
        payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        transformed = algo.transform(config, payload)
        prepared = algo.prepare_forward(config, transformed, payload, samples=episodes)
        return algo.forward_prepared(config, transformed, prepared, obs)

    executable = evoxc.build(kernel, *specs, backend=resolved_backend, device=resolved_device)
    return _attach(
        executable,
        graph="policy",
        rows=rows,
        n_policies=n_policies,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        episodes=episodes,
        pop_shapes=tuple(_host(array).shape for array in pop_arrays),
        device_name=_device_name(resolved_device),
    )


def run_policy(executable, pop_arrays, obs):
    """Run the policy graph; returns raw (unclipped) actions ``(rows, num_outputs)`` float32."""
    arrays = tuple(_host(array) for array in pop_arrays)
    if len(arrays) != len(POP_OPERANDS):
        raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
    for array, shape in zip(arrays, executable.pop_shapes, strict=True):
        if array.shape != shape:
            raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {array.shape}")
    expected = (int(executable.rows), int(executable.num_inputs))
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.shape != expected:
        raise ValueError(f"obs must have shape {expected} to match the compiled graph, got {obs_arr.shape}")
    outputs = evoxc.run(
        executable,
        *[_to_device_array(executable, np.ascontiguousarray(array)) for array in arrays],
        _to_device_array(executable, np.ascontiguousarray(obs_arr)),
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)
